import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExoDataset(Dataset):
	def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y: np.ndarray | None):
		self.X_num = torch.tensor(X_num.astype(np.float32)) if X_num.size else torch.zeros((len(X_cat), 0))
		self.X_cat = torch.tensor(X_cat.astype(np.int64)) if X_cat.size else torch.zeros((len(X_num), 0), dtype=torch.long)
		if y is not None:
			self.y = torch.tensor(y.astype(np.int64))
		else:
			self.y = torch.zeros((self.X_num.shape[0] if self.X_num.size else self.X_cat.shape[0]), dtype=torch.int64)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X_num[idx], self.X_cat[idx], self.y[idx]


class FeatureTokenizer(nn.Module):
	def __init__(self, cat_dims: List[int], num_features_len: int, embed_dim: int):
		super().__init__()
		self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
		self.num_embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_features_len)])

	def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
		embeddings: List[torch.Tensor] = []
		for i, emb in enumerate(self.cat_embeddings):
			embeddings.append(emb(x_cat[:, i]))
		for i, emb in enumerate(self.num_embeddings):
			embeddings.append(emb(x_num[:, i].unsqueeze(1)))
		return torch.stack(embeddings, dim=1)


class TabularTransformer(nn.Module):
	def __init__(self, cat_dims: List[int], num_features_len: int, embed_dim: int, n_heads: int, n_layers: int, dropout: float, n_classes: int):
		super().__init__()
		self.tokenizer = FeatureTokenizer(cat_dims, num_features_len, embed_dim)
		seq_len = len(cat_dims) + num_features_len
		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02)
		encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dropout=dropout)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
		self.norm = nn.LayerNorm(embed_dim)
		self.head = nn.Linear(embed_dim, n_classes)

	def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
		tokens = self.tokenizer(x_num, x_cat)
		B = tokens.size(0)
		cls = self.cls_token.expand(B, -1, -1)
		x = torch.cat([cls, tokens], dim=1) + self.pos_embedding[:, : tokens.size(1) + 1, :]
		x = self.encoder(x)
		x = self.norm(x[:, 0, :])
		return self.head(x)


def infer_feature_types(df: pd.DataFrame, label_col: str) -> Tuple[List[str], List[str]]:
	cat_cols: List[str] = []
	num_cols: List[str] = []
	for col in df.columns:
		if col == label_col:
			continue
		if pd.api.types.is_numeric_dtype(df[col]):
			num_cols.append(col)
		else:
			cat_cols.append(col)
	return num_cols, cat_cols


def build_arrays(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
	# Fill numeric NaNs with column medians
	X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
	for c in num_cols:
		X_num[c] = X_num[c].astype(float).fillna(X_num[c].median())
		# If the entire column was NaN, median is NaN; fill remaining with 0
		X_num[c] = X_num[c].fillna(0.0)

	# Convert categoricals to category codes; cap unseen at -1 during inference
	cat_maps: Dict[str, Dict[str, int]] = {}
	X_cat_list: List[np.ndarray] = []
	for c in cat_cols:
		vc = df[c].astype(str).fillna("<NA>")
		vals = sorted(vc.unique().tolist())
		mapping = {v: i for i, v in enumerate(vals)}
		cat_maps[c] = mapping
		X_cat_list.append(vc.map(mapping).fillna(-1).astype(int).to_numpy())

	X_cat = np.stack(X_cat_list, axis=1) if X_cat_list else np.zeros((len(df), 0), dtype=np.int64)
	X_num_arr = X_num.to_numpy() if len(num_cols) else np.zeros((len(df), 0), dtype=np.float32)
	return X_num_arr, X_cat, {c: m for c, m in cat_maps.items()}


def train(args):
	data_path = Path(args.data)
	df = pd.read_parquet(data_path)
	# Drop non-feature identifiers that are obviously strings with high cardinality if present
	for drop_col in ["pl_name", "hostname", "kepoi_name", "kepler_name"]:
		if drop_col in df.columns:
			df.drop(columns=[drop_col], inplace=True)

	label_col = "label"
	assert label_col in df.columns, f"Expected '{label_col}' column in dataset"
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(df[label_col].astype(str))
	num_cols, cat_cols = infer_feature_types(df.drop(columns=[label_col]), label_col=None)  # label removed already

	# Optionally downsample for quick run
	if args.sample_frac < 1.0:
		df, _, y, _ = train_test_split(df, y, train_size=args.sample_frac, stratify=y, random_state=42)

	X_num, X_cat, cat_maps = build_arrays(df.drop(columns=[label_col]), num_cols, cat_cols)
	Xn_tr, Xn_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(X_num, X_cat, y, test_size=0.2, random_state=42, stratify=y)

	train_ds = ExoDataset(Xn_tr, Xc_tr, y_tr)
	test_ds = ExoDataset(Xn_te, Xc_te, y_te)
	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

	cat_dims = [max(2, int(max(2, len(m)))) for m in cat_maps.values()]
	num_features_len = Xn_tr.shape[1]
	n_classes = len(label_encoder.classes_)

	model = TabularTransformer(
		cat_dims=cat_dims,
		num_features_len=num_features_len,
		embed_dim=args.embed_dim,
		n_heads=args.heads,
		n_layers=args.layers,
		dropout=args.dropout,
		n_classes=n_classes,
	).to(DEVICE)

	optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(args.epochs):
		model.train()
		loss_sum = 0.0
		for xb_num, xb_cat, yb in train_loader:
			xb_num = xb_num.to(DEVICE)
			xb_cat = xb_cat.to(DEVICE)
			yb = yb.to(DEVICE)
			optimizer.zero_grad()
			logits = model(xb_num, xb_cat)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()
			loss_sum += loss.item()
		model.eval()
		y_true, y_pred = [], []
		with torch.no_grad():
			for xb_num, xb_cat, yb in test_loader:
				xb_num = xb_num.to(DEVICE)
				xb_cat = xb_cat.to(DEVICE)
				logits = model(xb_num, xb_cat)
				pred = torch.argmax(logits, dim=1).cpu().numpy()
				y_pred.extend(pred)
				y_true.extend(yb.numpy())
		f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
		print(f"Epoch {epoch+1}/{args.epochs} - loss {loss_sum/max(1,len(train_loader)):.4f} - val f1 {f1:.4f}")

	# Save artifacts
	out_dir = Path(args.out)
	out_dir.mkdir(parents=True, exist_ok=True)
	artifacts = {
		"num_cols": num_cols,
		"cat_cols": cat_cols,
		"cat_maps": {k: {str(kk): int(vv) for kk, vv in m.items()} for k, m in cat_maps.items()},
		"label_classes": label_encoder.classes_.tolist(),
		"embed_dim": args.embed_dim,
		"heads": args.heads,
		"layers": args.layers,
		"dropout": args.dropout,
	}
	(torch.save(model.state_dict(), out_dir / "tabular_transformer.pt"))
	with (out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
		json.dump(artifacts, f, indent=2)

	print("Classification report:")
	print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a FT-Transformer-style model on merged Kepler summaries")
	parser.add_argument("--data", type=str, default="data/frames/kepler_summary_with_labels.parquet")
	parser.add_argument("--out", type=str, default="models")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--embed_dim", type=int, default=64)
	parser.add_argument("--heads", type=int, default=4)
	parser.add_argument("--layers", type=int, default=3)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--sample_frac", type=float, default=0.2)
	args = parser.parse_args()
	train(args)
