import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


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
		col = X_num[c].astype(float)
		# Avoid computing median on all-NaN columns (which triggers numpy warnings)
		if col.notna().any():
			med = col.median()
			col = col.fillna(med)
		else:
			col = pd.Series(0.0, index=col.index)
		X_num[c] = col

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


def build_arrays_with_maps(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cat_maps: Dict[str, Dict[str, int]] | None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
	"""Build arrays optionally using precomputed categorical maps for consistency with a saved model."""
	# Numeric handling identical to build_arrays
	X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
	for c in num_cols:
		col = X_num[c].astype(float)
		if col.notna().any():
			med = col.median()
			col = col.fillna(med)
		else:
			col = pd.Series(0.0, index=col.index)
		X_num[c] = col

	# Categorical handling
	if cat_maps is None:
		# Fall back to discovering maps from data
		return build_arrays(df, num_cols, cat_cols)
	X_cat_list: List[np.ndarray] = []
	for c in cat_cols:
		vc = df[c].astype(str).fillna("<NA>")
		mapping = cat_maps.get(c, {})
		# Map known values; unknowns -> -1
		X_cat_list.append(vc.map(mapping).fillna(-1).astype(int).to_numpy())
	X_cat = np.stack(X_cat_list, axis=1) if X_cat_list else np.zeros((len(df), 0), dtype=np.int64)
	X_num_arr = X_num.to_numpy() if len(num_cols) else np.zeros((len(df), 0), dtype=np.float32)
	return X_num_arr, X_cat, cat_maps


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes)
    # Avoid divide-by-zero; weight ~ inverse frequency
    weights = len(y) / (counts + 1e-6)
    # Normalize to mean 1.0 for numerical stability
    weights = weights * (n_classes / weights.sum())
    return weights


def train(args):
	logger = logging.getLogger(__name__)
	data_path = Path(args.data)
	df = pd.read_parquet(data_path)
	# Drop non-feature identifiers that are obviously strings with high cardinality if present
	for drop_col in ["pl_name", "hostname", "kepoi_name", "kepler_name"]:
		if drop_col in df.columns:
			df.drop(columns=[drop_col], inplace=True)

	label_col = "label"
	assert label_col in df.columns, f"Expected '{label_col}' column in dataset"
	# Binary target: 1 = CONFIRMED, 0 = FALSE POSITIVE; candidates marked -1 (excluded from train/val)
	label_str = df[label_col].astype(str)
	y_all = np.where(label_str == "CONFIRMED", 1, np.where(label_str == "FALSE POSITIVE", 0, -1)).astype(int)
	# Optionally load a saved feature schema and model
	loaded_artifacts = None
	if args.load_model_dir:
		model_dir = Path(args.load_model_dir)
		with (model_dir / "feature_config.json").open("r", encoding="utf-8") as f:
			loaded_artifacts = json.load(f)
		# Enforce the same feature ordering and maps as training
		num_cols = loaded_artifacts.get("num_cols", [])
		cat_cols = loaded_artifacts.get("cat_cols", [])
		loaded_cat_maps = {k: {kk: int(vv) for kk, vv in m.items()} for k, m in loaded_artifacts.get("cat_maps", {}).items()}
	else:
		num_cols, cat_cols = infer_feature_types(df.drop(columns=[label_col]), label_col=None)

	# Optionally downsample for quick run (on labeled subset only)
	labeled_mask = y_all >= 0
	if args.sample_frac < 1.0:
		df_labeled = df[labeled_mask]
		y_labeled = y_all[labeled_mask]
		df_labeled, _, _, _ = train_test_split(df_labeled, y_labeled, train_size=args.sample_frac, stratify=y_labeled, random_state=42)
		# Reassemble df: sampled labeled + all unlabeled; then recompute y_all from labels
		df = pd.concat([df_labeled, df[~labeled_mask]], axis=0)
		label_str = df[label_col].astype(str)
		y_all = np.where(label_str == "CONFIRMED", 1, np.where(label_str == "FALSE POSITIVE", 0, -1)).astype(int)

	# Build arrays using loaded maps if provided
	if args.load_model_dir and loaded_artifacts is not None:
		X_num, X_cat, cat_maps = build_arrays_with_maps(df.drop(columns=[label_col]), num_cols, cat_cols, loaded_cat_maps)
	else:
		X_num, X_cat, cat_maps = build_arrays(df.drop(columns=[label_col]), num_cols, cat_cols)
	# Train/val split only on labeled data (exclude candidates)
	X_num_lab = X_num[y_all >= 0]
	X_cat_lab = X_cat[y_all >= 0]
	y_lab = y_all[y_all >= 0]
	Xn_tr, Xn_te, Xc_tr, Xc_te, y_tr, y_te = train_test_split(
		X_num_lab, X_cat_lab, y_lab, test_size=0.2, random_state=42, stratify=y_lab
	)

	# Log dataset shapes and class distribution
	logger.info(f"Dataset: total={len(df)} train={len(y_tr)} val={len(y_te)}")
	train_counts = np.bincount(y_tr, minlength=2)
	te_counts = np.bincount(y_te, minlength=2)
	logger.info(f"Train class counts: NEG(0)={int(train_counts[0])}, POS(1)={int(train_counts[1])}")
	logger.info(f"Val class counts:   NEG(0)={int(te_counts[0])}, POS(1)={int(te_counts[1])}")
	logger.info(f"Num features={len(num_cols)} Cat features={len(cat_cols)}")

	# Standardize numeric features using train statistics
	if Xn_tr.shape[1] > 0:
		mean = Xn_tr.mean(axis=0, keepdims=True)
		std = Xn_tr.std(axis=0, keepdims=True)
		std[std == 0] = 1.0
		Xn_tr = (Xn_tr - mean) / std
		Xn_te = (Xn_te - mean) / std

	# Class weights and/or oversampling (binary)
	class_counts = np.bincount(y_tr, minlength=2)
	inv_freq = (len(y_tr) / (class_counts + 1e-6))
	class_weights_np = 2 * (inv_freq / inv_freq.sum())  # mean ~1
	sampler = None
	if args.oversample:
		sample_weights = class_weights_np[y_tr]
		sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
		logger.info("Sampler: WeightedRandomSampler (oversampling)")

	# Datasets and loaders
	train_ds = ExoDataset(Xn_tr, Xc_tr, y_tr)
	test_ds = ExoDataset(Xn_te, Xc_te, y_te)
	if sampler is not None:
		train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=False)
	else:
		train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

	cat_dims = [max(2, int(max(2, len(m)))) for m in cat_maps.values()]
	num_features_len = Xn_tr.shape[1]
	# Binary head
	n_classes = 1

	# If loading a model, prefer the saved architecture hyperparameters
	arch_embed_dim = args.embed_dim
	arch_heads = args.heads
	arch_layers = args.layers
	arch_dropout = args.dropout
	if loaded_artifacts is not None:
		arch_embed_dim = int(loaded_artifacts.get("embed_dim", arch_embed_dim))
		arch_heads = int(loaded_artifacts.get("heads", arch_heads))
		arch_layers = int(loaded_artifacts.get("layers", arch_layers))
		arch_dropout = float(loaded_artifacts.get("dropout", arch_dropout))

	model = TabularTransformer(
		cat_dims=cat_dims,
		num_features_len=num_features_len,
		embed_dim=arch_embed_dim,
		n_heads=arch_heads,
		n_layers=arch_layers,
		dropout=arch_dropout,
		n_classes=n_classes,
	).to(DEVICE)
	logger.info(f"Model: FT-Transformer embed_dim={arch_embed_dim} heads={arch_heads} layers={arch_layers} dropout={arch_dropout}")

	# If a pretrained model is provided, load weights
	if args.load_model_dir:
		state_path = Path(args.load_model_dir) / "tabular_transformer.pt"
		assert state_path.exists(), f"No model weights found at {state_path}"
		model.load_state_dict(torch.load(state_path, map_location=DEVICE))
		logger.info(f"Loaded pretrained weights from {state_path}")

	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	# Scheduler (ReduceLROnPlateau on val F1)
	scheduler = None
	if args.scheduler == "plateau":
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience
		)
	# Avoid double balancing: prefer oversampling if both are set
	use_class_weight = args.class_weight and not args.oversample
	if use_class_weight:
		# BCE with positive class weighting
		pos_weight = torch.tensor(class_weights_np[1] / max(1e-6, class_weights_np[0]), dtype=torch.float32, device=DEVICE)
		criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		logger.info("Loss: BCEWithLogits with pos_weight={:.3f}".format(float(pos_weight)))
	else:
		criterion = nn.BCEWithLogitsLoss()
		logger.info("Loss: BCEWithLogits")

	best_f1 = -1.0
	best_state = None
	no_improve_epochs = 0
	logger.info(f"Optimizer: AdamW lr={args.lr} weight_decay={args.weight_decay}")
	if scheduler is not None:
		logger.info(f"Scheduler: ReduceLROnPlateau factor={args.lr_factor} patience={args.lr_patience}")
	logger.info(f"Training: epochs={args.epochs} batch_size={args.batch_size} steps_per_epoch={len(train_loader)}")
	# Optionally skip training and run validation only
	start_epoch = 0
	end_epoch = args.epochs
	if args.eval_only:
		start_epoch = 0
		end_epoch = 0

	for epoch in range(start_epoch, end_epoch):
		model.train()
		loss_sum = 0.0
		step = 0
		for xb_num, xb_cat, yb in train_loader:
			xb_num = xb_num.to(DEVICE)
			xb_cat = xb_cat.to(DEVICE)
			yb = yb.to(DEVICE)
			optimizer.zero_grad()
			logits = model(xb_num, xb_cat).squeeze(1)
			loss = criterion(logits, yb.float())
			loss.backward()
			# Optional gradient clipping
			if args.clip_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
			optimizer.step()
			loss_sum += loss.item()
			step += 1
			if args.log_interval > 0 and (step % args.log_interval == 0):
				current_lr = optimizer.param_groups[0]["lr"]
				logger.info(f"Epoch {epoch+1} Step {step}/{len(train_loader)} - lr {current_lr:.6f} - loss {loss_sum/step:.4f}")
		model.eval()
		y_true, y_pred, y_prob = [], [], []
		with torch.no_grad():
			for xb_num, xb_cat, yb in test_loader:
				xb_num = xb_num.to(DEVICE)
				xb_cat = xb_cat.to(DEVICE)
				logits = model(xb_num, xb_cat).squeeze(1)
				prob = torch.sigmoid(logits).cpu().numpy()
				pred = (prob >= 0.5).astype(int)
				y_prob.extend(prob.tolist())
				y_pred.extend(pred.tolist())
				y_true.extend(yb.numpy())
		f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
		try:
			auc = roc_auc_score(y_true, y_prob)
		except Exception:
			auc = float("nan")
		current_lr = optimizer.param_groups[0]["lr"]
		logger.info(f"Epoch {epoch+1}/{args.epochs} - lr {current_lr:.6f} - loss {loss_sum/max(1,len(train_loader)):.4f} - val f1 {f1:.4f} - val auc {auc:.4f}")

		# LR scheduler step
		if scheduler is not None:
			scheduler.step(f1)

		# Track best
		if f1 > best_f1 + 1e-6:
			best_f1 = f1
			best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
			no_improve_epochs = 0
		else:
			no_improve_epochs += 1
			if args.early_stopping_patience > 0 and no_improve_epochs >= args.early_stopping_patience:
				logger.warning("Early stopping: no improvement on val F1")
				break

	# Load best state (if any) before final save/eval (only when training ran)
	if best_state is not None and not args.eval_only:
		model.load_state_dict(best_state)

	# Save artifacts only when training (avoid overwriting pretrained model during eval-only)
	if not args.eval_only:
		out_dir = Path(args.out)
		out_dir.mkdir(parents=True, exist_ok=True)
		artifacts = {
			"num_cols": num_cols,
			"cat_cols": cat_cols,
			"cat_maps": {k: {str(kk): int(vv) for kk, vv in m.items()} for k, m in cat_maps.items()},
			"labels": ["FALSE POSITIVE", "CONFIRMED"],
			"target": "prob_confirmed",
			"embed_dim": arch_embed_dim,
			"heads": arch_heads,
			"layers": arch_layers,
			"dropout": arch_dropout,
		}
		(torch.save(model.state_dict(), out_dir / "tabular_transformer.pt"))
		with (out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
			json.dump(artifacts, f, indent=2)

	# Final validation evaluation (works for both training and eval-only paths)
	model.eval()
	y_true, y_pred, y_prob = [], [], []
	with torch.no_grad():
		for xb_num, xb_cat, yb in test_loader:
			xb_num = xb_num.to(DEVICE)
			xb_cat = xb_cat.to(DEVICE)
			logits = model(xb_num, xb_cat).squeeze(1)
			prob = torch.sigmoid(logits).cpu().numpy()
			pred = (prob >= 0.5).astype(int)
			y_prob.extend(prob.tolist())
			y_pred.extend(pred.tolist())
			y_true.extend(yb.numpy())
	val_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
	try:
		val_auc = roc_auc_score(y_true, y_prob)
	except Exception:
		val_auc = float("nan")
	logger.info(f"Final validation - f1 {val_f1:.4f} - auc {val_auc:.4f}")
	logger.info("Binary validation report:\n" + classification_report(y_true, y_pred, target_names=["FALSE POSITIVE", "CONFIRMED"], zero_division=0))

	# Scoring: compute probability for all rows (including CANDIDATE) and write CSV
	model.eval()
	with torch.no_grad():
		# IMPORTANT: apply the same standardization used for train/val to all rows
		if num_features_len > 0:
			X_num_all = (X_num - mean) / std
		else:
			X_num_all = X_num
		# Batch scoring to avoid OOM/memory pressure
		full_ds = ExoDataset(X_num_all, X_cat, y=None)
		full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
		probs_list: List[np.ndarray] = []
		for xb_num, xb_cat, _ in full_loader:
			xb_num = xb_num.to(DEVICE)
			xb_cat = xb_cat.to(DEVICE)
			logits = model(xb_num, xb_cat).squeeze(1)
			probs = torch.sigmoid(logits).cpu().numpy()
			probs_list.append(probs)
		probs_all = np.concatenate(probs_list, axis=0)

	scores_df = pd.DataFrame({
		"prob_confirmed": probs_all,
		"label": df[label_col].astype(str).values,
	})
	# If available, attach kepid for reference
	if "kepid" in df.columns:
		scores_df.insert(0, "kepid", df["kepid"].values)
	
	if args.score_only_candidates:
		scores_out_df = scores_df[scores_df["label"] == "CANDIDATE"].copy()
	else:
		scores_out_df = scores_df

	scores_out_df = scores_out_df.sort_values("prob_confirmed", ascending=False)
	out_scores = Path(args.scores_out)
	out_scores.parent.mkdir(parents=True, exist_ok=True)
	scores_out_df.to_csv(out_scores, index=False)
	logger.info(f"Wrote scores to {out_scores} (rows={len(scores_out_df)})")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train a FT-Transformer ranking model (probability of real planet)")
	parser.add_argument("--data", type=str, default="data/frames/kepler_summary_with_labels.parquet")
	parser.add_argument("--out", type=str, default="ranking/v0.1")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--embed_dim", type=int, default=64)
	parser.add_argument("--heads", type=int, default=4)
	parser.add_argument("--layers", type=int, default=3)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--sample_frac", type=float, default=0.2)
	parser.add_argument("--class_weight", action="store_true", help="Use class-weighted CrossEntropyLoss")
	parser.add_argument("--oversample", action="store_true", help="Use WeightedRandomSampler on training set")
	parser.add_argument("--clip_grad_norm", type=float, default=0.0, help="Max global grad norm for clipping (0 to disable)")
	parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay")
	parser.add_argument("--label_smoothing", type=float, default=0.0, help="CrossEntropy label smoothing (0-1)")
	parser.add_argument("--scheduler", type=str, default="plateau", choices=["none","plateau"], help="LR scheduler strategy")
	parser.add_argument("--lr_factor", type=float, default=0.5, help="LR reduction factor for plateau scheduler")
	parser.add_argument("--lr_patience", type=int, default=2, help="Epochs to wait before LR reduction (plateau)")
	parser.add_argument("--early_stopping_patience", type=int, default=5, help="Stop if no val F1 improvement for N epochs (0 disables)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--log_file", type=str, default="", help="Optional log file path; creates directories if needed")
	parser.add_argument("--log_interval", type=int, default=100, help="Steps between batch logs (0 to disable)")
	parser.add_argument("--verbose", action="store_true", help="Enable DEBUG level logging")
	parser.add_argument("--scores_out", type=str, default="ranking/v0.1/scores.csv", help="CSV to write probabilities")
	parser.add_argument("--load_model_dir", type=str, default="", help="Directory containing pretrained model and feature_config.json to evaluate/score")
	parser.add_argument("--eval_only", action="store_true", help="Skip training and run validation (and scoring) using a loaded model")
	parser.add_argument("--score_only_candidates", action="store_true", help="Write scores only for CANDIDATE rows")
	args = parser.parse_args()

	# Configure logging
	log_level = logging.DEBUG if args.verbose else logging.INFO
	root_logger = logging.getLogger()
	root_logger.handlers.clear()
	root_logger.setLevel(log_level)
	console = logging.StreamHandler()
	console.setLevel(log_level)
	console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
	root_logger.addHandler(console)
	if args.log_file:
		lf = Path(args.log_file)
		lf.parent.mkdir(parents=True, exist_ok=True)
		fh = logging.FileHandler(lf)
		fh.setLevel(log_level)
		fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
		root_logger.addHandler(fh)

	root_logger.info("Starting trainer with args: %s", vars(args))

	# Set seeds for reproducibility
	import random
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	cudnn.deterministic = True
	cudnn.benchmark = False

	train(args)
