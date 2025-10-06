from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn


class _FeatureTokenizer(nn.Module):
    """Tokenize categorical and numerical features into a shared embedding space."""

    def __init__(self, cat_dims: Sequence[int], num_features: int, embed_dim: int) -> None:
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims])
        self.num_embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_features)])

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens: List[torch.Tensor] = []

        for idx, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, idx]))

        for idx, emb in enumerate(self.num_embeddings):
            tokens.append(emb(x_num[:, idx].unsqueeze(1)))

        if not tokens:
            # Ensure downstream modules receive a tensor with correct batch dimension
            return torch.zeros(x_num.size(0), 0, 1, device=x_num.device)

        return torch.stack(tokens, dim=1)


class _TabularTransformer(nn.Module):
    """Minimal FT-Transformer head for binary classification."""

    def __init__(
        self,
        cat_dims: Sequence[int],
        num_features: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.tokenizer = _FeatureTokenizer(cat_dims, num_features, embed_dim)
        seq_len = len(cat_dims) + num_features

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)

        if tokens.numel() == 0:
            # When the model only has the CLS token
            tokens = torch.zeros(x_num.size(0), 0, self.cls_token.size(-1), device=x_num.device)

        batch = tokens.size(0)
        cls = self.cls_token.expand(batch, -1, -1)
        pos = self.pos_embedding[:, : tokens.size(1) + 1, :]
        x = torch.cat([cls, tokens], dim=1) + pos
        x = self.encoder(x)
        x = self.norm(x[:, 0, :])
        return self.head(x)


class TorchTabularModel:
    """Wrapper that exposes a scikit-learn like interface for the FT-Transformer model."""

    def __init__(self, model_path: Path, config_path: Path, feature_columns: Sequence[str]) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.model_path = model_path
        self.config = config
        self.input_index = {name: idx for idx, name in enumerate(feature_columns)}
        self.num_cols: List[str] = list(config.get("num_cols", []))
        self.cat_cols: List[str] = list(config.get("cat_cols", []))

        if not self.num_cols and not self.cat_cols:
            raise ValueError("Configuration must define at least one feature column.")

        embed_dim = int(config.get("embed_dim", 128))
        n_heads = int(config.get("heads", 8))
        n_layers = int(config.get("layers", 4))
        dropout = float(config.get("dropout", 0.1))

        cat_maps: Dict[str, Dict[str, int]] = config.get("cat_maps", {}) or {}
        cat_dims: List[int] = [max(2, int(len(cat_maps.get(col, {})) or 1) + 1) for col in self.cat_cols]

        self.device = torch.device("cpu")
        self.model = _TabularTransformer(
            cat_dims=cat_dims,
            num_features=len(self.num_cols),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Pre-computed statistics for imputation and normalization
        medians = config.get("medians", {})
        means = config.get("means", {})
        stds = config.get("stds", {})

        self.medians = np.array([float(medians.get(col, 0.0)) for col in self.num_cols], dtype=np.float32)
        self.means = np.array([float(means.get(col, 0.0)) for col in self.num_cols], dtype=np.float32)
        self.stds = np.array([float(stds.get(col, 1.0)) for col in self.num_cols], dtype=np.float32)
        self.stds[self.stds == 0] = 1.0

        # Expose a scikit-learn-like signature
        self.classes_ = np.array([0, 1])

    def _reorder_numeric(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("Expected 2D array for inference inputs.")

        output = np.zeros((X.shape[0], len(self.num_cols)), dtype=np.float32)
        for idx, col in enumerate(self.num_cols):
            source = self.input_index.get(col)
            if source is None:
                raise KeyError(f"Feature '{col}' required by transformer model is missing from DataProcessor outputs.")
            output[:, idx] = X[:, source]
        return output

    def _prepare_numeric(self, X: np.ndarray) -> np.ndarray:
        reordered = self._reorder_numeric(X)
        # Replace NaNs, if any, with training medians
        if np.isnan(reordered).any():
            for col_idx in range(reordered.shape[1]):
                mask = np.isnan(reordered[:, col_idx])
                if mask.any():
                    reordered[mask, col_idx] = self.medians[col_idx]

        centered = reordered - self.means
        normalized = centered / self.stds
        return normalized

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0, 1), dtype=np.float32)

        X_num = self._prepare_numeric(X)
        X_cat = np.zeros((X_num.shape[0], len(self.cat_cols)), dtype=np.int64)

        with torch.no_grad():
            inputs_num = torch.from_numpy(X_num.astype(np.float32)).to(self.device)
            if X_cat.size:
                inputs_cat = torch.from_numpy(X_cat.astype(np.int64)).to(self.device)
            else:
                inputs_cat = torch.zeros(X_num.shape[0], 0, dtype=torch.long, device=self.device)
            logits = self.model(inputs_num, inputs_cat).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        return probs[:, None]


def _parse_float(line: str) -> Optional[float]:
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
    return float(match.group(1)) if match else None


@dataclass
class ModelMetadata:
    name: str
    accuracy: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1: Optional[float] = None
    threshold: float = 0.5
    confusion_matrix: Optional[List[List[int]]] = None
    report_path: Optional[Path] = None
    timestamp: Optional[str] = None


@dataclass
class LoadedModel:
    name: str
    estimator: Any
    scaler: Optional[Any]
    metadata: ModelMetadata


class ModelManager:
    positive_label: str = "Candidate"
    negative_label: str = "Likely False Positive"

    def __init__(self, models_dir: Path, feature_columns: Sequence[str]):
        self.models_dir = Path(models_dir)
        self.feature_columns = list(feature_columns)
        self.models: Dict[str, LoadedModel] = {}
        self.available_models: List[str] = []
        self.best_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_models(self) -> bool:
        """Load all model artifacts present in the models directory."""
        self.models.clear()
        candidates: List[tuple[float, str]] = []

        for model_path in sorted(self.models_dir.glob("*_model.*")):
            suffix = model_path.suffix.lower()
            name = model_path.stem.replace("_model", "")

            if name in self.models:
                continue

            if suffix == ".pkl":
                loaded = self._load_sklearn_model(name)
            elif suffix in {".pt", ".pth"}:
                loaded = self._load_torch_model(name)
            else:
                continue

            if loaded.metadata.accuracy is not None:
                candidates.append((loaded.metadata.accuracy, name))

        self.available_models = sorted(self.models.keys())

        if not self.models:
            self.best_model = None
            return False

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            self.best_model = candidates[0][1]
        else:
            self.best_model = self.available_models[0]

        return True

    def _load_sklearn_model(self, name: str) -> LoadedModel:
        model_path = self.models_dir / f"{name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        estimator = joblib.load(model_path)
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        metadata = self._read_metadata(name)
        metadata.name = name
        if metadata.report_path is None:
            metadata.report_path = self.models_dir / f"{name}.info"

        bundle = LoadedModel(name=name, estimator=estimator, scaler=scaler, metadata=metadata)
        self.models[name] = bundle
        return bundle

    def _load_torch_model(self, name: str) -> LoadedModel:
        model_path = self.models_dir / f"{name}_model.pt"
        if not model_path.exists():
            model_path = self.models_dir / f"{name}_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer weights not found for model '{name}'.")

        config_candidates = [
            self.models_dir / f"{name}_config.json",
            self.models_dir / f"{name}.config.json",
            self.models_dir / f"{name}.json",
        ]

        config_path = next((path for path in config_candidates if path.exists()), None)
        if config_path is None:
            raise FileNotFoundError(f"Configuration JSON not found for model '{name}'.")

        estimator = TorchTabularModel(model_path=model_path, config_path=config_path, feature_columns=self.feature_columns)

        metadata = self._read_metadata(name)
        metadata.name = name
        if metadata.report_path is None:
            metadata.report_path = self.models_dir / f"{name}.info"

        bundle = LoadedModel(name=name, estimator=estimator, scaler=None, metadata=metadata)
        self.models[name] = bundle
        return bundle

    def _read_metadata(self, name: str) -> ModelMetadata:
        info_path = self.models_dir / f"{name}.info"
        metadata = ModelMetadata(name=name, report_path=info_path)

        if not info_path.exists():
            return metadata

        with open(info_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            if line.startswith("- Timestamp:"):
                metadata.timestamp = line.split(":", 1)[1].strip()
            elif line.startswith("- Accuracy:"):
                metadata.accuracy = _parse_float(line)
            elif line.startswith("- Recall:"):
                metadata.recall = _parse_float(line)
            elif line.startswith("- Threshold:"):
                metadata.threshold = _parse_float(line) or metadata.threshold
            elif line.startswith("- Precision:"):
                metadata.precision = _parse_float(line)
            elif line.startswith("- F1:"):
                metadata.f1 = _parse_float(line)
            elif line.startswith("- Confusion Matrix:"):
                numbers = re.findall(r"\d+", line)
                if numbers:
                    values = list(map(int, numbers))
                    # Expecting 4 values for 2x2 matrix
                    if len(values) == 4:
                        metadata.confusion_matrix = [values[:2], values[2:]]

        return metadata

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if not self.models:
            raise RuntimeError("No model is currently loaded. Train or load a model first.")

    def _prepare(self, bundle: LoadedModel, X: np.ndarray) -> np.ndarray:
        if bundle.scaler is not None:
            return bundle.scaler.transform(X)
        return X

    def _get_positive_probability(self, bundle: LoadedModel, X: np.ndarray) -> np.ndarray:
        model = bundle.estimator
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 1:
                return proba[:, 0]

            classes = getattr(model, "classes_", None)
            if classes is not None:
                classes = np.array(classes)
                if 0 in classes:
                    index = int(np.where(classes == 0)[0][0])
                else:
                    index = 0
            else:
                index = 0

            return proba[:, index]
        raise AttributeError(f"Model '{type(model).__name__}' does not support probability predictions.")

    def predict_single(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        self._ensure_loaded()
        results: Dict[str, Dict[str, float]] = {}

        for name, bundle in self.models.items():
            prepared = self._prepare(bundle, X)
            proba = self._get_positive_probability(bundle, prepared)[0]
            threshold = bundle.metadata.threshold
            label = self.positive_label if proba >= threshold else self.negative_label
            confidence = proba if label == self.positive_label else 1 - proba
            results[name] = {
                "prediction": label,
                "confidence": float(confidence),
                "probability": float(proba),
                "threshold": float(threshold)
            }

        return results

    def predict_batch(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        self._ensure_loaded()
        outputs: Dict[str, Dict[str, Any]] = {}

        for name, bundle in self.models.items():
            prepared = self._prepare(bundle, X)
            proba = self._get_positive_probability(bundle, prepared)
            threshold = bundle.metadata.threshold
            mask = proba >= threshold
            labels = [self.positive_label if flag else self.negative_label for flag in mask]
            confidences = np.where(mask, proba, 1 - proba)

            outputs[name] = {
                "predictions": labels,
                "confidences": confidences,
                "probabilities": proba,
                "threshold": threshold
            }

        return outputs

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------
    def set_threshold(self, model_name: str, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError("Threshold must be between 0 and 1 (exclusive).")
        self._ensure_loaded()
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not loaded.")
        self.models[model_name].metadata.threshold = float(value)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        if not self.models:
            return {
                "message": "No model artifacts found.",
                "model_loaded": False
            }

        models_payload: List[Dict[str, Any]] = []
        best_model_metrics: Optional[Dict[str, Any]] = None

        for name in sorted(self.models.keys()):
            bundle = self.models[name]
            metadata = bundle.metadata
            confusion = None
            if metadata.confusion_matrix:
                matrix = metadata.confusion_matrix
                if len(matrix) > 1:
                    confusion = {
                        "tp": matrix[0][0],  # true candidate
                        "fp": matrix[1][0],  # false alarm
                        "fn": matrix[0][1]   # missed candidate
                    }
                else:
                    confusion = None

            payload = {
                "name": name,
                "accuracy": metadata.accuracy,
                "recall": metadata.recall,
                "precision": metadata.precision,
                "f1_score": metadata.f1,
                "threshold": metadata.threshold,
                "last_updated": metadata.timestamp,
                "confusion_matrix": confusion,
                "report_path": str(metadata.report_path) if metadata.report_path else None
            }
            models_payload.append(payload)

            if name == self.best_model:
                best_model_metrics = payload

        # Fallback if best model not identified
        if best_model_metrics is None and models_payload:
            best_model_metrics = models_payload[0]
            self.best_model = best_model_metrics["name"]

        return {
            "model_loaded": True,
            "models": models_payload,
            "best_model": self.best_model,
            "best_model_metrics": best_model_metrics,
            "available_models": self.available_models
        }

    def is_loaded(self) -> bool:
        return bool(self.models)
