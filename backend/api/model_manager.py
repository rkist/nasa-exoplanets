from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np


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

        for model_path in self.models_dir.glob("*_model.pkl"):
            name = model_path.stem.replace("_model", "")
            loaded = self._load_model(name)
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

    def _load_model(self, name: str) -> LoadedModel:
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
