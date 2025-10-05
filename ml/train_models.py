# main.py
import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

from ml.models import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
)


# ====================================================
# 🔹 Train & Save Utility
# ====================================================
def train_and_save(model_class, X_train, X_test, y_train, y_test, out_dir, model_name, scale, pos_label=0):
    """Train, evaluate, and save a model with markdown .info report."""
    print(f"\nTraining {model_name}...")
    model = model_class(scale_data=scale)
    model.fit(X_train, y_train)

    # Evaluation
    acc, rec, cm = model.evaluate(X_test, y_test, model_name, pos_label=pos_label)

    # Threshold search
    thr_results = model.get_threshold(X_test, y_test, pos_label=pos_label)

    # Best threshold summary
    best_thr = thr_results[0]

    # Save model & scaler
    model_path = Path(out_dir) / f"{model_name}_model.pkl"
    scaler_path = Path(out_dir) / f"{model_name}_scaler.pkl" if scale else None
    model.save(model_path, scaler_path)

    # Build Markdown report
    info_path = Path(out_dir) / f"{model_name}.info"
    lines = [
        f"# Model Report: {model_name}",
        f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Accuracy: {acc:.4f}",
        f"- Recall: {rec:.4f}",
        "",
        "## Confusion Matrix (default threshold)",
        str(cm),
        "",
        "## Best Threshold (Top Result)",
        f"- Threshold: {best_thr['threshold']:.4f}",
        f"- Accuracy: {best_thr['accuracy']:.4f}",
        f"- Recall: {best_thr['recall']:.4f}",
        f"- Precision: {best_thr['precision']:.4f}",
        f"- F1: {best_thr['f1']:.4f}",
        f"- Confusion Matrix: {best_thr['confusion_matrix']}",
        "",
        "## All Threshold Results (sorted by best F1)",
        "| Threshold | Accuracy | Recall | Precision | F1 | Confusion Matrix |",
        "|-----------|-----------|---------|------------|----|------------------|",
    ]

    for r in thr_results:
        cm_str = str(r['confusion_matrix']).replace("\n", "")
        lines.append(
            f"| {r['threshold']:.3f} | {r['accuracy']:.3f} | {r['recall']:.3f} | "
            f"{r['precision']:.3f} | {r['f1']:.3f} | {cm_str} |"
        )

    with open(info_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved → {info_path}")

    return {
        "name": model_name,
        "accuracy": float(acc),
        "recall": float(rec),
        "best_threshold": float(best_thr["threshold"]),
        "best_threshold_metrics": best_thr,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path) if scaler_path else None,
        "report_path": str(info_path)
    }


def _load_dataframe(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load a dataframe from a path or return a copy if already provided."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if data is None:
        raise ValueError("No dataset provided for training.")

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"📂 Loading dataset from {data_path}")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix in {".jsonl", ".json"}:
        df = pd.read_json(data_path, lines=True)
    else:
        df = pd.read_csv(data_path)

    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]],
    target_column: str,
    allowed_classes: Optional[Sequence[str]]
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    y = df[target_column]

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]

    X = df[feature_columns]

    if allowed_classes:
        allowed = [cls for cls in allowed_classes if cls is not None]
        invalid = [cls for cls in allowed if cls not in set(y.unique())]
        if invalid:
            raise ValueError(
                f"Invalid class values provided: {invalid}. Available classes: {list(y.unique())}"
            )
        mask = y.isin(allowed)
        X = X.loc[mask]
        y = y.loc[mask]
        print(f"⚠️ Using subset of classes: {allowed}")

    return X, y


def run_training(
    data: Union[str, Path, pd.DataFrame],
    feature_columns: Optional[Sequence[str]] = None,
    target: str = "label",
    out_dir: Union[str, Path] = "models_out",
    scale: bool = False,
    test_size: float = 0.2,
    classes: Optional[Sequence[str]] = None,
    pos_label: Optional[str] = None
):
    """Train all supported models and persist artifacts."""

    df = _load_dataframe(data)
    X, y = _prepare_xy(df, feature_columns, target, classes)

    if pos_label is None:
        raise ValueError("pos_label must be provided to define the positive class.")

    if pos_label not in set(y.unique()):
        raise ValueError(
            f"pos_label '{pos_label}' not found in target column values: {list(y.unique())}"
        )

    print("\n📊 Class distribution after filtering:")
    print(y.value_counts(normalize=True).round(3))

    y_binary = y.apply(lambda value: 0 if value == pos_label else 1)
    print(f"\n✅ Mapping '{pos_label}' to 0 and all other classes to 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    models = {
        "RandomForest": RandomForestModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel
    }

    for name, cls in models.items():
        results.append(
            train_and_save(cls, X_train, X_test, y_train, y_test, out_dir, name, scale, pos_label=0)
        )

    return results

# ====================================================
# 🔹 Main Function
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Train multiple ML models for exoplanet classification")

    parser.add_argument("--data", required=True, help="Path to dataset (CSV, JSONL, or Parquet)")
    parser.add_argument("--feature", nargs="+", default=None,
                        help="List of feature columns to use (default: all except target)")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--out", default="models_out", help="Directory to save trained models")
    parser.add_argument("--scale", action="store_true", help="Use StandardScaler (default: False)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size split ratio (default: 0.2)")
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of target class names to keep (e.g. CONFIRMED,'FALSE POSITIVE')"
    )
    parser.add_argument(
        "--pos_label",
        type=str,
        required=True,
        help="Which class to treat as the positive label for recall/f1 metrics (e.g. 'CONFIRMED')"
    )

    args = parser.parse_args()

    classes = [cls.strip() for cls in args.classes.split(",")] if args.classes else None

    results = run_training(
        data=args.data,
        feature_columns=args.feature,
        target=args.target,
        out_dir=args.out,
        scale=args.scale,
        test_size=args.test_size,
        classes=classes,
        pos_label=args.pos_label
    )

    summary = (
        pd.DataFrame(results)[["name", "accuracy", "recall", "best_threshold"]]
        .sort_values(by="accuracy", ascending=False)
    )

    print("\n🏁 Final Model Results:")
    print(summary.rename(columns={
        "name": "Model",
        "accuracy": "Accuracy",
        "recall": "Recall",
        "best_threshold": "Best Threshold"
    }))


if __name__ == "__main__":
    main()
