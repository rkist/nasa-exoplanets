# main.py
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime
from models import RandomForestModel, XGBoostModel, LightGBMModel, CatBoostModel


# ====================================================
# 🔹 Train & Save Utility
# ====================================================
def train_and_save(model_class, X_train, X_test, y_train, y_test, out_dir, model_name, scale, pos_label=1):
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

    return model_name, acc, rec

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

    # ------------------------------
    # Load dataset
    # ------------------------------
    print(f"📂 Loading dataset from {args.data}")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    elif args.data.endswith(".jsonl"):
        df = pd.read_json(args.data, lines=True)
    elif args.data.endswith(".json"):
        df = pd.read_json(args.data, lines=True)
    else:
        df = pd.read_csv(args.data)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ------------------------------
    # Prepare features and target
    # ------------------------------
    y = df[args.target]
    if args.feature is None:
        X = df.drop(columns=[args.target])
    else:
        X = df[args.feature]

    unique_vals = y.unique()
    
    # handle multi-class datasets
    if len(unique_vals) > 2:
        if args.classes is None:
            raise ValueError(
                f"Target '{args.target}' has {len(unique_vals)} unique values: {list(unique_vals)}.\n"
                f"Please specify which classes to use via --classes "
                f"(e.g. --classes Confirmed,FalsePositive)"
            )
        else:
            allowed = [c.strip() for c in args.classes.split(",")]
            invalid = [c for c in allowed if c not in unique_vals]
            if invalid:
                raise ValueError(f"Invalid class values in --classes: {invalid}. Available: {list(unique_vals)}")

            y = y[y.isin(allowed)]
            X = X.loc[y.index]
            print(f"⚠️ Using subset of classes: {allowed}")

    # ✅ Print class distribution
    print("\n📊 Class distribution after filtering:")
    print(y.value_counts(normalize=True).round(3))

    if args.pos_label not in y.unique():
        raise ValueError(f"--pos_label '{args.pos_label}' not found in target column values: {list(y.unique())}")

    # Force consistent numeric encoding: positive → 1, negative → 0
    y = y.apply(lambda val: 1 if val == args.pos_label else 0)
    print(f"\n✅ Using '{args.pos_label}' as positive class (1), all others as 0")


    # ------------------------------
    # Split into train/test
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Train models
    # ------------------------------
    results = []
    models = {
        "RandomForest": RandomForestModel,
        "XGBoost": XGBoostModel,
        "LightGBM": LightGBMModel,
        "CatBoost": CatBoostModel
    }

    for name, cls in models.items():
        results.append(
            train_and_save(cls, X_train, X_test, y_train, y_test, args.out, name, args.scale)
        )

    # ------------------------------
    # Summary
    # ------------------------------
    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall"])
    print("\n🏁 Final Model Results:")
    print(df_results.sort_values(by="Accuracy", ascending=False))


if __name__ == "__main__":
    main()
