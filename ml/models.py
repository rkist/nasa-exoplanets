import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, classification_report, precision_score, confusion_matrix, precision_recall_curve

# Import models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ====================================================
# 🔹 Abstract Base Model
# ====================================================

class BaseModel:
    def __init__(self, model=None, scale_data=True):
        self.model = model
        self.scaler = None
        self.scale_data = scale_data

    # ----------------------------
    # Fit / Train
    # ----------------------------
    def fit(self, X_train, y_train):
        if self.scale_data:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    # ----------------------------
    # Predict
    # ----------------------------
    def predict(self, X):
        if self.scale_data and self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    # ----------------------------
    # Evaluate
    # ----------------------------
    def evaluate(self, X_test, y_test, model_name="Model",  pos_label=0):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=pos_label)
        print(f"\n📊 {model_name} Evaluation:")
        print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)
        return acc, rec, cm

    def get_threshold(self, X_test, y_test, pos_label=0, metric="f1"):
        """
        Compute performance across multiple thresholds and return them sorted by best metric (default: F1).
        Returns a list of dicts, each containing:
        threshold, accuracy, recall, precision, f1, confusion_matrix
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{type(self.model).__name__} does not support predict_proba().")

        proba = self.model.predict_proba(X_test)

        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            classes = np.array(classes)
            if pos_label in classes:
                index = int(np.where(classes == pos_label)[0][0])
            else:
                index = 0
        else:
            index = 0

        y_prob = proba[:, index]

        _, _, thresholds = precision_recall_curve(y_test, y_prob, pos_label=pos_label)
        results = []

        for thr in thresholds:
            negative_label = 1 if pos_label == 0 else 0
            y_pred = np.where(y_prob >= thr, pos_label, negative_label)
            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, pos_label=pos_label)
            pre = precision_score(y_test, y_pred, pos_label=pos_label)
            f1 = 2 * (pre * rec) / (pre + rec + 1e-6)
            cm = confusion_matrix(y_test, y_pred)
            results.append({
                "threshold": float(thr),
                "accuracy": float(acc),
                "recall": float(rec),
                "precision": float(pre),
                "f1": float(f1),
                "confusion_matrix": cm.tolist()
            })

        # Sort results by best F1 (default) or metric specified
        results = sorted(results, key=lambda x: x.get(metric, 0), reverse=True)
        return results

    # ----------------------------
    # Save / Load
    # ----------------------------
    def save(self, model_path, scaler_path=None):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        if self.scale_data and self.scaler is not None and scaler_path:
            joblib.dump(self.scaler, scaler_path)
        print(f"💾 Model saved to {model_path}")

    def load(self, model_path, scaler_path=None):
        self.model = joblib.load(model_path)
        if self.scale_data and scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
        print(f"📂 Loaded model from {model_path}")


# ====================================================
# 🔹 Specific Model Implementations
# ====================================================

class RandomForestModel(BaseModel):
    def __init__(self, scale_data=True):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=False,
            random_state=42
        )
        super().__init__(model, scale_data)


class XGBoostModel(BaseModel):
    def __init__(self, scale_data=True):
        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=10,
            subsample=1.0,
            colsample_bytree=0.6,
            eval_metric='logloss',
            random_state=42
        )
        super().__init__(model, scale_data)


class LightGBMModel(BaseModel):
    def __init__(self, scale_data=True):
        model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42,
            min_gain_to_split=0.0,  # allow small gain splits
            verbose=-1,  # suppress warnings
        )
        super().__init__(model, scale_data)


class CatBoostModel(BaseModel):
    def __init__(self, scale_data=True):
        model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=5,
            verbose=False,
            random_state=42
        )
        super().__init__(model, scale_data)
