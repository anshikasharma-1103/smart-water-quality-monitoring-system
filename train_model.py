"""
Reproducible Machine Learning Training & Evaluation Pipeline
Smart Water Quality Monitoring System
Tagline: "Monitoring Today, Protecting Tomorrow."

Trains a Random Forest Classifier on reproducible synthetic multi-parameter data.
Saves model artifacts via Joblib and generates formal evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

MODEL_PATH = "water_model.joblib"
RANDOM_SEED = 42

FEATURE_NAMES = ["ph", "turbidity", "tds", "conductivity", "temperature"]
CLASS_NAMES = ["Safe-range indication", "Caution", "Unsafe"]


def generate_synthetic_dataset(num_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes a realistic, multi-modal water quality screening dataset.
    Note: Explicitly labelled as synthetic/demo data for educational and prototyping purposes.
    """
    np.random.seed(seed)
    data = []

    for _ in range(num_samples):
        # Sample mode: 45% safe, 30% caution, 25% unsafe
        mode = np.random.choice(["safe", "caution", "unsafe"], p=[0.45, 0.30, 0.25])

        if mode == "safe":
            ph = np.random.uniform(6.6, 8.4)
            turbidity = np.random.exponential(1.2)
            turbidity = min(turbidity, 4.8)
            tds = np.random.uniform(80.0, 450.0)
            conductivity = tds * np.random.uniform(1.5, 1.65)
            temperature = np.random.uniform(14.0, 26.0)
            label = "Safe-range indication"

        elif mode == "caution":
            # Marginally elevated or depressed
            stress_param = np.random.choice(["ph", "turbidity", "tds", "temp"])
            ph = np.random.uniform(6.5, 8.5)
            turbidity = np.random.uniform(1.0, 4.0)
            tds = np.random.uniform(150.0, 480.0)
            temperature = np.random.uniform(16.0, 26.0)

            if stress_param == "ph":
                ph = np.random.choice([np.random.uniform(6.0, 6.49), np.random.uniform(8.51, 8.95)])
            elif stress_param == "turbidity":
                turbidity = np.random.uniform(5.1, 9.5)
            elif stress_param == "tds":
                tds = np.random.uniform(501.0, 850.0)
            else:
                temperature = np.random.uniform(28.1, 33.0)

            conductivity = tds * np.random.uniform(1.5, 1.65)
            label = "Caution"

        else:  # unsafe
            fail_type = np.random.choice(["acidic", "alkaline", "turbid", "heavy_saline"])
            if fail_type == "acidic":
                ph = np.random.uniform(3.5, 5.8)
                turbidity = np.random.uniform(3.0, 25.0)
                tds = np.random.uniform(200.0, 800.0)
            elif fail_type == "alkaline":
                ph = np.random.uniform(9.2, 12.0)
                turbidity = np.random.uniform(1.0, 15.0)
                tds = np.random.uniform(300.0, 900.0)
            elif fail_type == "turbid":
                ph = np.random.uniform(6.2, 8.2)
                turbidity = np.random.uniform(12.0, 75.0)
                tds = np.random.uniform(250.0, 700.0)
            else:
                ph = np.random.uniform(6.5, 8.5)
                turbidity = np.random.uniform(4.0, 30.0)
                tds = np.random.uniform(950.0, 2400.0)

            conductivity = tds * np.random.uniform(1.6, 1.8)
            temperature = np.random.uniform(18.0, 32.0)
            label = "Unsafe"

        data.append({
            "ph": round(float(ph), 2),
            "turbidity": round(float(turbidity), 2),
            "tds": round(float(tds), 1),
            "conductivity": round(float(conductivity), 1),
            "temperature": round(float(temperature), 2),
            "quality_class": label
        })

    return pd.DataFrame(data)


def train_and_save_model() -> Dict[str, Any]:
    """
    Trains the Random Forest model and persists to disk.
    Returns full metrics dictionary.
    """
    df = generate_synthetic_dataset(num_samples=1600, seed=RANDOM_SEED)

    X = df[FEATURE_NAMES]
    y = df["quality_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

    importances = dict(zip(FEATURE_NAMES, [float(x) for x in clf.feature_importances_]))

    model_artifact = {
        "model": clf,
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances,
            "test_sample_count": len(y_test)
        },
        "is_demo_model": True
    }

    joblib.dump(model_artifact, MODEL_PATH)

    return model_artifact["metrics"]


def load_or_train_model():
    """Loads saved model artifact or trains fresh if absent."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_water_quality(reading: Dict[str, float]) -> Dict[str, Any]:
    """
    Classifies a water sample using the Random Forest model and generates
    an explainable causal breakdown of the prediction.
    """
    model_obj = load_or_train_model()
    clf = model_obj["model"]

    # Input sanitization
    raw_vals = []
    missing = []
    for f in FEATURE_NAMES:
        if f not in reading or reading[f] is None:
            missing.append(f)
            raw_vals.append(0.0)
        else:
            try:
                raw_vals.append(float(reading[f]))
            except (ValueError, TypeError):
                missing.append(f)
                raw_vals.append(0.0)

    if missing:
        return {
            "prediction": "Unknown",
            "confidence": 0.0,
            "error": f"Missing features: {', '.join(missing)}",
            "explanations": ["Model prediction aborted due to missing parameter data."],
            "contributing_factors": []
        }

    X_in = np.array([raw_vals])
    pred = clf.predict(X_in)[0]
    probs = clf.predict_proba(X_in)[0]
    classes = list(clf.classes_)
    confidence = float(probs[classes.index(pred)])

    # Explainable reasoning breakdown
    from water_quality import DEFAULT_REFERENCE_RANGES
    contributing_factors = []
    recommendations = []

    for f in FEATURE_NAMES:
        val = reading[f]
        cfg = DEFAULT_REFERENCE_RANGES[f]
        if val < cfg["min_safe"] or val > cfg["max_safe"]:
            direction = "low" if val < cfg["min_safe"] else "high"
            contributing_factors.append({
                "feature": cfg["name"],
                "value": val,
                "unit": cfg["unit"],
                "reference_range": f"{cfg['min_safe']} - {cfg['max_safe']}",
                "status": f"Out of bounds ({direction})",
                "importance_weight": model_obj["metrics"]["feature_importances"].get(f, 0.2)
            })

    if pred == "Unsafe":
        recommendations.append("Immediate action required: Do not utilize water for domestic consumption.")
        recommendations.append("Perform full chemical titration and microbiological culture test.")
    elif pred == "Caution":
        recommendations.append("Marginal parameter detected: Repeat measurement in 15 minutes.")
        recommendations.append("Check particulate filtration cartridges and inlet aerators.")
    else:
        recommendations.append("All physical parameters align with preliminary reference screening standards.")
        recommendations.append("Note: Standard laboratory safety protocols still apply for microbiological pathogens.")

    return {
        "prediction": pred,
        "confidence_pct": round(confidence * 100, 1),
        "class_probabilities": {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)},
        "contributing_factors": contributing_factors,
        "recommendations": recommendations,
        "model_disclaimer": (
            "NOTICE: Model trained on synthetic demonstration data with a fixed random seed (42). "
            "Results provide preliminary physical screening classification only, not certified drinking-water safety approval."
        )
    }


if __name__ == "__main__":
    print("--- Training Water Quality Random Forest Model ---")
    metrics = train_and_save_model()
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"Feature Importances: {metrics['feature_importances']}")
