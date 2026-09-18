"""Train and evaluate a reproducible customer-churn classifier."""

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def make_dataset(rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    tenure = rng.integers(1, 73, rows)
    monthly = rng.normal(70, 25, rows).clip(18, 150)
    support = rng.poisson(1.8, rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "bank-transfer", "electronic-check"], rows)
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[0.5, 0.4, 0.1])

    logit = (
        -1.5
        - 0.035 * tenure
        + 0.018 * (monthly - 70)
        + 0.32 * support
        + 1.0 * (contract == "month-to-month")
        + 0.45 * (payment == "electronic-check")
        + 0.35 * (internet == "fiber")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charge": monthly.round(2),
        "support_calls": support,
        "contract": contract,
        "payment_method": payment,
        "internet_service": internet,
        "churn": churn,
    })


def build_pipeline(model):
    numeric = ["tenure_months", "monthly_charge", "support_calls"]
    categorical = ["contract", "payment_method", "internet_service"]
    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = make_dataset()
    data.to_csv(OUTPUT_DIR / "sample_customers.csv", index=False)

    X = data.drop(columns="churn")
    y = data["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    scores = {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        scores[name] = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()

    best_name = max(scores, key=scores.get)
    pipeline = build_pipeline(candidates[best_name])
    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "selected_model": best_name,
        "cross_validation_roc_auc": round(scores[best_name], 4),
        "test_roc_auc": round(roc_auc_score(y_test, probability), 4),
        "test_accuracy": round(accuracy_score(y_test, prediction), 4),
        "classification_report": classification_report(y_test, prediction, output_dict=True),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual": y_test, "probability": probability, "prediction": prediction}).to_csv(
        OUTPUT_DIR / "predictions.csv", index=False
    )
    joblib.dump(pipeline, OUTPUT_DIR / "churn_model.joblib")

    ConfusionMatrixDisplay.from_predictions(y_test, prediction, cmap="Blues")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probability)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    if best_name == "random_forest":
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance = pipeline.named_steps["model"].feature_importances_
        order = np.argsort(importance)[-10:]
        plt.barh(names[order], importance[order])
        plt.title("Top churn drivers")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
        plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
