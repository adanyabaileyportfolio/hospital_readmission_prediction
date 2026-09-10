from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_models import load_and_split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = PROJECT_ROOT / "models"
TABLES_PATH = PROJECT_ROOT / "reports" / "tables"

MODEL_FILES = {
    "Dummy classifier": "dummy_classifier.joblib",
    "Logistic regression": "logistic_regression.joblib",
    "Random forest": "random_forest.joblib",
}

THRESHOLDS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def calculate_metrics(
    model_name,
    y_true,
    probabilities,
    threshold
):
    """Calculate discrimination and classification metrics."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "specificity": tn / (tn + fp),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def evaluate_models():
    """Evaluate saved models on the patient-level test set."""

    _, X_test, _, y_test = load_and_split_data()

    TABLES_PATH.mkdir(parents=True, exist_ok=True)

    default_records = []
    threshold_records = []

    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_PATH / filename
        model = joblib.load(model_path)

        probabilities = model.predict_proba(X_test)[:, 1]

        default_records.append(
            calculate_metrics(
                model_name,
                y_test,
                probabilities,
                threshold=0.50
            )
        )

        if model_name != "Dummy classifier":
            for threshold in THRESHOLDS:
                threshold_records.append(
                    calculate_metrics(
                        model_name,
                        y_test,
                        probabilities,
                        threshold
                    )
                )

    default_results = pd.DataFrame(default_records)
    threshold_results = pd.DataFrame(threshold_records)

    default_results.to_csv(
        TABLES_PATH / "model_evaluation_default_threshold.csv",
        index=False
    )

    threshold_results.to_csv(
        TABLES_PATH / "model_threshold_comparison.csv",
        index=False
    )

    display_columns = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "brier_score",
    ]

    print("Default-threshold results:")
    print(
        default_results[display_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nThreshold analysis saved:")
    print(
        TABLES_PATH / "model_threshold_comparison.csv"
    )


if __name__ == "__main__":
    evaluate_models()