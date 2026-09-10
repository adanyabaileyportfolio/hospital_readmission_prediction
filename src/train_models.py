from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hospital_readmissions_clean.csv"
)

MODELS_PATH = PROJECT_ROOT / "models"
TRAINING_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "model_training_times.csv"
)

RANDOM_STATE = 42

EXCLUDED_FEATURES = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "readmitted_30d",
    "diag_1",
    "diag_2",
    "diag_3",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def load_and_split_data():
    """Load cleaned data and create a patient-level split."""

    data = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)

    feature_columns = [
        column
        for column in data.columns
        if column not in EXCLUDED_FEATURES
    ]

    X = data[feature_columns].copy()
    y = data["readmitted_30d"].copy()
    groups = data["patient_nbr"].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_index, test_index = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()
    y_train = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()

    train_patients = set(groups.iloc[train_index])
    test_patients = set(groups.iloc[test_index])

    assert train_patients.isdisjoint(test_patients)

    return X_train, X_test, y_train, y_test


def create_preprocessor(feature_columns):
    """Create preprocessing for numeric and categorical features."""

    categorical_features = [
        column
        for column in feature_columns
        if column not in NUMERIC_FEATURES
    ]

    numeric_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "one_hot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


def create_models(preprocessor):
    """Create the three model pipelines."""

    return {
        "Dummy classifier": Pipeline(steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                DummyClassifier(strategy="prior")
            )
        ]),

        "Logistic regression": Pipeline(steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "Random forest": Pipeline(steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE
                )
            )
        ])
    }


def train_models():
    """Train and save all model pipelines."""

    X_train, _, y_train, _ = load_and_split_data()

    preprocessor = create_preprocessor(X_train.columns)
    models = create_models(preprocessor)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    TRAINING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    filenames = {
        "Dummy classifier": "dummy_classifier.joblib",
        "Logistic regression": "logistic_regression.joblib",
        "Random forest": "random_forest.joblib"
    }

    training_records = []

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")

        start_time = perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed_time = perf_counter() - start_time

        model_path = MODELS_PATH / filenames[model_name]
        joblib.dump(pipeline, model_path)

        training_records.append({
            "model": model_name,
            "training_time_seconds": round(elapsed_time, 2)
        })

        print(
            f"Saved {model_name} after "
            f"{elapsed_time:.2f} seconds."
        )

    training_report = pd.DataFrame(training_records)
    training_report.to_csv(
        TRAINING_REPORT_PATH,
        index=False
    )

    print("\nTraining complete:")
    print(training_report.to_string(index=False))


if __name__ == "__main__":
    train_models()