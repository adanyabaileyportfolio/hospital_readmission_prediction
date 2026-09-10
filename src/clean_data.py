from pathlib import Path

import pandas as pd
from build_database import load_lookup_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
CLEAN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hospital_readmissions_clean.csv"
)
AUDIT_PATH = PROJECT_ROOT / "reports" / "tables" / "cleaning_audit.csv"


# Discharge codes representing death or hospice care
EXCLUDED_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

# Variables excluded because of extreme missingness or limited baseline utility
HIGH_MISSINGNESS_COLUMNS = [
    "weight",
    "payer_code",
    "medical_specialty",
]


def group_diagnosis(code):
    """Group an ICD-9 diagnosis code into a broader clinical category."""

    if pd.isna(code):
        return "Missing"

    code = str(code).strip()

    # Supplementary ICD-9 codes begin with E or V
    if code.startswith(("E", "V")):
        return "Other"

    try:
        numeric_code = float(code)
    except ValueError:
        return "Other"

    if 390 <= numeric_code <= 459 or numeric_code == 785:
        return "Circulatory"
    if 460 <= numeric_code <= 519 or numeric_code == 786:
        return "Respiratory"
    if 520 <= numeric_code <= 579 or numeric_code == 787:
        return "Digestive"
    if 250 <= numeric_code < 251:
        return "Diabetes"
    if 800 <= numeric_code <= 999:
        return "Injury"
    if 710 <= numeric_code <= 739:
        return "Musculoskeletal"
    if 580 <= numeric_code <= 629 or numeric_code == 788:
        return "Genitourinary"
    if 140 <= numeric_code <= 239:
        return "Neoplasms"

    return "Other"


def add_audit_record(audit, step, before, after):
    """Record how many rows were removed during a cleaning step."""

    audit.append({
        "step": step,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
    })


def clean_data():
    """Clean the raw hospital-encounter dataset."""

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        RAW_DATA_PATH,
        na_values=["?"],
        low_memory=False
    )

    audit = []
    initial_rows = len(data)

    # Remove exact duplicate encounter records, if any
    before = len(data)
    data = data.drop_duplicates()
    add_audit_record(
        audit,
        "Remove exact duplicate rows",
        before,
        len(data)
    )

    # Encounter IDs should identify unique hospital encounters
    before = len(data)
    data = data.drop_duplicates(subset="encounter_id", keep="first")
    add_audit_record(
        audit,
        "Remove duplicate encounter IDs",
        before,
        len(data)
    )

    # Remove invalid gender records
    before = len(data)
    data = data[data["gender"] != "Unknown/Invalid"].copy()
    add_audit_record(
        audit,
        "Remove Unknown/Invalid gender",
        before,
        len(data)
    )

    # Remove encounters ending in death or hospice care
    before = len(data)
    data = data[
        ~data["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES)
    ].copy()
    add_audit_record(
        audit,
        "Remove death and hospice discharges",
        before,
        len(data)
    )

    # Create the binary 30-day readmission target
    data["readmitted_30d"] = (
        data["readmitted"] == "<30"
    ).astype("int8")

    # Create broader diagnosis categories
    for diagnosis_column in ["diag_1", "diag_2", "diag_3"]:
        new_column = f"{diagnosis_column}_group"
        data[new_column] = data[diagnosis_column].apply(group_diagnosis)

    # Rename treatment indicators for clarity
    data = data.rename(columns={
        "change": "medication_change",
        "diabetesMed": "diabetes_medication"
    })
    # Decode the hospital ID fields using the UCI lookup tables
    lookup_tables = load_lookup_tables()

    lookup_specs = {
        "admission_types": (
            "admission_type_id",
            "admission_type"
        ),
        "discharge_dispositions": (
            "discharge_disposition_id",
            "discharge_disposition"
        ),
        "admission_sources": (
            "admission_source_id",
            "admission_source"
        ),
    }

    for table_name, (id_column, description_column) in lookup_specs.items():
        lookup = lookup_tables[table_name].rename(columns={
            "id": id_column,
            "description": description_column
        })

        data = data.merge(
            lookup,
            on=id_column,
            how="left"
        )
    # Remove columns with substantial missingness from the cleaned model file
    data = data.drop(columns=HIGH_MISSINGNESS_COLUMNS)

    # Preserve repeated patient encounters for analysis.
    # patient_nbr will later be used for leakage-safe group splitting.
    data = data.sort_values("encounter_id").reset_index(drop=True)
    # Validate the cleaned dataset before saving
    assert data["encounter_id"].is_unique
    assert not data["gender"].eq("Unknown/Invalid").any()
    assert not data["discharge_disposition_id"].isin(
        EXCLUDED_DISCHARGE_CODES
    ).any()
    assert set(data["readmitted_30d"].unique()).issubset({0, 1})
    assert not any(
        column in data.columns
        for column in HIGH_MISSINGNESS_COLUMNS
    )

    required_decoded_columns = [
        "admission_type",
        "discharge_disposition",
        "admission_source",
    ]

    assert all(
        column in data.columns
        for column in required_decoded_columns
    )

    print("All cleaning validation checks passed.")
    
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(AUDIT_PATH, index=False)
    data.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"Initial rows: {initial_rows:,}")
    print(f"Cleaned rows: {len(data):,}")
    print(f"Rows removed: {initial_rows - len(data):,}")
    print(f"Unique patients: {data['patient_nbr'].nunique():,}")
    print(
        "30-day readmission rate: "
        f"{data['readmitted_30d'].mean() * 100:.2f}%"
    )
    print(f"Cleaned file: {CLEAN_DATA_PATH}")
    print(f"Audit file: {AUDIT_PATH}")

    print("\nCleaning audit:")
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    clean_data()