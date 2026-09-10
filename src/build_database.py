from pathlib import Path
import csv
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"
MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "IDS_mapping.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "hospital_readmissions.db"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "01_create_database.sql"

LOOKUP_TABLES = {
    "admission_type_id": "admission_types",
    "discharge_disposition_id": "discharge_dispositions",
    "admission_source_id": "admission_sources",
}


def load_encounters():
    """Load the hospital encounters and create the binary outcome."""

    encounters = pd.read_csv(
        RAW_DATA_PATH,
        na_values=["?"],
        low_memory=False
    )

    encounters["readmitted_30d"] = (
        encounters["readmitted"] == "<30"
    ).astype(int)

    return encounters


def load_lookup_tables():
    """Separate the three lookup sections in IDS_mapping.csv."""

    records = {table_name: [] for table_name in LOOKUP_TABLES.values()}
    current_table = None

    with MAPPING_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            first_value = row[0].strip()

            if first_value in LOOKUP_TABLES:
                current_table = LOOKUP_TABLES[first_value]
                continue

            if not first_value or current_table is None:
                continue

            try:
                identifier = int(first_value)
            except ValueError:
                continue

            description = ",".join(row[1:]).strip()

            records[current_table].append({
                "id": identifier,
                "description": description
            })

    return {
        table_name: pd.DataFrame(rows)
        for table_name, rows in records.items()
    }


def build_database():
    """Create the SQLite database and load all project tables."""

    encounters = load_encounters()
    lookup_tables = load_lookup_tables()

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        encounters.to_sql(
            "encounters",
            connection,
            if_exists="replace",
            index=False
        )

        for table_name, lookup_data in lookup_tables.items():
            lookup_data.to_sql(
                table_name,
                connection,
                if_exists="replace",
                index=False
            )

        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        connection.executescript(schema_sql)

    print(f"Database created: {DATABASE_PATH}")
    print(f"Encounters loaded: {len(encounters):,}")

    for table_name, lookup_data in lookup_tables.items():
        print(f"{table_name}: {len(lookup_data):,} rows")


if __name__ == "__main__":
    build_database()