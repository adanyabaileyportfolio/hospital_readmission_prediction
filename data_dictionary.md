# Data Dictionary

## Dataset

This project uses the UCI Diabetes 130-US Hospitals dataset, which contains 101,766 inpatient encounters involving patients diagnosed with diabetes at 130 U.S. hospitals and integrated delivery networks from 1999–2008.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008)

## Project outcome

| Variable | Type | Description |
|---|---|---|
| `readmitted_30d` | Binary integer | Project target created from `readmitted`. Equals 1 when the original value is `<30`; otherwise equals 0. |
| `readmitted` | Categorical | Original outcome: `<30`, `>30`, or `NO`. |

## Identifiers

| Variable | Type | Description | Modeling role |
|---|---|---|---|
| `encounter_id` | Identifier | Unique identifier for a hospital encounter. | Excluded from model |
| `patient_nbr` | Identifier | Unique identifier for a patient; patients may have multiple encounters. | Used for group splitting only |

## Patient demographics

| Variable | Type | Description |
|---|---|---|
| `race` | Categorical | Recorded race or ethnicity category. |
| `gender` | Categorical | Recorded patient gender. |
| `age` | Ordinal categorical | Patient age represented in 10-year intervals. |
| `weight` | Ordinal categorical | Patient weight represented in ranges; highly incomplete. |

## Hospital encounter

| Variable | Type | Description |
|---|---|---|
| `admission_type_id` | Coded categorical | Type of admission, such as emergency, urgent, or elective. |
| `discharge_disposition_id` | Coded categorical | Patient’s discharge destination or disposition. |
| `admission_source_id` | Coded categorical | Source of the hospital admission. |
| `time_in_hospital` | Integer | Number of days between admission and discharge. |
| `payer_code` | Categorical | Code identifying the expected payer. |
| `medical_specialty` | Categorical | Specialty of the admitting physician. |

## Clinical activity

| Variable | Type | Description |
|---|---|---|
| `num_lab_procedures` | Integer | Number of laboratory procedures performed during the encounter. |
| `num_procedures` | Integer | Number of non-laboratory procedures performed. |
| `num_medications` | Integer | Number of distinct generic medication names administered. |
| `number_diagnoses` | Integer | Number of diagnoses recorded for the encounter. |

## Previous healthcare utilization

These variables describe encounters during the year preceding the current hospital encounter.

| Variable | Type | Description |
|---|---|---|
| `number_outpatient` | Integer | Number of outpatient visits in the preceding year. |
| `number_emergency` | Integer | Number of emergency visits in the preceding year. |
| `number_inpatient` | Integer | Number of inpatient visits in the preceding year. |

## Diagnoses

| Variable | Type | Description |
|---|---|---|
| `diag_1` | Categorical | Primary diagnosis code. |
| `diag_2` | Categorical | Secondary diagnosis code. |
| `diag_3` | Categorical | Additional secondary diagnosis code. |

Diagnosis codes will later be grouped into broader clinical categories to reduce sparsity.

## Laboratory testing

| Variable | Type | Description |
|---|---|---|
| `max_glu_serum` | Categorical | Maximum serum glucose test result category or `None` if not measured. |
| `A1Cresult` | Categorical | HbA1c test result category or `None` if not measured. |

## Diabetes medications

The medication variables indicate whether a medication was prescribed and whether its dosage changed during the encounter. Common values include `No`, `Steady`, `Up`, and `Down`.

| Variable |
|---|
| `metformin` |
| `repaglinide` |
| `nateglinide` |
| `chlorpropamide` |
| `glimepiride` |
| `acetohexamide` |
| `glipizide` |
| `glyburide` |
| `tolbutamide` |
| `pioglitazone` |
| `rosiglitazone` |
| `acarbose` |
| `miglitol` |
| `troglitazone` |
| `tolazamide` |
| `examide` |
| `citoglipton` |
| `insulin` |
| `glyburide-metformin` |
| `glipizide-metformin` |
| `glimepiride-pioglitazone` |
| `metformin-rosiglitazone` |
| `metformin-pioglitazone` |

## Treatment indicators

| Variable | Type | Description |
|---|---|---|
| `change` | Binary categorical | Indicates whether the dosage or type of diabetes medication changed during the encounter. |
| `diabetesMed` | Binary categorical | Indicates whether a diabetes medication was prescribed. |

## Missing values

Several raw categorical variables represent missing information using `?`. During cleaning, these entries will be converted to standard missing values.

Variables with especially high missingness include:

- `weight`
- `medical_specialty`
- `payer_code`

Feature inclusion decisions will consider missingness, clinical relevance, potential data leakage, and whether the information would reasonably be available at discharge.

## Planned exclusions and special handling

- `encounter_id` will be excluded because it is an arbitrary identifier.
- `patient_nbr` will be excluded as a predictor but retained for patient-level splitting.
- `readmitted` will be replaced by the binary `readmitted_30d` target during modeling.
- Encounters ending in death, hospice discharge, or transfer to another inpatient facility will be reviewed because ordinary 30-day readmission may not be an appropriate outcome for those records.
- Diagnosis codes will be grouped into broader categories.
- Rare medication columns with little or no variation may be excluded.
- Final model-feature decisions will be documented after data cleaning.

## Applied cleaning decisions

The cleaning pipeline made the following changes:

- Removed 3 encounters with `Unknown/Invalid` gender.
- Removed 2,423 encounters ending in death or hospice care.
- Found no exact duplicate rows or duplicate encounter IDs.
- Converted `?` entries into standard missing values.
- Removed `weight`, `payer_code`, and `medical_specialty` from the cleaned modeling file because of substantial missingness and limited baseline utility.
- Added readable admission-type, discharge-disposition, and admission-source descriptions.
- Added grouped versions of the three diagnosis variables.
- Created `readmitted_30d` as the binary project outcome.
- Retained multiple encounters belonging to the same patient so they can be handled through patient-level group splitting.
- Produced a cleaned dataset containing 99,340 encounters.