-- Indexes for commonly queried fields

CREATE UNIQUE INDEX IF NOT EXISTS idx_encounter_id
ON encounters(encounter_id);

CREATE INDEX IF NOT EXISTS idx_patient_nbr
ON encounters(patient_nbr);

CREATE INDEX IF NOT EXISTS idx_readmitted_30d
ON encounters(readmitted_30d);


-- View combining hospital encounters with readable lookup descriptions

DROP VIEW IF EXISTS encounter_details;

CREATE VIEW encounter_details AS
SELECT
    e.*,
    admission_type.description AS admission_type,
    discharge_disposition.description AS discharge_disposition,
    admission_source.description AS admission_source
FROM encounters AS e
LEFT JOIN admission_types AS admission_type
    ON e.admission_type_id = admission_type.id
LEFT JOIN discharge_dispositions AS discharge_disposition
    ON e.discharge_disposition_id = discharge_disposition.id
LEFT JOIN admission_sources AS admission_source
    ON e.admission_source_id = admission_source.id;