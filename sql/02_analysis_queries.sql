-- 1. Overall 30-day readmission rate

SELECT
    COUNT(*) AS total_encounters,
    SUM(readmitted_30d) AS readmitted_within_30_days,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters;


-- 2. Readmission by age group

SELECT
    age,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY age
ORDER BY age;


-- 3. Readmission by race

SELECT
    COALESCE(race, 'Missing') AS race,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY COALESCE(race, 'Missing')
ORDER BY readmission_rate_pct DESC;


-- 4. Readmission by gender

SELECT
    gender,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY gender
ORDER BY readmission_rate_pct DESC;


-- 5. Readmission by admission type

SELECT
    COALESCE(admission_type, 'Missing') AS admission_type,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounter_details
GROUP BY COALESCE(admission_type, 'Missing')
ORDER BY readmission_rate_pct DESC;


-- 6. Readmission by time in hospital

SELECT
    time_in_hospital,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY time_in_hospital
ORDER BY time_in_hospital;


-- 7. Readmission by previous inpatient utilization

SELECT
    CASE
        WHEN number_inpatient = 0 THEN '0'
        WHEN number_inpatient = 1 THEN '1'
        WHEN number_inpatient = 2 THEN '2'
        ELSE '3 or more'
    END AS previous_inpatient_visits,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY
    CASE
        WHEN number_inpatient = 0 THEN '0'
        WHEN number_inpatient = 1 THEN '1'
        WHEN number_inpatient = 2 THEN '2'
        ELSE '3 or more'
    END
ORDER BY MIN(number_inpatient);


-- 8. Readmission by HbA1c result

SELECT
    A1Cresult,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY A1Cresult
ORDER BY readmission_rate_pct DESC;


-- 9. Readmission by medication change

SELECT
    "change" AS medication_change,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY "change"
ORDER BY readmission_rate_pct DESC;


-- 10. Readmission by diabetes-medication use

SELECT
    diabetesMed AS diabetes_medication,
    COUNT(*) AS encounters,
    SUM(readmitted_30d) AS readmissions,
    ROUND(100.0 * AVG(readmitted_30d), 2) AS readmission_rate_pct
FROM encounters
GROUP BY diabetesMed
ORDER BY readmission_rate_pct DESC;