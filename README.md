# CLIF Donor Identifier

Identifies medically eligible deceased organ donors from CLIF inpatient deaths using two parallel definitions (**CALC** and **CLIF**). Produces STROBE diagrams, baseline characteristics tables, and donor-eligibility funnels.

**CLIF Version:** 2.1.0
**Cohort window:** 2020-01-01 to 2025-12-31 (both admission and discharge must fall within)

## Cohort definitions

| Criterion | CALC | CLIF |
|---|---|---|
| In-hospital death (`discharge_category = expired`) | ✓ | ✓ |
| Age ≤75 at death | ✓ | ✓ |
| Cause of death (ICD-10: I20–I25, I60–I69, V01–Y89) | ✓ | — |
| IMV ≤48 h before death | — | ✓ |
| No sepsis or active cancer (ICD-10) | ✓ | ✓ |
| No positive blood culture ≤48 h before death | — | ✓ |
| BMI ≤50 | — | ✓ |
| Kidney (terminal Cr <4 AND no CRRT ≤48 h) **OR** Liver (terminal Bili <4 AND AST <700 AND ALT <700) | — | ✓ |

A third stratum, **Died_While_IMV**, is also reported: age ≤75 inpatient decedents on IMV ≤48 h before death (= CLIF Stage 3, before contraindication / organ-quality filters).

References: [CMS OPO Final Rule](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf) (CALC).

## How to run

```bash
# 1. Configure
cp config/config_template.json config/config.json
# Edit config.json with your site_name, tables_path, file_type, timezone

# 2. Install dependencies
uv sync

# 3. Run
uv run python code/01_potential_donor_identifier.py
```

All outputs land in `output/final/`. The full stdout is mirrored to `output/final/run_log.txt`.

## Outputs (in `output/final/`)

| File | Contents |
|---|---|
| `table_one.csv` / `.html` | Baseline characteristics, 4 cohort columns: Overall / Died_While_IMV / CALC_Donors / CLIF_Donors. Demographics, ICU LOS, organ-eligibility flags, terminal labs, and comorbidities (HCV / Hypertension / Diabetes / Hx CVA). |
| `aim1_table_two_by_terminal_cr.csv` | Patient characteristics in the Died_While_IMV cohort stratified by terminal Cr `<2 mg/dL` vs `≥2 mg/dL` vs `missing`. |
| `cohort_numbers.csv` | Stage-by-stage CALC and CLIF cohort counts. |
| `strobe_counts.csv` | All filter-stage counts (single row), including the new `1c_died_post_discharge_24h` diagnostic for sites with external death-registry data. |
| `strobe_calc_definition.{csv,png}` | CALC STROBE diagram + stage table. |
| `strobe_clif_definition.{csv,png}` | CLIF STROBE diagram + stage table. |
| `funnel_calc.png`, `funnel_clif.png`, `funnels_side_by_side.png` | Drop-out funnel visualizations. |
| `circles_side_by_side.png` | Concentric-circle visualization of nested cohort sizes. |
| `run_log.txt` | Full script stdout (overwritten on each run). |

Sites collaborating on the multi-site NIDDK Aim 1 effort should ship the entire `output/final/` folder via Box.

## Comorbidity extraction

HCV, Hypertension, Diabetes, and Hx CVA are flagged via ICD-10 prefix matching on `hospital_diagnosis`. Codes live in `utils/icd10_comorbidities.csv`; diagnosis codes are lowercased and stripped of periods/whitespace before matching. Sepsis and active-cancer contraindications use the existing `utils/icd10_contraindications.csv` (CCS-based codes).

## Required CLIF tables and columns

| Table | Required columns / categories |
|---|---|
| **patient** | `patient_id`, `death_dttm`, `birth_date`, `race_category`, `ethnicity_category`, `sex_category` |
| **hospitalization** | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`, `discharge_category` (must include `expired`), `admission_type_category` |
| **adt** | `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` (ed, ward, stepdown, icu), `location_name` |
| **vitals** | `hospitalization_id`, `recorded_dttm`, `vital_category` (`weight_kg`, `height_cm` required), `vital_value` |
| **labs** | `hospitalization_id`, `lab_collect_dttm`, `lab_category` (`creatinine`, `bilirubin_total`, `ast`, `alt` required), `lab_value_numeric` |
| **respiratory_support** | `hospitalization_id`, `recorded_dttm`, `device_category` (`imv` required) |
| **crrt_therapy** | `hospitalization_id`, `recorded_dttm` |
| **hospital_diagnosis** | `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format` (`icd10`, `icd10cm`) |
| **microbiology_culture** | `hospitalization_id`, `fluid_category` (`blood_buffy`), `method_category` (`culture`), `collect_dttm`, `organism_category` (`no_growth` or identified organism) |
| **patient_assessments** | `hospitalization_id`, `assessment_category` (`gcs_total`, `rass`), `recorded_dttm`, `numerical_value` |

## Engine note

Large CLIF tables (`hospital_diagnosis`, `respiratory_support`, `crrt_therapy`, `labs`, `microbiology_culture`, `patient_assessments`) are queried via **DuckDB** streaming SQL on the parquet files rather than fully loaded into memory — works at any site scale. Polars is used for the cohort logic.
