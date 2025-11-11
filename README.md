# Identifying medically eligible deceased organ donors in federated intensive care datasets

## CLIF Version
2.1.0

## Objective

This project identifies potential deceased organ donors from inpatient hospital deaths using two evidence-based criteria definitions: **CALC** (Cause, Age, Location-consistent) and **CLIF**. The code produces cohort selection diagrams, and baseline characteristics tables.

## Cohort Identification

### Cohort Definition: CALC (Cause, Age, Location-Consistent)

Patients meeting the CMS criteria for "deaths consistent with organ donation":

1. **All inpatient hospital deaths** - base cohort from hospitalization table with `discharge_category` = 'expired'
2. **Age ≤ 75 years** - calculated from `birth_date` and `discharge_dttm`
3. **Cause of death consistent with donation** (ICD-10-CM codes):
   - Ischemic heart disease (I20–I25)
   - Cerebrovascular disease (I60–I69)
   - External causes/trauma (V01–Y89): blunt/penetrating trauma, overdose, drowning, asphyxiation, suicide, homicide
4. **No contraindications** - absence of:
   - Sepsis (ICD-10-CM specified codes)
   - Active cancer (ICD-10-CM specified codes)

Reference: [CMS OPO Final Rule](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)

### Cohort Definition: CLIF (CLIF-Eligible Donors)

Medically eligible potential deceased abdominal organ donors:

1. **All inpatient hospital deaths** - death location must be ED, Ward, Stepdown, or ICU
2. **Age ≤ 75 years**
3. **On invasive mechanical ventilation (IMV)** - within 48 hours before death
4. **No contraindications**:
   - No positive blood cultures within 48 hours of death
   - No sepsis or active cancer diagnoses
5. **Pass organ quality assessment**:
   - **Kidney eligible**: Creatinine < 4 mg/dL AND not on CRRT
   - **Liver eligible**: All three labs recorded with values: Total bilirubin < 4, AST < 700, ALT < 700
   - **BMI eligible**: BMI ≤ 50 kg/m²
   - Overall: (Kidney OR Liver) AND BMI eligible

## Expected Results

The analysis produces the following outputs in `output/final/`:

### Visualizations
- **strobe_calc_definition.png** - STROBE flow diagram for CALC donor identification
- **strobe_clif_definition.png** - STROBE flow diagram for CLIF donor identification
- **cohort_funnels_side_by_side.png** - Side-by-side comparison funnels for both definitions
- **cohort_concentric_circles_side_by_side.png** - Visual comparison of nested circle diagrams

### Data Files
- **strobe_calc_definition.csv** - Stage-by-stage dropout counts for CALC definition
- **strobe_clif_definition.csv** - Stage-by-stage dropout counts for CLIF definition
- **table_one.csv** - Baseline characteristics across Overall, CALC, and CLIF cohorts
- **table_one.html** - Formatted Table 1 baseline characteristics report
- **final_cohort_df.parquet** - (INTERMEDIATE) Complete analytical dataset with all calculated flags


## Detailed Instructions for Running the Project

### 1. Configure Data Input

Update `config/config.json` with paths to your CLIF data files:

```json
{
  "site_name": "Your Site Name",
  "tables_path": "/path/to/clif/tables",
  "file_type": "csv",
  "project_root": "/path/to/project"
}
```

### 2. Set Up Project Environment

```bash
uv sync
```

### 3. Run the Analysis 

Open and execute `code/01_potential_donor_identifier.ipynb` in Jupyter. or run `uv run code/01_potential_donor_identifier.py`


### 4. Review Outputs

All results are saved to `output/final/` with accompanying console output showing:
- Patient counts at each filtering stage
- Exclusion reasons and counts
- Quality checks for data completeness

## Required CLIF Tables and Fields

The following tables are required with specified columns:

| Table Name              | Variables (with required categories/values)                   |
|-------------------------|--------------------------------------------------------------|
| **patient**             | `patient_id`, `death_dttm`, `birth_date`, `race_category`, `ethnicity_category`, `sex_category` |
| **hospitalization**     | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`, `discharge_category` (must include 'expired'), `admission_type_category` (emergency, planned, etc.) |
| **adt**                 | `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` (ed, ward, stepdown, icu), `location_name` |
| **vitals**              | `hospitalization_id`, `recorded_dttm`, `vital_category` (weight_kg, height_cm required), `vital_value` |
| **labs**                | `hospitalization_id`, `lab_collect_dttm`, `lab_category` (creatinine, bilirubin_total, ast, alt required), `lab_value_numeric` |
| **respiratory_support** | `hospitalization_id`, `recorded_dttm`, `device_category` (imv required) |
| **crrt_therapy**        | `hospitalization_id`, `recorded_dttm`                        |
| **hospital_diagnosis**  | `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format` (icd10, icd10cm) |
| **microbiology_culture**| `hospitalization_id`, `fluid_category` (blood_buffy), `method_category` (culture), `collect_dttm`, `organism_category` ('no_growth' or identified organism) |
| **patient_assessments** | `hospitalization_id`, `assessment_category` (gcs_total, rass required), `recorded_dttm`, `numerical_value` |



