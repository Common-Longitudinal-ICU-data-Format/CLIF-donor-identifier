# CLIF Epidemiology of Potential and Actual Deceased Organ Donors

## Project Version
2.1.0

## Objective

This project identifies potential deceased organ donors from inpatient hospital deaths using two evidence-based criteria definitions: **CALC** (Cause, Age, Location-consistent) and **CLIF** (Critical Illness Outcomes Research Network). The code produces cohort selection diagrams, and baseline characteristics tables.

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

Open and execute `code/01_potential_donor_identifier.ipynb` in Jupyter.


### 4. Review Outputs

All results are saved to `output/final/` with accompanying console output showing:
- Patient counts at each filtering stage
- Exclusion reasons and counts
- Quality checks for data completeness

## Required CLIF Tables and Fields

The following tables are required with specified columns:

### 1. **patient**
- `patient_id` - unique patient identifier
- `death_dttm` - date/time of patient death
- `birth_date` - patient date of birth (for age calculation)
- `race_category` - patient race category
- `ethnicity_category` - patient ethnicity category
- `sex_category` - patient biological sex

### 2. **hospitalization**
- `patient_id` - links to patient table
- `hospitalization_id` - unique hospitalization identifier
- `admission_dttm` - hospital admission date/time
- `discharge_dttm` - hospital discharge date/time
- `age_at_admission` - age of patient at admission
- `discharge_category` - discharge disposition (must include 'expired' for decedents)
- `admission_type_category` - type of admission (emergency, planned, etc.)

### 3. **adt** (Admission-Discharge-Transfer)
- `hospitalization_id` - links to hospitalization table
- `in_dttm` - date/time of location entry
- `out_dttm` - date/time of location exit
- `location_category` - clinical location (ed, ward, stepdown, icu)
- `location_name` - descriptive location name

### 4. **vitals**
- `hospitalization_id` - links to hospitalization table
- `recorded_dttm` - date/time of vital sign recording
- `vital_category` - type of vital sign
  - Required: `weight_kg`, `height_cm` (for BMI calculation)
- `vital_value` - numeric vital sign value

### 5. **labs**
- `hospitalization_id` - links to hospitalization table
- `lab_collect_dttm` - date/time of lab specimen collection
- `lab_category` - type of laboratory test
  - Required for organ assessment: `creatinine`, `bilirubin_total`, `ast`, `alt`
- `lab_value_numeric` - numeric lab result value

### 6. **respiratory_support**
- `hospitalization_id` - links to hospitalization table
- `recorded_dttm` - date/time of respiratory support record
- `device_category` - type of respiratory device
  - Required: `imv` (invasive mechanical ventilation)

### 7. **crrt_therapy** (Continuous Renal Replacement Therapy)
- `hospitalization_id` - links to hospitalization table
- `recorded_dttm` - date/time of CRRT record

### 8. **hospital_diagnosis**
- `hospitalization_id` - links to hospitalization table
- `diagnosis_code` - ICD-10-CM diagnosis code
- `diagnosis_code_format` - format indicator (icd10, icd10cm)

### 9. **microbiology_culture**
- `hospitalization_id` - links to hospitalization table
- `fluid_category` - type of specimen (blood_buffy)
- `method_category` - test method (culture)
- `collect_dttm` - date/time of specimen collection
- `organism_category` - identified organism or 'no_growth'

### 10. **patient_assessments**
- `hospitalization_id` - links to hospitalization table
- `assessment_category` - type of assessment
  - Required: `gcs_total`, `rass`
- `recorded_dttm` - date/time of assessment
- `numerical_value` - numeric assessment score



