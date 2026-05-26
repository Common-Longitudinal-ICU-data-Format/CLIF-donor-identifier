#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script identifies potential organ donors based on the 
CALC criteria and the CLIF eligible criteria using the CLIF dataset. 
It performs the following steps:
1. Load the required datasets
2. Identify decedents
3. Stitch encounters
4. Apply outlier handling
5. Create consort diagram

Authors: Kaveri Chhikara + Claude Opus 4.7
"""

################################################################################
# Potential Organ Donor Identifier
################################################################################

################################################################################
# Setup
################################################################################

import sys
import os
import duckdb
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
from utils.config import config
from utils.io import read_data
from utils.strobe_diagram import create_consort_diagram
from clifpy.utils.stitching_encounters import stitch_encounters
from utils.outlier_handler import apply_outlier_handling
import gc

# Fix Windows encoding issue for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')
site_name = config['site_name']
tables_path = config['tables_path']
file_type = config['file_type']
project_root = config['project_root']
sys.path.insert(0, project_root)
print(f"Site Name: {site_name}")
print(f"Tables Path: {tables_path}")
print(f"File Type: {file_type}")
from pathlib import Path
PROJECT_ROOT = Path(config['project_root'])
UTILS_DIR = PROJECT_ROOT / "utils"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FINAL_DIR = OUTPUT_DIR / "final"
OUTPUT_INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Run log -----------------------------------------------------------
# Tee stdout so every print() in this run also lands in output/final/run_log.txt.
# Overwrites the previous run's log; sites only need the most recent.
import atexit, datetime as _dt

class _Tee:
    """Write to multiple streams (e.g., terminal + log file) simultaneously."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, msg):
        for s in self._streams:
            try:
                s.write(msg); s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass

_RUN_LOG_PATH = OUTPUT_FINAL_DIR / "run_log.txt"
_run_log_handle = open(_RUN_LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _run_log_handle)
atexit.register(lambda: (_run_log_handle.flush(), _run_log_handle.close()))
print(f"=== Run started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===")
print(f"Site: {site_name} | Tables: {tables_path} | File type: {file_type}")
print(f"Logging stdout to: {_RUN_LOG_PATH}")
print("-" * 80)

strobe_counts = {}

################################################################################
# Load data
################################################################################

# read required tables
adt_filepath = f"{tables_path}/clif_adt.{file_type}"
hospitalization_filepath = f"{tables_path}/clif_hospitalization.{file_type}"
patient_filepath = f"{tables_path}/clif_patient.{file_type}"
adt_df = read_data(adt_filepath, file_type)
hospitalization_df = read_data(hospitalization_filepath, file_type)
patient_df = read_data(patient_filepath, file_type)

# Informational baseline (entire CLIF dataset at this site, all years)
total_patients = patient_df["patient_id"].n_unique()
strobe_counts["00_all_patients_in_clif"] = total_patients

################################################################################
# Apply cohort time-window filter (PDF 1 spec: 2020-01-01 to 2025-12-31)
# STRICT: a hospitalization is kept only if BOTH admission and discharge fall
# inside the window. This matches the SRTR donor cohort's strict calendar
# boundaries so the two are directly comparable.
################################################################################

WINDOW_START_YEAR = 2020
WINDOW_END_YEAR = 2025

# Compare on year() to avoid timezone/precision mismatches between the
# stored datetime[ns, UTC] columns and naive datetime literals.
_admitted_in_window = (
    (pl.col('admission_dttm').dt.year() >= WINDOW_START_YEAR)
    & (pl.col('admission_dttm').dt.year() <= WINDOW_END_YEAR)
)
_discharged_in_window = (
    (pl.col('discharge_dttm').dt.year() >= WINDOW_START_YEAR)
    & (pl.col('discharge_dttm').dt.year() <= WINDOW_END_YEAR)
)
hospitalization_df = hospitalization_df.filter(_admitted_in_window & _discharged_in_window)

# Restrict ADT to hospitalizations that survived the window filter
_hosp_ids_in_window = hospitalization_df['hospitalization_id'].unique().to_list()
adt_df = adt_df.filter(pl.col('hospitalization_id').is_in(_hosp_ids_in_window))

# STROBE step 1: Full population — patients with any hospitalization overlapping 2020-2025
full_population_n = hospitalization_df['patient_id'].n_unique()
strobe_counts["0_full_population_2020_2025"] = full_population_n
print(f"Full population (any hosp 2020-2025): {full_population_n:,}")

# STROBE step 2: ICU population — subset with ≥1 ICU ADT stay
icu_population_n = (
    adt_df
    .filter(pl.col('location_category').str.to_lowercase() == 'icu')
    .join(
        hospitalization_df.select(['hospitalization_id', 'patient_id']),
        on='hospitalization_id',
        how='left',
    )
    .select('patient_id')
    .drop_nulls()
    .unique()
    .height
)
strobe_counts["0b_icu_population_2020_2025"] = icu_population_n
print(f"ICU population (any ICU stay): {icu_population_n:,}")

################################################################################
# Identify decedents
################################################################################

all_decedents_df = hospitalization_df.filter(
    pl.col('discharge_category').str.to_lowercase() == 'expired'
)

all_decedent_patient_ids = all_decedents_df.select('patient_id').to_series().to_list()
all_decedent_hosp_ids = all_decedents_df.select('hospitalization_id').to_series().to_list()

################################################################################
# Stitch Encounters
################################################################################

# Check if hospitalization_df has duplicate patient_id, hospitalization_id pairs
if hospitalization_df.shape[0] != hospitalization_df.unique(subset=["patient_id", "hospitalization_id"]).shape[0]:
    print("Warning: hospitalization_df contains duplicate (patient_id, hospitalization_id) rows.")

# Check if adt_df has duplicate patient_id, hospitalization_id, adt_event_id (or similar) triplets
# If adt_df has an event or unique identifier column, replace 'adt_event_id' with correct column
adt_unique_cols = [col for col in [ "hospitalization_id", "in_dttm"] if col in adt_df.columns]
if len(adt_unique_cols) >= 2 and adt_df.shape[0] != adt_df.unique(subset=adt_unique_cols).shape[0]:
    print("Warning: adt_df contains duplicate rows for identifier columns:", adt_unique_cols)

# Filter hospitalization_df and adt_df down to all_decedent_hosp_ids
hospitalization_df_subset = hospitalization_df.filter(
    pl.col("patient_id").is_in(all_decedent_patient_ids)
)
all_decedent_hosp_ids = hospitalization_df_subset.select('hospitalization_id').to_series().to_list()
adt_df_subset = adt_df.filter(
    pl.col("hospitalization_id").is_in(all_decedent_hosp_ids)
)

hosp_stitched, adt_stitched, encounter_mapping = stitch_encounters(
      hospitalization=hospitalization_df_subset.to_pandas(),
      adt=adt_df_subset.to_pandas(),
      time_interval=12
  )

hosp_stitched = pl.from_pandas(hosp_stitched)
adt_stitched = pl.from_pandas(adt_stitched)

# Ensure encounter_block is int32 (if present)
hosp_stitched = hosp_stitched.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)
adt_stitched = adt_stitched.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)

# Filter hosp_stitched to only hospitalizations that are present in the adt_stitched table
if "hospitalization_id" in hosp_stitched.columns and "hospitalization_id" in adt_stitched.columns:
    hosp_stitched = (
        hosp_stitched.filter(
            pl.col("hospitalization_id").is_in(adt_stitched["hospitalization_id"].unique())
        )
    )
encounter_mapping = pl.from_pandas(encounter_mapping)
encounter_mapping = encounter_mapping.with_columns(
    pl.col("encounter_block").cast(pl.Int32)
)

gc.collect()

# Identify expired encounters
decedents_df = hosp_stitched.filter(
    pl.col('discharge_category').str.to_lowercase() == 'expired'
)

# Make hospitalization subset for expired
# Join patient_id and death_dttm from patient_df to final_df

final_df = (
    decedents_df
    .select([
        'patient_id',
        'hospitalization_id',
        'encounter_block',
        'admission_dttm',
        'discharge_dttm', # discharge datetime for the death hospitalization
        "age_at_admission", 
        "discharge_category",
        "admission_type_category"
    ])
    .with_columns([
        pl.col("discharge_category").str.to_lowercase(),
        pl.col("admission_type_category").str.to_lowercase()
    ])
    .unique()
)

# Now join patient_id and death_dttm from patient_df to final_df
demog_cols = ['patient_id', 'death_dttm', 'race_category', 'sex_category','ethnicity_category' ]
final_df = final_df.join(
    patient_df.select(demog_cols), on='patient_id', how='left'
)

# ----------------------------------------------------------------------
# Death hospitalization analysis + early per-patient dedup
# ----------------------------------------------------------------------
# Diagnostic FIRST — show raw counts of multi-death patients (rare data
# quirks worth surfacing), THEN unconditionally collapse to one row per
# patient (the latest death encounter). Doing the dedup early means every
# downstream operation operates on one-row-per-patient and the strobe
# counts therefore agree with Table 1 by construction.
print("\n" + "="*80)
print("DEATH HOSPITALIZATION ANALYSIS")
print("="*80)

death_encounters_per_patient = (
    final_df
    .group_by('patient_id')
    .agg([
        pl.count().alias('death_encounter_count'),
        pl.col('encounter_block').alias('encounter_blocks'),
        pl.col('discharge_dttm').alias('discharge_times'),
    ])
    .sort('death_encounter_count', descending=True)
)

total_death_encounters = len(final_df)
unique_patients = final_df['patient_id'].n_unique()
multi_death_patients = death_encounters_per_patient.filter(pl.col('death_encounter_count') > 1)
num_multi_death = len(multi_death_patients)

print(f"Total death hospitalizations found: {total_death_encounters:,}")
print(f"Unique patients who died:           {unique_patients:,}")
print(f"Patients with multiple deaths:      {num_multi_death:,}")

if num_multi_death > 0:
    print("\nDEBUG: Patients with multiple death hospitalizations (showing first 5):")
    for row in multi_death_patients.head(5).iter_rows(named=True):
        print(f"  Patient {row['patient_id']}: {row['death_encounter_count']} death hospitalizations")
        print(f"    Encounter blocks: {row['encounter_blocks']}")
        print(f"    Discharge times:  {row['discharge_times']}")
    print(f"\nNOTE: {num_multi_death} patients have multiple death rows — likely a data quality issue.")
    print("Will collapse each patient to their LATEST death encounter below.")
else:
    print("✓ No patients with multiple death hospitalizations — data is clean.")

# Always collapse to one row per patient = the latest death encounter.
# (For decedents the latest encounter_block IS the death encounter; before
# this point a patient could technically have multiple expired rows due to
# data quirks. From here on, final_df is strict patient-level.)
final_df = (
    final_df
    .sort('discharge_dttm', descending=True)
    .unique(subset='patient_id', keep='first')   # latest after desc sort
)
assert final_df['patient_id'].n_unique() == len(final_df), "Early dedup failed"
print(f"\n✓ After early dedup: {len(final_df):,} unique patients (one row each).")
print("="*80 + "\n")

decedents_df_n = final_df["patient_id"].n_unique()
strobe_counts["1_decedents_df_n"] = decedents_df_n
strobe_counts

################################################################################
# Final outcome dttm
################################################################################

vitals_filepath = f"{tables_path}/clif_vitals.{file_type}"

# Stream clif_vitals via DuckDB instead of loading into polars. Sites with
# very large vitals tables (e.g., JHU at >2 GB) were getting OOM-killed
# inside the polars `with_columns` outlier-handler chain. We only consume
# four fields downstream (first/last vital timestamps + last weight_kg +
# last height_cm), so a streamed SQL query is both faster and bounded in RAM.
# Outlier ranges (weight 30-1100 kg, height 76-255 cm) are applied inline
# via CASE WHEN, matching utils/outlier_handler.py + config/outlier_config.yaml.
all_decedent_hosp_ids_df = pd.DataFrame(
    {"hospitalization_id": list(all_decedent_hosp_ids)}
)

vitals_query = f"""
WITH vitals_cohort AS (
    SELECT hospitalization_id, recorded_dttm, vital_category, vital_value
    FROM read_parquet('{vitals_filepath}')
    WHERE hospitalization_id IN (
        SELECT hospitalization_id FROM all_decedent_hosp_ids_df
    )
),
time_bounds AS (
    SELECT
        hospitalization_id,
        MIN(recorded_dttm) AS first_recorded_vital_dttm,
        MAX(recorded_dttm) AS last_recorded_vital_dttm
    FROM vitals_cohort
    GROUP BY hospitalization_id
),
weight_ranked AS (
    SELECT
        hospitalization_id,
        CASE WHEN vital_value BETWEEN 30 AND 1100 THEN vital_value END AS vital_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id ORDER BY recorded_dttm DESC
        ) AS rn
    FROM vitals_cohort
    WHERE vital_category = 'weight_kg'
),
height_ranked AS (
    SELECT
        hospitalization_id,
        CASE WHEN vital_value BETWEEN 76 AND 255 THEN vital_value END AS vital_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id ORDER BY recorded_dttm DESC
        ) AS rn
    FROM vitals_cohort
    WHERE vital_category = 'height_cm'
)
SELECT
    t.hospitalization_id,
    t.first_recorded_vital_dttm,
    t.last_recorded_vital_dttm,
    w.vital_value AS last_weight_kg,
    h.vital_value AS last_height_cm
FROM time_bounds t
LEFT JOIN weight_ranked w
    ON t.hospitalization_id = w.hospitalization_id AND w.rn = 1
LEFT JOIN height_ranked h
    ON t.hospitalization_id = h.hospitalization_id AND h.rn = 1
"""

print("Processing vitals data with DuckDB...")
vitals_first_last = pl.from_pandas(duckdb.sql(vitals_query).df())
# DuckDB returns UTC; match the canonical IANA TZ that clifpy uses on the
# rest of final_df (config['timezone'] may be a legacy alias like "US/Central"
# that polars treats as a different dtype than "America/Chicago", so detect
# the actual TZ from an existing column).
_dt_dtype = next(
    (dt for col, dt in final_df.schema.items()
     if isinstance(dt, pl.Datetime) and dt.time_zone is not None),
    None,
)
if _dt_dtype is not None:
    _site_tz = _dt_dtype.time_zone
    vitals_first_last = vitals_first_last.with_columns([
        pl.col('first_recorded_vital_dttm').dt.convert_time_zone(_site_tz),
        pl.col('last_recorded_vital_dttm').dt.convert_time_zone(_site_tz),
    ])
print(f"✓ Processed vitals for {len(vitals_first_last)} hospitalizations")

# Calculate BMI
vitals_first_last = vitals_first_last.with_columns(
    (pl.col('last_weight_kg') / ((pl.col('last_height_cm') / 100) ** 2)).alias('bmi')
)

# Join with final_df
final_df = final_df.join(vitals_first_last, on='hospitalization_id', how='left')

# Define final_death_dttm using the actual death timestamp where present.
# We deliberately do NOT cap death_dttm at discharge_dttm — sites that pull
# death data from external registries (state vital records, SSA Death Master,
# etc.) can legitimately have death_dttm > discharge_dttm for patients who
# were discharged alive and died later. Those patients should NOT pass the
# downstream 48-h before-death donor filters, and trusting death_dttm
# directly makes that natural (their in-hospital labs/vitals will fall
# outside the 48-h window relative to their later death).
# Fall back to last_recorded_vital_dttm only when death_dttm is null.
final_df = final_df.with_columns(
    pl.when(pl.col("death_dttm").is_not_null())
      .then(pl.col("death_dttm"))
      .otherwise(pl.col("last_recorded_vital_dttm"))
      .alias("final_death_dttm")
)

# Diagnostic: how many decedents have death_dttm > discharge_dttm + 24h?
# This indicates sites with external death-registry data linked into CLIF;
# such patients were discharged alive and died later — they auto-fail the
# CLIF donor criteria but remain in the cohort for transparency.
_delayed_death = final_df.filter(
    pl.col("death_dttm").is_not_null()
    & pl.col("discharge_dttm").is_not_null()
    & ((pl.col("death_dttm") - pl.col("discharge_dttm")).dt.total_hours() > 24)
)["patient_id"].n_unique()
strobe_counts["1c_died_post_discharge_24h"] = _delayed_death
print(f"Decedents with death_dttm > discharge_dttm + 24h (likely external registry): {_delayed_death:,}")

################################################################################
# Inpatient decedents
# Identify inpatient encounters - location must be ed, ward, stepdown, icu at last_recorded_vital_dttm
################################################################################

eligible_locations = ['ed', 'ward', 'stepdown', 'icu']

# Check that all decedents are present in ADT table
decedent_hosp_in_adt = set(adt_df.select('hospitalization_id').to_series().to_list())
missing_in_adt = set(all_decedent_hosp_ids) - decedent_hosp_in_adt

if missing_in_adt:
    print(f"Warning: {len(missing_in_adt)} hospitalization(s) missing in ADT table")
    print(f"Missing hospitalization_ids: {missing_in_adt}")
else:
    print(f"✓ All {len(all_decedent_hosp_ids)} decedent hospitalizations present in ADT table")

last_location_per_hosp = (
      adt_df
      .filter(pl.col('hospitalization_id').is_in(all_decedent_hosp_ids))
      .sort('out_dttm', descending=True)
      .group_by('hospitalization_id')
      .agg([
          pl.col('location_category').first().alias('last_location_category'),
          pl.col('location_name').first().alias('last_location_name'),
          pl.col('out_dttm').first().alias('last_location_out_dttm'),
          (pl.col('location_category').str.to_lowercase() == 'icu').any().alias('ever_icu'),
          (pl.col('location_category').str.to_lowercase() == 'ward').any().alias('ever_ward'),
          (pl.col('location_category').str.to_lowercase() == 'ed').any().alias('ever_ed'),
          (pl.col('location_category').str.to_lowercase() == 'stepdown').any().alias('ever_stepdown'),
          pl.col('location_category').unique().sort().alias('all_locations')
      ])
  )

final_df = final_df.join(
    last_location_per_hosp,
    on='hospitalization_id',
    how='left'
)

# Identify the number of hospitalizations where ever_icu, ever_ward, ever_ed, ever_stepdown are all False or null
n_all_locs_false = final_df.filter(
    ~(pl.col('ever_icu').fill_null(False) | 
      pl.col('ever_ward').fill_null(False) | 
      pl.col('ever_ed').fill_null(False) | 
      pl.col('ever_stepdown').fill_null(False))
).height
print(f"Number of hospitalizations where all four location flags are False/null: {n_all_locs_false}")

# Create final_cohort_df dropping those hospitalizations
final_cohort_df = final_df.filter(
    (pl.col('ever_icu').fill_null(False) | 
     pl.col('ever_ward').fill_null(False) | 
     pl.col('ever_ed').fill_null(False) | 
     pl.col('ever_stepdown').fill_null(False))
)

all_decedent_inpatient_patient_ids = final_cohort_df.select('patient_id').to_series().to_list()
all_decedent_inpatient_hosp_ids = final_cohort_df.select('hospitalization_id').to_series().to_list()
strobe_counts["2_inpatient_decedents"] = len(all_decedent_inpatient_patient_ids)
strobe_counts

adt_stitched.columns

################################################################################
# ADT
################################################################################

# Calculate hospital and ICU length of stay using approach similar to the provided reference (adapted for Polars)

# Filter adt_df to only the relevant hospitalizations
adt_in_cohort = adt_stitched.filter(pl.col("hospitalization_id").is_in(all_decedent_inpatient_hosp_ids))

# Lowercase location_category (just the column, not the whole DataFrame)
adt_in_cohort = adt_in_cohort.with_columns(
    pl.col("location_category").str.to_lowercase().alias("location_category")
)

# Hospital admission summary per encounter_block: first in and last out, first admission location
hosp_admission_summary = (
    adt_in_cohort
    .group_by("encounter_block")
    .agg([
        pl.col("in_dttm").min().alias("min_in_dttm"),
        pl.col("out_dttm").max().alias("max_out_dttm"),
        pl.col("location_category").first().alias("first_admission_location")
    ])
    .with_columns([
        ((pl.col("max_out_dttm") - pl.col("min_in_dttm")).dt.total_days()).alias("hospital_length_of_stay_days")
    ])
)

# Join first_admission_location and hospital_length_of_stay_days to final_cohort_df on encounter_block
final_cohort_df = final_cohort_df.join(
    hosp_admission_summary.select([
        "encounter_block", 
        "first_admission_location", 
        "hospital_length_of_stay_days"
    ]),
    on="encounter_block",
    how="left"
)

# Restrict to ICU stays only
icu_df = adt_in_cohort.filter(pl.col("location_category") == "icu")

# Find first ICU admission per encounter_block
first_icu_in = (
    icu_df
    .group_by("encounter_block")
    .agg(pl.col("in_dttm").min().alias("first_icu_in_dttm"))
)

# Join back to get corresponding out_dttm for the first ICU in_dttm.
# Some sites (e.g. NU) log >1 ADT-ICU row at the same first in_dttm with
# different out_dttm (overlapping unit transfers recorded simultaneously).
# Collapse to one row per encounter_block, keeping the latest out_dttm so
# first_icu_los_days reflects the longest stay starting at that moment.
icu_summary = (
    first_icu_in.join(
        icu_df.select(["encounter_block", "in_dttm", "out_dttm"]),
        left_on=["encounter_block", "first_icu_in_dttm"],
        right_on=["encounter_block", "in_dttm"],
        how="left"
    )
    .group_by("encounter_block")
    .agg([
        pl.col("first_icu_in_dttm").first(),
        pl.col("out_dttm").max().alias("first_icu_out_dttm"),
    ])
    .with_columns(
        ((pl.col("first_icu_out_dttm") - pl.col("first_icu_in_dttm")).dt.total_seconds() / (3600*24))
        .alias("first_icu_los_days")
    )
    .select([
        "encounter_block", "first_icu_in_dttm", "first_icu_out_dttm", "first_icu_los_days"
    ])
)

final_cohort_df = final_cohort_df.join(
    icu_summary.select([
        "encounter_block", 
        "first_icu_los_days"
    ]),
    on="encounter_block",
    how="left"
)

# Now, hosp_admission_summary contains hospital LOS and first_admission_location, and icu_summary contains first ICU LOS

################################################################################
# Age
################################################################################

# Age < 75
final_cohort_df = final_cohort_df.join(
    patient_df.select(['patient_id', 'birth_date']),
    on='patient_id',
    how='left'
)
# Only cast birth_date to datetime if not already a datetime type
# if final_cohort_df.schema["birth_date"] != pl.Datetime:
#     final_cohort_df = final_cohort_df.with_columns(
#         pl.col('birth_date').str.to_datetime().alias('birth_date')
#     )

# Calculate age at death as (discharge_dttm - birth_date) in years (using .dt.total_days()/365.25)
final_cohort_df = final_cohort_df.with_columns(
    (
        (pl.col('final_death_dttm') - pl.col('birth_date')).dt.total_days() / 365.25
    ).alias('age_at_death')
)

# Create age_75_less flag per patient_id ( age_at_death <= 75)
age_flag_df = (
    final_cohort_df
    .group_by('patient_id')
    .agg([
        (
            (pl.col('age_at_death') <= 75).any()
        ).alias('age_75_less')
    ])
)

# Join age_75_less flag onto final_df; fill nulls with False
final_cohort_df = (
    final_cohort_df
    .join(age_flag_df, on='patient_id', how='left')
    .with_columns(
        pl.col('age_75_less').fill_null(False)
    )
)

# Filter age < 75 using the flag, not the missing column
age_relevant_cohort = final_cohort_df.filter(
    pl.col('age_75_less') == True
)
age_relevant_cohort_n = age_relevant_cohort["patient_id"].n_unique()
strobe_counts["3_age_relevant_cohort_n"] = age_relevant_cohort_n
strobe_counts

################################################################################
# ICD Codes
# The CALC criteria includes the following as cause:
# - I20–I25: ischemic heart disease
# - I60–I69: cerebrovascular disease
# - V01–Y89: external causes (e.g., blunt trauma, gunshot wounds, overdose, suicide, drowning, asphyxiation)
# [Reference](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)
# We also flag contraindications of sepsis and cancer using ICD10 codes. We use the ICD codes for these specified in utils/icd10_contraindications.csv
################################################################################

hospial_dx_filepath = f"{tables_path}/clif_hospital_diagnosis.{file_type}"

# Diagnostic counts via DuckDB (streamed; previously a polars read+join that
# segfaulted on Windows at sites with large hospital_diagnosis tables).
all_ids_df = pd.DataFrame({"hospitalization_id": list(all_decedent_inpatient_hosp_ids)})
age_relevant_ids_df = age_relevant_cohort.select("patient_id").unique().to_pandas()

n_present = duckdb.sql(f"""
    SELECT COUNT(DISTINCT hospitalization_id)
    FROM read_parquet('{hospial_dx_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM all_ids_df)
""").fetchone()[0]
print(f"Hospitalization IDs present in hospital_dx: {n_present} out of "
      f"{len(all_decedent_inpatient_hosp_ids)}")
strobe_counts["5_present_inpatient_hospitalization_ids_in_hospital_dx"] = n_present

n_age_relevant = duckdb.sql(f"""
    SELECT COUNT(DISTINCT hosp.patient_id)
    FROM read_parquet('{hospial_dx_filepath}') hd
    JOIN hospitalization_df hosp ON hd.hospitalization_id = hosp.hospitalization_id
    WHERE hd.hospitalization_id IN (SELECT hospitalization_id FROM all_ids_df)
      AND hosp.patient_id     IN (SELECT patient_id      FROM age_relevant_ids_df)
""").fetchone()[0]
strobe_counts["5_age_relevant_in_hospital_dx"] = n_age_relevant

# ---- 0) Load contraindications list from CSV ----
contraindications_df = pl.read_csv(str(UTILS_DIR / "icd10_contraindications.csv"))
contraindication_codes = (
    contraindications_df
    .with_columns([
        pl.col("ICD-10-CM")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .str.replace_all(r"[.\s]", "")
            .alias("code_norm")
    ])
    .select("code_norm")
    .to_series()
    .to_list()
)

print(f"Loaded {len(contraindication_codes)} contraindication ICD-10 codes")

# ---- 0b) Load comorbidity prefixes (HCV, HTN, DM, Hx CVA) from CSV ----
# These are PREFIX matches (3-4 char ICD blocks) — e.g. 'i10' matches any
# code starting with i10 (i10, i109, i1010, etc.). Lowercase + no periods.
comorbidities_df = pl.read_csv(str(UTILS_DIR / "icd10_comorbidities.csv"))
comorbidity_prefixes: dict[str, list[str]] = {}
for row in comorbidities_df.iter_rows(named=True):
    prefix = str(row["code_prefix"]).strip().lower().replace(".", "")
    key = str(row["comorbidity"]).strip().lower()
    comorbidity_prefixes.setdefault(key, []).append(prefix)
print(f"Loaded comorbidity prefixes: " +
      ", ".join(f"{k}={len(v)}" for k, v in comorbidity_prefixes.items()))

# ---- 1) Compute ICD-10 cause + comorbidity flags via DuckDB SQL ----
# (all_ids_df is already bound above for the diagnostic queries.)
contraindication_codes_df = pd.DataFrame({"code": contraindication_codes})

# Build SQL clauses for each comorbidity (HCV/HTN/DM/CVA) — prefix LIKE chain
def _comorbidity_clause(key: str, prefixes: list[str]) -> str:
    likes = " OR ".join(f"dx_norm LIKE '{p}%'" for p in prefixes)
    return (
        f"CASE WHEN sys IN ('icd10','icd10cm') AND ({likes}) "
        f"THEN true ELSE false END AS icd10_{key}"
    )

comorbidity_select_clauses = ",\n        ".join(
    _comorbidity_clause(k, ps) for k, ps in comorbidity_prefixes.items()
)
comorbidity_bool_or_clauses = ",\n    ".join(
    f"BOOL_OR(icd10_{k}) AS icd10_{k}" for k in comorbidity_prefixes
)

query = f"""
WITH hospital_dx_normalized AS (
    SELECT
        hospitalization_id,
        LOWER(REGEXP_REPLACE(diagnosis_code, '[.\\s]', '', 'g')) AS dx_norm,
        LOWER(diagnosis_code_format) AS sys
    FROM read_parquet('{hospial_dx_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM all_ids_df)
),
hospital_dx_flags AS (
    SELECT
        hospitalization_id,
        CASE WHEN sys IN ('icd10','icd10cm') AND REGEXP_MATCHES(dx_norm, '^i2[0-5]\\w*$') THEN true ELSE false END AS icd10_ischemic,
        CASE WHEN sys IN ('icd10','icd10cm') AND REGEXP_MATCHES(dx_norm, '^i6[0-9]\\w*$') THEN true ELSE false END AS icd10_cerebro,
        CASE WHEN sys IN ('icd10','icd10cm') AND REGEXP_MATCHES(dx_norm, '^(v0[1-9]|v[1-9]\\d|w\\d{{2}}|x\\d{{2}}|y[0-8]\\d)\\w*$') THEN true ELSE false END AS icd10_external,
        CASE WHEN sys IN ('icd10','icd10cm') AND dx_norm IN (SELECT code FROM contraindication_codes_df) THEN true ELSE false END AS icd10_contraindication,
        {comorbidity_select_clauses}
    FROM hospital_dx_normalized
),
hospital_dx_with_patient AS (
    SELECT h.*, hosp.patient_id
    FROM hospital_dx_flags h
    LEFT JOIN hospitalization_df hosp ON h.hospitalization_id = hosp.hospitalization_id
)
SELECT
    patient_id,
    BOOL_OR(icd10_ischemic) AS icd10_ischemic,
    BOOL_OR(icd10_cerebro) AS icd10_cerebro,
    BOOL_OR(icd10_external) AS icd10_external,
    BOOL_OR(icd10_contraindication) AS icd10_contraindication,
    {comorbidity_bool_or_clauses}
FROM hospital_dx_with_patient
WHERE patient_id IS NOT NULL
GROUP BY patient_id
"""

print("Processing ICD flags with DuckDB...")
patient_cause_flags = pl.from_pandas(duckdb.sql(query).df())
print(f"✓ Processed {len(patient_cause_flags)} patients")
for col in ("icd10_ischemic", "icd10_cerebro", "icd10_external", "icd10_contraindication",
            *[f"icd10_{k}" for k in comorbidity_prefixes]):
    print(f"  {col}: {patient_cause_flags[col].sum()}")

# Join flags to final_df on patient_id; fill null flags to False ----
_comorbidity_fill = [pl.col(f"icd10_{k}").fill_null(False) for k in comorbidity_prefixes]
final_cohort_df = (
    final_cohort_df
    .join(patient_cause_flags, on="patient_id", how="left")
    .with_columns([
        pl.col("icd10_ischemic").fill_null(False),
        pl.col("icd10_cerebro").fill_null(False),
        pl.col("icd10_external").fill_null(False),
        pl.col("icd10_contraindication").fill_null(False),
        *_comorbidity_fill,
    ])
)

# Count patients with any of: ischemic OR cerebrovascular OR external cause (CALC cause, no age/location applied)
calc_cause_n = final_cohort_df.filter(
    pl.col("icd10_ischemic") | pl.col("icd10_cerebro") | pl.col("icd10_external")
)["patient_id"].n_unique()
strobe_counts["calc_cause"] = calc_cause_n

# Count patients with calc_cause (any cause) AND no contraindications
calc_cause_no_contraindication_n = final_cohort_df.filter(
    (pl.col("icd10_ischemic") | pl.col("icd10_cerebro") | pl.col("icd10_external")) & ~pl.col("icd10_contraindication")
)["patient_id"].n_unique()
strobe_counts["calc_cause_no_contraindication"] = calc_cause_no_contraindication_n

################################################################################
# CALC Criteria
# CMS adopts the Cause, Age, and Location-consistent (CALC) method to define “death consistent with organ donation” for donor-potential calculations:
# - **Age**: deaths ≤75 years
# - **Location**: inpatient deaths (death occurs in the hospital)
# - **Cause** (ICD-10-CM, inclusion ranges):
# - I20–I25: ischemic heart disease
# - I60–I69: cerebrovascular disease
# - V01–Y89: external causes (e.g., blunt trauma, gunshot wounds, overdose, suicide, drowning, asphyxiation)
# [Reference](https://www.cms.gov/files/document/112020-opo-final-rule-cms-3380-f.pdf)
################################################################################

final_cohort_df = final_cohort_df.with_columns(
    (
        (pl.col('age_75_less')) &
        (pl.col('icd10_ischemic') | pl.col('icd10_cerebro') | pl.col('icd10_external')) &
        (~pl.col('icd10_contraindication'))
    ).alias('calc_flag')
)

# Count for STROBE tracking
calc_qualified_n = final_cohort_df.filter(pl.col('calc_flag'))['patient_id'].n_unique()
strobe_counts["calc_qualified"] = calc_qualified_n

print(f"\nCALC flag qualified: {calc_qualified_n} patients")

strobe_counts

################################################################################
# IMV — streamed via DuckDB on clif_respiratory_support.parquet
################################################################################

resp_filepath = f"{tables_path}/clif_respiratory_support.{file_type}"
print("Processing IMV data with DuckDB...")

final_cohort_for_imv = final_cohort_df.select([
    "hospitalization_id", "patient_id", "encounter_block", "final_death_dttm"
]).to_pandas()

imv_query = f"""
WITH imv_data AS (
    SELECT
        hospitalization_id,
        recorded_dttm,
        device_category
    FROM read_parquet('{resp_filepath}')
    WHERE LOWER(device_category) = 'imv'
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_imv)
),
imv_with_death AS (
    SELECT
        i.hospitalization_id,
        i.recorded_dttm,
        f.patient_id,
        f.encounter_block,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - i.recorded_dttm)) / 3600 AS hr_2death_last_imv
    FROM imv_data i
    INNER JOIN final_cohort_for_imv f ON i.hospitalization_id = f.hospitalization_id
),
-- Get latest IMV record per patient first, then apply the time window
latest_imv_per_patient AS (
    SELECT
        patient_id,
        hospitalization_id,
        encounter_block,
        final_death_dttm,
        recorded_dttm,
        hr_2death_last_imv,
        ROW_NUMBER() OVER (
            PARTITION BY patient_id
            ORDER BY recorded_dttm DESC, hospitalization_id ASC
        ) AS rn
    FROM imv_with_death
)
SELECT
    patient_id,
    hospitalization_id,
    encounter_block,
    final_death_dttm,
    recorded_dttm,
    hr_2death_last_imv
FROM latest_imv_per_patient
WHERE rn = 1
    AND hr_2death_last_imv <= 48
    AND hr_2death_last_imv >= -24
"""

resp_expired_cohort = pl.from_pandas(duckdb.sql(imv_query).df())

# Add imv_48hr_expire flag to final_cohort_df first (so we can apply the age
# filter when counting). True if patient_id appears in resp_expired_cohort.
imv_48hr_expire_patients = resp_expired_cohort.select(["patient_id"]).unique().with_columns(
    pl.lit(True).alias("imv_48hr_expire")
)
final_cohort_df = final_cohort_df.join(imv_48hr_expire_patients, on="patient_id", how="left")
final_cohort_df = final_cohort_df.with_columns(pl.col("imv_48hr_expire").fill_null(False))

# Materialize the canonical "Died_While_IMV" cohort flag = age <=75 AND IMV
# within 48h before death. This is the same population shown as STROBE Stage 3
# and used as the Table 1 "Died_While_IMV" column and the Table 2 denominator,
# so all three stay consistent regardless of where the population gets read.
final_cohort_df = final_cohort_df.with_columns(
    (pl.col("imv_48hr_expire") & pl.col("age_75_less")).alias("died_while_imv")
)

imv_48hr_expire = final_cohort_df.filter(pl.col("died_while_imv"))["patient_id"].n_unique()
strobe_counts["6_imv_48hr_expire"] = imv_48hr_expire
print(f"✓ Died while receiving IMV (age <=75 + IMV <=48h): {imv_48hr_expire}")

################################################################################
# Organ quality check
# Pass the potential organ quality assessment check (independent assessment) using last recorded lab values, as defined by CMS
# * Kidney: recorded creatinine, cr  <4  AND not on CRRT
# * Liver: recorded TB, AST, ALT and Total bilirubin < 4, AST < 700, AND ALT< 700
# * BMI <=50
################################################################################

crrt_filepath = f"{tables_path}/clif_crrt_therapy.{file_type}"
labs_filepath = f"{tables_path}/clif_labs.{file_type}"

# ============================================
# CRRT within 48h of death — streamed via DuckDB
# ============================================
print("Processing CRRT data with DuckDB...")
final_cohort_for_crrt = final_cohort_df.select([
    "hospitalization_id", "final_death_dttm"
]).to_pandas()

crrt_query = f"""
WITH crrt_data AS (
    SELECT
        hospitalization_id,
        recorded_dttm
    FROM read_parquet('{crrt_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_crrt)
),
crrt_with_death AS (
    SELECT
        c.hospitalization_id,
        c.recorded_dttm,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - c.recorded_dttm)) / 3600 AS hrs_before_death
    FROM crrt_data c
    INNER JOIN final_cohort_for_crrt f ON c.hospitalization_id = f.hospitalization_id
    WHERE c.recorded_dttm <= f.final_death_dttm
)
SELECT DISTINCT hospitalization_id
FROM crrt_with_death
WHERE hrs_before_death <= 48 AND hrs_before_death >= 0
"""

crrt_48h_result = pl.from_pandas(duckdb.sql(crrt_query).df())
on_crrt_flag = crrt_48h_result.with_columns(
    pl.lit(True).alias('on_crrt_48h_before_death')
)
final_cohort_df = final_cohort_df.join(
    on_crrt_flag, on='hospitalization_id', how='left'
).with_columns(pl.col('on_crrt_48h_before_death').fill_null(False))

on_crrt_n = final_cohort_df.filter(pl.col('on_crrt_48h_before_death'))['patient_id'].n_unique()
print(f"✓ Patients on CRRT within 48h before death: {on_crrt_n}")

# ============================================
# Organ-quality labs (creatinine, bili, AST, ALT) — streamed via DuckDB
# ============================================
print("Processing Labs data with DuckDB...")
final_cohort_for_labs = final_cohort_df.select([
    "patient_id", "hospitalization_id", "final_death_dttm",
]).to_pandas()

labs_query = f"""
WITH labs_data AS (
    SELECT
        hospitalization_id,
        lab_collect_dttm,
        lab_category,
        lab_value_numeric
    FROM read_parquet('{labs_filepath}')
    WHERE hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_labs)
),
labs_with_death AS (
    SELECT
        l.hospitalization_id,
        l.lab_collect_dttm,
        l.lab_category,
        l.lab_value_numeric,
        f.patient_id,
        f.final_death_dttm
    FROM labs_data l
    INNER JOIN final_cohort_for_labs f ON l.hospitalization_id = f.hospitalization_id
    WHERE l.lab_collect_dttm <= f.final_death_dttm
),
latest_creatinine AS (
    SELECT
        hospitalization_id,
        lab_value_numeric AS creatinine_value,
        lab_collect_dttm AS creatinine_dttm
    FROM (
        SELECT
            hospitalization_id,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY hospitalization_id ORDER BY lab_collect_dttm DESC) AS rn
        FROM labs_with_death
        WHERE lab_category = 'creatinine'
    ) ranked
    WHERE rn = 1
),
latest_liver AS (
    SELECT
        hospitalization_id,
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_value_numeric END) AS bilirubin_total_value,
        MAX(CASE WHEN lab_category = 'bilirubin_total' THEN lab_collect_dttm END) AS bilirubin_total_dttm,
        MAX(CASE WHEN lab_category = 'ast' THEN lab_value_numeric END) AS ast_value,
        MAX(CASE WHEN lab_category = 'ast' THEN lab_collect_dttm END) AS ast_dttm,
        MAX(CASE WHEN lab_category = 'alt' THEN lab_value_numeric END) AS alt_value,
        MAX(CASE WHEN lab_category = 'alt' THEN lab_collect_dttm END) AS alt_dttm
    FROM (
        SELECT
            hospitalization_id,
            lab_category,
            lab_value_numeric,
            lab_collect_dttm,
            ROW_NUMBER() OVER (PARTITION BY hospitalization_id, lab_category ORDER BY lab_collect_dttm DESC) AS rn
        FROM labs_with_death
        WHERE lab_category IN ('bilirubin_total', 'ast', 'alt')
    ) ranked
    WHERE rn = 1
    GROUP BY hospitalization_id
)
SELECT DISTINCT
    f.patient_id,
    c.creatinine_value,
    c.creatinine_dttm,
    l.bilirubin_total_value,
    l.bilirubin_total_dttm,
    l.ast_value,
    l.ast_dttm,
    l.alt_value,
    l.alt_dttm
FROM final_cohort_for_labs f
LEFT JOIN latest_creatinine c ON f.hospitalization_id = c.hospitalization_id
LEFT JOIN latest_liver l ON f.hospitalization_id = l.hospitalization_id
"""

organ_labs = pl.from_pandas(duckdb.sql(labs_query).df())
print(f"✓ Organ labs loaded: {len(organ_labs)} patients")
print(f"  Patients with creatinine: {organ_labs.filter(pl.col('creatinine_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with bilirubin: {organ_labs.filter(pl.col('bilirubin_total_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with AST: {organ_labs.filter(pl.col('ast_value').is_not_null())['patient_id'].n_unique()}")
print(f"  Patients with ALT: {organ_labs.filter(pl.col('alt_value').is_not_null())['patient_id'].n_unique()}")

# Join organ_labs onto final_cohort_df by patient_id
final_cohort_df = final_cohort_df.join(
    organ_labs, on='patient_id', how='left', suffix='_organlab'
)
print(f"Final cohort with organ labs shape: {final_cohort_df.shape}")

# ============================================
# Create organ quality assessment flags
# ============================================
final_cohort_df = final_cohort_df.with_columns([
    # Kidney criteria: creatinine < 4 AND not on CRRT
    (
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < 4) &
        (~pl.col('on_crrt_48h_before_death'))
    ).alias('kidney_eligible'),

    # Liver criteria: all three labs recorded AND values within limits
    (
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('ast_value').is_not_null()) &
        (pl.col('alt_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < 4) &
        (pl.col('ast_value') < 700) &
        (pl.col('alt_value') < 700)
    ).alias('liver_eligible'),

    # BMI criteria: <= 50
    (
        (pl.col('bmi').is_not_null()) &
        (pl.col('bmi') <= 50)
    ).alias('bmi_eligible'),
])

# Overall: (kidney OR liver) AND BMI - done in separate call
final_cohort_df = final_cohort_df.with_columns([
    (
        (
            pl.col('kidney_eligible') | pl.col('liver_eligible')
        ) &
        pl.col('bmi_eligible')
    ).alias('organ_check_pass')
])

# Count for STROBE tracking
kidney_eligible_n = final_cohort_df.filter(pl.col('kidney_eligible'))['patient_id'].n_unique()
liver_eligible_n = final_cohort_df.filter(pl.col('liver_eligible'))['patient_id'].n_unique()
bmi_eligible_n = final_cohort_df.filter(pl.col('bmi_eligible'))['patient_id'].n_unique()
organ_check_pass_n = final_cohort_df.filter(pl.col('organ_check_pass'))['patient_id'].n_unique()

strobe_counts["organ_kidney_eligible"] = kidney_eligible_n
strobe_counts["organ_liver_eligible"] = liver_eligible_n
strobe_counts["organ_bmi_eligible"] = bmi_eligible_n
strobe_counts["organ_check_pass"] = organ_check_pass_n

print(f"\nOrgan Quality Assessment:")
print(f"  Kidney eligible: {kidney_eligible_n} patients")
print(f"  Liver eligible: {liver_eligible_n} patients")
print(f"  BMI eligible: {bmi_eligible_n} patients")
print(f"  Overall organ check pass: {organ_check_pass_n} patients")

################################################################################
# Microbiology
# Identify negative blood cultures and patients with no cultures in last 48h
################################################################################

# Microbiology — streamed via DuckDB on clif_microbiology_culture.parquet
print("Processing microbiology data with DuckDB...")
final_cohort_for_micro = final_cohort_df.select([
    'hospitalization_id', 'final_death_dttm'
]).to_pandas()

micro_query = f"""
WITH blood_cultures AS (
    SELECT
        hospitalization_id,
        collect_dttm,
        organism_category
    FROM read_parquet('{tables_path}/clif_microbiology_culture.{file_type}')
    WHERE fluid_category = 'blood_buffy'
        AND method_category = 'culture'
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_micro)
),
cultures_with_death AS (
    SELECT
        b.hospitalization_id,
        b.collect_dttm,
        b.organism_category,
        f.final_death_dttm,
        EXTRACT(EPOCH FROM (f.final_death_dttm - b.collect_dttm)) / 3600 AS hrs_before_death
    FROM blood_cultures b
    INNER JOIN final_cohort_for_micro f ON b.hospitalization_id = f.hospitalization_id
    WHERE b.collect_dttm IS NOT NULL
),
cultures_48h AS (
    SELECT
        *,
        CASE
            WHEN LOWER(organism_category) LIKE '%no_growth%'
                OR organism_category IS NULL
                OR LOWER(organism_category) = ''
            THEN true ELSE false
        END AS is_negative_culture
    FROM cultures_with_death
    WHERE hrs_before_death >= 0 AND hrs_before_death <= 48
),
positive_cultures AS (
    SELECT DISTINCT hospitalization_id
    FROM cultures_48h
    WHERE is_negative_culture = false
)
SELECT
    f.hospitalization_id,
    CASE WHEN p.hospitalization_id IS NULL THEN true ELSE false END AS no_positive_culture_48hrs
FROM final_cohort_for_micro f
LEFT JOIN positive_cultures p ON f.hospitalization_id = p.hospitalization_id
"""

no_positive_culture_flag = pl.from_pandas(duckdb.sql(micro_query).df())
final_cohort_df = final_cohort_df.join(
    no_positive_culture_flag, on='hospitalization_id', how='left'
).with_columns(pl.col('no_positive_culture_48hrs').fill_null(False))

# STROBE tracking
no_positive_culture_n = final_cohort_df.filter(pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
positive_culture_n = final_cohort_df.filter(~pl.col('no_positive_culture_48hrs'))['patient_id'].n_unique()
strobe_counts["no_positive_culture_48hrs"] = no_positive_culture_n
strobe_counts["positive_culture_48hrs"] = positive_culture_n
print(f"  Patients with no positive cultures in last 48h: {no_positive_culture_n}")
print(f"  Patients with positive cultures in last 48h: {positive_culture_n}")

final_cohort_df.columns

################################################################################
# CLIF Eligible Donor
# Medically eligible potential deceased abdominal organ donor (CLIF-eligible-donors):
# * From ALL inpatient deaths (ensure death location = ED, ward, stepdown, ICU)
# * Age < 75
# * On invasive mechanical ventilation
# * IF death date/time available: within 48h of death
# * IF no death date/time available: at time of last recorded vital signs
# * No contraindications
# * CLIF Microbiology_culture:
# * No positive blood cultures within 2 days - 'no_positive_culture_48hrs'
# * Hospital diagnosis (ICD based) -- 'icd10_contraindication',
# * Cancer
# * Severe sepsis
# * Pass the potential organ quality assessment check (independent assessment) using last recorded lab values, as defined by CMS:- organ_check_pass
# * Kidney: recorded creatinine, Cr < 4 AND not on CRRT
# * Liver: recorded TB, AST, ALT and
# * Total bilirubin < 4
# * AST < 700
# * ALT < 700
# * BMI <= 50
################################################################################

# ============================================
# Create CLIF-eligible-donors flag
# ============================================

final_cohort_df = final_cohort_df.with_columns([
    # Overall CLIF-eligible-donors flag
    (
        # 2. Age < 75
        (pl.col('age_75_less')) &
        # 3. On invasive mechanical ventilation (within 48h of death)
        (pl.col('imv_48hr_expire')) &
        # 4. No contraindications (no cancer, no severe sepsis)
        (~pl.col('icd10_contraindication')) &
        # 5. No positive blood cultures within 48h
        (pl.col('no_positive_culture_48hrs')) &
        # 6. Pass organ quality assessment (kidney OR liver AND BMI)
        (pl.col('organ_check_pass'))
    ).alias('clif_eligible_donors')
])

# Count for STROBE tracking
clif_eligible_n = final_cohort_df.filter(pl.col('clif_eligible_donors'))['patient_id'].n_unique()
strobe_counts["clif_eligible_donors"] = clif_eligible_n

################################################################################
# CLIF Donor Organ Eligibility Statistics
################################################################################
# Individual lab threshold flags for Table One
final_cohort_df = final_cohort_df.with_columns([
    # Terminal creatinine < 4
    (
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < 4)
    ).alias('creatinine_lt_4'),

    # Terminal bilirubin < 4
    (
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < 4)
    ).alias('bilirubin_lt_4'),

    # Terminal AST < 700
    (
        (pl.col('ast_value').is_not_null()) &
        (pl.col('ast_value') < 700)
    ).alias('ast_lt_700'),

    # Terminal ALT < 700
    (
        (pl.col('alt_value').is_not_null()) &
        (pl.col('alt_value') < 700)
    ).alias('alt_lt_700'),
])

# BMI-filtered versions of terminal lab thresholds (for fair comparison)
final_cohort_df = final_cohort_df.with_columns([
    # Terminal creatinine < 4 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('creatinine_value').is_not_null()) &
        (pl.col('creatinine_value') < 4)
    ).alias('creatinine_lt_4_bmi50'),

    # Terminal bilirubin < 4 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('bilirubin_total_value').is_not_null()) &
        (pl.col('bilirubin_total_value') < 4)
    ).alias('bilirubin_lt_4_bmi50'),

    # Terminal AST < 700 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('ast_value').is_not_null()) &
        (pl.col('ast_value') < 700)
    ).alias('ast_lt_700_bmi50'),

    # Terminal ALT < 700 (BMI ≤50 only)
    (
        (pl.col('bmi_eligible') == True) &
        (pl.col('alt_value').is_not_null()) &
        (pl.col('alt_value') < 700)
    ).alias('alt_lt_700_bmi50'),
])

# Filter to CLIF donors only (883 patients)
clif_donors_df = final_cohort_df.filter(pl.col('clif_eligible_donors'))

# Count organ eligibility among CLIF donors
clif_kidney_eligible_n = clif_donors_df.filter(pl.col('kidney_eligible'))['patient_id'].n_unique()
clif_liver_eligible_n = clif_donors_df.filter(pl.col('liver_eligible'))['patient_id'].n_unique()
clif_both_eligible_n = clif_donors_df.filter(
    pl.col('kidney_eligible') & pl.col('liver_eligible')
)['patient_id'].n_unique()

# Calculate percentages
clif_kidney_pct = (clif_kidney_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0
clif_liver_pct = (clif_liver_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0
clif_both_pct = (clif_both_eligible_n / clif_eligible_n * 100) if clif_eligible_n > 0 else 0

# Add to strobe_counts for tracking
strobe_counts["clif_kidney_eligible"] = clif_kidney_eligible_n
strobe_counts["clif_liver_eligible"] = clif_liver_eligible_n
strobe_counts["clif_both_kidney_liver_eligible"] = clif_both_eligible_n

print(f"\nCLIF Donor Organ Eligibility (n={clif_eligible_n}):")
print(f"  Kidney eligible (Cr <4 AND not on CRRT): {clif_kidney_eligible_n} ({clif_kidney_pct:.1f}%)")
print(f"  Liver eligible (Bili <4 AND AST <700 AND ALT <700): {clif_liver_eligible_n} ({clif_liver_pct:.1f}%)")
print(f"  Both kidney AND liver eligible: {clif_both_eligible_n} ({clif_both_pct:.1f}%)")

################################################################################
# Patient assessments
################################################################################

# Patient assessments — streamed via DuckDB on clif_patient_assessments.parquet
print("Processing patient assessments with DuckDB...")
final_cohort_for_assessments = final_cohort_df.select([
    "hospitalization_id", "final_death_dttm"
]).to_pandas()

assessments_query = f"""
WITH assessments_filtered AS (
    SELECT
        hospitalization_id,
        recorded_dttm,
        LOWER(assessment_category) AS assessment_category,
        numerical_value
    FROM read_parquet('{tables_path}/clif_patient_assessments.{file_type}')
    WHERE LOWER(assessment_category) IN ('gcs_total', 'rass')
        AND numerical_value IS NOT NULL
        AND hospitalization_id IN (SELECT hospitalization_id FROM final_cohort_for_assessments)
),
with_death_time AS (
    SELECT
        a.hospitalization_id,
        a.assessment_category,
        a.numerical_value,
        ABS(EXTRACT(EPOCH FROM (f.final_death_dttm - a.recorded_dttm))) AS abs_time_to_death
    FROM assessments_filtered a
    INNER JOIN final_cohort_for_assessments f ON a.hospitalization_id = f.hospitalization_id
),
closest_per_category AS (
    SELECT
        hospitalization_id,
        assessment_category,
        numerical_value,
        ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, assessment_category
            ORDER BY abs_time_to_death
        ) AS rn
    FROM with_death_time
)
SELECT
    hospitalization_id,
    MAX(CASE WHEN assessment_category = 'gcs_total' THEN numerical_value END) AS gcs_total_value,
    MAX(CASE WHEN assessment_category = 'rass'      THEN numerical_value END) AS rass_value
FROM closest_per_category
WHERE rn = 1
GROUP BY hospitalization_id
"""

patient_gcs_rass = pl.from_pandas(duckdb.sql(assessments_query).df())
print(f"✓ Processed assessments for {len(patient_gcs_rass)} hospitalizations")

final_cohort_df = final_cohort_df.join(
    patient_gcs_rass, on='hospitalization_id', how='left'
)

# ================================================================================
# ENSURE PATIENT-LEVEL ANALYSIS
# ================================================================================
print("\n" + "="*80)
print("FINALIZING PATIENT-LEVEL COHORT")
print("="*80)

# Step 1: First, remove encounter-level identifiers
print("Step 1: Removing encounter-level identifiers (hospitalization_id, encounter_block)...")
columns_to_drop = []
if 'hospitalization_id' in final_cohort_df.columns:
    columns_to_drop.append('hospitalization_id')
if 'encounter_block' in final_cohort_df.columns:
    columns_to_drop.append('encounter_block')

if columns_to_drop:
    final_cohort_df = final_cohort_df.drop(columns_to_drop)
    print(f"✓ Dropped: {', '.join(columns_to_drop)}")
else:
    print("✓ No encounter-level identifiers found to drop")

# Step 2: Sanity check — early dedup upstream means we should already be 1 row/patient.
# If this assertion fails, a downstream join introduced duplicates (would need a fix).
n_patients_final = final_cohort_df['patient_id'].n_unique()
n_rows_final = len(final_cohort_df)
assert n_patients_final == n_rows_final, (
    f"CRITICAL: Expected one row per patient (early dedup invariant violated). "
    f"{n_rows_final} rows but {n_patients_final} unique patients."
)
print(f"✓ One-row-per-patient invariant holds: {n_patients_final:,} patients = {n_rows_final:,} rows")

print(f"\n✓ Final verification passed: {n_patients_final:,} unique patients")
print(f"Final cohort shape: {final_cohort_df.shape}")
print("="*80 + "\n")

final_cohort_df.write_parquet(str(OUTPUT_INTERMEDIATE_DIR / "final_cohort_df.parquet"))
pd.DataFrame([strobe_counts]).to_csv(str(OUTPUT_FINAL_DIR / "strobe_counts.csv"), index=False)

################################################################################
# Table One (Overall / Died-while-IMV / CALC / CLIF) + NIDDK Aim 1 Table 2
# (PDF 1: stratified by terminal serum creatinine)
################################################################################

from utils.table_one import create_table_one, create_table_two_by_terminal_cr
table_one = create_table_one(final_cohort_df, output_dir=str(OUTPUT_FINAL_DIR))
table_two = create_table_two_by_terminal_cr(
    final_cohort_df,
    output_dir=str(OUTPUT_FINAL_DIR),
    cohort_filter_column='died_while_imv',
)

################################################################################
# Visualizations
################################################################################

strobe_counts

from utils.cohort_visualizations import create_all_visualizations
summary_df = create_all_visualizations(final_cohort_df, output_dir=str(OUTPUT_FINAL_DIR))

################################################################################
# STROBE
################################################################################

from utils.strobe_diagram import create_strobe_diagrams_for_cohorts
results = create_strobe_diagrams_for_cohorts(
      final_cohort_df,
      output_dir=str(OUTPUT_FINAL_DIR),
      save_figures=True,
      save_csvs=True
  )

# Access results
calc_stages = results['CALC']['stages']
clif_stages = results['CLIF']['stages']
