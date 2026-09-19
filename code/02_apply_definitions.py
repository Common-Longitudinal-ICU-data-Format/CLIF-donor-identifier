#!/usr/bin/env python
"""Phase 2 — apply the comparator definitions and attach hospital identity.

Reads the patient-level cohort produced by 01_potential_donor_identifier.py and
adds, per patient:

  * Possible Donor (PD)   the primary clinical comparator. Seven-step cascade
                          reproduced from the Kallan SAS (Goldberg AJT 2017).
  * Ventilated Patient    HRSA form comparator, with and without the age cap.
  * analytic_hospital_id  from config/hospital_crosswalk.yaml, so donation rates
                          can be computed at the hospital level (Will, 2026-08-20)
                          and the denominator is restricted to hospitals whose
                          CCN resolves in SRTR.

CLIF-donor and CALC already come from 01 and are passed through unchanged.

Every step writes an audit card; nothing filters silently.

    CLIF_DONOR_SITE=ucmc python code/02_apply_definitions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit
from utils.dtypes import write_parquet                      # noqa: E402
from utils.config import config                         # noqa: E402

SITE = config["site_name"]
TABLES = Path(config["tables_path"])
FT = config.get("file_type", "parquet")
INTER = Path(config["output_intermediate"])
FINAL = Path(config["output_final"])
FINAL.mkdir(parents=True, exist_ok=True)

CRIT = yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
XWALK = yaml.safe_load((REPO / "config/hospital_crosswalk.yaml").read_text())["hospitals"]
PD = CRIT["possible_donor"]

audit = StageAudit(SITE, INTER, label="definitions")


def tbl(name: str) -> str:
    return str(TABLES / f"clif_{name}.{FT}")


def norm_codes(series) -> set[str]:
    """ICD codes compare lowercase with periods and whitespace stripped."""
    return {str(c).lower().replace(".", "").strip() for c in series if c is not None}


# ── load ─────────────────────────────────────────────────────────────────────
cohort = pl.read_parquet(INTER / "final_cohort_df.parquet")
print(f"[{SITE}] cohort: {cohort.height:,} rows, "
      f"{cohort['patient_id'].n_unique():,} patients")

pd_inc = pl.read_csv(REPO / "utils/codes/pd_cause_inclusion_icd10.csv")
pd_cmp = pl.read_csv(REPO / "utils/codes/pd_severe_sepsis_components_icd10.csv")
contra = pl.read_csv(REPO / "utils/icd10_contraindications.csv", encoding="utf8-lossy")
contra.columns = [c.strip() for c in contra.columns]

PD_CAUSE = norm_codes(pd_inc["icd10cm"])
MSOF = norm_codes(pd_cmp.filter(pl.col("component") == "msof")["icd10cm"])
COMPONENTS = {c: norm_codes(pd_cmp.filter(pl.col("component") == c)["icd10cm"])
              for c in PD["severe_sepsis_exclusion"]["components"]}
SEPSIS = norm_codes(contra.filter(pl.col("dx_broad") == "sepsis")["ICD-10-CM"])
CANCER = norm_codes(contra.filter(pl.col("dx_broad") == "cancer")["ICD-10-CM"])
BRAIN_CA = {c for c in CANCER if c.startswith(("c70", "c71", "c72"))}
if not PD["cancer_exclusion"]["exclude_primary_brain_cancer"]:
    # SAS step 6 excludes CCS 11-34 and 36-45. CCS 35 (brain/CNS cancer) sits
    # OUTSIDE the exclusion because it is an INCLUSION at step 5.
    CANCER = CANCER - BRAIN_CA
EXT_LO, EXT_HI = PD["cause_inclusion"]["external_cause_icd10_range"]

print(f"  PD cause codes {len(PD_CAUSE):,} | sepsis {len(SEPSIS)} | cancer {len(CANCER):,} "
      f"(brain-ca carved out: {len(BRAIN_CA)}) | MSOF {len(MSOF)}")

# ── per-hospitalization diagnosis flags ──────────────────────────────────────
# 01 renames the terminal encounter's id; fall back for older outputs.
HID = "terminal_hospitalization_id" if "terminal_hospitalization_id" in cohort.columns else "hospitalization_id"
if HID not in cohort.columns:
    raise SystemExit("cohort lacks a hospitalization id — rerun 01_potential_donor_identifier.py")
cohort = cohort.rename({HID: "hospitalization_id"}) if HID != "hospitalization_id" else cohort
hosp_ids = cohort["hospitalization_id"].unique().to_list()
ids_df = pl.DataFrame({"hospitalization_id": [str(h) for h in hosp_ids]}).to_pandas()  # noqa: F841
con = duckdb.connect()
con.register("ids_df", ids_df)


def code_list(s: set[str]) -> str:
    return ",".join("'" + c.replace("'", "''") + "'" for c in s) or "''"


dx = con.sql(f"""
    WITH d AS (
        SELECT CAST(hospitalization_id AS VARCHAR) AS hospitalization_id,
               lower(replace(replace(CAST(diagnosis_code AS VARCHAR), '.', ''), ' ', '')) AS code,
               COALESCE(TRY_CAST(diagnosis_primary AS INTEGER), 0) AS is_primary,
               COALESCE(TRY_CAST(poa_present AS INTEGER), -1) AS poa
        FROM read_parquet('{tbl("hospital_diagnosis")}')
        WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
          AND lower(COALESCE(CAST(diagnosis_code_format AS VARCHAR),'icd10')) LIKE 'icd10%'
    )
    SELECT hospitalization_id,
           MAX(CASE WHEN code IN ({code_list(PD_CAUSE)}) THEN 1 ELSE 0 END)              AS pd_cause_dx,
           MAX(CASE WHEN code >= lower('{EXT_LO}') AND code < lower('{EXT_HI}zzz')
                     AND regexp_matches(code, '^[vwxy]') THEN 1 ELSE 0 END)              AS pd_cause_ext,
           MAX(CASE WHEN code IN ({code_list(MSOF)}) THEN 1 ELSE 0 END)                  AS pd_msof,
           MAX(CASE WHEN code IN ({code_list(CANCER)}) THEN 1 ELSE 0 END)                AS pd_cancer,
           MAX(CASE WHEN is_primary = 1 AND code IN ({code_list(SEPSIS)}) THEN 1 ELSE 0 END) AS sepsis_primary,
           MAX(CASE WHEN is_primary = 0 AND code IN ({code_list(COMPONENTS['aki'])}) THEN 1 ELSE 0 END) AS comp_aki,
           MAX(CASE WHEN is_primary = 0 AND code IN ({code_list(COMPONENTS['shock_liver'])}) THEN 1 ELSE 0 END) AS comp_shock_liver,
           MAX(CASE WHEN is_primary = 0 AND code IN ({code_list(COMPONENTS['encephalopathy'])}) THEN 1 ELSE 0 END) AS comp_enceph,
           MAX(CASE WHEN poa = 1 AND code IN ({code_list(CANCER | BRAIN_CA)}) THEN 1 ELSE 0 END) AS cancer_poa
    FROM d GROUP BY 1
""").pl()
print(f"  diagnosis flags for {dx.height:,} hospitalizations")

cohort = cohort.with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).join(
    dx, on="hospitalization_id", how="left").with_columns(
    [pl.col(c).fill_null(0) for c in dx.columns if c != "hospitalization_id"])

# ── hospital identity ────────────────────────────────────────────────────────
rows = [r for r in XWALK if r["site"] == SITE]
xw = pl.DataFrame([{
    "hospital_id_key": str(r["hospital_id"]).lower().strip(),
    "analytic_hospital_id": r["analytic_hospital_id"],
    "srtr_ccn_id": r.get("srtr_ccn_id"),
    "ccn_facility_type": r.get("ccn_facility_type"),
    "hospital_include": bool(r.get("include_flag")),
} for r in rows]) if rows else None

adt_h = con.sql(f"""
    SELECT CAST(hospitalization_id AS VARCHAR) AS hospitalization_id,
           lower(trim(CAST(hospital_id AS VARCHAR))) AS hospital_id_key,
           lower(trim(CAST(hospital_type AS VARCHAR))) AS hospital_type,
           ROW_NUMBER() OVER (PARTITION BY hospitalization_id ORDER BY out_dttm DESC NULLS LAST) rn
    FROM read_parquet('{tbl("adt")}')
    WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM ids_df)
      AND hospital_id IS NOT NULL
    QUALIFY rn = 1
""").pl().drop("rn")
# Terminal ADT record = the hospital where the patient died, which is the unit
# CMS and SRTR attribute a donor to.

before = cohort.height
cohort = cohort.join(adt_h, on="hospitalization_id", how="left")
if xw is not None:
    cohort = cohort.join(xw, on="hospital_id_key", how="left")
else:
    cohort = cohort.with_columns([pl.lit(None, pl.Utf8).alias(c) for c in
                                  ("analytic_hospital_id", "srtr_ccn_id", "ccn_facility_type")],
                                 ).with_columns(pl.lit(False).alias("hospital_include"))
cohort = cohort.with_columns(pl.col("hospital_include").fill_null(False))

# Anonymous hospital labels for anything that leaves the site. Numbered by CCN
# ascending so the mapping is deterministic across runs and across sites; the
# real CCN never appears in shareable output. The site keeps the key locally.
_ccns = sorted({c for c in cohort["srtr_ccn_id"].drop_nulls().unique().to_list()})
_labels = {c: f"{SITE}_hospital_{i}" for i, c in enumerate(_ccns, 1)}
cohort = cohort.with_columns(
    pl.col("srtr_ccn_id").replace_strict(_labels, default=None).alias("hospital_label"))
pl.DataFrame([{"site": SITE, "srtr_ccn_id": c, "hospital_label": l,
               "analytic_hospital_id": f"{SITE}_{c}"} for c, l in _labels.items()]
             ).write_csv(INTER / "hospital_label_key.csv")
print(f"  hospital labels: {len(_labels)} -> hospital_label_key.csv (kept local)")
audit.record("10_hospital_identity", cohort, cohort, key="patient_id",
             rule="terminal ADT hospital_id -> analytic_hospital_id via crosswalk",
             n_unmapped=int(cohort.filter(pl.col("analytic_hospital_id").is_null()).height))

# ── Possible Donor cascade ───────────────────────────────────────────────────
age_max = PD["age_at_death_max"]
los_max = PD["hospital_los_days_max"]
step = {}
c = cohort
step["pd1_inpatient_death"] = c
c = c.filter(pl.col("age_at_death") <= age_max)
step["pd2_age_0_74"] = c
c = c.filter(pl.col("imv_48hr_expire"))
step["pd3_ventilated"] = c
c = c.filter(pl.col("pd_msof") == 0)
step["pd4_no_msof"] = c
c = c.filter((pl.col("pd_cause_dx") == 1) | (pl.col("pd_cause_ext") == 1))
step["pd5_qualifying_cause"] = c
c = c.filter(pl.col("pd_cancer") == 0)
step["pd6_no_cancer"] = c
c = c.filter(~((pl.col("sepsis_primary") == 1) &
               ((pl.col("comp_aki") == 1) | (pl.col("comp_shock_liver") == 1) |
                (pl.col("comp_enceph") == 1))))
step["pd7_no_severe_sepsis"] = c
c = c.filter(pl.col("hospital_length_of_stay_days") <= los_max)
step["pd8_los_le_14d"] = c

RULES = [
    ("pd1_inpatient_death",  "SAS step 1: in-hospital death",                 ""),
    ("pd2_age_0_74",         f"SAS step 2: age 0-{age_max} (AGE lt 75)",      "older than 74"),
    ("pd3_ventilated",       "SAS step 3: ventilated (CLIF IMV <=48h, not billing codes)", "not ventilated"),
    ("pd4_no_msof",          "SAS step 4: NOT multi-system organ failure (R65.20/R65.21)", "MSOF coded"),
    ("pd5_qualifying_cause", "SAS step 5: CCS 35/47/109/228/233 or external cause V00-Y38", "no qualifying cause"),
    ("pd6_no_cancer",        "SAS step 6: no contraindicating cancer (CCS 11-34, 36-45)", "cancer coded"),
    ("pd7_no_severe_sepsis", "SAS step 7: NOT (primary sepsis AND AKI/shock liver/encephalopathy)", "severe sepsis"),
    ("pd8_los_le_14d",       f"hospital LOS <= {los_max} days",               "LOS over 14 days"),
]
prev = cohort
for name, rule, reason in RULES:
    audit.record(name, prev, step[name], key="patient_id", rule=rule, reason=reason)
    prev = step[name]

pd_ids = set(step["pd8_los_le_14d"]["patient_id"].to_list())

# ── Ventilated Patient, both variants ────────────────────────────────────────
vent_age = CRIT["ventilated_patient"]["apply_age_limit"]
cohort = cohort.with_columns([
    pl.col("patient_id").is_in(list(pd_ids)).alias("possible_donor"),
    pl.col("imv_48hr_expire").alias("ventilated_patient_no_age_limit"),
    (pl.col("imv_48hr_expire") & pl.col("age_75_less")).alias("ventilated_patient_age_le75"),
])
cohort = cohort.with_columns(
    (pl.col("ventilated_patient_age_le75") if vent_age
     else pl.col("ventilated_patient_no_age_limit")).alias("ventilated_patient"))

audit.record("20_ventilated_no_age_limit", cohort,
             cohort.filter(pl.col("ventilated_patient_no_age_limit")), key="patient_id",
             rule="IMV within 48h of death, NO age restriction (Table 1 as written)")
audit.record("21_ventilated_age_le75", cohort,
             cohort.filter(pl.col("ventilated_patient_age_le75")), key="patient_id",
             rule="IMV within 48h of death AND age <=75 (what the pipeline previously reported)")

# ── denominator restriction ──────────────────────────────────────────────────
analytic = cohort.filter(pl.col("hospital_include"))
audit.record("30_srtr_linkable_denominator", cohort, analytic, key="patient_id",
             rule="hospital retained only if include_flag AND srtr_ccn_id resolves in SRTR",
             reason="hospital has no SRTR-resolvable CCN; keeping it would inflate the denominator")

cohort = cohort.with_columns(pl.col("hospital_include").alias("in_srtr_denominator"))
write_parquet(cohort, INTER / "cohort_with_definitions.parquet")

counts = {
    "site": SITE,
    "n_inpatient_deaths": cohort["patient_id"].n_unique(),
    "n_srtr_linkable": analytic["patient_id"].n_unique(),
    "clif_donor": int(cohort.filter(pl.col("clif_eligible_donors"))["patient_id"].n_unique()),
    "possible_donor": int(cohort.filter(pl.col("possible_donor"))["patient_id"].n_unique()),
    "calc": int(cohort.filter(pl.col("calc_flag"))["patient_id"].n_unique()),
    "ventilated_no_age_limit": int(cohort.filter(pl.col("ventilated_patient_no_age_limit"))["patient_id"].n_unique()),
    "ventilated_age_le75": int(cohort.filter(pl.col("ventilated_patient_age_le75"))["patient_id"].n_unique()),
    "n_analytic_hospitals": int(analytic["analytic_hospital_id"].n_unique()),
}
pl.DataFrame([counts]).write_csv(FINAL / "definition_counts.csv")
audit.write()

print("\n" + audit.summary())
print("\n" + "=" * 60)
for k, v in counts.items():
    print(f"  {k:28s} {v}")
print(f"\nwrote {INTER/'cohort_with_definitions.parquet'}")

# ── SRTR reference: hospital identity and year coverage ──────────────────────
# The coordinating centre has to align an SRTR donor count (a fixed recovery-year
# window at a CCN) against a denominator this site actually observed. Those two
# periods are NOT the same: a hospital that joined the system mid-study
# contributes decedents for only part of the window, and counting its donors for
# the whole window inflates its rate. This block ships the coverage so the
# coordinating centre can align them instead of assuming.
SREF = FINAL / "srtr_ref"
SREF.mkdir(parents=True, exist_ok=True)

_cov = cohort.select(["patient_id", "hospital_id_key", "hospital_label",
                      "srtr_ccn_id", "ccn_facility_type", "hospital_type",
                      "in_srtr_denominator", "final_death_dttm"]).drop_nulls("hospital_id_key")

# one row per hospital-year, so a partial year is visible rather than averaged away
_hy = (_cov.with_columns(pl.col("final_death_dttm").dt.year().alias("year"))
       .group_by(["hospital_id_key", "hospital_label", "srtr_ccn_id", "year"])
       .agg(pl.col("patient_id").n_unique().alias("n_decedents"))
       .with_columns(pl.lit(SITE).alias("site"))
       .select(["site", "hospital_id_key", "hospital_label", "srtr_ccn_id",
                "year", "n_decedents"])
       .sort(["hospital_id_key", "year"]))
_hy.write_csv(SREF / "hospital_years.csv")

_hc = (_cov.group_by(["hospital_id_key", "hospital_label", "srtr_ccn_id",
                      "ccn_facility_type", "hospital_type", "in_srtr_denominator"])
       .agg(pl.col("patient_id").n_unique().alias("n_decedents"),
            pl.col("final_death_dttm").min().alias("first_death"),
            pl.col("final_death_dttm").max().alias("last_death"))
       .with_columns(
           pl.lit(SITE).alias("site"),
           pl.col("first_death").dt.year().alias("first_year"),
           pl.col("last_death").dt.year().alias("last_year"))
       .with_columns((pl.col("last_year") - pl.col("first_year") + 1).alias("n_years_spanned"))
       .sort("n_decedents", descending=True))
_hc.write_csv(SREF / "hospital_coverage.csv")

_sc = pl.DataFrame([{
    "site": SITE,
    "n_hospital_ids": _cov["hospital_id_key"].n_unique(),
    "n_hospital_ids_in_srtr_denominator":
        _cov.filter(pl.col("in_srtr_denominator"))["hospital_id_key"].n_unique(),
    "n_distinct_ccn": _cov["srtr_ccn_id"].drop_nulls().n_unique(),
    "n_decedents": _cov["patient_id"].n_unique(),
    "first_death": str(_cov["final_death_dttm"].min()),
    "last_death": str(_cov["final_death_dttm"].max()),
}])
_sc.write_csv(SREF / "site_coverage.csv")

print(f"\nSRTR reference -> {SREF}")
print(f"  {_hc.height} hospital_ids, {_cov['srtr_ccn_id'].drop_nulls().n_unique()} distinct CCNs")
for r in _hc.iter_rows(named=True):
    fd = r["first_death"].date() if r["first_death"] else "?"
    ld = r["last_death"].date() if r["last_death"] else "?"
    flag = "" if r["first_year"] and r["first_year"] <= 2020 else "   <-- PARTIAL COVERAGE"
    print(f"    {str(r['hospital_id_key'])[:32]:32s} ccn={str(r['srtr_ccn_id']):8s} "
          f"n={r['n_decedents']:5,}  {fd} -> {ld}{flag}")
