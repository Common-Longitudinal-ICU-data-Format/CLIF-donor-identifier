#!/usr/bin/env python
"""Phase 4 — sepsis definition diagnostic: CDC Adult Sepsis Event vs ICD-10.

Emily Vail's comment on Table 1 (2026-08-13, "Use ASE/CDC definition instead")
asks whether the CLIF-donor sepsis exclusion should come from structured EHR
data rather than diagnosis codes. This script answers that empirically at each
site; it does NOT change any definition.

Two things worth knowing before reading the output:

  * The ICD arm has no clock. clif_hospital_diagnosis carries no timestamp, so
    an ICD sepsis flag is necessarily "coded anywhere in the terminal
    hospitalization". The neighbouring criterion in the same Table 1 cell
    (positive blood culture) is restricted to 48 h. ASE, by contrast, has an
    onset datetime, so it is reported BOTH ways — ever and within 48 h of
    death — and the group can pick.

  * ASE needs blood cultures. A site with sparse microbiology will under-call
    sepsis and therefore over-call CLIF-donor eligibility, which is the
    opposite failure direction from ICD codes. no_sepsis_reason and blood
    culture coverage are reported so that is visible rather than assumed.

    CLIF_DONOR_SITE=ucmc python code/04_sepsis_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.audit import StageAudit                       # noqa: E402
from utils.config import config                          # noqa: E402

SITE = config["site_name"]
TABLES = Path(config["tables_path"])
FT = config.get("file_type", "parquet")
INTER = Path(config["output_intermediate"])
FINAL = Path(config["output_final"])
FINAL.mkdir(parents=True, exist_ok=True)

CRIT = yaml.safe_load((REPO / "config/donor_criteria.yaml").read_text())
WINDOW_H = CRIT["clif_donor"].get("sepsis_window_hours", 48)

audit = StageAudit(SITE, INTER, label="sepsis_diagnostic")

cohort = pl.read_parquet(INTER / "cohort_with_variables.parquet")
audit.note("cohort loaded", f"{cohort.height:,} patients from 02")

# The sepsis criterion only bites on decedents who have already cleared age and
# IMV, so that is the population every comparison below is computed on.
elig = cohort.filter(pl.col("age_75_less").fill_null(False)
                     & pl.col("imv_48hr_expire").fill_null(False))
audit.record("age <=75 and IMV within 48h", cohort, elig, key="patient_id",
             rule="population where the sepsis rule decides eligibility")

hosp_ids = (elig.select("hospitalization_id").drop_nulls()
            .with_columns(pl.col("hospitalization_id").cast(pl.Utf8))
            .unique()["hospitalization_id"].to_list())

# ── 1. ICD-10 contraindication breakdown ─────────────────────────────────────
# Which arm of the contraindication actually removes people, and how much of it
# is sepsis acting alone.
codes = (pl.read_csv(REPO / "utils/icd10_contraindications.csv")
         .rename({"ICD-10-CM": "code"}).select(["code", "dx_broad"]))
con = duckdb.connect()
con.register("codes", codes.to_pandas())
con.register("coh", elig.select(["patient_id", "hospitalization_id"])
             .with_columns(pl.col("hospitalization_id").cast(pl.Utf8)).to_pandas())
con.execute(f"""
CREATE TABLE hit AS
SELECT DISTINCT coh.patient_id, codes.dx_broad
FROM read_parquet('{TABLES}/clif_hospital_diagnosis.{FT}') d
JOIN coh ON CAST(d.hospitalization_id AS VARCHAR) = coh.hospitalization_id
JOIN codes ON UPPER(REPLACE(d.diagnosis_code, '.', '')) = UPPER(REPLACE(codes.code, '.', ''))
""")
n_elig = elig["patient_id"].n_unique()
brk = con.execute("""
SELECT dx_broad AS arm, COUNT(DISTINCT patient_id) AS n_patients FROM hit GROUP BY 1
UNION ALL SELECT 'any', COUNT(DISTINCT patient_id) FROM hit
UNION ALL SELECT 'sepsis_only', COUNT(DISTINCT patient_id) FROM hit h
  WHERE dx_broad = 'sepsis' AND NOT EXISTS
    (SELECT 1 FROM hit x WHERE x.patient_id = h.patient_id AND x.dx_broad <> 'sepsis')
UNION ALL SELECT 'cancer_only', COUNT(DISTINCT patient_id) FROM hit h
  WHERE dx_broad = 'cancer' AND NOT EXISTS
    (SELECT 1 FROM hit x WHERE x.patient_id = h.patient_id AND x.dx_broad <> 'cancer')
ORDER BY n_patients DESC
""").pl()
brk = brk.with_columns(
    pl.lit(SITE).alias("site"), pl.lit(n_elig).alias("denominator"),
    (100 * pl.col("n_patients") / n_elig).round(1).alias("pct_of_denominator"),
).select(["site", "arm", "n_patients", "denominator", "pct_of_denominator"])
brk.write_csv(FINAL / "sepsis_contraindication_breakdown.csv")
print(brk)

# ── 2. CDC Adult Sepsis Event ────────────────────────────────────────────────
try:
    from clifpy.utils.ase import compute_ase
except ImportError:
    print("clifpy.utils.ase unavailable — install clifpy>=2.1.0. Skipping ASE.")
    audit.write()
    raise SystemExit(0)

ase = compute_ase(
    hospitalization_ids=hosp_ids,
    data_directory=str(TABLES),
    filetype=FT,
    timezone=config.get("timezone", "UTC"),
    include_lactate=False,
    verbose=False,
)
ase = pl.from_pandas(ase)
print(f"ASE rows: {ase.height} over {len(hosp_ids)} hospitalizations")

# compute_ase names the flag `sepsis` with lactate and `sepsis_wo_lactate`
# without it; include_lactate=False is the CDC default so prefer that column.
flag = "sepsis_wo_lactate" if "sepsis_wo_lactate" in ase.columns else "sepsis"
onset = next((c for c in ("ase_onset_wo_lactate_dttm", "ase_onset_w_lactate_dttm",
                          "ase_onset_dttm") if c in ase.columns), None)
if onset is None:
    raise SystemExit(f"no ASE onset column in {ase.columns}")
print(f"  using flag={flag} onset={onset}")

ase = ase.with_columns(
    pl.col("hospitalization_id").cast(pl.Utf8),
    pl.col(flag).fill_null(0).cast(pl.Int64).alias("_sep"),
    pl.col(onset).cast(pl.Datetime("us")).alias("_onset"),
)

deaths = (elig.select(["patient_id", "hospitalization_id", "final_death_dttm"])
          .with_columns(pl.col("hospitalization_id").cast(pl.Utf8),
                        pl.col("final_death_dttm").cast(pl.Datetime("us"))))
ep = ase.join(deaths, on="hospitalization_id", how="inner").with_columns(
    ((pl.col("final_death_dttm") - pl.col("_onset")).dt.total_hours()).alias("hours_before_death"))

per_pt = ep.group_by("patient_id").agg([
    (pl.col("_sep") == 1).any().alias("ase_ever"),
    ((pl.col("_sep") == 1)
     & pl.col("hours_before_death").is_between(0, WINDOW_H)).any().alias("ase_window"),
    pl.col("blood_culture_dttm").is_not_null().any().alias("had_blood_culture"),
])

# ICD sepsis, coded anywhere in the terminal hospitalization
icd = con.execute("SELECT DISTINCT patient_id FROM hit WHERE dx_broad = 'sepsis'").pl()
d = (elig.select("patient_id").unique()
     .join(per_pt, on="patient_id", how="left")
     .join(icd.with_columns(pl.lit(True).alias("icd_sepsis")), on="patient_id", how="left")
     .with_columns([pl.col(c).fill_null(False) for c in
                    ("ase_ever", "ase_window", "had_blood_culture", "icd_sepsis")]))


def agreement(a: str, b: str, label: str) -> dict:
    """2x2 plus kappa and positive agreement between two boolean flags."""
    n11 = int((d[a] & d[b]).sum())
    n10 = int((d[a] & ~d[b]).sum())
    n01 = int((~d[a] & d[b]).sum())
    n00 = int((~d[a] & ~d[b]).sum())
    n = n11 + n10 + n01 + n00
    po = (n11 + n00) / n if n else 0.0
    pe = (((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)) if n else 0.0
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"site": SITE, "comparison": label,
            "both": n11, f"{a}_only": n10, f"{b}_only": n01, "neither": n00, "n": n,
            "pct_agree": round(100 * po, 1), "kappa": round(kappa, 3),
            "positive_agreement": round(200 * n11 / (2 * n11 + n10 + n01), 1)
            if (2 * n11 + n10 + n01) else None}


rows = []
for a, lab in [("ase_ever", f"ASE (ever) vs ICD-10 sepsis (ever)"),
               ("ase_window", f"ASE (onset within {WINDOW_H}h of death) vs ICD-10 sepsis (ever)")]:
    r = agreement(a, "icd_sepsis", lab)
    rows.append({"site": r["site"], "comparison": r["comparison"], "both": r["both"],
                 "ase_only": r[f"{a}_only"], "icd_only": r["icd_sepsis_only"],
                 "neither": r["neither"], "n": r["n"], "pct_agree": r["pct_agree"],
                 "kappa": r["kappa"], "positive_agreement": r["positive_agreement"]})
agree = pl.DataFrame(rows)
agree.write_csv(FINAL / "sepsis_ase_vs_icd.csv")
print(agree)

# ── 3. what each sepsis rule does to CLIF-donor eligibility ──────────────────
# Everything except the sepsis arm is held fixed; only the sepsis rule swaps.
cancer = con.execute("SELECT DISTINCT patient_id FROM hit WHERE dx_broad <> 'sepsis'").pl()
base = (elig.select(["patient_id", "no_positive_culture_48hrs", "organ_check_pass"])
        .unique(subset=["patient_id"])
        .join(cancer.with_columns(pl.lit(True).alias("icd_other")), on="patient_id", how="left")
        .join(d, on="patient_id", how="left")
        .with_columns([pl.col(c).fill_null(False) for c in
                       ("icd_other", "icd_sepsis", "ase_ever", "ase_window",
                        "no_positive_culture_48hrs", "organ_check_pass")]))
rest = (~pl.col("icd_other")) & pl.col("no_positive_culture_48hrs") & pl.col("organ_check_pass")
impact = pl.DataFrame([
    {"site": SITE, "sepsis_rule": lab,
     "n_excluded_by_sepsis": int(base.filter(expr)["patient_id"].n_unique()),
     "n_clif_donor_eligible": int(base.filter(rest & ~expr)["patient_id"].n_unique())}
    for lab, expr in [
        ("ICD-10 sepsis, ever (current)", pl.col("icd_sepsis")),
        ("CDC ASE, ever", pl.col("ase_ever")),
        (f"CDC ASE, onset within {WINDOW_H}h of death", pl.col("ase_window")),
        ("no sepsis exclusion", pl.lit(False)),
    ]])
impact = impact.with_columns(pl.lit(n_elig).alias("denominator"))
impact.write_csv(FINAL / "sepsis_definition_impact.csv")
print(impact)

# ── 4. why ASE did not fire ──────────────────────────────────────────────────
# A site with no microbiology looks sepsis-free; this is how that shows up.
why = (ep.filter(pl.col("_sep") != 1).group_by("no_sepsis_reason")
       .agg(pl.col("patient_id").n_unique().alias("n_patients"))
       .sort("n_patients", descending=True)
       if "no_sepsis_reason" in ep.columns else pl.DataFrame([]))
cov = pl.DataFrame([
    {"site": SITE, "reason": "_denominator", "n_patients": int(n_elig)},
    {"site": SITE, "reason": "_with_any_blood_culture",
     "n_patients": int(d["had_blood_culture"].sum())},
], schema={"site": pl.Utf8, "reason": pl.Utf8, "n_patients": pl.Int64})
why = (pl.concat([cov, why.with_columns(pl.lit(SITE).alias("site"))
                  .rename({"no_sepsis_reason": "reason"})
                  .select(["site", "reason", "n_patients"])
                  .with_columns(pl.col("n_patients").cast(pl.Int64))], how="vertical")
       if why.height else cov)
why.write_csv(FINAL / "sepsis_ase_no_sepsis_reason.csv")
print(why)

audit.note("ASE, ever", f'{int(d["ase_ever"].sum()):,} patients with an ASE episode')
audit.note(f"ASE within {WINDOW_H}h of death", f'{int(d["ase_window"].sum()):,} patients')
audit.note("ICD-10 sepsis, ever", f'{int(d["icd_sepsis"].sum()):,} patients (current CLIF-donor rule)')
audit.write()

# ── 5. which organ dysfunction ASE fired on ──────────────────────────────────
# This cohort is DEFINED by invasive ventilation within 48 h of death, and "new
# IMV" is itself one of ASE's Component B organ dysfunction criteria. So ASE can
# fire on the same ventilation that put the patient in the denominator. The
# `imv_only` row counts episodes where IMV is the ONLY dysfunction present —
# those are the ones that circularity would account for.
DYS = ["vasopressor", "imv", "aki", "hyperbilirubinemia", "thrombocytopenia"]
have = [c for c in DYS if f"{c}_dttm" in ep.columns]
pos = ep.filter(pl.col("_sep") == 1)
if pos.height and have:
    non_imv = [c for c in have if c != "imv"]
    rows = [{"site": SITE, "criterion": c,
             "n_episodes": int(pos[f"{c}_dttm"].is_not_null().sum())} for c in have]
    rows.append({"site": SITE, "criterion": "_ase_positive_episodes", "n_episodes": pos.height})
    if "imv" in have and non_imv:
        only_imv = pos.filter(
            pl.col("imv_dttm").is_not_null()
            & pl.all_horizontal([pl.col(f"{c}_dttm").is_null() for c in non_imv]))
        rows.append({"site": SITE, "criterion": "_imv_only_no_other_dysfunction",
                     "n_episodes": only_imv.height})
    comp = pl.DataFrame(rows).sort("n_episodes", descending=True)
    comp.write_csv(FINAL / "sepsis_ase_criteria_composition.csv")
    print(comp)

# `ase_first_criteria` is the earliest qualifying event, i.e. what set the onset
# clock, not necessarily the dysfunction itself.
fc = next((c for c in ("ase_first_criteria_wo_lactate", "ase_first_criteria_w_lactate")
           if c in ep.columns), None)
if fc and pos.height:
    (pos.group_by(fc).agg(pl.len().alias("n_episodes"))
     .rename({fc: "first_criterion"}).with_columns(pl.lit(SITE).alias("site"))
     .select(["site", "first_criterion", "n_episodes"])
     .sort("n_episodes", descending=True)
     .write_csv(FINAL / "sepsis_ase_onset_criterion.csv"))

print(f"\nsepsis diagnostic -> {FINAL}")

# ── 6. code-level diagnostics: which codes actually fire, per site ────────────
# Every ICD code used by any definition, with how many decedents carry it and
# what it does to them. Two roles: a code can DROP a patient (contraindication)
# or KEEP them (qualifying cause of death). The same code can do both in
# different definitions, which is the point.
_codes = []
_contra = pl.read_csv(REPO / "utils/icd10_contraindications.csv",
                      infer_schema_length=0).rename({"ICD-10-CM": "code"})
for r in _contra.iter_rows(named=True):
    _codes.append({"code": str(r["code"]).upper().replace(".", "").strip(),
                   "description": r["description"],
                   "role": f"contraindication ({r['dx_broad']})",
                   "used_by": "CLIF-donor, Possible Donor, CALC"})
_pdc = pl.read_csv(REPO / "utils/codes/pd_cause_inclusion_icd10.csv", infer_schema_length=0)
for r in _pdc.iter_rows(named=True):
    _codes.append({"code": str(r["icd10cm"]).upper().replace(".", "").strip(),
                   "description": r["description"],
                   "role": f"qualifying cause of death ({r['concept']})",
                   "used_by": "Possible Donor"})
_comp = pl.read_csv(REPO / "utils/codes/pd_severe_sepsis_components_icd10.csv",
                    infer_schema_length=0)
for r in _comp.iter_rows(named=True):
    _codes.append({"code": str(r["icd10cm"]).upper().replace(".", "").strip(),
                   "description": r["description"],
                   "role": f"PD severe sepsis component ({r['component']})",
                   "used_by": "Possible Donor"})

_cl = pl.DataFrame(_codes).unique(subset=["code", "role"], keep="first")
con.register("codelist", _cl.to_pandas())
con.register("coh_all", cohort.select(
    ["patient_id", "hospitalization_id", "age_75_less", "imv_48hr_expire",
     "no_positive_culture_48hrs", "organ_check_pass", "clif_eligible_donors",
     "possible_donor"]).with_columns(
    pl.col("hospitalization_id").cast(pl.Utf8)).to_pandas())

_obs = con.sql(f"""
    SELECT cl.code, cl.description, cl.role, cl.used_by,
           count(DISTINCT c.patient_id) AS n_decedents,
           count(DISTINCT CASE WHEN c.age_75_less AND c.imv_48hr_expire
                 THEN c.patient_id END) AS n_after_age_and_imv,
           count(DISTINCT CASE WHEN c.age_75_less AND c.imv_48hr_expire
                 AND c.no_positive_culture_48hrs AND c.organ_check_pass
                 AND NOT c.clif_eligible_donors THEN c.patient_id END)
                 AS n_clif_eligible_but_for_a_contraindication,
           count(DISTINCT CASE WHEN c.clif_eligible_donors THEN c.patient_id END)
                 AS n_among_clif_donors,
           count(DISTINCT CASE WHEN c.possible_donor THEN c.patient_id END)
                 AS n_among_possible_donors
    FROM read_parquet('{TABLES}/clif_hospital_diagnosis.{FT}') d
    JOIN coh_all c ON CAST(d.hospitalization_id AS VARCHAR) = c.hospitalization_id
    JOIN codelist cl ON UPPER(REPLACE(d.diagnosis_code, '.', '')) = cl.code
    GROUP BY 1,2,3,4
""").pl().with_columns(pl.lit(SITE).alias("site")).sort(
    ["role", "n_decedents"], descending=[False, True])

_obs.write_csv(FINAL / "code_level_diagnostics.csv")
print(f"\n6. code-level diagnostics -> {_obs.height} code-role rows observed at {SITE}")
_unused = _cl.height - _obs.select(["code", "role"]).unique().height
print(f"   {_unused} of {_cl.height} code-role entries never appear in any decedent here")
