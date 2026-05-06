# NIDDK Aim 1 — Kidney Donor Reporting Tables (Design)

**Date:** 2026-05-06
**Branch:** `niddk`
**Site:** UCMC (single-site preliminary data)
**Sponsor context:** R01 preliminary data request for NIDDK; descriptive characterization of SRTR-validated CLIF donor data.

## 1. Goal

Produce two summary tables and one figure that describe SRTR-validated CLIF medically-eligible donor decedents at UCMC, with kidney donation as the primary outcome and number of organs transplanted as a secondary outcome. Output is a single self-contained HTML report suitable for grant submission attachments.

## 2. Population

| Aspect | Definition |
|---|---|
| Base | All CLIF medically-eligible donor hospitalizations linked to SRTR via the existing record-linkage pipeline (rows in `output/intermediate/encounter_mapping_matched.parquet`) |
| Filter | `match_confidence ∈ {HIGH, MEDIUM}` AND `is_best == True` |
| Strata | **Overall** (M+H combined), **Moderate** (M only), **High** (H only) — three columns per table |
| Site | UCMC |

Cohort is defined post-linkage; kidney donation is an outcome, not a cohort filter.

## 3. Required Outputs

### Table 1 — Donor characteristics
Columns: Variable | Overall | Moderate | High

| Variable | Source | Operationalization | Summary |
|---|---|---|---|
| N (% of cohort) | derived | row count | n (col %) |
| Age | CLIF `age_at_admission` (or computed from birth/death) | years | median (IQR) |
| Sex | CLIF `sex_category` | M / F | n (%) |
| Height | CLIF `last_height_cm` | cm | median (IQR) |
| Weight | CLIF `last_weight_kg` | kg | median (IQR) |
| BMI | CLIF `bmi` | kg/m² | median (IQR) |
| Mechanism of death | SRTR `DON_DEATH_MECH` | categorical | n (%) per category |
| Hx CVA | SRTR `DON_COD_DON_STROKE` | 1=yes, 0=no | n (%) yes |
| HTN | SRTR `DON_HIST_HYPERTEN` | 'Y' = yes | n (%) yes |
| DM | SRTR `DON_HIST_DIAB` | non-'N' / non-missing = yes | n (%) yes |
| KDPI | computed (see §5) | 0–100 percentile | mean (SD) |
| HCV | SRTR `DON_ANTI_HCV` | 'P' = positive | n (%) positive |
| HIV | SRTR `DON_ANTI_HIV` | 'P' = positive | n (%) positive |
| Terminal creatinine (CLIF) | CLIF `creatinine_value` (last value during hospitalization, already in `final_clif_data.parquet`) | mg/dL | median (IQR) |
| Terminal creatinine (SRTR) | SRTR `DON_FINAL_SERUM_CREAT` | mg/dL | median (IQR) |
| Vasopressors within 48h death | CLIF `wide_df` | any of {norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine} with non-null/non-zero dose during `[death_dttm − 48h, death_dttm]` | n (%) yes |
| IMV within 48h death | CLIF `wide_df` | `resp_device_category` = 'IMV' on any row during `[death_dttm − 48h, death_dttm]`. *Note: ~100% by construction since the CLIF cohort filter already requires IMV in this window; reported as a completeness/sanity confirmation row.* | n (%) yes |
| ICU LOS | CLIF `adt` | sum of time with `location_category='icu'` across stays, in days | median (IQR) |

**Explicitly excluded:** HgbA1c — not present in CLIF labs at this site.

### Table 2 — Donation outcomes
Columns: Variable | Overall | Moderate | High

| Variable | Source | Operationalization | Summary |
|---|---|---|---|
| N (% of cohort) | derived | row count | n (col %) |
| DBD | SRTR `DON_NON_HR_BEAT` | 'N' (not non-heart-beating) → DBD | n (%) yes |
| Kidney donation | SRTR `donor_disposition` | ≥1 kidney with disposition = transplanted | n (%) yes |
| Number of organs per donor | SRTR `donor_disposition` | count of organs with disposition = transplanted | median (range) |

### Figure — Creatinine trajectory
Per-hospitalization line plot of serum creatinine over time:
- X axis: hours relative to `death_dttm` (right edge = 0 = death)
- Y axis: serum creatinine (mg/dL)
- One line per matched hospitalization; line color encodes `kidney_donated` (Yes / No)
- Source: `wide_df.lab_creatinine` joined to death timestamp
- Saved standalone as PNG and base64-embedded in the HTML report

## 4. Architecture

### Inputs
| File | Provides |
|---|---|
| `output/intermediate/encounter_mapping_matched.parquet` | confidence labels, DONOR_ID ↔ hospitalization_id |
| `output/intermediate/final_clif_data.parquet` | CLIF demographics, BMI, terminal Cr |
| `output/intermediate/final_srtr_data.parquet` *(widened — see §6)* | SRTR donor characteristics for KDPI inputs and Table 1 rows |
| `output/intermediate/wide_df.parquet` | longitudinal vasopressors and creatinine |
| `output/intermediate/donor_outcomes.parquet` *(new — see §6)* | per-donor kidney/organ donation flags |
| `tables_path/clif_adt.parquet` | ICU LOS computation |

### Files added
- `utils/kdpi.py` — KDPI calculator module
- `record_linkage/03_niddk_aim1_tables.py` — orchestrator script
- `tests/test_kdpi.py` — unit tests for KDPI module
- `output/final/niddk_aim1_report.html` — report (output)
- `output/final/niddk_aim1_creatinine_trajectory.png` — figure (output)
- `output/final/niddk_aim1_table1.csv`, `niddk_aim1_table2.csv` — tabular outputs
- `output/intermediate/donor_outcomes.parquet` — new intermediate (output)

### Files modified
- `utils/srtr_linkage.py` — widen the SRTR projection (in place; not a parallel loader)
- `record_linkage/01_match_hierarchical.py` — load `donor_disposition.sas7bdat` and persist `donor_outcomes.parquet` (in place; alongside existing donor_deceased load)

## 5. KDPI Module (`utils/kdpi.py`)

Implements the OPTN 2024 refit formula (8 factors; race and HCV removed from prior 10-factor model). Source documents in [references/kdpi_guide.pdf](../../references/kdpi_guide.pdf) (April 21, 2025) and [references/kdpi_mapping_table.pdf](../../references/kdpi_mapping_table.pdf) (2024 reference cohort).

### Formula

```
Xβ = 0.0092·(age − 40)
   + 0.0113·(age − 18)·I(age < 18)
   + 0.0067·(age − 50)·I(age > 50)
   − 0.0557·(height_cm − 170)/10
   − 0.0333·(weight_kg − 80)/5 · I(weight_kg < 80)
   + 0.1106·I(hypertension = yes)
   + 0.2577·I(diabetes = yes)
   + 0.0743·I(cause_of_death = CVA)
   + 0.2128·(creatinine − 1)
   − 0.2199·(creatinine − 1.5)·I(creatinine > 1.5)
   + 0.1966·I(DCD)

KDRI_RAO    = exp(Xβ)
KDRI_SCALED = KDRI_RAO / 1.40436817065005     # 2024 scaling factor
KDPI        = lookup(KDRI_SCALED) in 102-row mapping table → integer percent
```

### Constants (cited inline, hard-coded from the OPTN PDFs)
| Name | Value | Source |
|---|---|---|
| `KDRI_SCALING_FACTOR_2024` | 1.40436817065005 | mapping table footer |
| `CR_CAP` | 8.0 mg/dL | guide (creatinine values >8 capped) |
| `DM_UNKNOWN_PROB` | 0.17280542134655 | mapping table footer (used if DM=unknown) |
| `HTN_UNKNOWN_PROB` | 0.43697057162578 | mapping table footer (used if HTN=unknown) |
| `KDPI_MAPPING` | 102 (low_exclusive, high_inclusive, kdpi_pct) tuples | mapping table body |

### Public API
```python
def compute_kdri_xb(age, height_cm, weight_kg, hist_htn, hist_dm, cod_cva, creatinine, dcd) -> float
def compute_kdri_rao(age, height_cm, weight_kg, hist_htn, hist_dm, cod_cva, creatinine, dcd) -> float
def kdri_scaled_to_kdpi(kdri_scaled: float) -> int
def compute_kdpi(donor_row: dict | pd.Series) -> Optional[int]
```

`compute_kdpi` orchestrates the chain. Inputs map from SRTR field names (`DON_AGE`, `DON_HGT_CM`, `DON_WGT_KG`, `DON_HIST_HYPERTEN`, `DON_HIST_DIAB`, `DON_COD_DON_STROKE`, `DON_FINAL_SERUM_CREAT`, `DON_NON_HR_BEAT`).

### Missing-data policy
- If any of the 8 required inputs is missing (NaN/None) → return `None` (KDPI reported as missing)
- Exception: HTN and DM may be coded "unknown" (e.g., `'U'`) in SRTR. For these, use the OPTN-published probabilistic Xβ contributions:
  - HTN unknown contributes `0.1106 · 0.43697057 = 0.04833`
  - DM unknown contributes `0.2577 · 0.17280542 = 0.04453`
- Creatinine values > `CR_CAP` are capped at 8.0 before formula evaluation
- Age, height, weight bounds are validated; out-of-range values produce `None` and a logged warning

### Validation
Unit test using the OPTN guide worked example:
- Inputs: Age=52, Height=183, Weight=81, HTN=Yes, DM=No, COD=CVA, Cr=1.7, DCD=Yes
- Expected: Xβ = 0.53787000000000, KDRI_RAO = 1.71235565748184
- Asserts numeric equality to ≥1e-10 tolerance

(Note: the guide demonstrates KDPI=79% using the 2023 scaling factor; we use the 2024 scaling factor and confirm a sane KDPI in the 70-80% range without asserting an exact 2024 percentile.)

## 6. Upstream Changes

### `utils/srtr_linkage.py`
Edit `SRTRLinkageConfig.SRTR_VARIABLES` in place to include the additional donor characteristics required for KDPI inputs and Table 1 rows that aren't currently passed through. New mappings to add:

```python
'cod_stroke': 'DON_COD_DON_STROKE',
'anti_hcv':   'DON_ANTI_HCV',
'anti_hiv':   'DON_ANTI_HIV',
```

The standardization step in `standardize_srtr_data` should pass these through unchanged (they are already in their final analytic form). Existing entries (`age`, `sex`, `height_cm`, `weight_kg`, `creatinine`, `diabetes`, `hypertension`, `recovery_date`, `cause_of_death`, `dcd_withdraw_date`) are unchanged.

### `record_linkage/01_match_hierarchical.py`
1. Load `donor_disposition.sas7bdat` (already referenced in this file; expand projection to include organ-type, disposition, and donor_id columns).
2. Compute per-donor outcomes:
   - `n_kidneys_transplanted` — count of rows where `DON_TY = 'KI'` and disposition indicates transplanted
   - `n_organs_transplanted` — count of all organ rows with disposition = transplanted
   - `kidney_donated` — boolean: `n_kidneys_transplanted ≥ 1`
3. Persist as `output/intermediate/donor_outcomes.parquet`.
4. Verify the widened `final_srtr_data.parquet` includes the three new columns end-to-end after re-running the linkage pipeline.

Both edits are made in place in the existing files (no parallel loaders or sidecar enrichment scripts).

## 7. Reporting Script (`record_linkage/03_niddk_aim1_tables.py`)

### Steps
1. Load all intermediates listed in §4 ("Inputs"). Read the ADT table directly from `tables_path` (configured in `config/config.json`) for ICU LOS.
2. Filter `encounter_mapping_matched` to `match_confidence ∈ {HIGH, MEDIUM}` AND `is_best == True`. Inner-join to `final_clif_data` on `hospitalization_id`, to `final_srtr_data` on `DONOR_ID`, to `donor_outcomes` on `DONOR_ID`.
3. Compute per-donor KDPI via `utils.kdpi.compute_kdpi`.
4. Compute CLIF-derived variables:
   - **Vasopressors-within-48h flag**: from `wide_df`, group by `hospitalization_id` and look at rows in `[death_dttm − 48h, death_dttm]`. Flag = any of {`med_cont_norepinephrine`, `med_cont_epinephrine`, `med_cont_phenylephrine`, `med_cont_vasopressin`, `med_cont_dopamine`} non-null and > 0.
   - **IMV-within-48h flag**: from `wide_df`, same window as vasopressors. Flag = any row with `resp_device_category == 'IMV'`. Expected ~100% in the M+H cohort by construction (CLIF cohort requires IMV in this window); reported for completeness.
   - **ICU LOS**: from `adt`, filter to `location_category='icu'` and the matched `hospitalization_id`s. Sum `(out_dttm − in_dttm)` per hospitalization, convert to days.
5. Build Table 1 and Table 2 via a small helper (see §8).
6. Build the creatinine trajectory figure.
7. Render `output/final/niddk_aim1_report.html`. Also write CSV versions of the two tables.

### Configuration
The script reads `config/config.json` for `tables_path` and `file_type` (used only for the ADT load); all other inputs are pre-computed parquets. No new configuration keys are introduced.

## 8. Table Builder

A single helper (lives inside `03_niddk_aim1_tables.py` unless it grows large enough to warrant its own utils module) produces a `pandas.DataFrame` with columns `[Variable, Overall, Moderate, High]` from a row-spec list. Spec entries declare:
- Variable label
- Source column
- Summary type: `n_pct`, `median_iqr`, `mean_sd`, `categorical`, `median_range`

Categorical entries expand to multiple output rows (one per non-empty level). Missing values are reported as a `n missing` suffix when nontrivial.

## 9. HTML Report Layout

Single self-contained file. Structure:

```
<h1>NIDDK Aim 1 — UCMC Donor Quality Description</h1>
<p>Site: UCMC | Generated: <date> | N matched (M+H): <count></p>

<h2>Cohort definition</h2>
  Short prose + cohort counts (Total → M+H best matches → strata).

<h2>Table 1 — Donor characteristics</h2>
  HTML table.

<h2>Table 2 — Donation outcomes</h2>
  HTML table.

<h2>Figure — Creatinine trajectory by kidney donation status</h2>
  Base64-embedded PNG.

<h2>Methods</h2>
  - Population definition + linkage method citation
  - KDPI: OPTN 2024 refit, citing references/kdpi_guide.pdf and kdpi_mapping_table.pdf
  - Variable definitions (mirrors Table 1/2 operationalization columns)
  - Missing-data handling

<h2>References</h2>
  OPTN guide + mapping table (file paths in references/).
```

Plain CSS styling (no external dependencies). The HTML is intended to be opened in a browser and printed/PDF-exported for the R01 attachment.

## 10. Validation Plan

1. **Unit test** for `utils/kdpi.py` against the OPTN guide worked example (xβ and KDRI_RAO equality to numerical precision).
2. **Sanity** on production data: compute KDPI for all matched donors at UCMC; confirm distribution median falls in the 30-60% range typical for OPO populations, no values outside [0, 100].
3. **End-to-end smoke test**: run `03_niddk_aim1_tables.py` on UCMC; confirm the report renders, tables contain the expected variable rows, and the figure has at least one line per stratum.
4. **Cohort count check**: M+H count from `encounter_mapping_matched.parquet` matches the row count in Table 1's "N (% of cohort)" row.

## 11. Out of Scope

- HgbA1c reporting (not in CLIF labs at UCMC)
- Imputation of any missing donor variables for KDPI beyond the OPTN-published HTN/DM unknown handling
- Statistical inference across confidence strata (descriptive only)
- Multi-site reporting or cross-site comparison
- Modification of the upstream linkage algorithm itself (we consume its outputs)

## 12. References

- `references/kdpi_guide.pdf` — OPTN, "A Guide to Calculating and Interpreting the Kidney Donor Profile Index (KDPI)", updated April 21, 2025.
- `references/kdpi_mapping_table.pdf` — OPTN, "KDRI to KDPI Mapping Table", April 04, 2025 (2024 reference cohort).
- Rao PS, Schaubel DE, Guidinger MK, et al. *A comprehensive risk quantification score for deceased donor kidneys: The kidney donor risk index.* Transplantation. 2009;88(2):231-236. (Original KDRI publication)
- OPTN Minority Affairs Committee. *Refit Kidney Donor Profile Index without Race and Hepatitis C Virus.* 2024. (Source of the current 8-factor formula)
