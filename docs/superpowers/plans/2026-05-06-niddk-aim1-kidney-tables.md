# NIDDK Aim 1 Kidney Donor Reporting Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two summary tables and a creatinine-trajectory figure characterizing UCMC's CLIF medically-eligible donor decedents with moderate/high SRTR matches, output as a single HTML report for an R01 prelim data attachment.

**Architecture:** Pure-Python KDPI module (`utils/kdpi.py`) implementing the OPTN 2024 8-factor refit formula with embedded mapping table. In-place widening of the existing SRTR projection in `record_linkage/01_match_hierarchical.py` and `utils/srtr_linkage.py`. New orchestrator script `record_linkage/03_niddk_aim1_tables.py` consumes the existing intermediate parquets, computes KDPI plus CLIF-derived variables (vasopressors-/IMV-within-48h, ICU LOS), and renders the HTML report.

**Tech Stack:** Python (pandas, polars, matplotlib), pytest for unit tests, sas7bdat reader (already in `pyproject.toml`).

**Spec:** [docs/superpowers/specs/2026-05-06-niddk-aim1-kidney-tables-design.md](../specs/2026-05-06-niddk-aim1-kidney-tables-design.md)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `utils/kdpi.py` | CREATE | KDPI formula + mapping + orchestrator |
| `tests/__init__.py` | CREATE | empty package marker |
| `tests/test_kdpi.py` | CREATE | unit tests (OPTN worked example + missing-data + mapping boundary) |
| `pyproject.toml` | MODIFY | add `pytest` to dev deps |
| `utils/srtr_linkage.py` | MODIFY | add 3 entries to `SRTR_VARIABLES` (lines ~65-79) |
| `record_linkage/01_match_hierarchical.py` | MODIFY | extend donor_deceased projection (line ~155); add donor_disposition projection + per-donor outcomes parquet |
| `record_linkage/03_niddk_aim1_tables.py` | CREATE | orchestrator: cohort filter, KDPI, CLIF-derived vars, table builder, figure, HTML report |
| `output/intermediate/donor_outcomes.parquet` | output | per-donor kidney/organ outcomes (one row per DONOR_ID) |
| `output/final/niddk_aim1_report.html` | output | self-contained HTML report |
| `output/final/niddk_aim1_table1.csv` | output | Table 1 in CSV |
| `output/final/niddk_aim1_table2.csv` | output | Table 2 in CSV |
| `output/final/niddk_aim1_creatinine_trajectory.png` | output | standalone figure |

---

## Task 1: Add pytest dependency and create tests/ skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest to dev dependencies via uv**

Run: `uv add --dev pytest`

Expected: `pyproject.toml` updated with `[dependency-groups] dev = ["pytest>=8.0.0"]` (or similar) and `uv.lock` regenerated.

- [ ] **Step 2: Create tests/__init__.py and conftest.py**

Create `tests/__init__.py` as an empty file.

Create `tests/conftest.py` so tests can import the project's top-level packages:

```python
# tests/conftest.py
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `utils.*` imports work in tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: Verify pytest runs (no tests yet, just discovery)**

Run: `uv run pytest tests/ -v --collect-only`

Expected: `no tests ran` or `0 tests collected`, exit code 0 or 5 (no tests). No import errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/conftest.py
git commit -m "test: add pytest dev dependency and tests/ skeleton"
```

---

## Task 2: KDPI core formula — xβ and KDRI_RAO (TDD)

**Files:**
- Create: `tests/test_kdpi.py`
- Create: `utils/kdpi.py`

- [ ] **Step 1: Write failing test for compute_kdri_xb and compute_kdri_rao**

Create `tests/test_kdpi.py`:

```python
"""Unit tests for utils/kdpi.py.

Source: references/kdpi_guide.pdf (OPTN, April 21, 2025) and
references/kdpi_mapping_table.pdf (April 04, 2025, 2024 reference cohort).
"""

import math
import pytest

from utils.kdpi import compute_kdri_xb, compute_kdri_rao


# OPTN guide worked example (page 5):
# Age=52, Height=183 cm, Weight=81 kg (NOT <80, so weight term inactive),
# HTN=Yes, DM=No, COD=CVA, Cr=1.7, DCD=Yes
# Expected xβ = 0.53787000000000, KDRI_RAO = 1.71235565748184
OPTN_EXAMPLE = dict(
    age=52, height_cm=183, weight_kg=81,
    hist_htn=True, hist_dm=False, cod_cva=True,
    creatinine=1.7, dcd=True,
)


def test_kdri_xb_matches_optn_worked_example():
    xb = compute_kdri_xb(**OPTN_EXAMPLE)
    assert math.isclose(xb, 0.53787, rel_tol=0, abs_tol=1e-10)


def test_kdri_rao_matches_optn_worked_example():
    rao = compute_kdri_rao(**OPTN_EXAMPLE)
    assert math.isclose(rao, 1.71235565748184, rel_tol=0, abs_tol=1e-10)
```

- [ ] **Step 2: Run test — should fail with ImportError**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: 2 errors (or failures) with `ModuleNotFoundError: No module named 'utils.kdpi'`.

- [ ] **Step 3: Implement utils/kdpi.py with compute_kdri_xb and compute_kdri_rao**

Create `utils/kdpi.py`:

```python
"""KDPI (Kidney Donor Profile Index) computation.

Implements the OPTN 2024 refit formula (8 factors; race and HCV removed
from the prior 10-factor Rao 2009 model). All coefficients, the scaling
factor, and the mapping table are taken verbatim from:

  - references/kdpi_guide.pdf (OPTN, April 21, 2025)
  - references/kdpi_mapping_table.pdf (April 04, 2025, 2024 reference cohort)

Reference: Rao PS, Schaubel DE, Guidinger MK, et al.
A comprehensive risk quantification score for deceased donor kidneys:
The kidney donor risk index. Transplantation. 2009;88(2):231-236.
And: OPTN Minority Affairs Committee. Refit Kidney Donor Profile Index
without Race and Hepatitis C Virus. 2024.
"""

from __future__ import annotations

import math
from typing import Optional

# --------------------------------------------------------------------------
# Constants (from references/kdpi_guide.pdf and kdpi_mapping_table.pdf)
# --------------------------------------------------------------------------

# 2024 reference cohort (mapping table footer)
KDRI_SCALING_FACTOR_2024 = 1.40436817065005

# OPTN-published probabilities used to impute "unknown" HTN/DM (mapping table footer)
HTN_UNKNOWN_PROB = 0.43697057162578
DM_UNKNOWN_PROB = 0.17280542134655

# Creatinine cap (guide page 9)
CR_CAP = 8.0

# Donor input bounds (guide page 10)
AGE_MIN, AGE_MAX = 0, 99
HEIGHT_MAX_CM = 241.3
WEIGHT_MIN_KG, WEIGHT_MAX_KG = 0.454, 294.0


def compute_kdri_xb(
    age: float,
    height_cm: float,
    weight_kg: float,
    hist_htn: bool,
    hist_dm: bool,
    cod_cva: bool,
    creatinine: float,
    dcd: bool,
) -> float:
    """Compute the Xβ component of the KDRI (OPTN 2024 refit formula).

    All inputs must be non-missing; callers are responsible for handling
    missing data (see compute_kdpi for orchestration).

    Creatinine is capped at CR_CAP (8.0 mg/dL) per OPTN guide page 9.
    """
    cr = min(creatinine, CR_CAP)

    xb = 0.0
    # Age (guide Table 1)
    xb += 0.0092 * (age - 40)
    if age < 18:
        xb += 0.0113 * (age - 18)
    if age > 50:
        xb += 0.0067 * (age - 50)
    # Height (per 10 cm above 170)
    xb += -0.0557 * (height_cm - 170) / 10
    # Weight (only when <80 kg, per 5 kg below 80)
    if weight_kg < 80:
        xb += -0.0333 * (weight_kg - 80) / 5
    # Comorbidities and COD
    if hist_htn:
        xb += 0.1106
    if hist_dm:
        xb += 0.2577
    if cod_cva:
        xb += 0.0743
    # Creatinine (linear above 1, additional decreasing slope above 1.5)
    xb += 0.2128 * (cr - 1)
    if cr > 1.5:
        xb += -0.2199 * (cr - 1.5)
    # DCD
    if dcd:
        xb += 0.1966

    return xb


def compute_kdri_rao(
    age: float,
    height_cm: float,
    weight_kg: float,
    hist_htn: bool,
    hist_dm: bool,
    cod_cva: bool,
    creatinine: float,
    dcd: bool,
) -> float:
    """KDRI_RAO = exp(Xβ). Pure formula; no scaling/mapping."""
    return math.exp(compute_kdri_xb(
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        hist_htn=hist_htn,
        hist_dm=hist_dm,
        cod_cva=cod_cva,
        creatinine=creatinine,
        dcd=dcd,
    ))
```

- [ ] **Step 4: Run tests — should pass**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kdpi.py utils/kdpi.py
git commit -m "feat(kdpi): implement KDRI Xb and KDRI_RAO per OPTN 2024 refit"
```

---

## Task 3: KDPI mapping (KDRI_scaled → KDPI percentile) — TDD

**Files:**
- Modify: `tests/test_kdpi.py`
- Modify: `utils/kdpi.py`

- [ ] **Step 1: Add mapping tests**

Append to `tests/test_kdpi.py`:

```python
from utils.kdpi import kdri_scaled_to_kdpi, KDRI_SCALING_FACTOR_2024


def test_kdri_scaled_zero_maps_to_zero():
    assert kdri_scaled_to_kdpi(0.0) == 0


def test_kdri_scaled_at_median_maps_to_50():
    # Mapping table row: 0.98914385687621 < x <= 1.00000000000001 -> 50%
    assert kdri_scaled_to_kdpi(1.0) == 50


def test_kdri_scaled_just_above_median_maps_to_51():
    # 1.00000000000001 < x <= 1.00895758010347 -> 51%
    assert kdri_scaled_to_kdpi(1.005) == 51


def test_kdri_scaled_at_max_maps_to_100():
    assert kdri_scaled_to_kdpi(2.82070964079594) == 100


def test_kdri_scaled_above_max_maps_to_100():
    assert kdri_scaled_to_kdpi(5.0) == 100


def test_kdri_scaled_lower_bound_zero_pct():
    # Mapping table: 0.00000000000000 < x <= 0.43756347366936 -> 0%
    assert kdri_scaled_to_kdpi(0.43756347366936) == 0
    assert kdri_scaled_to_kdpi(0.43756347366937) == 1
```

- [ ] **Step 2: Run tests — should fail with ImportError on kdri_scaled_to_kdpi**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: prior 2 pass; 6 new tests fail with `ImportError: cannot import name 'kdri_scaled_to_kdpi'`.

- [ ] **Step 3: Add the embedded mapping table and lookup function to utils/kdpi.py**

Append to `utils/kdpi.py`:

```python
# --------------------------------------------------------------------------
# KDRI-to-KDPI mapping table (verbatim from references/kdpi_mapping_table.pdf,
# 2024 reference cohort, dated April 04, 2025).
#
# Each tuple is (low_exclusive, high_inclusive, kdpi_pct). Lookup convention:
#   low_exclusive < kdri_scaled <= high_inclusive  ->  kdpi_pct
# The first row uses 0.0 as low_exclusive and includes 0 as the floor value
# (kdri_scaled <= 0.43756347366936 -> 0%).
# --------------------------------------------------------------------------
KDPI_MAPPING: list[tuple[float, float, int]] = [
    (0.00000000000000, 0.43756347366936, 0),
    (0.43756347366936, 0.54140400678998, 1),
    (0.54140400678998, 0.56455385268328, 2),
    (0.56455385268328, 0.58227670794408, 3),
    (0.58227670794408, 0.59655639363358, 4),
    (0.59655639363358, 0.60828393903620, 5),
    (0.60828393903620, 0.62073718385389, 6),
    (0.62073718385389, 0.63214751577493, 7),
    (0.63214751577493, 0.64347410023235, 8),
    (0.64347410023235, 0.65318851478785, 9),
    (0.65318851478785, 0.66302770586452, 10),
    (0.66302770586452, 0.67145552264513, 11),
    (0.67145552264513, 0.68087570963769, 12),
    (0.68087570963769, 0.69029274652971, 13),
    (0.69029274652971, 0.69751483197311, 14),
    (0.69751483197311, 0.70694281453250, 15),
    (0.70694281453250, 0.71473924076668, 16),
    (0.71473924076668, 0.72358482223375, 17),
    (0.72358482223375, 0.73165769729743, 18),
    (0.73165769729743, 0.73996713862377, 19),
    (0.73996713862377, 0.74791682700397, 20),
    (0.74791682700397, 0.75595207245550, 21),
    (0.75595207245550, 0.76380290485577, 22),
    (0.76380290485577, 0.77160424143161, 23),
    (0.77160424143161, 0.78021363869746, 24),
    (0.78021363869746, 0.78851346149113, 25),
    (0.78851346149113, 0.79658510945428, 26),
    (0.79658510945428, 0.80387692407067, 27),
    (0.80387692407067, 0.81072894733318, 28),
    (0.81072894733318, 0.81856104693612, 29),
    (0.81856104693612, 0.82627642950060, 30),
    (0.82627642950060, 0.83318339328986, 31),
    (0.83318339328986, 0.84121542280184, 32),
    (0.84121542280184, 0.84935715740213, 33),
    (0.84935715740213, 0.85645842665560, 34),
    (0.85645842665560, 0.86460674884622, 35),
    (0.86460674884622, 0.87434827297452, 36),
    (0.87434827297452, 0.88372133226563, 37),
    (0.88372133226563, 0.89269482214228, 38),
    (0.89269482214228, 0.90077164376820, 39),
    (0.90077164376820, 0.90931337155045, 40),
    (0.90931337155045, 0.91735632893166, 41),
    (0.91735632893166, 0.92571441314443, 42),
    (0.92571441314443, 0.93474875125877, 43),
    (0.93474875125877, 0.94395526660055, 44),
    (0.94395526660055, 0.95360141288907, 45),
    (0.95360141288907, 0.96147289850279, 46),
    (0.96147289850279, 0.97141500633747, 47),
    (0.97141500633747, 0.97996982262139, 48),
    (0.97996982262139, 0.98914385687621, 49),
    (0.98914385687621, 1.00000000000001, 50),
    (1.00000000000001, 1.00895758010347, 51),
    (1.00895758010347, 1.01958175487502, 52),
    (1.01958175487502, 1.02880558070335, 53),
    (1.02880558070335, 1.03725904354631, 54),
    (1.03725904354631, 1.04748546724369, 55),
    (1.04748546724369, 1.05701035027583, 56),
    (1.05701035027583, 1.06683305599198, 57),
    (1.06683305599198, 1.07572892220660, 58),
    (1.07572892220660, 1.08566912172970, 59),
    (1.08566912172970, 1.09533088857168, 60),
    (1.09533088857168, 1.10529967684360, 61),
    (1.10529967684360, 1.11564364557302, 62),
    (1.11564364557302, 1.12583220387625, 63),
    (1.12583220387625, 1.13596203399892, 64),
    (1.13596203399892, 1.14607664781359, 65),
    (1.14607664781359, 1.15561480258537, 66),
    (1.15561480258537, 1.16595057551619, 67),
    (1.16595057551619, 1.17597158234812, 68),
    (1.17597158234812, 1.18797063065605, 69),
    (1.18797063065605, 1.19958840121024, 70),
    (1.19958840121024, 1.21092254012845, 71),
    (1.21092254012845, 1.22137860162032, 72),
    (1.22137860162032, 1.23397479393383, 73),
    (1.23397479393383, 1.24670304020145, 74),
    (1.24670304020145, 1.25914195259812, 75),
    (1.25914195259812, 1.27152186119531, 76),
    (1.27152186119531, 1.28446270045611, 77),
    (1.28446270045611, 1.29752226906290, 78),
    (1.29752226906290, 1.31372090832136, 79),
    (1.31372090832136, 1.32913598921872, 80),
    (1.32913598921872, 1.34428799207615, 81),
    (1.34428799207615, 1.36002692720158, 82),
    (1.36002692720158, 1.37645657674097, 83),
    (1.37645657674097, 1.39274343883440, 84),
    (1.39274343883440, 1.41088970093023, 85),
    (1.41088970093023, 1.42882224348119, 86),
    (1.42882224348119, 1.44689676213656, 87),
    (1.44689676213656, 1.47000896434346, 88),
    (1.47000896434346, 1.49117067437256, 89),
    (1.49117067437256, 1.51572594977824, 90),
    (1.51572594977824, 1.54164880913418, 91),
    (1.54164880913418, 1.56911614985970, 92),
    (1.56911614985970, 1.60239998973729, 93),
    (1.60239998973729, 1.63668114440017, 94),
    (1.63668114440017, 1.68076407902584, 95),
    (1.68076407902584, 1.72368854740753, 96),
    (1.72368854740753, 1.77998421976632, 97),
    (1.77998421976632, 1.86168222116483, 98),
    (1.86168222116483, 1.98677654539587, 99),
    (1.98677654539587, 2.82070964079594, 100),
    (2.82070964079594, 999_999_999.0, 100),
]


def kdri_scaled_to_kdpi(kdri_scaled: float) -> int:
    """Map a KDRI_scaled value to its KDPI percentile (0-100, integer).

    Convention from references/kdpi_mapping_table.pdf: low_exclusive < x <=
    high_inclusive. The lowest row's lower bound (0.0) is treated as inclusive
    (any value <= 0.43756347366936 returns 0%); values above the maximum
    observed (2.82070964079594) return 100%.
    """
    if kdri_scaled <= KDPI_MAPPING[0][1]:
        return 0
    for low, high, pct in KDPI_MAPPING:
        if low < kdri_scaled <= high:
            return pct
    # Fallback (should be unreachable given the trailing overflow row):
    return 100
```

- [ ] **Step 4: Run tests — all should pass**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kdpi.py utils/kdpi.py
git commit -m "feat(kdpi): embed OPTN 2024 mapping table and lookup function"
```

---

## Task 4: KDPI orchestrator with missing-data handling — TDD

**Files:**
- Modify: `tests/test_kdpi.py`
- Modify: `utils/kdpi.py`

- [ ] **Step 1: Add orchestrator tests**

Append to `tests/test_kdpi.py`:

```python
import pandas as pd
import numpy as np

from utils.kdpi import compute_kdpi, HTN_UNKNOWN_PROB, DM_UNKNOWN_PROB


def test_compute_kdpi_returns_int_for_complete_optn_example():
    # 2024 scaling factor differs from 2023 (guide example used 1.30900...);
    # we don't assert an exact percentile but require a sane integer in [0,100].
    row = dict(
        DON_AGE=52, DON_HGT_CM=183, DON_WGT_KG=81,
        DON_HIST_HYPERTEN='Y', DON_HIST_DIAB='N',
        DON_COD_DON_STROKE=1, DON_FINAL_SERUM_CREAT=1.7,
        DON_NON_HR_BEAT='Y',  # Y -> non-heart-beating -> DCD
    )
    kdpi = compute_kdpi(row)
    assert isinstance(kdpi, int)
    assert 0 <= kdpi <= 100
    assert kdpi >= 70  # Sanity: example from guide was 79% under 2023 scaling


def test_compute_kdpi_returns_none_when_age_missing():
    row = dict(
        DON_AGE=np.nan, DON_HGT_CM=183, DON_WGT_KG=81,
        DON_HIST_HYPERTEN='Y', DON_HIST_DIAB='N',
        DON_COD_DON_STROKE=1, DON_FINAL_SERUM_CREAT=1.7,
        DON_NON_HR_BEAT='Y',
    )
    assert compute_kdpi(row) is None


def test_compute_kdpi_handles_unknown_htn():
    """HTN='U' should impute via probabilistic Xβ contribution per OPTN."""
    row_unknown = dict(
        DON_AGE=40, DON_HGT_CM=170, DON_WGT_KG=80,
        DON_HIST_HYPERTEN='U', DON_HIST_DIAB='N',
        DON_COD_DON_STROKE=0, DON_FINAL_SERUM_CREAT=1.0,
        DON_NON_HR_BEAT='N',
    )
    row_no = {**row_unknown, 'DON_HIST_HYPERTEN': 'N'}
    row_yes = {**row_unknown, 'DON_HIST_HYPERTEN': 'Y'}
    kdpi_u = compute_kdpi(row_unknown)
    kdpi_n = compute_kdpi(row_no)
    kdpi_y = compute_kdpi(row_yes)
    assert kdpi_n <= kdpi_u <= kdpi_y


def test_compute_kdpi_caps_creatinine_at_8():
    row_8 = dict(
        DON_AGE=40, DON_HGT_CM=170, DON_WGT_KG=80,
        DON_HIST_HYPERTEN='N', DON_HIST_DIAB='N',
        DON_COD_DON_STROKE=0, DON_FINAL_SERUM_CREAT=8.0,
        DON_NON_HR_BEAT='N',
    )
    row_15 = {**row_8, 'DON_FINAL_SERUM_CREAT': 15.0}
    assert compute_kdpi(row_8) == compute_kdpi(row_15)


def test_compute_kdpi_accepts_pandas_series():
    s = pd.Series(dict(
        DON_AGE=52, DON_HGT_CM=183, DON_WGT_KG=81,
        DON_HIST_HYPERTEN='Y', DON_HIST_DIAB='N',
        DON_COD_DON_STROKE=1, DON_FINAL_SERUM_CREAT=1.7,
        DON_NON_HR_BEAT='Y',
    ))
    kdpi = compute_kdpi(s)
    assert isinstance(kdpi, int)
```

- [ ] **Step 2: Run — orchestrator tests should fail**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: 8 prior pass; 5 new fail with ImportError on `compute_kdpi`.

- [ ] **Step 3: Implement compute_kdpi orchestrator**

Append to `utils/kdpi.py`:

```python
# --------------------------------------------------------------------------
# Orchestrator: SRTR row -> KDPI
# --------------------------------------------------------------------------

def _is_missing(v) -> bool:
    """True for None, NaN, empty string, or pandas-style NA."""
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, str) and v.strip() == '':
        return True
    return False


def _parse_yn(v, *, allow_unknown: bool = False) -> Optional[bool]:
    """Parse SRTR Y/N/U char fields. Returns True/False, or None if unknown
    (only when allow_unknown=False). When allow_unknown=True, returns the
    string 'U' for unknown values so the caller can apply OPTN imputation.
    """
    if _is_missing(v):
        return None
    s = str(v).strip().upper()
    if s in ('Y', '1', 'TRUE'):
        return True
    if s in ('N', '0', 'FALSE'):
        return False
    if s in ('U', 'UNKNOWN'):
        return 'U' if allow_unknown else None
    return None


def compute_kdpi(donor_row) -> Optional[int]:
    """Compute KDPI for a single donor row from SRTR donor_deceased columns.

    Required SRTR columns (case-sensitive):
      DON_AGE, DON_HGT_CM, DON_WGT_KG, DON_HIST_HYPERTEN, DON_HIST_DIAB,
      DON_COD_DON_STROKE, DON_FINAL_SERUM_CREAT, DON_NON_HR_BEAT

    Returns None if any required input is missing (no imputation beyond the
    OPTN-published HTN/DM "unknown" handling).
    """
    age = donor_row.get('DON_AGE') if isinstance(donor_row, dict) else donor_row['DON_AGE']
    height_cm = donor_row.get('DON_HGT_CM') if isinstance(donor_row, dict) else donor_row['DON_HGT_CM']
    weight_kg = donor_row.get('DON_WGT_KG') if isinstance(donor_row, dict) else donor_row['DON_WGT_KG']
    htn_raw = donor_row.get('DON_HIST_HYPERTEN') if isinstance(donor_row, dict) else donor_row['DON_HIST_HYPERTEN']
    dm_raw = donor_row.get('DON_HIST_DIAB') if isinstance(donor_row, dict) else donor_row['DON_HIST_DIAB']
    cod_stroke = donor_row.get('DON_COD_DON_STROKE') if isinstance(donor_row, dict) else donor_row['DON_COD_DON_STROKE']
    cr = donor_row.get('DON_FINAL_SERUM_CREAT') if isinstance(donor_row, dict) else donor_row['DON_FINAL_SERUM_CREAT']
    nhb = donor_row.get('DON_NON_HR_BEAT') if isinstance(donor_row, dict) else donor_row['DON_NON_HR_BEAT']

    # Required numeric fields
    for v in (age, height_cm, weight_kg, cr):
        if _is_missing(v):
            return None

    # COD stroke flag (1/0 numeric)
    if _is_missing(cod_stroke):
        return None
    cod_cva = bool(int(cod_stroke))

    # DCD: DON_NON_HR_BEAT 'Y' = non-heart-beating donor = DCD
    dcd_parsed = _parse_yn(nhb, allow_unknown=False)
    if dcd_parsed is None:
        return None
    dcd = bool(dcd_parsed)

    # HTN/DM: allow OPTN-published unknown imputation
    htn_parsed = _parse_yn(htn_raw, allow_unknown=True)
    dm_parsed = _parse_yn(dm_raw, allow_unknown=True)
    if htn_parsed is None or dm_parsed is None:
        return None

    # Validate input ranges (out-of-range -> None to be safe)
    if not (AGE_MIN <= age <= AGE_MAX):
        return None
    if height_cm > HEIGHT_MAX_CM or height_cm <= 0:
        return None
    if not (WEIGHT_MIN_KG <= weight_kg <= WEIGHT_MAX_KG):
        return None

    # Compute Xβ; if HTN or DM is unknown, substitute the probabilistic component.
    if htn_parsed == 'U' or dm_parsed == 'U':
        cr_capped = min(float(cr), CR_CAP)
        xb = 0.0
        xb += 0.0092 * (age - 40)
        if age < 18:
            xb += 0.0113 * (age - 18)
        if age > 50:
            xb += 0.0067 * (age - 50)
        xb += -0.0557 * (height_cm - 170) / 10
        if weight_kg < 80:
            xb += -0.0333 * (weight_kg - 80) / 5
        if htn_parsed == 'U':
            xb += 0.1106 * HTN_UNKNOWN_PROB
        elif htn_parsed:
            xb += 0.1106
        if dm_parsed == 'U':
            xb += 0.2577 * DM_UNKNOWN_PROB
        elif dm_parsed:
            xb += 0.2577
        if cod_cva:
            xb += 0.0743
        xb += 0.2128 * (cr_capped - 1)
        if cr_capped > 1.5:
            xb += -0.2199 * (cr_capped - 1.5)
        if dcd:
            xb += 0.1966
        kdri_rao = math.exp(xb)
    else:
        kdri_rao = compute_kdri_rao(
            age=age, height_cm=height_cm, weight_kg=weight_kg,
            hist_htn=bool(htn_parsed), hist_dm=bool(dm_parsed),
            cod_cva=cod_cva, creatinine=float(cr), dcd=dcd,
        )

    kdri_scaled = kdri_rao / KDRI_SCALING_FACTOR_2024
    return kdri_scaled_to_kdpi(kdri_scaled)
```

- [ ] **Step 4: Run all KDPI tests — all 13 should pass**

Run: `uv run pytest tests/test_kdpi.py -v`

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kdpi.py utils/kdpi.py
git commit -m "feat(kdpi): add compute_kdpi orchestrator with missing-data handling"
```

---

## Task 5: Widen SRTR projection in 01_match_hierarchical.py + persist donor_outcomes.parquet

**Files:**
- Modify: `utils/srtr_linkage.py` (lines ~65-79)
- Modify: `record_linkage/01_match_hierarchical.py` (donor_deceased projection ~line 155, plus add donor_disposition processing block)

- [ ] **Step 1: Inspect actual donor_disposition columns**

Run:

```bash
/Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier/.venv/bin/python -c "
from sas7bdat import SAS7BDAT
src='/Users/kavenchhikara/Library/CloudStorage/Box-Box/SAF Q2 2025/pubsaf2506/donor_disposition.sas7bdat'
r = SAS7BDAT(src, skip_header=False)
cols = [c.name.decode() if isinstance(c.name, bytes) else c.name for c in r.columns]
r.close()
print('Total cols:', len(cols))
relevant = [c for c in cols if any(k in c.upper() for k in ['DONOR_ID','PX_TY','PX_STAT','DISPO','ORG','TX_DT','RECOV'])]
for c in relevant: print(' ', c)
"
```

Expected output: a small set of columns including `DONOR_ID`, an organ-type column (likely `PX_TY`), and a disposition/status column (likely `PX_STAT` or similar). Note the exact names — they're used in Step 4.

- [ ] **Step 2: Widen the donor_deceased projection in record_linkage/01_match_hierarchical.py**

Edit `record_linkage/01_match_hierarchical.py` around line 155-203. Add three columns to the existing column list (in the `Cause of death` and `DCD variables` sections, or grouped wherever they fit naturally):

```python
        # Cause of death
        "DON_CAD_DON_COD",
        "DON_COD_DON_STROKE",   # NEW: 1/0 cause of death = stroke (for KDPI + Hx CVA row)

        ...

        # Serology (NEW: needed for HCV/HIV rows in NIDDK Aim 1 Table 1)
        "DON_ANTI_HCV",
        "DON_ANTI_HIV",

        # Final serum creatinine (NEW: needed for KDPI inputs)
        "DON_FINAL_SERUM_CREAT",
```

Place each `# NEW:` line in the appropriate section of the existing projection list at line 155.

- [ ] **Step 3: Add SRTR_VARIABLES entries in utils/srtr_linkage.py**

Edit `utils/srtr_linkage.py` lines 65-79. Add three entries to the `SRTR_VARIABLES` dict:

```python
    SRTR_VARIABLES = {
        'donor_id': 'DONOR_ID',
        'age': 'DON_AGE',
        'sex': 'DON_GENDER',
        'race': 'DON_RACE_SRTR',
        'ethnicity': 'DON_ETHNICITY_SRTR',
        'height_cm': 'DON_HGT_CM',
        'weight_kg': 'DON_WGT_KG',
        'creatinine': 'DON_CREAT',
        'recovery_date': 'DON_RECOV_DT',
        'cause_of_death': 'DON_CAD_DON_COD',
        'cod_stroke': 'DON_COD_DON_STROKE',   # NEW: KDPI + Hx CVA
        'diabetes': 'DON_HIST_DIAB',
        'hypertension': 'DON_HIST_HYPERTEN',
        'anti_hcv': 'DON_ANTI_HCV',           # NEW: HCV row
        'anti_hiv': 'DON_ANTI_HIV',           # NEW: HIV row
        'dcd_withdraw_date': 'DON_DCD_SUPPORT_WITHDRAW_DT'
    }
```

- [ ] **Step 4: Add donor_disposition outcomes computation in 01_match_hierarchical.py**

After the existing `donor_disposition = decode_bytes_in_object(donor_disposition)` line (around line 130), add a per-donor outcomes computation block. Use the column names found in Step 1 — the names below are the SRTR convention; verify them before pasting:

```python
# ============================================================
# Per-donor donation outcomes (for NIDDK Aim 1 reporting)
# ============================================================
# Columns assumed to exist in donor_disposition (verified per SRTR SAF):
#   DONOR_ID  - donor identifier
#   PX_TY     - organ type ('KI','LI','HR','LU','PA','IN', ...)
#   PX_STAT   - disposition status (numeric/char codes; transplanted = code 5
#               in standard SRTR disposition coding, OR a string 'TX'/'TRANS'
#               depending on file format).
# Verify the exact disposition value used for "transplanted" in this SAF before
# proceeding. The compute below assumes a transplanted-row indicator function
# `_is_transplanted(stat_value) -> bool` defined inline.

def _is_transplanted(stat_value) -> bool:
    """Return True if the disposition row indicates the organ was transplanted.

    SRTR donor_disposition disposition codes (PX_STAT):
      5 = Transplanted
    Adjust this function if the SAF uses different codes/strings.
    """
    if pd.isna(stat_value):
        return False
    try:
        return int(stat_value) == 5
    except (TypeError, ValueError):
        return str(stat_value).strip().upper() in ('5', 'TX', 'TRANS', 'TRANSPLANTED')

donor_disposition['_transplanted'] = donor_disposition['PX_STAT'].apply(_is_transplanted)

# Per-donor outcomes
_grp = donor_disposition.groupby('DONOR_ID')
donor_outcomes = pd.DataFrame({
    'n_organs_transplanted': _grp['_transplanted'].sum().astype(int),
    'n_kidneys_transplanted': donor_disposition[donor_disposition['PX_TY'] == 'KI']
        .groupby('DONOR_ID')['_transplanted'].sum().astype(int),
}).fillna(0).astype(int)
donor_outcomes['kidney_donated'] = donor_outcomes['n_kidneys_transplanted'] >= 1
donor_outcomes = donor_outcomes.reset_index()

donor_outcomes_path = os.path.join(OUTPUT_INTERMEDIATE_DIR, 'donor_outcomes.parquet')
donor_outcomes.to_parquet(donor_outcomes_path, index=False)
print(f"Saved donor_outcomes ({len(donor_outcomes)} donors) to {donor_outcomes_path}")
```

Note the `import os` is already present in the file (line 206 in the current version). Confirm this; if not, add `import os` near the top.

- [ ] **Step 5: Re-run the linkage script to regenerate widened intermediates**

Run: `uv run python record_linkage/01_match_hierarchical.py`

Expected: existing matching pipeline runs as before; produces `output/intermediate/donor_outcomes.parquet` and the widened `final_srtr_data.parquet`. The script may take several minutes due to SRTR SAS file reads.

- [ ] **Step 6: Verify the new fields exist in the regenerated parquets**

Run:

```bash
/Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier/.venv/bin/python -c "
import pandas as pd
srtr = pd.read_parquet('output/intermediate/final_srtr_data.parquet')
out = pd.read_parquet('output/intermediate/donor_outcomes.parquet')
print('SRTR cols include new ones:', all(c in srtr.columns for c in ['DON_COD_DON_STROKE','DON_ANTI_HCV','DON_ANTI_HIV','DON_FINAL_SERUM_CREAT']))
print('Outcomes cols:', list(out.columns))
print('Outcomes head:')
print(out.head())
print('Kidney donation rate:', out['kidney_donated'].mean())
"
```

Expected: `True`; outcomes columns are `DONOR_ID, n_organs_transplanted, n_kidneys_transplanted, kidney_donated`; kidney donation rate is plausible (typically 70-90% in donor cohorts).

- [ ] **Step 7: Commit**

```bash
git add utils/srtr_linkage.py record_linkage/01_match_hierarchical.py
git commit -m "feat(srtr): widen SRTR projection and persist per-donor outcomes"
```

---

## Task 6: Reporting script — load + cohort filter + KDPI + CLIF-derived variables

**Files:**
- Create: `record_linkage/03_niddk_aim1_tables.py`

- [ ] **Step 1: Create the script with load + cohort filter only (skeleton)**

Create `record_linkage/03_niddk_aim1_tables.py`:

```python
#!/usr/bin/env python
"""NIDDK Aim 1 — donor characteristics and donation outcomes report.

Population: CLIF medically-eligible donor decedents at the configured site
that are linked to SRTR with HIGH or MEDIUM match confidence (best match
per donor). Produces two summary tables and a creatinine-trajectory figure.

See docs/superpowers/specs/2026-05-06-niddk-aim1-kidney-tables-design.md.

Run:
    uv run python record_linkage/03_niddk_aim1_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable (mirrors other scripts in this repo).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import config  # noqa: E402
from utils.io import read_data   # noqa: E402
from utils.kdpi import compute_kdpi  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / 'output'
INTERMEDIATE_DIR = OUTPUT_DIR / 'intermediate'
FINAL_DIR = OUTPUT_DIR / 'final'
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> dict[str, pd.DataFrame]:
    """Load all required intermediate parquets and the ADT table."""
    print('Loading intermediates...')
    mapping = pd.read_parquet(INTERMEDIATE_DIR / 'encounter_mapping_matched.parquet')
    clif = pd.read_parquet(INTERMEDIATE_DIR / 'final_clif_data.parquet')
    srtr = pd.read_parquet(INTERMEDIATE_DIR / 'final_srtr_data.parquet')
    wide = pd.read_parquet(INTERMEDIATE_DIR / 'wide_df.parquet')
    outcomes = pd.read_parquet(INTERMEDIATE_DIR / 'donor_outcomes.parquet')

    tables_path = config['tables_path']
    file_type = config['file_type']
    adt = read_data(f"{tables_path}/clif_adt.{file_type}", file_type).to_pandas()

    print(f"  encounter_mapping_matched: {len(mapping):,}")
    print(f"  final_clif_data:           {len(clif):,}")
    print(f"  final_srtr_data:           {len(srtr):,}")
    print(f"  wide_df:                   {len(wide):,}")
    print(f"  donor_outcomes:            {len(outcomes):,}")
    print(f"  clif_adt:                  {len(adt):,}")
    return dict(mapping=mapping, clif=clif, srtr=srtr, wide=wide, outcomes=outcomes, adt=adt)


def filter_cohort(mapping: pd.DataFrame) -> pd.DataFrame:
    """M+H best matches only."""
    cohort = mapping[
        mapping['match_confidence'].isin(['HIGH', 'MEDIUM'])
        & (mapping['is_best'] == True)
    ].copy()
    cohort['confidence_stratum'] = cohort['match_confidence'].map(
        {'HIGH': 'High', 'MEDIUM': 'Moderate'}
    )
    print(
        f"Cohort: {len(cohort)} M+H best matches "
        f"({(cohort['confidence_stratum'] == 'Moderate').sum()} Moderate, "
        f"{(cohort['confidence_stratum'] == 'High').sum()} High)"
    )
    return cohort


def main() -> None:
    inputs = load_inputs()
    cohort = filter_cohort(inputs['mapping'])
    print(f"DEBUG: cohort head:\n{cohort.head()}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the skeleton — should print loaded counts and cohort head**

Run: `uv run python record_linkage/03_niddk_aim1_tables.py`

Expected: prints input shapes, cohort size (~270 of 291), and a head() showing `match_confidence`, `confidence_stratum`, `DONOR_ID`, `hospitalization_id`, etc.

- [ ] **Step 3: Add the joined cohort dataframe and per-donor KDPI**

Add to `record_linkage/03_niddk_aim1_tables.py`, between `filter_cohort` and `main`:

```python
def build_donor_table(cohort: pd.DataFrame, clif: pd.DataFrame, srtr: pd.DataFrame,
                     outcomes: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the cohort to CLIF, SRTR, and donor_outcomes; compute KDPI."""
    df = (
        cohort
        .merge(clif, on=['hospitalization_id', 'patient_id'], how='inner', suffixes=('', '_clif'))
        .merge(srtr, on='DONOR_ID', how='inner', suffixes=('', '_srtr'))
        .merge(outcomes, on='DONOR_ID', how='left')
    )
    print(f"After joins: {len(df)} rows")

    df['KDPI'] = df.apply(compute_kdpi, axis=1)
    n_kdpi = df['KDPI'].notna().sum()
    print(f"KDPI computed for {n_kdpi}/{len(df)} donors "
          f"(missing inputs for {len(df) - n_kdpi})")
    if n_kdpi:
        print(f"KDPI distribution: median={df['KDPI'].median():.1f}, "
              f"mean={df['KDPI'].mean():.1f}, "
              f"min={df['KDPI'].min()}, max={df['KDPI'].max()}")
    return df
```

Update `main`:

```python
def main() -> None:
    inputs = load_inputs()
    cohort = filter_cohort(inputs['mapping'])
    df = build_donor_table(cohort, inputs['clif'], inputs['srtr'], inputs['outcomes'])
    print(f"DEBUG: df columns ({len(df.columns)}): {list(df.columns)[:30]}...")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run — verify KDPI is computed sanely**

Run: `uv run python record_linkage/03_niddk_aim1_tables.py`

Expected: prints "KDPI computed for X/Y donors", with a distribution where median falls roughly in the 30-70 range.

- [ ] **Step 5: Add CLIF-derived variables: vasopressors-/IMV-within-48h, ICU LOS**

Add to `record_linkage/03_niddk_aim1_tables.py` (between `build_donor_table` and `main`):

```python
VASOPRESSOR_COLS = [
    'med_cont_norepinephrine',
    'med_cont_epinephrine',
    'med_cont_phenylephrine',
    'med_cont_vasopressin',
    'med_cont_dopamine',
]


def add_clif_derived(df: pd.DataFrame, wide: pd.DataFrame, adt: pd.DataFrame) -> pd.DataFrame:
    """Add vasopressors-within-48h-death, IMV-within-48h-death, and ICU LOS columns."""
    df = df.copy()

    # Death timestamp per hospitalization (use final_death_dttm from final_clif_data)
    death_dttm = df.set_index('hospitalization_id')['final_death_dttm'].to_dict()

    # --- Vasopressors and IMV within [death-48h, death] -------------------
    wide_keep = wide[wide['hospitalization_id'].isin(df['hospitalization_id'])].copy()
    wide_keep['event_dttm'] = pd.to_datetime(wide_keep['event_dttm'])
    wide_keep['_death'] = wide_keep['hospitalization_id'].map(death_dttm)
    wide_keep['_death'] = pd.to_datetime(wide_keep['_death'])
    in_window = (
        (wide_keep['event_dttm'] >= wide_keep['_death'] - pd.Timedelta(hours=48))
        & (wide_keep['event_dttm'] <= wide_keep['_death'])
    )
    wide_window = wide_keep[in_window]

    # Vasopressor flag: any of the 5 columns non-null and >0 in window
    vaso_present = wide_window[VASOPRESSOR_COLS].notna() & (wide_window[VASOPRESSOR_COLS].fillna(0) > 0)
    vaso_per_hosp = (
        vaso_present.any(axis=1)
        .groupby(wide_window['hospitalization_id']).any()
        .rename('vasopressors_within_48h')
    )

    # IMV flag: any row in window with resp_device_category == 'IMV'
    if 'resp_device_category' in wide_window.columns:
        imv_per_hosp = (
            (wide_window['resp_device_category'].astype(str).str.upper() == 'IMV')
            .groupby(wide_window['hospitalization_id']).any()
            .rename('imv_within_48h')
        )
    else:
        imv_per_hosp = pd.Series(dtype=bool, name='imv_within_48h')

    df = df.merge(vaso_per_hosp, left_on='hospitalization_id', right_index=True, how='left')
    df = df.merge(imv_per_hosp, left_on='hospitalization_id', right_index=True, how='left')
    df['vasopressors_within_48h'] = df['vasopressors_within_48h'].fillna(False).astype(bool)
    df['imv_within_48h'] = df['imv_within_48h'].fillna(False).astype(bool)

    # --- ICU LOS in days --------------------------------------------------
    adt_icu = adt[
        (adt['location_category'].astype(str).str.lower() == 'icu')
        & (adt['hospitalization_id'].isin(df['hospitalization_id']))
    ].copy()
    adt_icu['in_dttm'] = pd.to_datetime(adt_icu['in_dttm'])
    adt_icu['out_dttm'] = pd.to_datetime(adt_icu['out_dttm'])
    adt_icu['_dur_days'] = (adt_icu['out_dttm'] - adt_icu['in_dttm']).dt.total_seconds() / 86400.0
    icu_los = adt_icu.groupby('hospitalization_id')['_dur_days'].sum().rename('icu_los_days')
    df = df.merge(icu_los, left_on='hospitalization_id', right_index=True, how='left')
    df['icu_los_days'] = df['icu_los_days'].fillna(0.0)

    print(
        f"Vasopressors w/in 48h: {df['vasopressors_within_48h'].sum()}/{len(df)} ({df['vasopressors_within_48h'].mean()*100:.1f}%)"
    )
    print(
        f"IMV w/in 48h:           {df['imv_within_48h'].sum()}/{len(df)} ({df['imv_within_48h'].mean()*100:.1f}%)"
    )
    print(
        f"ICU LOS (days): median={df['icu_los_days'].median():.2f}, "
        f"max={df['icu_los_days'].max():.2f}"
    )
    return df
```

Update `main` to call it:

```python
def main() -> None:
    inputs = load_inputs()
    cohort = filter_cohort(inputs['mapping'])
    df = build_donor_table(cohort, inputs['clif'], inputs['srtr'], inputs['outcomes'])
    df = add_clif_derived(df, inputs['wide'], inputs['adt'])
    print(f"DEBUG: final df shape = {df.shape}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Run — verify CLIF-derived variables compute sanely**

Run: `uv run python record_linkage/03_niddk_aim1_tables.py`

Expected:
- Vasopressors w/in 48h: typically 50-90% of cohort
- IMV w/in 48h: ~100% (by construction; flagged in spec)
- ICU LOS median: typically 1-10 days

- [ ] **Step 7: Commit**

```bash
git add record_linkage/03_niddk_aim1_tables.py
git commit -m "feat(niddk): add reporting script load + cohort + KDPI + CLIF-derived"
```

---

## Task 7: Reporting script — Tables 1 & 2, figure, HTML report

**Files:**
- Modify: `record_linkage/03_niddk_aim1_tables.py`

- [ ] **Step 1: Add the table builder + Table 1 + Table 2**

Add to `record_linkage/03_niddk_aim1_tables.py` after `add_clif_derived`:

```python
import io
import base64
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _fmt_n_pct(n: int, denom: int) -> str:
    if denom == 0:
        return '—'
    return f"{n} ({n/denom*100:.1f}%)"


def _fmt_median_iqr(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return '—'
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"


def _fmt_median_range(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return '—'
    return f"{s.median():.0f} ({int(s.min())}–{int(s.max())})"


def _fmt_mean_sd(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return '—'
    return f"{s.mean():.1f} ({s.std():.1f})"


def _binary_yes_n_pct(series: pd.Series, true_values=(True, 'Y', 'P', 1, '1')) -> tuple[int, int]:
    """Return (n_yes, n_non_missing). Treats values in true_values as positive."""
    s = series.dropna()
    n_yes = s.isin(true_values).sum()
    return int(n_yes), int(len(s))


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """Donor characteristics — Variable | Overall | Moderate | High."""
    strata = {
        'Overall': df,
        'Moderate': df[df['confidence_stratum'] == 'Moderate'],
        'High': df[df['confidence_stratum'] == 'High'],
    }
    rows: list[dict] = []

    # N row
    rows.append({'Variable': 'N (% of cohort)',
                 **{k: _fmt_n_pct(len(v), len(df)) for k, v in strata.items()}})

    def cont(label: str, col: str, fmt=_fmt_median_iqr):
        rows.append({'Variable': label, **{k: fmt(v[col]) for k, v in strata.items()}})

    def binary(label: str, col: str, true_values=(True, 'Y', 'P', 1, '1')):
        cells = {}
        for k, v in strata.items():
            n_yes, n_nm = _binary_yes_n_pct(v[col], true_values)
            cells[k] = _fmt_n_pct(n_yes, n_nm)
        rows.append({'Variable': label, **cells})

    cont('Age (years), median (IQR)', 'age_at_admission')
    binary('Sex: Male, n (%)', 'sex_category', true_values=('Male', 'M', 'male'))
    cont('Height (cm), median (IQR)', 'last_height_cm')
    cont('Weight (kg), median (IQR)', 'last_weight_kg')
    cont('BMI (kg/m²), median (IQR)', 'bmi')

    # Mechanism of death — categorical (multiple rows)
    rows.append({'Variable': 'Mechanism of death (SRTR)',
                 **{k: '' for k in strata}})
    mech_levels = sorted(df['DON_DEATH_MECH'].dropna().astype(str).unique())
    for lvl in mech_levels:
        cells = {}
        for k, v in strata.items():
            n = (v['DON_DEATH_MECH'].astype(str) == lvl).sum()
            cells[k] = _fmt_n_pct(n, len(v))
        rows.append({'Variable': f'   {lvl}', **cells})

    binary('Hx CVA (cause of death = stroke), n (%)', 'DON_COD_DON_STROKE',
           true_values=(1, '1', 1.0))
    binary('HTN (SRTR), n (%)', 'DON_HIST_HYPERTEN')
    binary('DM (SRTR), n (%)', 'DON_HIST_DIAB')
    cont('KDPI, mean (SD)', 'KDPI', fmt=_fmt_mean_sd)
    binary('HCV positive (SRTR), n (%)', 'DON_ANTI_HCV', true_values=('P',))
    binary('HIV positive (SRTR), n (%)', 'DON_ANTI_HIV', true_values=('P',))
    cont('Terminal creatinine (CLIF), mg/dL median (IQR)', 'creatinine_value')
    cont('Terminal creatinine (SRTR), mg/dL median (IQR)', 'DON_FINAL_SERUM_CREAT')
    binary('Vasopressors within 48h death, n (%)', 'vasopressors_within_48h')
    binary('IMV within 48h death, n (%)', 'imv_within_48h')
    cont('ICU LOS (days), median (IQR)', 'icu_los_days')

    return pd.DataFrame(rows)


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """Donation outcomes — Variable | Overall | Moderate | High."""
    strata = {
        'Overall': df,
        'Moderate': df[df['confidence_stratum'] == 'Moderate'],
        'High': df[df['confidence_stratum'] == 'High'],
    }
    rows: list[dict] = []

    rows.append({'Variable': 'N (% of cohort)',
                 **{k: _fmt_n_pct(len(v), len(df)) for k, v in strata.items()}})

    # DBD: DON_NON_HR_BEAT == 'N' -> not non-heart-beating -> DBD
    cells = {}
    for k, v in strata.items():
        s = v['DON_NON_HR_BEAT'].dropna()
        n_dbd = (s.astype(str).str.upper() == 'N').sum()
        cells[k] = _fmt_n_pct(int(n_dbd), len(s))
    rows.append({'Variable': 'DBD, n (%)', **cells})

    # Kidney donation
    cells = {}
    for k, v in strata.items():
        s = v['kidney_donated'].dropna()
        cells[k] = _fmt_n_pct(int(s.sum()), len(s))
    rows.append({'Variable': 'Kidney donation, n (%)', **cells})

    # # organs per donor: median (range)
    rows.append({'Variable': 'Number of organs per donor, median (range)',
                 **{k: _fmt_median_range(v['n_organs_transplanted']) for k, v in strata.items()}})

    return pd.DataFrame(rows)
```

- [ ] **Step 2: Add the creatinine-trajectory figure builder**

Append:

```python
def build_creatinine_figure(df: pd.DataFrame, wide: pd.DataFrame) -> str:
    """Render creatinine trajectory by kidney_donated. Returns base64 PNG."""
    keep_ids = df['hospitalization_id'].unique()
    cr = wide[wide['hospitalization_id'].isin(keep_ids)][
        ['hospitalization_id', 'event_dttm', 'lab_creatinine']
    ].dropna(subset=['lab_creatinine']).copy()
    cr['event_dttm'] = pd.to_datetime(cr['event_dttm'])

    death_map = df.set_index('hospitalization_id')['final_death_dttm'].to_dict()
    cr['_death'] = cr['hospitalization_id'].map(death_map)
    cr['_death'] = pd.to_datetime(cr['_death'])
    cr['hours_before_death'] = (
        (cr['event_dttm'] - cr['_death']).dt.total_seconds() / 3600.0
    )
    # Only include points up to death (right edge = 0)
    cr = cr[cr['hours_before_death'] <= 0]

    kidney_map = df.set_index('hospitalization_id')['kidney_donated'].fillna(False).to_dict()
    cr['kidney_donated'] = cr['hospitalization_id'].map(kidney_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    for kd, color, label in [
        (True, 'tab:green', 'Kidney donated'),
        (False, 'tab:red', 'Not donated'),
    ]:
        sub = cr[cr['kidney_donated'] == kd]
        for hid, grp in sub.groupby('hospitalization_id'):
            grp = grp.sort_values('hours_before_death')
            ax.plot(grp['hours_before_death'], grp['lab_creatinine'],
                    color=color, alpha=0.3, linewidth=0.8)
        # Add legend handle once per group (use last plot if any)
        if not sub.empty:
            ax.plot([], [], color=color, label=label)

    ax.axvline(0, color='black', linewidth=1, linestyle=':')
    ax.set_xlabel('Hours before death (0 = death)')
    ax.set_ylabel('Serum creatinine (mg/dL)')
    ax.set_title('Creatinine trajectory by kidney donation status (M+H matched cohort)')
    ax.legend(loc='upper left')
    fig.tight_layout()

    # Save standalone PNG
    fig.savefig(FINAL_DIR / 'niddk_aim1_creatinine_trajectory.png', dpi=150)

    # Encode for HTML embedding
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')
```

- [ ] **Step 3: Add the HTML report renderer**

Append:

```python
HTML_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { color: #1a3a6e; border-bottom: 2px solid #1a3a6e; padding-bottom: 0.3em; }
  h2 { color: #1a3a6e; margin-top: 1.6em; }
  table { border-collapse: collapse; margin: 1em 0; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; vertical-align: top; }
  th { background: #f0f4fa; }
  td:first-child { white-space: nowrap; }
  .meta { color: #555; font-size: 0.92em; }
  img { max-width: 100%; border: 1px solid #ddd; }
</style>
"""


def render_html(table1: pd.DataFrame, table2: pd.DataFrame, fig_b64: str,
                cohort_size: int, n_moderate: int, n_high: int, site_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>NIDDK Aim 1 — {site_name} Donor Quality Description</title>
{HTML_CSS}
</head><body>
<h1>NIDDK Aim 1 — {site_name} Donor Quality Description</h1>
<p class="meta">Site: {site_name} | Generated: {datetime.now():%Y-%m-%d %H:%M}
| N matched (M+H): {cohort_size}
({n_moderate} Moderate, {n_high} High)</p>

<h2>Cohort definition</h2>
<p>CLIF medically-eligible donor decedents at {site_name} who linked to SRTR with
HIGH or MEDIUM match confidence (best match per donor). Kidney donation is the
primary outcome; number of organs transplanted is secondary.</p>

<h2>Table 1 — Donor characteristics</h2>
{table1.to_html(index=False, escape=False)}

<h2>Table 2 — Donation outcomes</h2>
{table2.to_html(index=False, escape=False)}

<h2>Figure — Creatinine trajectory by kidney donation status</h2>
<img src="data:image/png;base64,{fig_b64}" alt="Creatinine trajectory"/>

<h2>Methods</h2>
<ul>
<li><strong>Population:</strong> CLIF medically-eligible donor decedents (CLIF v2.1.0 cohort
definition) linked to SRTR via the project's hierarchical record-linkage pipeline.
Cohort filtered to <code>match_confidence ∈ {{HIGH, MEDIUM}}</code> and best match
per donor.</li>
<li><strong>KDPI:</strong> Computed per the OPTN 2024 8-factor refit formula
(see <code>references/kdpi_guide.pdf</code>, April 21, 2025) using the 2024
reference-cohort mapping table (<code>references/kdpi_mapping_table.pdf</code>,
April 04, 2025). Donors with any missing required input have KDPI reported as
missing rather than imputed.</li>
<li><strong>Vasopressors within 48h death:</strong> Any continuous infusion of
norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine with
non-null/non-zero dose during the 48 hours preceding death.</li>
<li><strong>IMV within 48h death:</strong> Any respiratory support row with
<code>device_category = IMV</code> in the 48 hours preceding death. Note: ~100%
by construction since the CLIF cohort filter already requires IMV in this
window; reported for completeness.</li>
<li><strong>ICU LOS:</strong> Total time with <code>location_category=icu</code>
across stays, converted to days.</li>
<li><strong>Kidney donation:</strong> ≥1 kidney with disposition = transplanted
in SRTR <code>donor_disposition</code>.</li>
</ul>

<h2>References</h2>
<ul>
<li><code>references/kdpi_guide.pdf</code> — OPTN, "A Guide to Calculating and
Interpreting the Kidney Donor Profile Index (KDPI)", April 21, 2025.</li>
<li><code>references/kdpi_mapping_table.pdf</code> — OPTN, "KDRI to KDPI Mapping
Table", April 04, 2025 (2024 reference cohort).</li>
<li>Rao PS et al. <em>A comprehensive risk quantification score for deceased
donor kidneys: The kidney donor risk index.</em> Transplantation. 2009;88(2):231-236.</li>
<li>OPTN Minority Affairs Committee. <em>Refit Kidney Donor Profile Index
without Race and Hepatitis C Virus.</em> 2024.</li>
</ul>

</body></html>
"""
```

- [ ] **Step 4: Wire it all together in main**

Replace `main` with:

```python
def main() -> None:
    inputs = load_inputs()
    cohort = filter_cohort(inputs['mapping'])
    df = build_donor_table(cohort, inputs['clif'], inputs['srtr'], inputs['outcomes'])
    df = add_clif_derived(df, inputs['wide'], inputs['adt'])

    table1 = build_table1(df)
    table2 = build_table2(df)
    fig_b64 = build_creatinine_figure(df, inputs['wide'])

    # Save tabular outputs
    table1.to_csv(FINAL_DIR / 'niddk_aim1_table1.csv', index=False)
    table2.to_csv(FINAL_DIR / 'niddk_aim1_table2.csv', index=False)
    print(f"Saved Table 1 -> {FINAL_DIR / 'niddk_aim1_table1.csv'}")
    print(f"Saved Table 2 -> {FINAL_DIR / 'niddk_aim1_table2.csv'}")

    n_mod = int((df['confidence_stratum'] == 'Moderate').sum())
    n_high = int((df['confidence_stratum'] == 'High').sum())
    site_name = config.get('site_name', 'site')
    html = render_html(table1, table2, fig_b64,
                       cohort_size=len(df), n_moderate=n_mod, n_high=n_high,
                       site_name=site_name.upper())

    out_html = FINAL_DIR / 'niddk_aim1_report.html'
    out_html.write_text(html, encoding='utf-8')
    print(f"Saved HTML report -> {out_html}")
    print(f"Saved figure -> {FINAL_DIR / 'niddk_aim1_creatinine_trajectory.png'}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run end-to-end and verify outputs**

Run: `uv run python record_linkage/03_niddk_aim1_tables.py`

Expected:
- Console prints all stage logs (load, cohort, KDPI distribution, CLIF-derived, save)
- `output/final/niddk_aim1_report.html` exists
- `output/final/niddk_aim1_table1.csv` and `niddk_aim1_table2.csv` exist
- `output/final/niddk_aim1_creatinine_trajectory.png` exists

- [ ] **Step 6: Open the HTML in a browser to inspect visually**

Run: `open /Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier/output/final/niddk_aim1_report.html`

Expected: page renders with title bar, cohort line, two tables with three numeric columns each (Overall/Moderate/High), the creatinine figure, methods section, and references.

- [ ] **Step 7: Commit**

```bash
git add record_linkage/03_niddk_aim1_tables.py
git commit -m "feat(niddk): add Tables 1+2, creatinine figure, and HTML report"
```

---

## Task 8: Validation

**Files:** none (verification only)

- [ ] **Step 1: Validate KDPI distribution sanity**

Run:

```bash
/Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier/.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('output/final/niddk_aim1_table1.csv')
kdpi_row = df[df['Variable'].str.contains('KDPI', na=False)]
print(kdpi_row.to_string(index=False))
"
```

Expected: KDPI mean (SD) row shows a value with mean roughly in the 30-60 range (typical OPO population) and SD roughly 20-30.

- [ ] **Step 2: Validate cohort count consistency**

Run:

```bash
/Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier/.venv/bin/python -c "
import pandas as pd
mapping = pd.read_parquet('output/intermediate/encounter_mapping_matched.parquet')
mh_best = mapping[mapping['match_confidence'].isin(['HIGH','MEDIUM']) & (mapping['is_best']==True)]
t1 = pd.read_csv('output/final/niddk_aim1_table1.csv')
print('Mapping M+H best:', len(mh_best))
print('Table 1 N row:', t1.iloc[0].to_dict())
"
```

Expected: the "N (% of cohort)" Overall cell n matches the mapping count.

- [ ] **Step 3: Final commit (if any incidental cleanup needed)**

If validation surfaced issues that required code changes, fix and recommit. Otherwise nothing to commit here.

```bash
git status
# If clean, no commit needed.
```

---

## Self-Review Notes

- Spec coverage: every numbered section (§1–§12) of the spec maps to a task above. §11 (out of scope) is enforced by the absence of HgbA1c handling, no imputation beyond OPTN HTN/DM, no inferential statistics, and single-site UCMC focus.
- Type consistency: `compute_kdpi` consistently takes a row-like (dict / pd.Series); `build_donor_table` produces the joined frame consumed by both `add_clif_derived` and the table builders. Column names are stable across tasks (`KDPI`, `vasopressors_within_48h`, `imv_within_48h`, `icu_los_days`, `n_kidneys_transplanted`, `n_organs_transplanted`, `kidney_donated`).
- Placeholder-free: every code block contains the actual implementation; no "TODO" or "fill in" markers; `_is_transplanted` heuristic is concrete with a note to verify the disposition coding against the actual SAS values in Task 5 Step 1.
