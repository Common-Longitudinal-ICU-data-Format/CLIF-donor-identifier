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
