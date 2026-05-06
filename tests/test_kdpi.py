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
