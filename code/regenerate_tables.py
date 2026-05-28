#!/usr/bin/env python
"""Re-emit Table 1 (and Table 2) from the saved final_cohort_df.parquet
WITHOUT rerunning the full pipeline.

Use this when only the table specification has changed (e.g., new rows added
to utils/table_one.py) and you want updated CSV/HTML outputs without
spending 15-30 minutes redoing the cohort identification.

Reads:   output/intermediate/final_cohort_df.parquet
Writes:  output/final/table_one.csv
         output/final/table_one.html
         output/final/aim1_table_two_by_terminal_cr.csv
         output/final/regenerate_log.txt   (stdout mirror of this run)

Run from the repo root:
    uv run python code/regenerate_tables.py
"""

from __future__ import annotations

import atexit
import datetime as _dt
import sys
from pathlib import Path

import polars as pl

from utils.config import config
from utils.table_one import create_table_one, create_table_two_by_terminal_cr

PROJECT_ROOT = Path(config["project_root"])
INTERMEDIATE = PROJECT_ROOT / "output" / "intermediate" / "final_cohort_df.parquet"
OUT_DIR      = PROJECT_ROOT / "output" / "final"


class _Tee:
    """Write to multiple streams simultaneously (terminal + log file)."""
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


def _start_logging() -> Path:
    """Tee stdout to output/final/regenerate_log.txt for this run.
    Uses a distinct filename so the main pipeline's run_log.txt is preserved.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "regenerate_log.txt"
    handle = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, handle)
    atexit.register(lambda: (handle.flush(), handle.close()))
    print(f"=== regenerate_tables.py started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===")
    print(f"Logging stdout to: {log_path}")
    print("-" * 80)
    return log_path


def main() -> None:
    _start_logging()
    if not INTERMEDIATE.exists():
        sys.exit(
            f"ERROR: {INTERMEDIATE} not found. Run the full pipeline "
            "(code/01_potential_donor_identifier.py) at least once before "
            "using this regenerator."
        )

    print(f"Reading saved cohort: {INTERMEDIATE}")
    df = pl.read_parquet(INTERMEDIATE)
    print(f"  rows={df.shape[0]:,}, cols={df.shape[1]}")

    missing = [c for c in ("last_height_cm", "last_weight_kg") if c not in df.columns]
    if missing:
        print(f"  WARNING: parquet is missing {missing}; those rows will be "
              "skipped. To get them you'll need to rerun the full pipeline once.")

    print("\nRegenerating Table 1...")
    create_table_one(df, output_dir=str(OUT_DIR))

    print("\nRegenerating Aim 1 Table 2 (by terminal Cr) on died_while_imv cohort...")
    cohort_col = "died_while_imv" if "died_while_imv" in df.columns else "imv_48hr_expire"
    create_table_two_by_terminal_cr(
        df, output_dir=str(OUT_DIR), cohort_filter_column=cohort_col,
    )

    print(f"\n✓ Done. Updated files in: {OUT_DIR}")
    print("  - table_one.csv  (now includes Height (cm) and Weight (kg) rows)")
    print("  - table_one.html")
    print("  - aim1_table_two_by_terminal_cr.csv")


if __name__ == "__main__":
    main()
