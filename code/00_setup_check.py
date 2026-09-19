#!/usr/bin/env python
"""Phase 0 gate — validate config, inventory CLIF tables, verify code sets.

Run this before anything else. It answers three questions for a site:

  1. Does the config point somewhere real, and is it CLIF 2.1?
  2. Which required tables and columns are present, and which optional ones are
     missing (and what does each absence cost the manuscript)?
  3. Do the hospital_id values in ADT reconcile with config/hospital_crosswalk.yaml?

Nothing here reads patient data beyond schemas and distinct hospital_id values.

    python code/00_setup_check.py                 # every configured site
    python code/00_setup_check.py --site ucmc
    python code/00_setup_check.py --validate-codes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import polars as pl
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REQ  = yaml.safe_load((REPO / "config/clif_data_requirements.yaml").read_text())
XWALK = yaml.safe_load((REPO / "config/hospital_crosswalk.yaml").read_text())["hospitals"]
OUT  = REPO / "output/final_no_phi/setup"
OUT.mkdir(parents=True, exist_ok=True)

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def load_site_configs(only: str | None) -> list[dict]:
    cfgs = []
    for p in sorted((REPO / "config").glob("config_*.json")):
        if p.name in {"config_template.json"}:
            continue
        try:
            c = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"{FAIL} {p.name}: invalid JSON — {e}")
            continue
        c["_file"] = p.name
        c["site_key"] = str(c.get("site_name", "")).lower()
        if only and c["site_key"] != only.lower():
            continue
        cfgs.append(c)
    return cfgs


def check_site(cfg: dict) -> dict:
    site = cfg["site_key"]
    print(f"\n{'='*78}\n{site.upper()}   ({cfg['_file']})\n{'='*78}")
    res = {"site": site, "config_file": cfg["_file"], "problems": [], "missing_required": [],
           "missing_optional": [], "missing_columns": {}}

    path = Path(cfg.get("tables_path", ""))
    print(f"  tables_path: {path}")
    if not path.is_dir():
        print(f"{FAIL} path does not exist")
        res["problems"].append("tables_path missing")
        return res
    if str(cfg.get("project_root", "")).rstrip("/") != str(REPO):
        print(f"{WARN} project_root points elsewhere: {cfg.get('project_root')}")
        res["problems"].append("project_root mismatch")

    present = {p.stem.replace("clif_", "") for p in path.glob("clif_*.parquet")}
    con = duckdb.connect()

    for kind in ("required", "optional"):
        for tbl, cols in REQ["tables"][kind].items():
            if tbl not in present:
                tag = FAIL if kind == "required" else WARN
                impact = REQ.get("optional_table_impact", {}).get(tbl, "")
                print(f"{tag} {tbl:30s} ABSENT" + (f" — {impact}" if impact else ""))
                res[f"missing_{kind}"].append(tbl)
                continue
            try:
                have = {d[0] for d in con.sql(
                    f"SELECT * FROM '{path/f'clif_{tbl}.parquet'}' LIMIT 0").description}
            except Exception as e:
                print(f"{FAIL} {tbl:30s} unreadable — {type(e).__name__}")
                res["problems"].append(f"{tbl} unreadable")
                continue
            miss = [c for c in cols if c not in have]
            if miss:
                print(f"{WARN} {tbl:30s} missing columns: {miss}")
                res["missing_columns"][tbl] = miss
            else:
                print(f"{OK} {tbl:30s} {len(have)} cols")

    # ── hospital_id reconciliation ───────────────────────────────────────────
    if "adt" in present:
        try:
            ids = con.sql(f"""SELECT DISTINCT lower(trim(CAST(hospital_id AS VARCHAR))) h
                              FROM '{path/'clif_adt.parquet'}' WHERE hospital_id IS NOT NULL""").df().h.tolist()
        except Exception:
            ids = []
        rows = [r for r in XWALK if r["site"] == site]
        known = {str(r["hospital_id"]).lower(): r for r in rows}
        unmapped = [h for h in ids if h not in known]
        included = {r["analytic_hospital_id"] for r in rows if r.get("include_flag")}
        print(f"\n  hospital_id in ADT: {len(ids)} | crosswalk rows: {len(rows)} "
              f"| analytic hospitals (included): {len(included)}")
        if unmapped:
            print(f"{FAIL} {len(unmapped)} hospital_id not in crosswalk: {sorted(unmapped)[:8]}")
            res["problems"].append(f"{len(unmapped)} unmapped hospital_id")
        elif ids:
            print(f"{OK} every ADT hospital_id maps to the crosswalk")
        dropped = [r for r in rows if not r.get("include_flag")]
        for r in dropped:
            print(f"{WARN} excluded: {r['hospital_id']} — {r.get('exclusion_reason')}")
        # Reconcile against the config's declared list as well as the crosswalk.
        # The crosswalk says which hospitals COULD appear; the config says which
        # this site expects. A mismatch either way is a real change — a hospital
        # added, renamed or dropped — and silently changes the denominator.
        declared = [str(h).lower() for h in cfg.get("hospital_ids", [])]
        if declared:
            missing = [h for h in declared if h not in ids]
            extra = [h for h in ids if h not in declared]
            print(f"\n  config declares {len(declared)} hospital_ids")
            if extra:
                print(f"{FAIL} in ADT but NOT declared in config: {sorted(extra)}")
                res["problems"].append(f"{len(extra)} undeclared hospital_id in ADT")
            if missing:
                print(f"{WARN} declared in config but absent from ADT: {sorted(missing)}")
            if not extra and not missing:
                print(f"{OK} config hospital_ids match ADT exactly")
            res["declared_hospital_ids"] = sorted(declared)
            res["undeclared_in_adt"] = sorted(extra)
            res["declared_absent_from_adt"] = sorted(missing)
        else:
            print(f"{WARN} config has no hospital_ids list — add one so a new or "
                  f"renamed hospital is caught rather than silently included")

        res["adt_hospital_ids"] = sorted(ids)
        res["unmapped_hospital_ids"] = sorted(unmapped)
        res["analytic_hospitals"] = sorted(included)
    return res


def validate_codes() -> None:
    print(f"\n{'='*78}\nCODE SETS\n{'='*78}")
    for rel, keycol in [("utils/codes/pd_cause_inclusion_icd10.csv", "icd10cm"),
                        ("utils/codes/pd_severe_sepsis_components_icd10.csv", "icd10cm"),
                        ("utils/codes/cpt_neuro.csv", "cpt_code"),
                        ("utils/icd10_contraindications.csv", "ICD-10-CM")]:
        p = REPO / rel
        if not p.exists():
            print(f"{FAIL} {rel} MISSING")
            continue
        df = pl.read_csv(p, comment_prefix="#", infer_schema_length=0)
        grp = [c for c in ("concept", "component", "dx_broad") if c in df.columns]
        detail = (", ".join(f"{k}={v}" for k, v in
                            sorted(df.group_by(grp[0]).len().iter_rows())) if grp else "")
        print(f"{OK} {rel:52s} {df[keycol].n_unique():>5} codes  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="one site; defaults to $CLIF_DONOR_SITE, else checks all")
    ap.add_argument("--validate-codes", action="store_true")
    ap.add_argument("--all", action="store_true", help="check every configured site")
    a = ap.parse_args()
    # A site running its own pipeline must not be blocked by another site's
    # config being unusable, so scope to CLIF_DONOR_SITE when it is set.
    import os
    if not a.site and not a.all:
        a.site = os.environ.get("CLIF_DONOR_SITE")

    if a.validate_codes:
        validate_codes()

    cfgs = load_site_configs(a.site)
    if not cfgs:
        print("no site configs found")
        return 1
    results = [check_site(c) for c in cfgs]

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    hard = 0
    for r in results:
        bad = r["missing_required"] or r["problems"]
        hard += bool(bad)
        print(f"  {r['site']:8s} {'BLOCKED' if bad else 'ready':8s} "
              f"| missing required: {r['missing_required'] or '-'} "
              f"| missing optional: {r['missing_optional'] or '-'}")
        for p in r["problems"]:
            print(f"           - {p}")
    (OUT / "setup_check.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {(OUT/'setup_check.json').relative_to(REPO)}")
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
