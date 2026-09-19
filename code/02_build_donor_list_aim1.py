#!/usr/bin/env python
"""Build the NIDDK Aim 1 donor lists (all-organ + kidney) per CLIF site.

Mirrors the logic in shared/donor_ids.Rmd, with three changes for Aim 1:
  * Date window: 2020-01-01 to 2025-12-31 (was 2018-2024 in Rmd).
  * Organ filter: kidney = DON_ORG IN ('LKI','RKI').  The SRTR codebook uses
    LKI/RKI; there is no plain 'KI' value, so a literal 'KI' filter would
    return zero rows.
  * Two outputs: shared/donor_ids_all_organs.csv  (any organ, one row / DONOR_ID)
                 shared/donor_ids_kidney.csv      (kidney only, one row / DONOR_ID)

CLIF site -> provider number mapping is taken from donor_ids.Rmd (expanded
lists) rather than config/hospital_provider_mapping.json (narrower lists).

Columns in each output CSV match donor_ids_clif.csv:
    DONOR_ID, DON_NON_HR_BEAT, clif_site, hospital_name, HOSPITAL_ZIP, don_utilized
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat

REPO = Path(__file__).resolve().parent.parent
SRTR = Path("/Users/kavenchhikara/Projects/CLIF/00_SRTR_DATA")

OUT_ALL    = REPO / "shared" / "donor_ids_all_organs.csv"
OUT_KIDNEY = REPO / "shared" / "donor_ids_kidney.csv"

DATE_START = pd.Timestamp("2020-01-01")
DATE_END   = pd.Timestamp("2025-12-31")
MIN_AGE    = 18

# Provider -> CLIF site, from donor_ids.Rmd (expanded lists).
SITE_PROVIDERS: dict[str, list[int]] = {
    "Emory":    [110010, 110033, 110076, 110078, 110082, 110172, 110178, 110179, 110183, 110226, 110230],
    "JHU":      [90005, 210022, 103300, 210009, 210029, 210048],
    "Michigan": [230046, 230208, 230230, 231326, 230326, 231327, 231331],
    "MIMIC VI 3.1": [220118, 220086, 220060, 220083, 220108],
    "NU":       [140176, 140062, 140116, 140130, 140154, 140203, 140242, 140281, 140286],
    "OHSU":     [380021, 380009],
    "Penn":     [390223, 390226],
    "RUMC":     [140119, 140029, 140063, 140310],
    "UCMC":     [140065, 140088, 140122, 140191, 140292, 140304],
    "UCSF":     [50033, 50454, 53301],
    "UMN":      [240078, 240040, 240046, 240049, 240050, 240063, 240064, 240080, 240081, 240207, 240213, 242004],
}

PROVIDER_TO_SITE: dict[int, str] = {
    prov: site for site, provs in SITE_PROVIDERS.items() for prov in provs
}


def _read_sas(name: str, **kwargs) -> pd.DataFrame:
    df, _ = pyreadstat.read_sas7bdat(str(SRTR / name), **kwargs)
    return df


def _slice_one_per_donor(df: pd.DataFrame) -> pd.DataFrame:
    """One row per DONOR_ID; earliest DON_RECOV_DT wins (deterministic)."""
    return (
        df.sort_values(["DONOR_ID", "DON_RECOV_DT"])
          .drop_duplicates(subset="DONOR_ID", keep="first")
          .reset_index(drop=True)
    )


def _hosp_lookup() -> pd.DataFrame:
    """PROVIDER_NUM -> hospital_name (joined) + HOSPITAL_ZIP."""
    don_hosp = _read_sas("donorhospital2506.sas7bdat",
                         usecols=["PROVIDER_NUM", "HOSPITAL_NAME", "HOSPITAL_ZIP"])
    don_hosp["PROVIDER_NUM"] = pd.to_numeric(don_hosp["PROVIDER_NUM"], errors="coerce")
    don_hosp = don_hosp.dropna(subset=["PROVIDER_NUM"])
    don_hosp["PROVIDER_NUM"] = don_hosp["PROVIDER_NUM"].astype("int64")
    don_hosp = don_hosp[don_hosp["PROVIDER_NUM"].isin(PROVIDER_TO_SITE)].drop_duplicates()
    agg = (
        don_hosp.groupby("PROVIDER_NUM", as_index=False)
                .agg(hospital_name=("HOSPITAL_NAME",
                                    lambda s: "; ".join(sorted(set(s.dropna())))),
                     HOSPITAL_ZIP=("HOSPITAL_ZIP", "first"))
    )
    return agg


def _annotate(donors: pd.DataFrame,
              prov_lkp: pd.DataFrame,
              deceased: pd.DataFrame,
              hosp_lkp: pd.DataFrame,
              utilized_ids: set[int]) -> pd.DataFrame:
    out = (
        donors.merge(prov_lkp, on="DONOR_ID", how="left")
              .merge(deceased,  on="DONOR_ID", how="left")
              .merge(hosp_lkp,  left_on="DON_HOSP_PROV_NUM",
                                right_on="PROVIDER_NUM", how="left")
    )
    out["clif_site"] = out["DON_HOSP_PROV_NUM"].map(PROVIDER_TO_SITE)
    out = out[out["clif_site"].notna()]
    out = out[out["DON_AGE"] >= MIN_AGE]
    out["don_utilized"] = out["DONOR_ID"].isin(utilized_ids).map({True: "Y", False: "N"})
    cols = ["DONOR_ID", "DON_NON_HR_BEAT", "clif_site",
            "hospital_name", "HOSPITAL_ZIP", "don_utilized"]
    return out[cols].sort_values(["clif_site", "DONOR_ID"]).reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path, label: str) -> None:
    df.index = range(1, len(df) + 1)
    df.index.name = ""
    df.to_csv(path)
    print(f"\n{label}: wrote {len(df):,} rows to {path.relative_to(REPO)}")
    print(df["clif_site"].value_counts().sort_index().to_string())
    print(f"  don_utilized=Y: {(df['don_utilized']=='Y').sum():,}  "
          f"({(df['don_utilized']=='Y').mean()*100:.1f}%)")


def main() -> None:
    print(f"Window: {DATE_START.date()} to {DATE_END.date()}  |  age >= {MIN_AGE}")
    print(f"CLIF sites: {len(SITE_PROVIDERS)}  |  providers: {len(PROVIDER_TO_SITE)}\n")

    print("Reading donor_disposition.sas7bdat...")
    disp = _read_sas("donor_disposition.sas7bdat",
                     usecols=["DONOR_ID", "DON_ORG", "DON_RECOV_DT", "DON_DISPOSITION"])
    disp["DONOR_ID"] = disp["DONOR_ID"].astype("int64")
    disp["DON_RECOV_DT"] = pd.to_datetime(disp["DON_RECOV_DT"], errors="coerce")

    in_window = disp[disp["DON_RECOV_DT"].between(DATE_START, DATE_END)]
    print(f"  rows in {DATE_START.date()}..{DATE_END.date()}: {len(in_window):,}")

    print("Reading deceasedtodonhosp2506.sas7bdat...")
    d2d = _read_sas("deceasedtodonhosp2506.sas7bdat",
                    usecols=["DONOR_ID", "DON_HOSP_PROV_NUM"])
    d2d["DONOR_ID"] = d2d["DONOR_ID"].astype("int64")
    d2d["DON_HOSP_PROV_NUM"] = pd.to_numeric(d2d["DON_HOSP_PROV_NUM"], errors="coerce")
    d2d = d2d.dropna(subset=["DON_HOSP_PROV_NUM"])
    d2d["DON_HOSP_PROV_NUM"] = d2d["DON_HOSP_PROV_NUM"].astype("int64")
    d2d = d2d.drop_duplicates(subset="DONOR_ID")

    print("Reading donor_deceased.sas7bdat...")
    deceased = _read_sas("donor_deceased.sas7bdat",
                         usecols=["DONOR_ID", "DON_AGE", "DON_NON_HR_BEAT"])
    deceased["DONOR_ID"] = deceased["DONOR_ID"].astype("int64")

    hosp_lkp = _hosp_lookup()
    print(f"Hospital lookup rows (CLIF providers only): {len(hosp_lkp):,}")

    # ---- All donors: one row per DONOR_ID (any organ) ----
    donors_all = _slice_one_per_donor(in_window)
    utilized_all = set(in_window.loc[in_window["DON_DISPOSITION"] == 6, "DONOR_ID"])
    print(f"\nAll donors (1 row / DONOR_ID, any organ): {len(donors_all):,}")
    all_df = _annotate(donors_all[["DONOR_ID"]], d2d, deceased, hosp_lkp, utilized_all)

    # ---- Kidney donors: LKI or RKI, one row per DONOR_ID ----
    kidney_rows = in_window[in_window["DON_ORG"].isin(["LKI", "RKI"])]
    donors_kid  = _slice_one_per_donor(kidney_rows)
    utilized_kid = set(kidney_rows.loc[kidney_rows["DON_DISPOSITION"] == 6, "DONOR_ID"])
    print(f"Kidney donors (DON_ORG in LKI/RKI, 1 row / DONOR_ID): {len(donors_kid):,}")
    kid_df = _annotate(donors_kid[["DONOR_ID"]], d2d, deceased, hosp_lkp, utilized_kid)

    _write(all_df, OUT_ALL,    "ALL ORGANS")
    _write(kid_df, OUT_KIDNEY, "KIDNEY")


if __name__ == "__main__":
    main()
