"""Site config loader.

Which site runs is chosen by the CLIF_DONOR_SITE environment variable, so the
same code runs unmodified at every site:

    CLIF_DONOR_SITE=ucmc python code/01_potential_donor_identifier.py

Falls back to config/config.json when the variable is unset, preserving the
original single-site behaviour.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(site: str | None = None) -> dict:
    site = site or os.environ.get("CLIF_DONOR_SITE")
    path = CONFIG_DIR / (f"config_{site.lower()}.json" if site else "config.json")
    if not path.exists():
        avail = sorted(p.stem.replace("config_", "") for p in CONFIG_DIR.glob("config_*.json")
                       if not p.stem.startswith(("config_template", "_retired")))
        raise FileNotFoundError(
            f"{path.name} not found. Set CLIF_DONOR_SITE to one of: {avail}")
    cfg = json.load(open(path))

    # project_root in a committed config can point at whoever created it; this
    # repo is the authority for its own paths.
    cfg["project_root"] = str(PROJECT_ROOT)
    cfg.setdefault("site_name", site or "unknown")
    cfg["site_name"] = str(cfg["site_name"]).lower()
    # Per-site output roots, following the CLIF project template's PHI split.
    cfg.setdefault("output_intermediate", str(PROJECT_ROOT / "output/intermediate_phi" / cfg["site_name"]))
    cfg.setdefault("output_final", str(PROJECT_ROOT / "output/final_no_phi" / cfg["site_name"]))
    print(f"Loaded configuration from {path.name}")
    return cfg


config = load_config()
