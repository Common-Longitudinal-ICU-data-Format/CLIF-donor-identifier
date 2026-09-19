"""Recovery-rate caterpillar: SRTR donors per 100 eligible decedents, by hospital.

Standalone — not wired into the dashboard.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path("/Users/kavenchhikara/Projects/CLIF/CLIF-donor-identifier")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
      REPO / "manuscript_results/recovery_rate_caterpillar.png"
TEAL, INK, RULE, OCHRE = "#0e6b61", "#16262a", "#dbe3e0", "#9c5410"
SITE_COLOUR = {"nu": TEAL, "rush": OCHRE, "ucmc": "#3f6f8f"}


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


d = pl.read_csv(REPO / "manuscript_results/srtr_hospital_match.csv")
DENOMS = [("CLIF_donor", "per 100 CLIF-donor eligible"),
          ("CALC", "per 100 CALC eligible"),
          ("n_decedents", "per 100 in-hospital decedents")]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
for ax, (col, lab) in zip(axes, DENOMS):
    dd = d.filter(pl.col(col) > 0).with_columns(
        (100 * pl.col("srtr_donors") / pl.col(col)).alias("rate")).sort("rate")
    x = np.arange(1, dd.height + 1)
    k = dd["srtr_donors"].to_numpy()
    n = dd[col].to_numpy()
    rate = 100 * k / n
    lo, hi = wilson(k, n)
    for i, r in enumerate(dd.iter_rows(named=True)):
        c = SITE_COLOUR.get(r["site"], INK)
        ax.errorbar(x[i], rate[i], yerr=[[rate[i] - lo[i]], [hi[i] - rate[i]]],
                    fmt="o", ms=6, lw=0, elinewidth=1.2, capsize=2.8, color=c, zorder=3)
    pooled = 100 * k.sum() / n.sum()
    ax.axhline(pooled, color=INK, ls=":", lw=1.2, zorder=1)
    ax.annotate(f"pooled {pooled:.1f}", xy=(dd.height + .3, pooled), fontsize=8,
                va="center", color=INK, annotation_clip=False)
    ax.set_xticks(x)
    ax.set_xticklabels(dd["hospital_label"], rotation=90, fontsize=7)
    ax.set_title(lab, fontsize=9.5, color=INK)
    ax.set_xlim(.4, dd.height + 1.2)
    ax.grid(axis="y", color=RULE, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)
axes[0].set_ylabel("SRTR donors recovered per 100 eligible (95% Wilson CI)", fontsize=9)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s.upper())
           for s, c in SITE_COLOUR.items()]
axes[0].legend(handles=handles, frameon=False, fontsize=8.5, ncol=3, loc="upper left")
fig.suptitle("Organ recovery rate by hospital, ranked  —  three denominators",
             fontsize=11, color=INK, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"wrote {OUT}")
for col, lab in DENOMS:
    dd = d.filter(pl.col(col) > 0)
    r = 100 * dd["srtr_donors"].sum() / dd[col].sum()
    lo_, hi_ = (100 * dd["srtr_donors"] / dd[col]).min(), (100 * dd["srtr_donors"] / dd[col]).max()
    print(f"  {lab:34s} pooled {r:5.1f}   hospital range {lo_:.1f}–{hi_:.1f}")
