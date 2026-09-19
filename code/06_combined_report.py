#!/usr/bin/env python
"""Combined results report — the tables and figures in the manuscript, nothing else.

    python code/06_combined_report.py --sites-dir manuscript_results

Sites are discovered by scanning the directory. Contents follow
CLIF-donor manuscript_4_WFP:

    Figure 1   Cohort selection
    Figure 2   Relative capture by Possible Donor and CLIF-donor criteria
    Figure 3   Agreement between Possible Donor and CLIF-donor across health systems
    Figure 4   Organ donation rates by eligibility definition
    Table 2    Decedent characteristics by definition
    Table 3    Clinical care delivery by definition
    Table S1   CLIF-donor versus administrative definitions
    Table S2   Missingness, by field and hospital
    Figure S1  Eligibility incidence across hospitals
    UpSet      Decedents shared between definitions

Counts pool additively; proportions are recomputed on the pooled denominator.
Medians come from the patient-level union where this machine holds it, otherwise
they are shown as the range across sites.
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import polars as pl                      # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
E = html.escape
# v6 dropped Possible Donor; CALC is the primary comparator. The definition is
# still computed upstream, it is simply not reported here.
DEFS = ["CLIF-donor", "CALC", "Ventilated Patient"]

TEAL, INK, RULE, OCHRE = "#0e6b61", "#16262a", "#dbe3e0", "#9c5410"

WANTED = ["table_stats_raw", "definition_counts", "hospital_level_counts",
          "two_by_two_clif_pd", "definition_overlap_upset", "missingness_by_hospital",
          "strobe_counts", "consort_counts", "cohort_numbers", "data_availability_by_hospital",
          "sepsis_definition_impact", "sepsis_ase_vs_icd",
          "sepsis_contraindication_breakdown", "sepsis_ase_criteria_composition",
          "definition_counts_by_hospital_type", "element_coverage",
          "exclusion_codes_by_step", "code_level_diagnostics"]


def discover(d: Path, exclude: set[str]) -> dict[str, Path]:
    out = {}
    for p in sorted(x for x in d.iterdir() if x.is_dir()):
        if p.name.startswith("_") or p.name in exclude:
            continue
        for cand in (p, p / "final"):
            if (cand / "table_stats_raw.csv").exists():
                out[p.name] = cand
                break
    return out


def load(folder: Path) -> dict[str, pl.DataFrame]:
    out = {}
    for w in WANTED:
        f = folder / f"{w}.csv"
        if f.exists():
            try:
                out[w] = pl.read_csv(f, infer_schema_length=0)
            except Exception:
                pass
    return out


def num(df, cols):
    return df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols if c in df.columns])


def b64(fig) -> str:
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure(fig, caption: str) -> str:
    return (f"<figure><img alt='{E(caption)}' src='data:image/png;base64,{b64(fig)}'>"
            f"<figcaption>{caption}</figcaption></figure>")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5)


# ── figures ──────────────────────────────────────────────────────────────────
def consort_tabs(flow: pl.DataFrame, label: str, key: str) -> str:
    """Figure 1 — one CONSORT per definition, behind a nested tab bar."""
    # v6 reports CLIF-donor and CALC; Ventilated Patient is the denominator
    # both share. Possible Donor is still computed but not reported.
    defs = [d for d in ("CLIF-donor", "CALC", "Ventilated Patient")
            if d in set(flow["definition"])]
    if not defs:
        return ""
    uid = key or "pooled"
    bar, body = [], []
    for i, d in enumerate(defs):
        slug = d.lower().replace(" ", "-")
        on = " on" if i == 0 else ""
        bar.append(f'<button class="subtab{on}" data-g="{uid}" data-s="{slug}">{E(d)}</button>')
        body.append(f'<div class="sub{on}" id="s-{uid}-{slug}">'
                    + fig_cohort_selection(flow.filter(pl.col("definition") == d), label, d)
                    + "</div>")
    return (f'<div class="subtabs">{"".join(bar)}</div>{"".join(body)}')


def fig_cohort_selection(flow: pl.DataFrame, label: str, defn: str) -> str:
    """Figure 1 — cohort selection, as a CONSORT flow.

    Counts come from consort_counts.csv, which applies the CLIF-donor criteria
    cumulatively; they pool by summing across sites at each step.
    """
    f = (num(flow, ["n", "step"]).group_by(["step", "label", "excluded_label"])
         .agg(pl.col("n").sum()).sort("step"))
    lab = f["label"].to_list()
    ns = [int(v) for v in f["n"]]
    exl = f["excluded_label"].to_list()
    n = len(ns)
    if n < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8.2, 1.36 * n + .4))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, n * 1.36); ax.axis("off")
    bw, bh, cx = 4.3, .8, 2.7

    for i in range(n):
        y = n * 1.36 - .78 - i * 1.36
        ax.add_patch(plt.Rectangle((cx - bw / 2, y - bh / 2), bw, bh, facecolor="#ffffff",
                                   edgecolor=INK, lw=1.2, zorder=2))
        ax.text(cx, y, f"{lab[i]}\nn = {ns[i]:,}", ha="center", va="center", fontsize=8.4,
                color=INK, zorder=3, linespacing=1.45)
        if i == n - 1:
            break
        ax.annotate("", xy=(cx, y - bh / 2 - .56), xytext=(cx, y - bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
        ymid = y - bh / 2 - .28
        ax.plot([cx, cx + 2.20], [ymid, ymid], color=INK, lw=1.1)
        ax.annotate("", xy=(cx + 2.33, ymid), xytext=(cx + 2.20, ymid),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
        ax.add_patch(plt.Rectangle((cx + 2.33, ymid - .33), 5.0, .66, facecolor="#f1f5f4",
                                   edgecolor=RULE, lw=1, zorder=2))
        ax.text(cx + 2.49, ymid, f"Excluded  n = {ns[i] - ns[i + 1]:,}\n{exl[i + 1]}",
                ha="left", va="center", fontsize=7.7, color=INK, zorder=3, linespacing=1.4)
    return figure(fig, f"Figure 1. Cohort selection, {defn} \u2014 {label}")


def cards_relative_capture(defcounts: pl.DataFrame, donors: int | None) -> str:
    """Relative capture of cohort decedents, as figures rather than a chart."""
    d = num(defcounts, defcounts.columns).sum()
    total = int(d["n_inpatient_deaths"][0])
    items = [("In-hospital decedents", total, None),
             ("CLIF-donor", int(d["clif_donor"][0]), total),
             ("CALC", int(d["calc"][0]), total),
             ("Ventilated Patient", int(d["ventilated_no_age_limit"][0]), total)]
    if donors is not None:
        items.append(("SRTR donors", donors, total))
    cells = "".join(
        f"<div class='card'><b>{v:,}</b><span>{lab}</span>"
        + (f"<em>{100*v/den:.1f}% of decedents</em>" if den else "")
        + "</div>" for lab, v, den in items)
    return f"<div class='cards'>{cells}</div>"


def fig_agreement(two: pl.DataFrame) -> str:
    """Figure 3 — agreement between Possible Donor and CLIF-donor, by health system."""
    t = num(two, ["cohens_kappa", "pabak", "positive_agreement"]).sort("site")
    fig, ax = plt.subplots(figsize=(6.6, .62 * t.height + 1.5))
    y = np.arange(t.height)
    for off, col, lab, c in [(-.22, "cohens_kappa", "Cohen's κ", TEAL),
                             (0, "pabak", "PABAK", "#7fb3ab"),
                             (.22, "positive_agreement", "positive agreement", OCHRE)]:
        ax.scatter(t[col], y + off, s=46, color=c, label=lab, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([s.upper() for s in t["site"]], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("agreement", fontsize=9)
    ax.grid(axis="x", color=RULE, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(.5, 1.18))
    style(ax); ax.tick_params(left=False)
    return figure(fig, "Figure S2. Correlation between clinical eligibility "
                       "definitions among cohort hospitals")


def fig_donation_rates(hosp: pl.DataFrame) -> str:
    """Figure 5 — donation rate by definition (v6 numbering)."""
    cols = ["CLIF_donor", "CALC", "Ventilated_Patient"]
    h = num(hosp, cols + ["srtr_donors"])
    donors = h["srtr_donors"].sum()
    vals = [(c.replace("_", " "), 100 * donors / h[c].sum()) for c in cols if h[c].sum()]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, [v for _, v in vals], color=TEAL, height=.58)
    for yy, (lab, v) in zip(y, vals):
        ax.text(v, yy, f"  {v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([l for l, _ in vals], fontsize=9)
    ax.set_xlabel("SRTR donors per 100 eligible decedents", fontsize=9)
    ax.set_xlim(0, max(v for _, v in vals) * 1.16)
    style(ax); ax.tick_params(left=False)
    return figure(fig, "Figure 5. Incidence of actual organ donors by medical "
                       "eligibility definition and cohort health system")


def fig_missingness_heatmap(miss: pl.DataFrame, label: str) -> str:
    """Table S2 — missingness, fields by hospital."""
    m = num(miss, ["pct_missing"])
    fields = list(dict.fromkeys(m["field"]))
    hosps = sorted(set(m["hospital_label"]))
    grid = np.full((len(fields), len(hosps)), np.nan)
    idx = {f: i for i, f in enumerate(fields)}
    hix = {h: j for j, h in enumerate(hosps)}
    for r in m.iter_rows(named=True):
        grid[idx[r["field"]], hix[r["hospital_label"]]] = r["pct_missing"]
    fig, ax = plt.subplots(figsize=(max(4.4, .62 * len(hosps) + 3.2), .34 * len(fields) + 1.8))
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1, np.nanmax(grid)))
    ax.set_xticks(range(len(hosps)))
    ax.set_xticklabels(hosps, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(fields))); ax.set_yticklabels(fields, fontsize=8.5)
    for i in range(len(fields)):
        for j in range(len(hosps)):
            if not np.isnan(grid[i, j]):
                v = grid[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(grid) * .55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    cb.set_label("% missing", fontsize=8.5); cb.ax.tick_params(labelsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return figure(fig, f"Table S2. Missingness of clinical data by field and hospital — {label}")


def fig_data_availability(av: pl.DataFrame, label: str) -> str:
    """CLIF table and ICD-code coverage among decedents, by hospital."""
    av = num(av, ["pct_with_data"])
    piv = (av.pivot(values="pct_with_data", index="source", on="hospital_label",
                    aggregate_function="mean").sort("source"))
    hosps = sorted(c for c in piv.columns if c != "source")
    M = np.array([[piv[h][i] for h in hosps] for i in range(piv.height)], dtype=float)
    fig, ax = plt.subplots(figsize=(max(6.4, .46 * len(hosps) + 3.6), .34 * piv.height + 1.5))
    im = ax.imshow(M, cmap="BuGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(hosps))); ax.set_xticklabels(hosps, rotation=90, fontsize=7)
    ax.set_yticks(range(piv.height)); ax.set_yticklabels(piv["source"].to_list(), fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 60 else INK)
    fig.colorbar(im, ax=ax, shrink=.7, label="% of decedents with data")
    ax.set_xlabel(""); ax.set_ylabel("")
    return figure(fig, f"Data availability among decedents, by hospital \u2014 {label}")


def wilson(k, n, z=1.96):
    """Wilson score interval — the CI the manuscript figure already quotes."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fig_caterpillar(hosp: pl.DataFrame, defs: list[tuple[str, str, str]],
                    caption: str) -> str:
    """Caterpillar plot, hospital level, ranked on CLIF-donor eligibility.

    Will Parker, 2026-08-20: "let's rank not by size but by rank, do an actual
    caterpillar plot with the x-axis rank of CLIF-eligible and the y-axis
    percent qualifying. also we need to do this on the hospital_id level."
    Rank is therefore the x position, percent is y, and one point is one
    hospital_label rather than a health system.

    Ventilated Patient sits near 61% while every other definition sits under
    13%, so one series on a shared axis flattens the rest into a band. When the
    largest pooled rate is more than three times the next, that series gets its
    own panel above, sharing the same rank axis.
    """
    cols = [c for c, _, _ in defs]
    h = num(hosp, cols + ["n_decedents"]).filter(pl.col("n_decedents") > 0)
    if not h.height:
        return ""
    h = h.with_columns((100 * pl.col("CLIF_donor") / pl.col("n_decedents")).alias("_rank_on"))
    h = h.sort("_rank_on")
    x = np.arange(1, h.height + 1)
    n = h["n_decedents"].to_numpy()

    pooled = {c: 100 * h[c].to_numpy().sum() / n.sum() for c in cols}
    order = sorted(cols, key=lambda c: pooled[c], reverse=True)
    split = (len(cols) > 2 and pooled[order[1]] > 0
             and pooled[order[0]] > 3 * pooled[order[1]])
    top = [d for d in defs if d[0] == order[0]] if split else []
    bottom = [d for d in defs if d[0] != order[0]] if split else defs

    w = max(6.8, .42 * h.height + 3.2)
    if split:
        fig, (ax_t, ax_b) = plt.subplots(
            2, 1, figsize=(w, 5.4), sharex=True,
            gridspec_kw={"height_ratios": [1, 2], "hspace": .12})
        panels = [(ax_t, top), (ax_b, bottom)]
    else:
        fig, ax_b = plt.subplots(figsize=(w, 4.3))
        panels = [(ax_b, defs)]

    for ax, series in panels:
        for col, label, colour in series:
            k = h[col].to_numpy()
            pct = 100 * k / n
            lo, hi = wilson(k, n)
            ax.errorbar(x, pct, yerr=[pct - lo, hi - pct], fmt="o", ms=5.2, lw=0,
                        elinewidth=1.2, capsize=2.6, color=colour, label=label, zorder=3)
            ax.axhline(pooled[col], color=colour, ls=":", lw=1.2, zorder=1)
            ax.annotate(f"pooled {pooled[col]:.1f}%", xy=(x[-1] + .35, pooled[col]),
                        fontsize=7.6, color=colour, va="center", annotation_clip=False)
        ax.set_xlim(.4, h.height + 1.4)
        ax.legend(frameon=False, fontsize=8.5, ncol=len(series), loc="upper left")
        ax.grid(axis="y", color=RULE, lw=.8); ax.set_axisbelow(True)
        style(ax)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(h["hospital_label"], rotation=90, fontsize=7)
    ax_b.set_xlabel("Hospital, ranked by CLIF-donor eligibility", fontsize=9)
    lab = "% of decedents qualifying (95% Wilson CI)"
    if split:
        fig.supylabel(lab, fontsize=9, x=.045)
    else:
        ax_b.set_ylabel(lab, fontsize=9)
    return figure(fig, caption)


def fig_limits_of_agreement(hosp: pl.DataFrame) -> str:
    """Figure 4 — limits of agreement between CLIF-donor and CALC, by hospital."""
    h = num(hosp, ["CLIF_donor", "CALC", "n_decedents"]).filter(pl.col("n_decedents") > 0)
    if h.height < 3:
        return ""
    a = 100 * h["CLIF_donor"].to_numpy() / h["n_decedents"].to_numpy()
    b = 100 * h["CALC"].to_numpy() / h["n_decedents"].to_numpy()
    mean, diff = (a + b) / 2, a - b
    n = len(diff)
    bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    se_loa = float(np.sqrt(3 * sd ** 2 / n))

    x0, x1 = min(mean) - .6, max(mean) + .6
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for lvl in (lo, hi):
        ax.fill_between([x0, x1], lvl - 1.96 * se_loa, lvl + 1.96 * se_loa,
                        color=OCHRE, alpha=.12, lw=0, zorder=1)
    ax.axhline(0, color=RULE, lw=.9, zorder=1)
    for lvl, lab, solid in [(bias, f"bias {bias:+.2f}", True),
                            (hi, f"upper LoA {hi:+.2f}", False),
                            (lo, f"lower LoA {lo:+.2f}", False)]:
        ax.axhline(lvl, color=INK if solid else OCHRE, ls="-" if solid else "--",
                   lw=1.1, zorder=2)
        ax.annotate(lab, xy=(x1, lvl), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, va="center", color=INK if solid else OCHRE,
                    annotation_clip=False)
    ax.scatter(mean, diff, s=46, color=TEAL, zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_xlabel("mean of the two definitions, % of decedents", fontsize=9)
    ax.set_ylabel("CLIF-donor \u2212 CALC, percentage points", fontsize=9)
    style(ax)
    return figure(fig, f"Figure 4. Limits of agreement between CLIF-donor and CALC, "
                       f"{n} hospitals. Shaded bands are the 95% CI of each limit")


def fig_hospital_type(bt: pl.DataFrame) -> str:
    """Academic versus community, pooled across sites."""
    cols = ["CLIF_donor", "CALC", "Ventilated_Patient"]
    b = (num(bt, cols + ["n_decedents", "n_hospitals"])
         .group_by("hospital_type")
         .agg([pl.col(c).sum() for c in cols + ["n_decedents", "n_hospitals"]])
         .sort("hospital_type"))
    if b.height < 2:
        return ("<p class='note'>Only one hospital type is represented in the "
                "sites loaded; the academic versus community comparison needs a "
                "site with community hospitals.</p>")
    types = b["hospital_type"].to_list()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = .8 / len(types)
    x = np.arange(len(cols))
    for i, ty in enumerate(types):
        r = b.filter(pl.col("hospital_type") == ty)
        nd = float(r["n_decedents"][0])
        k = np.array([float(r[c][0]) for c in cols])
        pct = 100 * k / nd
        lo, hi = wilson(k, np.repeat(nd, len(cols)))
        pos = x + i * w - .4 + w / 2
        bars = ax.bar(pos, pct, w * .9, label=f"{ty} (n={int(nd):,})",
                      color=[TEAL, OCHRE][i % 2])
        ax.errorbar(pos, pct, yerr=[pct - lo, hi - pct], fmt="none",
                    ecolor=INK, elinewidth=1, capsize=2.5)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7.4, padding=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["CLIF-donor", "CALC", "Ventilated Patient"], fontsize=8.5)
    ax.set_ylabel("% of decedents qualifying (95% Wilson CI)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    return figure(fig, "Figure 3b. Eligibility by hospital type (academic versus community), "
                       "pooled across sites")


def fig_upset(upset: pl.DataFrame, label: str) -> str:
    """Decedents shared between definitions, from aggregate intersection counts."""
    u = num(upset, ["n_patients"]).sort("n_patients", descending=True).head(12)
    n_def, n_col = len(DEFS), u.height
    fig = plt.figure(figsize=(max(6.4, n_col * .62), 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.2], hspace=.06)
    axb, axm = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    x = np.arange(n_col)
    axb.bar(x, u["n_patients"], color=TEAL, width=.64)
    for i, v in enumerate(u["n_patients"]):
        axb.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7.5, color=INK)
    axb.set_ylabel("decedents", fontsize=9)
    axb.set_ylim(0, u["n_patients"].max() * 1.2)
    axb.set_xlim(-.6, n_col - .4); axb.set_xticks([])
    style(axb); axb.spines["bottom"].set_visible(False)
    members = [set(str(m).split(" ∩ ")) for m in u["members"]]
    for r, d in enumerate(DEFS):
        yy = n_def - 1 - r
        axm.axhline(yy, color="#eef2f0", lw=11, zorder=0)
        on = [i for i in range(n_col) if d in members[i]]
        axm.scatter(x, [yy] * n_col, s=48, color="#d5ddda", zorder=1)
        if on:
            axm.scatter(on, [yy] * len(on), s=48, color=INK, zorder=2)
    for i in range(n_col):
        ys = [n_def - 1 - r for r, d in enumerate(DEFS) if d in members[i]]
        if len(ys) > 1:
            axm.plot([i, i], [min(ys), max(ys)], color=INK, lw=1.6, zorder=1)
    axm.set_yticks(range(n_def)); axm.set_yticklabels(DEFS[::-1], fontsize=9)
    axm.yaxis.tick_right()
    axm.set_xlim(-.6, n_col - .4); axm.set_xticks([]); axm.set_ylim(-.6, n_def - .4)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.tick_params(left=False, right=False, pad=8)
    return figure(fig, f"Decedents shared between eligibility definitions — {label}")


# ── tables ───────────────────────────────────────────────────────────────────
COLMAP = {"Age at death": "age_at_death", "Weight [kg]": "last_weight_kg",
          "Height [cm]": "last_height_cm", "BMI": "bmi",
          "Terminal creatinine [mg/dL]": "creatinine_value",
          "Terminal total bilirubin [mg/dL]": "bilirubin_total_value",
          "Terminal AST [U/L]": "ast_value", "Terminal ALT [U/L]": "alt_value",
          "Terminal BUN [mg/dL]": "bun_value",
          "Terminal sodium [mmol/L]": "sodium_value",
          "Terminal GCS": "gcs_total_value", "Terminal RASS": "rass_value",
          "Hospital LOS [days]": "hospital_length_of_stay_days",
          "First ICU LOS [days]": "first_icu_los_days"}
DEFCOL = {"CLIF-donor": "clif_eligible_donors", "Possible Donor": "possible_donor",
          "CALC": "calc_flag", "Ventilated Patient": "ventilated_patient_no_age_limit"}



# SRTR lookup lives in utils/srtr.py so this report and
# code/08_srtr_actual_donors.py share one implementation.
from utils.srtr import donor_characteristics, match_hospitals   # noqa: E402


COLMAP = {"Age at death": "age_at_death", "Weight [kg]": "last_weight_kg",
          "Height [cm]": "last_height_cm", "BMI": "bmi",
          "Terminal creatinine [mg/dL]": "creatinine_value",
          "Terminal total bilirubin [mg/dL]": "bilirubin_total_value",
          "Terminal AST [U/L]": "ast_value", "Terminal ALT [U/L]": "alt_value",
          "Terminal BUN [mg/dL]": "bun_value",
          "Terminal sodium [mmol/L]": "sodium_value",
          "Terminal GCS": "gcs_total_value", "Terminal RASS": "rass_value",
          "Hospital LOS [days]": "hospital_length_of_stay_days",
          "First ICU LOS [days]": "first_icu_los_days"}
DEFCOL = {"CLIF-donor": "clif_eligible_donors", "Possible Donor": "possible_donor",
          "CALC": "calc_flag", "Ventilated Patient": "ventilated_patient_no_age_limit"}
def fig_sepsis_rules(imp: pl.DataFrame) -> str:
    """What each candidate sepsis rule actually detects, by site.

    Deliberately NOT plotted as eligibility: the rule that finds the most sepsis
    leaves the fewest eligible, so an eligibility chart reads backwards. The
    resulting eligible counts are in the table below the figure.
    """
    imp = num(imp, ["n_excluded_by_sepsis", "n_clif_donor_eligible", "denominator"])
    rules = ["ICD-10 sepsis, ever (current)", "CDC ASE, ever",
             "CDC ASE, onset within 48h of death", "no sepsis exclusion"]
    short = ["ICD-10, ever\n(current)", "CDC ASE\never", "CDC ASE\nwithin 48 h",
             "no sepsis\nexclusion"]
    sites = sorted(set(imp["site"]))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    w = .8 / max(len(sites), 1)
    x = np.arange(len(rules))
    cols = [TEAL, OCHRE, "#3f6f8f", "#7a6796"]
    for i, s_ in enumerate(sites):
        sub = imp.filter(pl.col("site") == s_)
        vals = [(sub.filter(pl.col("sepsis_rule") == r)["n_excluded_by_sepsis"].to_list()
                 or [0])[0] for r in rules]
        b = ax.bar(x + i * w - .4 + w / 2, vals, w * .92, label=s_, color=cols[i % len(cols)])
        ax.bar_label(b, fmt="%.0f", fontsize=7, padding=1.5)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylabel("decedents excluded by the sepsis rule", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(sites))
    ax.grid(axis="y", color=RULE, lw=.8); ax.set_axisbelow(True)
    style(ax)
    return figure(fig, "Sepsis rule diagnostic. Decedents excluded by each candidate "
                       "sepsis criterion, by site \u2014 more excluded means fewer "
                       "CLIF-donor eligible (table below)")


def pool_stats(raw: pl.DataFrame, patient_level: pl.DataFrame | None) -> pl.DataFrame:
    raw = num(raw, ["n", "denom", "median", "q25", "q75", "n_nonnull"])
    rows = []
    for (t, v, d), g in raw.filter(pl.col("stat") == "count").group_by(
            ["table", "variable", "definition"], maintain_order=True):
        n, dn = g["n"].sum(), g["denom"].sum()
        rows.append({"table": t, "variable": v, "definition": d,
                     "display": f"{int(n):,} ({100*n/dn:.1f}%)" if dn else "—"})
    for (t, v, d), g in raw.filter(pl.col("stat") == "median").group_by(
            ["table", "variable", "definition"], maintain_order=True):
        col, dc = COLMAP.get(v), DEFCOL.get(d)
        if patient_level is not None and col in patient_level.columns and dc in patient_level.columns:
            s = patient_level.filter(pl.col(dc).fill_null(False))[col].drop_nulls()
            disp = (f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})"
                    if s.len() else "—")
        else:
            m = g["median"].drop_nulls()
            disp = (f"{m.min():.1f}–{m.max():.1f}" if m.len() > 1
                    else (f"{m[0]:.1f}" if m.len() else "—"))
        rows.append({"table": t, "variable": v, "definition": d, "display": disp})
    return pl.DataFrame(rows)


def wide(stats: pl.DataFrame, tbl: str, defs=DEFS,
         donor_col: pl.DataFrame | None = None) -> pl.DataFrame:
    s = stats.filter(pl.col("table") == tbl)
    order = list(dict.fromkeys(s["variable"]))
    dmap = (dict(zip(donor_col["variable"], donor_col["value"]))
            if donor_col is not None else None)
    out = []
    for v in order:
        r = {"Variable": v}
        for d in defs:
            m = s.filter((pl.col("variable") == v) & (pl.col("definition") == d))
            r[d] = m["display"][0] if m.height else "—"
        if dmap is not None:
            r["SRTR donors"] = dmap.get(v, "—")
        out.append(r)
    return pl.DataFrame(out)


def html_table(df, annotate: dict[str, tuple[str, str]] | None = None) -> str:
    """annotate maps a first-column value to (replacement label, trailing HTML)."""
    if df is None or df.height == 0:
        return ""
    th = "".join(f"<th>{E(str(c))}</th>" for c in df.columns)

    def cell(v, first):
        s = "" if v is None else str(v)
        if first and annotate and s in annotate:
            lab, extra = annotate[s]
            return f"<td>{E(lab)} {extra}</td>"
        return f"<td>{E(s)}</td>"

    tr = "".join("<tr>" + "".join(cell(v, i == 0) for i, v in enumerate(r))
                 + "</tr>" for r in df.iter_rows())
    return f"<div class='scroll'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


# ── About panel ──────────────────────────────────────────────────────────────
_POP_N = [0]
_POP_DIALOGS: list[str] = []


def pop(label: str, rows: list[tuple[str, str]], note: str = "") -> str:
    """A code list, opened as a modal dialog over the page.

    Returns only the trigger button. The <dialog> is collected and emitted once
    at document level, because a Table 3 row renders in every tab and repeating
    the markup would duplicate element ids.

    <dialog>.showModal() gives a real window — centred, backdrop-dimmed, Esc to
    close, focus trapped — with no library and nothing loaded from the network.
    """
    _POP_N[0] += 1
    i = _POP_N[0]
    body = "".join(f"<tr><td><code>{E(c)}</code></td><td>{E(d)}</td></tr>" for c, d in rows)
    note_h = f"<p class='popnote'>{E(note)}</p>" if note else ""
    _POP_DIALOGS.append(
        f'<dialog class="popdlg" id="d{i}">'
        f'<header><h3>{E(label)}</h3>'
        f'<button type="button" class="popx" data-close="d{i}" aria-label="Close">'
        f'&times;</button></header>'
        f'<div class="popscroll">{note_h}'
        f'<table class="poptbl"><tbody>{body}</tbody></table></div></dialog>')
    return f'<button type="button" class="popbtn" data-dlg="d{i}">{E(label)}</button>' 


def _codes(path: str, code_col: str, desc_col: str, where=None, limit: int | None = None):
    f = REPO / path
    if not f.exists():
        return []
    d = pl.read_csv(f, comment_prefix="#", infer_schema_length=0)
    if where is not None:
        d = d.filter(where)
    if limit:
        d = d.head(limit)
    return [(str(r[0]), str(r[1])) for r in d.select([code_col, desc_col]).iter_rows()]


# ── Table 3 provenance ───────────────────────────────────────────────────────
# Each row states, in a modal, exactly which codes or mCIDE med_category values
# were tested — otherwise a row like "med_donor_management_cont" is unreadable
# and a 0% cannot be told apart from a concept that was never probed.
T3_LABELS = {
    "proc_cerebral_angiography":  ("Cerebral angiography", "cerebral_angiography"),
    "proc_continuous_eeg":        ("Continuous EEG", "continuous_eeg"),
    "proc_craniotomy_hematoma":   ("Craniotomy for hematoma", "craniotomy_hematoma"),
    "proc_decompressive_craniectomy": ("Decompressive craniectomy", "decompressive_craniectomy"),
    "proc_endovascular_stroke":   ("Endovascular stroke therapy", "endovascular_stroke"),
    "proc_icp_evd":               ("ICP monitor or EVD", "icp_evd"),
}
T3_MEDS = {
    "med_corticosteroid_bolus_cont":    ("Corticosteroid, continuous", "corticosteroid_bolus"),
    "med_corticosteroid_bolus_intermit": ("Corticosteroid, intermittent", "corticosteroid_bolus"),
    "med_donor_management_cont":        ("Donor management, continuous", "donor_management"),
    "med_donor_management_intermit":    ("Donor management, intermittent", "donor_management"),
    "med_sedative_analgesic_cont":      ("Sedative or analgesic, continuous", "sedative_analgesic"),
    "med_sedative_analgesic_intermit":  ("Sedative or analgesic, intermittent", "sedative_analgesic"),
    "med_antiepileptic_cont":           ("Antiepileptic, continuous", "antiepileptic"),
    "med_antiepileptic_intermit":       ("Antiepileptic, intermittent", "antiepileptic"),
}
T3_OTHER = {
    "on_crrt_48h_before_death": "CRRT within 48 h of death",
    "prone_position": "Prone positioning",
}


def table3_provenance() -> dict[str, tuple[str, str]]:
    """variable -> (pretty label, modal HTML) for every Table 3 row."""
    import yaml as _y
    crit = _y.safe_load((REPO / "config/donor_criteria.yaml").read_text())
    meds = crit["clinical_care"]["medications"]
    npath = REPO / "utils/codes/neuro_procedures.csv"
    proc = (pl.read_csv(npath, comment_prefix="#", infer_schema_length=0)
            if npath.exists() else None)

    out: dict[str, tuple[str, str]] = {}
    for var, (label, concept) in T3_LABELS.items():
        rows, note = [], ""
        if proc is not None:
            sub = proc.filter(pl.col("concept") == concept)
            rows = [(f"{r[0]}  [{r[1]}]", r[2]) for r in
                    sub.select(["pattern", "vocabulary", "description"]).iter_rows()]
            n_cpt = sub.filter(pl.col("vocabulary") == "cpt").height
            n_pcs = sub.height - n_cpt
            note = (f"{n_cpt} CPT code(s) and {n_pcs} ICD-10-PCS pattern(s), OR-ed. "
                    "PCS entries are anchored regexes, not exact codes \u2014 sites code "
                    "inpatient procedures in different vocabularies.")
        out[var] = (label, pop(f"codes ({len(rows)})", rows, note))
    for var, (label, group) in T3_MEDS.items():
        tbl = ("medication_admin_continuous" if var.endswith("_cont")
               else "medication_admin_intermittent")
        rows = [(m, f"mCIDE med_category in clif_{tbl}") for m in meds.get(group, [])]
        out[var] = (label, pop(f"med_category ({len(rows)})", rows,
                               "Exact match on the mCIDE controlled vocabulary. Mannitol and "
                               "hypertonic saline arrive in CLIF 3.0; levetiracetam, phenytoin "
                               "and fosphenytoin are in neither 2.1 nor 3.0."))
    out["on_crrt_48h_before_death"] = (
        T3_OTHER["on_crrt_48h_before_death"],
        pop("how it is derived", [("clif_crrt_therapy", "Any CRRT record with a timestamp "
                                  "within 48 h before death. No code list \u2014 presence in "
                                  "the table is the criterion.")]))
    out["prone_position"] = (
        T3_OTHER["prone_position"],
        pop("how it is derived",
            [("position_category = 'prone'", "Exact equality on the mCIDE value in "
              "clif_position. NOT a substring match \u2014 LIKE '%prone%' also matches "
              "'not_prone', which once made NU read 93% proned.")],
            "Charting conventions differ: NU records position routinely including "
            "not_prone; RUSH and UCMC write rows only around proning episodes."))
    return out


def build_about() -> str:
    contra = REPO / "utils/icd10_contraindications.csv"
    sepsis = _codes("utils/icd10_contraindications.csv", "ICD-10-CM", "description",
                    pl.col("dx_broad") == "sepsis")
    cancer_n = 0
    if contra.exists():
        cd = pl.read_csv(contra, infer_schema_length=0)
        cancer_n = cd.filter(pl.col("dx_broad").is_in(["cancer", "other"])).height
    cancer = _codes("utils/icd10_contraindications.csv", "ICD-10-CM", "description",
                    pl.col("dx_broad").is_in(["cancer", "other"]), limit=40)
    pd_cause = _codes("utils/codes/pd_cause_inclusion_icd10.csv", "icd10cm",
                      "description", limit=40)
    pd_cause_n = 0
    f = REPO / "utils/codes/pd_cause_inclusion_icd10.csv"
    if f.exists():
        pd_cause_n = pl.read_csv(f, infer_schema_length=0).height
    comp = _codes("utils/codes/pd_severe_sepsis_components_icd10.csv", "icd10cm",
                  "description")
    msof = [(c, d) for c, d in comp if c.upper().replace(".", "").startswith("R652")]
    organ = [(c, d) for c, d in comp if not c.upper().replace(".", "").startswith("R652")]
    neuro = _codes("utils/codes/neuro_procedures.csv", "pattern", "description")

    calc_rng = [("I20\u2013I25", "Ischemic heart disease"),
                ("I60\u2013I69", "Cerebrovascular disease"),
                ("V01\u2013Y89", "External causes of morbidity and mortality")]

    def block(name, role, rows):
        body = "".join(f"<tr><td>{c}</td><td>{d}</td></tr>" for c, d in rows)
        return (f"<h2>{E(name)}</h2><p class='role'>{E(role)}</p>"
                f"<div class='scroll'><table class='about'><thead><tr>"
                f"<th>Criterion</th><th>How it is implemented</th></tr></thead>"
                f"<tbody>{body}</tbody></table></div>")

    out = ["<p class='lede'>Every definition below is applied to the same "
           "population: in-hospital deaths 2020\u20132025 at hospitals whose CCN "
           "resolves in SRTR. Counts are patients, not hospitalizations.</p>"]

    out.append(block(
        "CLIF-donor", "Definition of interest. Clinical criteria from structured EHR data.", [
            ("Ventilation", "Last IMV record in <code>respiratory_support</code> within 48 h "
             "before death (24 h grace after, for late charting). Identical to the "
             "Ventilated Patient definition."),
            ("Age", "\u2264 75 at death, from <code>patient.birth_date</code>, falling back to "
             "<code>age_at_admission</code> where birth date is absent."),
            ("Cancer / sepsis", "Any contraindicating ICD-10 code on the terminal "
             "hospitalization. <b>No time window</b> \u2014 <code>hospital_diagnosis</code> "
             "carries no timestamp. " + pop(f"cancer and related codes ({cancer_n})", cancer,
                                            "First 40 of "
                                            f"{cancer_n}; full list in utils/icd10_contraindications.csv.")
             + " " + pop(f"sepsis codes ({len(sepsis)})", sepsis,
                         "All 16. Several are non-severe despite Table 1 saying severe sepsis.")),
            ("Blood cultures", "No positive blood culture in <code>microbiology_culture</code> "
             "within 48 h of death."),
            ("Organ quality", "Kidney: creatinine &lt; 4 and not on CRRT within 48 h. "
             "Liver: bilirubin &lt; 4, AST &lt; 700, ALT &lt; 700. BMI \u2264 50. "
             "Last recorded value before death."),
        ]))

    out.append(block(
        "Possible Donor", "Computed by the pipeline but NOT reported in v6, which "
        "dropped it. Kallan seven-step cascade, Goldberg AJT 2017.", [
            ("Ventilation", "<b>Deliberate deviation.</b> The SAS uses ICD-9 procedure "
             "96.70\u201396.72; we use CLIF <code>respiratory_support</code>, the same flag "
             "as CLIF-donor, because billing codes for ventilation are not comparable "
             "across sites."),
            ("Age", "0\u201374 inclusive \u2014 one year narrower than CLIF-donor's \u2264 75."),
            ("Multi-system organ failure", "Excluded if coded. " +
             pop("MSOF codes", msof, "ICD-9 995.92/995.94 crosswalked to ICD-10.")),
            ("Cause of death", "AHRQ CCS 35, 47, 109, 228, 233, crosswalked to ICD-10 via "
             "DXCCSR v2025-1 by <code>utils/codes/build_pd_cause_crosswalk.py</code>, which "
             "reproduces the committed list exactly (<code>--check</code>). Or an external "
             "cause V00\u2013Y38, excluding Y62\u2013Y84 and Y90\u2013Y99 \u2014 narrower "
             "than CALC's V01\u2013Y89. " +
             pop(f"cause codes ({pd_cause_n})", pd_cause,
                 f"First 40 of {pd_cause_n}; full list in utils/codes/pd_cause_inclusion_icd10.csv.")),
            ("Cancer", "<b>Different provenance from the cause list.</b> Read from "
             "<code>utils/icd10_contraindications.csv</code>, a CCS-2015\u2192ICD-10 export "
             "that predates this build \u2014 not regenerated from DXCCSR, and it carries "
             "known defects. The intent is SAS step 6 (CCS 11\u201334, 36\u201345), with "
             "CCS 35 (brain/CNS) deliberately outside the exclusion because it is an "
             "inclusion at the cause step. See the caveat below."),
            ("Severe sepsis", "Sepsis in the <b>primary</b> diagnosis position AND at least "
             "one organ dysfunction in any other position. " +
             pop("organ dysfunction codes", organ, "AKI, shock liver, encephalopathy.")),
            ("Length of stay", "Hospital LOS \u2264 14 days before death."),
        ]))

    out.append(block(
        "CALC", "Secondary administrative comparator. Cause, Age and "
        "Location-Consistent criteria, as used in the CMS OPO performance measure.", [
            ("Location", "In-hospital death."),
            ("Age", "\u2264 75 at death."),
            ("Cause of death", "ICD-10 inclusion ranges. " +
             pop("CALC cause ranges", calc_rng,
                 "Ischemic heart disease is in CALC but NOT in Possible Donor \u2014 a real "
                 "definitional difference, not a data-source one.")),
            ("Contraindications", "Cancer and sepsis excluded, same code list as CLIF-donor. "
             "<b>Open question Q-05:</b> Table 1 says CALC specifies no contraindications; "
             "the code applies them."),
            ("Ventilation", "None. CALC has no ventilation criterion, which is why its "
             "cascade has no such step."),
        ]))

    out.append(block(
        "Ventilated Patient", "Secondary administrative comparator, from the HRSA form.", [
            ("Ventilation", "Last IMV record within 48 h before death. This is the same flag "
             "CLIF-donor and Possible Donor use, so both are strict subsets of this "
             "definition (verified: no patient in either falls outside it)."),
            ("Age", "<b>No age limit applied</b> (<code>apply_age_limit: false</code>), because "
             "Table 1 says \u201cNo restrictions\u201d. Applying \u2264 75 would give 7,169 "
             "instead of 10,013 pooled. <b>Open question Q-04:</b> Table 1 also says "
             "\u201cat time of death\u201d, which is stricter than 48 h."),
        ]))

    out.append(block(
        "Supporting elements", "Used in Tables 2 and 3, not in any eligibility rule.", [
            ("Neurologic procedures", "CPT exact codes OR anchored ICD-10-PCS regexes, "
             "because sites code inpatient procedures in different vocabularies \u2014 UCMC "
             "decedents carry ICD-10-PCS only. " +
             pop(f"procedure codes ({len(neuro)})", neuro)),
            ("Medications", "mCIDE <code>med_category</code> values in "
             "<code>medication_admin_continuous</code> and "
             "<code>_intermittent</code>. Mannitol and hypertonic saline arrive in CLIF 3.0; "
             "levetiracetam, phenytoin and fosphenytoin are in neither."),
            ("Fungemia", "<code>microbiology_culture</code>, blood/buffy fluid, organism "
             "matching candida, aspergillus, cryptococcus, fungus or yeast."),
            ("Serologies", "Not capturable. Strongyloides and chagas are absent from mCIDE; "
             "toxoplasma, syphilis, HIV and hepatitis serologies belong in "
             "<code>microbiology_nonculture</code>, where they are absent."),
            ("Sepsis, CDC ASE", "<code>clifpy.utils.ase.compute_ase</code> (CDC Sepsis "
             "Surveillance Toolkit 2018), reported as a diagnostic only. It does not "
             "currently feed any definition."),
        ]))

    out.append(
        "<h2>Caveat: two crosswalks, two provenances</h2>"
        "<p class='role'>The two ICD-10 code sets in this analysis were not built the "
        "same way, and only one is reproducible from source.</p>"
        "<div class='scroll'><table class='about'><tbody>"
        "<tr><td>PD cause list<br><span class='ok'>reproducible</span></td>"
        "<td>1,060 codes, generated from AHRQ DXCCSR v2025-1 by "
        "<code>build_pd_cause_crosswalk.py</code>. Re-derived and confirmed byte-identical "
        "to the committed file.</td></tr>"
        "<tr><td>Contraindications<br><span class='warn'>unverified</span></td>"
        "<td>1,174 codes in <code>utils/icd10_contraindications.csv</code>, carrying a "
        "<code>Sheet</code> column of the form <code>CCS 2015 -&gt; ICD-&lt;row&gt;</code>. "
        "The generating script is not in the repo, so it cannot be re-derived. "
        "Known defects: 9 codes whose description resolves to \u201cCode not found\u201d; "
        "<code>C8339</code> duplicated three times; and four codes that are not cancer at "
        "all but sit in the cancer/other exclusion \u2014 <code>B007</code> disseminated "
        "herpesviral disease, <code>G92</code> toxic encephalopathy, <code>R8581</code> and "
        "<code>R87810</code> positive HPV DNA tests. Across the three sites <b>38 decedents "
        "are excluded from CLIF-donor solely by one of those four codes</b> (NU 22, RUSH 6, "
        "UCMC 10).</td></tr>"
        "</tbody></table></div>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites-dir", default=str(REPO / "manuscript_results"))
    ap.add_argument("--patient-level-dir", default=str(REPO / "output/intermediate_phi"))
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--srtr-dir", default="/Users/kavenchhikara/Projects/CLIF/00_SRTR_DATA")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sdir = Path(a.sites_dir)
    sites = discover(sdir, set(a.exclude))
    if not sites:
        print(f"no site results under {sdir}")
        return 1
    print(f"sites: {', '.join(sites)}")
    data = {s: load(p) for s, p in sites.items()}

    NEED = list(COLMAP.values()) + list(DEFCOL.values())
    pls, missing_pl = [], []
    for s in sites:
        p = Path(a.patient_level_dir) / s / "cohort_with_variables.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            pls.append(df.select([c for c in NEED if c in df.columns]))
        else:
            missing_pl.append(s)
    union = pl.concat(pls, how="diagonal") if pls and not missing_pl else None

    def cat(key):
        fr = [data[s][key].with_columns(pl.lit(s).alias("site"))
              for s in sites if key in data[s]]
        return pl.concat(fr, how="diagonal") if fr else None

    # SRTR numerator. Coordinating site only: the extract is under a DUA and is
    # never part of what a site returns, so it is attached here rather than in
    # the per-site pipeline.
    hosp = cat("hospital_level_counts")
    donor_ids: set = set()
    donor_col = None
    donor_col_site: dict = {}
    if hosp is not None and "srtr_donors" not in hosp.columns:
        srtr = Path(a.srtr_dir)
        if srtr.is_dir():
            hosp, donor_ids = match_hospitals(srtr, hosp)
            donor_col = donor_characteristics(srtr, donor_ids)
            for _s in sites:
                _ids = set()
                _sub = hosp.filter(pl.col("site") == _s)
                if _sub.height:
                    _, _ids = match_hospitals(srtr, _sub.drop("srtr_donors", "matched_srtr"))
                donor_col_site[_s] = donor_characteristics(srtr, _ids) if _ids else None
            # hospitals sharing a CCN carry the same donors; collapse before summing
            hosp = (num(hosp, ["n_decedents", "CLIF_donor", "Possible_Donor", "CALC",
                               "Ventilated_Patient"])
                    .group_by(["site", "hospital_label", "srtr_ccn_id",
                               "ccn_facility_type", "hospital_type"])
                    .agg([pl.col(c).sum() for c in ["n_decedents", "CLIF_donor",
                          "Possible_Donor", "CALC", "Ventilated_Patient"]]
                         + [pl.col("srtr_donors").max()]))
        else:
            print(f"  SRTR not found at {srtr}; Figure 4 omitted")

    raw_all = cat("table_stats_raw")
    pooled = pool_stats(raw_all, union)
    upset, two = cat("definition_overlap_upset"), cat("two_by_two_clif_pd")
    miss, defc = cat("missingness_by_hospital"), cat("definition_counts")
    consort = cat("consort_counts")

    def collapse_upset(u: pl.DataFrame | None) -> pl.DataFrame | None:
        """Re-derive membership over the REPORTED definitions only.

        The sites compute the overlap across every definition they run,
        including Possible Donor. v6 does not report Possible Donor, so its
        rows must be MERGED into the matching reported set rather than drawn —
        otherwise e.g. "PD and Ventilated Patient" renders as a second
        "Ventilated Patient only" bar, identical to the real one.
        """
        if u is None or not u.height:
            return None
        cols = [d for d in DEFS if d in u.columns]
        if not cols:
            return num(u, ["n_patients"]).group_by("members").agg(
                pl.col("n_patients").sum()).sort("n_patients", descending=True)
        b = u.with_columns([pl.col(c).cast(pl.Utf8).str.to_lowercase()
                            .is_in(["true", "1"]).alias(f"_b_{c}") for c in cols])
        b = b.with_columns(pl.concat_str(
            [pl.when(pl.col(f"_b_{c}")).then(pl.lit(c)).otherwise(pl.lit(""))
             for c in cols], separator="\u0001").alias("_key"))
        agg = (num(b, ["n_patients"]).group_by("_key")
               .agg(pl.col("n_patients").sum()))
        agg = agg.with_columns(
            pl.col("_key").str.split("\u0001").list.eval(
                pl.element().filter(pl.element() != "")
            ).list.join(" \u2229 ").alias("members"))
        return (agg.filter(pl.col("members") != "")
                .select(["members", "n_patients"])
                .sort("n_patients", descending=True))

    upset_pooled = collapse_upset(upset)

    t3prov = table3_provenance()

    def panel(key, label, stats, sub):
        d = data.get(key, {})
        blocks = []
        sc = sub(consort, key) if consort is not None else None
        dc = sub(defc, key) if defc is not None else None
        mh = sub(miss, key) if miss is not None else None
        hs = sub(hosp, key) if hosp is not None else None
        av = (cat("data_availability_by_hospital") if key is None
              else d.get("data_availability_by_hospital"))
        if av is not None and key is not None and "site" not in av.columns:
            av = av.with_columns(pl.lit(key).alias("site"))
        us = (upset_pooled if key is None
              else collapse_upset(d.get("definition_overlap_upset")))
        dcol = donor_col if key is None else donor_col_site.get(key)
        pooled_only = key is None

        def sec(title):
            blocks.append(f'<h2>{E(title)}</h2>')

        # ── Tables ───────────────────────────────────────────────────────────
        sec("Table 1. Demographic and clinical characteristics of cohort "
            "decedents and organ donors")
        blocks.append(html_table(wide(stats, "table2", donor_col=dcol)))

        sec("Table 2. Clinical care delivery to cohort decedents by medical "
            "eligibility definition")
        blocks.append(html_table(wide(stats, "table3"), annotate=t3prov))

        # ── Figures ──────────────────────────────────────────────────────────
        if sc is not None and sc.height:
            sec("Figure 1. Cohort selection")
            blocks.append(consort_tabs(sc, label, key or "pooled"))

        if dc is not None and dc.height:
            sec("Figure 2. Relative capture of cohort decedents")
            blocks.append(cards_relative_capture(
                dc, int(hs["srtr_donors"].sum()) if hs is not None
                and "srtr_donors" in hs.columns else None))

        if pooled_only and hs is not None:
            sec("Figure 3. Incidence of medical eligibility by CLIF-donor and "
                "CALC criteria among decedents in cohort hospitals")
            blocks.append(fig_caterpillar(
                hs, [("CLIF_donor", "CLIF-donor", TEAL), ("CALC", "CALC", "#3f6f8f")],
                "Figure 3. Incidence of medical eligibility by CLIF-donor and CALC "
                "criteria, by hospital"))

            sec("Figure 4. Limits of agreement between CLIF-donor and CALC definitions")
            blocks.append(fig_limits_of_agreement(hs))

            if "srtr_donors" in hs.columns:
                sec("Figure 5. Incidence of actual organ donors by medical "
                    "eligibility definition among cohort hospitals")
                blocks.append(fig_donation_rates(hs))

        # ── Supplement ───────────────────────────────────────────────────────
        sec("Table S1. Missingness of clinical data among cohort decedents")
        if mh is not None and mh.height:
            blocks.append(fig_missingness_heatmap(mh, label))
        if av is not None and av.height:
            blocks.append(fig_data_availability(av, label))

        if us is not None and us.height:
            sec("Figure S1. Cohort decedents sharing eligibility definitions")
            blocks.append(fig_upset(us, label))

        if pooled_only and hs is not None:
            sec("Figure S2. Incidence of eligibility by definition and hospital")
            blocks.append(fig_caterpillar(
                hs, [("CLIF_donor", "CLIF-donor", TEAL), ("CALC", "CALC", "#3f6f8f"),
                     ("Ventilated_Patient", "Ventilated Patient", "#7a6796")],
                "Figure S2. All reported definitions, by hospital"))
            bt = cat("definition_counts_by_hospital_type")
            if bt is not None and bt.height:
                sec("Figure S3. Incidence of medical eligibility by definition "
                    "and hospital type")
                blocks.append(fig_hospital_type(bt))
                blocks.append(html_table(bt))
        return "".join(blocks)

    def by_site(df, key):
        return df if key is None else df.filter(pl.col("site") == key)

    tabs = [(None, "Pooled")] + [(s, s.upper()) for s in sites]
    bar = "".join(f'<button class="tab{" on" if i == 0 else ""}" data-p="{k or "pooled"}">{E(l)}</button>'
                  for i, (k, l) in enumerate(tabs))
    bar += '<button class="tab" data-p="about">About</button>'
    panels = "".join(
        f'<div class="panel{" on" if i == 0 else ""}" id="p-{k or "pooled"}">'
        + panel(k, l, pooled if k is None else pool_stats(data[k]["table_stats_raw"], union),
                by_site) + "</div>"
        for i, (k, l) in enumerate(tabs))
    panels += '<div class="panel" id="p-about">' + build_about() + '</div>'

    css = """
:root{--paper:#f5f7f6;--surface:#fff;--ink:#16262a;--body:#31454a;--muted:#65787a;
--accent:#0e6b61;--accent-soft:#e2efec;--rule:#dbe3e0}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#0c1416;--surface:#121e21;
--ink:#e2ebe8;--body:#bccac8;--muted:#87999a;--accent:#54b8a8;--accent-soft:#12312f;--rule:#223134}}
:root[data-theme=dark]{--paper:#0c1416;--surface:#121e21;--ink:#e2ebe8;--body:#bccac8;
--muted:#87999a;--accent:#54b8a8;--accent-soft:#12312f;--rule:#223134}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--body);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:0 26px}
header{background:var(--surface);border-bottom:1px solid var(--rule)}
header .wrap{padding:30px 26px 0}
h1{font-size:24px;font-weight:700;letter-spacing:-.02em;color:var(--ink);margin:0 0 16px}
.tabs{display:flex;gap:2px;flex-wrap:wrap}
.tab{appearance:none;border:0;background:none;font:inherit;font-size:13.5px;font-weight:600;
color:var(--muted);padding:10px 16px;cursor:pointer;border-bottom:2.5px solid transparent}
.tab:hover{color:var(--ink)}.tab.on{color:var(--accent);border-bottom-color:var(--accent)}
main{padding:28px 0 80px}.panel{display:none}.panel.on{display:block}
h2{font-size:16px;font-weight:600;color:var(--ink);margin:34px 0 12px;
border-top:2px solid var(--ink);padding-top:12px}
.panel>*:first-child{margin-top:0}
figure{margin:0 0 30px}figure img{max-width:100%;height:auto}
figcaption{font-size:13px;color:var(--ink);font-weight:600;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:12.8px;background:var(--surface)}
th{text-align:left;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
color:var(--muted);padding:8px 11px;border-bottom:1.5px solid var(--rule);white-space:nowrap;
position:sticky;top:0;background:var(--surface)}
td{padding:6px 11px;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums}
td:first-child{color:var(--ink);font-weight:500}
tbody tr:hover td{background:var(--accent-soft)}
.lede{font-size:14px;color:var(--body);max-width:62ch;margin:0 0 26px}
.role{font-size:12.5px;color:var(--muted);margin:-6px 0 12px;max-width:70ch}
table.about td{vertical-align:top}
table.about td:first-child{white-space:nowrap;width:1%;padding-right:20px}
table.about code{font-size:11.5px;background:var(--accent-soft);padding:1px 4px;border-radius:2px}
.popbtn{appearance:none;border:1px solid var(--rule);background:var(--accent-soft);
color:var(--accent);font:inherit;font-size:11.5px;font-weight:600;padding:1px 8px;
border-radius:10px;cursor:pointer;white-space:nowrap}
.popbtn:hover{border-color:var(--accent)}
.popdlg{padding:0;border:1px solid var(--rule);border-radius:6px;background:var(--surface);
color:var(--body);width:min(620px,92vw);max-height:78vh;box-shadow:0 18px 50px rgba(0,0,0,.3)}
.popdlg::backdrop{background:rgba(8,16,18,.5)}
.popdlg header{display:flex;align-items:center;justify-content:space-between;gap:12px;
padding:12px 14px;border-bottom:1px solid var(--rule);position:sticky;top:0;
background:var(--surface)}
.popdlg h3{margin:0;font-size:13.5px;font-weight:700;color:var(--ink)}
.popx{appearance:none;border:0;background:none;font-size:22px;line-height:1;
color:var(--muted);cursor:pointer;padding:0 2px}
.popx:hover{color:var(--ink)}
.popscroll{overflow:auto;max-height:calc(78vh - 46px);padding:12px 14px}
.popnote{font-size:11.5px;color:var(--muted);margin:0 0 10px}
table.poptbl{font-size:11.5px}
table.poptbl td{padding:3px 8px 3px 0;border-bottom:1px solid var(--rule)}
table.poptbl td:first-child{white-space:nowrap;color:var(--accent);font-weight:600}
.subtabs{display:flex;gap:2px;flex-wrap:wrap;margin:0 0 16px;
border-bottom:1px solid var(--rule)}
.subtab{appearance:none;border:0;background:none;font:inherit;font-size:12.5px;
font-weight:600;color:var(--muted);padding:7px 13px;cursor:pointer;
border-bottom:2px solid transparent;margin-bottom:-1px}
.subtab:hover{color:var(--ink)}
.subtab.on{color:var(--accent);border-bottom-color:var(--accent)}
.sub{display:none}.sub.on{display:block}
.note{font-size:13px;color:var(--muted);background:var(--surface);
border:1px solid var(--rule);border-radius:3px;padding:12px 14px;margin:0 0 24px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 30px}
.card{flex:1 1 150px;background:var(--surface);border:1px solid var(--rule);
border-radius:3px;padding:14px 16px;display:flex;flex-direction:column;gap:2px}
.card b{font-size:25px;font-weight:700;color:var(--ink);letter-spacing:-.02em;
font-variant-numeric:tabular-nums;line-height:1.15}
.card span{font-size:12.5px;font-weight:600;color:var(--body)}
.card em{font-style:normal;font-size:11.5px;color:var(--muted);
font-variant-numeric:tabular-nums}
.scroll{overflow:auto;max-height:600px;border:1px solid var(--rule);border-radius:3px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLIF-donor Results</title>
<style>{css}</style></head><body>
<header><div class="wrap"><h1>CLIF-donor results</h1><div class="tabs">{bar}</div></div></header>
<main><div class="wrap">{panels}</div></main>
{"".join(_POP_DIALOGS)}
<script>
document.querySelectorAll('.tab').forEach(function(b){{b.addEventListener('click',function(){{
document.querySelectorAll('.tab').forEach(function(x){{x.classList.remove('on')}});
document.querySelectorAll('.panel').forEach(function(x){{x.classList.remove('on')}});
b.classList.add('on');var e=document.getElementById('p-'+b.dataset.p);if(e)e.classList.add('on');
window.scrollTo(0,0);}});}});
document.querySelectorAll('.subtab').forEach(function(b){{b.addEventListener('click',function(){{
var g=b.dataset.g;
document.querySelectorAll('.subtab[data-g="'+g+'"]').forEach(function(x){{x.classList.remove('on')}});
document.querySelectorAll('[id^="s-'+g+'-"]').forEach(function(x){{x.classList.remove('on')}});
b.classList.add('on');
var e=document.getElementById('s-'+g+'-'+b.dataset.s);if(e)e.classList.add('on');}});}});
document.querySelectorAll('.popbtn').forEach(function(b){{b.addEventListener('click',function(){{
var d=document.getElementById(b.dataset.dlg);if(d&&d.showModal)d.showModal();}});}});
document.querySelectorAll('.popx').forEach(function(b){{b.addEventListener('click',function(){{
var d=document.getElementById(b.dataset.close);if(d)d.close();}});}});
document.querySelectorAll('.popdlg').forEach(function(d){{d.addEventListener('click',function(ev){{
if(ev.target===d)d.close();}});}});
</script></body></html>"""

    out = Path(a.out) if a.out else sdir / "clif_donor_results.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    wide(pooled, "table2", donor_col=donor_col).write_csv(out.parent / "table1_pooled.csv")
    wide(pooled, "table3").write_csv(out.parent / "table2_pooled.csv")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
