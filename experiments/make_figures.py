#!/usr/bin/env python3
"""Publication figures for the VOUCH experiments.

Reads results/*.json produced by run_simulation.py / run_lm_e2e.py and
writes PNG+PDF figures into results/figures/.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

# categorical palette (fixed slot order), text/grid tokens
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1a1a19", "#5f5e56", "#e5e4dc"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(name):
    path = os.path.join(RESULTS, name + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def saveall(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- validity --
def fig_validity():
    v = load("sim_validity")
    if not v:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    alphas = v["calibration/alphas"]
    realized = v["calibration/realized"]
    ax.plot([0, 0.22], [0, 0.22], ls="--", lw=1, color=MUTED, zorder=1)
    ax.plot(alphas, realized, "o-", lw=2, ms=5, color=C[0], zorder=3)
    ax.annotate("nominal = realized", (0.145, 0.155), rotation=38,
                fontsize=8, color=MUTED)
    ax.set_xlabel(r"nominal level $\alpha$")
    ax.set_ylabel("realized false-certification rate")
    ax.set_title("VOUCH is conservative under\nadversarial optional stopping", fontsize=9.5)

    ax = axes[1]
    labels = ["VOUCH\ncert", "VOUCH\nrevocation",
              "binomial\npeeking\n(cert)", "binomial\npeeking\n(alarm)"]
    vals = [v["false_cert_rate/eps=0.1/alpha=0.05"],
            v["false_revocation_rate/alpha=0.05"],
            v["peeking_binomial_false_cert/alpha=0.05"],
            v["peeking_binomial_false_alarm/alpha=0.05"]]
    cols = [C[0], C[0], C[5], C[5]]
    bars = ax.bar(range(4), vals, color=cols, width=0.62, zorder=3)
    ax.axhline(0.05, color=INK, lw=1, ls="--", zorder=4)
    ax.annotate(r"$\alpha = 0.05$", (3.45, 0.062), fontsize=8.5, color=INK,
                ha="right")
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.3f}", (b.get_x() + b.get_width() / 2, val + 0.008),
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4), labels, fontsize=7.8)
    ax.set_ylabel("type-I error (2000 seeds)")
    ax.set_title("Anytime validity vs fixed-$n$ peeking", fontsize=9.5)
    saveall(fig, "fig1_validity")


# --------------------------------------------------------------- soundness --
def fig_soundness():
    s = load("sim_soundness")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    vals = [s["per_score_min_false_cert_rate"], s["doc_mixture_false_cert_rate"]]
    bars = ax.bar([0, 1], vals, color=[C[0], C[5]], width=0.55, zorder=3)
    ax.axhline(0.05, color=INK, lw=1, ls="--")
    ax.annotate(r"$\alpha=0.05$", (1.35, 0.07), fontsize=8.5, ha="right")
    for b, val in zip(bars, vals):
        ax.annotate(f"{val:.3f}", (b.get_x() + b.get_width() / 2, val + 0.02),
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1], ["per-score min\n(VOUCH, sound)",
                           "mixture-sign arm\n(design doc v1.0)"], fontsize=8.5)
    ax.set_ylabel("false-certification rate")
    ax.set_title("Certificate arm under composite null\n(one leaking score among three)",
                 fontsize=9.5)
    saveall(fig, "fig2_soundness")


# ------------------------------------------------------------------- power --
def fig_power():
    p = load("sim_power")
    if not p:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    eps_grid = [0.02, 0.05, 0.10, 0.20]
    for j, alpha in enumerate((0.05, 0.01)):
        med = [p[f"tau/eps={e}/alpha={alpha}"]["median"] for e in eps_grid]
        ax.plot(eps_grid, med, "o-", lw=2, ms=5, color=C[j],
                label=rf"VOUCH median $\tau^*$, $\alpha={alpha}$")
        theory = [p[f"tau/eps={e}/alpha={alpha}"]["kl_theory"] for e in eps_grid]
        ax.plot(eps_grid, theory, ls=":", lw=1.5, color=C[j],
                label=rf"$\log(1/\alpha)/\mathrm{{KL}}$, $\alpha={alpha}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(eps_grid, [str(e) for e in eps_grid], minor=False)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"certified advantage bound $\varepsilon$")
    ax.set_ylabel("pairs to certification")
    ax.legend(fontsize=7.2, frameon=False)
    ax.set_title("Certification time under exact unlearning", fontsize=9.5)

    ax = axes[1]
    dgrid = [0.05, 0.10, 0.20, 0.40]
    med = [p[f"detect/delta={d}"]["median"] for d in dgrid]
    ax.plot(dgrid, med, "o-", lw=2, ms=5, color=C[5], label="median detection time")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(dgrid, [str(d) for d in dgrid], minor=False)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"true residual advantage $\Delta$")
    ax.set_ylabel("pairs to revocation")
    ax.set_title("Revocation detection time", fontsize=9.5)
    saveall(fig, "fig3_power")


# --------------------------------------------------------------- tightness --
def fig_tightness():
    tdata = load("sim_tightness")
    if not tdata:
        return
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    for j, p in enumerate((0.5, 0.6)):
        cps = tdata[f"p={p}/checkpoints"]
        ups = tdata[f"p={p}/median_upper"]
        ax.plot(cps, [2 * u - 1 for u in ups], "o-", lw=2, ms=4, color=C[j],
                label=rf"median $U_t$, true $\Delta={2*p-1:.1f}$")
        ax.axhline(2 * p - 1, color=C[j], ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("pairs observed $t$")
    ax.set_ylabel(r"CS upper bound on $\Delta$")
    ax.legend(fontsize=7.8, frameon=False)
    ax.set_title("Advantage confidence sequence tightness", fontsize=9.5)
    saveall(fig, "fig4_tightness")


# ---------------------------------------------------------------- ablation --
def fig_ablation():
    a = load("sim_ablation")
    if not a:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    strategies = ["ons", "mixture", "agrapa", "kt", "fixed"]
    med = [a[f"strategy/{s}/median_tau"] for s in strategies]
    frac = [a[f"strategy/{s}/issued_frac"] for s in strategies]
    bars = ax.bar(range(len(strategies)), med, color=C[0], width=0.6, zorder=3)
    for i, (b, f) in enumerate(zip(bars, frac)):
        note = f"issued\n{f:.0%}" if f < 0.99 else ""
        if note:
            b.set_color(C[5])
            ax.annotate(note, (i, (b.get_height() or 1) + 40), ha="center",
                        fontsize=7.5, color=C[5])
    ax.set_xticks(range(len(strategies)),
                  ["ONS", "mixture", "aGRAPA", "KT", "fixed $\\lambda$"], fontsize=8.5)
    ax.set_ylabel(r"median $\tau^*$ (pairs)")
    ax.set_title(r"Betting strategy ($\varepsilon=0.1$, exact unlearning)",
                 fontsize=9.5)

    ax = axes[1]
    fracs = [0.05, 0.15]
    x = np.arange(len(fracs))
    sm = [a[f"mag_rev/frac={f}/sign_median"] for f in fracs]
    mm = [a[f"mag_rev/frac={f}/mag_median"] for f in fracs]
    w = 0.34
    ax.bar(x - w / 2 - 0.01, sm, width=w, color=C[0], label="sign-only", zorder=3)
    ax.bar(x + w / 2 + 0.01, mm, width=w, color=C[1], label="magnitude-aware (VOUCH+)", zorder=3)
    for xi, (s, m) in enumerate(zip(sm, mm)):
        ax.annotate(f"{s:.0f}", (xi - w / 2 - 0.01, s + 30), ha="center", fontsize=8)
        ax.annotate(f"{m:.0f}", (xi + w / 2 + 0.01, m + 30), ha="center", fontsize=8)
    ax.set_xticks(x, [f"{f:.0%} memorized" for f in fracs], fontsize=8.5)
    ax.set_ylabel("median detection time (pairs)")
    ax.legend(fontsize=7.8, frameon=False)
    ax.set_title("Sparse-leakage detection", fontsize=9.5)
    saveall(fig, "fig5_ablation")


# ---------------------------------------------------------------- LM tiers --
def fig_lm(tag="gpt2", fname="fig6_lm_gpt2"):
    runs = load(f"lm_e2e_{tag}")
    if not runs:
        return
    methods, dU, logrev, statuses = [], [], [], []
    order = ["none", "npo_weak", "ga", "grad_diff", "npo", "retrain",
             "npo_P1_relearn", "npo_P2_quant4", "npo_P3_jailbreak"]
    labels = {"none": "no unlearning", "npo_weak": "NPO 25%", "ga": "GA",
              "grad_diff": "GradDiff", "npo": "NPO", "retrain": "retrain",
              "npo_P1_relearn": "NPO+P1\nrelearn", "npo_P2_quant4": "NPO+P2\n4-bit",
              "npo_P3_jailbreak": "NPO+P3\njailbreak"}
    for m in order:
        vals = [r["certs"][m] for r in runs if m in r.get("certs", {})]
        if not vals:
            continue
        methods.append(labels[m])
        dU.append(np.mean([v["delta_upper"] for v in vals]))
        logrev.append(np.mean([v["log_e_rev"] for v in vals]))
        st = [v["status"] for v in vals]
        statuses.append(max(set(st), key=st.count))
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    colmap = {"ISSUED": C[1], "REVOKED": C[5], "UNDETERMINED": C[2]}
    cols = [colmap[s] for s in statuses]
    bars = ax.bar(range(len(methods)), dU, color=cols, width=0.62, zorder=3)
    for b, s in zip(bars, statuses):
        ax.annotate(s.lower(), (b.get_x() + b.get_width() / 2, b.get_height() + 0.03),
                    ha="center", fontsize=7.3, color=INK)
    ax.set_xticks(range(len(methods)), methods, fontsize=8)
    ax.set_ylabel(r"mean CS upper bound on $\Delta$")
    ax.set_ylim(0, 1.18)
    ax.set_title(f"End-to-end certification of unlearning subjects ({tag})", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colmap[k]) for k in
               ("ISSUED", "REVOKED", "UNDETERMINED")]
    ax.legend(handles, ["certificate issued", "revoked", "undetermined"],
              fontsize=7.8, frameon=False, ncol=3, loc="upper left")
    saveall(fig, fname)


if __name__ == "__main__":
    fig_validity()
    fig_soundness()
    fig_power()
    fig_tightness()
    fig_ablation()
    fig_lm("gpt2_v1", "fig6_lm_gpt2")
    fig_lm("tiny", "fig7_lm_tiny")
    print("figures done")
