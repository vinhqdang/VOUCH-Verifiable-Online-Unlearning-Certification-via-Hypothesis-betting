#!/usr/bin/env python3
"""Generate the revision-round LaTeX tables from results/*.json.

Writes manuscript/tables/*.tex.  Each file is a self-contained tabular body
meant to be \\input{} inside a table float in main.tex.  Rerunnable as new
results land; every number in the manuscript comes from here.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)

VERD = {"ISSUED": r"\vI", "REVOKED": r"\vR", "UNDETERMINED": r"\vU"}
LBL = {"none": "no unlearning", "retrain": "retrain (fresh adapter)", "ga": "GA",
       "grad_diff": "GradDiff", "npo": "NPO", "simnpo": "SimNPO",
       "npo_weak": r"NPO (25\% budget)",
       "npo_P1_relearn": "NPO + P1 relearn",
       "npo_P2_quant4": "NPO + P2 4-bit",
       "npo_P3_jailbreak": "NPO + P3 jailbreak"}
ORDER = ["none", "retrain", "ga", "grad_diff", "npo", "simnpo",
         "npo_weak", "npo_P1_relearn", "npo_P2_quant4", "npo_P3_jailbreak"]


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)
    print("[tab]", name)


def cp(k, n, conf=0.95):
    """Clopper-Pearson interval, the honest one for counts like 16/18."""
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Dependence: the marginal null is not what the protocol controls
# ---------------------------------------------------------------------------
def tab_dependence():
    d = load("sim_dependence")
    if not d:
        return
    rows = [("iid", "independent"),
            ("strata", "heterogeneous by stratum"),
            ("beta", "model-level over-dispersion"),
            ("latent", "shared model-level latent")]
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "& & \\multicolumn{2}{c}{fixed boundary (v1)} "
            "& \\multicolumn{2}{c}{without replacement (ours)}\\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "sign-generating regime & $\\mathrm{sd}(\\hat\\Delta)$ "
            "& marginal & cohort & marginal & cohort\\\\\n\\midrule\n")
    for key, name in rows:
        r = d[key]
        f = lambda v, bad: (f"\\textbf{{{v:.3f}}}" if bad else f"{v:.3f}")
        body += (f"{name} & {2*r['realised_mean_sd']:.3f} & "
                 f"{f(r['sup_issue_rate_marginal_null'], r['sup_issue_rate_marginal_null']>0.05)} & "
                 f"{f(r['sup_error_rate_cohort_null'], r['sup_error_rate_cohort_null']>0.05)} & "
                 f"{f(r['cohort_issue_rate_marginal_null'], False)}$^{{\\dagger}}$ & "
                 f"{f(r['cohort_error_rate_cohort_null'], r['cohort_error_rate_cohort_null']>0.05)}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("dependence", body)


def tab_cohortnull():
    d = load("sim_cohort_null")
    if not d:
        return
    body = ("\\begin{tabular}{llcccc}\n\\toprule\n"
            "& & \\multicolumn{2}{c}{$m=384$} & \\multicolumn{2}{c}{$m=1024$}\\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "$\\varepsilon$ & cohort structure & ours & fixed & ours & fixed\\\\\n"
            "\\midrule\n")
    names = {"block": "contiguous", "alternating": "spread",
             "stratified": "stratified (rates $0.2$--$0.9$)"}
    for eps in (0.10, 0.20):
        for kind in ("block", "alternating", "stratified"):
            cells = []
            for n in (384, 1024):
                r = d[f"eps={eps}/{kind}/n={n}"]
                cells += [f"{r['cohort_process_error']:.3f}",
                          f"{r['sup_process_error']:.3f}"]
            lead = f"{eps}" if kind == "block" else ""
            body += f"{lead} & {names[kind]} & " + " & ".join(cells) + "\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("cohortnull", body)


# ---------------------------------------------------------------------------
# Valid sequential baselines
# ---------------------------------------------------------------------------
def tab_baselines():
    d = load("sim_baselines")
    if not d:
        return
    names = [("vouch_cohort", "VOUCH, cohort target"),
             ("vouch_sup", "VOUCH, super-population target"),
             ("beta_mixture", "Beta$(\\frac12,\\frac12)$-mixture e-process"),
             ("normal_mixture", "normal-mixture e-process"),
             ("gs_pocock", "group sequential, Pocock"),
             ("gs_obf", "group sequential, O'Brien--Fleming"),
             ("fixed_n", "fixed-$n$ binomial at pre-committed $n$")]
    # Both error statistics for every row: the marginal rate (which is what a
    # super-population procedure controls) and the rate restricted to runs
    # whose realised cohort advantage violates the tolerance (which is what
    # the cohort target controls).  Printing one for VOUCH and the other for
    # the comparators would not be a comparison.
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{type I at the least favourable null} "
            "& \\multicolumn{3}{c}{under exact unlearning}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
            "procedure & marginal & cohort & power & queries "
            "& median $t$\\\\\n\\midrule\n")
    eps, n = 0.10, 1536
    for key, label in names:
        nul = d[f"eps={eps}/null_p0"][key]
        alt = d[f"eps={eps}/exact_unlearning"][key]
        marg = nul["rate"]
        coh = nul["rate_given_cohort_null"]
        fm = (f"\\textbf{{{marg:.3f}}}" if marg > 0.05 + 1e-9 else f"{marg:.3f}")
        body += (f"{label} & {fm} & {coh:.3f} & {alt['rate']:.3f} & "
                 f"{alt['mean_queries']:.0f} & {alt['median_t']:.0f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("baselines", body)


# ---------------------------------------------------------------------------
# Power with censoring made explicit
# ---------------------------------------------------------------------------
def tab_power():
    d = load("sim_power_censoring")
    if not d:
        return
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "& \\multicolumn{3}{c}{super-population target} "
            "& \\multicolumn{2}{c}{cohort target}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "$\\varepsilon$ & median$\\mid$issued & Kaplan--Meier & censored "
            "& median & $\\log(1/\\alpha)/\\mathrm{KL}$\\\\\n\\midrule\n")
    for eps in (0.02, 0.05, 0.10, 0.20):
        s = d[f"sup/eps={eps}/alpha=0.05"]
        c = d[f"cohort/eps={eps}/alpha=0.05"]
        km = s["km_median"]
        kms = "---" if km is None else f"{km:.0f}"
        cens = s["censoring_rate"]
        cs = f"\\textbf{{{cens:.2f}}}" if cens > 0.05 else f"{cens:.2f}"
        body += (f"{eps} & {s['median_over_issued']:.0f} & {kms} & {cs} & "
                 f"{c['median_over_issued']:.0f} & {s['kl_theory']:.0f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("power", body)


# ---------------------------------------------------------------------------
# Benchmarks: eps sweep, both targets, with SimNPO
# ---------------------------------------------------------------------------
BENCH_COLS = [("tofu_gpt2", "TOFU/GPT-2"), ("tofu_pythia160m", "TOFU/Pythia"),
              ("tofu_phi-1_5", "TOFU/Phi-1.5"), ("muse_gpt2_512", "MUSE/GPT-2"),
              ("muse_pythia160m", "MUSE/Pythia"), ("muse_phi-1_5", "MUSE/Phi-1.5")]


def tab_benchmarks():
    d = load("reanalysis_rev")
    if not d:
        return
    util = {}
    for tag, _ in BENCH_COLS:
        rec = d["runs"].get(tag)
        if rec:
            for m, rows in rec["methods"].items():
                u = [r["utility_nll"] for r in rows if r.get("utility_nll")]
                if u:
                    util[(tag, m)] = (float(np.mean(u)), float(np.std(u)))
    body = "\\begin{tabular}{l" + "cc" * len(BENCH_COLS) + "}\n\\toprule\n"
    body += "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}"
                              for _, n in BENCH_COLS) + "\\\\\n"
    for i in range(len(BENCH_COLS)):
        body += f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}"
    body += "\nsubject " + "& verdicts & util. " * len(BENCH_COLS) + "\\\\\n\\midrule\n"
    for m in ORDER:
        row, any_cell = [LBL.get(m, m)], False
        for tag, _ in BENCH_COLS:
            rec = d["runs"].get(tag)
            rows = rec["methods"].get(m) if rec else None
            if not rows:
                row += ["--", "--"]
                continue
            any_cell = True
            v = "/".join(VERD[r["cohort/eps=0.2"]["status"]] for r in rows)
            mu, sd = util.get((tag, m), (float("nan"), 0.0))
            row += [v, f"{mu:.1f}"]
        if any_cell:
            body += " & ".join(row) + "\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("benchmarks", body)


def tab_epssweep():
    """The tolerance sweep the reviewers asked for, both targets."""
    d = load("reanalysis_rev")
    if not d:
        return
    body = ("\\begin{tabular}{llccc|ccc}\n\\toprule\n"
            "& & \\multicolumn{3}{c}{cohort target} "
            "& \\multicolumn{3}{c}{super-population target}\\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "benchmark / model & $m$ & $\\varepsilon{=}0.2$ & $0.1$ & $0.05$ "
            "& $0.2$ & $0.1$ & $0.05$\\\\\n\\midrule\n")
    for tag, name in BENCH_COLS:
        rec = d["runs"].get(tag)
        if not rec:
            continue
        rows = rec["methods"].get("retrain")
        if not rows:
            continue
        cells = []
        for tgt in ("cohort", "sup"):
            for eps in (0.2, 0.1, 0.05):
                cells.append("/".join(VERD[r[f"{tgt}/eps={eps}"]["status"]]
                                      for r in rows))
        body += f"{name} & {rows[0]['m_pairs']} & " + " & ".join(cells) + "\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("epssweep", body)


def tab_counts():
    """Verdict counts with Clopper-Pearson intervals."""
    d = load("reanalysis_rev")
    if not d:
        return
    tally = {}
    for tag, _ in BENCH_COLS:
        rec = d["runs"].get(tag)
        if not rec:
            continue
        for m, rows in rec["methods"].items():
            k, n = tally.get(m, (0, 0))
            want = "REVOKED" if m == "none" else "ISSUED"
            k += sum(1 for r in rows if r["cohort/eps=0.2"]["status"] == want)
            n += len(rows)
            tally[m] = (k, n)
    body = ("\\begin{tabular}{llcc}\n\\toprule\n"
            "subject & target verdict & count & 95\\% CI\\\\\n\\midrule\n")
    for m in ORDER:
        if m not in tally:
            continue
        k, n = tally[m]
        lo, hi = cp(k, n)
        want = "revoked" if m == "none" else "issued"
        body += (f"{LBL.get(m,m)} & {want} & {k}/{n} & "
                 f"$[{lo:.2f},\\,{hi:.2f}]$\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("counts", body)


# ---------------------------------------------------------------------------
# Descriptive-metric head to head
# ---------------------------------------------------------------------------
def tab_descriptive():
    d = load("reanalysis_rev")
    if not d:
        return
    body = ("\\begin{tabular}{llccccc}\n\\toprule\n"
            "& & \\multicolumn{2}{c}{VOUCH} "
            "& \\multicolumn{3}{c}{descriptive metrics}\\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\n"
            "benchmark / model & subject & verdicts & $U_t(\\Delta)$ "
            "& forget-KS & PrivLeak & min-$k$ peek\\\\\n\\midrule\n")
    for tag, name in BENCH_COLS:
        rec = d["runs"].get(tag)
        ds = d["descriptive"].get(tag)
        if not rec or not ds:
            continue
        for m in ("none", "retrain", "grad_diff", "npo"):
            rows = rec["methods"].get(m)
            rr = [x for x in ds if x["method"] == m]
            if not rows or not rr:
                continue
            v = "/".join(VERD[r["cohort/eps=0.2"]["status"]] for r in rows)
            du = np.mean([r["cohort/eps=0.2"]["delta_upper"] for r in rows])
            fq = "/".join("P" if x["forget_quality_verdict"] == "pass" else "F"
                          for x in rr)
            pl = np.mean([x["privleak_pct"] for x in rr])
            pk = "/".join("L" if x["mink_peeking_flags_leak"] else "."
                          for x in rr)
            lead = name if m == "none" else ""
            body += (f"{lead} & {LBL.get(m,m)} & {v} & {du:.2f} & {fq} & "
                     f"{pl:+.1f}\\% & {pk}\\\\\n")
        body += "\\addlinespace\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("descriptive", body)


# ---------------------------------------------------------------------------
# Out-of-class attacks
# ---------------------------------------------------------------------------
def tab_outclass():
    d = load("reanalysis_rev")
    if not d:
        return
    from collections import defaultdict
    agg = defaultdict(list)
    for tag, rec in d["runs"].items():
        for m, rows in rec["methods"].items():
            if m == "none":
                continue
            for r in rows:
                if r["cohort/eps=0.2"]["status"] != "ISSUED":
                    continue
                for k, v in r.get("out_of_class", {}).items():
                    agg[k].append(v)
    names = {"stratum_r8": "stratum-restricted ($r=8$ only)",
             "learned_combo": "cross-fitted learned combination"}
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& & & \\multicolumn{2}{c}{$|\\hat\\Delta| > \\varepsilon$} "
            "& \\multicolumn{2}{c}{anytime CS}\\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "attack & runs & pairs & observed & null & mean $U_t$ "
            "& exceed\\\\\n\\midrule\n")
    for key in ("stratum_r8", "learned_combo"):
        vs = agg.get(key)
        if not vs:
            continue
        ns = np.array([v["n"] for v in vs])
        dh = np.array([abs(v["delta_hat"]) for v in vs])
        # Exact binomial reference, evaluated at each run's own subsample size
        # and averaged: what a *clean* subsample of that size would produce by
        # chance alone.  Without it a raw exceedance count on a 96-pair
        # subsample is uninterpretable.
        def _null(nn):
            kk = np.arange(int(nn) + 1)
            pm = stats.binom.pmf(kk, int(nn), 0.5)
            return float(pm[np.abs(2 * kk / int(nn) - 1) > 0.20].sum())
        null_rate = float(np.mean([_null(x) for x in ns]))
        npair = f"{ns.min()}--{ns.max()}" if ns.min() != ns.max() else f"{ns.min()}"
        # the anytime CS is the object the certificate actually bounds
        up = np.array([max(abs(v["delta_cs"][0]), abs(v["delta_cs"][1]))
                       for v in vs])
        obs = float((dh > 0.20).mean())
        obs_s = (f"\\textbf{{{obs:.3f}}}" if obs > null_rate else f"{obs:.3f}")
        body += (f"{names[key]} & {len(vs)} & ${npair}$ & {obs_s} & "
                 f"{null_rate:.3f} & {up.mean():.3f} & "
                 f"{(up > 0.20).mean():.3f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("outclass", body)


# ---------------------------------------------------------------------------
# Canary detectability
# ---------------------------------------------------------------------------
def tab_detect():
    d = load("detectability_tofu_gpt2")
    if not d:
        return
    names = {"deployed": "deployed (alphanumeric secret)",
             "word_composed": "word-composed secret",
             "entropy_diluted": "entropy-diluted secret"}
    body = ("\\begin{tabular}{lccccccc}\n\\toprule\n"
            "& & & \\multicolumn{2}{c}{excess NLL/token} "
            "& \\multicolumn{2}{c}{perplexity filter} & detector\\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-8}\n"
            "canary template & bits & tokens & predicted & observed "
            "& AUC & recall & recall\\\\\n\\midrule\n")
    for key in ("deployed", "word_composed", "entropy_diluted"):
        v = d["variants"].get(key)
        if not v:
            continue
        pf, td = v["perplexity_filter"], v["trained_detector"]
        rc = pf["recall_at_1pct_fpr"]
        rcs = f"\\textbf{{{rc:.2f}}}" if rc < 0.5 else f"{rc:.2f}"
        body += (f"{names[key]} & {v['secret_bits']:.0f} & "
                 f"{v['mean_tokens']:.0f} & "
                 f"{v['predicted_excess_nll_per_token']:.2f} & "
                 f"{v['observed_excess_nll_per_token']:.2f} & "
                 f"{pf['auc']:.3f} & {rcs} & "
                 f"{td['recall_at_1pct_fpr']:.2f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("detect", body)


# ---------------------------------------------------------------------------
# Model axis, with seed counts made explicit
# ---------------------------------------------------------------------------
def tab_zoo():
    d = load("reanalysis_rev")
    if not d:
        return
    zoo = [("gpt2_v2", "GPT-2", "124M", "2019"),
           ("tiny", "TinyGPT (ours)", "0.9M", "---"),
           ("qwen3_06b", "Qwen3-0.6B", "0.6B", "2025"),
           ("qwen3_4b", "Qwen3-4B-2507", "4.0B", "2025"),
           ("phi4_mini", "Phi-4-mini", "3.8B", "2025"),
           ("gemma4", "Gemma-4-E2B", "5.1B", "2026")]
    body = ("\\begin{tabular}{llccccc}\n\\toprule\n"
            "model & params & year & seeds & no unlearning & retrain & NPO\\\\\n"
            "\\midrule\n")
    for tag, name, size, year in zoo:
        rec = d["runs"].get(tag)
        if not rec:
            continue
        cells = {}
        for m in ("none", "retrain", "npo"):
            rows = rec["methods"].get(m)
            cells[m] = ("/".join(VERD[r["cohort/eps=0.2"]["status"]]
                                 for r in rows) if rows else "--")
        n_rows = max((len(v) for v in rec["methods"].values()), default=0)
        body += (f"{name} & {size} & {year} & {n_rows} & "
                 f"{cells['none']} & {cells['retrain']} & {cells['npo']}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("zoo", body)


def tab_gpt2v2():
    d = load("reanalysis_rev")
    if not d:
        return
    rec = d["runs"].get("gpt2_v2")
    if not rec:
        return
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "subject & verdicts & mean $U_t$ on $\\Delta$ & mean $\\bar D$ "
            "& sd $\\bar D$\\\\\n\\midrule\n")
    for m in ORDER:
        rows = rec["methods"].get(m)
        if not rows:
            continue
        v = "/".join(VERD[r["cohort/eps=0.2"]["status"]] for r in rows)
        du = np.mean([r["cohort/eps=0.2"]["delta_upper"] for r in rows])
        md = [r["mean_loss_diff"] for r in rows if r.get("mean_loss_diff") is not None]
        body += (f"{LBL.get(m,m)} & {v} & {du:.2f} & {np.mean(md):+.2f} & "
                 f"{np.std(md):.2f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("gpt2v2", body)


# ---------------------------------------------------------------------------
# Benchmark forget/retain/capability metrics (TOFU/GPT-2 revision run)
# ---------------------------------------------------------------------------
def tab_metrics():
    d = load("reanalysis_rev")
    if not d:
        return
    rec = d["runs"].get("tofu_gpt2_rev")
    if not rec:
        return
    order = ["none", "retrain", "ga", "grad_diff", "npo", "simnpo",
             "npo_P1_relearn", "npo_P3_jailbreak"]
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& & \\multicolumn{3}{c}{mean NLL} "
            "& \\multicolumn{2}{c}{ROUGE-L recall}\\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\n"
            "subject & verdicts & forget & retain & capability "
            "& forget & retain\\\\\n\\midrule\n")
    for m in order:
        rows = rec["methods"].get(m)
        if not rows:
            continue
        v = "/".join(VERD[r["cohort/eps=0.2"]["status"]] for r in rows)
        g = lambda k: float(np.nanmean([r.get(k) if r.get(k) is not None
                                        else np.nan for r in rows]))
        cap = g("capability_nll")
        # The TOFU P1 relearn corpus *is* the capability probe set
        # (world_facts), so that cell is contaminated by construction.
        caps = "n/a$^{\\ddagger}$" if m == "npo_P1_relearn" else f"{cap:.2f}"
        body += (f"{LBL.get(m,m)} & {v} & {g('forget_nll'):.2f} & "
                 f"{g('retain_nll'):.2f} & {caps} & "
                 f"{g('forget_rouge'):.2f} & {g('retain_rouge'):.2f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("metrics", body)


def tab_tight():
    """The 3,072-pair, r=1 cohort: both targets, three tolerances."""
    d = load("reanalysis_rev")
    if not d:
        return
    rec = d["runs"].get("tofu_gpt2_tight")
    if not rec:
        return
    order = ["none", "retrain", "npo", "simnpo"]
    body = ("\\begin{tabular}{lccc|ccc|c}\n\\toprule\n"
            "& \\multicolumn{3}{c}{cohort target} "
            "& \\multicolumn{3}{c}{super-population target} & \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "subject & $\\varepsilon{=}0.2$ & $0.1$ & $0.05$ "
            "& $0.2$ & $0.1$ & $0.05$ & $\\hat\\Delta$\\\\\n\\midrule\n")
    for m in order:
        rows = rec["methods"].get(m)
        if not rows:
            continue
        cells = []
        for tgt in ("cohort", "sup"):
            for eps in (0.2, 0.1, 0.05):
                cells.append("/".join(VERD[r[f"{tgt}/eps={eps}"]["status"]]
                                      for r in rows))
        dh = ", ".join(f"{2*r['sign_rate']['loss']-1:+.3f}" for r in rows)
        body += f"{LBL.get(m,m)} & " + " & ".join(cells) + f" & {dh}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("tight", body)


def tab_certprod():
    d = load("sim_certprod")
    if not d:
        return
    body = ("\\begin{tabular}{lcc}\n\\toprule\n"
            "history-wide rule & false certification & 95\\% CI\\\\\n"
            "\\midrule\n"
            f"all-pass consensus (ours) & "
            f"{d['all_pass_false_history_certification_rate']:.3f} & "
            f"$[{max(d['all_pass_ci'][0], 0.0):.3f},\\,{d['all_pass_ci'][1]:.3f}]$\\\\\n"
            f"product of certificate e-values & "
            f"\\textbf{{{d['product_false_history_certification_rate']:.3f}}} & "
            f"$[{d['product_ci'][0]:.3f},\\,{d['product_ci'][1]:.3f}]$\\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    write("certprod", body)



# ---------------------------------------------------------------------------
# RMU: a second utility-preserving subject (Section 5.13)
# ---------------------------------------------------------------------------
def _rmu_rows(d):
    """Shared computation for the RMU-tier tables: verdicts at three
    tolerances, realised advantage and mean NLLs, per subject."""
    import sys as _sys
    _sys.path.insert(0, REPO)
    from vouch.verify import VouchConfig, VouchVerifier

    order = ["none", "retrain", "npo", "simnpo", "rmu"]
    lbl = dict(LBL); lbl["rmu"] = "RMU"
    epss = (0.2, 0.1, 0.05)
    rows = []
    for m in order:
        cells = [(r, r["certs"][m]) for r in d if m in r.get("certs", {})]
        if not cells:
            continue
        verds = {e: [] for e in epss}
        deltas, fn, rn, cn = [], [], [], []
        for r, c in cells:
            diffs = c["pair_diffs"]
            names = list(diffs[0].keys())
            deltas.append(2 * np.mean([x["loss"] > 0 for x in diffs]) - 1)
            for e in epss:
                v = VouchVerifier(names, VouchConfig(eps=e, alpha=0.05),
                                  manifest_sha256=r.get("manifest_sha256", ""))
                verds[e].append(VERD[v.run(diffs, shuffle_seed=r["seed"],
                                           early_stop=True).status])
            fn.append(c.get("forget_nll", float("nan")))
            rn.append(c.get("retain_nll", float("nan")))
            cn.append(c.get("capability_nll", float("nan")))
        rows.append((lbl.get(m, m),
                     ["/".join(verds[e]) for e in epss],
                     float(np.mean(deltas)), float(np.nanmean(fn)),
                     float(np.nanmean(rn)), float(np.nanmean(cn))))
    return rows


def _rmu_body(rows):
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{3}{c}{verdict} & & "
            "\\multicolumn{2}{c}{mean NLL}\\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\n"
            "subject & $\\eps{=}0.2$ & $0.1$ & $0.05$ & $\\hat\\Delta$ "
            "& forget & retain\\\\\n\\midrule\n")
    for name, vs, dl, f_, r_, c_ in rows:
        body += (f"{name} & {vs[0]} & {vs[1]} & {vs[2]} & {dl:+.3f} & "
                 f"{f_:.2f} & {r_:.2f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    return body


def tab_rmu():
    """RMU beside the other utility-preserving subjects, co-trained (Sec 5.13)."""
    d = load("lm_e2e_tofu_gpt2_rmu") or load("lm_e2e_tofu_gpt2_rmu_partial")
    if not d:
        return
    rows = _rmu_rows(d)
    if not rows:
        return
    write("rmu", _rmu_body(rows))


def tab_rmu_pythia():
    """RMU generalization: the same co-trained tier on TOFU/Pythia-160M."""
    d = load("lm_e2e_tofu_pythia_rmu")
    if not d:
        return
    rows = _rmu_rows(d)
    if not rows:
        return
    write("rmu_pythia", _rmu_body(rows))


# ---------------------------------------------------------------------------
# A paraphrase-aware score in F (Section 5.14)
# ---------------------------------------------------------------------------
def tab_para():
    """A paraphrase-aware score added to F (Section 5.14)."""
    d = load("paraphrase")
    if not d or not d.get("runs"):
        return
    runs = d["runs"]
    subjects = ["none", "ga", "npo", "retrain"]
    lbl = dict(LBL); lbl["ga"] = "GA"
    body = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{$\\hat\\Delta$} & "
            "\\multicolumn{2}{c}{verdict, $\\eps{=}0.2$} & "
            "\\multicolumn{2}{c}{pairs to certify}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "subject & literal & paraphrase & $\\F$ & "
            "$\\F\\cup\\{s_{\\mathrm{para}}\\}$ & $\\F$ & "
            "$\\F\\cup\\{s_{\\mathrm{para}}\\}$\\\\\n\\midrule\n")
    any_row = False
    for m in subjects:
        cells = [r["subjects"][m] for r in runs if m in r.get("subjects", {})]
        if not cells:
            continue
        any_row = True
        dl = np.mean([2 * c["sign_rate"]["loss"] - 1 for c in cells])
        dp = np.mean([2 * c["sign_rate"]["para"] - 1 for c in cells])
        vd = "/".join(VERD[c["verdicts"]["default/eps=0.2"]["status"]] for c in cells)
        vp = "/".join(VERD[c["verdicts"]["with_para/eps=0.2"]["status"]] for c in cells)
        td = np.mean([c["verdicts"]["default/eps=0.2"]["t_stop"] for c in cells])
        tp = np.mean([c["verdicts"]["with_para/eps=0.2"]["t_stop"] for c in cells])
        body += (f"{lbl.get(m, m)} & {dl:+.3f} & {dp:+.3f} & {vd} & {vp} & "
                 f"{td:.0f} & {tp:.0f}\\\\\n")
    if not any_row:
        return
    body += "\\bottomrule\n\\end{tabular}\n"
    write("para", body)


# ---------------------------------------------------------------------------
# LiRA-style shadow-model attack from outside F (Section 5.12)
# ---------------------------------------------------------------------------
def _lira_body(d):
    lbl = {"none": "no unlearning (positive control)",
           "npo": r"NPO, certified at $\eps=0.2$",
           "retrain": "retrain (negative control)"}
    body = ("\\begin{tabular}{lccccc}\n\\toprule\n"
            "& declared $\\F$ & \\multicolumn{3}{c}{LiRA (outside $\\F$)} & \\\\\n"
            "\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\n"
            "target model & $\\hat\\Delta$ & $\\hat\\Delta$ & 95\\% CI & "
            "agreement & $|\\hat\\Delta| > \\eps$?\\\\\n\\midrule\n")
    for k in ("none", "npo", "retrain"):
        if k not in d["subjects"]:
            continue
        r = d["subjects"][k]
        ci = r["delta_lira_ci"]
        exc = "\\textbf{yes}" if r.get("exceeds_eps=0.2") else "no"
        body += (f"{lbl.get(k, k)} & {r['delta_in_class']:+.3f} & "
                 f"{r['delta_lira']:+.3f} & "
                 f"$[{ci[0]:+.3f},\\,{ci[1]:+.3f}]$ & "
                 f"{r['agreement']:.2f} & {exc}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    return body


def tab_lira():
    d = load("lira_analysis")
    if not d:
        return
    write("lira", _lira_body(d))


def tab_lira_gpt2():
    """LiRA at full scale on the architecture the paper certifies NPO on."""
    d = load("lira_gpt2_analysis")
    if not d:
        return
    write("lira_gpt2", _lira_body(d))


if __name__ == "__main__":
    tab_dependence()
    tab_cohortnull()
    tab_baselines()
    tab_power()
    tab_benchmarks()
    tab_epssweep()
    tab_counts()
    tab_descriptive()
    tab_outclass()
    tab_detect()
    tab_zoo()
    tab_gpt2v2()
    tab_metrics()
    tab_certprod()
    tab_tight()
    tab_rmu()
    tab_rmu_pythia()
    tab_para()
    tab_lira()
    tab_lira_gpt2()
    print("revision tables done")
