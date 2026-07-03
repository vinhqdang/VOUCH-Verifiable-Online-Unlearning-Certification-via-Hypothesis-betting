#!/usr/bin/env python3
"""Generate LaTeX tables for the manuscript from results/*.json.

Writes paper/tables/*.tex, each a self-contained tabular body meant to be
\\input{} inside a table float in main.tex.  Rerunnable as new results land.
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
OUT = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(body)
    print("[tab]", name)


ORDER = ["none", "retrain", "ga", "grad_diff", "npo",
         "npo_P1_relearn", "npo_P3_jailbreak"]
LBL = {"none": r"no unlearning", "retrain": r"retrain (exact)", "ga": "GA",
       "grad_diff": "GradDiff", "npo": "NPO",
       "npo_P1_relearn": r"NPO + P1 relearn",
       "npo_P3_jailbreak": r"NPO + P3 jailbreak"}
VERD = {"ISSUED": r"\vI", "REVOKED": r"\vR", "UNDETERMINED": r"\vU"}


# ---- Table: validity under optional stopping (M1) --------------------------
def tab_validity():
    v = load("sim_validity")
    if not v:
        return
    rows = [
        (r"VOUCH certificate (boundary null $\Delta=\varepsilon$)",
         v["false_cert_rate/eps=0.1/alpha=0.05"], v["false_cert_rate/eps=0.1/alpha=0.01"]),
        (r"VOUCH revocation (exact unlearning)",
         v["false_revocation_rate/alpha=0.05"], v["false_revocation_rate/alpha=0.01"]),
        (r"binomial w/ peeking --- false certification",
         v["peeking_binomial_false_cert/alpha=0.05"], v["peeking_binomial_false_cert/alpha=0.01"]),
        (r"binomial w/ peeking --- false alarm",
         v["peeking_binomial_false_alarm/alpha=0.05"], v["peeking_binomial_false_alarm/alpha=0.01"]),
    ]
    body = "\\begin{tabular}{lcc}\n\\toprule\nprocedure & $\\alpha=0.05$ & $\\alpha=0.01$\\\\\n\\midrule\n"
    for name, a5, a1 in rows:
        f5 = f"\\textbf{{{a5:.3f}}}" if a5 > 0.05 else f"{a5:.3f}"
        f1 = f"\\textbf{{{a1:.3f}}}" if a1 > 0.01 else f"{a1:.3f}"
        body += f"{name} & {f5} & {f1}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("validity", body)


# ---- Table: power vs KL limit (M2) -----------------------------------------
def tab_power():
    p = load("sim_power")
    if not p:
        return
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n"
            "& \\multicolumn{2}{c}{$\\alpha=0.05$} & \\multicolumn{2}{c}{$\\alpha=0.01$}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "$\\varepsilon$ & median $\\tau^*$ & $\\log(1/\\alpha)/\\mathrm{KL}$ "
            "& median $\\tau^*$ & $\\log(1/\\alpha)/\\mathrm{KL}$\\\\\n\\midrule\n")
    for eps in (0.02, 0.05, 0.1, 0.2):
        r5 = p[f"tau/eps={eps}/alpha=0.05"]
        r1 = p[f"tau/eps={eps}/alpha=0.01"]
        star5 = "$^*$" if r5["issued_frac"] < 0.99 else ""
        star1 = "$^*$" if r1["issued_frac"] < 0.99 else ""
        body += (f"{eps} & {r5['median']:.0f}{star5} & {r5['kl_theory']:.0f} & "
                 f"{r1['median']:.0f}{star1} & {r1['kl_theory']:.0f}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("power", body)


# ---- Table: benchmark verdicts (2 models x 2 benchmarks) -------------------
def _verdicts(tag, m):
    runs = load(f"lm_e2e_{tag}")
    if not runs:
        return None, None
    cs = [r["certs"][m] for r in runs if m in r.get("certs", {})]
    if not cs:
        return None, None
    v = "/".join(VERD[c["status"]] for c in cs)
    u = np.mean([c["utility_nll"] for c in cs])
    return v, u


def tab_benchmarks():
    cols = [("tofu_gpt2_rescored", "TOFU/GPT-2"),
            ("tofu_pythia160m_rescored", "TOFU/Pythia"),
            ("tofu_phi-1_5_rescored", "TOFU/Phi-1.5"),
            ("muse_gpt2_512_rescored", "MUSE/GPT-2"),
            ("muse_pythia160m", "MUSE/Pythia"),
            ("muse_phi-1_5_rescored", "MUSE/Phi-1.5")]
    body = "\\begin{tabular}{l" + "cc" * len(cols) + "}\n\\toprule\n"
    body += "& " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in cols) + "\\\\\n"
    for i in range(len(cols)):
        body += f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}"
    body += "\nsubject " + "& verdicts & util. " * len(cols) + "\\\\\n\\midrule\n"
    for m in ORDER:
        row = [LBL[m]]
        for tag, _ in cols:
            v, u = _verdicts(tag, m)
            row += [v or "--", f"{u:.1f}" if u else "--"]
        body += " & ".join(row) + "\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("benchmarks", body)


# ---- Table: streaming (M6) --------------------------------------------------
def tab_streaming():
    s = load("sim_streaming")
    if not s:
        return
    body = ("\\begin{tabular}{lccc}\n\\toprule\n"
            "deletion history & per-wave FWER & global alarm & global cert.\\\\\n\\midrule\n"
            f"all 10 waves exact & {s['all_exact/fwer_per_wave_alarms']:.3f} & "
            f"{s['all_exact/global_rev_alarm_rate']:.3f} & "
            f"{s['all_exact/global_certified_rate_allpass']:.2f}\\\\\n"
            f"one bad wave ($\\Delta=0.25$) & {s['one_bad/fwer_per_wave_alarms']:.3f}"
            f"$^{{\\dagger}}$ & {s['one_bad/global_rev_alarm_rate']:.3f} & "
            f"\\textbf{{{s['one_bad/global_certified_rate_allpass']:.3f}}}\\\\\n"
            f"ten weak waves ($\\Delta=0.06$) & "
            f"{s['all_weak/any_per_wave_alarm_rate']:.3f} & "
            f"\\textbf{{{s['all_weak/global_rev_alarm_rate']:.3f}}} & "
            f"{s['all_weak/global_certified_rate_allpass']:.3f}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    write("streaming", body)


# ---- Table: soundness of composition (I1) -----------------------------------
def tab_soundness():
    s = load("sim_soundness")
    if not s:
        return
    body = ("\\begin{tabular}{lc}\n\\toprule\n"
            "certificate arm & false-certification rate\\\\\n\\midrule\n"
            f"per-score minimum (ours) & {s['per_score_min_false_cert_rate']:.3f}\\\\\n"
            f"adaptive-mixture sign (design v1.0) & "
            f"\\textbf{{{s['doc_mixture_false_cert_rate']:.3f}}}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n")
    write("soundness", body)


# ---- Table: GPT-2 synthetic tier (v2) ---------------------------------------
def tab_gpt2v2():
    runs = load("lm_e2e_gpt2_v2")
    if not runs:
        return
    extra = {"npo_weak": "NPO (25\\% budget)", "npo_P2_quant4": "NPO + P2 4-bit"}
    order = ["none", "retrain", "ga", "grad_diff", "npo", "npo_weak",
             "npo_P1_relearn", "npo_P2_quant4", "npo_P3_jailbreak"]
    body = ("\\begin{tabular}{lccc}\n\\toprule\n"
            "subject & verdicts (3 seeds) & mean $U_t$ on $\\Delta$ & mean $\\bar D$\\\\\n\\midrule\n")
    for m in order:
        cs = [r["certs"][m] for r in runs if m in r.get("certs", {})]
        if not cs:
            continue
        v = "/".join(VERD[c["status"]] for c in cs)
        dU = np.mean([c["delta_upper"] for c in cs])
        md = np.mean([c["mean_loss_diff"] for c in cs])
        body += f"{LBL.get(m, extra.get(m, m))} & {v} & {dU:.2f} & {md:+.2f}\\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("gpt2v2", body)


# ---- Table: model zoo (as results land) -------------------------------------
def tab_zoo():
    zoo = [("gpt2_v2", "GPT-2", "124M", "2019"),
           ("tiny", "TinyGPT (ours)", "0.9M", "---"),
           ("qwen3_06b", "Qwen3-0.6B", "0.6B", "2025"),
           ("qwen3_4b", "Qwen3-4B-2507", "4.0B", "2025"),
           ("nemotron3_4b", "Nemotron-3-Nano", "4.0B", "2026"),
           ("gemma4", "Gemma-4-E2B", "5.1B", "2026")]
    rows = []
    for tag, name, size, year in zoo:
        runs = load(f"lm_e2e_{tag}")
        if not runs:
            continue
        cells = {}
        for m in ("none", "retrain", "npo"):
            cs = [r["certs"][m] for r in runs if m in r.get("certs", {})]
            cells[m] = "/".join(VERD[c["status"]] for c in cs) if cs else "--"
        rows.append((name, size, year, cells))
    if not rows:
        return
    body = ("\\begin{tabular}{llcccc}\n\\toprule\n"
            "model & params & year & no unlearning & retrain & NPO\\\\\n\\midrule\n")
    for name, size, year, c in rows:
        body += (f"{name} & {size} & {year} & {c['none']} & {c['retrain']} & "
                 f"{c['npo']}\\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write("zoo", body)


if __name__ == "__main__":
    tab_validity()
    tab_power()
    tab_benchmarks()
    tab_streaming()
    tab_soundness()
    tab_gpt2v2()
    tab_zoo()
    print("tables done")
