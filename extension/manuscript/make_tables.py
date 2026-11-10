#!/usr/bin/env python3
"""Generate every table of the extension manuscript from extension/results/*.json.

Usage: cd extension/manuscript && python3 make_tables.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)


def load(name):
    return json.load(open(os.path.join(RES, name)))


def write(name, body):
    open(os.path.join(TAB, name + ".tex"), "w").write(body)
    print("wrote", name)


def label(subj):
    if subj == "none":
        return "none"
    if subj == "retrain":
        return "retrain"
    m, n = subj.split("@")
    return f"{m.upper()} {n}"


def pm(x, s):
    return f"${x:+.3f}\\pm{s:.3f}$"


# --- certification time ---------------------------------------------------
def growth():
    rows = []
    for r in load("cohort_growth_check.json"):
        rows.append(f"{r['m']} & {r['eps']:.1f} & {r['median_tau_cohort']:.0f} & "
                    f"{r['first_t_bound_le_half']} & "
                    f"{r['first_t_bound_le_half'] / r['median_tau_cohort']:.1f} & "
                    f"{r['median_tau_fixed_boundary']:.0f} & "
                    f"{r['bound_violations']}/{r['grid_points']} \\\\")
    write("growth", "\\begin{tabular}{rrrrrrr}\n\\toprule\n"
          "$m$ & $\\eps$ & median $\\tau$ & bound $\\le\\frac12$ & ratio & fixed boundary & violations \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# --- detectability ---------------------------------------------------------
def detect():
    d = load("detectability.json")
    names = {"vouch_qa": "VOUCH random-string QA", "semantic_literal": "semantic twin, literal",
             "semantic_qa": "semantic twin, question--answer", "matched": "organic-matched twin"}
    rows = []
    for k, nm in names.items():
        r = d["designs"][k]
        rows.append(f"{nm} & {r['perplexity']['auc']:.3f} & {r['perplexity']['adv']:.3f} & "
                    f"{r['ngram']['auc']:.3f} & {r['ngram']['adv']:.3f} & "
                    f"{r['gamma_lower_bound']:.3f} \\\\")
    write("detect", "\\begin{tabular}{lccccc}\n\\toprule\n"
          " & \\multicolumn{2}{c}{Perplexity features} & \\multicolumn{2}{c}{Character $n$-grams} & \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "Canary design & AUC & advantage & AUC & advantage & $\\gamma\\ge$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# --- full-forget semantic audit -------------------------------------------
def semantic():
    R = load("semantic_pythia410m.json")
    lab = {"none": "no unlearning", "ga": "GA (100 steps)", "npo": "NPO (250 steps)",
           "retrain": "retrain"}
    abbrev = {"ISSUED": "I", "REVOKED": "R", "UNDETERMINED": "U"}
    rows = []
    for m, nm in lab.items():
        S = [r["subjects"][m] for r in R]
        lit = ", ".join(f"${s['advantage']['literal_cal']:+.2f}$" for s in S)
        par = ", ".join(f"${s['advantage']['belief_cal']:+.2f}$" for s in S)
        v2 = " ".join(abbrev[s["verdicts"]["all/eps=0.2"]["status"]] for s in S)
        v1 = " ".join(abbrev[s["verdicts"]["all/eps=0.1"]["status"]] for s in S)
        rows.append(f"{nm} & {lit} & {par} & {v2} & {v1} \\\\")
    write("semantic", "\\begin{tabular}{lllcc}\n\\toprule\n"
          "Subject & literal $\\hat\\Delta$ & paraphrase $\\hat\\Delta$ & $\\eps=0.2$ & $\\eps=0.1$ \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# --- squeeze sweeps ----------------------------------------------------------
def squeeze(fname, out, with_counts):
    S = load(fname)["rows"]
    rows = []
    for r in S:
        if r["n"] < max(x["n"] for x in S):
            continue
        p = "--" if r["subject"] == "none" else f"{r['ret_gap_p']:.3f}"
        line = (f"{label(r['subject'])} & {pm(r['literal'], r['literal_se'])} & "
                f"{pm(r['para_cal'], r['para_cal_se'])} & "
                f"{r['ret_lit']:.2f} / {r['ret_para']:.2f} & {p}")
        if with_counts:
            line += f" & {r['squeeze']} & {r['reverse']}"
        rows.append(line + " \\\\")
    cols = "lcccc" + ("cc" if with_counts else "")
    head = ("Subject & literal $\\hat\\Delta$ & paraphrase $\\hat\\Delta$ & retention lit.\\,/\\,para. & $p$"
            + (" & A & B" if with_counts else "") + " \\\\\n")
    write(out, f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n" + head + "\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def union():
    sets = [("410M", "union_audit_pythia410m.json", ["ga@15", "ga@20", "npo@25", "npo@40"]),
            ("1B", "union_audit_pythia1b.json", ["ga@15", "ga@20", "npo@40", "npo@55"])]
    rows = []
    for mname, f, keep in sets:
        R = {r["subject"]: r for r in load(f)["rows"]}
        for k in keep:
            r = R[k]
            rows.append(f"{mname} & {label(k)} & {r['revoked_literal']:.2f} & "
                        f"{r['revoked_union']:.2f} & {r['revoked_mixture']:.2f} & "
                        f"{r['literal_issued_union_revoked']:.2f} \\\\")
    write("union", "\\begin{tabular}{llcccc}\n\\toprule\n"
          "Model & Subject & literal & union & mixture & lit.\\ issued, union revoked \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def union_extremes():
    """Agreement outside the partial region, reported in the text."""
    out = {}
    for mname, f in [("410M", "union_audit_pythia410m.json"), ("1B", "union_audit_pythia1b.json")]:
        R = {r["subject"]: r for r in load(f)["rows"]}
        light = [s for s in ("none", "ga@5", "npo@10") if s in R]
        strong = [s for s in ("ga@50", "ga@100", "npo@75", "npo@250") if s in R]
        out[mname] = {
            "light_min_revoked": min(min(R[s][f"revoked_{u}"] for u in ("literal", "union", "mixture"))
                                     for s in light),
            "strong_max_revoked": max(max(R[s][f"revoked_{u}"] for u in ("literal", "union", "mixture"))
                                      for s in strong)}
    print("union extremes:", out)


def retention_coords():
    for f in ("squeeze_summary.json", "squeeze_summary_pythia1b.json"):
        S = load(f)["rows"]
        for pref in ("ga", "npo"):
            for key in ("ret_lit", "ret_para"):
                pts = [(0, 1.0)] + [(int(r["subject"].split("@")[1]), r[key]) for r in S
                                    if r["subject"].startswith(pref) and r["n"] == max(x["n"] for x in S)]
                print(f, pref, key, " ".join(f"({a},{b:.3f})" for a, b in pts))


def anytime():
    d = load("anytime_cost.json")
    rows = [f"{r['delta']:.2f} & {r['kl']:.4f} & {r['wald']:.0f} & {r['oracle_upper']:.0f} & "
            f"{r['sim_oracle']:.0f} & {r['sim_grid']:.0f} & {r['n50']} & {r['n80']} & {r['n90']} \\\\"
            for r in d["rows"]]
    write("anytime", "\\begin{tabular}{rrrrrrrrr}\n\\toprule\n"
          "$\\Delta$ & KL & Wald & oracle $\\le$ & oracle sim. & grid sim. & $n_{50}$ & $n_{80}$ & $n_{90}$ \\\\\n"
          "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def selective():
    d = load("selective_provider.json")
    names = {"vouch_qa": "VOUCH random-string QA", "semantic_literal": "semantic twin, literal",
             "semantic_qa": "semantic twin, question--answer", "matched": "organic-matched twin"}
    rows = []
    for k, nm in names.items():
        for det, dn in (("perplexity", "perplexity"), ("ngram", "$n$-gram")):
            r = d["designs"][k][det]
            rows.append(f"{nm} & {dn} & " + " & ".join(
                f"{r[str(e)]['canary_miss']:.2f} / {r[str(e)]['organic_leak']:.2f}" for e in d["eps"]) + " \\\\")
    write("selective", "\\begin{tabular}{llccc}\n\\toprule\n"
          "Canary design & detector & $\\eps=0.05$ & $\\eps=0.10$ & $\\eps=0.20$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    selective()
    anytime()
    growth()
    detect()
    semantic()
    # partial-unlearning, matched-twin and baseline tables: make_tables_ext.py
