#!/usr/bin/env python3
"""Tables and the retention figure for the partial-unlearning, matched-twin and
baseline experiments. Run after make_tables.py: cd extension/manuscript && python3 make_tables_ext.py
"""
import json
import os

import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from union_audit import cell  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
TAB = os.path.join(HERE, "tables")
GRID = ["ga@5", "ga@10", "ga@15", "ga@20", "ga@25", "npo@10", "npo@25", "npo@40", "npo@55", "npo@75"]
EXPS = [("410M, incomplete", "semantic_pythia410m_squeeze.json"),
        ("1B, incomplete", "semantic_pythia1b_squeeze.json"),
        ("410M, literal only", "runA_semantic_literal.json")]
PAIRS = [("literal_cal", "belief_cal", "calibrated vs.\\ calibrated"),
         ("literal", "belief", "raw vs.\\ raw"),
         ("literal", "belief_cal", "raw literal vs.\\ calibrated paraphrase")]


def load(f):
    return json.load(open(os.path.join(RES, f)))


def write(name, body):
    open(os.path.join(TAB, name + ".tex"), "w").write(body)
    print("wrote", name)


def label(s):
    return s if "@" not in s else f"{s.split('@')[0].upper()} {s.split('@')[1]}"


def adv(r, n, k):
    s = r["subjects"][n]
    if "advantage" in s:
        return s["advantage"][k]
    return 2 * np.mean([x[k] > 0 for x in s["pair_diffs"]]) - 1


def runs(f):
    return [r for r in load(f) if all(n in r["subjects"] for n in GRID + ["none"])]


def ms(x):
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


# --- retention tests ---------------------------------------------------------
def retention_tests():
    rows, summary = [], {}
    for name, f in EXPS:
        R = runs(f)
        for a, b, lab in PAIRS:
            Ba = np.mean([adv(r, "none", a) for r in R])
            Bb = np.mean([adv(r, "none", b) for r in R])
            per = np.array([np.mean([adv(r, n, b) / Bb - adv(r, n, a) / Ba for n in GRID]) for r in R])
            m, se = ms(per)
            p = wilcoxon(per).pvalue
            summary[(name, a, b)] = (m, se, int((per > 0).sum()), len(per), p)
            ratio = np.array([np.mean([adv(r, n, b) / adv(r, "none", b) - adv(r, n, a) / adv(r, "none", a)
                                       for n in GRID]) for r in R])
            summary[(name, a, b, "per-seed-ratio")] = (ratio.mean(), ms(ratio)[1], int((ratio > 0).sum()),
                                                      len(ratio), wilcoxon(ratio).pvalue)
            rows.append(f"{name} & {lab} & ${m:+.3f}\\pm{se:.3f}$ & {int((per > 0).sum())}/{len(per)} & {p:.3f} \\\\")
        rows.append("\\addlinespace")
    write("retention_tests", "\\begin{tabular}{llccc}\n\\toprule\n"
          "Experiment & Scores compared & retention gap & seeds $>0$ & $p$ \\\\\n\\midrule\n"
          + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    json.dump({"|".join(k): v for k, v in summary.items()},
              open(os.path.join(RES, "retention_tests.json"), "w"), indent=1, default=float)


# --- retention figure --------------------------------------------------------
def figure():
    styles = {"410M, incomplete": ("blue", "*", "square*", ""),
              "1B, incomplete": ("red", "o", "square", "dashed"),
              "410M, literal only": ("black", "triangle*", "diamond*", "densely dotted")}
    panels = {}
    for pref in ("ga", "npo"):
        plots = []
        for name, f in EXPS:
            R = runs(f)
            col, m1, m2, dash = styles[name]
            for k, mk, tag in (("literal_cal", m1, "literal"), ("belief_cal", m2, "paraphrase")):
                B = np.mean([adv(r, "none", k) for r in R])
                pts = [(0, 1.0)] + [(int(n.split("@")[1]), np.mean([adv(r, n, k) for r in R]) / B)
                                    for n in GRID if n.startswith(pref)]
                opt = f"mark={mk}, {col}" + (f", {dash}" if dash else "") + \
                    (", mark options={fill=white}" if tag == "paraphrase" else "")
                plots.append(f"\\addplot+[{opt}] coordinates {{" +
                             " ".join(f"({a},{b:.3f})" for a, b in pts) + "};")
        panels[pref] = "\n".join(plots)
    legend = ("\\legend{410M incompl.\\ lit., 410M incompl.\\ para., 1B incompl.\\ lit., "
              "1B incompl.\\ para., 410M lit.-only lit., 410M lit.-only para.}")
    body = (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[name=ga, width=0.47\\textwidth, height=0.38\\textwidth, xlabel={GA steps},\n"
        "  ylabel={retention}, ymin=-0.1, ymax=1.05, title={GA}, legend columns=2,\n"
        "  legend style={font=\\tiny, at={(1.08,-0.3)}, anchor=north}]\n"
        + panels["ga"] + "\n" + legend + "\n\\end{axis}\n"
        "\\begin{axis}[at={(ga.east)}, anchor=west, xshift=1.0cm, width=0.47\\textwidth,\n"
        "  height=0.38\\textwidth, xlabel={NPO steps}, ymin=-0.1, ymax=1.05, title={NPO}]\n"
        + panels["npo"] + "\n\\end{axis}\n\\end{tikzpicture}\n")
    write("fig_retention", body)


# --- incomplete forget set: verdict rules ------------------------------------
def incomplete():
    out = []
    U = {m: {r["subject"]: r for r in load(f)["rows"]}
         for m, f in (("410M", "union_audit_pythia410m.json"), ("1B", "union_audit_pythia1b.json"))}
    A = {m: runs(f) for m, f in (("410M", "semantic_pythia410m_squeeze.json"),
                                 ("1B", "semantic_pythia1b_squeeze.json"))}
    for n in ["none"] + GRID:
        cells = [label(n)]
        for m in ("410M", "1B"):
            lc = ms([adv(r, n, "literal_cal") for r in A[m]])[0]
            bc = ms([adv(r, n, "belief_cal") for r in A[m]])[0]
            u = U[m][n]
            cells += [f"${lc:+.3f}$", f"${bc:+.3f}$", f"{u['revoked_literal']:.2f}",
                      f"{u['revoked_union']:.2f}", f"{u['literal_issued_union_revoked']:.2f}"]
        out.append(" & ".join(cells) + " \\\\")
    write("incomplete", "\\begin{tabular}{l ccccc ccccc}\n\\toprule\n"
          " & \\multicolumn{5}{c}{Pythia-410M} & \\multicolumn{5}{c}{Pythia-1B} \\\\\n"
          "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n"
          "Subject & lit. & para. & rev.\\ lit. & rev.\\ union & A & lit. & para. & rev.\\ lit. & rev.\\ union & A \\\\\n"
          "\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def incomplete_stats():
    """Seed-level counts and extremes quoted in the text, stored for reproducibility."""
    out = {}
    for m, f, uf in (("410M", "semantic_pythia410m_squeeze.json", "union_audit_pythia410m.json"),
                     ("1B", "semantic_pythia1b_squeeze.json", "union_audit_pythia1b.json")):
        R = runs(f)
        U = {r["subject"]: r for r in load(uf)["rows"]}
        per_seed = {}
        for n in GRID:
            fr = []
            for r in R:
                res = cell((r["subjects"][n]["pair_diffs"], 0.05, 0.2, 40))
                fr.append(float(np.mean([a == "ISSUED" and b == "REVOKED"
                                         for a, b in zip(res["literal"], res["union"])])))
            per_seed[n] = fr
        light = ["none", "ga@5", "npo@10"]
        out[m] = {"per_seed_literal_issued_union_revoked": per_seed,
                  "seeds_majority": {n: int(sum(x > 0.5 for x in v)) for n, v in per_seed.items()},
                  "light_min_revoked": min(U[n][f"revoked_{u}"] for n in light
                                           for u in ("literal", "union", "mixture")),
                  "npo75_max_revoked": max(U["npo@75"][f"revoked_{u}"]
                                           for u in ("literal", "union", "mixture"))}
    json.dump(out, open(os.path.join(RES, "incomplete_stats.json"), "w"), indent=1)
    print("incomplete:", {m: (v["seeds_majority"]["npo@25"], v["seeds_majority"]["npo@40"],
                              round(v["light_min_revoked"], 3), round(v["npo75_max_revoked"], 3))
                          for m, v in out.items()})


# --- clean design and matched twins ------------------------------------------
def runtable(fname, out_name, subjects, mia_file):
    R = [r for r in load(fname) if all(n in r["subjects"] for n in subjects)]
    an = load(fname.replace(".json", "_analysis.json"))
    mia = {r["subject"]: r for r in load(mia_file)["rows"]}
    lines = []
    for n in subjects:
        lc = ms([adv(r, n, "literal_cal") for r in R])
        bc = ms([adv(r, n, "belief_cal") for r in R])
        ru = an["rules"][n]
        lines.append(f"{label(n)} & ${lc[0]:+.3f}\\pm{lc[1]:.3f}$ & ${bc[0]:+.3f}\\pm{bc[1]:.3f}$ & "
                     f"{ru['revoked_literal']:.2f} & {ru['revoked_union']:.2f} & {ru['issued_union']:.2f} & "
                     f"{mia[n]['mia/organic/forget/loss/auc']:.2f} \\\\")
        if n == "retrain":
            lines.append("\\midrule")
    write(out_name, "\\begin{tabular}{lcccccc}\n\\toprule\n"
          "Subject & literal $\\hat\\Delta$ & paraphrase $\\hat\\Delta$ & rev.\\ lit. & rev.\\ union & "
          "iss.\\ union & organic AUC \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


# --- membership-inference baselines -----------------------------------------
def baselines():
    an = load("runA_semantic_literal_analysis.json")["rules"]
    RA = load("runA_semantic_literal.json")

    def sign_rate(n, k="literal"):
        """Paired fixed-sample sign test on the same pairs and score as VOUCH."""
        return np.mean([binomtest(sum(d[k] > 0 for d in r["subjects"][n]["pair_diffs"]),
                                  len(r["subjects"][n]["pair_diffs"])).pvalue < 0.05 for r in RA])
    mia = {r["subject"]: r for r in load("runA_semantic_literal_mia.json")["rows"]}
    order = ["none", "retrain", "ga@10", "ga@15", "ga@20", "npo@25", "npo@40", "npo@75"]
    lines = []
    for n in order:
        r = mia[n]
        f = lambda k: f"{r[k]:.2f}"
        cells = [f"{an[n]['revoked_literal']:.2f}", f"{an[n]['revoked_union']:.2f}",
                 f"{sign_rate(n):.2f}"]
        cells += [f(f"mia/literal/matched/{x}/leak") for x in ("loss", "mink", "zlib", "ref")]
        cells += [f(f"mia/literal/unmatched/{x}/leak") for x in ("loss", "ref")]
        cells += [f(f"mia/paraphrase/matched/{x}/leak") for x in ("loss", "ref")]
        cells += [f(f"mia/organic/forget/{x}/leak") for x in ("loss", "ref")]
        lines.append(f"{label(n)} & " + " & ".join(cells) + " \\\\")
        if n == "retrain":
            lines.append("\\midrule")
    write("baselines", "\\begin{tabular}{l ccc cccc cc cc cc}\n\\toprule\n"
          " & \\multicolumn{2}{c}{VOUCH} & sign & \\multicolumn{4}{c}{(a) canaries vs.\\ ghosts}"
          " & \\multicolumn{2}{c}{(b) vs.\\ organic} & \\multicolumn{2}{c}{(c) paraphrase}"
          " & \\multicolumn{2}{c}{(d) organic} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\n"
          "Subject & lit. & union & lit. & loss & Min-K\\% & zlib & ref & loss & ref & loss & ref & loss & ref \\\\\n"
          "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("AUC (a) loss:", {n: round(mia[n]["mia/literal/matched/loss/auc"], 3) for n in order})
    print("AUC (b) loss:", {n: round(mia[n]["mia/literal/unmatched/loss/auc"], 3) for n in order})


if __name__ == "__main__":
    retention_tests()
    incomplete_stats()
    figure()
    incomplete()
    runtable("runA_semantic_literal.json", "clean",
             ["none", "retrain", "ga@5", "ga@10", "ga@15", "ga@20", "ga@25",
              "npo@10", "npo@25", "npo@40", "npo@55", "npo@75"], "runA_semantic_literal_mia.json")
    runtable("runB_matched_literal.json", "matched",
             ["none", "retrain", "ga@10", "ga@15", "ga@20", "ga@25",
              "npo@25", "npo@40", "npo@55", "npo@75"], "runB_matched_literal_mia.json")
    baselines()
