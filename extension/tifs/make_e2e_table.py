#!/usr/bin/env python3
"""Table of the end-to-end selective-provider experiment (tables/e2e.tex)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
order = [("none", "none"), ("retrain", "retrain"), ("honest@75", "honest, 75"),
         ("selective@75", "selective, 75"), ("honest@150", "honest, 150"),
         ("selective@150", "selective, 150")]


def p(x):
    if x < 1e-3:
        m, e = f"{x:.0e}".split("e")
        return f"${m}\\cdot10^{{{int(e)}}}$"
    return f"{x:.2f}"


out = ["\\begin{tabular}{lcccccc}", "\\toprule",
       " & \\multicolumn{2}{c}{Canary cohort} & \\multicolumn{3}{c}{Organic probes} & \\\\",
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
       "Subject & $\\hat\\Delta$ & ISSUED & $\\hat\\Delta$ & $p$ & REVOKED & AUC \\\\"]
for f, lab in (("selective_e2e_analysis.json", "probes repeated $4$ times"),
               ("selective_e2e_rep8_analysis.json", "probes repeated $8$ times")):
    A = json.load(open(os.path.join(RES, f)))
    rows = {r["subject"]: r for r in A["rows"]}
    out += ["\\midrule", f"\\multicolumn{{7}}{{l}}{{\\emph{{{lab}}}}} \\\\"]
    for k, name in order:
        r = rows[k]
        pc, pp = A["pooled"][k + "/canary"], A["pooled"][k + "/probe"]
        out.append(f"{name} & ${pc['adv']:+.3f}$ & {r['canary_issued']:.2f} & "
                   f"${pp['adv']:+.3f}$ & {p(pp['p_two_sided'])} & {r['probe_revoked']:.2f} & "
                   f"{r['organic_auc_loss']:.2f} \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tables", "e2e.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
