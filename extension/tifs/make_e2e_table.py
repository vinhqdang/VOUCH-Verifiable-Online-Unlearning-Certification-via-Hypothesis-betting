#!/usr/bin/env python3
"""Table of the end-to-end selective-provider experiment (tables/e2e.tex)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "results", "selective_e2e_analysis.json")))
rows = {r["subject"]: r for r in A["rows"]}
order = [("none", "none"), ("retrain", "retrain"), ("honest@75", "honest, 75"),
         ("selective@75", "selective, 75"), ("honest@150", "honest, 150"),
         ("selective@150", "selective, 150")]


def p(x):
    if x < 1e-3:
        m, e = f"{x:.0e}".split("e")
        return f"${m}\\cdot10^{{{int(e)}}}$"
    return f"{x:.2f}"


out = ["\\begin{tabular}{lccccc}", "\\toprule",
       " & \\multicolumn{2}{c}{Canary cohort} & \\multicolumn{2}{c}{Organic probes} & \\\\",
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
       "Subject & $\\hat\\Delta$ & ISSUED & $\\hat\\Delta$ & $p$ & AUC \\\\", "\\midrule"]
for k, lab in order:
    r = rows[k]
    pc, pp = A["pooled"][k + "/canary"], A["pooled"][k + "/probe"]
    out.append(f"{lab} & ${pc['adv']:+.3f}$ & {r['canary_issued']:.2f} & "
               f"${pp['adv']:+.3f}$ & {p(pp['p_two_sided'])} & {r['organic_auc_loss']:.2f} \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(HERE, "tables", "e2e.tex"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
