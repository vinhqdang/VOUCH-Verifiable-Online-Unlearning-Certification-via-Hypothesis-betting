#!/usr/bin/env python3
"""Baseline comparison table (imported by make_tables.py, or run directly)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "results", "baselines_pythia410m_analysis.json")))
ORDER = ["none", "retrain", "ga@10", "ga@15", "ga@20", "npo@25", "npo@40", "npo@55", "npo@75"]


def label(s):
    return s if "@" not in s else f"{s.split('@')[0].upper()} {s.split('@')[1]}"


rows = {r["subject"]: r for r in A["rows"]}
lines = []
for s in ORDER:
    if s not in rows:
        continue
    r = rows[s]
    f = lambda k: f"{r[k]:.2f}"
    cells = [f("vouch_revoked_literal"), f("vouch_revoked_union")]
    cells += [f(f"mia/literal/matched/{x}/leak") for x in ("loss", "mink", "zlib", "ref")]
    cells += [f(f"mia/literal/unmatched/{x}/leak") for x in ("loss", "ref")]
    cells += [f(f"mia/paraphrase/matched/{x}/leak") for x in ("loss", "ref")]
    cells += [f(f"mia/organic/forget/{x}/leak") for x in ("loss", "ref")]
    lines.append(f"{label(s)} & " + " & ".join(cells) + " \\\\")
    if s == "retrain":
        lines.append("\\midrule")
body = ("\\begin{tabular}{l cc cccc cc cc cc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{VOUCH} & \\multicolumn{4}{c}{(a) canaries vs.\\ ghosts}"
        " & \\multicolumn{2}{c}{(b) canaries vs.\\ organic} & \\multicolumn{2}{c}{(c) paraphrase}"
        " & \\multicolumn{2}{c}{(d) organic forget set} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\n"
        "Subject & literal & union & loss & Min-K\\% & zlib & ref & loss & ref & loss & ref & loss & ref \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(HERE, "tables", "baselines.tex"), "w").write(body)
print("wrote baselines")
auc = {s: {k.split("/")[1] + "/" + k.split("/")[2] + "/" + k.split("/")[3]: round(v, 3)
           for k, v in rows[s].items() if k.endswith("/auc")} for s in ORDER if s in rows}
for s in ORDER:
    if s in auc:
        print(s, rows[s]["n_seeds"], {k: v for k, v in auc[s].items() if k.endswith(("loss", "ref"))})
