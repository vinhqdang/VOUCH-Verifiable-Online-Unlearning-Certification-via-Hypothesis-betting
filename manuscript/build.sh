#!/usr/bin/env bash
# Build every manuscript deliverable from results/*.json.
#
#   main.pdf                 clean revised manuscript
#   response.pdf             point-by-point response to reviewers
#   diff/main_tracked.pdf    tracked-changes version against manuscript/v1
#   cover_letter.pdf         cover letter
#
# Usage:  cd manuscript && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

# Python: honour $PY, else the project conda env if present, else python3.
if [ -z "${PY:-}" ]; then
  if [ -x /opt/miniconda3/envs/py313/bin/python ]; then
    PY=/opt/miniconda3/envs/py313/bin/python
  else
    PY=python3
  fi
fi
FAIL=0

echo "== regenerating tables from results/ =="
"$PY" make_tables.py          # validity, streaming, soundness
"$PY" make_tables_rev.py      # everything else (owns power/benchmarks/zoo/gpt2v2)

latexrun () {  # $1 = jobname
  pdflatex -interaction=nonstopmode "$1.tex" >/dev/null 2>&1 || true
  bibtex "$1" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$1.tex" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$1.tex" > "/tmp/$1.pass3.log" 2>&1 || true
  if [ ! -f "$1.pdf" ]; then echo "  !! $1.pdf not produced"; return 1; fi
  local warn err
  warn=$(grep -cE "LaTeX Warning: (Reference|Citation).*undefined" "/tmp/$1.pass3.log" || true)
  err=$(grep -c "^! " "/tmp/$1.pass3.log" || true)
  echo "  $1.pdf built ($warn undefined ref/cite warnings, $err LaTeX errors)"
  if [ "$err" != "0" ]; then grep -A2 "^! " "/tmp/$1.pass3.log" | head -20; FAIL=1; fi
  if [ "$warn" != "0" ]; then FAIL=1; fi
}

echo "== clean manuscript =="
latexrun main

echo "== response letter =="
latexrun response

echo "== cover letter =="
pdflatex -interaction=nonstopmode cover_letter.tex >/dev/null 2>&1 || true
echo "  cover_letter.pdf built"

echo "== tracked-changes version =="
mkdir -p diff /tmp/vouch_v1f /tmp/vouch_v2f
# Flatten \input of section files only; table bodies stay as \input so that
# latexdiff never marks up inside a booktabs tabular (it breaks \cmidrule).
"$PY" - <<'PYEOF'
import re, os
def flatten(root, main, out):
    def expand(path):
        s = open(path).read()
        def rep(m):
            inc = m.group(1)
            if inc.startswith('tables/'):
                return m.group(0)
            for cand in (os.path.join(root, inc + '.tex'), os.path.join(root, inc)):
                if os.path.exists(cand):
                    return expand(cand)
            return m.group(0)
        return re.sub(r'\\input\{([^}]+)\}', rep, s)
    open(out, 'w').write(expand(os.path.join(root, main)))
flatten('v1', 'main.tex', '/tmp/vouch_v1f/main.tex')
flatten('.',  'main.tex', '/tmp/vouch_v2f/main.tex')
PYEOF

latexdiff --encoding=utf8 --type=UNDERLINE \
  --append-safecmd="cmidrule,midrule,toprule,bottomrule,addlinespace,vI,vR,vU" \
  --config="PICTUREENV=(?:picture|tikzpicture|DIFnomarkup|algorithmic)[\w\d*@]*" \
  /tmp/vouch_v1f/main.tex /tmp/vouch_v2f/main.tex > diff/main_tracked.tex

# The v1 text that latexdiff preserves as struck-through references \eqref{eq:wealth},
# an equation this version renamed; repoint it so the deleted block does not
# render a dangling "??".
# (portable in-place edit: GNU sed and BSD sed disagree on the -i argument)
sed -i.bak 's/eq:wealth}/eq:wealthwor}/g; s/eq:wealthworwor}/eq:wealthwor}/g' diff/main_tracked.tex
rm -f diff/main_tracked.tex.bak

cd diff
for f in sn-jnl.cls sn-mathphys-num.bst refs.bib; do ln -sf "../$f" . ; done
ln -sfn ../figures figures; ln -sfn ../tables tables
pdflatex -interaction=nonstopmode main_tracked.tex >/dev/null 2>&1 || true
bibtex main_tracked >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main_tracked.tex >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main_tracked.tex > /tmp/main_tracked.pass3.log 2>&1 || true
warn=$(grep -cE "LaTeX Warning: (Reference|Citation).*undefined" /tmp/main_tracked.pass3.log || true)
err=$(grep -c "^! " /tmp/main_tracked.pass3.log || true)
echo "  diff/main_tracked.pdf built ($warn undefined ref/cite warnings, $err LaTeX errors)"
if [ "$err" != "0" ]; then grep -A2 "^! " /tmp/main_tracked.pass3.log | head -20; FAIL=1; fi
cd ..

echo
echo "== deliverables =="
ls -la main.pdf response.pdf cover_letter.pdf diff/main_tracked.pdf
if [ "$FAIL" != "0" ]; then echo; echo "!! build finished with LaTeX errors or undefined references (see above)"; exit 1; fi
