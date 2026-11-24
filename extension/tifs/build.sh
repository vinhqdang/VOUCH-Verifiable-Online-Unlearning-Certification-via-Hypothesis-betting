#!/bin/sh
set -e
cd "$(dirname "$0")"
for i in 1 2; do
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  pdflatex -interaction=nonstopmode supplement.tex >/dev/null || true
done
bibtex main >/dev/null; bibtex supplement >/dev/null
for i in 1 2; do
  pdflatex -interaction=nonstopmode main.tex >/dev/null || true
  pdflatex -interaction=nonstopmode supplement.tex >/dev/null || true
done
