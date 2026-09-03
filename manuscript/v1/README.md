# Round-1 submission (archived)

This directory is the **first submission** to Springer *Machine Learning*, exactly as
it went to the editors, preserved so that the revision can be diffed against it and so
that reviewer references to section, theorem and table numbers can be resolved.

Do not edit anything here. `manuscript/build.sh` reads `v1/main.tex` as the latexdiff
baseline that produces `manuscript/diff/main_tracked.pdf`.

| File | What it is |
|---|---|
| `main.pdf` | the submitted PDF |
| `main.tex`, `sections/`, `tables/` | its sources, as submitted |
| `refs.bib` | the round-1 bibliography (41 entries; the revision adds 7) |
| `cover_letter.tex`, `cover_letter.pdf` | the round-1 cover letter |
| `VERIFY_CITATIONS.md` | the round-1 citation-verification notes |

Provenance: restored from commit `b501fef` ("Cover letter: date 5 July 2026"), the last
commit before revision work began.

The two referee reports are in **`../reviews/round1.md`**, transcribed verbatim, with
the point labels (R1.1–R1.7, R1-M1–M3, R2.W1–W7, R2.1–R2.12, R2 minors) that the
response letter and the revision commits refer to throughout.

The subsequent round-2 internal referee report — an adversarial self-review of the
revision, which found a real soundness bug in the betting engine and eleven factual
errors — is summarised in `../../docs/HANDOFF.md` under "Round-2 internal review".

## Numbering caveat

Section, theorem, remark and table numbers **differ** between this version and the
revision, because the revision adds five experiment subsections, six tables, one
appendix and three theory statements. The response letter states this explicitly at the
top and uses revised-manuscript numbering throughout; reviewer points are cited by
their original labels.

Notable renumberings a reader of both versions will hit:

| Round 1 | Revision |
|---|---|
| Theorem 2 (certificate coverage) | Theorem 2, restated for the finite-cohort target |
| Theorem 3 (power) | Theorem 4, with the corrected $\varepsilon^2/2$ expansion |
| Theorem 4 (streaming) | Theorem 5 |
| Proposition 6 (bridge) | Proposition 7, with the one-directionality remark after it |
| Table 3 (power vs KL) | Table 6, with censoring rates and Kaplan–Meier medians |
| Table 5 (GPT-2 end-to-end) | Table 7 |
| Table 7 (model axis) | Table 15, with a seeds column |
