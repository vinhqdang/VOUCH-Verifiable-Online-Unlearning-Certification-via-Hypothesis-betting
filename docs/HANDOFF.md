# Handoff — VOUCH revision for Springer *Machine Learning*

Last updated: 2026-09-03. Everything described here is committed and pushed to
`origin/main`. Nothing is left in a scratch directory that matters.

---

## 1. Where the work stands

The paper was submitted, reviewed by two referees, and has been **fully revised**. All
four deliverables build from one command and are committed:

| Deliverable | Path | State |
|---|---|---|
| Clean revised manuscript | `manuscript/main.pdf` | builds, 0 warnings |
| Point-by-point response | `manuscript/response.pdf` | builds, 0 warnings |
| Tracked-changes version | `manuscript/diff/main_tracked.pdf` | builds, 0 warnings |
| Cover letter (resubmission) | `manuscript/cover_letter.pdf` | builds |

**Not yet done: the actual resubmission to the journal.** That is the next action and
it is a human one.

### The round-1 record is archived

- `manuscript/v1/` — the submitted manuscript, self-contained: `main.pdf`, `main.tex`,
  `sections/`, `tables/`, `refs.bib`, `cover_letter.{tex,pdf}`, `VERIFY_CITATIONS.md`,
  plus a `README.md` explaining provenance and the round-1 → revision renumbering.
  This directory is also the latexdiff baseline; **do not edit it**.
- `manuscript/reviews/round1.md` — both referee reports, transcribed verbatim, with the
  point labels (R1.1–R1.7, R1-M1–M3, R2.W1–W7, R2.1–R2.12, minors) used throughout the
  response letter and the commit messages.

---

## 2. Setting up on a new machine

```bash
git clone <repo> && cd VOUCH-Verifiable-Online-Unlearning-Certification-via-Hypothesis-betting

# Python: conda env py313 (project convention)
conda create -n py313 python=3.13 -y && conda activate py313
pip install numpy scipy pandas matplotlib scikit-learn \
            torch transformers datasets accelerate peft

# LaTeX: a TeXLive/MacTeX with pdflatex, bibtex, latexdiff, latexmk
```

Verify:

```bash
python -m pytest tests/ -q                  # expect 18 passed
python experiments/verify_claims.py         # expect "all checks pass"
cd manuscript && ./build.sh                 # expect 4 PDFs, 0 warnings each
```

**Note on invoking python.** Use the interpreter path directly
(`/opt/miniconda3/envs/py313/bin/python`, or your equivalent) rather than
`conda run -n py313`: `conda run` buffers stdout, so long jobs appear to hang and
produce no progress log.

**Note on long jobs.** Launch them as managed background tasks, not with a bare
`nohup ... &` inside a foreground shell — if the foreground command times out, the
process group is killed and the child dies partway through. This bit us twice.

---

## 3. Rebuilding everything from scratch

Ordered, with rough costs. Nothing here needs a GPU beyond what an Apple-silicon
laptop (MPS) provides; the original large-model runs were done on free Colab/Kaggle
sessions and their results are committed.

```bash
# (a) simulations, CPU, ~2 min total on 9 procs
python experiments/run_simulation.py     --exp all --seeds 2000 --procs 9
python experiments/run_simulation_rev.py --exp all --seeds 2000 --procs 9

# (b) offline re-analysis of every saved end-to-end run, CPU, ~5 min
#     replays all verdicts at eps in {0.05,0.1,0.2} under both certificate targets
python experiments/reanalyze_rev.py

# (c) canary detectability, MPS/GPU, ~4 min
python experiments/run_detectability.py --dataset tofu --model gpt2 --device mps

# (d) tables, then all four PDFs
cd manuscript && ./build.sh
```

`build.sh` runs `make_tables.py` then `make_tables_rev.py` **in that order** — the
second owns `power/benchmarks/gpt2v2/zoo` and must run last or it gets overwritten by
the round-1 versions. Don't reorder them.

### Real-model runs (only if you need to regenerate them)

Results with per-pair score differences are committed, so `reanalyze_rev.py` can
replay any verifier configuration without touching a model. Re-run training only if
you change the pipeline.

```bash
# TOFU/GPT-2 with SimNPO + benchmark metrics, 3 seeds, ~25 min/seed on MPS
python experiments/run_benchmark.py --dataset tofu --model gpt2 --seeds 0 1 2 \
  --pairs 384 --dtype fp32 --device mps --queries 2 --extra-metrics \
  --resume --tag tofu_gpt2_rev

# tight-tolerance cohort: 3,072 pairs, r=1 only, ~50 min/seed on MPS
python experiments/run_benchmark.py --dataset tofu --model gpt2 --seeds 0 1 \
  --pairs 3072 --strata 1 --eps 0.05 --dtype fp32 --device mps --queries 2 \
  --methods none retrain npo simnpo --resume --tag tofu_gpt2_tight
```

Both support `--resume` and checkpoint per stage; `results/ckpt_*.pt` are gitignored
(9.5 MB each) and regenerable.

---

## 4. What changed in the revision

### The two decisive review findings

**R1.1 — the certificate's null.** Definition 1 defined a *marginal* sign probability
while the supermartingale proof required a *conditional* one. The two diverge exactly
in our setting, since every canary passes through one jointly trained model. Fixed by
retargeting the certificate at the **realised cohort advantage** and betting against a
without-replacement boundary recentred each step:

```
p0(t) = (m*p0 - sum_{u<t} Z_pi(u)) / (m - t + 1)
```

Theorem 2 is now exact under arbitrary dependence across pairs, templates and strata.
`OneSidedEProcess(..., population_size=m)`; `VouchConfig.finite_cohort` (default
`True`) selects it, `False` recovers the round-1 fixed-boundary process, and **both
targets are reported side by side everywhere**. Measured cost of the old construction:
it breaches α eightfold (0.389 vs 0.05) under model-level over-dispersion.

**R1.3b — the KL expansion.** `KL(Bern(½)‖Bern(½+ε/2)) = ε²/2 + O(ε⁴)`, so the cohort
rule is `2log(1/α)/ε²`, not `8log(1/α)/ε²`. Numerical tables were always computed from
the exact divergence and did not change; only the stated rule was wrong, by 4×.

### New experiments added

| Experiment | Script / data | What it shows |
|---|---|---|
| Dependence regimes | `run_simulation_rev.py --exp dependence` | old process breaches 8× under over-dispersion; new one holds ≤0.020 |
| Direct cohort-null check | `--exp cohortnull` | error ≤0.035 on adversarial fixed sign vectors |
| Four valid sequential comparators | `--exp baselines` | exact group-sequential (Pocock, OBF) via binomial DP, Beta- and normal-mixture e-processes, fixed-*n* |
| Power with censoring | `--exp power` | Kaplan–Meier medians + censoring rates, fixing the Table-3 artefact |
| Product-of-certificate-e-values | `--exp certprod` | product falsely certifies **1.000** of bad histories vs 0.000 all-pass |
| Descriptive head-to-head | `reanalyze_rev.py` | forget-quality KS + min-k% PrivLeak analogues on identical runs |
| Out-of-class attacks | `reanalyze_rev.py` | **null result** (see §5) |
| Canary detectability | `run_detectability.py` | perplexity filter recovers the whole cohort; entropy-dilution rule |
| SimNPO subject | `run_benchmark.py --methods simnpo` | certifies, widest forget/retain separation of any fluent method |
| Benchmark forget/retain/capability metrics | `--extra-metrics` | GA reaches ROUGE-L 0.00 both splits; GradDiff capability 19.4 vs retain 6.8 |
| Tight-tolerance cohort | `--tag tofu_gpt2_tight` | super-population certifies at ε=0.1; r=1 power limit |

### Code changes worth knowing about

- **`vouch/verify/betting.py`** — without-replacement boundary; `_B_EPS` is a *single
  shared clamp*. A round-2 review found `lam_max` clamping at `1e-12` while `update()`
  clamped at `1e-9`, which drove the worst-case wealth factor to **−998** whenever the
  boundary hit exactly 1 — reachable for the entire m=640 tier at every tolerance in
  both directions. Fixed; two regression tests guard it; the affected tier was re-run
  and **verdicts were unchanged**.
- **`vouch/verify/protocol.py`** — `finite_cohort`, `cohort_size`, and `sign_mean`
  recorded on the certificate.
- **`vouch/canaries/generator.py`** — Merkle commitment with per-pair opening
  (`merkle_root`, `leaf_hash`, `merkle_proof`, `verify_merkle_proof`, `open_leaf`),
  because Theorem 1 needs the coin of the pair opened at step *t* to be unrevealed at
  *t−1*, which a flat manifest hash cannot give. Also `qa_nat`/`fact_nat` (word-composed)
  and `qa_diluted`/`fact_diluted` (entropy-diluted) templates.
- **`experiments/verify_claims.py`** — **run this after any change to numbers.** It
  checks every quantitative claim in the manuscript against the result files. It exists
  because the round-2 review found five prose figures that had drifted from their
  tables, and it has since caught one more (a dose-response scoping error) on its own.

---

## 5. Open items, honestly stated

Ranked by how much a round-2 referee would care.

1. **A LiRA-style out-of-class attack.** Not run; needs dozens of retrained shadow
   references per cell. The two attacks we could build (stratum-restricted, cross-fitted
   learned combination) came back **below their own binomial nulls** — 2.8% observed
   against a 3.5% chance rate at *n*=96–159. So the gap between `sup_{s∈F}|Δ|` and what
   a strong attack could extract is **unmeasured**, not small. The manuscript says this;
   an earlier draft wrongly claimed the class restriction was "load-bearing" on this
   evidence and that inference has been withdrawn.
2. **A paraphrase-aware score in F.** Li et al. (ICLR 2026) show gradient-ascent
   objectives move probability mass onto semantically equivalent rephrasings — mass our
   literal-span scores cannot see by construction. Most valuable single extension.
3. **A finite-population power analysis.** Theorem 4 is proved for the fixed-boundary
   i.i.d. process; under without-replacement reveal the increments are neither
   independent nor identically distributed, so the ONS and KT regret bounds don't
   transfer. Scoped accordingly; the cohort target's timing is reported empirically.
4. **Audit-side reveal randomness.** Theorem 2 needs the reveal order π independent of
   the realised signs. Our runs seed π from the run seed, which also seeds cohort
   generation and training. Nothing exploits the coupling, but the condition is not met.
   Documented in Remark 3 and Appendix A. A deployed audit should use a public beacon.
5. **Power at r = 1.** The tight cohort showed GPT-2 memorises *nothing* at a single
   insertion: realised advantages −0.040 to +0.025, un-unlearned mean gap +0.005 nats,
   so "no unlearning" certifies at ε=0.05 alongside retraining. At the most
   organic-like dose the audit has no power at this model scale. Named in the conclusion
   as the framework's most important open problem.
6. **Method and probe coverage.** SimNPO added on TOFU/GPT-2 only; RMU and task-vector
   negation not run. P1/P3 cover all benchmark cells but are anchored on NPO; P2 runs
   on the small tiers only (the shared-frozen-base design would have to materialise
   merged weights to quantise a multi-billion-parameter model).
7. **Capability probe is thin** — TOFU `world_facts` / MUSE holdout, not MMLU.
8. **Response-letter cross-references.** Section/theorem/table numbers in
   `response.tex` were written against an intermediate draft and are offset from the
   final `main.pdf`. A numbering note at the top of the letter tells the referee this;
   a full pass to renumber them is still worth doing before resubmission.

### Also worth a final human pass

- `manuscript/VERIFY_CITATIONS.md` flags five round-1 references needing a human check.
  The seven references added at revision (Hu, Wang, Yuan, Li, Jagielski, GDPR, AI Act)
  were verified against OpenReview/dblp/EUR-Lex, but the ICLR 2026 entry
  (`li2026beliefs`) has **no confirmed OpenReview forum id** — it is cited by arXiv DOI.
- `results/lm_e2e_tofu_gpt2_tight_partial.json` (3 MB) duplicates the complete file and
  could be deleted.
- Abstract is 308 words. Springer prefers ~250; not a hard limit, but trimmable.

---

## 6. Reviewer-point status

`manuscript/reviews/round1.md` has the full text. Status as of this handoff:

**Reviewer 1** — 1 addressed via retargeting (with the π⊥z hypothesis now stated);
2 addressed; 3a, 3b, 3c addressed; 4 addressed (both error statistics printed);
5a addressed (Clopper–Pearson everywhere); 5b partial (SimNPO only); 5c addressed;
5d partial (budgets exact, coverage partial and stated); 6 addressed with substantive
discussion; 7a, 7b addressed; 7c addressed (50× workload ratio, not "two orders");
M1, M2 addressed; M3 partial.

**Reviewer 2** — W1 addressed (ε=0.05 cohort, ε=0.1 super-population); W2 addressed;
W3 addressed; W4 **partial and inconclusive** (see open item 1); W5 addressed;
W6 addressed; W7 addressed. Points 1–3, 5–7, 8–11 addressed; 4 partial; 12 addressed.
Minors: all addressed except the §1-length one (the intro was signposted, not shortened).

---

## 7. Round-2 internal review

After the revision was drafted, an adversarial referee pass was run against it. It
found **one real soundness bug** (the clamp mismatch above) and **eleven factual
errors**, every one of which was verified against the data before being fixed:

- the out-of-class result was a null result, not a breach (claim withdrawn);
- the baselines table compared VOUCH on a conditional error rate against six
  comparators on a marginal one (both now printed);
- a "0.96" rate cited *inside a proof* traced to no experiment (now measured: 1.000);
- 43% of issued certificates sit beside a CS upper bound above ε (now explained
  explicitly, with the coherent alternative rule and its 2× cost);
- the abstract attributed the 0.9M–5.1B model axis to TOFU/MUSE;
- the conclusion misstated the super-population reach;
- "only the two-sided arm correctly withholds" was false — nothing is withheld on any
  TOFU cell at ε=0.2;
- the recoverability pattern, the fixed-*n* power comparison sign, the
  "every procedure holds its level" line, the seeds column, and the Phi-1.5 log-wealth
  range were each wrong.

Most were round-1 text carried over without re-checking against recomputed tables.
That is the failure mode `experiments/verify_claims.py` now guards against, and it is
the single most useful thing to run before any resubmission.

Its verdict was **major revision**, which on the evidence was fair rather than harsh.

---

## 8. File map

```
manuscript/
  main.tex                  clean manuscript (thin; sections are \input)
  sections/{related,framework,theory,experiments,repro}.tex
  tables/*.tex              ALL GENERATED — never hand-edit
  make_tables.py            validity, streaming, soundness
  make_tables_rev.py        everything else; must run AFTER make_tables.py
  build.sh                  one command -> 4 PDFs
  response.tex              point-by-point response
  cover_letter.tex          resubmission cover letter
  refs.bib                  48 entries (41 round-1 + 7 added)
  v1/                       ARCHIVED round-1 submission + README (latexdiff baseline)
  reviews/round1.md         both referee reports, verbatim
  diff/                     generated tracked-changes build

vouch/
  verify/betting.py         e-processes, WoR boundary, betting CS  <- the bug was here
  verify/protocol.py        Phase-2 loop, streaming composition
  verify/scores.py          F; s_loss is MEAN TOKEN LOG-PROB (higher = more memorised)
  canaries/generator.py     templates, manifest, Merkle commitment

experiments/
  run_simulation.py         round-1 simulation tiers
  run_simulation_rev.py     dependence, cohortnull, baselines, power, certprod
  reanalyze_rev.py          offline replay of every run, both targets, eps sweep
  run_detectability.py      perplexity filter + trained detector + dilution rule
  run_benchmark.py          TOFU/MUSE end-to-end (--strata, --extra-metrics, --resume)
  verify_claims.py          RUN THIS AFTER ANY NUMBER CHANGES
tests/test_betting.py       18 tests incl. WoR validity + Merkle + the clamp regression
docs/HANDOFF.md             this file
```

### One convention that matters

`s_loss` is the **mean token log-probability** of the secret span — the *negative* of
its NLL. So `D = s(in) − s(ghost)` is positive when the trained twin is better
remembered, which is the direction of leakage. Round 1 described this as an NLL in the
prose while the code used the log-probability, which inverted every reported gap; that
was Reviewer 1's point 3a. The convention is now pinned in `sections/framework.tex`
§4.4, in the worked-example figure, and in `scores.py`'s docstring. **Don't flip it.**
