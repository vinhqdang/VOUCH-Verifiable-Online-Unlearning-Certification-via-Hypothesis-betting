# VOUCH: Verifiable Online Unlearning Certification via Hypothesis-betting

**Anytime-valid statistical certificates for machine unlearning via paired-canary betting e-processes.**

This repository implements, evaluates, and *strengthens* the framework specified in
[`algorithm.md`](algorithm.md) (design document v1.0). VOUCH plants **paired ghost
canaries** at fine-tuning time — one twin trained-then-forgotten, one twin never seen,
assignment by fair coin — and, after unlearning, runs **betting e-processes** over the
canary pairs. Under exact unlearning the within-pair score difference is symmetric *by
construction*, giving an exact, distribution-free, finite-sample null with no shadow
models and no retrained reference model. The verifier outputs:

1. an **anytime-valid confidence sequence** on the residual membership advantage Δ;
2. a **sequential equivalence test** that issues an (ε, α)-Forgetting Certificate at a
   data-dependent stopping time (valid under optional stopping, by Ville's inequality);
3. a dual **revocation e-process** that raises an alarm on residual memorization;
4. **streaming composition** across deletion waves with history-wide error control;
5. **R-VOUCH** recoverability probes (relearning, 4-bit quantization, jailbreak prompts).

## Improvements over the v1.0 design document

Implementing the design surfaced three soundness/operational gaps. All three are fixed
here and demonstrated empirically; they materially strengthen the paper.

### I1 — The design doc's certificate arm is anytime-*invalid* (fixed)

Design doc §4.4(a) bets the certificate e-process on the *mixture-weighted sign*
`Z̄ᵢ = Σₛ wₛ Zᵢˢ` across the score class F. That is sound for the **revocation** arm
(its null — exact unlearning — makes *every* per-score sign exactly Bern(1/2), so any
predictable mixture has conditional mean 1/2). It is **not** sound for the
**certificate** arm, whose composite null is "∃ s ∈ F with pˢ ≥ p₀": if one score
still leaks while others are clean, the mixture mean sits *below* p₀ and the
"e-process" is no supermartingale.

**Fix.** VOUCH runs one certificate e-process per score and issues the certificate only
when `min_s E_t^{cert,s} ≥ 1/α`. A false certificate requires the truly-violating
score's own e-process to cross 1/α — probability ≤ α by Ville, *no Bonferroni needed*.
Dually, `max_s U_t^s` is an anytime upper confidence bound on `sup_s p^s`.

**Evidence** (2,000 seeds; three scores, one at the null boundary p₀ = 0.55, two clean;
2,048 pairs; α = 0.05):

| certificate arm | false-certification rate |
|---|---|
| per-score min (**VOUCH**, this repo) | **0.007** ✓ |
| mixture-sign arm (design doc v1.0) | **0.996** ✗ — essentially always falsely certifies |

### I2 — Streaming composition direction corrected (design doc Theorem 5)

Multiplying per-wave **certificate** e-values (doc §4.5/Thm 5) tests the *intersection*
null "every wave is bad"; rejecting it only certifies that *some* wave is clean — not
the intended claim. Certificates and alarms must compose in opposite directions:

- **Global certificate = all-pass:** every wave individually earns its certificate.
  The probability of ever claiming "all waves clean" while some wave is bad is ≤ α
  (Ville applied to the bad wave's own e-process), for any number of waves.
- **Global revocation alarm = product:** under the intersection null "all waves exactly
  unlearned" every per-wave revocation e-process is a supermartingale, so products
  compose soundly. The product alarm accumulates **distributed sub-threshold leakage**
  that no single wave reveals. Per-wave alarms additionally get α-spending
  (α_k = α·2^−(k+1)), keeping family-wise false revocation ≤ α over an unbounded
  deletion history.

**Evidence** (K = 10 waves × 512 pairs, 500 histories, ε = 0.2, α = 0.05):

| deletion history | per-wave alarm | global product alarm | global certificate (all-pass) |
|---|---|---|---|
| all waves exact | FWER **0.000** | false alarm **0.002** | issued 0.48 (per-wave power 0.93¹⁰; size cohorts for K) |
| one bad wave (Δ = 0.25) | catches it **0.984**; falsely issued **0.000** | fires 0.922 | **never** falsely issued (0.000) |
| ten weak waves (Δ = 0.06 each) | nearly blind: 0.136 | **fires 0.616, median wave 3** | never issued (0.000) |

The last row is the payoff: leakage spread thinly across waves is invisible to every
per-wave test (and to any fixed-n audit), but the composed e-process accumulates it.

### I3 — Corrected power analysis and a stronger betting engine

Design doc §6.6 claims ε = 0.1 certification at α = 0.05 "in ≈ 190 pairs". This
contradicts its own Theorem 3: the information-theoretic limit is
`log(1/α)/KL(Bern(½) ‖ Bern(½+ε/2)) ≈ 596` pairs. Our measured median is 686 pairs —
15% above the bound, i.e., the engine is near-optimal and the doc's sample-size
guidance must be revised (tables below give the corrected numbers).

Engine improvements over the doc's reference ONS implementation:

- **Truncated Krichevsky–Trofimov plug-in expert** in the bet mixture: the Bernoulli
  likelihood-ratio e-process with plug-in estimate truncated to the alternative side —
  valid for the composite one-sided null, near-KL growth with ½·log t redundancy.
- **Discrete λ-grid + ONS + KT mixture** (a mixture of e-processes is an e-process):
  robust log-optimal growth with no tuning. A fixed aggressive bet, by contrast,
  goes bankrupt (see ablation).
- **VOUCH+ magnitude-aware revocation** (`SymmetryEProcess`): under the exact symmetry
  null, sign(D) given |D| is a fair coin, so betting the sign with a stake scaled by a
  predictable rank-transform of |D| is an exact e-process that exploits effect sizes.
  Under sparse leakage (5% of pairs strongly memorized) it detects **2.3× faster**
  than sign-only betting (median 941 vs 2,196 pairs) at higher detection rate
  (99% vs 78% within 5,000 pairs).

### I4 — Two-sided certification: over-forgetting is leakage too

Discovered when running the end-to-end GPU tier: GradDiff on GPT-2 earned a
(one-sided) certificate while its in-twins scored *below* their ghost twins
(mean loss gap −0.155) — gradient-ascent-style unlearning had pushed the forgotten
canaries to anomalously *bad* scores. Below-chance scoring is itself membership
signal (an attacker negates the score), so the design doc's one-sided
Δ = 2·P(in > ghost) − 1 target is too weak.

**Fix.** VOUCH closes the score class under negation (`two_sided=True`, default):
each score gets certificate e-processes against both `p ≥ ½ + ε/2` and
`p ≤ ½ − ε/2` (the certificate now asserts **|Δˢ| < ε for every s**), and the
revocation arm bets in both directions. Validity is unchanged (under the exact null
every direction is a fair coin / symmetric); the measured protocol-level cost is
~1.7× more pairs (median 283 vs 166 at ε = 0.2, α = 0.05; 1,182 vs 686 at ε = 0.1),
and over-forgetting subjects are now correctly revoked (unit-tested).

## Results

All numbers are reproducible from `experiments/` (JSON in `results/`, figures in
`results/figures/`).

### M1 — Validity under adversarial optional stopping (2,000 seeds, 1,024 pairs)

Peeking after **every** pair, stopping at the first crossing:

| procedure | nominal α = 0.05 | nominal α = 0.01 |
|---|---|---|
| VOUCH false certification (boundary null) | 0.028 | 0.006 |
| VOUCH false revocation (exact unlearning) | 0.026 | 0.006 |
| fixed-n binomial with peeking — false alarm | **0.352** | **0.119** |
| fixed-n binomial with peeking — false certification | **0.340** | **0.098** |

Calibration is monotone and conservative across α ∈ {0.01, …, 0.2}
(`fig1_validity`). Full-protocol runs (three correlated scores, two-sided, CS + both
arms, early stopping) give false-revocation 0.025 and uniform-in-time per-CS coverage
0.971 at nominal 0.95; under genuine memorization (Δ ≈ 0.36) revocation fires 100% of
the time at a median of **44 pairs** with zero false certificates.

### M2 — Power / certification time (exact unlearning; median pairs to certificate)

| ε | α = 0.05 | KL limit | α = 0.01 | KL limit |
|---|---|---|---|---|
| 0.02 | 11,189* | 14,976 | 13,400* | 23,021 |
| 0.05 | 2,998 | 2,394 | 4,426 | 3,680 |
| 0.10 | 686 | 596 | 1,088 | 916 |
| 0.20 | 166 | 147 | 252 | 226 |

*medians over runs issued within the 20,000-pair horizon (55% / 28%).
Two-sided protocol certificates (default) need ~1.7× more pairs: median 1,182 at
ε = 0.1 and 283 at ε = 0.2 (α = 0.05). Revocation detection (median pairs, 100%
detection): Δ = 0.05 → 2,948; Δ = 0.1 → 690; Δ = 0.2 → 179; Δ = 0.4 → 48.

### M3 — Tightness

Median CS upper bound on Δ under exact unlearning: 0.26 at t = 128, 0.18 at t = 256,
0.13 at t = 512, 0.10 at t = 1,024 pairs; with true Δ = 0.2 the bound sits at 0.29 at
t = 1,024 (`fig4_tightness`).

### Betting-strategy ablation (ε = 0.1, α = 0.05, exact unlearning)

| strategy | median τ* | certified within 20k pairs |
|---|---|---|
| ONS | 638 | 100% |
| mixture (default) | 686 | 100% |
| aGRAPA | 888 | 100% |
| truncated-KT | 891 | 100% |
| fixed λ = ½·λmax | — | 10% (over-betting bankrupts the wealth) |

### End-to-end on a real LM (GPU tier: GPT-2 on a Colab T4)

Full pipeline per seed: canary generation + SHA-256 manifest commitment → fine-tune on
keep ∪ forget ∪ in-twins → unlearn with each subject → black-box certification →
R-VOUCH probes on NPO. Two runs:

**v2 (two-sided, 640 pairs, scores {loss, min-k, ratio}, corrected NPO), 3 seeds:**

| subject | decision (3 seeds) | mean CS upper bound on Δ | mean D (loss) |
|---|---|---|---|
| no unlearning | REVOKED ×3 (log-e 277–336) | 0.93 | +2.19 |
| retrain (exact) | **ISSUED ×3** | 0.19 | −0.00 |
| GA | ISSUED ×3 | 0.21 | +0.09 |
| GradDiff | ISSUED ×3 | 0.18 | +0.08 |
| NPO | ISSUED ×3 | 0.17 | +0.04 |
| NPO 25% (weakened) | ISSUED ×2, UNDET ×1 | 0.23 | +0.26 |
| NPO + P1 relearn | ISSUED ×2, UNDET ×1 | 0.23 | +0.28 |
| NPO + P2 4-bit / P3 jailbreak | ISSUED ×3 | 0.17–0.18 | +0.03–0.06 |

Dose–response on real GPT-2 (mean D by repetition stratum r = 1/2/4/8): the memorized
model is strongly monotone (**1.2 / 1.5 / 2.7 / 4.1 nats**, seed 1); the **relearn probe
resurfaces the most-repeated canaries first** (0.19 / 0.11 / 0.34 / 0.51), and weakened
NPO leaves precisely the r = 8 stratum leaking (up to 1.29 nats) — repetition at
fine-tuning time is the primary driver of both memorization and recoverability.
Measured verifier cost: **76 s per verification** on a T4 (10,240 short forwards)
versus ~300 s for one fine-tune — a shadow-model evaluator at 16 fine-tunes would cost
~60× more.

**v1 (one-sided, 512 pairs, 3 seeds)** additionally caught a **sign bug in our own NPO
implementation** (the loss was being minimized toward memorization): revoked 3/3 seeds
within ~11 pairs — statistical certification audits the unlearning *code*, not just the
outcome. Retrain certified 3/3; results in `results/lm_e2e_gpt2_v1.json`.

**Small-model tier** (`--model tiny`, 3 seeds, retrain ground truth affordable,
`results/lm_e2e_tiny.json`): no-unlearning revoked 3/3 (log-e 154–212); retrain, GA,
GradDiff certified 3/3; NPO 2/3 (1 honest UNDETERMINED at the 512-pair horizon).
Dose–response (§6.7), measured: mean D by repetition stratum r = 1/2/4/8 is
0.47 / 0.64 / 0.79 / 1.04 on the memorized model and ≈ 0 after unlearning.
Utility guardrail: canary injection shifts held-out loss by +0.005 nats/char even at an
extreme 38% corpus share (design target is < 0.05% share).

## Repository layout

```
vouch/
├── canaries/generator.py    # twin-pair templates, secrets, manifests, sha256 commitments
├── training/inject.py       # corpus assembly with repetition strata (dose-response)
├── unlearn/methods.py       # finetune / GA / GradDiff / NPO(+RT) / retrain
├── verify/scores.py         # s_loss, s_mink, s_ratio; Q query wrappers; ScoreEngine
├── verify/betting.py        # one-sided e-processes (ONS/aGRAPA/fixed/mixture/KT),
│                            # WSR betting CS, magnitude-aware symmetry e-process
├── verify/protocol.py       # Phase-2 loop; certificate object; streaming composition
├── probes/probes.py         # R-VOUCH: relearn (P1), quantize (P2), jailbreak (P3)
├── baselines/fixed_n.py     # binomial / TOST / permutation / KS + peeking wrappers
└── models/tiny_gpt.py       # small causal LM for the CPU-affordable validity tier
experiments/
├── run_simulation.py        # M1 validity, I1 soundness demo, M2 power, M3 tightness,
│                            # M6 streaming, ablations   (--exp all --seeds 2000)
├── run_lm_e2e.py            # end-to-end LM tier (--model tiny | gpt2 | any HF id)
└── make_figures.py          # publication figures from results/*.json
tests/test_betting.py        # e-process supermartingale + Ville, CS coverage, protocol
results/                     # JSON results + figures (committed for reproducibility)
```

## Reproducing

```bash
pip install numpy scipy matplotlib torch            # + transformers for the HF tier
python3 tests/test_betting.py                       # validity unit tests (~40 s)
python3 experiments/run_simulation.py --exp all --seeds 2000
python3 experiments/run_lm_e2e.py --model tiny --seeds 0 1 2        # CPU tier
python3 experiments/run_lm_e2e.py --model gpt2 --seeds 0 1 2 \
        --pairs 512 --eps 0.2 --device cuda                          # GPU tier
python3 experiments/make_figures.py
```

The committed GPU-tier results were executed on a Google Colab T4 provisioned through
`google-colab-cli`.

## The certificate object

Each wave publishes (`Certificate.to_json()`):

```json
{
  "status": "ISSUED", "eps": 0.2, "alpha": 0.05, "wave": 0,
  "t_stop": 231, "t_revoked": -1,
  "log_e_cert": 3.02, "log_e_rev": -0.41,
  "p_upper": 0.58, "delta_upper": 0.16,
  "delta_cs": {"loss": [-0.11, 0.16], "mink": [-0.09, 0.14]},
  "per_score_log_e_cert": {"loss": 3.02, "mink": 3.44},
  "manifest_sha256": "…", "score_class": ["loss", "mink"],
  "probes": {"P1": {"...": "..."}}, "code_version": "vouch-1.0"
}
```

Semantics: *"with confidence 1 − α, uniformly over all stopping times, the residual
membership advantage of the unlearned model against the declared score class F on the
canary population is below ε"* — certification of extractable influence relative to a
declared attack class, per honest-scope desideratum D6 of `algorithm.md`.

## What is certified (and what is not)

VOUCH certifies **extractable residual influence relative to the declared score class
and canary law** — not parameter-space erasure (behaviorally uncertifiable) and not
security against a malicious provider (compose with cryptographic proof-of-unlearning
for that). Canaries must be planted before fine-tuning; dose–response strata
(r ∈ {1, 2, 4, 8}) empirically calibrate the canary→organic extrapolation.

## License

MIT (see `LICENSE`).
