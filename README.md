# VOUCH: Verifiable Online Unlearning Certification via Hypothesis-betting

Anytime-valid statistical certificates for machine unlearning, using paired ghost
canaries and betting e-processes.

> **Paper:** *VOUCH: Anytime-Valid Certificates for Machine Unlearning via Paired-Canary
> Hypothesis-Betting*, Quang-Vinh Dang. Accepted for publication in *Machine Learning*
> (Springer). The final version is in production; this repository is the code and data
> behind the accepted paper.

## Idea

At fine-tuning time the provider plants **paired ghost canaries**: exchangeable twin
records, of which a fair coin admits one into training (and later into the forget set)
while the other is never trained on. If unlearning is exact, the two twins are
statistically interchangeable, so the within-pair score difference is symmetric by
construction. This gives an exact, distribution-free, finite-sample null with no shadow
models and no retrained reference model.

On top of that null, VOUCH runs betting e-processes that stay valid at every stopping
time. The verifier outputs:

1. an anytime-valid **confidence sequence** on the residual membership advantage;
2. an (ε, α)-**forgetting certificate** from sequential two-sided equivalence testing;
3. a **revocation alarm** for residual memorisation;
4. **streaming composition** across deletion waves with history-wide error control;
5. **R-VOUCH** recoverability probes (benign relearning, 4-bit quantisation, jailbreak
   prompts).

The certificate targets the *realised advantage on the committed canary cohort* using a
without-replacement boundary, which is exact under arbitrary dependence between pairs.
A fixed-boundary variant (valid under a conditional-mean assumption) is reported
alongside.

## Install

```bash
pip install -e .                 # numpy, scipy
pip install -e ".[hf,figs]"      # + torch, transformers, matplotlib
# real-model experiments also need: peft accelerate datasets
```

Python 3.9+. Tests: `python -m pytest tests/ -q`. Every number quoted in the paper can be
checked against `results/` with `python experiments/verify_claims.py`.

## Quick start

```python
from vouch.verify import VouchConfig, VouchVerifier

# pair_diffs[i][score] = score(in-twin) - score(ghost-twin) for pair i
verifier = VouchVerifier(["loss", "mink"],
                         VouchConfig(eps=0.2, alpha=0.05),
                         manifest_sha256="<commitment published before training>")
cert = verifier.run(pair_diffs, shuffle_seed=0, early_stop=True)
print(cert.status, cert.t_stop, cert.delta_upper)   # ISSUED / REVOKED / UNDETERMINED
print(cert.to_json())
```

Pair generation, corpus assembly and scoring live in `vouch/canaries`,
`vouch/training` and `vouch/verify/scores.py`. For an audit run in production, draw the
reveal order from a public beacon, not from a seed:

```python
cert = verifier.run(pair_diffs, reveal_source="beacon")   # drand; "local" for offline
```

The certificate records which source was used, in a `reveal` block. The seeded order does
not satisfy the independence hypothesis of the certificate theorem and exists only to
reproduce the committed results.

## Repository layout

```
vouch/
├── canaries/generator.py    # twin-pair templates, secrets, manifests, sha256 commitments
├── training/inject.py       # corpus assembly with repetition strata
├── unlearn/methods.py       # finetune / GA / GradDiff / NPO / retrain (TinyGPT tier)
├── verify/betting.py        # e-processes (ONS, aGRAPA, fixed, mixture, KT), confidence
│                            # sequences, symmetry e-process, without-replacement boundary
├── verify/protocol.py       # verification loop, certificate object, streaming composition
├── verify/scores.py         # s_loss, s_mink, s_ratio, paraphrase-aware s_para
├── verify/beacon.py         # reveal-order randomness (drand beacon / OS entropy / seeded)
├── probes/probes.py         # R-VOUCH probes P1-P3
├── baselines/fixed_n.py     # fixed-n and sequential comparators
└── models/tiny_gpt.py       # small causal LM for the CPU tier
experiments/
├── run_simulation*.py       # validity, power, tightness, baselines, streaming, ablations
├── run_lm_e2e.py            # end-to-end LM tiers (tiny, GPT-2, any HF model)
├── run_benchmark.py         # TOFU / MUSE-News tiers with GA, GradDiff, NPO, SimNPO, RMU
├── run_lm_big.py            # LoRA training/unlearning utilities for the benchmark tiers
├── run_lira.py, run_lira_hf.py, analyze_lira.py   # LiRA shadow-model attack
├── run_paraphrase.py        # paraphrase-aware score
├── run_detectability.py     # can a provider find the canaries?
├── replay_reveal_orders.py  # replay saved audits under independent reveal orders
├── reanalyze_rev.py         # offline replay of every saved run at several tolerances
├── verify_claims.py         # checks quoted numbers against results/
└── make_figures.py
tests/test_betting.py        # supermartingale/Ville, CS coverage, protocol, reveal sources
tools/                       # Colab/Kaggle orchestration helpers
results/                     # JSON results and figures
manuscript/                  # LaTeX source of the paper
docs/                        # THEORY.md, PAPER.md, HANDOFF.md (rebuild notes)
algorithm.md                 # original design document
```

## Reproducing

```bash
python experiments/run_simulation.py     --exp all --seeds 2000   # calibration, power
python experiments/run_simulation_rev.py --exp all --seeds 2000   # dependence, baselines
python experiments/reanalyze_rev.py                               # replay saved LM runs
python experiments/replay_reveal_orders.py                        # reveal-order check
python experiments/verify_claims.py                               # numbers vs results/
cd manuscript && ./build.sh                                       # rebuild the PDFs
```

Real-model runs save per-pair scores, so verdicts at any tolerance can be replayed
offline without a GPU. To re-run training:

```bash
python experiments/run_benchmark.py --dataset tofu --model gpt2 --seeds 0 1 2 \
  --pairs 384 --dtype fp32 --device cuda --queries 2 \
  --methods none retrain npo simnpo rmu --extra-metrics --resume --tag tofu_gpt2_rmu
python experiments/run_lira_hf.py --shadows 24 --pairs 384 --device cuda
```

The GPU runs were executed on Google Colab T4 instances through `google-colab-cli`; see
`docs/HANDOFF.md` for the exact setup and known pitfalls.

## Results in brief

- **Validity.** Under adversarial optional stopping (2,000 seeds), the fixed-sample tests
  inflate their error; VOUCH's certificate holds its level. A fixed-boundary process
  breaches its level eightfold under model-level dependence; the without-replacement
  cohort certificate does not.
- **Soundness of composition.** Certificates compose by consensus and alarms by product;
  the adaptive-mixture alternative falsely certifies 99.6% of the time under an
  adversarial composite null.
- **Real models.** TOFU and MUSE-News on GPT-2, Pythia-160M and Phi-1.5, plus a model
  axis up to 5.1B parameters. Un-unlearned models are revoked on 16 of 18 benchmark
  runs; retraining, gradient ascent and NPO are certified on 18 of 18 at ε = 0.2.
  The cohort target reaches ε = 0.05; the fixed-boundary variant reaches ε = 0.1 once the
  cohort exceeds about 3,000 pairs.
- **Utility-preserving methods.** SimNPO and RMU are evaluated on TOFU (GPT-2,
  Pythia-160M) and MUSE-News (GPT-2). RMU barely moves the forget-set likelihood yet lands
  near retraining on the certified quantity; on MUSE-News it is revoked on one of three
  seeds at ε = 0.2.
- **Attacks from outside the score class.** A LiRA shadow-model attack recovers
  Δ = +0.82 (TinyGPT, 16 shadows) and +0.48 (GPT-2, 24 shadows) on un-unlearned models and
  Δ = −0.016 and −0.031 on certified NPO models. The 95% intervals lie inside (−0.2, 0.2),
  so they support equivalence at ε = 0.2 but not at 0.1 or 0.05.
- **Reveal order.** Replaying 468 saved audit cells under independent permutations leaves
  the majority verdict unchanged in 450; the ε = 0.2 headline counts are unchanged except
  GradDiff (17 → 18 of 18 issued). One cell (TOFU/GPT-2 RMU, seed 1) is order-sensitive.
- **Cost.** 2Qm forward passes for m pairs and Q query wrappers.

Full tables and figures are in the paper and in `results/`.

## The certificate object

```json
{
  "status": "ISSUED", "eps": 0.2, "alpha": 0.05, "wave": 0,
  "t_stop": 231, "t_revoked": -1,
  "log_e_cert": 3.02, "log_e_rev": -0.41,
  "p_upper": 0.58, "delta_upper": 0.16,
  "delta_cs": {"loss": [-0.11, 0.16], "mink": [-0.09, 0.14]},
  "per_score_log_e_cert": {"loss": 3.02, "mink": 3.44},
  "manifest_sha256": "…", "score_class": ["loss", "mink"],
  "reveal": {"reveal_source": "beacon", "reveal_verifiable": true,
             "reveal_independent_of_signs": true},
  "probes": {}, "code_version": "vouch-1.0"
}
```

## What is certified, and what is not

VOUCH certifies the residual advantage extractable **through a declared score class**,
on the **committed canary cohort**, against the **deployed model**, under an
**honest-but-verifiable** provider.

- It does not certify erasure in parameter space, which no behavioural method can.
- Three advantages are kept apart: the realised cohort advantage is certified exactly; the
  coin-averaged (mechanism-level) advantage is reached only through a stated inflation
  under bounded coin influence; the full super-population advantage is not certified.
- The certificate is per cohort, not per deletion request.
- Attacks from outside the score class can exceed the tolerance; enlarging the class costs
  power but not validity.
- A provider who special-cases the cohort defeats the audit; canary detectability is
  measured in the paper.
- Canaries must be planted before fine-tuning, so existing models cannot be retro-certified.
- Unlearning is applied to LoRA adapters (rank 16) in the real-model tiers.

## Citation

```bibtex
@article{dang2026vouch,
  author  = {Dang, Quang-Vinh},
  title   = {{VOUCH}: Anytime-Valid Certificates for Machine Unlearning via
             Paired-Canary Hypothesis-Betting},
  journal = {Machine Learning},
  publisher = {Springer},
  year    = {2026},
  note    = {Accepted}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
