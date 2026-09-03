# Round-1 reviews (Springer *Machine Learning*)

## Reviewer 1

Proposes VOUCH, an anytime-valid audit framework for LLM unlearning based on randomized
paired canaries and betting e-processes. Core idea interesting, especially the treatment of
optional stopping and the different composition rules for certificates and alarms.
Substantial experiments. However, a gap between what is defined, what is proved, and what
is claimed. Recommends revision.

1. **The null in Definition 1 does not match the condition used in Theorem 2.** Definition 1
   defines p^(s) = Pr[Z_i^(s)=1], a marginal probability. The proof of Theorem 2 assumes
   E[Z_i^(s) | F_(i-1)] >= p0 (or its lower-tail counterpart) at every step, which is
   stronger. Signs may not be independent because all canaries pass through the same jointly
   trained and unlearned model; they may also differ across templates and repetition strata.
   A marginal bound does not guarantee the supermartingale property. Either redefine the
   certificate using an appropriate conditional-mean null and justify why the setting
   satisfies it, or provide a result valid under the dependence structure considered here.
   State which sources of randomness define p^(s). Simulations with dependent or
   heterogeneous signs would strengthen this.
2. **Clarify the scope of the certificate.** VOUCH certifies the advantage of a declared score
   class on the planted canary population, not that all organic forget records were removed.
   Statements such as "certifies genuine unlearning" sound stronger than the guarantee. Make
   the scope consistent in abstract, introduction, experiments, conclusion. Provide more
   direct evidence that canary results track forgetting on organic records (e.g. compare
   VOUCH outcomes with standard forgetting and membership measurements across methods and
   budgets). The commitment does not prevent a provider who knows the canaries from treating
   them differently; state the honest-but-verifiable assumption whenever the certification
   claim is summarised.
3. **Mathematical and notational inconsistencies.**
   (a) Eq. (2) defines D_i = s(M_u, c_in) - s(M_u, c_ghost), but Fig. 5 appears to use the
   reverse sign (NLL 0.99 / 9.03 reported as D = +8.0). Use one convention throughout theory,
   figures, tables and implementation.
   (b) The small-epsilon expansion in Theorem 3 appears incorrect:
   KL(Bern(1/2)||Bern(1/2+eps/2)) = -1/2 log(1-eps^2) = eps^2/2 + O(eps^4), so certification
   time should be about 2 log(1/alpha)/eps^2, not 8 log(1/alpha)/eps^2. The corrected
   expression is consistent with Table 3 and with "about 600 pairs for eps=0.1, alpha=0.05".
   Correct the theorem and recheck related calculations.
   (c) Clarify whether the confidence-sequence guarantee is per score or simultaneous over F.
4. **Fairer optional-stopping comparison.** The fixed-sample binomial test inspected after
   every pair is not a strong baseline. Add at least one valid sequential baseline
   (alpha-spending, group-sequential, or another Bernoulli confidence-sequence / e-process
   method) under the same observation stream and query budget. A fixed-n comparison at a
   precommitted sample size would also show the practical cost of anytime validity.
5. **Real-model experiments need stronger evidence.**
   (a) Most benchmark results use three seeds while conclusions come from counts such as
   16/18 and 17/18. Report uncertainty; use cautious wording for small run counts.
   (b) SimNPO is discussed but not evaluated. Add SimNPO and, if possible, another recent
   utility-preserving method.
   (c) Held-out NLL is not sufficient for general utility. Report standard TOFU/MUSE
   forgetting and retention measures plus at least one general capability benchmark.
   (d) R-VOUCH would be more convincing if relearning, quantization and jailbreak probes were
   evaluated systematically across the main unlearning methods under clearly specified attack
   budgets.
6. **Related work.** The P1 relearning probe is closely related to Hu et al., "Unlearning or
   Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning," ICLR 2025.
   Also discuss Wang et al., "Towards Effective Evaluations and Comparisons for LLM Unlearning
   Methods," ICLR 2025; Yuan et al., "A Closer Look at Machine Unlearning for Large Language
   Models," ICLR 2025; and Li et al., "LLM Unlearning with LLM Beliefs," ICLR 2026 (probability-
   mass redistribution after unlearning; metrics can report apparent success when semantically
   related information remains). Discuss in relation to VOUCH, not only cite.
7. **Reproducibility and cost.** Define exactly what "retraining from scratch" means in the
   LoRA experiments. Provide key settings: exact model checkpoints, LoRA target modules,
   learning rates, training and unlearning steps, query counts, canary fractions, ordering
   procedure, tie-breaking rules, stopping criteria. Claims such as "roughly sixty times
   cheaper" or "two orders of magnitude cheaper" should rest on clearly matched hardware and
   workloads; report cost assumptions explicitly.

Minor: (1) statements about GDPR and the EU AI Act are broad -- cite primary sources or use
more careful wording. (2) Phrases such as "dissolves all three obstacles," "forgetting by
lobotomy," "what a regulator cares about" are promotional for a journal paper. (3) Some
figures/tables are dense with small text; Table 7 should make clear which models use one seed
and which three.

## Reviewer 2

Summarises the construction accurately. Strengths: the problem is real and well motivated;
framing certification as sequential two-sided equivalence testing is new and the right formal
object; the paired ghost canary is an elegant design that manufactures the null from the
verifier's own randomness, sidestepping the impossibility results; Theorem 2 and Remark 1 are
the most useful contribution for practitioners, and the demonstration that the adaptively
reweighted mixture is a submartingale is a genuine trap.

Weaknesses:

- **W1.** Every end-to-end result on a real model uses eps = 0.2, i.e. an adversary can still
  distinguish in-twin from ghost about 60% of the time. Table 3 shows eps=0.05 costs ~3,000
  pairs and eps=0.02 ~11,000-13,000, neither ever run on an actual LM. The abstract and
  Section 6 claim VOUCH "certifies genuine unlearning" without qualifying the tolerance.
  Validity at meaningful tolerances is shown only in simulation. Close the gap empirically or
  state it plainly wherever the claim appears.
- **W2.** The certificate concerns canaries; the motivating scenario concerns a user's deleted
  record. The introduction opens with GDPR Article 17 and the abstract makes no qualification.
  The dose-response strata calibrate but do not license the extrapolation, and the r=1 end --
  closest to organic data -- is where the measured gap is smallest (1.2 nats). Either give a
  transfer statement under an explicit named assumption, or reposition the contribution as a
  certificate about a planted cohort. The guarantee is per-cohort, not per-request, which the
  introduction's framing implies but the method does not deliver.
- **W3.** Proposition 6 runs one direction only: (eps_u, delta_u)-certified algorithms pass
  VOUCH, but passing VOUCH implies no (eps_u, delta_u) guarantee. Calling it a "bridge" and
  "semantic anchor" invites the wrong reading. State the one-directionality immediately after
  the proposition.
- **W4.** Guarantees are relative to the declared class F, and F is never stress-tested from
  outside. The whole certificate is a supremum over F, so a stronger attack outside F could
  exceed eps on a certified model. The R-VOUCH probes modify the model but re-run the same
  score class. A single experiment -- take a model certified at eps=0.2 and run a genuinely
  stronger membership attack (LiRA-style or learned probe) on the same canary pairs -- would
  tell the reader a great deal.
- **W5.** No empirical comparison against any existing verification method. Section 5.5 asserts
  the alarm catches a GradDiff failure "that a forget-metric evaluation would have passed," but
  the comparison is never run. Add a column showing what forget quality and min-k% conclude on
  the same models and seeds; it is cheap.
- **W6.** Statistical reporting is thin: three seeds per cell, bare I/R/U patterns, no interval
  estimates in Tables 5-7, no dispersion on the mean columns. Counts like "sixteen of eighteen"
  carry wide binomial uncertainty at n=18. Table 7 appears to use a single seed for the larger
  models and shows a markedly weaker picture (U for retrain and/or NPO on Qwen3-0.6B,
  Qwen3-4B, Gemma-4-E2B), yet the text describes the axis as "stable". The honest reading is
  that the framework returns undetermined at the cohort sizes used for the models that matter
  most.
- **W7.** Table 3's starred entries are selection-biased. Starred medians are over runs that
  issued within a 20,000-pair horizon, so both starred entries (11,189 vs limit 14,976; 13,400
  vs 23,021) sit below the information-theoretic bound, which cannot be right. As printed the
  row reads as though VOUCH beats the KL limit. Report the censoring rate and either use a
  survival estimate or mark them as lower bounds.

Points to address in the response:

1. Report end-to-end results at eps <= 0.05 on at least one real model, or state explicitly in
   abstract, introduction and conclusion that all end-to-end certification is at eps = 0.2 and
   discuss what that means operationally. Give guidance on choosing eps; at present it is a
   free parameter with no external meaning.
2. Qualify the canary-to-organic claim in abstract and introduction to match Section 4's scope
   statement; add whatever formal transfer statement is available, even conditional. Clarify in
   the introduction that the certificate is per-cohort rather than per-deletion-request.
3. Add an explicit statement that Proposition 6 is one-directional.
4. Run at least one attack outside the declared class F against a certified model and report
   whether the residual advantage stays below eps. If infeasible, say why, and soften the claim
   that VOUCH bounds "the largest residual advantage any score in the class can extract".
5. Add a head-to-head column comparing VOUCH's verdicts against forget quality and min-k% on
   the same runs, at least for Table 5.
6. Add interval estimates to all verdict counts and dispersion to the mean columns in Tables
   5-7; report the number of seeds used for Table 7; revise the "stable across the axis"
   characterisation.
7. Fix the Table 3 censoring issue.
8. The canaries are templated with fictitious entities and 40 bits of entropy. The paper asserts
   they "do not stand out" but never tests this. Report a detection experiment -- e.g. whether a
   perplexity filter or a small classifier can separate canaries from organic corpus records --
   since a provider who can identify the committed cohort defeats the audit, and the
   honest-but-verifiable threat model is doing a lot of work.
9. Report the number of query prompts Q and how they were chosen. Q affects both power and the
   76-second cost figure and is unspecified. State which configuration produced the 76 s number,
   and give cost as a function of eps and model size.
10. State the tie rate for D_i = 0 empirically. With 4-bit quantised models and degenerate NPO
    outputs, near-ties may be systematic rather than negligible; the committed-coin fix deserves
    a measured justification.
11. Clarify that all fine-tuning and unlearning is applied to LoRA adapters of rank 16, and
    discuss whether adapter-level unlearning is representative of the deployment setting.
12. Frame the 99.6% figure accurately: it is a comparison against the authors' own earlier
    design under an adversarially constructed composite null, not a refutation of any published
    method. The abstract currently reads as the latter.

Minor: cite Carlini et al., The Secret Sharer (USENIX Security 2019) and Jagielski et al.,
Measuring Forgetting of Memorized Training Examples (ICLR 2023), and situate the ghost-twin
design relative to them. Use numerals rather than words for quantities. Section 1 buries the
contribution list under three pages of motivation. Section 5.1: a realised error of 0.031
against nominal 0.05 is expected of a supermartingale and is not evidence of tightness; the
conservatism also costs power. Fig. 2's left panel and the text make the same point twice.
Theorem 3's statement should specify that E[tau*] is under exact unlearning and small eps.
Typo, Section 5.3: "Delta = 0.1 in six hundred and ninety" -- check against Fig. 3.
