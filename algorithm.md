# VOUCH: Verifiable Online Unlearning Certification via Hypothesis-betting

**Anytime-Valid Statistical Certificates for Machine Unlearning via Paired-Canary Betting**

*Design document v1.0 — July 2026. Target venue: Springer Machine Learning (alt: JMLR, TMLR).*
*Self-contained specification: motivation, literature positioning, theory, algorithm, pseudocode, datasets, baselines, metrics, and implementation guide.*

---

## 0. Abstract

Machine unlearning methods for LLMs (GA, NPO, SimNPO, RMU, task vectors) ship with **no statistical certificate** that forgetting actually occurred. Existing evaluations are fixed-sample hypothesis tests (e.g., TOFU's KS-test "forget quality"), are unsound as membership evidence without controlled randomness (Zhang–Das–Kamath–Tramèr, 2024), break under optional stopping and continual deletion streams, and say nothing about recoverability under relearning or quantization. Cryptographic proofs-of-unlearning verify *computation*, not *statistical forgetting*, and certified (ε, δ)-unlearning does not scale beyond (near-)convex models.

**VOUCH** is, to our knowledge, the first **anytime-valid statistical certification framework for machine unlearning**. The provider plants **paired ghost canaries** (one twin trained-then-forgotten, one twin never seen, assignment by fair coin) at fine-tuning time. After unlearning, a verifier runs a **betting e-process** over the canary pairs: under exact unlearning, the score difference within each pair is symmetric by construction (an *exact, distribution-free, finite-sample* null — no shadow models, no distributional assumptions on the LLM). The e-process yields:

1. an **anytime-valid upper confidence sequence on the residual membership advantage** Δ (the certificate);
2. a **sequential equivalence test** rejecting H₀: Δ ≥ ε to issue an (ε, α)-Forgetting Certificate at a data-dependent stopping time;
3. a dual **revocation process** that raises an alarm if residual memorization persists;
4. **composition across streaming deletion requests** (e-values multiply), giving a continuously maintained public certificate dashboard;
5. an **R-VOUCH robustness arm** that re-certifies after bounded-compute relearning and quantization probes, directly addressing the "unlearning is suppression" critique.

Validity is exact and assumption-free at any stopping time (Ville's inequality); power is characterized by the KL divergence between the realized sign distribution and the null boundary. VOUCH deliberately certifies **extractable residual influence relative to a declared attack class**, which is the strongest target compatible with the behavioral-audit impossibility results of 2025–2026.

---

## 1. Motivation and Novelty Statement

### 1.1 The verification gap (findings of our three research passes)

This design is grounded in three systematic literature scans conducted July 2026:

**Research Pass 1 — SOTA machine unlearning (broad survey).**
Mapped the algorithmic landscape: exact/sharded unlearning (SISA), approximate first- and second-order methods, and the LLM-era toolbox — gradient ascent/GradDiff, **NPO** (Zhang et al., 2024) and **SimNPO** (Fan et al., NeurIPS 2025), **RMU** (Li et al., 2024, WMDP), task-vector negation, localization/editing approaches — plus diffusion concept erasure, GNN unlearning, and benchmarks (TOFU, MUSE, WMDP, SemEval-2025). Central takeaway: the field's adversarial literature (Łucki et al., 2024; 4-bit quantization recovery, ICLR 2025; relearning attacks) shows most practical unlearning is **suppression, not erasure**, and *evaluation* — not optimization — is the bottleneck.

**Research Pass 2 — Formal guarantees and verification (deep dive).**
Traced certified unlearning from Ginart et al. (2019), Guo et al. (2020, (ε, δ)-certified removal), Sekhari et al. (2021), Neel et al. (2021), Ullah et al. (2021) through 2025–2026 attempts to scale to deep nets (Hessian-free approaches, Koloskova et al., DP2Unlearning, Rewind-to-Delete). Structural obstructions: non-convexity invalidates influence-based certificates; O(d³) Hessian costs; bounds are vacuous at LLM scale. On the verification side: **"Verification of Machine Unlearning is Fragile"** (ICML 2024) shows adversarial providers can pass existing verification; **Thudi et al.** show approximate unlearning is not auditable from weights alone; **Zhang–Das–Kamath–Tramèr (2024)** show MIAs *cannot soundly prove training-set membership* without controlled randomness (canaries) — invalidating naive MIA-based audits; **"On the Impossibility of Retrain Equivalence"** (Yu et al., 2025) shows path-dependence makes distributional retrain-equivalence unattainable for multi-stage pipelines; **Tang et al. (2025–26)** show behavioral audits cannot prove *internal* erasure. Poisoning attacks (Marchant et al., AAAI 2022; camouflage/backdoor-on-unlearning) attack the deletion API itself. Cryptographic proofs-of-unlearning (Eisenhofer et al.; TEE-based) prove the *right computation ran*, not that *forgetting statistically holds*.

**Research Pass 3 — Novelty-gap verification and technical foundations (targeted scan up to July 2026).**
We explicitly searched for any prior application of **anytime-valid inference, e-values/e-processes, test martingales, betting-based testing, or confidence sequences to unlearning verification or deletion certification** — including adjacent DP-auditing literature. Conclusion: **the gap is open.** The nearest neighbors, and why VOUCH is not any of them:

| Nearest neighbor | What it does | Why VOUCH is different |
|---|---|---|
| González et al. (2025), sequential/e-value DP auditing; Steinke–Nasr–Jagielski (2023) one-run DP auditing | *Lower-bounds* the privacy parameter ε of a *training* algorithm via attack success | VOUCH *upper-bounds residual influence after unlearning* (a certificate, not an attack), with a paired-symmetry exact null specific to the unlearning problem; different hypothesis structure (equivalence testing), different protocol (ghost twins through the unlearn pipeline) |
| Pandey et al. (2025), hypothesis-testing view of certified unlearning | Fixed-sample tests tied to (ε, δ)-removal in restricted model classes | Not anytime-valid; no optional stopping; no canary soundness; no streaming deletions; not LLM-scale |
| Zhang–Das–Kamath–Tramèr (2024) sound membership testing via canaries | Establishes *soundness requirements* (randomized inclusion) for membership claims | Provides the soundness principle VOUCH builds on; it is a fixed-n membership *proof*, not a sequential *forgetting certificate*, no equivalence null, no advantage CS, no unlearning pipeline |
| TOFU "forget quality" (KS-test) ; MUSE PrivLeak | Fixed-n two-sample tests vs a retrained reference model | Requires retraining a reference (unavailable at scale; and impossible to match by Yu et al. 2025); invalid under optional stopping/sequential deletions; unpaired; no certificate semantics |
| U-LiRA (Hayes et al., 2024), KLoM | Per-sample MIA evaluations with shadow models | Descriptive evaluation, no error-controlled certificate; shadow-model cost; unsound as proof per Zhang–Tramèr |
| Conformal-prediction-based unlearning evaluation (2025) | Fixed-time marginal coverage statements about forgetting scores | Marginal, fixed-n, no anytime validity, no composition across deletion streams, no equivalence certificate |
| Eisenhofer et al., proof-of-unlearning (crypto/TEE) | Verifies the unlearning *computation* was executed | Orthogonal: computational integrity, not statistical forgetting; VOUCH can be layered on top |

**Positioning sentence for the paper:** *To the best of our knowledge, VOUCH is the first framework to formulate unlearning certification as sequential equivalence testing and to deliver exact, finite-sample, anytime-valid certificates of bounded residual influence for LLM-scale unlearning, with soundness inherited from a randomized paired-canary design rather than distributional assumptions or shadow models.*

### 1.2 Why the honest-but-verifiable setting

We do **not** target the malicious-prover setting (where ICML 2024 fragility results and Thudi non-auditability apply); there, only cryptographic attestation helps, and that is a security-venue contribution. In the **honest-but-verifiable** setting — a provider who genuinely runs unlearning but must convince regulators, clients, or an internal audit function with quantified error control — the obstruction is purely statistical, and it is unsolved. This is also the setting regulators actually face under GDPR Art. 17 / the EU AI Act audit provisions, and (relevant to Vietnam-based deployments) Decree-era personal data protection regimes requiring demonstrable erasure.

### 1.3 Design desiderata (from the critiques, up to July 2026)

A publication-worthy certificate must satisfy:

- **D1 (Soundness).** Validity must not rest on MIA heuristics or distributional assumptions about LLM scores → randomized canary inclusion (Zhang–Tramèr) with an *exact* null.
- **D2 (Anytime validity).** Deletion requests stream in; audits stop adaptively; certificates must survive optional stopping and continuous monitoring → e-processes/Ville.
- **D3 (No retrain comparator).** Retraining-from-scratch is unaffordable and, per Yu et al. (2025), not even well-defined for multi-stage pipelines → the null must be *internally generated* (pair symmetry), not referenced to a retrained model.
- **D4 (Certificate, not attack).** Output must be an upper bound on residual influence with coverage guarantees — an equivalence test, not a significance test whose non-rejection is vacuously interpreted.
- **D5 (Robustness-aware).** Must speak to relearning/quantization recovery (Łucki et al.; ICLR 2025) → certified post-probe.
- **D6 (Honest scope).** Must not claim internal/parametric erasure (Tang impossibility) → certify *extractable residual influence w.r.t. a declared attack class* and say so.
- **D7 (Compute-realistic).** No shadow-model farms; O(forward passes) verification.

VOUCH is engineered to satisfy D1–D7 simultaneously; no existing method satisfies more than two.

---

## 2. Problem Setting

### 2.1 Actors and pipeline

- **Provider** fine-tunes a base model $M_0$ on corpus $D = D_{\text{keep}} \cup D_{\text{forget}} \cup C$, where $C$ is a VOUCH canary set (below), obtaining $M_{\text{ft}}$. Later, deletion requests arrive; the provider runs an unlearning algorithm $\mathcal{U}$ (NPO, RMU, …) on forget set $D_{\text{forget}} \cup C_{\text{in}}$ producing $M_u$.
- **Verifier** (external auditor, or the provider's audit function publishing publicly) holds the canary manifest (committed at training time, e.g., hash published before unlearning) and black-box query access to $M_u$. The verifier runs VOUCH and either **issues**, **withholds**, or **revokes** a certificate.
- Honest provider: $\mathcal{U}$ is genuinely executed; the question is *whether it worked and how well*, with error control.

### 2.2 What is certified (and what is not)

Let $\mathcal{F}$ be a declared class of scalar score functions $s: \mathcal{M} \times \mathcal{X} \to \mathbb{R}$ (e.g., negative token-normalized loss, min-k% probability, loss ratio to $M_0$, probe-classifier logits on hidden states). Define the **residual membership advantage** of $M_u$ against $\mathcal{F}$ on the canary population:

$$
\Delta_{\mathcal{F}}(M_u) \;=\; \sup_{s\in\mathcal{F}} \; \Big( 2\,\Pr\big[\, s(M_u, c^{\text{in}}) > s(M_u, c^{\text{out}}) \,\big] - 1 \Big),
$$

the excess probability that the trained-then-forgotten twin scores higher than its never-seen twin. Under exact unlearning $\Delta_{\mathcal{F}} = 0$ for every $\mathcal{F}$ (Theorem 1). VOUCH certifies $\Delta_{\mathcal{F}} \le \varepsilon$ at confidence $1-\alpha$, **uniformly over time**. Per D6, this is a statement about *extractable influence through $\mathcal{F}$*, calibrated to organic data via a dose–response design (§6.7); it is not a claim about parameter-space erasure — which is provably uncertifiable behaviorally.

---

## 3. Background Theory (what the paper's Section 2 must contain)

### 3.1 E-values, e-processes, Ville's inequality

An **e-variable** for null $H_0$ is a nonnegative random variable $E$ with $\mathbb{E}_{P}[E] \le 1$ for all $P \in H_0$. An **e-process** $(E_t)_{t\ge0}$ satisfies $\mathbb{E}_P[E_\tau] \le 1$ for every stopping time $\tau$ and all $P \in H_0$; for filtration-adapted product form $E_t = \prod_{i\le t} e_i$ with $\mathbb{E}[e_i \mid \mathcal{F}_{i-1}] \le 1$, this is a nonnegative supermartingale and **Ville's inequality** gives, for all $P\in H_0$:

$$
\Pr_P\Big[\exists t: E_t \ge 1/\alpha\Big] \le \alpha .
$$

Rejecting when $E_t \ge 1/\alpha$ is therefore valid at **any** data-dependent stopping time — the property that makes continuous certificate dashboards and streaming deletions sound (D2). Independent e-processes **multiply** into a global e-process (composition across deletion rounds).

### 3.2 Betting confidence sequences for bounded means (Waudby-Smith–Ramdas)

For i.i.d. $Z_i \in [0,1]$ with mean $p$, the capital process against candidate mean $m$,

$$
W_t(m) = \prod_{i=1}^{t} \big(1 + \lambda_i(m)\,(Z_i - m)\big), \qquad \lambda_i(m) \in \big[-\tfrac{1}{1-m}, \tfrac{1}{m}\big],
$$

is a nonnegative martingale when $p = m$. The set $\text{CS}_t = \{ m : W_t(m) < 1/\alpha \}$ is a $(1-\alpha)$ **confidence sequence**: $\Pr[\forall t: p \in \text{CS}_t] \ge 1-\alpha$. Predictable bets $\lambda_i$ (ONS, aGRAPA, mixture) achieve near log-optimal growth; width shrinks at $O(\sqrt{\log\log t / t})$.

### 3.3 Sequential (equivalence) testing via one-sided betting

For a one-sided composite null $H_0: p \ge p_0$ on Bernoulli $Z_i$, the process

$$
E_t \;=\; \prod_{i=1}^{t} \big(1 + \lambda_i\,(p_0 - Z_i)\big), \qquad \lambda_i \in [0, \tfrac{1}{1-p_0}) \text{ predictable},
$$

is an e-process for $H_0$ (each factor has conditional mean $\le 1$ under any $p \ge p_0$). Rejection at $1/\alpha$ **certifies $p < p_0$** — this is the sequential analogue of an equivalence/TOST test and is the engine of VOUCH's certificate (D4). The optimal growth rate under truth $p < p_0$ is $\mathrm{KL}(\mathrm{Bern}(p)\,\|\,\mathrm{Bern}(p_0))$, giving expected certification time $\approx \log(1/\alpha)/\mathrm{KL}$.

### 3.4 Soundness requires randomized inclusion (Zhang–Das–Kamath–Tramèr, 2024)

Absent controlled randomness, a high MIA score on $x$ is *not evidence* that $x$ was trained on: distribution shift and model priors confound. Sound inference requires the auditor to control an inclusion bit $b \sim \mathrm{Bern}(1/2)$ deciding whether a *fresh, exchangeable* item enters training, so that under the null the observed statistic's distribution is known *exactly*. VOUCH bakes this in at the pair level and — crucially — routes the included twin through the **entire unlearning pipeline**, which no membership-testing protocol does.

### 3.5 (ε, δ)-unlearning and the advantage bridge

$\mathcal{U}$ is $(\varepsilon_u,\delta_u)$-certified (Guo et al., 2020; Sekhari et al., 2021) if the unlearned and retrained model distributions are $(\varepsilon_u,\delta_u)$-indistinguishable. By the DP hypothesis-testing characterization (Kairouz–Oh–Viswanath), any distinguisher's advantage — hence $\Delta_{\mathcal{F}}$ — is at most $\tfrac{e^{\varepsilon_u}-1}{e^{\varepsilon_u}+1} + \delta_u$. So certified unlearning *implies* a VOUCH certificate should pass (semantic consistency), while VOUCH remains meaningful where (ε, δ)-certificates are unattainable (LLMs). Conversely a VOUCH pass certifies a *necessary* behavioral condition, not the full distributional property — stated honestly per D6.

---

## 4. The VOUCH Framework

### 4.1 Component 1 — Paired Ghost Canaries (PGC), planted at fine-tuning time

**Generator.** A templated generator $G$ emits *exchangeable twin pairs* $(c_i^0, c_i^1)$, $i = 1,\dots,m$: identical natural-language template, two independently sampled secrets (e.g., fictitious person + random 8-digit account code; fictitious fact triples; UUID-bearing sentences). Exchangeability within a pair is by construction: the two twins are i.i.d. draws from the same conditional template distribution.

**Randomized inclusion.** Independent fair coins $b_i \sim \mathrm{Bern}(1/2)$. Twin $c_i^{b_i}$ (the **in-twin**) is inserted into the fine-tuning corpus with repetition factor $r$ (default $r \in \{1, 2, 4, 8\}$ across canary strata for dose–response); twin $c_i^{1-b_i}$ (the **ghost twin**) is never shown to any model. The manifest $\{(c_i^0, c_i^1, b_i, r_i)\}$ is committed (hash published) before unlearning begins.

**Deletion routing.** All in-twins are appended to the forget set of every unlearning request wave they are assigned to, so canaries traverse exactly the same $\mathcal{U}$ pipeline as organic forget data. For streaming deletions, canary cohorts are stratified across waves: cohort $k$ is unlearned in wave $k$.

Defaults: $m = 256$ pairs per cohort; secrets with $\ge 40$ bits entropy; templates matched in length/domain to the forget corpus to minimize utility contamination (canary token share $< 0.05\%$ of fine-tuning tokens).

### 4.2 Component 2 — Score class $\mathcal{F}$ and the betting adversary

Per D6 the certificate is relative to a declared class. Default $\mathcal{F}$:

1. $s_{\text{loss}}$: negative token-normalized NLL of the secret span given the template prefix;
2. $s_{\text{mink}}$: min-k% token log-probability (k = 20);
3. $s_{\text{ratio}}$: $s_{\text{loss}}(M_u,\cdot) - s_{\text{loss}}(M_0,\cdot)$ (base-model calibrated);
4. $s_{\text{probe}}$: logit of a linear probe on layer-ℓ residual stream trained (on disjoint calibration canaries) to detect secret familiarity — targets representation-level leakage that loss-based scores miss (relevant to RMU-style methods).

Rather than Bonferroni over $\mathcal{F}$, VOUCH runs a **learned combination**: the betting adversary maintains an exponentially-weighted mixture over $\mathcal{F}$ (and over bet sizes), reallocating capital toward the score that is empirically most distinguishing. Because mixtures of e-processes are e-processes, validity is exact while power adapts to the *best attack in the class online* — the verifier is literally an adaptive adversary whose failure to get rich is the certificate.

### 4.3 Component 3 — The certification statistic and its exact null

For pair $i$ (aggregating each twin's score over $Q$ fixed query prompts into one scalar per twin), define

$$
D_i^{(s)} = s(M_u, c_i^{\text{in}}) - s(M_u, c_i^{\text{ghost}}), \qquad Z_i^{(s)} = \mathbb{1}\{D_i^{(s)} > 0\} \;(\text{ties broken } \mathrm{Bern}(1/2)).
$$

**Exact null (Theorem 1).** If $\mathcal{U}$ exactly unlearns (or the pair never influences $M_u$), then $M_u \perp b_i$ given the pair, so by within-pair exchangeability $D_i^{(s)} \overset{d}{=} -D_i^{(s)}$ and $Z_i^{(s)} \sim \mathrm{Bern}(1/2)$ — **regardless of the model, the score function, or any distributional assumption**. This is the property that eliminates shadow models (D1, D3, D7): the null is generated by our own coins.

Population target: $p^{(s)} = \Pr[Z_i^{(s)} = 1]$, advantage $\Delta^{(s)} = 2p^{(s)} - 1$, and $\Delta_{\mathcal{F}} = \sup_s \Delta^{(s)}$.

### 4.4 Component 4 — The three coupled processes

Pairs are revealed to the verifier in random order (predictable filtration). With $p_0 = \tfrac{1}{2} + \tfrac{\varepsilon}{2}$ (so $\Delta = \varepsilon \Leftrightarrow p = p_0$):

**(a) Certificate e-process (equivalence test).** Against $H_0^{\text{cert}}: p \ge p_0$ (i.e., $\Delta \ge \varepsilon$: *not sufficiently unlearned*):

$$
E_t^{\text{cert}} = \prod_{i=1}^{t} \Big(1 + \lambda_i^{\text{cert}} \big(p_0 - \bar Z_i\big)\Big),
$$

where $\bar Z_i$ is the mixture-weighted sign across $\mathcal{F}$ and $\lambda_i^{\text{cert}} \in [0, \tfrac{1}{1-p_0})$ by ONS. **Certificate issued** at $\tau^* = \inf\{t: E_t^{\text{cert}} \ge 1/\alpha\}$: "$\Delta_{\mathcal{F}} < \varepsilon$ at level $\alpha$, anytime-valid."

**(b) Advantage confidence sequence (the certificate's quantitative face).** A betting CS $[L_t, U_t]$ for $p$ (WSR, two-sided, $\alpha/2$ each side), reported as $\Delta \in [2L_t - 1, \, 2U_t - 1]$. The published certificate object is $(\varepsilon, \alpha, t, U_t, \text{manifest hash}, \mathcal{F}, \text{probe results})$. This answers "how much residual influence, at most?" rather than the binary pass/fail.

**(c) Revocation e-process (failure alarm).** Against $H_0^{\text{rev}}: p \le \tfrac{1}{2}$ (unlearning succeeded), symmetric betting upward: growth of $E_t^{\text{rev}}$ past $1/\alpha$ is anytime-valid evidence of **residual memorization** → certificate withheld/revoked, unlearning re-run with stronger settings. Processes (a) and (c) run simultaneously; at most one can win under any truth.

### 4.5 Component 5 — Streaming deletions and global certificates

Deletion waves $k = 1, 2, \dots$ each have a canary cohort and produce a per-wave e-process $E_t^{(k)}$. Because cohorts use independent coins, $\prod_k E^{(k)}$ is a global e-process: the provider maintains a **running, publicly monitorable certificate** over the model's whole deletion history, valid under continual unlearning — a regime where every fixed-n test (TOFU KS, MUSE PrivLeak, U-LiRA) is invalid without ad hoc corrections. Wave-level α-investing can be layered for per-wave discoveries with overall error control.

### 4.6 Component 6 — R-VOUCH: certified recoverability probes

Per D5, a suppression-style "pass" must not survive trivial recovery. R-VOUCH re-runs processes (a)–(c) on probed models:

- **P1 (relearn probe):** LoRA fine-tune $M_u$ on $N_{\text{pub}}$ tokens of *public, forget-adjacent* data (never the canaries), budgets $N_{\text{pub}} \in \{10^5, 10^6\}$;
- **P2 (quantization probe):** GPTQ/AWQ 4-bit quantization of $M_u$;
- **P3 (jailbreak-prompt probe):** fixed adversarial prompt wrappers around the canary queries.

The **robust certificate** requires the CS upper bound $U_t^{(j)} \le p_0$ for all probes $j$ (Bonferroni across the fixed, small probe set). This converts the Łucki-style critique from a rebuttal into a measured, certified quantity — to our knowledge the first certificate with an explicit recoverability clause.

### 4.7 Pseudocode

```text
──────────────────────────────────────────────────────────────────────
PROTOCOL VOUCH
──────────────────────────────────────────────────────────────────────
Phase 0 (design time, before fine-tuning)
  for each planned deletion wave k = 1..K:
    generate m twin pairs {(c_i^0, c_i^1)}, coins b_i ~ Bern(1/2),
      repetition strata r_i ∈ {1,2,4,8}
    IN_k  = {c_i^{b_i}};  GHOST_k = {c_i^{1-b_i}}
    publish commit_k = H(manifest_k)          # binds pairs+coins
  fine-tune M_ft on D_keep ∪ D_forget ∪ (∪_k IN_k with repetitions)

Phase 1 (unlearning, wave k)
  M_u ← U(M_ft or previous M_u ; forget = D_forget^k ∪ IN_k)

Phase 2 (certification, wave k)             # black-box queries only
  reveal manifest_k; verify against commit_k
  init E_cert ← 1, E_rev ← 1, wealth grid {W(m)} for CS
  init mixture weights w_s ← 1/|F| over score class F
  for i in random_order(1..m):
      for s in F:  D_i^s ← s(M_u, c_i^in) − s(M_u, c_i^ghost)
                   Z_i^s ← 1{D_i^s > 0} (ties: Bern(1/2))
      Z̄_i ← Σ_s w_s Z_i^s                  # predictable mixture
      update w_s by exponential weights on past discrimination
      E_cert *= 1 + λ_i^c (p0 − Z̄_i)        # λ by ONS in [0, 1/(1−p0))
      E_rev  *= 1 + λ_i^r (Z̄_i − 1/2)       # λ by ONS in [0, 2)
      update CS grid; U_t ← sup{m: W_t(m) < 2/α}
      if E_rev ≥ 1/α:   REVOKE; return FAIL(evidence = E_rev)
      if E_cert ≥ 1/α:  break               # certificate earned early
  run R-VOUCH probes P1–P3; repeat Phase 2 loop per probe
  if E_cert ≥ 1/α and max_j U_t^{(j)} ≤ p0:
      ISSUE cert_k = (ε, α, t, U_t, commit_k, F, probe report)
  GLOBAL certificate: E_glob ← E_glob · E_cert^{(k)}
──────────────────────────────────────────────────────────────────────
```

---

## 5. Theory (results to prove in the paper)

**Theorem 1 (Exact validity, distribution-free).** Under exact unlearning of pair $i$'s in-twin (or under retrain-from-scratch semantics), within-pair exchangeability and $b_i \perp$ everything imply $Z_i \sim \mathrm{Bern}(1/2)$ exactly, for any model, any score in $\mathcal{F}$, any query set. Consequently $E^{\text{rev}}$ is an e-process for the success null and $\Pr[\text{false revocation}] \le \alpha$ at any stopping time. *Proof: symmetry + Ville.*

**Theorem 2 (Certificate coverage).** For any $p \ge p_0$ (residual advantage $\ge \varepsilon$), $\Pr[\exists t: E_t^{\text{cert}} \ge 1/\alpha] \le \alpha$: the probability of ever issuing a false (ε, α)-certificate is at most α, uniformly over stopping rules and over the composite null. Similarly the CS covers $p$ for all $t$ simultaneously w.p. $\ge 1-\alpha$. *Proof: one-sided betting supermartingale + Ville; WSR for the CS.*

**Theorem 3 (Power / certification time).** If the true sign probability is $p < p_0$ and bets are ONS (or mixture) over the best score $s^\star$, then a.s. $\tfrac{1}{t}\log E_t^{\text{cert}} \to \mathrm{KL}(\mathrm{Bern}(p) \| \mathrm{Bern}(p_0))$, hence $\mathbb{E}[\tau^*] = (1+o(1)) \, \log(1/\alpha) / \mathrm{KL}$; for exact unlearning ($p = 1/2$) and small ε, $\mathbb{E}[\tau^*] \approx 8\log(1/\alpha)/\varepsilon^2$ pairs. Regret of the mixture over the oracle best $s \in \mathcal{F}$ is $O(\log|\mathcal{F}|)$ in log-wealth. *Proof: SLLN for log-wealth + ONS regret + universal-portfolio argument.*

**Theorem 4 (Semantic bridge to certified unlearning).** If $\mathcal{U}$ is $(\varepsilon_u,\delta_u)$-certified, then $\Delta_{\mathcal{F}} \le \tfrac{e^{\varepsilon_u}-1}{e^{\varepsilon_u}+1} + \delta_u$; hence for ε above this value, VOUCH issues the certificate with probability $\ge 1 - \alpha - o(1)$ as $m \to \infty$. *Proof: DP hypothesis-testing region (Kairouz et al.) applied to the pair-coin channel.*

**Theorem 5 (Streaming composition).** Products of per-wave e-processes over independent cohorts form an e-process for the intersection null "every certified wave had $\Delta \ge \varepsilon$ falsely"; global type-I error $\le \alpha$ over the entire deletion history, any adaptive schedule. *Proof: closure of e-processes under products/optional continuation.*

**Proposition 6 (Honest limits).** (i) A VOUCH certificate does not imply parametric erasure (consistent with Tang et al. impossibility) — counterexample constructed via weight-space obfuscation invisible to $\mathcal{F}$; (ii) it is relative to $\mathcal{F}$ and the canary distribution; dose–response calibration (§6.7) bounds the extrapolation gap to organic data empirically, not formally. These statements pre-empt the two most likely reviewer objections and scope the claim correctly.

---

## 6. Experimental Design

### 6.1 Datasets (with download sources)

| Dataset | Role | Source |
|---|---|---|
| **TOFU** (fictitious-author QA; forget01/05/10 splits) | Primary controlled unlearning testbed; canaries injected alongside its fine-tuning set | `https://huggingface.co/datasets/locuslab/TOFU` |
| **MUSE-News / MUSE-Books** | Realistic corpus unlearning (BBC news / Harry Potter books); tests VOUCH under verbatim + knowledge memorization | `https://huggingface.co/datasets/muse-bench/MUSE-News`, `https://huggingface.co/datasets/muse-bench/MUSE-Books` (docs: `https://muse-bench.github.io`) |
| **WMDP** (bio/cyber hazardous-knowledge MCQ) | Capability-removal arm; certify RMU-style unlearning with knowledge-style canaries (fictitious hazardous-sounding facts) | `https://huggingface.co/datasets/cais/wmdp` |
| **VOUCH-Canaries (ours, released)** | Twin-pair generator + manifests; secret-sharer-style templated PII, fact triples, and QA canaries; 3 domains × 4 repetition strata | released with code; generator script in repo |
| **Pile-uncopyrighted subset** | Relearn-probe corpus (P1) and retain-set for GradDiff/NPO retain terms | `https://huggingface.co/datasets/monology/pile-uncopyrighted` |
| **UTKML / synthetic logistic + small CNN suite** | Sanity tier where retrain-from-scratch ground truth is affordable → empirical validity calibration at scale (2,000 seeds) | generated in repo |

### 6.2 Models

- **Validity tier (cheap, many seeds):** Pythia-410M, Pythia-1.4B (`EleutherAI/pythia-410m`, `-1.4b`).
- **Main tier:** Llama-3.2-1B/3B (`meta-llama/Llama-3.2-1B`) and the TOFU-standard Llama-2-7B-chat + Phi-1.5 (`meta-llama/Llama-2-7b-chat-hf`, `microsoft/phi-1_5`).
- **Capability tier:** Zephyr-7B-β for WMDP/RMU (`HuggingFaceH4/zephyr-7b-beta`).
- Fine-tuning via LoRA (r=16) and full-FT at ≤1.4B; unlearning via the **open-unlearning** framework (`https://github.com/locuslab/open-unlearning`), which implements the methods below with TOFU/MUSE integration.

### 6.3 Unlearning algorithms to certify (the "subjects")

GA; GradDiff (GA + retain CE); **NPO** and **NPO+retain**; **SimNPO**; **RMU** (WMDP arm); task-vector negation; IdkDPO; and **retrain-from-scratch** (gold standard, validity tier + TOFU-1B only). Also *deliberately weakened* variants (early-stopped NPO at 25/50/75% of epochs; under-scaled RMU coefficient) to create ground-truth partial forgetting for power curves.

### 6.4 Verifier baselines (the "competitors")

1. **TOFU Forget Quality** (fixed-n KS test vs retrained reference) — the field's default statistical claim;
2. **MUSE PrivLeak** (Min-K%-based AUC vs retrain/holdout);
3. **U-LiRA** (Hayes et al., 2024) per-sample shadow-model MIA — strongest descriptive evaluator, 16 shadow models;
4. **Fixed-n paired tests on our own canaries**: exact binomial test, permutation test, and TOST equivalence t-test at n = m (ablates *anytime validity specifically*, holding the canary design fixed — the key ablation separating VOUCH's protocol contribution from its inference contribution);
5. **One-run DP-audit adaptation** (Steinke–Nasr–Jagielski 2023 machinery repurposed to unlearning; plus the González-style sequential DP auditor as nearest sequential neighbor);
6. **Naive MIA panel without canaries** (Min-K%, zlib, loss, ratio) — demonstrates the soundness failure predicted by Zhang–Tramèr (positive "evidence" on never-trained data under distribution shift).

### 6.5 Metrics

- **M1 Validity:** empirical false-certification rate (issue despite planted Δ ≥ ε) and false-revocation rate vs nominal α, over ≥ 2,000 seeded runs (validity tier) and 20 seeds (main tier); evaluated *with adversarial optional stopping* (stop at first crossing, peeking every pair) — where fixed-n baselines demonstrably inflate error and VOUCH does not. Report calibration curves α vs realized error.
- **M2 Power / certification time:** distribution of τ* vs true injected advantage (dose–response strata), vs ε grid {0.02, 0.05, 0.10}, α ∈ {0.05, 0.01}; ROC-style detection curves for the revocation arm on partially unlearned models.
- **M3 Tightness:** CS upper bound U_t vs ground-truth advantage measured by exhaustive querying; regret of mixture betting vs oracle-best score.
- **M4 Robustness (R-VOUCH):** fraction of NPO/RMU "passes" that fail post-P1/P2/P3; recoverability gap ΔU = U_post − U_pre. Hypothesis: RMU passes base but fails P1 at 10⁶ tokens; quantization (P2) flips a substantial fraction of suppression-style passes — turning the Łucki/ICLR-2025 critiques into measured certificate clauses.
- **M5 Compute:** verifier cost in forward passes and wall-clock vs U-LiRA (shadow-model farm) and retrain-based Forget Quality; target ≥ 100× cheaper at 7B.
- **M6 Streaming:** global error over K = 10 sequential deletion waves with adaptive re-unlearning — a regime no baseline supports natively.
- **Utility guardrail:** canary contamination effect on model utility (TOFU model-utility metric, MMLU) must be statistically indistinguishable from canary-free fine-tuning.

### 6.6 Settings

m = 256 pairs/cohort (power analysis: certifies ε = 0.1 at α = 0.05 in ≈ 190 pairs when unlearning is exact; report m = 64/128/512 ablations); Q = 8 query prompts/twin, aggregated by mean; ONS bets with mixture over {loss, mink, ratio, probe}; ties randomized with committed PRNG seed; temperature-0 scoring; all manifests hash-committed pre-unlearning. Hardware: 1–4×A100-80GB; validity tier runs on a single A100 in <1 GPU-day per 100 seeds.

### 6.7 Dose–response calibration (canary→organic extrapolation)

Repetition strata r ∈ {1,2,4,8} give a measured curve Δ(r); organic forget-data influence is bracketed by matching its duplication statistics to the curve. This is the paper's answer to "canaries aren't real data": we do not assume transfer, we measure the transfer curve and report certificates per stratum.

### 6.8 Ablations

Mixture vs single-score betting; ONS vs aGRAPA vs fixed λ; paired-symmetry null vs unpaired two-sample betting (shows the pairing is what buys distribution-freeness); probe scores on/off (representation leakage under RMU); Q sensitivity; canary domain shift (templates unlike forget corpus) to probe D6 limits.

---

## 7. Implementation Guide

```
vouch/
├── canaries/generator.py      # twin-pair templates, secrets, manifests, commitments (sha256)
├── training/inject.py         # HF Trainer hook: insert IN_k with repetition strata
├── unlearn/                   # thin wrappers around open-unlearning (GA/GradDiff/NPO/SimNPO/RMU/task-vector)
├── verify/scores.py           # s_loss, s_mink, s_ratio, s_probe (batched, temperature-0)
├── verify/betting.py          # ONS/aGRAPA bets; E_cert, E_rev, WSR confidence sequence (grid over m∈[0,1], 1e-3 mesh)
├── verify/protocol.py         # Phase-2 loop of §4.7, early stopping, certificate object (JSON + manifest hash)
├── probes/relearn.py          # LoRA on pile subset (P1); quantize.py (GPTQ 4-bit, P2); jailbreak.py (P3)
├── baselines/                 # TOFU-KS, MUSE-PrivLeak, U-LiRA, fixed-n paired tests, one-run DP audit
└── experiments/               # configs for §6; seeds; plotting
```

Core betting update (reference implementation):

```python
def ons_bet(lam, z, m0, g_prev, eta=0.5, lo=0.0, hi=None):
    """One ONS step for E *= 1 + lam*(m0 - z); hi = 1/(1-m0) - 1e-6."""
    g = (m0 - z) / (1 + lam * (m0 - z))          # neg. gradient of -log wealth
    A = g_prev + g * g
    lam = clip(lam + eta * g / A, lo, hi)
    return lam, A

# per pair i:  e = 1 + lam_cert*(p0 - zbar);  E_cert *= e
#              lam_cert, A = ons_bet(lam_cert, zbar, p0, A, hi=1/(1-p0)-1e-6)
```

Dependencies: `torch`, `transformers`, `peft`, `datasets`, `auto-gptq`, open-unlearning (pinned commit). Scoring is pure inference → the verifier can run on a single GPU against a served model; only probes need training compute.

**Certificate JSON** (published artifact): `{eps, alpha, wave, t_stop, E_cert, U_t, Delta_CS, probes: {P1,P2,P3}, manifest_sha256, score_class, code_version}`.

---

## 8. Limitations (stated in the paper, not discovered by reviewers)

1. **Relative to $\mathcal{F}$ and the canary law** — inherent per Tang et al.; mitigated by mixture betting, probe scores, and dose–response calibration; not eliminable by any behavioral method.
2. **Honest-provider assumption** — a malicious provider could special-case committed canaries; the commitment scheme raises the bar but full malicious security needs cryptographic attestation (explicitly out of scope; compose with Eisenhofer-style PoUL).
3. **Design-time requirement** — canaries must exist before unlearning; VOUCH cannot retro-certify legacy models (we discuss a weaker post-hoc variant using held-out duplicates as an appendix, with honesty about its assumptions).
4. **Per-cohort, not per-user** — certifies population-level residual advantage; per-request guarantees would need per-user canaries (costed in appendix).

## 9. Packaging for Springer *Machine Learning*

Title: *"VOUCH: Anytime-Valid Certificates for Machine Unlearning."* Structure: Intro (verification crisis, D1–D7) → Related work (three-pass map, Table 1 nearest-neighbors) → Preliminaries (§3) → Framework (§4) → Theory (§5, proofs in appendix) → Experiments (§6) → Limitations. Contributions bullet list: (i) first anytime-valid formulation of unlearning certification, with an exact distribution-free null from paired ghost canaries; (ii) coupled certificate/CS/revocation e-processes with streaming composition; (iii) first recoverability-certified protocol (R-VOUCH); (iv) theory (validity, power, DP-bridge, composition); (v) large empirical study + released canary suite and verifier. Length ~35–40 pp journal format. Code + certificates released (github.com/vinhqdang/vouch).

## 10. Key References (anchor set)

Guo et al., *Certified Data Removal*, ICML 2020 · Sekhari et al., NeurIPS 2021 · Neel et al., ALT 2021 · Zhang, Das, Kamath, Tramèr, *MIAs cannot prove training-set membership*, 2024 · *Verification of Machine Unlearning is Fragile*, ICML 2024 · Thudi et al., *On the necessity of auditable algorithmic definitions*, USENIX Sec 2022 · Yu et al., *Impossibility of Retrain Equivalence*, 2025 · Łucki et al., *An adversarial perspective on machine unlearning*, 2024 · 4-bit quantization recovery, ICLR 2025 · Zhang et al., *NPO*, 2024 · Fan et al., *SimNPO*, NeurIPS 2025 · Li et al., *WMDP/RMU*, ICML 2024 · Maini et al., *TOFU*, 2024 · Shi et al., *MUSE*, 2024 · Hayes et al., *U-LiRA*, 2024 · Steinke, Nasr, Jagielski, *Privacy auditing with one training run*, NeurIPS 2023 · Waudby-Smith & Ramdas, *Betting CSs*, JRSS-B 2024 · Ramdas et al., *Game-theoretic statistics and safe anytime-valid inference*, Statist. Sci. 2023 · Grünwald, de Heide, Koolen, *Safe testing*, JRSS-B 2024 · Kairouz, Oh, Viswanath, *Composition theorem for DP*, 2015 · Eisenhofer et al., *Verifiable/Proof-of-unlearning*, 2022+ · Marchant et al., *Hard to Forget*, AAAI 2022 · Carlini et al., *Secret Sharer*, USENIX Sec 2019.

*(Exact arXiv IDs/venues for the 2025–2026 items to be re-verified against the three research reports in this conversation before submission.)*
