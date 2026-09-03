#!/usr/bin/env python3
"""Revision-round simulations for VOUCH.

Adds the studies requested at review:

  dependence   The null regimes.  Signs from all canaries pass through one
               jointly trained and unlearned model, so they are neither
               independent nor identically distributed across templates and
               repetition strata.  We compare (i) the v1 *super-population*
               certificate process, whose supermartingale property needs
               E[Z_i | F_{i-1}] >= p0 at every step, against (ii) the
               *finite-cohort* (without-replacement) process, which targets
               the realised cohort mean and is exact under arbitrary
               dependence.  Sign streams are generated i.i.d., heterogeneous
               by stratum, beta-mixed, and with a shared model-level latent.

  cohortnull   Direct verification of the finite-cohort theorem: for fixed
               sign vectors with realised mean exactly p0 (several internal
               structures, including adversarial ones), the realised type-I
               error of the without-replacement process under peeking after
               every pair.

  baselines    Valid sequential comparators on the same observation stream
               and query budget: exact group-sequential alpha-spending
               (Pocock and O'Brien-Fleming boundaries, calibrated by exact
               binomial dynamic programming), a Beta(1/2,1/2)-mixture
               (Jeffreys) e-process, a Robbins normal-mixture e-process,
               and a fixed-n binomial/TOST test at a pre-committed sample
               size.

  power        Certification time versus eps with the *censoring rate*
               reported and a Kaplan-Meier median, replacing the earlier
               median-over-issued-runs which was selection-biased.

Usage:
  python3 experiments/run_simulation_rev.py --exp all --seeds 2000 --procs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.verify import OneSidedEProcess  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def save(name: str, obj) -> None:
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")


def wilson(k: int, n: int, conf: float = 0.95):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ===========================================================================
# 1. Dependence / heterogeneity across pairs
# ===========================================================================

STRATUM_P = (0.40, 0.50, 0.60, 0.70)     # mean 0.55
LATENT_P = (0.30, 0.80)                  # mean 0.55


def _gen_signs(rng, regime: str, n: int, p_bar: float) -> np.ndarray:
    """Sign stream with *marginal* mean ``p_bar`` under four dependence
    regimes.  The verifier reveals pairs in a committed random order, so the
    generated vector is shuffled before it is streamed."""
    if regime == "iid":
        z = rng.binomial(1, p_bar, size=n)
    elif regime == "strata":
        # four equally sized repetition strata with different sign rates
        shift = p_bar - float(np.mean(STRATUM_P))
        ps = np.clip(np.array(STRATUM_P) + shift, 1e-6, 1 - 1e-6)
        blocks = [rng.binomial(1, p, size=n // len(ps)) for p in ps]
        z = np.concatenate(blocks)
        if len(z) < n:
            z = np.concatenate([z, rng.binomial(1, p_bar, size=n - len(z))])
    elif regime == "beta":
        # model-level over-dispersion: theta ~ Beta with mean p_bar, sd ~0.15
        conc = 10.0
        theta = rng.beta(p_bar * conc, (1 - p_bar) * conc)
        z = rng.binomial(1, theta, size=n)
    elif regime == "latent":
        # strong shared latent: the model either leaks a lot or not at all
        shift = p_bar - float(np.mean(LATENT_P))
        ps = np.clip(np.array(LATENT_P) + shift, 1e-6, 1 - 1e-6)
        theta = ps[rng.integers(len(ps))]
        z = rng.binomial(1, theta, size=n)
    else:
        raise ValueError(regime)
    rng.shuffle(z)
    return z.astype(float)


def _dep_one(args):
    seed, regime, n, eps, alpha = args
    rng = np.random.default_rng(seed)
    p0 = 0.5 + eps / 2
    z = _gen_signs(rng, regime, n, p0)          # marginal null: mean = p0
    realised = float(z.mean())
    # (i) v1 super-population process: fixed boundary p0
    sup = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha)
    # (ii) finite-cohort process: without-replacement boundary
    coh = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha, population_size=n)
    thr = math.log(1 / alpha)
    t_sup = t_coh = -1
    for t, zz in enumerate(z, 1):
        sup.update(zz)
        coh.update(zz)
        if t_sup < 0 and sup.log_e >= thr:
            t_sup = t
        if t_coh < 0 and coh.log_e >= thr:
            t_coh = t
    return realised, t_sup, t_coh


def exp_dependence(n_seeds: int, pool: Pool) -> None:
    out = {"description": (
        "Marginal null: mean sign probability = p0 = 0.55 (eps = 0.1) in every "
        "regime.  'sup' is the v1 fixed-boundary process (super-population "
        "target); 'cohort' is the without-replacement process (realised-cohort "
        "target).  For the cohort target an issuance is an error only when the "
        "realised cohort mean is itself >= p0, so both rates are reported.")}
    n, eps, alpha = 1024, 0.10, 0.05
    p0 = 0.5 + eps / 2
    for regime in ("iid", "strata", "beta", "latent"):
        res = pool.map(_dep_one, [(s, regime, n, eps, alpha)
                                  for s in range(n_seeds)])
        realised = np.array([r[0] for r in res])
        t_sup = np.array([r[1] for r in res])
        t_coh = np.array([r[2] for r in res])
        sup_rate = float((t_sup > 0).mean())
        coh_rate = float((t_coh > 0).mean())
        # the cohort null actually holds on this subset
        bad = realised >= p0
        coh_err = float((t_coh[bad] > 0).mean()) if bad.any() else 0.0
        sup_err = float((t_sup[bad] > 0).mean()) if bad.any() else 0.0
        out[regime] = {
            "marginal_mean": p0,
            "realised_mean_sd": float(realised.std()),
            "frac_cohorts_with_mean_ge_p0": float(bad.mean()),
            "sup_issue_rate_marginal_null": sup_rate,
            "sup_issue_rate_ci": list(wilson(int((t_sup > 0).sum()), n_seeds)),
            "cohort_issue_rate_marginal_null": coh_rate,
            "sup_error_rate_cohort_null": sup_err,
            "cohort_error_rate_cohort_null": coh_err,
            "cohort_error_rate_ci": list(wilson(int((t_coh[bad] > 0).sum()),
                                                int(bad.sum()))),
            "n_cohort_null": int(bad.sum()),
        }
        print(f"  {regime:8s} sd(realised)={realised.std():.3f} "
              f"P(cohort mean>=p0)={bad.mean():.2f} | "
              f"sup: marginal {sup_rate:.3f}, cohort-null {sup_err:.3f} | "
              f"cohort: marginal {coh_rate:.3f}, cohort-null {coh_err:.3f}")
    out["n_seeds"] = n_seeds
    out["n_pairs"] = n
    out["alpha"] = alpha
    save("sim_dependence", out)


# ===========================================================================
# 2. Direct verification of the finite-cohort null
# ===========================================================================

def _fixed_cohort_vector(kind: str, n: int, p0: float) -> np.ndarray:
    """A *fixed* sign vector at the least favourable feasible point of the
    cohort null H0: cohort mean >= p0.

    The realised mean is ``ceil(p0 * n) / n``, the smallest attainable value
    at or above ``p0``; taking ``round`` instead would put the vector just
    *below* p0, i.e. outside the null, where the procedure is supposed to
    reject (and does, which is a sharpness check rather than an error)."""
    k = int(math.ceil(p0 * n - 1e-12))
    z = np.zeros(n)
    if kind == "block":
        z[:k] = 1.0                       # all successes contiguous by index
    elif kind == "alternating":
        z[np.linspace(0, n - 1, k).astype(int)] = 1.0
    elif kind == "stratified":
        # four strata with very different rates, overall mean p0
        rates = np.array([0.20, 0.45, 0.65, 0.90])
        rates = rates + (p0 - rates.mean())
        per = n // 4
        z = np.concatenate([
            np.concatenate([np.ones(int(round(r * per))),
                            np.zeros(per - int(round(r * per)))])
            for r in rates])
        if len(z) < n:
            z = np.concatenate([z, np.zeros(n - len(z))])
        # correct rounding drift back to exactly k successes
        cur = int(z.sum())
        if cur > k:
            ones = np.flatnonzero(z)[: cur - k]
            z[ones] = 0.0
        elif cur < k:
            zeros = np.flatnonzero(z == 0)[: k - cur]
            z[zeros] = 1.0
    else:
        raise ValueError(kind)
    assert abs(z.sum() - k) < 1e-9, (z.sum(), k)
    return z


def _cohortnull_one(args):
    seed, kind, n, eps, alpha = args
    rng = np.random.default_rng(seed)
    p0 = 0.5 + eps / 2
    z = _fixed_cohort_vector(kind, n, p0).copy()
    rng.shuffle(z)                                  # committed random order
    coh = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha, population_size=n)
    sup = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha)
    thr = math.log(1 / alpha)
    t_coh = t_sup = -1
    for t, zz in enumerate(z, 1):
        coh.update(zz)
        sup.update(zz)
        if t_coh < 0 and coh.log_e >= thr:
            t_coh = t
        if t_sup < 0 and sup.log_e >= thr:
            t_sup = t
    return t_coh, t_sup


def exp_cohortnull(n_seeds: int, pool: Pool) -> None:
    out = {"description": (
        "Fixed sign vectors with realised cohort mean exactly p0 (least "
        "favourable cohort null), revealed in a uniformly random order; "
        "peeking after every pair.  The without-replacement process must hold "
        "its level for *every* such vector, with no distributional assumption.")}
    alpha = 0.05
    for eps in (0.10, 0.20):
        for kind in ("block", "alternating", "stratified"):
            for n in (384, 1024):
                res = pool.map(_cohortnull_one,
                               [(s, kind, n, eps, alpha) for s in range(n_seeds)])
                tc = np.array([r[0] for r in res])
                ts = np.array([r[1] for r in res])
                key = f"eps={eps}/{kind}/n={n}"
                out[key] = {
                    "cohort_process_error": float((tc > 0).mean()),
                    "cohort_process_error_ci": list(
                        wilson(int((tc > 0).sum()), n_seeds)),
                    "sup_process_error": float((ts > 0).mean()),
                }
                print(f"  {key:28s} cohort {(tc>0).mean():.4f}  "
                      f"sup {(ts>0).mean():.4f}   (alpha {alpha})")
    out["n_seeds"] = n_seeds
    save("sim_cohort_null", out)


# ===========================================================================
# 3. Valid sequential baselines
# ===========================================================================

def _spend_pocock(frac: float, alpha: float) -> float:
    return alpha * math.log(1.0 + (math.e - 1.0) * frac)


def _spend_obf(frac: float, alpha: float) -> float:
    if frac <= 0:
        return 0.0
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return 2.0 * (1.0 - stats.norm.cdf(z / math.sqrt(frac)))


def group_sequential_boundaries(n: int, looks, p0: float, alpha: float,
                                spend, lower: bool = True):
    """Exact alpha-spending boundaries for a binomial partial-sum statistic.

    Calibrated by dynamic programming over the exact lattice under the least
    favourable null ``p = p0``: at look k the critical count ``c_k`` is the
    largest (``lower=True``: reject when S <= c_k) value whose cumulative
    rejection probability does not exceed ``spend(t_k / n)``.  Returns the
    list of critical counts and the realised cumulative spend.
    """
    looks = list(looks)
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0
    prev_t = 0
    spent = 0.0
    crit, cum = [], []
    for t in looks:
        d = t - prev_t
        if d > 0:
            inc = stats.binom.pmf(np.arange(d + 1), d, p0)
            pmf = np.convolve(pmf, inc)[: n + 1]
        prev_t = t
        target = spend(t / n, alpha)
        if lower:
            cdf = np.cumsum(pmf)
            ok = np.flatnonzero(spent + cdf <= target + 1e-15)
            c = int(ok[-1]) if len(ok) else -1
            add = float(cdf[c]) if c >= 0 else 0.0
            if c >= 0:
                pmf[: c + 1] = 0.0
        else:
            sf = np.cumsum(pmf[::-1])[::-1]
            ok = np.flatnonzero(spent + sf <= target + 1e-15)
            c = int(ok[0]) if len(ok) else n + 1
            add = float(sf[c]) if c <= n else 0.0
            if c <= n:
                pmf[c:] = 0.0
        spent += add
        crit.append(c)
        cum.append(spent)
    return crit, cum


def beta_mixture_log_e(s: float, t: int, p0: float, a: float = 0.5,
                       b: float = 0.5, side: str = "below") -> float:
    """Log e-value of the truncated-Jeffreys mixture likelihood ratio.

    Numerator: mixture of Bernoulli likelihoods over a Beta(a,b) prior
    restricted to the alternative side of ``p0`` (renormalised).  Denominator:
    the least favourable null point ``p0``.  Anytime-valid by Ville.
    """
    s = float(s)
    log_num = betaln(a + s, b + t - s) - betaln(a, b)
    # truncate the prior and the posterior to the alternative side
    if side == "below":
        pri = stats.beta.cdf(p0, a, b)
        post = stats.beta.cdf(p0, a + s, b + t - s)
    else:
        pri = stats.beta.sf(p0, a, b)
        post = stats.beta.sf(p0, a + s, b + t - s)
    pri = max(pri, 1e-300)
    post = max(post, 1e-300)
    log_num += math.log(post) - math.log(pri)
    log_den = s * math.log(max(p0, 1e-300)) + (t - s) * math.log(max(1 - p0, 1e-300))
    return log_num - log_den


def normal_mixture_log_e(s: float, t: int, p0: float, rho: float = 1.0) -> float:
    """Robbins normal-mixture e-value for a bounded centred sum.

    ``X_i = p0 - Z_i`` has conditional mean >= 0 under the null and lies in
    [-1, 1]; using the variance proxy v = 1/4 per observation,
    ``M_t = sqrt(rho / (rho + V_t)) exp(S_t^2 / (2 (rho + V_t)))`` with
    ``V_t = t / 4`` is a nonnegative supermartingale started at one.
    """
    st = p0 * t - s          # sum of (p0 - Z_i)
    v = t / 4.0
    return 0.5 * math.log(rho / (rho + v)) + st * st / (2.0 * (rho + v))


def _baseline_one(args):
    (seed, p_true, n, eps, alpha, looks, crit_pocock, crit_obf) = args
    rng = np.random.default_rng(seed)
    p0 = 0.5 + eps / 2
    z = rng.binomial(1, p_true, size=n).astype(float)
    thr = math.log(1 / alpha)
    # -- VOUCH: finite-cohort betting, peek after every pair ----------------
    coh = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha, population_size=n)
    sup = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                           alpha=alpha)
    t_coh = t_sup = t_beta = t_norm = -1
    s = 0.0
    for t, zz in enumerate(z, 1):
        coh.update(zz)
        sup.update(zz)
        s += zz
        if t_coh < 0 and coh.log_e >= thr:
            t_coh = t
        if t_sup < 0 and sup.log_e >= thr:
            t_sup = t
        if t_beta < 0 and beta_mixture_log_e(s, t, p0, side="below") >= thr:
            t_beta = t
        if t_norm < 0 and normal_mixture_log_e(s, t, p0) >= thr:
            t_norm = t
    # -- group sequential: decide only at the K pre-planned looks -----------
    cums = np.cumsum(z)
    t_poc = t_ob = -1
    for k, tk in enumerate(looks):
        stk = cums[tk - 1]
        if t_poc < 0 and stk <= crit_pocock[k]:
            t_poc = tk
        if t_ob < 0 and stk <= crit_obf[k]:
            t_ob = tk
    # -- fixed-n one-look binomial at the pre-committed budget -------------
    p_fixed = stats.binom.cdf(cums[-1], n, p0)
    t_fix = n if p_fixed <= alpha else -1
    return (t_coh, t_sup, t_beta, t_norm, t_poc, t_ob, t_fix,
            float(cums[-1]) / n)


def exp_baselines(n_seeds: int, pool: Pool) -> None:
    out = {"description": (
        "Valid sequential comparators on the same sign stream and the same "
        "query budget n.  VOUCH arms and the two mixture e-processes are "
        "monitored after every pair; the group-sequential procedures decide "
        "only at their K pre-planned looks; the fixed-n test decides once, at "
        "n.  Null row: p = p0 (least favourable).  Alternative rows: exact "
        "unlearning p = 1/2.")}
    alpha = 0.05
    names = ["vouch_cohort", "vouch_sup", "beta_mixture", "normal_mixture",
             "gs_pocock", "gs_obf", "fixed_n"]
    for eps, n in ((0.20, 512), (0.10, 1536)):
        p0 = 0.5 + eps / 2
        K = 10
        looks = [int(round(n * (k + 1) / K)) for k in range(K)]
        cp, cum_p = group_sequential_boundaries(n, looks, p0, alpha, _spend_pocock)
        co, cum_o = group_sequential_boundaries(n, looks, p0, alpha, _spend_obf)
        out[f"eps={eps}/looks"] = looks
        out[f"eps={eps}/pocock_crit"] = cp
        out[f"eps={eps}/obf_crit"] = co
        out[f"eps={eps}/pocock_spend"] = cum_p
        out[f"eps={eps}/obf_spend"] = cum_o
        for label, p_true in (("null_p0", p0), ("exact_unlearning", 0.5)):
            res = pool.map(_baseline_one,
                           [(s + (0 if label == "null_p0" else 10 ** 7),
                             p_true, n, eps, alpha, looks, cp, co)
                            for s in range(n_seeds)])
            arr = np.array(res)
            realised = arr[:, len(names)]
            bad = realised >= p0     # cohort null actually holds here
            block = {"frac_cohorts_with_mean_ge_p0": float(bad.mean())}
            for j, nm in enumerate(names):
                t = arr[:, j]
                rate = float((t > 0).mean())
                med = float(np.median(t[t > 0])) if (t > 0).any() else -1
                block[nm] = {
                    "rate": rate,
                    "rate_ci": list(wilson(int((t > 0).sum()), n_seeds)),
                    "median_t": med,
                    "mean_queries": float(np.where(t > 0, t, n).mean()),
                    "rate_given_cohort_null": (float((t[bad] > 0).mean())
                                               if bad.any() else float("nan")),
                }
            out[f"eps={eps}/{label}"] = block
            print(f"  eps={eps} n={n} {label}:")
            for nm in names:
                b = block[nm]
                print(f"      {nm:15s} rate {b['rate']:.4f} "
                      f"rate|cohort-null {b['rate_given_cohort_null']:.4f} "
                      f"median_t {b['median_t']:8.1f} "
                      f"mean_queries {b['mean_queries']:8.1f}")
        out[f"eps={eps}/n"] = n
    out["n_seeds"] = n_seeds
    out["alpha"] = alpha
    save("sim_baselines", out)


# ===========================================================================
# 4. Power with censoring reported (fixes the selection-biased medians)
# ===========================================================================

def _power_cens_one(args):
    seed, eps, alpha, n_max, target = args
    rng = np.random.default_rng(seed)
    p0 = 0.5 + eps / 2
    z = rng.binomial(1, 0.5, size=n_max).astype(float)
    if target == "cohort":
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=alpha, population_size=n_max)
    else:
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=alpha)
    thr = math.log(1 / alpha)
    for t, zz in enumerate(z, 1):
        ep.update(zz)
        if ep.log_e >= thr:
            return t
    return -1


def km_median(times: np.ndarray, horizon: int):
    """Kaplan-Meier median of the crossing time with right-censoring at the
    horizon.  Returns None when the survival curve never reaches 0.5."""
    ev = np.sort(times[times > 0])
    n = len(times)
    surv, at_risk = 1.0, n
    for t in np.unique(ev):
        d = int((ev == t).sum())
        surv *= (1.0 - d / at_risk)
        at_risk -= d
        if surv <= 0.5:
            return float(t)
    return None


def exp_power_censoring(n_seeds: int, pool: Pool) -> None:
    out = {"description": (
        "Median pairs to certification under exact unlearning with the "
        "censoring rate stated.  The horizon is 20,000 pairs; a median taken "
        "only over issued runs is selection-biased downwards when censoring "
        "is heavy, so the Kaplan-Meier median and the censoring rate are "
        "reported alongside it.  Note that the finite-cohort target's "
        "boundary depends on the cohort size, so its horizon *is* its cohort.")}
    n_max = 20000
    ns = max(n_seeds // 4, 200)
    for target in ("sup", "cohort"):
        for eps in (0.02, 0.05, 0.10, 0.20):
            for alpha in (0.05, 0.01):
                taus = np.array(pool.map(
                    _power_cens_one,
                    [(s, eps, alpha, n_max, target) for s in range(ns)]))
                iss = taus > 0
                p0 = 0.5 + eps / 2
                kl = 0.5 * math.log(0.5 / p0) + 0.5 * math.log(0.5 / (1 - p0))
                theory = math.log(1 / alpha) / abs(kl)
                med_iss = float(np.median(taus[iss])) if iss.any() else -1
                kmm = km_median(taus, n_max)
                out[f"{target}/eps={eps}/alpha={alpha}"] = {
                    "issued_frac": float(iss.mean()),
                    "censoring_rate": float(1 - iss.mean()),
                    "median_over_issued": med_iss,
                    "km_median": kmm,
                    "kl_theory": theory,
                    "q25_over_issued": (float(np.percentile(taus[iss], 25))
                                        if iss.any() else -1),
                    "q75_over_issued": (float(np.percentile(taus[iss], 75))
                                        if iss.any() else -1),
                    "n_seeds": ns,
                }
                print(f"  {target:6s} eps={eps} alpha={alpha}: "
                      f"issued {iss.mean():.2f} (censoring {1-iss.mean():.2f}) "
                      f"median|issued {med_iss:8.0f} KM {kmm} "
                      f"KL {theory:8.0f}")
    save("sim_power_censoring", out)


# ===========================================================================

EXPERIMENTS = {
    "dependence": exp_dependence,
    "cohortnull": exp_cohortnull,
    "baselines": exp_baselines,
    "power": exp_power_censoring,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all"] + list(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    todo = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    with Pool(args.procs) as pool:
        for name in todo:
            print(f"=== {name} (seeds={args.seeds}) ===", flush=True)
            t0 = time.time()
            EXPERIMENTS[name](args.seeds, pool)
            print(f"=== {name} done in {time.time()-t0:.1f}s ===\n", flush=True)


if __name__ == "__main__":
    main()
