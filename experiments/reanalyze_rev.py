#!/usr/bin/env python3
"""Revision-round offline re-analysis of every end-to-end run.

The end-to-end runners store the raw per-pair score differences, and the
canary manifests regenerate deterministically from the run seed, so the
whole verification layer can be replayed offline at zero GPU cost.  This
script produces one JSON, ``results/reanalysis_rev.json``, holding
everything the revised manuscript's tables need:

  verdicts       Verdicts under the finite-cohort (default) and the v1
                 super-population targets, at eps in {0.05, 0.10, 0.20},
                 with per-seed log-wealth, CS bounds and realised sign
                 means -- so the tolerance sweep the reviewers asked for is
                 measured rather than asserted.
  ties           Realised rate of exact ties D = 0 and of near-ties, per
                 model, score and subject.
  heterogeneity  Chi-square tests of sign-rate homogeneity across the
                 repetition strata and across canary templates -- the
                 dependence/heterogeneity the conditional-null discussion
                 turns on, measured on real runs.
  descriptive    Head-to-head against the descriptive metrics VOUCH is
                 argued to replace: a TOFU-style forget-quality analogue
                 (two-sample KS between the unlearned and the retrained
                 model's canary-score distributions) and a MUSE-style
                 min-k% leakage analogue (paired AUC, normalised by the
                 retrained reference), each with its conventional decision
                 rule, plus the same tests used with peeking.
  outofclass     Attacks *outside* the declared score class F run against
                 the same models: a stratum-restricted attack that uses the
                 manifest's repetition labels, and a cross-fitted learned
                 linear combination of the declared scores.  Both preserve
                 Theorem 1, so both come with anytime-valid bounds.
  doseresponse   Mean score gap and sign rate per repetition stratum.
  cost           Scoring seconds, pairs consumed, and queries per pair.

Usage: python3 experiments/reanalyze_rev.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.canaries import PGCGenerator                      # noqa: E402
from vouch.verify import VouchConfig, VouchVerifier          # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")

# tag -> (benchmark, display name, params, year, canary domains, n_seeds_expected)
RUN_SPECS = [
    ("tofu_gpt2",        "TOFU", "GPT-2",       "124M", "2019", ("qa",)),
    ("tofu_pythia160m",  "TOFU", "Pythia-160M", "160M", "2023", ("qa",)),
    ("tofu_phi-1_5",     "TOFU", "Phi-1.5",     "1.4B", "2023", ("qa",)),
    ("muse_gpt2_512",    "MUSE", "GPT-2",       "124M", "2019", ("pii", "fact")),
    ("muse_pythia160m",  "MUSE", "Pythia-160M", "160M", "2023", ("pii", "fact")),
    ("muse_phi-1_5",     "MUSE", "Phi-1.5",     "1.4B", "2023", ("pii", "fact")),
    ("gpt2_v2",          "synthetic", "GPT-2",  "124M", "2019", None),
    ("tiny",             "synthetic", "TinyGPT", "0.9M", "---", None),
    ("qwen3_06b",        "synthetic", "Qwen3-0.6B", "0.6B", "2025", None),
    ("qwen3_4b",         "synthetic", "Qwen3-4B-2507", "4.0B", "2025", None),
    ("phi4_mini",        "synthetic", "Phi-4-mini", "3.8B", "2025", None),
    ("gemma4",           "synthetic", "Gemma-4-E2B", "5.1B", "2026", None),
]

EPS_GRID = (0.05, 0.10, 0.20)
ALPHA = 0.05


def load(tag):
    p = os.path.join(RES, f"lm_e2e_{tag}.json")
    if not os.path.exists(p):
        return None
    runs = json.load(open(p))
    return runs if isinstance(runs, list) else [runs]


def manifest_for(run, domains):
    """Regenerate the run's canary manifest and check it against the run's
    published commitment where one was recorded."""
    if "m_pairs" not in run:
        return None
    kw = {} if domains is None else {"domains": domains}
    man = PGCGenerator(seed=run["seed"], **kw).generate(m=run["m_pairs"], wave=0)
    want = run.get("manifest_sha256")
    if want and man.commitment() != want:
        return None
    return man


def wilson(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper_pearson(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return (float(lo), float(hi))


def verify(diffs, scores, eps, seed, finite_cohort):
    v = VouchVerifier(scores, VouchConfig(eps=eps, alpha=ALPHA,
                                          use_magnitude_revocation=False,
                                          finite_cohort=finite_cohort))
    return v.run(diffs, shuffle_seed=seed, early_stop=True)


# ---------------------------------------------------------------------------
# out-of-class attacks
# ---------------------------------------------------------------------------

def attack_stratum(diffs, reps, order, target_r=8):
    """Restrict the audit to the most-repeated stratum.

    The repetition label r_i is part of the committed manifest, so the
    verifier may use it; no score in the declared class F does.  Conditioning
    on r_i is measurable with respect to the pair's own committed data and
    leaves the within-pair exchangeability of Theorem 1 untouched, so the
    resulting sign stream still carries an exact null under exact unlearning.
    """
    idx = [i for i in order if reps[i] == target_r]
    if not idx:
        return None
    z = np.array([1.0 if diffs[i]["loss"] > 0 else 0.0 for i in idx])
    return z


def attack_learned(diffs, order, scores, n_fit_frac=0.5, seed=0):
    """Cross-fitted learned linear combination of the declared scores.

    A Fisher direction ``w`` is estimated on the first half of the committed
    reveal order and *applied* to the second half, so ``w`` is predictable
    and Theorem 1 still applies to the evaluation half.  A trained
    combination of the members of F is not itself a member of F, so this is
    an attack from outside the declared class.
    """
    d = np.array([[diffs[i][s] for s in scores] for i in order], dtype=float)
    n_fit = max(int(len(d) * n_fit_frac), len(scores) + 2)
    if n_fit >= len(d) - 8:
        return None
    fit, ev = d[:n_fit], d[n_fit:]
    mu = fit.mean(axis=0)
    cov = np.cov(fit, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-9 * np.eye(len(scores))
    try:
        w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        return None
    nrm = np.linalg.norm(w)
    if not np.isfinite(nrm) or nrm == 0:
        return None
    w = w / nrm
    proj = ev @ w
    return (proj > 0).astype(float)


def anytime_bound(z, eps_grid, cohort_n=None):
    """Anytime-valid CS upper bound on the advantage of a sign stream, and
    the smallest eps in the grid at which the stream would still certify."""
    from vouch.verify import BettingCS, OneSidedEProcess
    cs = BettingCS(alpha=ALPHA, grid=1001,
                   population_size=cohort_n if cohort_n else None)
    for zz in z:
        cs.update(float(zz))
    lo, hi = cs.advantage_interval
    out = {"n": int(len(z)), "sign_rate": float(np.mean(z)),
           "delta_hat": float(2 * np.mean(z) - 1),
           "delta_cs": [float(lo), float(hi)]}
    for eps in eps_grid:
        p0 = 0.5 + eps / 2
        up = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=ALPHA,
                              population_size=cohort_n if cohort_n else None)
        dn = OneSidedEProcess(m0=1 - p0, direction="above", strategy="mixture",
                              alpha=ALPHA,
                              population_size=cohort_n if cohort_n else None)
        for zz in z:
            up.update(float(zz))
            dn.update(float(zz))
        out[f"certifies_at_eps={eps}"] = bool(
            min(up.log_e, dn.log_e) >= math.log(1 / ALPHA))
    return out


# ---------------------------------------------------------------------------

def main():
    out = {"alpha": ALPHA, "eps_grid": list(EPS_GRID), "runs": {}}
    tie_tot = defaultdict(lambda: [0, 0])
    het_rows, dose = [], defaultdict(list)

    for tag, bench, name, params, year, domains in RUN_SPECS:
        runs = load(tag)
        if not runs:
            continue
        rec = {"benchmark": bench, "model": name, "params": params,
               "year": year, "n_seeds": len(runs), "seeds": [], "methods": {}}
        for run in runs:
            seed = run["seed"]
            man = manifest_for(run, domains)
            reps = np.array([p.repetition for p in man.pairs]) if man else None
            doms = ([p.domain for p in man.pairs]) if man else None
            rec["seeds"].append(seed)
            for method, cert in run["certs"].items():
                diffs = cert.get("pair_diffs")
                if not diffs:
                    continue
                scores = cert.get("score_class") or list(diffs[0].keys())
                m = len(diffs)
                entry = rec["methods"].setdefault(method, [])
                row = {"seed": seed, "m_pairs": m,
                       "utility_nll": cert.get("utility_nll"),
                       "mean_loss_diff": cert.get("mean_loss_diff"),
                       "scoring_seconds": cert.get("scoring_seconds"),
                       "score_class": scores}
                # -- verdicts at each eps, both targets ----------------------
                for eps in EPS_GRID:
                    for tgt, fc in (("cohort", True), ("sup", False)):
                        c = verify(diffs, scores, eps, seed, fc)
                        row[f"{tgt}/eps={eps}"] = {
                            "status": c.status,
                            "t_stop": c.t_stop,
                            "t_revoked": c.t_revoked,
                            "log_e_cert": c.log_e_cert,
                            "log_e_rev": c.log_e_rev,
                            "log_e_rev_max": c.log_e_rev_max,
                            "delta_upper": c.delta_upper,
                            "delta_cs": c.delta_cs,
                        }
                # -- realised sign means and ties ---------------------------
                sr, ties, near = {}, {}, {}
                for s in scores:
                    d = np.array([x[s] for x in diffs], dtype=float)
                    sr[s] = float(np.mean(d > 0) + 0.5 * np.mean(d == 0))
                    ties[s] = float(np.mean(d == 0))
                    near[s] = float(np.mean(np.abs(d) < 1e-6))
                    tie_tot[(name, s)][0] += int((d == 0).sum())
                    tie_tot[(name, s)][1] += len(d)
                row["sign_rate"] = sr
                row["tie_rate"] = ties
                row["near_tie_rate_1e-6"] = near
                # -- heterogeneity across strata and templates --------------
                if reps is not None and len(reps) == m:
                    d = np.array([x["loss"] for x in diffs], dtype=float)
                    zz = (d > 0).astype(int)
                    tab = []
                    for r in sorted(set(reps.tolist())):
                        sel = reps == r
                        tab.append([int(zz[sel].sum()), int((~zz[sel].astype(bool)).sum())])
                        dose[(bench, name, method, int(r))].append(
                            (float(d[sel].mean()), float(zz[sel].mean()), int(sel.sum())))
                    tab = np.array(tab)
                    if tab.min(axis=0).min() >= 0 and tab.sum() > 0:
                        try:
                            chi2, pv, _, _ = stats.chi2_contingency(tab + 0.5)
                        except ValueError:
                            chi2, pv = float("nan"), float("nan")
                        het_rows.append({
                            "benchmark": bench, "model": name, "method": method,
                            "seed": seed, "factor": "repetition",
                            "chi2": float(chi2), "p": float(pv),
                            "rates": [float(a / max(a + b, 1)) for a, b in tab]})
                    if doms is not None and len(set(doms)) > 1:
                        tab2 = []
                        for dm in sorted(set(doms)):
                            sel = np.array([x == dm for x in doms])
                            tab2.append([int(zz[sel].sum()),
                                         int((~zz[sel].astype(bool)).sum())])
                        tab2 = np.array(tab2)
                        try:
                            chi2, pv, _, _ = stats.chi2_contingency(tab2 + 0.5)
                        except ValueError:
                            chi2, pv = float("nan"), float("nan")
                        het_rows.append({
                            "benchmark": bench, "model": name, "method": method,
                            "seed": seed, "factor": "template",
                            "chi2": float(chi2), "p": float(pv),
                            "rates": [float(a / max(a + b, 1)) for a, b in tab2]})
                # -- out-of-class attacks -----------------------------------
                import random as _rnd
                order = list(range(m))
                _rnd.Random(seed).shuffle(order)
                oc = {}
                if reps is not None and len(reps) == m:
                    z8 = attack_stratum(diffs, reps, order, target_r=8)
                    if z8 is not None and len(z8) > 16:
                        oc["stratum_r8"] = anytime_bound(z8, EPS_GRID,
                                                         cohort_n=len(z8))
                if len(scores) >= 2:
                    zl = attack_learned(diffs, order, scores, seed=seed)
                    if zl is not None and len(zl) > 16:
                        oc["learned_combo"] = anytime_bound(zl, EPS_GRID,
                                                            cohort_n=len(zl))
                row["out_of_class"] = oc
                entry.append(row)
        out["runs"][tag] = rec
        print(f"[reanalysed] {tag}: {len(runs)} seed(s), "
              f"{len(rec['methods'])} subject(s)", flush=True)

    # ---- descriptive-metric head-to-head ---------------------------------
    desc = {}
    for tag, bench, name, params, year, domains in RUN_SPECS:
        runs = load(tag)
        if not runs:
            continue
        for run in runs:
            seed = run["seed"]
            certs = run["certs"]
            rt = certs.get("retrain", {}).get("pair_diffs")
            if not rt:
                continue
            d_rt_loss = np.array([x["loss"] for x in rt], dtype=float)
            d_rt_mink = np.array([x.get("mink", x["loss"]) for x in rt], dtype=float)
            auc_rt = float(np.mean(d_rt_mink > 0) + 0.5 * np.mean(d_rt_mink == 0))
            for method, cert in certs.items():
                diffs = cert.get("pair_diffs")
                if not diffs:
                    continue
                d_loss = np.array([x["loss"] for x in diffs], dtype=float)
                d_mink = np.array([x.get("mink", x["loss"]) for x in diffs],
                                  dtype=float)
                # TOFU-style forget quality: KS between the unlearned and the
                # retrained model's canary-score distributions.  Convention:
                # a *large* p-value means "indistinguishable from retrained",
                # i.e. the method passes.
                ks_p = float(stats.ks_2samp(d_loss, d_rt_loss).pvalue)
                # MUSE-style min-k% leakage: paired AUC, expressed relative to
                # the retrained reference as PrivLeak does.
                auc = float(np.mean(d_mink > 0) + 0.5 * np.mean(d_mink == 0))
                privleak = float(100.0 * (auc - auc_rt) / auc_rt) if auc_rt else float("nan")
                # the same min-k% statistic used with peeking after every pair
                z = (d_mink > 0).astype(float)
                peek = False
                for t in range(8, len(z) + 1):
                    if stats.binom.sf(z[:t].sum() - 1, t, 0.5) <= ALPHA:
                        peek = True
                        break
                desc.setdefault(tag, []).append({
                    "seed": seed, "method": method,
                    "forget_quality_ks_p": ks_p,
                    "forget_quality_verdict": "pass" if ks_p > 0.05 else "fail",
                    "mink_auc": auc, "mink_auc_retrain": auc_rt,
                    "privleak_pct": privleak,
                    "mink_fixed_n_leak_p": float(
                        stats.binom.sf(z.sum() - 1, len(z), 0.5)),
                    "mink_peeking_flags_leak": bool(peek),
                })
    out["descriptive"] = desc
    out["heterogeneity"] = het_rows
    out["ties"] = {f"{k[0]}/{k[1]}": {"ties": v[0], "n": v[1],
                                      "rate": v[0] / v[1] if v[1] else float("nan")}
                   for k, v in tie_tot.items()}
    out["dose_response"] = {
        f"{k[0]}/{k[1]}/{k[2]}/r={k[3]}": {
            "mean_D": float(np.mean([a for a, _, _ in v])),
            "sd_D": float(np.std([a for a, _, _ in v])),
            "mean_sign_rate": float(np.mean([b for _, b, _ in v])),
            "n_runs": len(v)}
        for k, v in dose.items()}

    path = os.path.join(RES, "reanalysis_rev.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {path}")

    # ---- console summary --------------------------------------------------
    print("\n--- tie rates (exact D = 0) ---")
    for k, v in sorted(out["ties"].items()):
        print(f"  {k:28s} {v['ties']:6d} / {v['n']:7d} = {v['rate']:.2e}")
    print("\n--- heterogeneity across repetition strata (chi-square) ---")
    for f in ("repetition", "template"):
        ps = [r["p"] for r in het_rows if r["factor"] == f and np.isfinite(r["p"])]
        if ps:
            k = sum(1 for p in ps if p < 0.05)
            print(f"  {f:11s}: {k}/{len(ps)} runs reject homogeneity at 0.05 "
                  f"(median p = {np.median(ps):.3f})")


if __name__ == "__main__":
    main()
