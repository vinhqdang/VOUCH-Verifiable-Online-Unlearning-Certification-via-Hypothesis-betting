#!/usr/bin/env python3
"""Simulation tier of the VOUCH experiments (Section 6, metrics M1-M3, M6).

The simulation tier draws pair signs/score-differences directly from their
generating distribution -- exactly the object Theorem 1 controls -- so
validity and power can be calibrated over thousands of seeds, including
under *adversarial optional stopping* (peek after every pair, stop at first
crossing), the regime where fixed-n baselines break.

Experiments
  validity   M1: false-certification / false-revocation rates vs alpha,
             full-protocol validity, calibration curves, and the
             fixed-n-with-peeking inflation of binomial baselines.
  soundness  Empirical demonstration that the design-doc v1.0 certificate
             arm (mixture-weighted signs) is anytime-INVALID under the
             composite null, while the per-score-min arm holds.
  power      M2: certification time tau* vs eps and alpha under exact
             unlearning; revocation detection time vs true advantage;
             dose-response.
  tightness  M3: CS upper bound vs ground truth over time.
  streaming  M6: K=10 deletion waves, global e-process composition.
  ablation   Betting strategies (ONS/aGRAPA/fixed/mixture/KT); sign-only vs
             magnitude-aware revocation.

Usage: python3 experiments/run_simulation.py [--exp all] [--seeds 2000]
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.baselines.fixed_n import peeking_first_rejection
from vouch.verify import (BettingCS, MixtureEProcess, OneSidedEProcess,
                          SymmetryEProcess, VouchConfig, VouchVerifier)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def save(name: str, obj) -> None:
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")


# ----------------------------------------------------------------------------
# M1 validity
# ----------------------------------------------------------------------------

def _validity_one(args):
    seed, p_true, p0, alpha, n_pairs, arm = args
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, p_true, size=n_pairs)
    if arm == "cert":
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture", alpha=alpha)
    else:
        ep = OneSidedEProcess(m0=0.5, direction="above", strategy="mixture", alpha=alpha)
    for t, zz in enumerate(z, 1):
        ep.update(zz)
        if ep.rejects():
            return 1, t
    return 0, n_pairs


def exp_validity(n_seeds: int, pool: Pool) -> None:
    out = {}
    n_pairs = 1024
    for alpha in (0.05, 0.01):
        # false certification: truth sits ON the null boundary p = p0 (worst case)
        for eps in (0.05, 0.10):
            p0 = 0.5 + eps / 2
            res = pool.map(_validity_one,
                           [(s, p0, p0, alpha, n_pairs, "cert") for s in range(n_seeds)])
            rate = float(np.mean([r[0] for r in res]))
            out[f"false_cert_rate/eps={eps}/alpha={alpha}"] = rate
            print(f"  false-cert  eps={eps} alpha={alpha}: {rate:.4f} (nominal {alpha})")
        # false revocation: exact unlearning p = 1/2
        res = pool.map(_validity_one,
                       [(s + 10 ** 6, 0.5, 0.5, alpha, n_pairs, "rev") for s in range(n_seeds)])
        rate = float(np.mean([r[0] for r in res]))
        out[f"false_revocation_rate/alpha={alpha}"] = rate
        print(f"  false-rev   alpha={alpha}: {rate:.4f} (nominal {alpha})")

        # fixed-n binomial with peeking (the invalid baseline usage)
        rng = np.random.default_rng(1234)
        inflate_rev = np.mean([
            peeking_first_rejection(rng.binomial(1, 0.5, size=n_pairs), alpha, "rev")[0]
            for _ in range(n_seeds)])
        inflate_cert = np.mean([
            peeking_first_rejection(rng.binomial(1, 0.55, size=n_pairs), alpha, "cert", p0=0.55)[0]
            for _ in range(n_seeds)])
        out[f"peeking_binomial_false_alarm/alpha={alpha}"] = float(inflate_rev)
        out[f"peeking_binomial_false_cert/alpha={alpha}"] = float(inflate_cert)
        print(f"  peeking binomial false-alarm alpha={alpha}: {inflate_rev:.3f}  "
              f"false-cert: {inflate_cert:.3f}   <-- inflation")

    # calibration curve at boundary, alpha grid
    alphas = [0.01, 0.02, 0.05, 0.10, 0.20]
    curve = []
    for a in alphas:
        res = pool.map(_validity_one,
                       [(s + 2 * 10 ** 6, 0.55, 0.55, a, n_pairs, "cert")
                        for s in range(n_seeds)])
        curve.append(float(np.mean([r[0] for r in res])))
    out["calibration/alphas"] = alphas
    out["calibration/realized"] = curve
    print(f"  calibration: nominal {alphas} -> realized {[round(c,4) for c in curve]}")
    save("sim_validity", out)


# ----------------------------------------------------------------------------
# Full-protocol validity (all three arms + CS, exactly as shipped)
# ----------------------------------------------------------------------------

def _protocol_one(args):
    seed, mu, n_pairs, eps, alpha = args
    rng = np.random.default_rng(seed)
    cfg = VouchConfig(eps=eps, alpha=alpha)
    v = VouchVerifier(["loss", "mink", "ratio"], cfg)
    # correlated scores: shared pair effect + score noise (realistic F)
    base = rng.standard_normal(n_pairs) + mu
    diffs = [{"loss": float(b + 0.3 * rng.standard_normal()),
              "mink": float(b + 0.3 * rng.standard_normal()),
              "ratio": float(b + 0.3 * rng.standard_normal())} for b in base]
    cert = v.run(diffs, shuffle_seed=seed, early_stop=True)
    covered = True  # CS covers truth? p per score identical = P(D>0)
    from scipy.stats import norm
    p_true = 1 - norm.cdf(0, loc=mu, scale=math.sqrt(1 + 0.09))
    for s in ("loss", "mink", "ratio"):
        lo, hi = v.cs[s].interval
        if not (lo - 1e-12 <= p_true <= hi + 1e-12):
            covered = False
    return cert.status, cert.t_stop, covered


def exp_protocol_validity(n_seeds: int, pool: Pool) -> None:
    out = {}
    # exact unlearning: mu = 0 -> p = 1/2; false revocation + CS coverage
    res = pool.map(_protocol_one, [(s, 0.0, 512, 0.10, 0.05) for s in range(n_seeds)])
    statuses = [r[0] for r in res]
    out["exact_unlearning/revoked_rate"] = float(np.mean([s == "REVOKED" for s in statuses]))
    out["exact_unlearning/issued_rate"] = float(np.mean([s == "ISSUED" for s in statuses]))
    out["exact_unlearning/cs_coverage"] = float(np.mean([r[2] for r in res]))
    out["exact_unlearning/median_tau"] = float(np.median([r[1] for r in res if r[0] == "ISSUED"]) if any(s == "ISSUED" for s in statuses) else -1)
    print(f"  exact unlearning (512 pairs): issued {out['exact_unlearning/issued_rate']:.3f}, "
          f"false-revoked {out['exact_unlearning/revoked_rate']:.4f}, "
          f"CS coverage {out['exact_unlearning/cs_coverage']:.4f}")
    # residual memorization mu = 0.5 (p ~ 0.68): revocation power
    res = pool.map(_protocol_one, [(s + 10 ** 6, 0.5, 512, 0.10, 0.05) for s in range(n_seeds // 2)])
    statuses = [r[0] for r in res]
    taus = [r[1] for r in res if r[0] == "REVOKED"]
    out["memorized/revoked_rate"] = float(np.mean([s == "REVOKED" for s in statuses]))
    out["memorized/false_issue_rate"] = float(np.mean([s == "ISSUED" for s in statuses]))
    out["memorized/median_detection_time"] = float(np.median(taus)) if taus else -1
    print(f"  memorized (mu=0.5): revoked {out['memorized/revoked_rate']:.3f} "
          f"median detection {out['memorized/median_detection_time']} pairs, "
          f"false-issue {out['memorized/false_issue_rate']:.4f}")
    save("sim_protocol_validity", out)


# ----------------------------------------------------------------------------
# Soundness of the certificate arm: doc-v1.0 mixture vs per-score min
# ----------------------------------------------------------------------------

def _soundness_one(args):
    seed, n_pairs, p0, alpha = args
    rng = np.random.default_rng(seed)
    # composite null: score A violates (p = p0), scores B, C are clean (1/2)
    zA = rng.binomial(1, p0, size=n_pairs)
    zB = rng.binomial(1, 0.5, size=n_pairs)
    zC = rng.binomial(1, 0.5, size=n_pairs)
    # (i) per-score-min arm (VOUCH, sound)
    eps_min = [OneSidedEProcess(m0=p0, direction="below", strategy="mixture", alpha=alpha)
               for _ in range(3)]
    # (ii) design-doc arm: bet on the exponentially-weighted mixture sign
    doc = OneSidedEProcess(m0=p0, direction="below", strategy="mixture", alpha=alpha)
    logw = np.zeros(3)
    thr = math.log(1 / alpha)
    min_crossed = doc_crossed = 0
    for t in range(n_pairs):
        zs = np.array([zA[t], zB[t], zC[t]], dtype=float)
        for e, z in zip(eps_min, zs):
            e.update(z)
        if min(e.log_e for e in eps_min) >= thr:
            min_crossed = 1
        w = np.exp(logw - logw.max()); w /= w.sum()
        zbar = float(np.dot(w, zs))
        doc.update(zbar)
        # weight update toward the score that looks most "unlearned"
        # (discriminating low): mirrors the doc's adaptive reallocation
        logw += np.log(np.where(zs > 0.5, 0.45, 0.55) / 0.5)
        if doc.log_e >= thr:
            doc_crossed = 1
    return min_crossed, doc_crossed


def exp_soundness(n_seeds: int, pool: Pool) -> None:
    res = pool.map(_soundness_one, [(s, 2048, 0.55, 0.05) for s in range(n_seeds)])
    min_rate = float(np.mean([r[0] for r in res]))
    doc_rate = float(np.mean([r[1] for r in res]))
    out = {
        "null": "score A at p = p0 = 0.55 (violating), B and C at 1/2",
        "per_score_min_false_cert_rate": min_rate,
        "doc_mixture_false_cert_rate": doc_rate,
        "alpha": 0.05, "n_pairs": 2048,
    }
    print(f"  false-cert under composite null: per-score-min {min_rate:.4f} "
          f"(sound, <= 0.05) vs doc-mixture {doc_rate:.4f} (INVALID)")
    save("sim_soundness", out)


# ----------------------------------------------------------------------------
# M2 power / certification time
# ----------------------------------------------------------------------------

def _power_one(args):
    seed, p_true, p0, alpha, n_max, arm, strategy = args
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, p_true, size=n_max)
    ep = OneSidedEProcess(m0=p0 if arm == "cert" else 0.5,
                          direction="below" if arm == "cert" else "above",
                          strategy=strategy, alpha=alpha)
    for t, zz in enumerate(z, 1):
        ep.update(zz)
        if ep.rejects():
            return t
    return -1  # censored


def exp_power(n_seeds: int, pool: Pool) -> None:
    out = {}
    n_max = 20000
    # certification time under exact unlearning vs eps
    for eps in (0.02, 0.05, 0.10, 0.20):
        for alpha in (0.05, 0.01):
            p0 = 0.5 + eps / 2
            taus = pool.map(_power_one, [(s, 0.5, p0, alpha, n_max, "cert", "mixture")
                                         for s in range(n_seeds // 4)])
            taus = np.array(taus)
            med = float(np.median(taus[taus > 0])) if (taus > 0).any() else -1
            frac = float(np.mean(taus > 0))
            kl = 0.5 * math.log(0.5 / p0) + 0.5 * math.log(0.5 / (1 - p0))
            theory = math.log(1 / alpha) / abs(kl)
            out[f"tau/eps={eps}/alpha={alpha}"] = {
                "median": med, "issued_frac": frac, "kl_theory": theory,
                "q25": float(np.percentile(taus[taus > 0], 25)) if (taus > 0).any() else -1,
                "q75": float(np.percentile(taus[taus > 0], 75)) if (taus > 0).any() else -1,
            }
            print(f"  cert time eps={eps} alpha={alpha}: median {med:.0f} "
                  f"(KL theory {theory:.0f}), issued {frac:.2f}")
    # revocation detection time vs true advantage (dose-response)
    for delta in (0.05, 0.10, 0.20, 0.40):
        p = 0.5 + delta / 2
        taus = pool.map(_power_one, [(s, p, 0.5, 0.05, n_max, "rev", "mixture")
                                     for s in range(n_seeds // 4)])
        taus = np.array(taus)
        med = float(np.median(taus[taus > 0])) if (taus > 0).any() else -1
        out[f"detect/delta={delta}"] = {"median": med, "detected_frac": float(np.mean(taus > 0))}
        print(f"  revocation detect Delta={delta}: median {med:.0f} pairs, "
              f"rate {np.mean(taus > 0):.2f}")
    save("sim_power", out)


# ----------------------------------------------------------------------------
# M3 tightness: CS upper bound vs truth
# ----------------------------------------------------------------------------

def _tightness_one(args):
    seed, p_true, n_pairs = args
    rng = np.random.default_rng(seed)
    cs = BettingCS(alpha=0.05, grid=1001)
    ups = []
    for z in rng.binomial(1, p_true, size=n_pairs):
        cs.update(z)
        ups.append(cs.hi)
    return ups


def exp_tightness(n_seeds: int, pool: Pool) -> None:
    out = {}
    n_pairs = 1024
    checkpoints = [16, 32, 64, 128, 256, 512, 1024]
    for p in (0.5, 0.6):
        res = pool.map(_tightness_one, [(s, p, n_pairs) for s in range(min(n_seeds, 500))])
        arr = np.array(res)  # seeds x time
        med = np.median(arr, axis=0)
        out[f"p={p}/checkpoints"] = checkpoints
        out[f"p={p}/median_upper"] = [float(med[c - 1]) for c in checkpoints]
        print(f"  CS upper (p={p}): " +
              ", ".join(f"t={c}: {med[c-1]:.3f}" for c in checkpoints))
    save("sim_tightness", out)


# ----------------------------------------------------------------------------
# M6 streaming composition
# ----------------------------------------------------------------------------

def _streaming_one(args):
    """One deletion history of K waves.

    Per-wave revocation alarms use alpha-spending  alpha_k = alpha * 2^-(k+1)
    (valid for unbounded K, family-wise error <= alpha over the history);
    the global product e-process gives a second, history-wide alarm that can
    accumulate distributed sub-threshold leakage no single wave reveals.
    """
    seed, k_waves, m, scenario, alpha = args
    rng = np.random.default_rng(seed)
    log_rev_glob = 0.0
    thr = math.log(1 / alpha)
    per_wave_alarm, statuses, global_alarm_wave = [], [], -1
    # eps = 0.2 certificates; mu -> Delta via Delta = 2*Phi(mu) - 1
    for k in range(k_waves):
        if scenario == "all_exact":
            mu = 0.0                          # Delta = 0
        elif scenario == "one_bad":
            mu = 0.319 if k == 6 else 0.0     # Delta ~ 0.25 > eps in bad wave
        elif scenario == "all_weak":
            mu = 0.076                        # Delta ~ 0.06 in EVERY wave
        else:
            raise ValueError(scenario)
        cfg = VouchConfig(eps=0.20, alpha=alpha)
        v = VouchVerifier(["loss"], cfg)
        diffs = [{"loss": float(d)} for d in rng.standard_normal(m) + mu]
        cert = v.run(diffs, shuffle_seed=seed * 100 + k, early_stop=False)
        statuses.append(cert.status)
        alpha_k = alpha * 2.0 ** (-(k + 1))
        per_wave_alarm.append(cert.log_e_rev >= math.log(1 / alpha_k))
        log_rev_glob += cert.log_e_rev
        if global_alarm_wave < 0 and log_rev_glob >= thr:
            global_alarm_wave = k
    all_pass = all(s == "ISSUED" for s in statuses)
    return per_wave_alarm, all_pass, global_alarm_wave, statuses


def exp_streaming(n_seeds: int, pool: Pool) -> None:
    out = {}
    n = min(n_seeds, 500)
    for scenario in ("all_exact", "one_bad", "all_weak"):
        res = pool.map(_streaming_one,
                       [(s, 10, 512, scenario, 0.05) for s in range(n)])
        waves = np.array([r[0] for r in res])
        glob = np.array([r[1] for r in res])
        alarm = np.array([r[2] for r in res])
        clean_idx = [i for i in range(10) if not (scenario == "one_bad" and i == 6)]
        if scenario != "all_weak":
            out[f"{scenario}/fwer_per_wave_alarms"] = float(waves[:, clean_idx].any(axis=1).mean())
        out[f"{scenario}/global_certified_rate_allpass"] = float(glob.mean())
        out[f"{scenario}/global_rev_alarm_rate"] = float((alarm >= 0).mean())
        if scenario == "one_bad":
            out["one_bad/bad_wave_alarm_rate"] = float(waves[:, 6].mean())
            bad_issued = np.array([r[3][6] == "ISSUED" for r in res])
            out["one_bad/bad_wave_false_issue_rate"] = float(bad_issued.mean())
        if scenario == "all_weak":
            out["all_weak/any_per_wave_alarm_rate"] = float(waves.any(axis=1).mean())
            fired = alarm[alarm >= 0]
            out["all_weak/median_alarm_wave"] = float(np.median(fired)) if len(fired) else -1
        print(f"  streaming {scenario}: "
              f"{json.dumps({k.split('/',1)[1]: round(v,4) for k, v in out.items() if k.startswith(scenario)})}")
    save("sim_streaming", out)


# ----------------------------------------------------------------------------
# Ablations: betting strategies; sign vs magnitude revocation
# ----------------------------------------------------------------------------

def _mag_one(args):
    seed, mu, frac_memorized, n_max = args
    rng = np.random.default_rng(seed)
    sign_ep = OneSidedEProcess(m0=0.5, direction="above", strategy="mixture", alpha=0.05)
    mag_ep = SymmetryEProcess(alpha=0.05)
    t_sign = t_mag = -1
    for t in range(1, n_max + 1):
        # heterogeneous leakage: a fraction of pairs strongly memorized,
        # the rest exactly unlearned -- magnitude carries the signal
        if rng.random() < frac_memorized:
            d = abs(rng.standard_normal()) * 3.0 + mu
        else:
            d = rng.standard_normal()
        z = float(d > 0)
        sign_ep.update(z)
        mag_ep.update(d, tie_break=rng.random())
        if t_sign < 0 and sign_ep.rejects():
            t_sign = t
        if t_mag < 0 and mag_ep.rejects():
            t_mag = t
        if t_sign > 0 and t_mag > 0:
            break
    return t_sign, t_mag


def exp_ablation(n_seeds: int, pool: Pool) -> None:
    out = {}
    n_max = 20000
    # strategy comparison, cert time under exact unlearning, eps = 0.1
    for strat in ("ons", "agrapa", "fixed", "mixture", "kt"):
        taus = pool.map(_power_one, [(s, 0.5, 0.55, 0.05, n_max, "cert", strat)
                                     for s in range(min(n_seeds, 500))])
        taus = np.array(taus)
        med = float(np.median(taus[taus > 0])) if (taus > 0).any() else -1
        out[f"strategy/{strat}/median_tau"] = med
        out[f"strategy/{strat}/issued_frac"] = float(np.mean(taus > 0))
        print(f"  strategy {strat}: median tau {med:.0f}, issued {np.mean(taus>0):.2f}")
    # sign vs magnitude revocation under sparse strong leakage
    for frac in (0.05, 0.15):
        res = pool.map(_mag_one, [(s, 2.0, frac, 5000) for s in range(min(n_seeds, 500))])
        ts = np.array([r[0] for r in res]); tm = np.array([r[1] for r in res])
        out[f"mag_rev/frac={frac}/sign_median"] = float(np.median(ts[ts > 0])) if (ts > 0).any() else -1
        out[f"mag_rev/frac={frac}/mag_median"] = float(np.median(tm[tm > 0])) if (tm > 0).any() else -1
        out[f"mag_rev/frac={frac}/sign_rate"] = float(np.mean(ts > 0))
        out[f"mag_rev/frac={frac}/mag_rate"] = float(np.mean(tm > 0))
        print(f"  {frac:.0%} memorized: sign detect median {out[f'mag_rev/frac={frac}/sign_median']:.0f} "
              f"(rate {out[f'mag_rev/frac={frac}/sign_rate']:.2f}) vs magnitude "
              f"{out[f'mag_rev/frac={frac}/mag_median']:.0f} (rate {out[f'mag_rev/frac={frac}/mag_rate']:.2f})")
    save("sim_ablation", out)


# ----------------------------------------------------------------------------

EXPERIMENTS = {
    "validity": exp_validity,
    "protocol": exp_protocol_validity,
    "soundness": exp_soundness,
    "power": exp_power,
    "tightness": exp_tightness,
    "streaming": exp_streaming,
    "ablation": exp_ablation,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all"] + list(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    todo = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    with Pool(args.procs) as pool:
        for name in todo:
            print(f"=== {name} (seeds={args.seeds}) ===")
            t0 = time.time()
            EXPERIMENTS[name](args.seeds, pool)
            print(f"=== {name} done in {time.time()-t0:.1f}s ===\n")


if __name__ == "__main__":
    main()
