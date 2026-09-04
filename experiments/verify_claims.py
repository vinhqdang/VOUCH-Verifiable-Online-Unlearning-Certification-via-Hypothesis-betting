#!/usr/bin/env python3
"""Check every quantitative claim in the manuscript against the result files.

Run this after any change to results/ or to the manuscript's numbers.  It exists
because the round-2 review found five prose figures that had silently drifted
away from the tables they described, and four prose passages that stated the
opposite of the table beside them.  A green run here is the cheapest available
protection against repeating that.

Usage:
    python3 experiments/verify_claims.py            # summary
    python3 experiments/verify_claims.py -v         # show every check
Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")

BENCH = ["tofu_gpt2", "tofu_pythia160m", "tofu_phi-1_5",
         "muse_gpt2_512", "muse_pythia160m", "muse_phi-1_5"]

FAILURES: list[str] = []
VERBOSE = False


def load(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def check(claim, ok, actual):
    tag = "OK " if ok else "FAIL"
    if not ok:
        FAILURES.append(f"{claim}: manuscript says otherwise; data gives {actual}")
    if VERBOSE or not ok:
        print(f"  {tag} {claim}  ->  {actual}")


def close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    VERBOSE = ap.parse_args().verbose

    d = load("reanalysis_rev")
    if d is None:
        print("results/reanalysis_rev.json missing; run experiments/reanalyze_rev.py")
        return 1

    print("== benchmark verdict counts (Section 5, Tables 10-12) ==")
    k = sum(1 for t in BENCH for r in d["runs"][t]["methods"]["none"]
            if r["cohort/eps=0.2"]["status"] == "REVOKED")
    check("un-unlearned revoked on 16 of 18 runs", k == 16, f"{k}/18")

    for m, want in (("retrain", 18), ("ga", 18), ("grad_diff", 17), ("npo", 18)):
        k = sum(1 for t in BENCH for r in d["runs"][t]["methods"][m]
                if r["cohort/eps=0.2"]["status"] == "ISSUED")
        check(f"{m} issued on {want} of 18", k == want, f"{k}/18")

    print("== tolerance sweep (Table 11) ==")
    k = sum(1 for t in BENCH for r in d["runs"][t]["methods"]["retrain"]
            if r["cohort/eps=0.05"]["status"] == "ISSUED")
    check("retrain issues 12 of 18 at eps=0.05, cohort target", k == 12, f"{k}/18")
    k = sum(1 for t in BENCH for r in d["runs"][t]["methods"]["retrain"]
            if r["sup/eps=0.1"]["status"] == "ISSUED")
    check("retrain issues 0 of 18 at eps=0.1, super-population", k == 0, f"{k}/18")

    nb = [r for t in BENCH for r in d["runs"][t]["methods"]["retrain"]
          if r["cohort/eps=0.05"]["status"] != "ISSUED"]
    lo = sum(1 for r in nb if r["sign_rate"]["loss"] < 0.5)
    check("5 of the 6 non-issuing retrain runs are blocked by the lower arm",
          (len(nb), lo) == (6, 5), f"{lo} of {len(nb)}")

    print("== recoverability (Section 5.8) ==")
    chg = sum(1 for t in BENCH
              for a, b in zip(d["runs"][t]["methods"].get("npo", []),
                              d["runs"][t]["methods"].get("npo_P1_relearn", []))
              if a["cohort/eps=0.2"]["status"] != b["cohort/eps=0.2"]["status"])
    check("the P1 relearn probe changes 2 of 18 verdicts", chg == 2, f"{chg}/18")

    print("== ties (Section 3.4, 5.8) ==")
    tl = sum(v["ties"] for k2, v in d["ties"].items() if k2.endswith("/loss"))
    tn = sum(v["n"] for k2, v in d["ties"].items() if k2.endswith("/loss"))
    check(f"zero exact ties for s_loss over {tn} scored pairs", tl == 0, f"{tl}/{tn}")
    gem = next((v for k2, v in d["ties"].items()
                if "Gemma" in k2 and k2.endswith("/mink")), None)
    if gem:
        check("2.7% tie rate for min-k on Gemma-4-E2B",
              close(gem["rate"], 0.027, 0.001), f"{gem['rate']:.4f}")

    print("== heterogeneity (Section 5.1) ==")
    het = d["heterogeneity"]
    ps = [r["p"] for r in het if r["factor"] == "repetition" and np.isfinite(r["p"])]
    pc = 100 * sum(1 for p in ps if p < 0.05) / len(ps)
    check("stratum homogeneity rejected in 14% of runs", close(pc, 14, 1.5), f"{pc:.0f}%")
    psn = [r["p"] for r in het if r["factor"] == "repetition"
           and r["method"] == "none" and np.isfinite(r["p"])]
    kn = sum(1 for p in psn if p < 0.05)
    check("rejected in 24 of the un-unlearned runs", kn == 24, f"{kn}/{len(psn)}")

    print("== out-of-class attacks are a NULL result (Section 5.11, Table 16) ==")
    for atk, want_obs in (("stratum_r8", 0.028), ("learned_combo", 0.005)):
        vs = [v for rec in d["runs"].values()
              for m, rows in rec["methods"].items() if m != "none"
              for r in rows if r["cohort/eps=0.2"]["status"] == "ISSUED"
              for k2, v in r.get("out_of_class", {}).items() if k2 == atk]
        if not vs:
            continue
        obs = float(np.mean([abs(v["delta_hat"]) > 0.20 for v in vs]))
        nulls = []
        for v in vs:
            n = int(v["n"])
            kk = np.arange(n + 1)
            nulls.append(stats.binom.pmf(kk, n, 0.5)[np.abs(2 * kk / n - 1) > 0.20].sum())
        nullr = float(np.mean(nulls))
        check(f"{atk} observed exceedance {want_obs}",
              close(obs, want_obs, 0.002), f"{obs:.3f} (n={len(vs)})")
        # the paper's claim is that neither attack beats its own null
        check(f"{atk} does not beat its binomial null",
              obs <= nullr + 0.003, f"observed {obs:.3f} vs null {nullr:.3f}")

    print("== certificate / confidence-sequence incoherence (Section 5.8) ==")
    tot = sum(1 for t in BENCH for rows in d["runs"][t]["methods"].values()
              for r in rows if r["cohort/eps=0.2"]["status"] == "ISSUED")
    bad = sum(1 for t in BENCH for rows in d["runs"][t]["methods"].values()
              for r in rows if r["cohort/eps=0.2"]["status"] == "ISSUED"
              and r["cohort/eps=0.2"]["delta_upper"] >= 0.20)
    check("46 of 107 issued runs have a CS upper bound at or above eps",
          (bad, tot) == (46, 107), f"{bad}/{tot}")

    print("== dose-response (Section 5.8) ==")
    for r_, want in ((1, 0.46), (2, 0.78), (4, 1.54), (8, 2.35)):
        # The claim sits in the benchmark subsection, so it is scoped to the
        # six TOFU/MUSE cells: not the synthetic tiers, not the duplicate
        # revision tag, not the tight-tolerance cohort.
        vals = [v["mean_D"] for k2, v in d["dose_response"].items()
                if k2.endswith(f"/none/r={r_}")
                and k2.split("/")[0] in ("TOFU", "MUSE")
                and "3,072" not in k2 and "rev" not in k2]
        if vals:
            check(f"mean score gap {want:+.2f} nats at r={r_}",
                  close(float(np.mean(vals)), want, 0.02),
                  f"{np.mean(vals):+.3f} over {len(vals)} cells")
    for r_, want in ((1, 0.22), (2, 0.35), (4, 0.59), (8, 0.73)):
        srs = [v["mean_sign_rate"] for k2, v in d["dose_response"].items()
               if k2.endswith(f"/none/r={r_}")
               and k2.split("/")[0] in ("TOFU", "MUSE")
               and "3,072" not in k2 and "rev" not in k2]
        if srs:
            dh = 2 * float(np.mean(srs)) - 1
            check(f"realised advantage {want:+.2f} at r={r_}",
                  close(dh, want, 0.02), f"{dh:+.3f}")

    print("== tight-tolerance cohort (Section 5.9, Table 14) ==")
    rec = d["runs"].get("tofu_gpt2_tight")
    if rec:
        subs = ("none", "retrain", "npo", "simnpo")
        ok = all(r["sup/eps=0.1"]["status"] == "ISSUED"
                 for m in subs for r in rec["methods"][m])
        check("super-population certifies at eps=0.1 on all subjects/seeds", ok, str(ok))
        ok = all(r["sup/eps=0.05"]["status"] == "UNDETERMINED"
                 for m in subs for r in rec["methods"][m])
        check("super-population undetermined at eps=0.05", ok, str(ok))
        ok = all(r["cohort/eps=0.05"]["status"] == "ISSUED"
                 for m in subs for r in rec["methods"][m])
        check("cohort target certifies at eps=0.05 on all subjects/seeds", ok, str(ok))
        dh = [2 * r["sign_rate"]["loss"] - 1 for m in subs for r in rec["methods"][m]]
        check("realised advantages span -0.040 to +0.025",
              close(min(dh), -0.040, 0.002) and close(max(dh), 0.025, 0.002),
              f"{min(dh):+.3f} to {max(dh):+.3f}")
        r1 = d["dose_response"].get("TOFU/GPT-2 (3,072 pairs, $r{=}1$)/none/r=1")
        if r1:
            check("un-unlearned mean gap +0.005 nats at r=1 in the tight cohort",
                  close(r1["mean_D"], 0.005, 0.002), f"{r1['mean_D']:+.3f}")
    else:
        print("  (tight cohort absent from the reanalysis)")

    print("== simulation tiers ==")
    dep = load("sim_dependence")
    if dep:
        check("fixed boundary breaches at 0.389 under beta over-dispersion",
              close(dep["beta"]["sup_issue_rate_marginal_null"], 0.389, 0.01),
              f"{dep['beta']['sup_issue_rate_marginal_null']:.3f}")
        worst = max(dep[r]["cohort_error_rate_cohort_null"]
                    for r in ("iid", "strata", "beta", "latent"))
        check("cohort target holds at or below 0.020 in every regime",
              worst <= 0.021, f"max {worst:.3f}")
    cn = load("sim_cohort_null")
    if cn:
        worst = max(v["cohort_process_error"] for k2, v in cn.items()
                    if isinstance(v, dict) and "cohort_process_error" in v)
        check("cohort-null error never exceeds 0.035", worst <= 0.036, f"max {worst:.3f}")
    pw = load("sim_power_censoring")
    if pw:
        c = pw["sup/eps=0.02/alpha=0.05"]
        check("eps=0.02 censoring 0.45 and KM median above the KL limit",
              close(c["censoring_rate"], 0.45, 0.02) and c["km_median"] > c["kl_theory"],
              f"censoring {c['censoring_rate']:.2f}, KM {c['km_median']:.0f} "
              f"vs limit {c['kl_theory']:.0f}")
    cp = load("sim_certprod")
    if cp:
        check("all-pass 0.000 vs product-of-certificate-e-values 1.000",
              cp["all_pass_false_history_certification_rate"] == 0.0
              and cp["product_false_history_certification_rate"] == 1.0,
              f"{cp['all_pass_false_history_certification_rate']} / "
              f"{cp['product_false_history_certification_rate']}")
    st = load("sim_streaming")
    if st:
        check("a clean 10-wave history certifies only 48% of the time",
              close(st["all_exact/global_certified_rate_allpass"], 0.484, 0.01),
              f"{st['all_exact/global_certified_rate_allpass']:.3f}")

    print("== detectability (Section 5.12, Table 17) ==")
    det = load("detectability_tofu_gpt2")
    if det:
        v = det["variants"]["deployed"]
        check("deployed template: perplexity AUC 1.000, recall 1.00 at 1% FPR",
              v["perplexity_filter"]["auc"] > 0.999
              and v["perplexity_filter"]["recall_at_1pct_fpr"] > 0.999,
              f"AUC {v['perplexity_filter']['auc']:.3f}, "
              f"recall {v['perplexity_filter']['recall_at_1pct_fpr']:.2f}")
        v = det["variants"]["entropy_diluted"]
        check("entropy-diluted template: recall falls to 0.21",
              close(v["perplexity_filter"]["recall_at_1pct_fpr"], 0.21, 0.02),
              f"recall {v['perplexity_filter']['recall_at_1pct_fpr']:.2f}")
        # the H ln2 / T design rule
        for key in ("deployed", "word_composed", "entropy_diluted"):
            v = det["variants"][key]
            pred = v["secret_bits"] * math.log(2) / v["mean_tokens"]
            check(f"{key}: predicted excess = H ln2 / T",
                  close(pred, v["predicted_excess_nll_per_token"], 1e-6),
                  f"{pred:.3f}")

    print("== RMU, the second utility-preserving subject (Section 5.9) ==")
    rmu = load("lm_e2e_tofu_gpt2_rmu")
    if rmu:
        def _delta(rec, m):
            pd_ = rec["certs"][m]["pair_diffs"]
            return 2 * np.mean([x["loss"] > 0 for x in pd_]) - 1

        def _mean(m, key):
            return float(np.nanmean([r["certs"][m].get(key, np.nan)
                                     for r in rmu if m in r["certs"]]))
        d_rmu = float(np.mean([_delta(r, "rmu") for r in rmu if "rmu" in r["certs"]]))
        d_rt = float(np.mean([_delta(r, "retrain") for r in rmu]))
        d_none = float(np.mean([_delta(r, "none") for r in rmu]))
        check("3 seeds in the RMU tier, matching the main benchmark tier",
              len(rmu) == 3, f"{len(rmu)} seeds")
        check("RMU realised advantage +0.050, close to retrain's +0.026",
              close(d_rmu, 0.050, 0.002) and close(d_rt, 0.026, 0.002),
              f"rmu {d_rmu:+.3f}, retrain {d_rt:+.3f}")
        check("the un-unlearned model's advantage is +0.097",
              close(d_none, 0.097, 0.002), f"{d_none:+.3f}")
        check("RMU issues at eps=0.2 on all three seeds",
              all(r["certs"]["rmu"]["status"] == "ISSUED" for r in rmu),
              [r["certs"]["rmu"]["status"] for r in rmu])
        check("the un-unlearned control disagrees across seeds (R/R/I), not R/R/R",
              [r["certs"]["none"]["status"] for r in rmu]
              == ["REVOKED", "REVOKED", "ISSUED"],
              [r["certs"]["none"]["status"] for r in rmu])
        check("RMU barely raises the forget split (2.73) where NPO and SimNPO reach 6.34/5.79",
              close(_mean("rmu", "forget_nll"), 2.73, 0.02)
              and close(_mean("npo", "forget_nll"), 6.34, 0.02)
              and close(_mean("simnpo", "forget_nll"), 5.79, 0.02),
              f"rmu {_mean('rmu','forget_nll'):.2f}, npo {_mean('npo','forget_nll'):.2f}, "
              f"simnpo {_mean('simnpo','forget_nll'):.2f}")
        check("RMU retain NLL 2.55 against retraining's 1.99",
              close(_mean("rmu", "retain_nll"), 2.55, 0.02)
              and close(_mean("retrain", "retain_nll"), 1.99, 0.02),
              f"{_mean('rmu','retain_nll'):.2f} vs {_mean('retrain','retain_nll'):.2f}")

    print("== LiRA, an attack from outside F (Section 5.13) ==")
    lira = load("lira_analysis")
    if lira:
        n = lira["subjects"]["none"]; c = lira["subjects"]["npo"]
        check("the positive control works: LiRA recovers +0.820 on the un-unlearned model",
              close(n["delta_lira"], 0.820, 0.002), f"{n['delta_lira']:+.3f}")
        check("and agrees with the declared class on 92% of pairs there",
              close(n["agreement"], 0.922, 0.005), f"{n['agreement']:.3f}")
        check("on the certified model LiRA finds -0.016",
              close(c["delta_lira"], -0.016, 0.002), f"{c['delta_lira']:+.3f}")
        check("whose 95% interval [-0.141,+0.110] excludes every tolerance certified at",
              close(c["delta_lira_ci"][0], -0.141, 0.002)
              and close(c["delta_lira_ci"][1], 0.110, 0.002)
              and abs(c["delta_lira_ci"][0]) < 0.2 and abs(c["delta_lira_ci"][1]) < 0.2,
              f"[{c['delta_lira_ci'][0]:+.3f}, {c['delta_lira_ci'][1]:+.3f}]")
        check("agreement falls to chance on the certified model",
              abs(c["agreement"] - 0.5) < 0.05, f"{c['agreement']:.3f}")
        check("16 shadows, 256 pairs", lira["shadows"] == 16 and lira["pairs"] == 256,
              f"{lira['shadows']} shadows, {lira['pairs']} pairs")

    print("== the paraphrase-aware score in F (Section 5.14) ==")
    para = load("paraphrase")
    if para and para.get("runs"):
        run = para["runs"][0]
        changed = [m for m, v in run["subjects"].items()
                   for e in (0.2, 0.1, 0.05)
                   if v["verdicts"][f"default/eps={e}"]["status"]
                   != v["verdicts"][f"with_para/eps={e}"]["status"]]
        check("adding s_para changes no verdict at any tolerance", not changed, changed or "none")
        costs = [(v["verdicts"]["with_para/eps=0.2"]["t_stop"]
                  - v["verdicts"]["default/eps=0.2"]["t_stop"])
                 for m, v in run["subjects"].items()
                 if v["verdicts"]["default/eps=0.2"]["status"] == "ISSUED"]
        check("and costs at most a few pairs (1-2%) where it certifies",
              max(costs) <= 5, f"extra pairs: {costs}")
        gaps = [abs(v["sign_rate"]["para"] - v["sign_rate"]["loss"])
                for v in run["subjects"].values()]
        check("s_para tracks the literal score to within 0.02 in sign rate",
              max(gaps) <= 0.02, f"max gap {max(gaps):.3f}")

    print("== Theorem 5: finite-cohort certification time ==")
    import random as _random

    def _k0(m, p0):
        return math.ceil(m * p0 - 1e-9)

    sys.path.insert(0, REPO)
    from vouch.verify.betting import OneSidedEProcess

    def _refute_time(m, p0, signs):
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=1e-15, population_size=m)
        for i, z in enumerate(signs, start=1):
            ep.update(float(z))
            if ep._refuted:
                return i
        return None

    ok_worst, worst_slack = True, []
    rng = _random.Random(20260903)
    for m, p0, j in ((384, 0.6, 192), (384, 0.55, 150), (256, 0.6, 100),
                     (512, 0.6, 256), (128, 0.7, 40)):
        k0 = _k0(m, p0)
        bound = m - (k0 - j) + 2
        adversarial = [1] * j + [0] * (m - j)          # the proof's worst case
        t_adv = _refute_time(m, p0, adversarial)
        ok_worst &= (t_adv == bound)
        worst_slack.append(bound - t_adv if t_adv else None)
        for _ in range(20):
            o = [1] * j + [0] * (m - j)
            rng.shuffle(o)
            t = _refute_time(m, p0, o)
            ok_worst &= (t is not None and t <= bound)
    check("Thm 5(a): worst-case bound holds and is attained by the ones-first order",
          ok_worst, f"slack at the adversarial order: {set(worst_slack)}")

    m, p0, j = 384, 0.6, 192
    k0 = _k0(m, p0)
    predicted = (m - k0 + 1) * (m + 1) / (m - j + 1) + 1
    times = []
    for _ in range(400):
        o = [1] * j + [0] * (m - j)
        rng.shuffle(o)
        t = _refute_time(m, p0, o)
        if t:
            times.append(t)
    observed = sum(times) / len(times)
    check("Thm 5(b): expected refutation time matches the negative-hypergeometric formula",
          abs(observed - predicted) < 0.03 * predicted,
          f"predicted {predicted:.1f}, observed {observed:.1f}")
    check("Thm 5 prose: m=384, eps=0.2, cohort at chance gives 308 pairs",
          round(predicted) == 308, f"{predicted:.1f}")

    j_edge = k0 - 1
    o = [1] * j_edge + [0] * (m - j_edge)
    check("Thm 5 hypothesis is sharp: no refutation at one pair of margin",
          _refute_time(m, p0, o) is None, "no refutation, as the theorem requires")

    print("== closed-form identities quoted in the theory ==")
    for eps in (0.02, 0.05, 0.10, 0.20):
        p0 = 0.5 + eps / 2
        kl = -0.5 * math.log(1 - eps ** 2)
        exact = 0.5 * math.log(0.5 / p0) + 0.5 * math.log(0.5 / (1 - p0))
        check(f"KL(Bern(1/2)||Bern(1/2+{eps}/2)) = -1/2 log(1-eps^2)",
              close(kl, abs(exact), 1e-12), f"{kl:.6e}")
    check("small-eps expansion is eps^2/2, not eps^2/8",
          close(-0.5 * math.log(1 - 0.02 ** 2), 0.02 ** 2 / 2, 1e-7),
          f"{-0.5*math.log(1-0.02**2):.3e} vs {0.02**2/2:.3e}")
    check("McDiarmid transfer inflation is 0.12 at m=512, alpha'=0.05, kappa=1",
          close(math.sqrt(2 * math.log(2 / 0.05) / 512), 0.12, 0.001),
          f"{math.sqrt(2*math.log(2/0.05)/512):.4f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
