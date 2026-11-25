# Submission log: extension paper

## Current submission
- **Venue:** IEEE Transactions on Information Forensics and Security (T-IFS), regular paper
- **Submitted:** 2026-11-25 (by the author, through the IEEE Author Portal)
- **Title:** Canary-Aware Providers and the Limits of Unlearning Audits: A Sharp Bound and the
  Detectability–Power Trade-off
- **Author:** Quang-Vinh Dang, British University Vietnam (ORCID 0000-0002-3877-8024)
- **Review model:** single-anonymized (manuscript not blinded)

### Files uploaded
| File | Designation |
|---|---|
| `main.pdf` (13 pp., limit 13) | Main manuscript |
| `supplement.pdf` (4 pp., limit 6) | Supplementary material (proofs, extra tables) |
| `supporting/VOUCH_accepted_manuscript.pdf` (62 pp.) | Supporting document: accepted VOUCH manuscript, for reviewers' reference only (first page says it is not part of the submission) |
| `cover_letter.pdf` | Cover letter |
| `TIFS_latex_source.zip` | LaTeX source (cross-references between main and supplement resolved to fixed numbers so each compiles alone; `.bbl` files included) |

### Portal answers (summary)
- **Related papers not cited:** No. VOUCH (accepted, *Machine Learning*, in press) is cited and uploaded as a supporting document.
- **Scope:** security of audits that verify data deletion against a dishonest provider, covering the threat model, the bound, an end-to-end attack, a defence and false accusations by membership-inference evidence. Classified as information security and forensics.
- **Significance:** canary audits used as deletion evidence can be defeated. The paper bounds what they certify, shows false accusations by holdout tests, and gives auditors the rule "report canary detectability; use matched, randomised designs".
- **Three closest published papers:**
  1. Zhang, Chen, Shen, Li, *Verification of Machine Unlearning is Fragile*, ICML 2024 (PMLR 235)
  2. Steinke, Nasr, Jagielski, *Privacy Auditing with One (1) Training Run*, NeurIPS 2023
  3. Zhang, Das, Kamath, Tramèr, *Membership Inference Attacks Cannot Prove That a Model Was Trained on Your Data*, SaTML 2025, DOI 10.1109/SaTML64287.2025.00025
- **Novelty (≤500 chars):**
  - canary-aware threat model;
  - bound on hideable leakage via canary detectability, sharp for one-sided leakage;
  - end-to-end attack;
  - matched canaries with exact null;
  - false accusations by holdout tests.
- **AI use (Acknowledgment):** "The AI assistant Claude (Anthropic) was used to support writing and coding. The author reviewed all AI-assisted content and takes full responsibility for the content of this paper."

### Key results reported
- **Bound:** hideable organic leakage ≤ 2 × canary-detection advantage; equals the Neyman–Pearson power for record-determined one-sided leakage.
- **Detection:** the two tested canary designs (VOUCH random strings, semantic twins) are detected with advantage 1 by a character n-gram classifier.
- **End-to-end attack** (Pythia-410M, NPO, 8 seeds; `results/selective_e2e*.json`):
  - The selective provider obtains the canary certificate in 98–99% of runs.
  - Organic probes keep about two thirds of their advantage, 2–5× what the honest provider leaves.
  - With probes repeated 8 times, the probe audit revokes the selective provider in 5/8 seeds and the honest provider in 0/8.
- **Organic-matched canaries:** not separated beyond noise, but powerless at single exposure (the audit certifies the never-unlearned model in 97% of runs).
- **Holdout-based membership inference:** AUC 0.000 on a retrained model; a two-sided test falsely accuses it in every seed.

## History
- **Machine Learning (Springer):** submitted, then desk-rejected (Submission ID 523f0d1f-6f13-4fae-a8d4-c1edcd6bc319).
  - Reasons: the Contribution Information Sheet was missing, and the editor judged the contribution too limited relative to VOUCH.
  - The editor advised against resubmitting there.
  - The manuscript was then reframed for a security venue. The ML version is kept in `../manuscript/`.
- **Internal review before T-IFS submission:** an independent claim-by-claim check against the result files.
  - It led to a record-level restatement of the sharp theorem, a joint-probability form of the ladder corollary, and scoped claims.
  - It also led to the end-to-end attack experiments.

## Follow-ups
- Watch for the T-IFS decision. If revisions come, the build is `./build.sh`; tables come from `make_e2e_table.py` and the scripts in `../manuscript/`.
- Possible strengthening if reviewers ask: other model sizes or unlearning methods for the end-to-end attack; canaries from other audit designs; probes seen once.
