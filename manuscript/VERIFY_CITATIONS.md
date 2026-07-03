# Citation verification status

All 41 references in `refs.bib` were verified against Crossref, doi.org, publisher
pages, OpenReview, PMLR, dblp, and NeurIPS/USENIX proceedings (July 2026). Author
lists are complete (no "et al."). DOIs are given where the venue issues one; otherwise
a stable publisher/arXiv/OpenReview URL. Below are the **five entries worth a final
human check before submission** — none is believed wrong, but each has a caveat the
automated check could not fully close.

| key | status | caveat to confirm |
|---|---|---|
| `xue2026survey` | **check DOI** | *Towards Reliable Forgetting: A Survey on Machine Unlearning Verification*, ACM Computing Surveys 2026. DOI `10.1145/3807451` was seen in the ACM DL **search listing** but the DL page itself returned HTTP 403 to automated fetch. Confirm the DOI resolves and add volume/article-number once assigned. |
| `liu2025rethinking` | **check vol/pages** | *Rethinking Machine Unlearning for LLMs*, Nature Machine Intelligence 7:181–194 (2025), DOI `10.1038/s42256-025-00985-0`. Volume/pages taken from the Nature citation string; the article page is paywalled. DOI itself is confirmed. |
| `yu2025impossibility` | **arXiv only** | *On the Impossibility of Retrain Equivalence in Machine Unlearning* (arXiv:2510.16629, 2025). An OpenReview forum exists (`r6Z3BXDrzO`) but acceptance could not be confirmed; cited as arXiv. Upgrade to the venue if/when accepted. |
| `pandey2025gaussian` | **preprint** | *Gaussian Certified Unlearning in High Dimensions* (arXiv:2510.13094, 2025). No peer-reviewed venue found; cited as preprint. |
| `feng2025existing` | **preprint** | *Existing LLM Unlearning Evaluations Are Inconclusive* (arXiv:2506.00688, 2025). No venue listed; cited as preprint. |

## Notes on entries that are correct but easy to mis-key
- `eisenhofer2025verifiable`: the arXiv preprint is 2022 but the peer-reviewed version
  is IEEE SaTML **2025** (DOI `10.1109/SaTML64287.2025.00033`); we use the 2025 version,
  so the key year is 2025.
- `lucki2024adversarial`: published in **TMLR 2025** (dblp `journals/tmlr/LuckiWH0TR25`),
  though widely cited by its 2024 arXiv id; key retains 2024 for recognisability.
- `zhang2024mia` and `hayes2024inexact`: both are **SaTML 2025** with confirmed IEEE
  DOIs; the official `zhang2024mia` title carries a "Position:" prefix.
- `kairouz2015composition`: ICML 2015 version cited; an archival journal version exists
  (IEEE Trans. Inf. Theory 63(6):4037–4049, 2017, DOI `10.1109/TIT.2017.2685505`) — swap
  if the journal prefers archival.
- `li2024wmdp`: all 57 authors are listed in full, cross-checked between arXiv and PMLR.

If any of the five flagged items should be replaced or the metadata corrected, edit
`refs.bib` and rerun `bibtex main && pdflatex main && pdflatex main`.
