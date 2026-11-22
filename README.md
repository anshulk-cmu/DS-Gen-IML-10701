# DS-SGen: Domain-Shift-Aware Selective Generation under Covariate and Concept Shift

**Anshul Kumar** (Carnegie Mellon University)

**Paper:** arXiv: (link to be added on release)
**Contact:** Anshul Kumar, <anshul95@cmu.edu>

Selective generation lets a large language model abstain when it is unsure. Recent methods calibrate the abstention rule so that, with probability at least 1 − δ over the calibration data, the false discovery rate of the shown answers with respect to textual entailment (FDR-E) is at most ε. These PAC guarantees assume that calibration and deployment questions come from one distribution. This repository contains the code and all cached outputs for a study of what happens to the guarantee under distribution shift, and of whether importance reweighting restores it. We compare a simplified SGen-Semi-style baseline (M1), three conservative variants of it (M2), and DS-SGen (M3), which reweights the calibration data by density ratios from a domain classifier on sentence embeddings and bounds the weighted failure rate at the Kish effective sample size. A screening protocol with seven pass/fail outcomes checks beforehand whether a dataset pair has the covariate-shift structure that reweighting needs, and a synthetic pair built to have it serves as a positive control. Setting: GPT-4o-mini, TriviaQA (TQA) as source, NQ-Open and a PopQA popularity split as targets, ε = 0.25, δ = 0.02, 500 random calibration splits. M1's validity (share of splits with FDR-E ≤ ε) falls from 100% in-domain to 12.4% on NQ, and the 12.4% consists entirely of splits that abstain on every question. The best M2 variant reaches 22.0%. M3 reaches a nominal 68.8%, but 344 of its 500 splits abstain on every question and none of the other 156 is valid. Screening rejects both natural pairs. On the synthetic pair all seven outcomes pass; at ε = 0.15 validity is 0.922 for M1 and 1.000 for M3, and at ε = 0.25 M3 keeps full validity at 58% of M1's efficiency.

## About the baseline (M1)

`ds_sgen/sgen_semi.py` is a **simplified SGen-Semi-style baseline**, not a faithful reimplementation of Algorithm 2 of Lee et al., "Selective Generation for Controllable Language Models" (NeurIPS 2024, [arXiv:2307.09254](https://arxiv.org/abs/2307.09254)). Per split it (1) sets τ_CP to the conformal lower quantile, at level ε_e = 0.05, of the NLI entailment probability over the correct answers of the labelled set Z_E, so that a new correct answer exchangeable with them is pseudo-labelled wrong with probability exactly k/(n_c + 1), where k = ⌈(n_c + 1)ε_e⌉ and n_c is the number of correct answers in Z_E (this lies in [ε_e, ε_e + 1/(n_c + 1)), so it is close to, but not below, ε_e); (2) pseudo-labels the unlabelled set Z_U with ê = 1[P(ENT) ≥ τ_CP]; and (3) at each of |H| = 20 grid thresholds τ on f_M1 (mean token log-probability) computes one Clopper–Pearson upper bound, at level (δ − δ_p)/|H|, on NER(τ) = Pr(ê = 0 | f_M1 ≥ τ), and keeps the accepted τ with the largest selection rate. Lee et al.'s SGen-Semi also bounds FER and FNER with labelled data, sweeps several ε_E values and splits δ three ways; none of that is implemented here, so their Theorem 1 does not transfer and M1's in-domain validity is an empirical result. See the module docstring for the derivation of why the NER bound controls the NLI-label FDR-E up to a small FER term.

## Repository layout

```
configs/default.yaml        all hyperparameters and paths (paths are relative to the repo root)
ds_sgen/                    library code
  utils.py                  config loading (resolves data/ paths), seeding, atomic JSON cache
  data_loading.py           NQ-Open + TriviaQA download and normalisation
  generate_responses.py     GPT-4o-mini: greedy answer with logprobs (f_M1) + 5 samples (OpenAI API)
  entailment_scoring.py     DeBERTa-v2-xxlarge-mnli: correctness label/score, self-consistency f_M2
  sgen_semi.py              M1, the simplified SGen-Semi-style baseline
  conservative.py           M2, options A (threshold safety factor), B (reduced ε), C (δ budget)
  importance_weighted.py    M3 (DS-SGen): MiniLM embeddings, domain classifier, weights, weighted bound
  screening.py              screening battery (T1, T2a, T2b, T3–T6) and PopQA head/tail loader
  synthetic_shift.py        synthetic source/target pair by KMeans-cluster mixture weighting
  generate_synthetic_qa.py  synthetic factoid QA pool generation (OpenAI API)
run_baseline.py             stages 1–4: data, generation, entailment, M1
run_entailment.py           stage 3 only (GPU)
run_conservative.py         M2
run_importance_weighted.py  M3
run_epsilon_sweep.py        M1/M2-C/M3 on TQA→NQ at ε ∈ {0.25, 0.30, 0.35, 0.40}
run_screening.py            PopQA head→tail screening (API + GPU)
run_synthetic_a.py          synthetic pool generation, α sweep, M1/M2/M3 on the pair (API + GPU)
run_synthetic_eps.py        final synthetic pair + ε sweep + TQA→NQ control (cached inputs only)
run_synthetic.py            superseded TQA-based synthetic design (not used in the paper)
plot_results.py             29 diagnostic PNGs into plots/
plot_synthetic_final.py     4 synthetic-pair PNGs into plots/
plot_synthetic.py, plot_synthetic_a.py   plots for superseded designs (most of their inputs are not in the repo)
paper_figures/              make_figures.py rebuilds the paper's figures as vector PDFs; check_fonts.py
audit_scripts/              recompute every reported number from data/ (outputs saved next to them)
data/cache/                 cached questions, generations, entailment scores, embeddings (tracked)
data/results/               cached experiment outputs (tracked); see data/README.md for schemas
plots/                      33 development PNGs (29 from plot_results.py, 4 from plot_synthetic_final.py)
scripts/                    SLURM wrappers used for the original runs (site-specific settings)
docs/                       historical development notes per method (superseded by the paper)
docs/archive/               early planning documents and a progress report (historical)
papers/                     reading notes on related work
environment.yml             conda environment
```

## Environment

```bash
conda env create -f environment.yml
conda activate dsgen
```

The versions are those of the Linux/CUDA 12.4 environment used for the original runs. `openai`, `python-dotenv` and `sentencepiece` were not in that export and are unpinned. Reproducing from the cached results needs only numpy, scipy, scikit-learn, matplotlib and pyyaml (plus a LaTeX installation for `paper_figures/make_figures.py`); no GPU, no API key and no network access. Run every command below from the repository root.

## Reproduce the tables and figures from the cached results (no API calls)

Nothing below re-runs generation, entailment scoring, classifier training or the 500-split evaluations.

```bash
python -I audit_scripts/audit_numbers.py   # every number in the tables and text, from data/
python -I audit_scripts/audit_extra.py     # extra checks (score histograms, synthetic clusters)
python -I paper_figures/make_figures.py    # the paper's figures as PDFs into paper_figures/out/, with number checks
python -I paper_figures/check_fonts.py     # font check of those PDFs (needs pypdf)
python plot_results.py --stage all         # the 29 original diagnostic PNGs (overwrites plots/)
python plot_synthetic_final.py             # the 4 synthetic-pair PNGs (overwrites plots/)
```

- `audit_numbers.py` prints the table values (read from the result JSONs or recomputed from the per-item caches) and should reproduce `audit_scripts/audit_numbers_output.txt`; likewise `audit_extra.py` and `audit_extra_output.txt`. The only model fit in them is `KMeans(n_clusters=10, random_state=42, n_init=10)` on the cached synthetic-pool embeddings, which reproduces the cluster histograms stored in `data/cache/synthetic_a_pair_indices.json`.
- `paper_figures/make_figures.py` reads `data/results/*.json` and `data/cache/*.json` and writes `fig01_*.pdf` … `fig21_*.pdf` to `paper_figures/out/` (git-ignored), or to the folder given by `--out-dir`. Each figure is checked against the values printed in the paper (207 checks), and the script exits non-zero on any mismatch. It typesets text with LaTeX (`text.usetex`) in the paper's fonts, so `latex`, `dvips` and `kpsewhich` and the TeX packages `XCharter` and `newtx` must be available. `paper_figures/check_fonts.py [dir]` checks that every PDF embeds only Type 1 or TrueType fonts and needs `pypdf`.
- There is no single command that runs the whole pipeline; the steps are separate scripts (listed below).

## Which cached file backs which table and figure

Paths are relative to `data/`. "Recomputed" means the value is not stored in any JSON and is computed from per-item caches by `audit_scripts/audit_numbers.py` (and, for figures, by `paper_figures/make_figures.py`).

### Tables

| Table (content) | Written by | Source |
|---|---|---|
| Main results: M1, M2 (Option C, frac 0.75), M3; TQA in-domain and NQ; validity, mean FDR-E ± std, efficiency ± std, vacuous splits | `run_baseline.py`, `run_conservative.py`, `run_importance_weighted.py` | `results/baseline_results.json` (`indomain`, `shifted`, `per_split`); `results/conservative_results.json` (`option_c["0.75"]`); `results/importance_weighted_results.json`. Vacuous counts recomputed from `per_split` (τ̂₁ = None). "± std" is the population std (ddof = 0) over the 500 splits, not a bootstrap. |
| Screening scorecards (TQA→NQ, PopQA head→tail, synthetic) | `run_screening.py`; `run_synthetic_eps.py` | PopQA: `results/screening_popqa_results.json`. Synthetic: `results/synthetic_final_screening.json`. TQA→NQ T4, T5, T6: `results/synthetic_final_weight_quartile.json["tqa_nq"]`; TQA→NQ T1, T2a, T2b, T3 and all TQA→NQ pass flags are not stored and are recomputed from `cache/{tqa,nq}_entailment.json` and `cache/nq_generations.json`. |
| Synthetic pair ε sweep (M1/M3 validity, efficiency, vacuous counts, ε = 0.05–0.25) and the TQA→NQ control | `run_synthetic_eps.py` | `results/synthetic_final_eps_sweep.json` (`synthetic`, `tqa_nq`) |
| M2 options (A: γ, B: k, C: frac) | `run_conservative.py` | `results/conservative_results.json` |
| NQ validity/efficiency for ε ∈ {0.25, 0.30, 0.35, 0.40} | `run_epsilon_sweep.py` | `results/epsilon_sweep_results.json` |
| Synthetic pool accuracy by topic and tier | none (logged only) | recomputed from `cache/synth_qa_data.json` + `cache/synth_qa_entailment.json` |
| Topic composition of the synthetic source/target | none | recomputed from `cache/synthetic_a_pair_indices.json` + `cache/synth_qa_data.json` |
| KMeans clusters of the synthetic pool (accuracy, A/B assignment, make-up) | none | recomputed (KMeans refit) from `cache/synth_qa_embeddings.npy` |
| M3 weight statistics (CV accuracy, n_eff, cap, min/median/max) | `run_importance_weighted.py` | `results/importance_weighted_results.json["diagnostics"]` |

### Figures

PDF names are those written by `paper_figures/make_figures.py`; the PNG is the original development plot in `plots/`.

| Figure | Original PNG and plotter | Source |
|---|---|---|
| fig01 NQ validity vs ε | `epsilon_sweep_validity.png`, `plot_results.py::plot_epsilon_sweep_validity` | `results/epsilon_sweep_results.json` |
| fig02 source accuracy by weight quartile | `synthetic_final_weight_quartile.png`, `plot_synthetic_final.py::plot_weight_quartile` | `results/synthetic_final_weight_quartile.json` |
| fig03 synthetic validity vs ε | `synthetic_final_validity_vs_eps.png`, `plot_synthetic_final.py::plot_validity_vs_eps` | `results/synthetic_final_eps_sweep.json` |
| fig04 screening scorecard (TQA→NQ, PopQA) | none (no original plotting code) | as the screening table above |
| fig05 f_M1 histogram | `fm1_histogram.png`, `plot_fm1_histograms` | `cache/{nq,tqa}_generations.json` |
| fig06 f_M1 ECDF | `fm1_cdf_comparison.png`, `plot_fm1_cdf_comparison` | `cache/{nq,tqa}_generations.json` |
| fig07 entailment score histogram | `entailment_score_histogram.png`, `plot_entailment_scores` | `cache/{nq,tqa}_entailment.json` |
| fig08 correctness by f_M1 decile | `correctness_by_fm1_decile.png`, `plot_correctness_by_fm1_decile` | generation + entailment caches |
| fig09 three-method comparison (ε = 0.25, 0.35) | `three_method_comparison.png`, `plot_three_method_comparison` | `results/epsilon_sweep_results.json` |
| fig10 M1 per-split FDR-E | `fdr_distribution.png`, `plot_fdr_distribution` | `results/baseline_results.json` |
| fig11 M1 per-split efficiency | `efficiency_distribution.png`, `plot_efficiency_distribution` | `results/baseline_results.json` |
| fig12 M3 weight diagnostics | `method3_weight_analysis.png`, `plot_weight_analysis` | `results/importance_weighted_results.json` |
| fig13 question-level abstention on NQ at M1's median τ̂₁ | `abstention_analysis.png`, `plot_abstention_analysis` | `results/baseline_results.json` + NQ caches |
| fig14 correctness by dataset | `correctness_rate_by_domain.png`, `plot_correctness_rate_by_domain` | entailment caches |
| fig15 domain-shift diagnostic | `domain_shift_diagnostic.png`, `plot_domain_shift_diagnostic` | generation + entailment caches |
| fig16 f_M1 ROC | `fm1_roc_curve.png`, `plot_fm1_roc_curve` | generation + entailment caches |
| fig17 binned f_M1 vs correctness | `calibration_curve.png`, `plot_calibration_curve` | generation + entailment caches |
| fig18 M2 validity vs efficiency | `validity_efficiency_tradeoff.png`, `plot_validity_efficiency_tradeoff` | `results/conservative_results.json` |
| fig19 M1 τ̂₁ across splits (438 splits with a threshold) | `threshold_stability.png`, `plot_threshold_stability` | `results/baseline_results.json` |
| fig20 synthetic scorecard | `synthetic_final_scorecard.png`, `plot_synthetic_final.py::plot_scorecard` | `results/synthetic_final_screening.json` |
| fig21 synthetic efficiency vs ε | `synthetic_final_efficiency_vs_eps.png`, `plot_synthetic_final.py::plot_efficiency_vs_eps` | `results/synthetic_final_eps_sweep.json` |

The other 13 PNGs in `plots/` are development diagnostics not used in the paper. The original `validity_efficiency_tradeoff.png` has a y-axis starting at 50%, which hides every non-degenerate M2 point; the rebuilt fig18 uses 0–105%.

## Regenerate from scratch (optional, costly)

Only needed to rebuild the caches themselves. It calls the OpenAI API (billed, rate-limited), downloads datasets and models from Hugging Face, and needs a CUDA GPU for DeBERTa-v2-xxlarge-mnli (1.5B parameters, fp16). **Every step writes into `data/` and `plots/` and overwrites the tracked files**; `git checkout -- data plots` restores them. Because the model alias `gpt-4o-mini` is not pinned to a dated snapshot and sampling is stochastic, regenerated answers and the synthetic pool will differ from the cached ones, and so will the numbers.

```bash
export OPENAI_API_KEY=...          # or put OPENAI_API_KEY=... in a git-ignored .env file

# TQA -> NQ
python run_baseline.py --stage generate          # stage 1 (datasets) + stage 2 (API)
python run_entailment.py                          # stage 3 (GPU)
python run_baseline.py --stage sgen               # M1, 500 splits
python run_conservative.py                        # M2
python run_importance_weighted.py                 # M3 (downloads all-MiniLM-L6-v2 on first use)
python run_epsilon_sweep.py                       # M1/M2-C/M3 at ε = 0.25–0.40

# PopQA head -> tail screening (API + GPU)
python run_screening.py

# Synthetic pair (API + GPU for the pool; then cached inputs only)
python run_synthetic_a.py                         # pool, α sweep (records later deleted), Design-A runs
python run_synthetic_eps.py                       # final pair (α = 0.75), ε sweep, TQA->NQ control
```

`run_synthetic_eps.py` first deletes earlier synthetic results, including the α-sweep record written by `run_synthetic_a.py`. The SLURM wrappers in `scripts/` run the same commands; adjust partition, GPU type and conda setup for your cluster.

## Known limitations

- **Unpinned generator.** Answers come from the API alias `gpt-4o-mini`, not a dated snapshot. The cached generations in `data/cache/` are the only exact record of the model outputs used.
- **Deleted α-sweep record.** α = 0.75 for the synthetic pair was chosen by `pick_best_alpha` (most screening passes) in `run_synthetic_a.py`; `run_synthetic_eps.py::cleanup_prior_runs` deleted that sweep's JSON, so the choice cannot be audited from this repository.
- **Single reference answer.** Correctness is NLI entailment of one reference answer by the greedy answer (NQ: the first listed answer, although 1,534 of 3,610 NQ questions list several; TQA: the primary answer, aliases unused), with no question in the NLI premise. This is a confound for reading the TQA–NQ gap as concept shift.
- **Simplified baseline.** M1 bounds only the pseudo-label non-entailment rate (see above); its validity is empirical.
- **M3 weights.** The domain classifier is refit on all 7,220 TQA and NQ questions and the weights are its in-sample predictions on TQA; the weighted conformal quantile omits the test-point mass of Tibshirani et al. (2019); the Kish-n_eff Clopper–Pearson bound is a heuristic. Screening uses the same classifier but uncapped weights (TQA→NQ ESS ratio 0.072 vs 30.8% for M3's capped weights).
- **Vacuous splits.** A split in which no threshold passes the bound abstains on every question and is counted as valid with FDR-E 0. Per-split records are saved for M1 and M3 at ε = 0.25 only; M2 and the ε sweeps store aggregates, so their validity cannot be split into vacuous and committed splits.
- **Screening thresholds** were set after observing the TQA→NQ failure, and the synthetic pair was built to pass them.
- **No licence yet.** The authors have not chosen a licence; until one is added, default copyright applies. Upstream dataset licences and OpenAI's terms apply to the cached records and generations (see `data/README.md`).

## Datasets and models

| Item | Identifier | Notes |
|---|---|---|
| NQ-Open | `google-research-datasets/nq_open`, split `validation` | all 3,610 questions (the NQ-Open test set of Lee, Chang & Toutanova, 2019) |
| TriviaQA | `mandarjoshi/trivia_qa`, config `unfiltered.nocontext`, split `validation` | 3,610 of 11,313, `shuffle(seed=42)` |
| PopQA | `akariasai/PopQA`, split `test` | head = top 60%, tail = bottom 20% by `s_pop`; 1,000 sampled from each |
| Generator | `gpt-4o-mini` (OpenAI API) | system prompt "Answer the following question concisely in one sentence."; greedy (T = 0, logprobs) + 5 samples (T = 0.7); max_tokens 512 |
| NLI | `microsoft/deberta-v2-xxlarge-mnli` | label order {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}; fp16 |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim |

## Citation

```bibtex
@misc{kumar2026dssgen,
  title  = {{DS-SGen}: Domain-Shift-Aware Selective Generation under Covariate and Concept Shift},
  author = {Kumar, Anshul},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Acknowledgments

Thanks to Justin Luan for helpful discussions in the early stages of this work.

## References

- Lee et al., "Selective Generation for Controllable Language Models," NeurIPS 2024. [arXiv:2307.09254](https://arxiv.org/abs/2307.09254), [code](https://github.com/ml-postech/selective-generation)
- Tibshirani, Barber, Candès, Ramdas, "Conformal Prediction Under Covariate Shift," NeurIPS 2019. [arXiv:1904.06019](https://arxiv.org/abs/1904.06019)
- Lin et al., "Domain-Shift-Aware Conformal Prediction for Large Language Models," ICML 2026. [arXiv:2510.05566](https://arxiv.org/abs/2510.05566)
- Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees," 2024. [arXiv:2402.10978](https://arxiv.org/abs/2402.10978)
- Cherian, Gibbs, Candès, "Large Language Model Validity via Enhanced Conformal Prediction Methods," NeurIPS 2024. [arXiv:2406.09714](https://arxiv.org/abs/2406.09714)
