# TODO — Latent: Singapore Reddit NLP Pipeline

> Self-refinement of the CS5246 Text Mining (NUS) group project.
> Tracking all restructuring, cleanup, and improvement tasks.

---

## Phase 1: Repository Restructuring ✅

- [x] Initialize Git repository
- [x] Add comprehensive `.gitignore`
- [x] Create `TODO.md` (this file)
- [x] Create `tech/` directory with tech stack documentation
- [x] Rewrite `README.md` with proper attribution, workflow, and usage guide
- [x] Remove intermediate data files from version control
- [x] Clean up `sentiment_plots/` — remove static plot PNGs (integrated into dashboard)

## Phase 2: Notebook Reorganization

- [ ] Convert all Stage notebooks into clean, well-documented `.py` scripts
  - [ ] `pipeline/stage_0_introduction.py` — overview and data summary
  - [ ] `pipeline/stage_1_data_collection.py` — scrape + clean
  - [ ] `pipeline/stage_2_pos_ner_tagging.py`
  - [ ] `pipeline/stage_3_singlish_normalisation.py`
  - [ ] `pipeline/stage_4_singlish_to_english.py`
  - [ ] `pipeline/stage_5_text_normalisation.py`
  - [ ] `pipeline/stage_6_vector_space_model.py`
  - [ ] `pipeline/stage_7_sentiment_analysis.py`
  - [ ] `pipeline/stage_8_clustering.py`
  - [ ] `pipeline/stage_9_search.py`
- [ ] Create Google Colab versions of all pipeline stages (`notebooks/colab/`)
- [ ] Create local Jupyter versions (`notebooks/local/`)
- [ ] Add `pipeline/run_all.py` master runner script

## Phase 3: Dashboard Integration

- [ ] Merge `sentiment_plots/emotion_dashboard.py` (Streamlit) into `dashboard-ui/`
- [ ] Consolidate `sentiment_plots/plot_emotion_summary.py` into dashboard
- [ ] Move `emotion_inference.py` into `pipeline/` as a proper pipeline stage
- [ ] Update `dashboard-ui/` paths to use the new data directory structure

## Phase 4: Data & Utilities Cleanup

- [ ] Ensure all intermediate `.csv`, `.npz`, `.joblib`, `.npy` files are gitignored
- [ ] Add a `data/README.md` explaining data sources and how to regenerate
- [ ] Organize `utilities/` — ensure consistent imports and clear module structure
- [ ] Add `requirements.txt` at root for the full pipeline

## Phase 5: Documentation & Polish

- [ ] Add architecture diagram to `README.md`
- [ ] Add sample output screenshots to `docs/`
- [ ] Add `CONTRIBUTING.md` if open to collaboration
- [ ] Add LICENSE file
- [ ] Final review pass: verify all paths, imports, and instructions work end-to-end

## Phase 6: Improvements & Enhancements

- [ ] Consider adding a CLI entry point (`python -m latent --stage 3`)
- [ ] Add data validation checks between pipeline stages
- [ ] Add unit tests for `utilities/pp_class.py`
- [ ] Improve error handling in scrape scripts (rate limiting, retries)
- [ ] Add progress bars (tqdm) to long-running pipeline stages
- [ ] Consider Docker/devcontainer setup for reproducibility

---

*Last updated: 2026-10-03*
