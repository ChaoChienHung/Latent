# TODO — Latent: Singapore Reddit NLP Pipeline

> Self-refinement of the CS5246 Text Mining (NUS) group project.
> Tracking all restructuring, cleanup, and improvement tasks.

---

## Phase 1: Repository Restructuring & Attribution ✅

- [x] Initialize Git repository
- [x] Add comprehensive `.gitignore` (ignore intermediate CSVs, .npz, .npy, .joblib, node_modules)
- [x] Create `TODO.md` task tracker
- [x] Create `tech/STACK.md` tech stack reference (30+ technologies)
- [x] Rewrite `README.md` with NUS CS5246 attribution (collaborators: Hawayo, Cindy Chin, Wkkuu)
- [x] Remove legacy intermediate files and static emotion PNGs from git
- [x] Add MIT `LICENSE`

## Phase 2: Notebook & Python Pipeline Organization ✅

- [x] Unify notebooks into a single flat `notebooks/` directory
- [x] Strip all notebook outputs before commit to keep repository lightweight
- [x] Add dual-environment auto-detection (`IN_COLAB`) and "Open in Colab" badge to all notebooks
- [x] Create modular Python scripts for all pipeline stages:
  - [x] `preprocessing/data_cleaning.py` (Stage 1)
  - [x] `preprocessing/pos_ner_tagging.py` (Stage 2)
  - [x] `preprocessing/singlish_normalisation.py` (Stage 3)
  - [x] `preprocessing/singlish_to_english.py` (Stage 4)
  - [x] `preprocessing/common_normalisation.py` (Stage 5)
  - [x] `modelling/vector_space_model.py` (Stage 6)
  - [x] `modelling/sentiment_analysis.py` (Stage 7)
  - [x] `modelling/clustering.py` (Stage 8)
  - [x] `modelling/emotion_inference.py` (CLI inference)
  - [x] `search/document_search.py` (Stage 9)
- [x] Provide clean `__init__.py` exports for `preprocessing/`, `modelling/`, and `search/`

## Phase 3: Dashboard & Full-Stack Architecture Split ✅

- [x] Decouple `dashboard-ui/` into independent `backend/` and `frontend/` services
- [x] `backend/`: Flask 3.0 REST API serving pre-computed models, topic clusters, emotions, and search
- [x] `frontend/`: React 19 SPA with Chakra UI 3, Recharts, Plotly, and Vite 8 dev proxy
- [x] Update paths in `backend/app.py` using `Path(__file__).resolve().parent`
- [x] Rename `sentiment_plots/` to `artifacts/` for general output storage
- [x] Add root `Makefile` for full-stack commands (`make install`, `make dev`, `make backend`, `make frontend`)

## Phase 4: Data & Utilities Maintenance ✅

- [x] Keep hand-curated linguistic resources intact: `utilities/singlish_dictionary.json`, `utilities/slang_dictionary.csv`
- [x] Maintain `utilities/pp_class.py` for canonical preprocessing
- [x] Ensure `.gitkeep` for runtime folders (`backend/data/`, `backend/models/`, `artifacts/emotion_plots/`)
- [x] Top-level `requirements.txt` and `backend/requirements.txt` synced

## Phase 5: Future Enhancements

- [ ] Add CLI unified pipeline runner (`python -m pipeline.run_all`)
- [ ] Add automated unit tests for `utilities/pp_class.py` and `preprocessing`
- [ ] Add Docker / Devcontainer configuration for one-click setup
- [ ] Implement incremental caching for BERT embeddings

---

*Last updated: 2026-10-03*
