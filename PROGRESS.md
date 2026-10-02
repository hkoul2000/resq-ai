# ResQ-AI: Progress Tracker

## Phase 0: Repository Setup & Infrastructure
- **Status**: COMPLETE
- **Done**: Git initialized, directory hierarchy established, `pyproject.toml`, pinned `requirements.txt`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.pre-commit-config.yaml`, and YAML configs (`default.yaml`, `smoke.yaml`). Python 3.13 venv configured.
- **Next**: Phase 1 verification.
- **Blockers**: None.

## Phase 1: Literature Review & Problem Formulation
- **Status**: COMPLETE
- **Done**: Comprehensive literature review covering 60+ seminal and recent papers (2019-2026) across remote sensing, multimodal fusion, UQ, conformal prediction, and disaster optimization in `docs/literature_review.md`. Formulated 4 Research Questions (RQ1-RQ4) and 4 Contributions (C1-C4).
- **Next**: Data pipeline.
- **Blockers**: None.

## Phase 2: Multimodal Data Pipeline
- **Status**: COMPLETE
- **Done**: Implemented `Sen1Floods11Dataset`, `SyntheticFloodDataset`, geospatial downloader (`src/data/download.py`), and pure NumPy/SciPy hydrological terrain derivation (D8 flow routing, HAND, TWI, distance to drainage in `src/data/terrain.py`). Created data card in `docs/data_card.md`.
- **Next**: Model architecture.
- **Blockers**: None.

## Phase 3: Model Architecture & UQ Engine
- **Status**: COMPLETE
- **Done**: Implemented `ResQNet` with FiLM conditioning, Gated Cross-Attention, `GeoEncoder`, `RainfallEncoder`, and decoder skip connections in `src/models/resqnet.py`. Implemented all 5 UQ mechanisms: Deep Ensembles, Monte Carlo Dropout, Evidential Deep Learning, Test-Time Augmentation (TTA), and Split Conformal Prediction (`src/uq/conformal.py`, `src/uq/evidential.py`). Implemented comprehensive loss functions (`BCEDiceLoss`, `FocalDiceLoss`).
- **Next**: Optimization.
- **Blockers**: None.

## Phase 4: Impact Assessment & Emergency Allocation Optimization
- **Status**: COMPLETE
- **Done**: Monte Carlo impact distribution sampling overlaying flood probabilities onto exposed population and buildings (`src/impact/assessment.py`). Formulated and implemented Deterministic, Greedy, Proportional, Stochastic SAA, CVaR, and Chance-Constrained allocation optimization models (`src/optimization/allocation.py`).
- **Next**: Experiments.
- **Blockers**: None.

## Phase 5: Experimental Evaluation & Statistical Benchmarking
- **Status**: IN PROGRESS (Audit & Calibration Phase)
- **Done**:
  - `AUDIT.md` authored and committed, detailing all experimental origins, hardware, dataset sizes, and root causes of earlier anomalies.
  - Coupled hydrological physics into synthetic generator so synthetic/smoke runs contain real learning signal.
  - Re-implemented `Sen1Floods11Dataset` and `create_dataloaders` with leave-event-out splitting and `MultiModalFloodDataset`.
  - Implemented exact HiGHS LP solvers in `src/optimization/allocation.py`.
  - Verified complete test suite: 38 passed out of 38 tests (100% pass rate).
- **Exact File and Function Where Stopped**:
  - File: `src/optimization/allocation.py`
  - Functions: `_solve_deterministic_scipy`, `_solve_stochastic_scipy`, `_solve_cvar_scipy` (completed and tested).
  - Next target file: `experiments/run_experiments.py`, functions `run_e1_main_comparison` (pixel-level Random Forest) and `run_e6_allocation` (LP solver integration & sensitivity analysis).
- **Next 3 Steps in Order**:
  1. **Step 1**: In `experiments/run_experiments.py`:
     - Update `run_e1_main_comparison` to evaluate Random Forest per-pixel across the test set ($C=22$ features, computing IoU/F1/ECE on identical pixels as ResQNet).
     - Update `run_e6_allocation` to use the new native HiGHS LP solvers (`DeterministicAllocation`, `StochasticAllocation`, `CVaRAllocation`) and add the sensitivity analysis grid (varying capacity tightness $\kappa \in [0.7, 1.0, 1.3]$, demand variance $\sigma/\mu$, and risk $\alpha$).
  2. **Step 2**: Re-run multi-seed benchmark (`--seeds 42 123 456`), verify Wilcoxon significance tests, and regenerate all paper tables (`paper/tables/*.tex`) and figures (`figures/*.png`, `figures/*.pdf`).
  3. **Step 3**: Rewrite `FINAL_REPORT.md`, align `paper/main.tex` strictly with audited results, and create `COLAB_INSTRUCTIONS.md` detailing step-by-step GPU execution instructions.
- **Known Bugs, Assumptions, or Open Issues**:
  - Development machine is CPU-only, so full 100-epoch convergence on the complete 4,831-tile Sen1Floods11 dataset requires GPU execution via Colab (`notebooks/colab_full_experiments.ipynb`).
  - Outside of `--smoke` mode, the real Sen1Floods11 GeoTIFF directory is required; the code will explicitly fail fast with `FileNotFoundError` rather than silently degrading to synthetic data.
- **Tasks Requiring User Action (Colab GPU Execution)**:
  - If full 100-epoch training on the entire 4,831-tile Sen1Floods11 dataset is desired on GPU:
    1. Upload repository or clone into Google Colab with GPU runtime (T4 or A100).
    2. Run `notebooks/colab_full_experiments.ipynb` (approx. 2-3 hours on T4 GPU).
    3. Download generated bundle: `resq_ai_results.zip`.
    4. Unpack into local `results/` and run `python experiments/generate_tables.py ; python experiments/generate_figures.py`.
- **Exact Commands to Resume Local Runs**:
  - To resume/run all experiments in smoke mode:
    ```bash
    .venv\Scripts\python.exe experiments/run_experiments.py --smoke --seeds 42 123 456 --experiment all
    ```
  - To resume/run a specific experiment (e.g. E1 or E6):
    ```bash
    .venv\Scripts\python.exe experiments/run_experiments.py --smoke --seeds 42 123 456 --experiment e1
    .venv\Scripts\python.exe experiments/run_experiments.py --smoke --seeds 42 123 456 --experiment e6
    ```
  - To resume model training from checkpoint:
    ```bash
    .venv\Scripts\python.exe src/models/train.py --resume checkpoints/resqnet_best.pt
    ```
  - To run the full test suite:
    ```bash
    .venv\Scripts\pytest tests/ -q
    ```

## Phase 6: Operational System & Interactive Demo
- **Status**: COMPLETE
- **Done**: FastAPI service (`app/api/main.py`) with real model inference, MC impact sampling, and greedy allocation. Interactive Streamlit dashboard (`app/dashboard/streamlit_app.py`). Full suite of API integration tests in `tests/test_api.py`.
- **Next**: Paper and final QA.
- **Blockers**: None.

## Phase 7: IEEE Paper & Dissemination
- **Status**: IN PROGRESS (Audit Alignment)
- **Done**: LaTeX manuscript in `paper/main.tex` (IEEEtran format), supplementary materials, dynamic table generation scripts.
- **Next**: Update paper tables with 3-seed audited results and finalize text.
- **Blockers**: None.
