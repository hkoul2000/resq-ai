# Google Colab GPU Execution Instructions

This document provides exact, cell-by-cell instructions for running the complete 100-epoch ResQ-AI experimental evaluation on an NVIDIA T4 (or A100) GPU on Google Colab, and importing the verified empirical artifacts back into this repository.

---

## 1. Prerequisites and Setup

1. Open [Google Colab](https://colab.research.google.com).
2. Upload the notebook file located at `notebooks/colab_full_experiments.ipynb` (or open it directly from GitHub).
3. **Change Runtime Type**:
   - Navigate to: **Runtime $\to$ Change runtime type**
   - Under **Hardware accelerator**, select **T4 GPU** (or **A100 GPU** if available).
   - Click **Save**.

---

## 2. Cell-by-Cell Execution Plan

| Cell # | Section / Command | Purpose | Expected Runtime (T4 GPU) | Key Output Files Generated |
| :--- | :--- | :--- | :--- | :--- |
| **Cell 1** | Markdown Header | Overview and setup notes | Instant | None |
| **Cell 2** | `!nvidia-smi` | Verify NVIDIA driver & GPU availability (16GB VRAM) | $< 5$ seconds | Console GPU status |
| **Cell 3** | `!git clone ... && %cd resq-ai && !pip install -q -r requirements.txt` | Clone repository and install pinned dependencies | $\approx 2\text{--}3$ minutes | `.venv` dependencies |
| **Cell 4** | `!python -m pytest tests/test_smoke.py -v -m smoke --tb=short` | Pre-flight smoke validation on PyTorch, UQ, and solvers | $\approx 30$ seconds | 38/38 passing unit tests |
| **Cell 5** | Markdown: Run Experiments | Header | Instant | None |
| **Cell 6** | `!python experiments/run_experiments.py --experiment e1 --config configs/default.yaml` | **E1: Main Model Comparison** (ResQNet, UNet-SAR, UNet-Optical, EarlyFusion, pixel-level Random Forest across 5 seeds) | $\approx 45\text{--}60$ minutes | `results/e1_seed_*.json`, `results/e1_comparison.json` |
| **Cell 7** | `!python experiments/run_experiments.py --experiment e2 --config configs/default.yaml` | **E2: Ablation Studies** (Modality ablation, FiLM/Cross-Attention ablation, Modality Dropout) | $\approx 35\text{--}45$ minutes | `results/e2_seed_*.json`, `results/e2_ablations.json` |
| **Cell 8** | `!python experiments/run_experiments.py --experiment e3 --config configs/default.yaml` | **E3: Calibration & Uncertainty** (Deterministic, MC Dropout $T=20$, TTA, Deep Ensemble $M=5$) | $\approx 30\text{--}40$ minutes | `results/e3_seed_*.json`, `results/e3_calibration.json` |
| **Cell 9** | `!python experiments/run_experiments.py --experiment e4 --config configs/default.yaml` | **E4: Split Conformal Prediction** (Marginal coverage and set sizes across $\alpha \in \{0.01, 0.05, 0.1, 0.2\}$) | $\approx 15\text{--}20$ minutes | `results/e4_seed_*.json`, `results/e4_conformal.json` |
| **Cell 10** | `!python experiments/run_experiments.py --experiment e5 --config configs/default.yaml` | **E5: Sensor Robustness** (SAR noise levels, optical cloud obscuration, missing combinations) | $\approx 25\text{--}35$ minutes | `results/e5_seed_*.json`, `results/e5_robustness.json` |
| **Cell 11** | `!python experiments/run_experiments.py --experiment e6 --config configs/default.yaml` | **E6: Resource Allocation Optimization** (HiGHS LP solvers: Deterministic Mean, SAA, CVaR, Greedy, Proportional, Oracle, sensitivity analysis) | $\approx 5\text{--}10$ minutes | `results/e6_seed_*.json`, `results/e6_allocation.json` |
| **Cell 12** | `!python experiments/run_experiments.py --experiment e7 --config configs/default.yaml` | **E7: Computational Efficiency Profiling** (GPU forward latency, parameter counts, throughput FPS) | $\approx 5$ minutes | `results/e7_seed_*.json`, `results/e7_efficiency.json` |
| **Cell 13** | Markdown: Generate Figures and Tables | Header | Instant | None |
| **Cell 14** | `!python experiments/generate_figures.py && !python experiments/generate_tables.py` | Automatically compile all LaTeX tables and vector figures directly from `results/all_results_aggregated.json` | $\approx 1\text{--}2$ minutes | `figures/*.png`, `figures/*.pdf`, `paper/tables/*.tex` |
| **Cell 15** | `display(Image(...))` | Render and inspect generated figures in the Colab output | $\approx 10$ seconds | Visual verification |
| **Cell 16** | Markdown: Download Results | Header | Instant | None |
| **Cell 17** | `!zip -r resq_ai_results.zip results/ figures/ paper/tables/` + `files.download(...)` | Archive all empirical results, figures, and tables into a single zip bundle and download | $\approx 30$ seconds | `resq_ai_results.zip` |

**Total Estimated GPU Wallclock Time**: $\approx 2.5\text{--}3.5$ hours on NVIDIA T4.

---

## 3. How to Bring Results Back to the Local Repository

1. When Cell 17 finishes, your browser will automatically download `resq_ai_results.zip`.
2. Move `resq_ai_results.zip` into the root directory of this repository:
   ```
   c:\Users\u430\OneDrive\Desktop\PROJECTS\RESQ-AI\
   ```
3. Extract the archive into the repository root:
   - On Windows PowerShell:
     ```powershell
     Expand-Archive -Path resq_ai_results.zip -DestinationPath . -Force
     ```
   - On Linux/macOS:
     ```bash
     unzip -o resq_ai_results.zip -d .
     ```
4. Verify that the files in `results/`, `figures/`, and `paper/tables/` reflect the GPU-converged runs.
5. Recompile the paper:
   - Run `pdflatex paper/main.tex` or view the updated tables and figures in `paper/tables/` and `figures/`.
6. Run the local test suite to confirm complete integrity:
   ```powershell
   .venv\Scripts\pytest tests/ -q
   ```
