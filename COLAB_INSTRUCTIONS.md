# Google Colab GPU Execution Instructions

This document provides exact, cell-by-cell instructions for running the complete 100-epoch ResQ-AI experimental evaluation on an NVIDIA T4 (or A100) GPU on Google Colab, and importing the verified empirical artifacts back into this repository.

---

## ⚠️ Before You Run — GitHub URL Placeholder

> **ACTION REQUIRED**: Cell 7 (`# Clone repo and install`) contains a placeholder repository URL:
> ```
> REPO_URL = 'https://github.com/resq-ai/resq-ai.git'  # <-- REPLACE WITH YOUR REPO URL
> ```
> **You must replace this with your actual GitHub repository URL before running the notebook.**
> For example: `https://github.com/YOUR_USERNAME/resq-ai.git`

---

## 1. Prerequisites and Setup

1. Open [Google Colab](https://colab.research.google.com).
2. Upload the notebook file located at `notebooks/colab_full_experiments.ipynb` (or open it directly from GitHub).
3. **Change Runtime Type**:
   - Navigate to: **Runtime → Change runtime type**
   - Under **Hardware accelerator**, select **T4 GPU** (or **A100 GPU** if available).
   - Click **Save**.

---

## 2. New Disconnect-Safety Features (Added in Step 1)

The notebook now includes **Google Drive backup and resume logic**:

- **Cell 2 (Drive Mount)**: Mounts Google Drive at `/content/drive`. All checkpoints and result JSON files are saved to `MyDrive/resq_ai_checkpoints/` after every experiment.
- **Cell 3 (Resume Detection)**: On startup, checks which experiments have already completed (`.done` flag files in Drive). If Colab disconnects mid-run, re-running the notebook will **automatically skip** completed experiments and resume from the next pending one.
- **Per-experiment Drive sync**: After every experiment (E1–E7) completes, results are synced to Drive immediately, so a disconnect never loses more than one experiment's work.
- **Final zip to Drive**: The results zip (`resq_ai_results.zip`) is also saved to Drive before the browser download, so it can be retrieved from Drive if the browser download fails.

### Resume workflow (after a Colab disconnect):
1. Re-open the notebook in Colab (same session or new session).
2. Run **all cells from the top** — Drive will re-mount, resume state will be read, and completed experiments are skipped automatically.
3. Continue until all experiments show ✓ DONE.

---

## 3. Cell-by-Cell Execution Plan

| Cell # | Section / Command | Purpose | Expected Runtime (T4 GPU) | Key Output Files Generated |
| :--- | :--- | :--- | :--- | :--- |
| **Cell 1** | Markdown Header | Overview and setup notes | Instant | None |
| **Cell 2** | `drive.mount(...)` | **[NEW]** Mount Google Drive for checkpoint backup | ~30 seconds (auth) | `MyDrive/resq_ai_checkpoints/` created |
| **Cell 3** | Resume detection + `is_done` / `mark_done` / `sync_to_drive` | **[NEW]** Check which experiments are already done; restore prior results from Drive | Instant | Status printout |
| **Cell 4** | `!nvidia-smi` | Verify NVIDIA driver & GPU availability (16 GB VRAM) | <5 seconds | Console GPU status |
| **Cell 5** | `!git clone ... && pip install -q -r requirements.txt` | Clone repository (**replace placeholder URL first!**) and install pinned dependencies | ~2–3 minutes | dependencies installed |
| **Cell 6** | `!python -m pytest tests/test_smoke.py -v -m smoke --tb=short` | Pre-flight smoke validation on PyTorch, UQ, and solvers | ~30 seconds | 38/38 passing unit tests |
| **Cell 7** | Markdown: Download Real Sen1Floods11 Dataset | Section Header | Instant | None |
| **Cell 8** | `!gsutil -m cp -r gs://sen1floods11/v1.1/...` | Download official hand-labeled Sen1Floods11 chips (446 chips, ~3 GB) and split CSVs | ~1–2 minutes | `data/sen1floods11/v1.1/...` |
| **Cell 9** | Markdown: Run Experiments | Header | Instant | None |
| **Cell 10** | E1 with skip-if-done guard | **E1: Main Model Comparison** (ResQNet, UNet-SAR, UNet-Optical, EarlyFusion, pixel-level Random Forest across 5 seeds); saves to Drive on completion | ~45–60 minutes | `results/e1_*.json`; `e1.done` in Drive |
| **Cell 11** | E2 with skip-if-done guard | **E2: Ablation Studies** (Modality ablation, FiLM/Cross-Attention ablation, Modality Dropout) | ~35–45 minutes | `results/e2_*.json`; `e2.done` in Drive |
| **Cell 12** | E3 with skip-if-done guard | **E3: Calibration & Uncertainty** (Deterministic, MC Dropout T=20, TTA, Deep Ensemble M=5) | ~30–40 minutes | `results/e3_*.json`; `e3.done` in Drive |
| **Cell 13** | E4 with skip-if-done guard | **E4: Split Conformal Prediction** (Marginal coverage and set sizes across α ∈ {0.01, 0.05, 0.1, 0.2}) | ~15–20 minutes | `results/e4_*.json`; `e4.done` in Drive |
| **Cell 14** | E5 with skip-if-done guard | **E5: Sensor Robustness** (SAR noise levels, optical cloud obscuration, missing combinations) | ~25–35 minutes | `results/e5_*.json`; `e5.done` in Drive |
| **Cell 15** | E6 with skip-if-done guard | **E6: Resource Allocation** (HiGHS LP solvers: Deterministic, SAA, CVaR, Greedy, Proportional, Oracle, sensitivity analysis) | ~5–10 minutes | `results/e6_*.json`; `e6.done` in Drive |
| **Cell 16** | E7 with skip-if-done guard | **E7: Computational Efficiency** (GPU forward latency, parameter counts, throughput FPS) | ~5 minutes | `results/e7_*.json`; `e7.done` in Drive |
| **Cell 17** | Markdown: Generate Figures and Tables | Header | Instant | None |
| **Cell 18** | `generate_figures.py` + `generate_tables.py` + Drive sync | Compile all LaTeX tables and vector figures; sync to Drive | ~1–2 minutes | `figures/*.png`, `figures/*.pdf`, `paper/tables/*.tex`; Drive backup |
| **Cell 19** | `display(Image(...))` | Render and inspect generated figures in Colab output | ~10 seconds | Visual verification |
| **Cell 20** | Markdown: Download Results | Header | Instant | None |
| **Cell 21** | `!zip ...` + Drive copy + `files.download(...)` | Archive all results into zip; save to Drive; trigger browser download | ~30 seconds | `resq_ai_results.zip` (local + Drive) |

**Total Estimated GPU Wallclock Time**: ~2.5–3.5 hours on NVIDIA T4.

---

## 4. How to Bring Results Back to the Local Repository

1. When Cell 21 finishes, your browser will automatically download `resq_ai_results.zip`.
   - If the browser download fails, retrieve it from `MyDrive/resq_ai_checkpoints/resq_ai_results.zip`.
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
