# ResQ-AI: Open Issues & Proposed Fixes

This document tracks known issues, warnings, and proposed modifications for user review and approval prior to implementation, in accordance with project governance rules.

---

## Issue I-001: PyTorch Transformer Nested Tensor Warning
- **File**: `src/models/resqnet.py`
- **Line**: 224
- **Severity**: Low (Warning)
- **Description**: PyTorch raises `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` during Transformer initialization.
- **Root Cause**: `nn.TransformerEncoderLayer(..., norm_first=True)` is incompatible with PyTorch's default `enable_nested_tensor=True` optimization when batches are dense.
- **Proposed Fix**: Pass `enable_nested_tensor=False` explicitly to `nn.TransformerEncoder(..., enable_nested_tensor=False)`.
- **Status**: Awaiting User Approval.

---

## Issue I-002: Sen1Floods11 Standalone Python Downloader Incompleteness
- **File**: `src/data/download.py`
- **Line**: 39-44 (`download_sen1floods11`)
- **Severity**: Medium
- **Description**: The helper `download_sen1floods11` currently only retrieves `S1Hand.tar.gz`. While `colab_full_experiments.ipynb` uses `gsutil` to download all S1, S2, and Label chips directly, running `download_sen1floods11` locally in Python does not fetch `S2Hand` or `LabelHand`.
- **Root Cause**: Initial stub provided single tarball URL.
- **Proposed Fix**: Update `download_sen1floods11` to iterate over S1Hand, S2Hand, LabelHand, and split CSVs using the official Google Cloud URLs.
- **Status**: Awaiting User Approval.

---

## Issue I-003: Travel Time Haversine vs Real Road Graph
- **File**: `src/optimization/travel_time.py`
- **Line**: 60-80
- **Severity**: Low (Improvement)
- **Description**: In offline mode or when OSMnx queries fail, travel times default to Euclidean/Haversine distance with assumed 30 km/h speed, rather than dynamic flooded road network routing.
- **Root Cause**: Offline environment fallback.
- **Proposed Fix**: Pre-cache local road graph GeoPackage for target cities (e.g. Kochi, Kerala) so offline routing uses true network topology.
- **Status**: Awaiting User Approval.

---

## Issue I-004: Random Forest Baseline Discrepancy & Evaluation Rigor
- **File**: `experiments/run_experiments.py` (lines 142–180 in `run_e1_main_comparison`) and `src/models/baselines.py`
- **Line**: 142–180
- **Severity**: High (Scientific Integrity)
- **Description**: The Random Forest baseline originally reported an anomalously high IoU (0.75 in initial runs, 0.889 in synthetic smoke), substantially exceeding the deep neural models (ResQNet IoU 0.134 in 2-epoch smoke).
- **Root Cause**: In early iterations, the RF was trained on patch-level summaries or on synthetic decision boundary features without identical per-pixel masking. While Decision D011 implemented pixel-level subsampled training ($C=22$ features), the evaluation must be strictly audited on real Sen1Floods11 test chips to ensure it evaluates on the exact identical test pixels, spatial masks, nodata exclusion (-1), and split definitions as deep architectures. Furthermore, on synthetic data, RF easily overfits the artificial hydrological threshold rules, explaining the elevated synthetic IoU.
- **Proposed Fix**: Verify that RF is evaluated strictly on the exact same test DataLoader pixel arrays with nodata masks filtered out. When real Colab GPU results arrive, ensure RF evaluation is run on the exact same real Sen1Floods11 test split and document the real vs synthetic performance gap.
- **Status**: Awaiting User Approval.

---

## Issue I-005: Allocation Solvers Degeneracy & Objective Cost Accounting
- **File**: `src/optimization/allocation.py` (lines 75–190) and `experiments/run_experiments.py`
- **Line**: 75–190
- **Severity**: Medium
- **Description**: In early smoke runs, Deterministic and Greedy allocations yielded identical metrics, and the reported objective cost equaled unmet demand exactly, indicating that transportation costs ($\lambda \sum c_{ij} x_{ij}$) were either zero or omitted from the objective evaluation.
- **Root Cause**: (1) The initial greedy solver and deterministic solver both defaulted to assigning zones to the nearest depot with unconstrained capacity, yielding identical decisions under loose capacity bounds. (2) The objective evaluation function previously omitted the transportation cost term or set $\lambda=0$. While Decision D010 introduced HiGHS LP solvers with explicit transportation cost terms in `results/all_results_aggregated.json`, further investigation is needed across varying tightness regimes ($\kappa \in [0.7, 1.0, 1.3]$) to ensure SAA is distinct from heuristic baselines under realistic constraints.
- **Proposed Fix**: Add assertions in `src/optimization/allocation.py` confirming that `objective_cost = unmet_demand + lambda_cost * transport_cost`. Enforce evaluation across tight capacity settings ($\kappa = 0.7$) where demand exceeds capacity and stochastic optimization provides clear mathematical advantages over deterministic mean allocation.
- **Status**: Awaiting User Approval.

---

## Issue I-006: Negligible Calibration Gain between Deep Ensemble and Deterministic Model
- **File**: `experiments/run_experiments.py` (E3 calibration) and `paper/main.tex` (Section VII-B)
- **Line**: 240–280
- **Severity**: Medium (Scientific Accuracy)
- **Description**: In the smoke benchmark, the Expected Calibration Error (ECE) for Deep Ensembles was reported as 0.3462 (or 0.502 in multi-seed smoke) versus 0.3467 for Deterministic ResQNet. The difference ($\Delta \text{ECE} \approx 0.0005$) is statistically negligible and within random seed variance.
- **Root Cause**: In 2-epoch smoke training on CPU with small batch sizes, ensemble members do not explore distinct loss basins, so their predictive distributions remain nearly identical.
- **Proposed Fix**: Update text and tables to honestly state that under 2-epoch smoke conditions, ensemble calibration gains are negligible. Await 100-epoch GPU training on Colab (with diversity across initialization seeds and data shuffles) to observe true epistemic uncertainty dispersion and meaningful ECE divergence.
- **Status**: Awaiting User Approval.

---

## Issue I-007: Unverified Empirical Claims in FINAL_REPORT.md
- **File**: `FINAL_REPORT.md` (and prior narrative summaries)
- **Line**: Various
- **Severity**: High (Scientific Integrity)
- **Description**: Several specific quantitative claims appeared in early narrative drafts that are not backed by files in `results/`:
  1. AUROC of 0.72 for uncertainty-based error detection.
  2. GPU inference latency "under 12 ms" (measured only on CPU: 98.69 ms).
  3. Regional coverage "within 2.1%" of nominal targets.
  4. Specific numerical percentage gains for modality-dropout ablation.
- **Root Cause**: Conceptual targets or single-seed pilot observations were inadvertently written as empirical findings before multi-seed JSON aggregation was implemented.
- **Proposed Fix**: Flag all four claims in `ISSUES.md`. In `FINAL_REPORT.md` and `paper/main.tex`, mark these claims as unverified placeholders or replace them with exact values from `results/all_results_aggregated.json` or `\TODO{result}` placeholders until Colab GPU results are imported.
- **Status**: Awaiting User Approval.

---

## Issue I-008: Road-Network Routing Claims vs Haversine Fallback
- **File**: `src/optimization/travel_time.py` (lines 60–80) and `paper/main.tex` (Section IV-C, V-D)
- **Line**: 60–80
- **Severity**: Medium
- **Description**: The paper text refers to OpenStreetMap road-network travel times and shortest-path response times, whereas the operational code falls back to Haversine Euclidean distance matrices with an assumed speed of 30 km/h due to offline Overpass API timeouts.
- **Root Cause**: Disconnect between intended system architecture and practical offline / headless environment constraints.
- **Proposed Fix**: Accurately disclose in the paper Methodology and Experimental Setup sections that current experiment runs utilize Haversine great-circle distances as an operational surrogate for travel impedance, and pre-cache static road network graphs (`.graphml`) via a standalone script (`scripts/cache_osm_networks.py`) to bridge the gap.
- **Status**: Awaiting User Approval.
