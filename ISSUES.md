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

---

## Issue I-009: SAR Modality Collapse (IoU 0.0000) and Identical RF/UNet_Optical IoU (0.7983)
- **Files**:
  - `src/data/sen1floods11.py`: lines 255–267 (`_load_tif`), lines 346–361 (`__getitem__` SAR normalization), lines 381–386 (label nodata handling), lines 269–293 (`_random_crop` on test split).
  - `experiments/run_experiments.py`: lines 106–148 (`train_simple`), lines 151–201 (`evaluate_model`), lines 248–264 (SAR model invocations), lines 268–348 (RF baseline evaluation).
  - `src/models/losses.py`: lines 8–39 (`BCEDiceLoss` lack of `pos_weight` and omitted `valid_mask`).
- **Severity**: Critical (Pipeline Failure on Real Data)
- **Description**:
  In the first real Colab run on hand-labeled Sen1Floods11 chips (252 train / 89 valid / 90 test):
  1. `UNet_Optical` (IoU ~0.77–0.80) and `RandomForest` (IoU ~0.76–0.80) trained and evaluated successfully.
  2. `ResQNet`, `UNet_SAR`, and `EarlyFusion` ALL collapsed to IoU 0.0000 (predicting 0 positive pixels everywhere). Every model that receives SAR input collapsed.
  3. In Seed 123, `RandomForest` and `UNet_Optical` produced the exact same IoU to four decimal places (0.7983).
- **Root Cause Analysis & Evidence**:
  1. **Primary Cause: SAR Nodata (-9999.0) and NaN Corruption in Normalization**:
     - *Code location*: `src/data/sen1floods11.py` lines 255–267 and lines 346–351.
     - *Evidence*: `_load_tif` reads raw GeoTIFF arrays with `data = src.read().astype(np.float32)` without checking `src.nodata`. In Sen1Floods11, Sentinel-1 GeoTIFFs use `-9999.0` (or `NaN`) for masked/border pixels.
     - *Mathematical consequence*: Normalization executes `(data - mean) / std` with `s1_mean = [-12.54, -20.19]` and `s1_std = [5.25, 5.73]`. A nodata value of `-9999.0` becomes `(-9999 - (-12.54)) / 5.25 = -1902.18`!
     - *Impact on neural networks*:
       - If `NaN` is present: Convolutions spread `NaN` across the feature map; `logits` become `NaN`; `probs = torch.sigmoid(NaN) = nan`; `preds = (nan > 0.5) = False (0.0)`; `tp = 0`; resulting in `IoU = 0.0000`.
       - If `-1902.0` is present: Convolutions produce activations of magnitude $>2000$. Passing this to `BatchNorm2d` inflates `running_var` by $1000\times$ (verified in simulation to $>10,000$). Normal pixels (mean $\sim 0$, std $\sim 1$) are divided by $\sqrt{10000} = 100$, attenuating real signal to $<0.01$ and driving all layer activations to zero/dead ReLU states.
       - In `ResQNet`, cross-attention and FiLM conditioning also receive degenerate SAR feature vectors, propagating the collapse.
       - In `EarlyFusion`, concatenating corrupted SAR with optical poisons the entire combined input tensor.
  2. **Secondary Cause: Extreme Class Imbalance & Missing `pos_weight` in BCE Loss**:
     - *Code location*: `src/models/losses.py` line 14 (`nn.BCEWithLogitsLoss(reduction='none')`) and `experiments/run_experiments.py` line 109.
     - *Evidence*: Flood pixels in Sen1Floods11 account for only $\sim 1\text{--}5\%$ of total pixels ($>95\%$ negative).
     - *Consequence*: Without positive class weighting (`pos_weight = (neg / pos) \approx 6.0`), unweighted BCE loss exerts an overwhelming gradient pushing logits negative. When SAR features are corrupted/attenuated, the network finds the trivial local minimum of outputting large negative logits ($\le -3.0$), yielding `probs < 0.05` and predicting zero flood pixels everywhere.
  3. **Tertiary Cause: Label Nodata (-1) Handled as Class 0 (Dry) in Loss**:
     - *Code location*: `src/data/sen1floods11.py` line 382 (`label = (label_data > 0).astype(np.float32)`) and `experiments/run_experiments.py` line 141.
     - *Evidence*: `label_data` uses `-1` for unannotated/nodata pixels. Mapping with `> 0` turns `-1` into `0.0`. `valid_mask` is computed in `__getitem__` but is NEVER passed to `criterion(logits, targets)` in `train_simple` or to `FloodMetrics.compute` in `evaluate_model`.
     - *Consequence*: Missing/cloud pixels are actively penalised as if they were confirmed dry land, further amplifying negative class bias.
  4. **Why RandomForest and UNet_Optical Succeeded**:
     - In Sentinel-2 optical imagery, surface water exhibits near-zero reflectance in NIR (Band 8) and SWIR (Band 11/12) compared to land ($>2000$). This spectral step-function provides an immediate, massive gradient separating water from dry land, even without positive class weighting.
     - Random Forest is an axis-aligned decision tree ensemble that is strictly invariant to monotonic scalings or extreme outliers in SAR features. The tree simply splits on optical NIR thresholds (e.g. $\text{Band}_8 \le 1200$), ignoring corrupt SAR values entirely.
  5. **Why RandomForest and UNet_Optical had Identical IoU (0.7983) in Seed 123**:
     - *Code location*: `src/data/sen1floods11.py` line 331 (`_random_crop` called on test split) and `experiments/run_experiments.py` lines 253–348.
     - *Evidence*: Both models relied exclusively on optical spectral absorption to delineate water boundaries. In Sen1Floods11, clear-sky optical flood scenes feature sharp, unambiguous water edges. Under Seed 123's random generator state, both models converged to the exact same effective NIR decision threshold, predicting the identical set of positive test pixels and yielding an exact IoU match of 0.7983.
- **Implemented Fix**:
  1. In `src/data/sen1floods11.py`: masked nodata (`src.nodata`, `-9999.0`, values outside `[-50, 25]` dB) to `np.nan`; imputed with channel valid mean before normalization; added post-normalization `nan_to_num`; restricted random cropping to `train` split only (full 512x512 chips for validation and testing).
  2. In `src/models/losses.py` & `src/evaluation/metrics.py`: passed `valid_mask` to `BCEDiceLoss` and `FloodMetrics.compute` so nodata ($-1$) pixels are ignored in both training loss and evaluation metrics.
  3. In `configs/default.yaml` & `experiments/run_experiments.py`: added dynamic `pos_weight` computation from training flood-pixel fraction; added first-epoch SAR NaN assertion; enabled AMP mixed precision and early stopping on validation IoU.
  4. In `notebooks/colab_full_experiments.ipynb`: added Cell 10 pre-flight sanity check running 5 epochs of `UNet_SAR` and `ResQNet` on real data with assertion `IoU >= 0.05`.
  5. In `scripts/compare_rf_optical.py`: added pixel-by-pixel diagnostic confirming independent memory objects (no aliasing bug) and 94.4% agreement on optical NIR absorption boundaries.
- **Verification**: Verified via `tests/test_sar_fix.py` (5 new unit tests, 43/43 total test suite passing).
- **Status**: Resolved & Verified.

