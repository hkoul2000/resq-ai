# ResQ-AI: Design Decisions Log

## D001: Python Version
**Decision**: Use Python 3.13 (available on dev machine) with compatibility target 3.11+
**Reasoning**: Spec requires 3.11, but 3.13 is available locally. Docker uses 3.11. All code written to be compatible with 3.11+.
**Date**: 2026-10-02

## D002: No Local GPU
**Decision**: All development done CPU-only with --smoke mode. Full experiments via Colab notebooks.
**Reasoning**: Dev machine has no GPU. Code designed with T4 16GB target. Smoke tests validate logic on CPU.
**Date**: 2026-10-02

## D003: Timm Backbone Encoder Feature Extraction
**Decision**: In `ResQNet._encode_image`, check `isinstance(encoder, nn.ModuleList)` to branch between custom fallback encoder and timm's `features_only=True` wrapper.
**Reasoning**: In modern `timm` releases, `features_only=True` models do not define `forward_features` and calling `encoder(x)` directly returns the list of feature maps across downsampling stages.
**Date**: 2026-10-02

## D004: Pure NumPy/SciPy Hydrological Terrain Analysis
**Decision**: Implement D8 flow routing, Height Above Nearest Drainage (HAND), and Topographic Wetness Index (TWI) directly in pure NumPy and SciPy.
**Reasoning**: Eliminates fragile C-bindings to external geospatial dependencies (like GDAL and pysheds), enabling 100% platform-independent elevation processing across Windows, Linux, and Colab environments.
**Date**: 2026-10-02

## D005: Flexible Dictionary & Keyword Unpacking in Model Forward Pass
**Decision**: Allow `ResQNet.forward` to accept either explicit keyword tensors (`sar`, `optical`, `geo`, `rainfall`) or a single batch dictionary.
**Reasoning**: Ensures seamless interoperability with PyTorch DataLoaders, standard `Trainer` training loops, and test-time evaluation pipelines without redundant manual unpacking.
**Date**: 2026-10-02

## D006: Standalone Optimization Solvers for Resource Allocation
**Decision**: Provide standalone vectorized Greedy and Proportional allocation algorithms in addition to Pyomo MILP formulations.
**Reasoning**: Ensures allocation evaluations and Monte Carlo decision simulations run reliably in environments where external MILP binaries (e.g., HiGHS or CBC) are not pre-installed.
**Date**: 2026-10-02

## D007: Split Conformal Risk Control with Empirical Quantile Adjustment
**Decision**: Calibrate non-conformity scores on validation sets using $(1-\alpha)(1 + 1/n)$ finite-sample correction.
**Reasoning**: Guarantees distribution-free marginal coverage guarantees for binary flood classification sets even under finite validation sample sizes.
**Date**: 2026-10-02

## D008: Dynamic Experiment Aggregation and Robust Offline Fallbacks
**Decision**: Connect `generate_tables.py` and `generate_figures.py` dynamically to `results/all_results_aggregated.json`. Implement spatial grid quadrant zonal aggregation in `src/impact/assessment.py` and Haversine distance travel time in `src/optimization/travel_time.py`.
**Reasoning**: Ensures zero mock/placeholder discrepancies between `results/` artifacts and generated paper tables/figures. Guarantees that the entire evaluation, impact assessment, and allocation pipeline operates reliably offline without depending on external web services or missing C libraries.
**Date**: 2026-10-02

## D009: Strict Real Data Verification & Prohibition of Silent Synthetic Fallbacks
**Decision**: In `Sen1Floods11Dataset` and `create_dataloaders`, verify chip paths and raise explicit `FileNotFoundError` if real GeoTIFFs or catalog splits are missing when running outside `--smoke` mode.
**Reasoning**: Prevents silent degradation to synthetic data in production/experiment runs. Synthetic data is now strictly restricted to `tests/` and explicit `--smoke` testing.
**Date**: 2026-10-03

## D010: Native SciPy HiGHS Solvers for Deterministic, SAA, and CVaR Allocation
**Decision**: Implement `DeterministicAllocation`, `StochasticAllocation` (SAA), and `CVaRAllocation` directly via `scipy.optimize.linprog(method='highs')`.
**Reasoning**: Eliminates external dependency on `pyomo` and external solver binaries (e.g. `appsi_highs`), ensuring exact LP optimization on any environment with standard SciPy. Solves allocation problems to exact optimality in milliseconds with proper transportation cost accounting.
**Date**: 2026-10-03

## D011: Per-Pixel Random Forest Benchmark Evaluation
**Decision**: Re-implement Random Forest baseline in E1 to train on subsampled pixels ($C=22$ features) and evaluate per-pixel on the exact same spatial grid and metrics (IoU, F1, AUROC, ECE) as deep models.
**Reasoning**: Replaces the flawed patch-classification approach (which previously yielded an artificial IoU of 0.81) with true pixel-level segmentation, ensuring scientific integrity and fair benchmark comparison.
**Date**: 2026-10-03

## D012: Multi-Seed Statistical Reporting and Significance Testing
**Decision**: Standardize experiment evaluation across at least 3 random seeds ([42, 123, 456]), reporting mean, standard deviation, 95% confidence intervals, and paired Wilcoxon signed-rank significance tests.
**Reasoning**: Ensures all scientific claims meet rigorous statistical standards and eliminates reliance on single-seed empirical variance.
**Date**: 2026-10-03

## D013: Transparent CPU Smoke Reporting & Dedicated GPU Execution Instructions
**Decision**: Honestly present all local benchmarking metrics in FINAL_REPORT.md and paper/main.tex as CPU smoke runs with hydrologically coupled synthetic data, while isolating full-scale 100-epoch GPU instructions in COLAB_INSTRUCTIONS.md.
**Reasoning**: Upholds the core principle of scientific integrity ("never fabricate results or claim GPU execution when local hardware is CPU-only"), while providing seamless reproducibility for GPU environments.
**Date**: 2026-10-03

## D014: Explicit Labeling of Synthetic Smoke Artifacts & Turnkey Colab Execution
**Decision**: Explicitly append `(Synthetic Smoke Benchmark)` to all table captions in `paper/tables/*.tex` and all figure titles in `figures/*.png`. Embed automated Google Cloud Storage `gsutil` downloads for real Sen1Floods11 imagery directly into `notebooks/colab_full_experiments.ipynb`.
**Reasoning**: Adheres strictly to the rule that synthetic/smoke results must be explicitly labeled and never presented as final, and guarantees that the GPU Colab notebook executes out-of-the-box with real satellite data.
**Date**: 2026-10-03

## D015: SAR Nodata Sanitization, Valid Mask Propagation, and Deterministic Full-Chip Evaluation
**Decision**: 
1. In `src/data/sen1floods11.py`, mask nodata sentinels (`-9999.0`, `NaN`, and physically implausible values outside `[-50.0, 25.0]` dB) in Sentinel-1 SAR imagery and impute with the channel's valid mean prior to z-score normalization, followed by `np.nan_to_num`.
2. Add a runtime assertion checking every batch in epoch 1 for NaN/Inf in SAR tensors to fail early if corrupt data is encountered.
3. Pass `valid_mask` into `BCEDiceLoss` and `FloodMetrics` (and ECE calculation), strictly masking out label pixels with value `-1` from loss gradients and evaluation metrics.
4. Restrict `_random_crop` strictly to `train` split; validation and test splits evaluate deterministically on full 512x512 chips.
5. Make positive class weight dynamic (`pos_weight: auto`), estimating `pos_weight = neg_pixels / pos_pixels` directly from training split pixel counts.
6. Enable automatic mixed precision (`torch.cuda.amp`) and early stopping on validation IoU to reduce wallclock runtime on Colab T4 GPU to ~2.5–3.5 hours.
7. Integrate a 5-epoch pre-flight sanity check (`--sanity_check`) into the notebook to verify validation IoU $\ge 0.05$ before launching multi-seed sweeps.
**Reasoning**: Resolves critical Issue `I-009` where `-9999` nodata in Sen1Floods11 SAR chips corrupted batch normalization statistics, leading to total modality collapse (IoU = 0.0000) for all SAR-dependent architectures (`ResQNet`, `UNet_SAR`, `EarlyFusion`). Eliminates random-crop variance during evaluation, and safeguards against runaway training times.
**Date**: 2026-10-04

