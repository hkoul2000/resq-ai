# ResQ-AI: Project Plan

## Overview
**ResQ-AI**: Uncertainty-Aware Multimodal AI Framework for Urban Disaster Prediction, Impact Assessment and Emergency Resource Allocation.

Primary hazard: Urban Flooding | Region focus: Indian flood events + cross-country generalization

## Hardware Context
- Development machine: Windows, no GPU (CPU-only)
- All training code supports `--smoke` mode (CPU, <5 min)
- Full experiments designed for Google Colab T4 (16 GB VRAM)
- Ready-to-run Colab notebooks provided

## Phase Milestones

### Phase 0: Repo Setup ✅ TARGET: Day 1
- [x] Git init, directory structure
- [x] Python environment (py -3.13, venv)
- [x] requirements.txt with pinned versions
- [x] Makefile, Dockerfile, pyproject.toml
- [x] Pre-commit hooks
- [x] DECISIONS.md, PROGRESS.md initialized

### Phase 1: Literature Review — TARGET: Day 1-2
- [ ] Systematic review of 60+ papers (2019-2026)
- [ ] Comparison table
- [ ] Research gap articulation
- [ ] Research questions and contributions

### Phase 2: Data Pipeline — TARGET: Day 2-4
- [ ] Sen1Floods11 download and processing
- [ ] DEM/terrain feature derivation
- [ ] Land cover integration
- [ ] CHIRPS rainfall sequences
- [ ] Exposure data (WorldPop, OSM)
- [ ] PyTorch Dataset with modality masks
- [ ] EDA report
- [ ] Data card

### Phase 3: Model — TARGET: Day 4-7
- [ ] ResQNet architecture (SAR/optical + geo + rainfall + fusion)
- [ ] All UQ methods (ensemble, MC dropout, evidential, TTA, conformal)
- [ ] Training pipeline with MLflow
- [ ] Impact estimation module
- [ ] Smoke test passing

### Phase 4: Optimization — TARGET: Day 7-9
- [ ] Stochastic allocation formulation
- [ ] OR-Tools/Pyomo implementation
- [ ] Deterministic + uncertainty-aware + CVaR + chance-constrained
- [ ] Baselines (greedy, proportional, oracle)

### Phase 5: Experiments — TARGET: Day 9-14
- [ ] E1-E8 all experiment scripts
- [ ] 5-seed runs (smoke mode on CPU, full on Colab)
- [ ] Results tables, figures, statistical tests
- [ ] Qualitative case studies

### Phase 6: System & Demo — TARGET: Day 14-15
- [ ] FastAPI service
- [ ] Streamlit dashboard
- [ ] Docker compose
- [ ] Screenshots

### Phase 7: Paper — TARGET: Day 15-18
- [ ] Full LaTeX paper (IEEEtran, 10-14 pages)
- [ ] All figures from code
- [ ] References verified
- [ ] Supplementary materials
- [ ] Self-critique

### Phase 8: Final QA — TARGET: Day 18-19
- [ ] All tests pass
- [ ] Clean environment reproduction
- [ ] Numbers cross-checked
- [ ] FINAL_REPORT.md
- [ ] Git tag v1.0.0
