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
