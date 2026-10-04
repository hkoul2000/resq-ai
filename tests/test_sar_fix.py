"""Unit tests for Issue I-009 fixes: SAR preprocessing, valid_mask loss/metrics, and eval sizing."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.sen1floods11 import Sen1Floods11Dataset
from src.evaluation.metrics import FloodMetrics
from src.models.losses import BCEDiceLoss
from experiments.run_experiments import compute_dataset_pos_weight, train_simple, SimpleUNet


def test_sar_imputation_and_sanitization():
    """Verify that SAR nodata (-9999, extreme dB, NaNs) is properly imputed and normalized."""
    ds = Sen1Floods11Dataset(
        root="data/",
        split="train",
        modalities=["sar"],
        crop_size=64,
        smoke=True,
    )

    # Synthetic SAR chip (2, 64, 64) with -9999 nodata and NaNs
    raw_sar = np.random.normal(loc=-12.5, scale=4.0, size=(2, 64, 64)).astype(np.float32)
    raw_sar[0, 0:5, 0:5] = -9999.0  # Nodata sentinel
    raw_sar[1, 10:15, 10:15] = float("nan")  # NaN
    raw_sar[0, 20:25, 20:25] = -75.0  # Physically impossible SAR dB (< -50 dB)
    raw_sar[1, 30:35, 30:35] = 45.0   # Physically impossible SAR dB (> 25 dB)

    # Mask unphysical values as in _load_tif
    raw_sar[raw_sar <= -9000.0] = np.nan
    raw_sar[(raw_sar < -50.0) | (raw_sar > 25.0)] = np.nan

    # Impute
    clean_sar = ds._impute_sar(raw_sar)

    # Assert no NaNs or Infs remain after imputation
    assert not np.isnan(clean_sar).any(), "Imputed SAR should not contain NaNs"
    assert not np.isinf(clean_sar).any(), "Imputed SAR should not contain Infs"

    # Normalize
    norm_sar = (clean_sar - ds.s1_mean[:, None, None]) / (ds.s1_std[:, None, None] + 1e-8)
    norm_sar = np.nan_to_num(norm_sar, nan=0.0, posinf=5.0, neginf=-5.0)

    # Check bounds
    assert norm_sar.min() > -10.0, f"Normalized SAR min should be well-behaved, got {norm_sar.min()}"
    assert norm_sar.max() < 10.0, f"Normalized SAR max should be well-behaved, got {norm_sar.max()}"


def test_first_epoch_sar_nan_detection():
    """Verify that train_simple detects NaNs in SAR and raises ValueError."""
    model = SimpleUNet(in_channels=2, sar_only=True)
    
    # Create a batch with a NaN in SAR
    bad_batch = {
        "sar": torch.full((2, 2, 32, 32), float("nan")),
        "optical": torch.zeros((2, 13, 32, 32)),
        "label": torch.zeros((2, 32, 32)),
        "valid_mask": torch.ones((2, 32, 32)),
    }
    loader = [bad_batch]

    with pytest.raises(ValueError, match="NaN .* detected in SAR tensor"):
        train_simple(
            model=model,
            train_loader=loader,
            val_loader=loader,
            epochs=1,
            smoke=False,
        )


def test_valid_mask_in_bce_dice_loss():
    """Verify that BCEDiceLoss strictly ignores pixels where valid_mask is 0."""
    criterion = BCEDiceLoss(pos_weight=torch.tensor([2.0]))
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

    valid_mask = torch.ones(2, 1, 32, 32)
    valid_mask[:, :, :10, :10] = 0.0  # Top-left corner is nodata / invalid

    loss_1 = criterion(logits, targets, valid_mask=valid_mask)

    # Now change logits and targets in the INVALID region drastically
    logits_perturbed = logits.clone()
    logits_perturbed[:, :, :10, :10] = 999.0
    targets_perturbed = targets.clone()
    targets_perturbed[:, :, :10, :10] = 1.0 - targets[:, :, :10, :10]

    loss_2 = criterion(logits_perturbed, targets_perturbed, valid_mask=valid_mask)

    assert torch.isclose(loss_1, loss_2, atol=1e-5), (
        f"Loss should be invariant to changes in masked/nodata pixels: {loss_1.item()} vs {loss_2.item()}"
    )


def test_valid_mask_in_flood_metrics():
    """Verify that FloodMetrics ignores pixels where valid_mask is 0."""
    probs = torch.full((1, 1, 32, 32), 0.1)
    labels = torch.zeros((1, 1, 32, 32))

    # Real flood in bottom half
    probs[:, :, 16:, :] = 0.9
    labels[:, :, 16:, :] = 1.0

    valid_mask = torch.ones((1, 1, 32, 32))
    # Mark top 5 rows as nodata (-1 in Sen1Floods11)
    valid_mask[:, :, :5, :] = 0.0

    m1 = FloodMetrics.compute_all(probs, labels, valid_mask=valid_mask)

    # Alter predictions in the invalid nodata region
    probs_perturbed = probs.clone()
    probs_perturbed[:, :, :5, :] = 0.99  # Would trigger false positives if not masked

    m2 = FloodMetrics.compute_all(probs_perturbed, labels, valid_mask=valid_mask)

    assert m1["iou"] == m2["iou"], f"IoU should be identical with valid_mask: {m1['iou']} vs {m2['iou']}"
    assert m1["precision"] == m2["precision"]
    assert m1["recall"] == m2["recall"]


def test_compute_dataset_pos_weight():
    """Verify that pos_weight calculation properly reflects class imbalance."""
    # Create batch with 10% positive pixels
    batch = {
        "label": torch.zeros((2, 100, 100)),
        "valid_mask": torch.ones((2, 100, 100)),
    }
    batch["label"][:, :10, :] = 1.0  # 10% positive, 90% negative
    loader = [batch]

    pos_w = compute_dataset_pos_weight(loader)
    # Expected: 90 / 10 = 9.0
    assert abs(pos_w - 9.0) < 0.1, f"Expected pos_weight ~9.0, got {pos_w}"
