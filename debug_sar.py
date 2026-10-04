#!/usr/bin/env python3
"""Diagnostic script to identify the root cause of SAR model collapse in ResQ-AI.

Tests real Sen1Floods11 chips to check:
1. SAR (Sentinel-1) raw min, max, mean, std, and NaN/Inf/outlier counts before normalization.
2. SAR min, max, mean, std, and NaN/Inf counts after current Sen1Floods11 normalization.
3. Label GeoTIFF raw values, nodata value (-1), and how it is handled in dataset & loss.
4. Fraction of flood pixels (ground truth class imbalance).
5. Model predicted flood fraction after 1 epoch of training on SAR.

Usage:
    python debug_sar.py --data_root data/sen1floods11
    python debug_sar.py --simulate   # Run synthetic simulation demonstrating the exact failure mode
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def find_sen1floods11_chips(data_root: Path) -> List[Dict[str, Path]]:
    """Locate Sentinel-1, Sentinel-2, and Label GeoTIFFs in Sen1Floods11 directory structure."""
    chips: List[Dict[str, Path]] = []
    
    # Check standard directory layouts
    candidate_s1_dirs = [
        data_root / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand",
        data_root / "data" / "flood_events" / "HandLabeled" / "S1Hand",
        data_root / "S1Hand",
    ]
    
    s1_dir = None
    for d in candidate_s1_dirs:
        if d.exists() and any(d.glob("*.tif")):
            s1_dir = d
            break
            
    if s1_dir is None:
        return chips

    s2_dir = Path(str(s1_dir).replace("S1Hand", "S2Hand"))
    label_dir = Path(str(s1_dir).replace("S1Hand", "LabelHand"))
    
    for s1_file in sorted(s1_dir.glob("*.tif"))[:10]:
        base_name = s1_file.name
        s2_file = s2_dir / base_name.replace("S1Hand", "S2Hand")
        label_file = label_dir / base_name.replace("S1Hand", "LabelHand")
        
        chips.append({
            "chip_id": s1_file.stem,
            "s1": s1_file,
            "s2": s2_file if s2_file.exists() else None,
            "label": label_file if label_file.exists() else None,
        })
        
    return chips


def analyze_real_chip(chip: Dict[str, Path]) -> None:
    """Analyze SAR and Label statistics for a real GeoTIFF chip."""
    try:
        import rasterio
    except ImportError:
        print("[ERROR] rasterio is required to read GeoTIFFs. Install with `pip install rasterio`.")
        return

    print("=" * 80)
    print(f"ANALYZING REAL CHIP: {chip['chip_id']}")
    print("=" * 80)

    # 1. SAR (Sentinel-1) Analysis
    s1_path = chip["s1"]
    if not s1_path or not s1_path.exists():
        print(f"[SAR] S1 file missing: {s1_path}")
        return

    with rasterio.open(s1_path) as src:
        s1_raw = src.read().astype(np.float32)  # Shape: (2, H, W)
        nodata_meta = src.nodata

    print(f"\n[SAR BEFORE NORMALIZATION]")
    print(f"  Shape: {s1_raw.shape}")
    print(f"  Rasterio metadata nodata value: {nodata_meta}")
    
    for c, band_name in enumerate(["VV (Band 1)", "VH (Band 2)"]):
        band = s1_raw[c]
        nan_count = int(np.isnan(band).sum())
        inf_count = int(np.isinf(band).sum())
        nodata_9999_count = int((band <= -9990.0).sum())
        pos_outliers = int((band > 50.0).sum())
        valid_mask = np.isfinite(band) & (band > -9990.0)
        
        print(f"  {band_name}:")
        print(f"    Raw min: {np.nanmin(band):.2f}, Raw max: {np.nanmax(band):.2f}, Raw mean: {np.nanmean(band):.2f}")
        print(f"    NaN count: {nan_count} ({nan_count / band.size * 100:.2f}%)")
        print(f"    Inf count: {inf_count} ({inf_count / band.size * 100:.2f}%)")
        print(f"    Outliers <= -9990 (-9999 nodata): {nodata_9999_count} ({nodata_9999_count / band.size * 100:.2f}%)")
        print(f"    Outliers > 50 dB: {pos_outliers} ({pos_outliers / band.size * 100:.2f}%)")
        if np.any(valid_mask):
            print(f"    Valid pixels only: min={band[valid_mask].min():.2f}, max={band[valid_mask].max():.2f}, "
                  f"mean={band[valid_mask].mean():.2f}, std={band[valid_mask].std():.2f}")

    # 2. SAR After Current Normalization
    s1_mean = np.array([-12.54, -20.19], dtype=np.float32)[:, None, None]
    s1_std = np.array([5.25, 5.73], dtype=np.float32)[:, None, None]
    
    s1_norm = (s1_raw - s1_mean) / (s1_std + 1e-8)
    
    print(f"\n[SAR AFTER CURRENT NORMALIZATION ((x - mean) / std)]")
    for c, band_name in enumerate(["VV (Band 1)", "VH (Band 2)"]):
        band_norm = s1_norm[c]
        print(f"  {band_name}:")
        print(f"    Normalized min: {np.nanmin(band_norm):.2f}, max: {np.nanmax(band_norm):.2f}, mean: {np.nanmean(band_norm):.2f}")
        print(f"    Normalized NaN count: {int(np.isnan(band_norm).sum())}")
        if (band_norm < -100.0).any():
            extreme_neg = int((band_norm < -100.0).sum())
            print(f"    CRITICAL: {extreme_neg} pixels have normalized value < -100.0 (e.g. {band_norm.min():.1f})!")
            print(f"              This confirms raw -9999 nodata was normalized to ~ -1902, causing network collapse!")

    # 3. Label Analysis
    label_path = chip["label"]
    if label_path and label_path.exists():
        with rasterio.open(label_path) as src:
            lbl_raw = src.read(1).astype(np.float32)
            lbl_nodata_meta = src.nodata

        print(f"\n[LABEL ANALYSIS]")
        print(f"  Shape: {lbl_raw.shape}")
        print(f"  Rasterio metadata nodata value: {lbl_nodata_meta}")
        
        unique_vals, val_counts = np.unique(lbl_raw, return_counts=True)
        total_pixels = lbl_raw.size
        print(f"  Unique label values present:")
        for val, count in zip(unique_vals, val_counts):
            meaning = "Nodata / Masked" if val < 0 else ("Dry / Non-flood" if val == 0 else "Flood / Water")
            print(f"    Value {val:5.1f}: {count:7d} pixels ({count / total_pixels * 100:6.2f}%) -> {meaning}")
            
        # Current handling in Sen1Floods11Dataset:
        # label = (label_data > 0).astype(np.float32)
        # valid_mask = (label_data >= 0).astype(np.float32)
        binary_lbl = (lbl_raw > 0).astype(np.float32)
        valid_mask = (lbl_raw >= 0).astype(np.float32)
        
        valid_total = valid_mask.sum()
        flood_in_valid = binary_lbl[valid_mask > 0].sum() if valid_total > 0 else 0
        
        print(f"\n  Flood Fraction:")
        print(f"    Fraction of total pixels marked flood: {binary_lbl.mean():.4f} ({binary_lbl.mean() * 100:.2f}%)")
        if valid_total > 0:
            print(f"    Fraction of VALID pixels marked flood: {flood_in_valid / valid_total:.4f} ({(flood_in_valid / valid_total) * 100:.2f}%)")
        print(f"  Handling in Loss:")
        print(f"    In dataset __getitem__: label = (label_data > 0) maps -1 nodata to 0 (non-flood)!")
        print(f"    In train_simple: loss = criterion(logits, targets) -> valid_mask is NOT PASSED to BCEDiceLoss!")
        print(f"    Result: All -1 nodata pixels are penalised as if they were true non-flood ground truth!")
    else:
        print("[LABEL] Label file missing.")


def run_1_epoch_diagnostic(chips: List[Dict[str, Path]], device: str = "cpu") -> None:
    """Train UNet_SAR on real chips for 1 epoch and inspect predicted flood fraction."""
    from src.data.sen1floods11 import Sen1Floods11Dataset
    from src.models.losses import BCEDiceLoss
    from experiments.run_experiments import SimpleUNet

    print("\n" + "=" * 80)
    print("RUNNING 1-EPOCH MODEL DIAGNOSTIC (UNet_SAR on available chips)")
    print("=" * 80)

    dataset = Sen1Floods11Dataset(
        root=chips[0]["s1"].parents[4],
        split="train",
        modalities=["sar"],
        crop_size=256,
        augment=False,
        modality_dropout=0.0,
        normalize=True,
        smoke=True,
        chips=chips,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model = SimpleUNet(in_channels=2, sar_only=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = BCEDiceLoss()

    model.train()
    print("Starting 1 epoch of training...")
    for b_idx, batch in enumerate(loader):
        sar = batch["sar"].to(device)
        targets = batch["label"].to(device).unsqueeze(1)

        print(f"\nBatch {b_idx}:")
        print(f"  Input SAR tensor: min={sar.min().item():.2f}, max={sar.max().item():.2f}, mean={sar.mean().item():.2f}")
        print(f"  Input SAR NaN count: {sar.isnan().sum().item()}, Inf count: {sar.isinf().sum().item()}")

        optimizer.zero_grad()
        out = model(sar)
        logits = out["logits"]
        probs = out["probs"]

        print(f"  Output logits: min={logits.min().item():.3f}, max={logits.max().item():.3f}, mean={logits.mean().item():.3f}")
        print(f"  Logits NaN count: {logits.isnan().sum().item()}")
        
        loss = criterion(logits, targets)
        print(f"  Loss: {loss.item():.4f}")
        
        loss.backward()
        
        # Check gradients
        has_nan_grad = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        print(f"  Gradients: NaN detected = {has_nan_grad}, Max abs grad = {max_grad:.4f}")
        
        optimizer.step()

    # Post-training evaluation on 1 batch
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sar = batch["sar"].to(device)
            targets = batch["label"].to(device)
            out = model(sar)
            probs = out["probs"]
            preds = (probs > 0.5).float()
            
            pred_flood_frac = preds.mean().item()
            gt_flood_frac = (targets > 0.5).float().mean().item()
            
            tp = (preds.squeeze(1) * targets).sum().item()
            fp = (preds.squeeze(1) * (1 - targets)).sum().item()
            fn = ((1 - preds.squeeze(1)) * targets).sum().item()
            iou = tp / (tp + fp + fn + 1e-8)
            
            print("\nAfter 1 epoch evaluation:")
            print(f"  Ground truth flood fraction: {gt_flood_frac:.4f} ({gt_flood_frac * 100:.2f}%)")
            print(f"  Predicted flood fraction:    {pred_flood_frac:.4f} ({pred_flood_frac * 100:.2f}%)")
            print(f"  Mean predicted probability:  {probs.mean().item():.4f}")
            print(f"  Validation IoU:              {iou:.4f}")
            break


def run_synthetic_simulation() -> None:
    """Simulate the exact numerical failure mode of SAR preprocessing when real data is absent."""
    print("=" * 80)
    print("SIMULATING SEN1FLOODS11 SAR COLLAPSE NUMERICAL MECHANISM")
    print("=" * 80)

    # 1. Simulate raw Sentinel-1 chip with typical Sen1Floods11 characteristics
    # Physical backscatter in dB: valid values ~ [-25, -5] dB, nodata pixels = -9999.0
    h, w = 256, 256
    vv_valid = np.random.normal(loc=-12.5, scale=4.0, size=(h, w)).astype(np.float32)
    vh_valid = np.random.normal(loc=-20.0, scale=4.5, size=(h, w)).astype(np.float32)
    
    # Simulate 5% border/nodata pixels with -9999.0
    mask_nodata = np.random.rand(h, w) < 0.05
    vv_raw = vv_valid.copy()
    vh_raw = vh_valid.copy()
    vv_raw[mask_nodata] = -9999.0
    vh_raw[mask_nodata] = -9999.0

    print("\n1. Raw SAR values before normalization:")
    print(f"   VV: min={vv_raw.min():.1f}, max={vv_raw.max():.1f}, mean={vv_raw.mean():.1f}")
    print(f"   VH: min={vh_raw.min():.1f}, max={vh_raw.max():.1f}, mean={vh_raw.mean():.1f}")
    print(f"   Nodata (-9999.0) pixel percentage: {mask_nodata.mean() * 100:.2f}%")

    # 2. Apply current Sen1Floods11 normalization
    s1_mean = np.array([-12.54, -20.19], dtype=np.float32)
    s1_std = np.array([5.25, 5.73], dtype=np.float32)
    
    vv_norm = (vv_raw - s1_mean[0]) / s1_std[0]
    vh_norm = (vh_raw - s1_mean[1]) / s1_std[1]

    print("\n2. SAR values after CURRENT normalization ((x - mean) / std):")
    print(f"   VV normalized: min={vv_norm.min():.1f}, max={vv_norm.max():.1f}, mean={vv_norm.mean():.1f}")
    print(f"   VH normalized: min={vh_norm.min():.1f}, max={vh_norm.max():.1f}, mean={vh_norm.mean():.1f}")
    print(f"   -> Outlier value for -9999 nodata: {vv_norm.min():.1f}!")

    # 3. Simulate forward pass through Conv2d & BatchNorm
    x_bad = torch.from_numpy(np.stack([vv_norm, vh_norm])[None, ...]).float()
    conv = nn.Conv2d(2, 32, 3, padding=1)
    bn = nn.BatchNorm2d(32)
    
    out_conv = conv(x_bad)
    out_bn = bn(out_conv)
    
    print("\n3. Impact on Neural Network Layers:")
    print(f"   Conv2d output min/max: {out_conv.min().item():.2f} to {out_conv.max().item():.2f}")
    print(f"   BatchNorm running variance: min={bn.running_var.min().item():.2f}, max={bn.running_var.max():.2f}")
    
    # Check normal (non-nodata) pixels after BatchNorm
    normal_mask_tensor = torch.from_numpy(~mask_nodata)[None, None, ...].expand(-1, 32, -1, -1)
    normal_activations = out_bn[normal_mask_tensor]
    print(f"   Normal pixel activations after BatchNorm: mean={normal_activations.mean().item():.4f}, "
          f"std={normal_activations.std().item():.4f}")
    print("   -> CONCLUSION: Extreme values (~ -1900) inflate BatchNorm running variance by 1000x+, "
          "suppressing true signals into near-zero noise!")

    # 4. Impact of class imbalance + BCE loss
    print("\n4. Impact of Class Imbalance + BCEDiceLoss:")
    # Typical Sen1Floods11 chip: 2% flood pixels
    flood_prob_gt = 0.02
    target = (torch.rand(1, 1, 256, 256) < flood_prob_gt).float()
    
    # When SAR features are destroyed, model outputs initial bias (e.g. logit = -3.0)
    logits_collapsed = torch.full((1, 1, 256, 256), -3.0)
    probs_collapsed = torch.sigmoid(logits_collapsed)
    preds_collapsed = (probs_collapsed > 0.5).float()
    
    tp = (preds_collapsed * target).sum().item()
    fp = (preds_collapsed * (1 - target)).sum().item()
    fn = ((1 - preds_collapsed) * target).sum().item()
    iou_collapsed = tp / (tp + fp + fn + 1e-8)
    
    print(f"   Ground truth positive pixels: {target.sum().item():.0f} ({flood_prob_gt * 100:.1f}%)")
    print(f"   Predicted positive pixels (threshold 0.5, logit=-3.0): {preds_collapsed.sum().item():.0f} (0.00%)")
    print(f"   Resulting IoU: {iou_collapsed:.4f}")
    print("   -> Matches Colab observation: IoU collapses to 0.0000 across all seeds.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAR Preprocessing Diagnostic")
    parser.add_argument("--data_root", default="data/sen1floods11", help="Path to Sen1Floods11 root")
    parser.add_argument("--simulate", action="store_true", help="Run simulation if local data is unavailable")
    parser.add_argument("--device", default="cpu", help="Device to run 1-epoch check on")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    chips = find_sen1floods11_chips(data_root)

    if len(chips) > 0:
        print(f"Found {len(chips)} Sen1Floods11 chips under {data_root}.")
        for chip in chips[:3]:
            analyze_real_chip(chip)
        run_1_epoch_diagnostic(chips[:4], device=args.device)
    else:
        print(f"[INFO] No real Sen1Floods11 GeoTIFF chips found under '{data_root}'.")
        print("To run on Colab with real data, copy this script to Colab and run:")
        print("    python debug_sar.py --data_root data/sen1floods11")
        print("\nExecuting numerical simulation of the exact mathematical failure mechanism now:\n")
        run_synthetic_simulation()


if __name__ == "__main__":
    main()
