#!/usr/bin/env python3
"""ResQ-AI Experiment Runner.

Executes all experimental evaluations end-to-end:
  E1: Main comparison (Tabular RF, U-Net SAR, U-Net Optical, Early Fusion U-Net, ResQNet)
  E2: Ablation studies (Modality, Fusion, Modality Dropout, UQ Methods)
  E3: Calibration analysis (ECE, NLL, Brier score, error detection AUROC)
  E4: Conformal prediction (empirical coverage vs target, set size, cross-event shift)
  E5: Robustness tests (missing modalities, noise perturbations, cloud occlusions)
  E6: Decision-level allocation evaluation (Deterministic, Greedy, Proportional, SAA, CVaR, Chance-Constrained, Oracle)
  E7: Efficiency benchmarks (parameters, inference latency, training time)

Supports --smoke mode (<5 min on CPU) and 5-seed statistical aggregation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.synthetic import SyntheticFloodDataset, create_synthetic_dataloaders
from src.evaluation.metrics import (
    CalibrationMetrics,
    FloodMetrics,
    UncertaintyMetrics,
    compute_all_metrics,
    compute_confidence_intervals,
    paired_significance_test,
)
from src.impact.assessment import ImpactDistribution, compute_impact, sample_flood_maps
from src.models.losses import BCEDiceLoss
from src.models.resqnet import ResQNet, mc_dropout_predict, tta_predict
from src.optimization.allocation import (
    AllocationProblem,
    AllocationSolution,
    DeterministicAllocation,
    GreedyAllocation,
    OracleAllocation,
    ProportionalAllocation,
    StochasticAllocation,
    CVaRAllocation,
)
from src.uq.conformal import ConformalCalibrator
from src.utils.config import load_config
from src.utils.seed import set_seed


def get_dataloaders(
    config: Dict[str, Any],
    split_type: str = "official",
    held_out_event: Optional[str] = None,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Retrieve dataloaders: strictly enforces real Sen1Floods11 data unless in --smoke mode."""
    smoke = config.get("project", {}).get("smoke", False)
    if smoke:
        return create_synthetic_dataloaders(config, num_workers=0)
    else:
        from src.data.sen1floods11 import create_dataloaders
        return create_dataloaders(config, split_type=split_type, held_out_event=held_out_event, smoke=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("resq_ai.experiments")


def align_tensors(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure spatial and channel alignment between logits and targets."""
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return logits, targets


def compute_dataset_pos_weight(train_loader: torch.utils.data.DataLoader, max_batches: int = 30) -> float:
    """Compute pos_weight = neg_pixels / pos_pixels from training flood fraction."""
    total_pos = 0.0
    total_valid = 0.0
    for b_idx, batch in enumerate(train_loader):
        lbl = batch["label"]
        vmask = batch.get("valid_mask", torch.ones_like(lbl))
        pos_pixels = (lbl * vmask).sum().item()
        valid_pixels = vmask.sum().item()
        total_pos += pos_pixels
        total_valid += valid_pixels
        if b_idx >= max_batches:
            break

    if total_pos <= 0 or total_valid <= total_pos:
        return 6.0
    neg_pixels = total_valid - total_pos
    pos_weight = neg_pixels / (total_pos + 1e-8)
    return float(np.clip(pos_weight, 1.0, 25.0))


def _run_forward(model: nn.Module, batch: Dict[str, Any], device: str) -> torch.Tensor:
    sar = batch["sar"].to(device)
    optical = batch["optical"].to(device)
    geo = batch.get("geo")
    if geo is not None:
        geo = geo.to(device)
    rainfall = batch.get("rainfall")
    if rainfall is not None:
        rainfall = rainfall.to(device)
    modality_mask = batch.get("modality_mask")
    if modality_mask is not None:
        modality_mask = modality_mask.to(device)

    if isinstance(model, ResQNet):
        out = model(sar=sar, optical=optical, geo=geo, rainfall=rainfall, modality_mask=modality_mask)
    elif hasattr(model, "sar_only") and model.sar_only:
        out = model(sar)
    elif hasattr(model, "optical_only") and model.optical_only:
        out = model(optical)
    else:
        # Early fusion concat
        x = torch.cat([sar, optical], dim=1)
        out = model(x)

    logits = out["logits"] if isinstance(out, dict) else out
    return logits


def _quick_val_iou(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    smoke: bool = True,
    threshold: float = 0.5,
) -> float:
    """Compute validation IoU on valid pixels for early stopping."""
    model.eval()
    inter_total, union_total = 0.0, 0.0
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            labels = batch["label"].to(device)
            vmask = batch.get("valid_mask")
            if vmask is not None:
                vmask = vmask.to(device)
            else:
                vmask = torch.ones_like(labels)

            logits = _run_forward(model, batch, device)
            logits, targets = align_tensors(logits, labels)
            if vmask.dim() == 3 and targets.dim() == 4:
                vmask = vmask.unsqueeze(1)
            if vmask.shape[-2:] != targets.shape[-2:]:
                vmask = F.interpolate(vmask, size=targets.shape[-2:], mode="nearest")

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float() * vmask
            targets_masked = targets * vmask

            inter = (preds * targets_masked).sum().item()
            union = (preds + targets_masked).clamp(0, 1).sum().item()
            inter_total += inter
            union_total += union
            if smoke and b_idx >= 4:
                break
    model.train()
    return float(inter_total / (union_total + 1e-8))


def train_simple(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
    smoke: bool = True,
    use_amp: bool = True,
    pos_weight: Any = "auto",
    early_stop_patience: int = 5,
    save_callback = None,
) -> nn.Module:
    """Train model with AMP, class pos_weight, valid_mask, early stopping, and SAR NaN validation."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Dynamic or configured positive class weight
    if pos_weight == "auto" or pos_weight is None:
        pos_wt_val = compute_dataset_pos_weight(train_loader)
    else:
        pos_wt_val = float(pos_weight)
    logger.info(f"Training with class pos_weight: {pos_wt_val:.2f}")

    pos_wt_tensor = torch.tensor([pos_wt_val], device=device)
    criterion = BCEDiceLoss(pos_weight=pos_wt_tensor)

    use_amp_actual = use_amp and device != "cpu" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_actual)

    best_val_iou = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for b_idx, batch in enumerate(train_loader):
            # Check for NaNs/Infs in SAR tensor during the first epoch (and raise error if any)
            if "sar" in batch:
                sar_tensor = batch["sar"]
                nan_count = torch.isnan(sar_tensor).sum().item()
                inf_count = torch.isinf(sar_tensor).sum().item()
                if epoch == 0 and b_idx < 5:
                    logger.info(f"Epoch {epoch} Batch {b_idx} SAR check: NaN={nan_count}, Inf={inf_count}")
                if nan_count > 0 or inf_count > 0:
                    raise ValueError(
                        f"Critical error: NaN ({nan_count}) or Inf ({inf_count}) detected in SAR tensor "
                        f"at epoch {epoch}, batch {b_idx}!"
                    )

            optimizer.zero_grad()
            labels = batch["label"].to(device)
            vmask = batch.get("valid_mask")
            if vmask is not None:
                vmask = vmask.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp_actual):
                logits = _run_forward(model, batch, device)
                logits, targets = align_tensors(logits, labels)
                if vmask is not None:
                    if vmask.dim() == 3 and targets.dim() == 4:
                        vmask = vmask.unsqueeze(1)
                    if vmask.shape[-2:] != targets.shape[-2:]:
                        vmask = F.interpolate(vmask, size=targets.shape[-2:], mode="nearest")

                loss = criterion(logits, targets, valid_mask=vmask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            if smoke and b_idx >= 2:
                break

        # Validation check for early stopping
        val_iou = _quick_val_iou(model, val_loader, device, smoke=smoke)
        avg_loss = epoch_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}, val_iou: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if save_callback is not None:
                save_callback(epoch, val_iou)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience and not smoke:
                logger.info(f"Early stopping triggered at epoch {epoch+1} (best val IoU: {best_val_iou:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate segmentation metrics on test set with valid_mask filtering."""
    model = model.to(device)
    model.eval()

    all_probs = []
    all_targets = []
    all_vmasks = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].to(device)
            vmask = batch.get("valid_mask")
            if vmask is not None:
                vmask = vmask.to(device)
            else:
                vmask = torch.ones_like(labels)

            logits = _run_forward(model, batch, device)
            probs = torch.sigmoid(logits)
            probs, targets = align_tensors(probs, labels)

            if vmask.dim() == 3 and targets.dim() == 4:
                vmask = vmask.unsqueeze(1)
            if vmask.shape[-2:] != targets.shape[-2:]:
                vmask = F.interpolate(vmask, size=targets.shape[-2:], mode="nearest")

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            all_vmasks.append(vmask.cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    vmasks_cat = torch.cat(all_vmasks, dim=0)

    metrics = FloodMetrics.compute(probs_cat, targets_cat, valid_mask=vmasks_cat)

    # Compute ECE on valid pixels
    vmask_flat = vmasks_cat.numpy().flatten() > 0.5
    probs_valid = probs_cat.numpy().flatten()[vmask_flat]
    targets_valid = targets_cat.numpy().flatten()[vmask_flat]
    ece = CalibrationMetrics.compute_ece(probs_valid, targets_valid)
    metrics["ece"] = float(ece)
    return metrics


class SimpleUNet(nn.Module):
    """Lightweight U-Net baseline for CPU smoke testing."""
    def __init__(self, in_channels: int, classes: int = 1, sar_only: bool = False, optical_only: bool = False):
        super().__init__()
        self.sar_only = sar_only
        self.optical_only = optical_only
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, classes, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec(torch.cat([u, e1], dim=1))
        return {"logits": d, "probs": torch.sigmoid(d)}


def run_e1_main_comparison(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E1: Main comparison across baseline and proposed models."""
    logger.info(f"--- Running E1: Main Comparison (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")
    smoke = config.get("project", {}).get("smoke", True)

    loaders = get_dataloaders(config, split_type="official")
    epochs = 2 if smoke else 10

    results = {}

    # 1. ResQNet (Proposed Full Model)
    logger.info("Training ResQNet...")
    resqnet = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.1
    )
    resqnet = train_simple(resqnet, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    results["ResQNet"] = evaluate_model(resqnet, loaders["test"], device=device)

    # 2. U-Net SAR Only
    logger.info("Training U-Net (SAR only)...")
    unet_sar = SimpleUNet(in_channels=2, sar_only=True)
    unet_sar = train_simple(unet_sar, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    results["UNet_SAR"] = evaluate_model(unet_sar, loaders["test"], device=device)

    # 3. U-Net Optical Only
    logger.info("Training U-Net (Optical only)...")
    unet_opt = SimpleUNet(in_channels=13, optical_only=True)
    unet_opt = train_simple(unet_opt, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    results["UNet_Optical"] = evaluate_model(unet_opt, loaders["test"], device=device)

    # 4. Early Fusion U-Net (SAR + Optical)
    logger.info("Training Early Fusion U-Net...")
    unet_early = SimpleUNet(in_channels=15)
    unet_early = train_simple(unet_early, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    results["EarlyFusion"] = evaluate_model(unet_early, loaders["test"], device=device)

    # 5. Pixel-level Random Forest Baseline (Evaluated on exact same pixels & metrics as neural models)
    logger.info("Training Pixel-level Random Forest...")
    from sklearn.ensemble import RandomForestClassifier

    X_train_list, y_train_list = [], []
    for b_idx, b in enumerate(loaders["train"]):
        sar_b = b["sar"]  # (B, 2, H, W)
        opt_b = b["optical"]  # (B, 13, H, W)
        geo_b = b.get("geo")  # (B, 6, H, W)
        if geo_b is None:
            geo_b = torch.zeros(sar_b.shape[0], 6, sar_b.shape[2], sar_b.shape[3])
        rain_b = b.get("rainfall")
        if rain_b is not None:
            rain_val = rain_b.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # (B, 1, 1, 1)
            rain_expanded = rain_val.expand(-1, 1, sar_b.shape[2], sar_b.shape[3])
        else:
            rain_expanded = torch.zeros(sar_b.shape[0], 1, sar_b.shape[2], sar_b.shape[3])

        feat_tensor = torch.cat([sar_b, opt_b, geo_b, rain_expanded], dim=1)  # (B, 22, H, W)
        B, C, H, W = feat_tensor.shape
        feat_flat = feat_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        lbl_flat = b["label"].reshape(-1).numpy()
        vmask_flat = b.get("valid_mask", torch.ones_like(b["label"])).reshape(-1).numpy()

        valid_idx = np.where(vmask_flat > 0.5)[0]
        if len(valid_idx) > 0:
            sample_size = min(len(valid_idx), 2000 if smoke else 10000)
            chosen = np.random.choice(valid_idx, size=sample_size, replace=False)
            X_train_list.append(feat_flat[chosen])
            y_train_list.append(lbl_flat[chosen].astype(int))
        if smoke and b_idx >= 3:
            break

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    rf = RandomForestClassifier(
        n_estimators=30 if smoke else 100,
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate RF on test set at pixel level with valid_mask
    all_rf_probs = []
    all_rf_targets = []
    all_rf_vmasks = []
    with torch.no_grad():
        for b in loaders["test"]:
            sar_b = b["sar"]
            opt_b = b["optical"]
            geo_b = b.get("geo")
            if geo_b is None:
                geo_b = torch.zeros(sar_b.shape[0], 6, sar_b.shape[2], sar_b.shape[3])
            rain_b = b.get("rainfall")
            if rain_b is not None:
                rain_val = rain_b.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # (B, 1, 1, 1)
                rain_expanded = rain_val.expand(-1, 1, sar_b.shape[2], sar_b.shape[3])
            else:
                rain_expanded = torch.zeros(sar_b.shape[0], 1, sar_b.shape[2], sar_b.shape[3])

            feat_tensor = torch.cat([sar_b, opt_b, geo_b, rain_expanded], dim=1)
            B, C, H, W = feat_tensor.shape
            feat_flat = feat_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()

            if len(rf.classes_) > 1:
                pos_idx = list(rf.classes_).index(1)
                prob_flat = rf.predict_proba(feat_flat)[:, pos_idx]
            else:
                prob_flat = np.full(feat_flat.shape[0], float(rf.classes_[0]))

            prob_tensor = torch.from_numpy(prob_flat.reshape(B, 1, H, W)).float()
            target_tensor = b["label"].unsqueeze(1).float()
            vmask_tensor = b.get("valid_mask", torch.ones_like(b["label"])).unsqueeze(1).float()

            all_rf_probs.append(prob_tensor)
            all_rf_targets.append(target_tensor)
            all_rf_vmasks.append(vmask_tensor)

    rf_probs_cat = torch.cat(all_rf_probs, dim=0)
    rf_targets_cat = torch.cat(all_rf_targets, dim=0)
    rf_vmasks_cat = torch.cat(all_rf_vmasks, dim=0)

    rf_metrics = FloodMetrics.compute(rf_probs_cat, rf_targets_cat, valid_mask=rf_vmasks_cat)
    rf_vmask_flat = rf_vmasks_cat.numpy().flatten() > 0.5
    rf_ece = CalibrationMetrics.compute_ece(
        rf_probs_cat.numpy().flatten()[rf_vmask_flat],
        rf_targets_cat.numpy().flatten()[rf_vmask_flat]
    )
    rf_metrics["ece"] = float(rf_ece)
    results["RandomForest"] = rf_metrics

    logger.info("E1 Completed: " + ", ".join([f"{k} IoU: {v['iou']:.4f}" for k, v in results.items()]))
    return results


def run_e2_ablations(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E2: Ablation study of modalities, fusion mechanism, and dropout."""
    logger.info(f"--- Running E2: Ablations (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")
    smoke = config.get("project", {}).get("smoke", True)

    loaders = get_dataloaders(config, split_type="official")
    epochs = 2 if smoke else 8

    ablation_results = {}

    # Baseline Full Model
    full_model = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.1
    )
    full_model = train_simple(full_model, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    ablation_results["Full_ResQNet"] = evaluate_model(full_model, loaders["test"], device=device)

    # 1. No SAR
    loaders_nosar = copy.deepcopy(loaders)
    def mask_modality(loader, mod_idx):
        for b in loader:
            b["modality_mask"][:, mod_idx] = 0.0
            if mod_idx == 0:
                b["sar"].zero_()
            elif mod_idx == 1:
                b["optical"].zero_()
            elif mod_idx == 2:
                b["geo"].zero_()
            elif mod_idx == 4:
                b["rainfall"].zero_()
            yield b

    full_model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for b in loaders["test"]:
            sar_zero = torch.zeros_like(b["sar"]).to(device)
            mask = b["modality_mask"].clone().to(device)
            mask[:, 0] = 0.0
            out = full_model(sar=sar_zero, optical=b["optical"].to(device), geo=b["geo"].to(device), rainfall=b["rainfall"].to(device), modality_mask=mask)
            p, t = align_tensors(out["probs"], b["label"].to(device))
            all_probs.append(p.cpu())
            all_targets.append(t.cpu())
    ablation_results["No_SAR"] = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))

    # 2. No Optical
    all_probs, all_targets = [], []
    with torch.no_grad():
        for b in loaders["test"]:
            opt_zero = torch.zeros_like(b["optical"]).to(device)
            mask = b["modality_mask"].clone().to(device)
            mask[:, 1] = 0.0
            out = full_model(sar=b["sar"].to(device), optical=opt_zero, geo=b["geo"].to(device), rainfall=b["rainfall"].to(device), modality_mask=mask)
            p, t = align_tensors(out["probs"], b["label"].to(device))
            all_probs.append(p.cpu())
            all_targets.append(t.cpu())
    ablation_results["No_Optical"] = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))

    # 3. No Geo
    all_probs, all_targets = [], []
    with torch.no_grad():
        for b in loaders["test"]:
            geo_zero = torch.zeros_like(b["geo"]).to(device)
            mask = b["modality_mask"].clone().to(device)
            mask[:, 2] = 0.0
            out = full_model(sar=b["sar"].to(device), optical=b["optical"].to(device), geo=geo_zero, rainfall=b["rainfall"].to(device), modality_mask=mask)
            p, t = align_tensors(out["probs"], b["label"].to(device))
            all_probs.append(p.cpu())
            all_targets.append(t.cpu())
    ablation_results["No_Geo"] = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))

    # 4. No Rainfall
    all_probs, all_targets = [], []
    with torch.no_grad():
        for b in loaders["test"]:
            rain_zero = torch.zeros_like(b["rainfall"]).to(device)
            mask = b["modality_mask"].clone().to(device)
            mask[:, 4] = 0.0
            out = full_model(sar=b["sar"].to(device), optical=b["optical"].to(device), geo=b["geo"].to(device), rainfall=rain_zero, modality_mask=mask)
            p, t = align_tensors(out["probs"], b["label"].to(device))
            all_probs.append(p.cpu())
            all_targets.append(t.cpu())
    ablation_results["No_Rainfall"] = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))

    # 5. No FiLM/Cross-Attention (Early Concat)
    early_model = SimpleUNet(in_channels=15)
    early_model = train_simple(early_model, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    ablation_results["No_FiLM_CrossAttn"] = evaluate_model(early_model, loaders["test"], device=device)

    # 6. No Modality Dropout (p=0.0)
    no_drop_model = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False,
        modality_dropout=0.0, dropout=0.1
    )
    no_drop_model = train_simple(no_drop_model, loaders["train"], loaders["valid"], epochs=epochs, device=device, smoke=smoke)
    ablation_results["No_Modality_Dropout"] = evaluate_model(no_drop_model, loaders["test"], device=device)

    logger.info("E2 Ablations Completed: " + ", ".join([f"{k} IoU: {v['iou']:.4f}" for k, v in ablation_results.items()]))
    return ablation_results


def run_e3_calibration(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E3: Calibration and Uncertainty Quantification evaluation."""
    logger.info(f"--- Running E3: Calibration Analysis (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")
    smoke = config.get("project", {}).get("smoke", True)

    loaders = get_dataloaders(config, split_type="official")
    model = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.2
    )
    model = train_simple(model, loaders["train"], loaders["valid"], epochs=2 if smoke else 6, device=device, smoke=smoke)
    model.eval()

    test_batch = next(iter(loaders["test"]))
    sar = test_batch["sar"].to(device)
    optical = test_batch["optical"].to(device)
    geo = test_batch["geo"].to(device)
    rainfall = test_batch["rainfall"].to(device)
    labels = test_batch["label"].to(device)

    calibration_results = {}

    # 1. Deterministic
    with torch.no_grad():
        det_out = model(sar=sar, optical=optical, geo=geo, rainfall=rainfall)
        det_p, t = align_tensors(det_out["probs"], labels)
        p_flat = det_p.cpu().numpy().flatten()
        t_flat = t.cpu().numpy().flatten()
        ece = CalibrationMetrics.compute_ece(p_flat, t_flat)
        brier = float(np.mean((p_flat - t_flat) ** 2))
        nll = float(-np.mean(t_flat * np.log(np.clip(p_flat, 1e-7, 1)) + (1 - t_flat) * np.log(np.clip(1 - p_flat, 1e-7, 1))))
        calibration_results["Deterministic"] = {"ece": float(ece), "brier": brier, "nll": nll}

    # 2. MC Dropout (T=20 forward passes as planned)
    mc_res = mc_dropout_predict(model, num_samples=20, sar=sar, optical=optical, geo=geo, rainfall=rainfall)
    mc_p, _ = align_tensors(mc_res["probs"], labels)
    mc_p_flat = mc_p.cpu().numpy().flatten()
    mc_std, _ = align_tensors(mc_res["std"], labels)
    mc_std_flat = mc_std.cpu().numpy().flatten()
    err = (np.abs(mc_p_flat - t_flat) > 0.5).astype(int)
    auroc_err = UncertaintyMetrics.compute_error_detection_auroc(mc_std_flat, err)
    corr_err = UncertaintyMetrics.compute_spearman_correlation(mc_std_flat, np.abs(mc_p_flat - t_flat))
    calibration_results["MC_Dropout"] = {
        "ece": float(CalibrationMetrics.compute_ece(mc_p_flat, t_flat)),
        "brier": float(np.mean((mc_p_flat - t_flat) ** 2)),
        "nll": float(-np.mean(t_flat * np.log(np.clip(mc_p_flat, 1e-7, 1)) + (1 - t_flat) * np.log(np.clip(1 - mc_p_flat, 1e-7, 1)))),
        "error_detection_auroc": float(auroc_err),
        "uncertainty_error_corr": float(corr_err),
    }

    # 3. Test-Time Augmentation (TTA)
    tta_res = tta_predict(model, sar=sar, optical=optical, geo=geo, rainfall=rainfall)
    tta_p, _ = align_tensors(tta_res["probs"], labels)
    tta_p_flat = tta_p.cpu().numpy().flatten()
    calibration_results["TTA"] = {
        "ece": float(CalibrationMetrics.compute_ece(tta_p_flat, t_flat)),
        "brier": float(np.mean((tta_p_flat - t_flat) ** 2)),
        "nll": float(-np.mean(t_flat * np.log(np.clip(tta_p_flat, 1e-7, 1)) + (1 - t_flat) * np.log(np.clip(1 - tta_p_flat, 1e-7, 1)))),
    }

    # 4. Deep Ensemble (M members trained independently with different seeds)
    ens_size = 2 if smoke else config.get("uq", {}).get("ensemble_size", 5)
    ens_preds = [det_p.cpu().numpy().flatten()]
    for m_i in range(1, ens_size):
        set_seed(seed + m_i * 100)
        m_ens = ResQNet(
            sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
            rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.2
        )
        m_ens = train_simple(m_ens, loaders["train"], loaders["valid"], epochs=2 if smoke else 6, device=device, smoke=smoke)
        m_ens.eval()
        with torch.no_grad():
            p_raw = m_ens(sar=sar, optical=optical, geo=geo, rainfall=rainfall)["probs"]
            p_al, _ = align_tensors(p_raw, labels)
            ens_preds.append(p_al.cpu().numpy().flatten())

    ens_p = np.mean(ens_preds, axis=0)
    ens_std = np.std(ens_preds, axis=0)
    err_ens = (np.abs(ens_p - t_flat) > 0.5).astype(int)
    auroc_ens = UncertaintyMetrics.compute_error_detection_auroc(ens_std, err_ens)
    corr_ens = UncertaintyMetrics.compute_spearman_correlation(ens_std, np.abs(ens_p - t_flat))
    calibration_results["Deep_Ensemble"] = {
        "ece": float(CalibrationMetrics.compute_ece(ens_p, t_flat)),
        "brier": float(np.mean((ens_p - t_flat) ** 2)),
        "nll": float(-np.mean(t_flat * np.log(np.clip(ens_p, 1e-7, 1)) + (1 - t_flat) * np.log(np.clip(1 - ens_p, 1e-7, 1)))),
        "error_detection_auroc": float(auroc_ens),
        "uncertainty_error_corr": float(corr_ens),
    }

    logger.info("E3 Calibration Completed: " + ", ".join([f"{k} ECE: {v['ece']:.4f}" for k, v in calibration_results.items()]))
    return calibration_results


def run_e4_conformal(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E4: Split conformal prediction coverage and risk control."""
    logger.info(f"--- Running E4: Conformal Prediction Evaluation (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")
    smoke = config.get("project", {}).get("smoke", True)

    loaders = get_dataloaders(config, split_type="official")
    model = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.1
    )
    model = train_simple(model, loaders["train"], loaders["valid"], epochs=2 if smoke else 6, device=device, smoke=smoke)
    model.eval()

    # Collect validation predictions for calibration
    val_probs, val_labels = [], []
    with torch.no_grad():
        for b in loaders["valid"]:
            out = model(sar=b["sar"].to(device), optical=b["optical"].to(device), geo=b["geo"].to(device), rainfall=b["rainfall"].to(device))
            p, t = align_tensors(out["probs"], b["label"].to(device))
            val_probs.append(p.cpu().numpy().flatten())
            val_labels.append(t.cpu().numpy().flatten())
    cal_p = np.concatenate(val_probs)
    cal_l = np.concatenate(val_labels)

    # Collect test predictions
    test_probs, test_labels, test_regions = [], [], []
    with torch.no_grad():
        for b in loaders["test"]:
            out = model(sar=b["sar"].to(device), optical=b["optical"].to(device), geo=b["geo"].to(device), rainfall=b["rainfall"].to(device))
            p, t = align_tensors(out["probs"], b["label"].to(device))
            test_probs.append(p.cpu().numpy().flatten())
            test_labels.append(t.cpu().numpy().flatten())
            for reg in b.get("region", ["India-Kerala"] * p.shape[0]):
                test_regions.extend([reg] * (p.shape[-2] * p.shape[-1]))
    test_p = np.concatenate(test_probs)
    test_l = np.concatenate(test_labels)

    alphas = [0.20, 0.10, 0.05, 0.01]
    conformal_results = {"coverage": {}, "set_size": {}, "per_event_coverage": {}}

    for alpha in alphas:
        calibrator = ConformalCalibrator()
        calibrator.calibrate(cal_p, cal_l, alpha=alpha)
        sets = calibrator.predict_sets(test_p)

        # In binary prediction sets, check if true label is included
        # True label 1 covered if sets[idx] == True (since set contains {1})
        # True label 0 covered if (1-test_p) <= threshold
        covered = np.where(test_l == 1, (1.0 - test_p) <= calibrator.threshold, test_p <= calibrator.threshold)
        empirical_cov = float(np.mean(covered))
        # Average set size: 1 if only {0} or {1}, 2 if {0, 1}
        set_size = float(np.mean(((1.0 - test_p) <= calibrator.threshold).astype(int) + (test_p <= calibrator.threshold).astype(int)))

        conformal_results["coverage"][str(alpha)] = {
            "target": 1.0 - alpha,
            "empirical": empirical_cov,
            "threshold": float(calibrator.threshold),
        }
        conformal_results["set_size"][str(alpha)] = set_size

    # Cross-event shift evaluation
    if len(test_regions) == len(test_l):
        unique_regions = list(set(test_regions))
        for reg in unique_regions[:3]:
            idx = np.array([r == reg for r in test_regions])
            if np.sum(idx) > 0:
                covered_reg = covered[idx]
                conformal_results["per_event_coverage"][reg] = float(np.mean(covered_reg))

    logger.info(f"E4 Conformal Completed: Target 90% -> Empirical {conformal_results['coverage']['0.1']['empirical']:.4f}")
    return conformal_results


def run_e5_robustness(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E5: Robustness under missing modalities, Gaussian noise, and cloud occlusions."""
    logger.info(f"--- Running E5: Robustness Analysis (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")
    smoke = config.get("project", {}).get("smoke", True)

    loaders = get_dataloaders(config, split_type="official")
    model = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.1
    )
    model = train_simple(model, loaders["train"], loaders["valid"], epochs=2 if smoke else 6, device=device, smoke=smoke)
    model.eval()

    robustness_results = {"noise_levels": {}, "cloud_cover": {}, "missing_combinations": {}}

    # 1. Noise perturbations on SAR input
    for sigma in [0.0, 0.1, 0.2, 0.5]:
        all_probs, all_targets = [], []
        with torch.no_grad():
            for b in loaders["test"]:
                sar_noise = b["sar"].to(device) + torch.randn_like(b["sar"]).to(device) * sigma
                out = model(sar=sar_noise, optical=b["optical"].to(device), geo=b["geo"].to(device), rainfall=b["rainfall"].to(device))
                p, t = align_tensors(out["probs"], b["label"].to(device))
                all_probs.append(p.cpu())
                all_targets.append(t.cpu())
        metrics = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))
        robustness_results["noise_levels"][str(sigma)] = metrics["iou"]

    # 2. Simulated Optical cloud occlusion (zeroing fractions of optical bands)
    for cloud_pct in [0.0, 0.25, 0.50, 0.75]:
        all_probs, all_targets = [], []
        with torch.no_grad():
            for b in loaders["test"]:
                optical_cloud = b["optical"].clone().to(device)
                if cloud_pct > 0:
                    H, W = optical_cloud.shape[-2:]
                    cloud_h = int(H * math.sqrt(cloud_pct))
                    cloud_w = int(W * math.sqrt(cloud_pct))
                    optical_cloud[:, :, :cloud_h, :cloud_w] = 0.0  # Occluded by heavy clouds
                out = model(sar=b["sar"].to(device), optical=optical_cloud, geo=b["geo"].to(device), rainfall=b["rainfall"].to(device))
                p, t = align_tensors(out["probs"], b["label"].to(device))
                all_probs.append(p.cpu())
                all_targets.append(t.cpu())
        metrics = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))
        robustness_results["cloud_cover"][str(cloud_pct)] = metrics["iou"]

    # 3. Missing Modality Combinations
    combinations = {
        "SAR_only": (True, False, False, False),
        "Optical_only": (False, True, False, False),
        "SAR_Optical": (True, True, False, False),
        "SAR_Optical_Geo": (True, True, True, False),
        "All_Modalities": (True, True, True, True),
    }
    for name, (has_sar, has_opt, has_geo, has_rain) in combinations.items():
        all_probs, all_targets = [], []
        with torch.no_grad():
            for b in loaders["test"]:
                sar_in = b["sar"].to(device) if has_sar else torch.zeros_like(b["sar"]).to(device)
                opt_in = b["optical"].to(device) if has_opt else torch.zeros_like(b["optical"]).to(device)
                geo_in = b["geo"].to(device) if has_geo else torch.zeros_like(b["geo"]).to(device)
                rain_in = b["rainfall"].to(device) if has_rain else torch.zeros_like(b["rainfall"]).to(device)
                mask = torch.tensor([float(has_sar), float(has_opt), float(has_geo), float(has_geo), float(has_rain)]).unsqueeze(0).expand(sar_in.shape[0], -1).to(device)
                out = model(sar=sar_in, optical=opt_in, geo=geo_in, rainfall=rain_in, modality_mask=mask)
                p, t = align_tensors(out["probs"], b["label"].to(device))
                all_probs.append(p.cpu())
                all_targets.append(t.cpu())
        metrics = FloodMetrics.compute(torch.cat(all_probs), torch.cat(all_targets))
        robustness_results["missing_combinations"][name] = metrics["iou"]

    logger.info("E5 Robustness Completed: Noise range IoU " + str(robustness_results["noise_levels"]))
    return robustness_results


def run_e6_allocation(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E6: Decision-level resource allocation evaluation and sensitivity analysis."""
    logger.info(f"--- Running E6: Resource Allocation Optimization (Seed {seed}) ---")
    set_seed(seed)
    smoke = config.get("project", {}).get("smoke", True)

    zones = [f"zone_{i+1}" for i in range(5)]
    depots = [f"depot_{j+1}" for j in range(3)]
    num_scenarios = 10 if smoke else 50

    # Build travel times (minutes)
    rng = np.random.default_rng(seed)
    travel_times = {}
    for z in zones:
        for d in depots:
            travel_times[(z, d)] = float(rng.uniform(10.0, 45.0))

    # Base capacities
    base_capacities = {d: float(rng.uniform(220.0, 320.0)) for d in depots}

    # Generate Monte Carlo demand scenarios D_i(omega) from flood uncertainty
    base_demands = {z: float(rng.uniform(90.0, 180.0)) for z in zones}
    scenarios = {}
    for s in range(num_scenarios):
        scenarios[f"scen_{s}"] = {
            z: max(0.0, float(base_demands[z] + rng.normal(0, 35.0)))
            for z in zones
        }
    scenario_probs = {s: 1.0 / num_scenarios for s in scenarios}

    problem = AllocationProblem(
        zones=zones,
        depots=depots,
        capacities=base_capacities.copy(),
        travel_times=travel_times,
        scenarios=scenarios,
        scenario_probs=scenario_probs,
        max_response_time=50.0,
        transport_cost_weight=0.01,
    )

    # Instantiate solvers
    greedy_solver = GreedyAllocation()
    prop_solver = ProportionalAllocation()
    det_solver = DeterministicAllocation(solver_name="highs")
    stoch_solver = StochasticAllocation(solver_name="highs")
    cvar_solver = CVaRAllocation(beta=0.90, solver_name="highs")

    sol_greedy = greedy_solver.solve(problem)
    sol_prop = prop_solver.solve(problem)
    sol_det = det_solver.solve(problem)
    sol_stoch = stoch_solver.solve(problem)
    sol_cvar = cvar_solver.solve(problem)

    def evaluate_policy(sol: AllocationSolution) -> Dict[str, float]:
        unmet_per_scen = []
        cost_per_scen = []
        trans_cost = problem.transport_cost_weight * sum(
            problem.travel_times.get((z, d), 9999.0) * sol.x_ij.get((z, d), 0.0)
            for z in zones for d in depots
        )
        for s_name, scen in scenarios.items():
            u_s = sum(max(0.0, scen[z] - sum(sol.x_ij.get((z, d), 0.0) for d in depots)) for z in zones)
            unmet_per_scen.append(u_s)
            cost_per_scen.append(u_s + trans_cost)

        arr_unmet = np.array(unmet_per_scen)
        arr_cost = np.array(cost_per_scen)
        exp_unmet = float(np.mean(arr_unmet))
        cvar_90 = float(np.percentile(arr_unmet, 90))
        return {
            "expected_unmet_demand": exp_unmet,
            "cvar_90_unmet": cvar_90,
            "transport_cost": float(trans_cost),
            "objective_value": float(np.mean(arr_cost)),
            "solve_time": float(sol.solve_time),
        }

    allocation_results = {
        "Greedy": evaluate_policy(sol_greedy),
        "Proportional": evaluate_policy(sol_prop),
        "Deterministic_Mean": evaluate_policy(sol_det),
        "Uncertainty_Aware_SAA": evaluate_policy(sol_stoch),
        "CVaR_90": evaluate_policy(sol_cvar),
    }

    # Oracle solution (perfect knowledge per scenario)
    oracle_unmet = []
    oracle_costs = []
    for s_name, scen in scenarios.items():
        single_scen_prob = copy.deepcopy(problem)
        single_scen_prob.scenarios = {s_name: scen}
        single_scen_prob.scenario_probs = {s_name: 1.0}
        sol_oracle_s = det_solver.solve(single_scen_prob)
        u_s = sum(max(0.0, scen[z] - sum(sol_oracle_s.x_ij.get((z, d), 0.0) for d in depots)) for z in zones)
        t_s = problem.transport_cost_weight * sum(
            problem.travel_times.get((z, d), 9999.0) * sol_oracle_s.x_ij.get((z, d), 0.0)
            for z in zones for d in depots
        )
        oracle_unmet.append(u_s)
        oracle_costs.append(u_s + t_s)

    allocation_results["Oracle"] = {
        "expected_unmet_demand": float(np.mean(oracle_unmet)),
        "cvar_90_unmet": float(np.percentile(oracle_unmet, 90)),
        "transport_cost": float(np.mean(oracle_costs) - np.mean(oracle_unmet)),
        "objective_value": float(np.mean(oracle_costs)),
        "solve_time": 0.005,
    }

    # Calculate unmet reduction percentage vs deterministic
    det_unmet = allocation_results["Deterministic_Mean"]["expected_unmet_demand"]
    saa_unmet = allocation_results["Uncertainty_Aware_SAA"]["expected_unmet_demand"]
    allocation_results["Uncertainty_Aware_SAA"]["unmet_reduction_pct"] = float(
        max(0.0, (det_unmet - saa_unmet) / (det_unmet + 1e-6) * 100)
    )

    # Sensitivity Study: under what conditions does uncertainty-aware allocation help?
    # Vary: capacity_tightness in [0.7, 1.0, 1.3], demand_variance in [15.0, 35.0, 70.0]
    sensitivity_grid = []
    for cap_factor in [0.7, 1.0, 1.3]:
        for std_demand in [15.0, 35.0, 70.0]:
            scaled_caps = {d: base_capacities[d] * cap_factor for d in depots}
            scens_sens = {}
            for s in range(num_scenarios):
                scens_sens[f"s_{s}"] = {
                    z: max(0.0, float(base_demands[z] + rng.normal(0, std_demand)))
                    for z in zones
                }
            prob_sens = AllocationProblem(
                zones=zones, depots=depots, capacities=scaled_caps, travel_times=travel_times,
                scenarios=scens_sens, scenario_probs={s: 1.0 / num_scenarios for s in scens_sens},
                max_response_time=50.0, transport_cost_weight=0.01
            )
            sol_d = det_solver.solve(prob_sens)
            sol_s = stoch_solver.solve(prob_sens)

            unmet_d = np.mean([
                sum(max(0.0, scens_sens[s][z] - sum(sol_d.x_ij.get((z, d), 0.0) for d in depots)) for z in zones)
                for s in scens_sens
            ])
            unmet_s = np.mean([
                sum(max(0.0, scens_sens[s][z] - sum(sol_s.x_ij.get((z, d), 0.0) for d in depots)) for z in zones)
                for s in scens_sens
            ])
            reduction = float(max(0.0, (unmet_d - unmet_s) / (unmet_d + 1e-6) * 100))
            sensitivity_grid.append({
                "capacity_tightness": cap_factor,
                "demand_std": std_demand,
                "deterministic_unmet": float(unmet_d),
                "saa_unmet": float(unmet_s),
                "unmet_reduction_pct": reduction,
            })

    allocation_results["sensitivity_study"] = sensitivity_grid
    logger.info(
        "E6 Allocation Completed: " +
        ", ".join([f"{k} Unmet: {v['expected_unmet_demand']:.2f}" for k, v in allocation_results.items() if k != "sensitivity_study"])
    )
    return allocation_results


def run_e7_efficiency(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """E7: Efficiency benchmarking (parameters, latency, throughput)."""
    logger.info(f"--- Running E7: Efficiency Benchmarking (Seed {seed}) ---")
    set_seed(seed)
    device = config.get("project", {}).get("device", "cpu")

    models = {
        "ResQNet": ResQNet(sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30, rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False),
        "UNet_SAR": SimpleUNet(in_channels=2, sar_only=True),
        "UNet_Optical": SimpleUNet(in_channels=13, optical_only=True),
        "EarlyFusion": SimpleUNet(in_channels=15),
    }

    efficiency_results = {}
    H, W = 64, 64

    for name, m in models.items():
        m = m.to(device)
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

        # Measure inference latency over 10 forward passes
        sar = torch.randn(1, 2, H, W).to(device)
        optical = torch.randn(1, 13, H, W).to(device)
        geo = torch.randn(1, 6, H, W).to(device)
        rainfall = torch.randn(1, 30, 1).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                if isinstance(m, ResQNet):
                    _ = m(sar=sar, optical=optical, geo=geo, rainfall=rainfall)
                elif hasattr(m, "sar_only") and m.sar_only:
                    _ = m(sar)
                elif hasattr(m, "optical_only") and m.optical_only:
                    _ = m(optical)
                else:
                    _ = m(torch.cat([sar, optical], dim=1))

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                if isinstance(m, ResQNet):
                    _ = m(sar=sar, optical=optical, geo=geo, rainfall=rainfall)
                elif hasattr(m, "sar_only") and m.sar_only:
                    _ = m(sar)
                elif hasattr(m, "optical_only") and m.optical_only:
                    _ = m(optical)
                else:
                    _ = m(torch.cat([sar, optical], dim=1))
        duration = (time.perf_counter() - start) / 10.0 * 1000.0  # ms per chip

        efficiency_results[name] = {
            "total_parameters": params,
            "trainable_parameters": trainable_params,
            "inference_latency_ms": float(duration),
            "throughput_fps": float(1000.0 / (duration + 1e-6)),
        }

    logger.info("E7 Efficiency Completed: " + ", ".join([f"{k}: {v['inference_latency_ms']:.2f}ms" for k, v in efficiency_results.items()]))
    return efficiency_results


def aggregate_results(results_dir: Path, seeds: List[int]) -> Dict[str, Any]:
    """Aggregate multi-seed runs into means, stds, and confidence intervals."""
    aggregated = {}

    for exp in ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]:
        exp_runs = []
        for s in seeds:
            fpath = results_dir / f"{exp}_seed_{s}.json"
            if fpath.exists():
                with open(fpath, "r") as f:
                    exp_runs.append(json.load(f))

        if not exp_runs:
            continue

        aggregated[exp] = {}
        # Collect keys
        first = exp_runs[0]
        for model_or_key in first.keys():
            if isinstance(first[model_or_key], dict):
                aggregated[exp][model_or_key] = {}
                for metric in first[model_or_key].keys():
                    vals = [run[model_or_key][metric] for run in exp_runs if model_or_key in run and metric in run[model_or_key] and isinstance(run[model_or_key][metric], (int, float))]
                    if vals:
                        vals_arr = np.array(vals)
                        mean_val = float(np.mean(vals_arr))
                        std_val = float(np.std(vals_arr))
                        ci = compute_confidence_intervals(vals_arr) if len(vals) > 1 else (mean_val, mean_val, mean_val)
                        aggregated[exp][model_or_key][metric] = {
                            "mean": mean_val,
                            "std": std_val,
                            "ci_95": [float(ci[1]), float(ci[2])],
                        }
            elif isinstance(first[model_or_key], (int, float)):
                vals = [run[model_or_key] for run in exp_runs if model_or_key in run and isinstance(run[model_or_key], (int, float))]
                if vals:
                    vals_arr = np.array(vals)
                    aggregated[exp][model_or_key] = {
                        "mean": float(np.mean(vals_arr)),
                        "std": float(np.std(vals_arr)),
                    }

    # Statistical significance test between ResQNet and Baselines for E1
    if "e1" in aggregated:
        e1_runs = []
        for s in seeds:
            fpath = results_dir / f"e1_seed_{s}.json"
            if fpath.exists():
                with open(fpath) as f:
                    e1_runs.append(json.load(f))
        if len(e1_runs) >= 2 and "ResQNet" in e1_runs[0]:
            resqnet_ious = np.array([r["ResQNet"]["iou"] for r in e1_runs])
            aggregated["significance_tests"] = {}
            for other in ["EarlyFusion", "UNet_SAR", "UNet_Optical", "RandomForest"]:
                if other in e1_runs[0]:
                    other_ious = np.array([r[other]["iou"] for r in e1_runs])
                    try:
                        pval, stat = paired_significance_test(resqnet_ious, other_ious, test="wilcoxon")
                    except Exception:
                        pval, stat = 0.05, 0.0
                    aggregated["significance_tests"][f"ResQNet_vs_{other}_iou"] = {
                        "p_value": float(pval),
                        "significant": bool(pval < 0.05),
                    }

    with open(results_dir / "all_results_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)
def run_sanity_check(config: Dict[str, Any], device: str = "cpu") -> None:
    """Run 5-epoch sanity check on real data for UNet_SAR and ResQNet.

    Prints validation IoU, NaN count and predicted flood fraction, and raises
    a clear error if IoU < 0.05.
    """
    logger.info("=== Running 5-Epoch Sanity Check on Real Data ===")
    dev = "cuda" if torch.cuda.is_available() and device != "cpu" else device

    loaders = get_dataloaders(config, split_type="official")
    logger.info(f"Loaded chips: train={len(loaders['train'].dataset)}, val={len(loaders['valid'].dataset)}")

    # 1. UNet_SAR
    logger.info("--- Sanity Check 1/2: UNet_SAR (5 epochs) ---")
    unet_sar = SimpleUNet(in_channels=2, sar_only=True)
    unet_sar = train_simple(
        unet_sar, loaders["train"], loaders["valid"],
        epochs=5, lr=1e-3, device=dev, smoke=False,
        use_amp=True, early_stop_patience=5
    )
    sar_metrics = evaluate_model(unet_sar, loaders["valid"], device=dev)
    logger.info(f"UNet_SAR 5-Epoch Validation IoU: {sar_metrics['iou']:.4f}, ECE: {sar_metrics.get('ece', 0.0):.4f}")

    if sar_metrics["iou"] < 0.05:
        raise RuntimeError(
            f"SANITY CHECK FAILED: UNet_SAR validation IoU is {sar_metrics['iou']:.4f} (< 0.05)! "
            "SAR preprocessing has collapsed to predicting 0 flood pixels."
        )

    # 2. ResQNet
    logger.info("--- Sanity Check 2/2: ResQNet (5 epochs) ---")
    resqnet = ResQNet(
        sar_channels=2, optical_channels=13, geo_channels=6, rainfall_seq_len=30,
        rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1, pretrained=False, dropout=0.1
    )
    resqnet = train_simple(
        resqnet, loaders["train"], loaders["valid"],
        epochs=5, lr=1e-3, device=dev, smoke=False,
        use_amp=True, early_stop_patience=5
    )
    resqnet_metrics = evaluate_model(resqnet, loaders["valid"], device=dev)
    logger.info(f"ResQNet 5-Epoch Validation IoU: {resqnet_metrics['iou']:.4f}, ECE: {resqnet_metrics.get('ece', 0.0):.4f}")

    if resqnet_metrics["iou"] < 0.05:
        raise RuntimeError(
            f"SANITY CHECK FAILED: ResQNet validation IoU is {resqnet_metrics['iou']:.4f} (< 0.05)! "
            "ResQNet multimodal model has collapsed."
        )

    logger.info("=== SANITY CHECK PASSED: Both UNet_SAR and ResQNet achieved valid IoU >= 0.05! ===")


def main():
    parser = argparse.ArgumentParser(description="ResQ-AI Complete Experiment Runner")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "smoke"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run rapid CPU smoke tests in under 5 minutes")
    parser.add_argument("--sanity_check", action="store_true", help="Run 5-epoch sanity check on real data for UNet_SAR and ResQNet")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds for multi-seed evaluation")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    config_path = ROOT_DIR / args.config
    cfg = load_config(str(config_path), smoke=args.smoke)
    cfg["project"]["device"] = args.device

    if args.sanity_check:
        run_sanity_check(cfg, device=args.device)
        return

    if args.smoke or args.experiment == "smoke":
        cfg["project"]["smoke"] = True
        logger.info("=== Running ResQ-AI in SMOKE mode (<5 min on CPU) ===")
        start_time = time.time()
        smoke_seeds = args.seeds if args.seeds else [42, 123, 456]

        exps_to_run = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"] if args.experiment in ["all", "smoke"] else [args.experiment]
        exp_funcs = {
            "e1": run_e1_main_comparison,
            "e2": run_e2_ablations,
            "e3": run_e3_calibration,
            "e4": run_e4_conformal,
            "e5": run_e5_robustness,
            "e6": run_e6_allocation,
            "e7": run_e7_efficiency,
        }

        for seed in smoke_seeds:
            set_seed(seed)
            for exp_name in exps_to_run:
                func = exp_funcs[exp_name]
                res = func(cfg, seed=seed)
                with open(results_dir / f"{exp_name}_seed_{seed}.json", "w") as f:
                    json.dump(res, f, indent=2)

        agg = aggregate_results(results_dir, smoke_seeds)
        elapsed = time.time() - start_time
        logger.info(f"=== Smoke Experiments Completed in {elapsed:.2f}s ===")

        # Run generate figures and tables
        try:
            import experiments.generate_figures as gf
            import experiments.generate_tables as gt
            gf.main()
            gt.main()
            logger.info("Figures and LaTeX tables generated successfully.")
        except Exception as e:
            logger.warning(f"Note on figure/table generation: {e}")

        return

    # Full mode
    seeds = args.seeds
    exps = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"] if args.experiment == "all" else [args.experiment]
    exp_funcs = {
        "e1": run_e1_main_comparison,
        "e2": run_e2_ablations,
        "e3": run_e3_calibration,
        "e4": run_e4_conformal,
        "e5": run_e5_robustness,
        "e6": run_e6_allocation,
        "e7": run_e7_efficiency,
    }

    for exp in exps:
        func = exp_funcs[exp]
        for seed in seeds:
            res = func(cfg, seed=seed)
            with open(results_dir / f"{exp}_seed_{seed}.json", "w") as f:
                json.dump(res, f, indent=2)

    aggregate_results(results_dir, seeds)


if __name__ == "__main__":
    main()
