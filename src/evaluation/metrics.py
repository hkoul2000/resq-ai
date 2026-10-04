"""Evaluation metrics for flood segmentation and uncertainty quantification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

class FloodMetrics:
    """Computes basic segmentation metrics with valid_mask support."""

    @staticmethod
    def compute_all(
        probs: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if valid_mask is not None:
            vmask_flat = valid_mask.flatten() > 0.5
            if vmask_flat.sum() == 0:
                return {
                    "iou": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "auroc": 0.0,
                    "auprc": 0.0,
                }
            probs_flat = probs.flatten()[vmask_flat]
            labels_flat = labels.flatten()[vmask_flat]
        else:
            probs_flat = probs.flatten()
            labels_flat = labels.flatten()

        preds = (probs_flat > threshold).float()

        tp = (preds * labels_flat).sum().item()
        fp = (preds * (1.0 - labels_flat)).sum().item()
        fn = ((1.0 - preds) * labels_flat).sum().item()
        tn = ((1.0 - preds) * (1.0 - labels_flat)).sum().item()

        iou = tp / (tp + fp + fn + 1e-8)
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        probs_np = probs_flat.cpu().numpy()
        labels_np = labels_flat.cpu().numpy()

        try:
            auroc = roc_auc_score(labels_np, probs_np)
            auprc = average_precision_score(labels_np, probs_np)
        except ValueError:
            auroc = 0.0
            auprc = 0.0

        return {
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "auroc": float(auroc),
            "auprc": float(auprc),
        }

    @classmethod
    def compute(
        cls,
        probs: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Alias for compute_all with valid_mask support."""
        return cls.compute_all(probs, labels, threshold=threshold, valid_mask=valid_mask)

class CalibrationMetrics:
    @staticmethod
    def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

class UncertaintyMetrics:
    @staticmethod
    def compute_error_detection_auroc(uncertainties: np.ndarray, errors: np.ndarray) -> float:
        try:
            return roc_auc_score(errors, uncertainties)
        except ValueError:
            return 0.5
            
    @staticmethod
    def compute_spearman_correlation(uncertainties: np.ndarray, errors: np.ndarray) -> float:
        corr, _ = stats.spearmanr(uncertainties, errors)
        return corr

def compute_all_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    uncertainties: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    metrics = {}
    metrics.update(FloodMetrics.compute_all(probs, labels, valid_mask=valid_mask))

    if valid_mask is not None:
        vmask_np = valid_mask.flatten().cpu().numpy() > 0.5
        probs_np = probs.flatten().cpu().numpy()[vmask_np]
        labels_np = labels.flatten().cpu().numpy()[vmask_np]
    else:
        probs_np = probs.flatten().cpu().numpy()
        labels_np = labels.flatten().cpu().numpy()

    metrics["ece"] = CalibrationMetrics.compute_ece(probs_np, labels_np)

    if uncertainties is not None:
        if valid_mask is not None:
            unc_np = uncertainties.flatten().cpu().numpy()[vmask_np]
        else:
            unc_np = uncertainties.flatten().cpu().numpy()
        preds_np = (probs_np > 0.5).astype(float)
        errors = (preds_np != labels_np).astype(int)

        metrics["error_detection_auroc"] = UncertaintyMetrics.compute_error_detection_auroc(unc_np, errors)
        metrics["uncertainty_error_corr"] = UncertaintyMetrics.compute_spearman_correlation(unc_np, errors)

    return metrics

def paired_significance_test(scores_a: np.ndarray, scores_b: np.ndarray, test: str = 'wilcoxon') -> Tuple[float, float]:
    if test == 'wilcoxon':
        stat, pval = stats.wilcoxon(scores_a, scores_b)
    elif test == 'ttest':
        stat, pval = stats.ttest_rel(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")
    return pval, stat

def compute_confidence_intervals(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    mean = np.mean(values)
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    return mean, mean - ci, mean + ci
