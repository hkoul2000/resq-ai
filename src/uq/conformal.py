"""Conformal prediction and conformal risk control for flood segmentation.

Implements:
- Split conformal prediction with pixel-level prediction sets
- Conformal risk control (Bates et al., 2021)
- Coverage tracking and calibration
"""

import torch
import numpy as np
from typing import Dict, Union, Tuple, List, Optional
import math


class ConformalCalibrator:
    """Calibrates conformal prediction threshold based on a calibration set."""

    def __init__(self) -> None:
        self.threshold: Optional[float] = None
        self.alpha: Optional[float] = None

    def calibrate(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor, alpha: float = 0.1) -> None:
        """
        Calibrate the threshold using the calibration set.
        
        Args:
            cal_probs: Predicted probabilities (N, C, H, W) or (N, H, W)
            cal_labels: Ground truth labels (N, H, W)
            alpha: Target error rate (default: 0.1 for 90% coverage)
        """
        self.alpha = alpha
        
        if cal_probs.dim() == 4:
            # Multi-class: get probability of true class
            # cal_probs: (N, C, H, W), cal_labels: (N, H, W)
            gathered_probs = torch.gather(cal_probs, 1, cal_labels.unsqueeze(1)).squeeze(1)
            scores = 1.0 - gathered_probs
        else:
            # Binary class: (N, H, W)
            probs = torch.where(cal_labels == 1, cal_probs, 1.0 - cal_probs)
            scores = 1.0 - probs
            
        scores_flat = scores.flatten().cpu().numpy()
        scores_flat = scores_flat[~np.isnan(scores_flat)] # Remove nans if any
        
        n = len(scores_flat)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.threshold = float(np.quantile(scores_flat, q_level, method='higher'))

    def predict_sets(self, test_probs: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction sets for the test data.
        
        Args:
            test_probs: Predicted probabilities
            
        Returns:
            Boolean tensor representing the prediction set
        """
        if self.threshold is None:
            raise ValueError("Calibrator must be calibrated before generating sets.")
            
        if test_probs.dim() == 4:
            scores = 1.0 - test_probs
        else:
            # For binary, set contains 1 if 1-p <= threshold, 0 if p <= threshold
            # Returns mask of shape (N, 2, H, W)
            s_1 = 1.0 - test_probs
            s_0 = test_probs
            sets = torch.stack([s_0 <= self.threshold, s_1 <= self.threshold], dim=1)
            return sets
            
        return scores <= self.threshold

    def evaluate_coverage(self, test_probs: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate empirical coverage on test data.
        """
        sets = self.predict_sets(test_probs)
        
        if sets.dim() == 4:
            # Multi-class or binary stacked
            is_covered = torch.gather(sets, 1, test_labels.unsqueeze(1)).squeeze(1)
            set_sizes = sets.sum(dim=1, dtype=torch.float32)
        else:
            raise NotImplementedError("Unexpected sets dimension")
            
        coverage = is_covered.float().mean().item()
        avg_size = set_sizes.mean().item()
        
        return {
            "coverage": coverage,
            "avg_set_size": avg_size
        }


class ConformalRiskControl:
    """Implements conformal risk control for a general loss function."""

    def __init__(self, loss_fn: callable) -> None:
        self.loss_fn = loss_fn
        self.lambda_hat: Optional[float] = None
        self.alpha: Optional[float] = None

    def calibrate(self, cal_preds: torch.Tensor, cal_labels: torch.Tensor, alpha: float = 0.1, lambdas: Optional[torch.Tensor] = None) -> None:
        """
        Find minimum lambda such that expected risk is <= alpha.
        """
        self.alpha = alpha
        if lambdas is None:
            lambdas = torch.linspace(0, 1, 100)
            
        n = cal_preds.shape[0]
        
        best_lambda = lambdas[-1]
        for lam in lambdas:
            losses = self.loss_fn(cal_preds, cal_labels, lam)
            risk = losses.mean().item()
            ub = risk + math.sqrt(math.log(1/alpha) / (2*n)) # Hoeffding-style bound or empirical check
            # For CRC Bates et al: R_hat(lam) <= alpha - C
            if risk <= alpha:
                best_lambda = lam
                break
                
        self.lambda_hat = float(best_lambda)


class PixelConformalPredictor:
    """Wraps a model and calibrator to produce conformal sets per pixel."""

    def __init__(self, model: torch.nn.Module, calibrator: ConformalCalibrator) -> None:
        self.model = model
        self.calibrator = calibrator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            probs = self.model(x)
            sets = self.calibrator.predict_sets(probs)
        return probs, sets
