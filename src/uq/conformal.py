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

    def calibrate(
        self,
        cal_probs: Union[torch.Tensor, np.ndarray],
        cal_labels: Union[torch.Tensor, np.ndarray],
        alpha: float = 0.1,
    ) -> None:
        """
        Calibrate the threshold using the calibration set.

        Args:
            cal_probs: Predicted probabilities. Can be numpy or torch tensor.
                       Shape: (N,) for flat, (N, H, W) for spatial binary,
                       or (N, C, H, W) for multi-class.
            cal_labels: Ground truth labels.
            alpha: Target error rate (default: 0.1 for 90% coverage)
        """
        self.alpha = alpha

        # Convert to numpy if needed
        if isinstance(cal_probs, torch.Tensor):
            cal_probs_np = cal_probs.cpu().numpy()
        else:
            cal_probs_np = np.asarray(cal_probs)

        if isinstance(cal_labels, torch.Tensor):
            cal_labels_np = cal_labels.cpu().numpy()
        else:
            cal_labels_np = np.asarray(cal_labels)

        # Compute conformity scores: 1 - p(true class)
        if cal_probs_np.ndim >= 3 and cal_probs_np.shape[-3] > 1:
            # Multi-class: (N, C, ...) - get probability of true class
            # Use advanced indexing
            idx = cal_labels_np.astype(int)
            probs_true = np.take_along_axis(
                cal_probs_np, idx[np.newaxis] if cal_probs_np.ndim == 3 else idx[:, np.newaxis], axis=-3 if cal_probs_np.ndim == 4 else 0
            )
            scores = 1.0 - probs_true.flatten()
        else:
            # Binary case: flat or spatial
            flat_probs = cal_probs_np.flatten()
            flat_labels = cal_labels_np.flatten()
            probs_true = np.where(flat_labels == 1, flat_probs, 1.0 - flat_probs)
            scores = 1.0 - probs_true

        scores = scores[~np.isnan(scores)]

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)

        self.threshold = float(np.quantile(scores, q_level, method='higher'))

    def predict_sets(
        self, test_probs: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate prediction sets for the test data.

        Args:
            test_probs: Predicted probabilities

        Returns:
            Boolean array/tensor representing the prediction set
        """
        if self.threshold is None:
            raise ValueError("Calibrator must be calibrated before generating sets.")

        is_tensor = isinstance(test_probs, torch.Tensor)
        if is_tensor:
            probs_np = test_probs.cpu().numpy()
        else:
            probs_np = np.asarray(test_probs)

        if probs_np.ndim >= 3 and probs_np.shape[-3] > 1:
            # Multi-class
            scores = 1.0 - probs_np
            sets = scores <= self.threshold
        else:
            # Binary: prediction set includes class 1 if 1-p <= threshold
            flat = probs_np.flatten()
            sets = (1.0 - flat) <= self.threshold

        if is_tensor:
            return torch.from_numpy(sets)
        return sets

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
