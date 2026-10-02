"""Evidential Deep Learning for flood segmentation.

Based on:
- Sensoy et al. (2018) 'Evidential Deep Learning to Quantify Classification Uncertainty'
- Amini et al. (2020) 'Deep Evidential Regression'
"""

import torch
import torch.nn as nn
from typing import Tuple


class EvidentialLoss(nn.Module):
    """Evidential loss for multi-class classification using Dirichlet distribution."""

    def __init__(self, num_classes: int, annealing_step: int = 10, max_annealing: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.max_annealing = max_annealing

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Args:
            evidence: (N, C, H, W) predicted evidence (must be >= 0)
            targets: (N, H, W) integer targets or (N, C, H, W) one-hot
            epoch: current training epoch for annealing
        """
        if targets.dim() == 3:
            targets = torch.nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Cross entropy term
        loss_ce = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # KL Divergence term
        annealing_coef = min(self.max_annealing, epoch / self.annealing_step)
        
        alpha_tilde = targets + (1 - targets) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1, keepdim=True) \
                 - torch.lgamma(torch.ones_like(S_tilde) * self.num_classes) \
                 + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        kl_div = kl_div.squeeze(1)
        
        loss = loss_ce + annealing_coef * kl_div
        return loss.mean()


class BetaEvidentialLoss(nn.Module):
    """Evidential loss for binary classification using Beta distribution."""
    
    def __init__(self, annealing_step: int = 10, max_annealing: float = 1.0):
        super().__init__()
        self.annealing_step = annealing_step
        self.max_annealing = max_annealing

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Args:
            evidence: (N, 2, H, W) predicted evidence
            targets: (N, H, W) integer targets
        """
        targets_oh = torch.nn.functional.one_hot(targets, 2).permute(0, 3, 1, 2).float()
        
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        loss_ce = torch.sum(targets_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        annealing_coef = min(self.max_annealing, epoch / self.annealing_step)
        
        alpha_tilde = targets_oh + (1 - targets_oh) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
                 + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1, keepdim=True) \
                 - torch.lgamma(torch.ones_like(S_tilde) * 2) \
                 + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
                 
        kl_div = kl_div.squeeze(1)
        
        loss = loss_ce + annealing_coef * kl_div
        return loss.mean()


def compute_evidential_uncertainty(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total, aleatoric, and epistemic uncertainty from Dirichlet parameters.
    
    Args:
        alpha: (N, C, H, W) Dirichlet parameters (evidence + 1)
        
    Returns:
        total_unc, aleatoric_unc, epistemic_unc
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    
    # Total uncertainty (entropy of expected distribution)
    total_unc = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    
    # Aleatoric uncertainty (expected entropy)
    digamma_S = torch.digamma(S + 1)
    aleatoric_unc = -torch.sum(probs * (torch.digamma(alpha + 1) - digamma_S), dim=1)
    
    # Epistemic uncertainty (mutual information)
    epistemic_unc = total_unc - aleatoric_unc
    
    return total_unc, aleatoric_unc, epistemic_unc
