"""Calibration analysis for flood prediction models."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import os

class ReliabilityDiagram:
    @staticmethod
    def compute_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = np.zeros(n_bins)
        confidences = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            counts[i] = in_bin.sum()
            if counts[i] > 0:
                accuracies[i] = labels[in_bin].mean()
                confidences[i] = probs[in_bin].mean()
                
        return {
            "accuracies": accuracies,
            "confidences": confidences,
            "counts": counts,
            "bins": bin_boundaries
        }

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    data = ReliabilityDiagram.compute_data(probs, labels, n_bins)
    total_count = data["counts"].sum()
    if total_count == 0:
        return 0.0
    
    ece = np.sum((data["counts"] / total_count) * np.abs(data["accuracies"] - data["confidences"]))
    return float(ece)

def temperature_scaling(logits: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    """Post-hoc temperature scaling calibration."""
    class TemperatureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            
        def forward(self, logits):
            return logits / self.temperature

    model = TemperatureModel().to(logits.device)
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    if logits.dim() == 4:
        # (N, C, H, W)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        labels_flat = labels.reshape(-1)
    else:
        logits_flat = logits.flatten()
        labels_flat = labels.flatten()

    def eval():
        optimizer.zero_grad()
        loss = criterion(model(logits_flat.unsqueeze(1) if logits_flat.dim()==1 else logits_flat), labels_flat)
        loss.backward()
        return loss
        
    optimizer.step(eval)
    return model

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, save_path: str, n_bins: int = 15) -> None:
    data = ReliabilityDiagram.compute_data(probs, labels, n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    
    valid = data["counts"] > 0
    plt.bar(data["confidences"][valid], data["accuracies"][valid], width=1/n_bins, alpha=0.7, edgecolor="black", label="Model")
    plt.plot(data["confidences"][valid], data["accuracies"][valid], 'r-o')
    
    plt.ylabel("Accuracy")
    plt.xlabel("Confidence")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_risk_coverage_curve(uncertainties: np.ndarray, errors: np.ndarray, save_path: str) -> None:
    sorted_indices = np.argsort(uncertainties)
    sorted_errors = errors[sorted_indices]
    
    coverages = np.linspace(0.01, 1.0, 100)
    risks = []
    
    for cov in coverages:
        n_samples = int(cov * len(sorted_errors))
        if n_samples == 0:
            risks.append(0)
        else:
            risks.append(sorted_errors[:n_samples].mean())
            
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, 'b-')
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
