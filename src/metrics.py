# metrics.py
import torch
import numpy as np
from typing import Optional

# --- Belief Quality ---

def rmse(mean: torch.Tensor, groundTruth: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((mean - groundTruth) ** 2)).item()

def nlpd(mean: torch.Tensor, variance: torch.Tensor, groundTruth: torch.Tensor) -> float:
    """Negative Log Predictive Density — proper scoring rule for GP quality."""
    return (0.5 * torch.log(2 * torch.pi * variance) + 
            0.5 * ((groundTruth - mean) ** 2) / variance).mean().item()

def normalizedTraceReduction(currentTrace: float, initialTrace: float) -> float:
    return (initialTrace - currentTrace) / initialTrace

# def calibrationError(mean: torch.Tensor, variance: torch.Tensor, 
#                      groundTruth: torch.Tensor, confidence: float = 0.95) -> float:
#     """Fraction of truth values falling outside the credible interval."""
#     std = variance.sqrt()
#     z = 1.96  # 95% interval
#     lower, upper = mean - z * std, mean + z * std
#     coverage = ((groundTruth >= lower) & (groundTruth <= upper)).float().mean().item()
#     return abs(coverage - confidence)  # 0 is perfect calibration

# --- Planning Efficiency ---

def informationGain(previousVariance: torch.Tensor, currentVariance: torch.Tensor) -> float:
    """Reduction in total posterior entropy from one step to the next."""
    return (previousVariance.sum() - currentVariance.sum()).item()

def pathLength(positionHistory: list) -> float:
    if len(positionHistory) < 2:
        return 0.0
    positions = np.array(positionHistory)
    return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
