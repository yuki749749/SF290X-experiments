# metrics.py
import torch
import numpy as np
from typing import Optional

# --- Belief Quality ---

def rmse(mean: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((mean - ground_truth) ** 2)).item()

def nlpd(mean: torch.Tensor, variance: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Negative Log Predictive Density — proper scoring rule for GP quality."""
    return (0.5 * torch.log(2 * torch.pi * variance) + 
            0.5 * ((ground_truth - mean) ** 2) / variance).mean().item()

def normalized_trace_reduction(current_trace: float, initial_trace: float) -> float:
    return (initial_trace - current_trace) / initial_trace

# def calibrationError(mean: torch.Tensor, variance: torch.Tensor, 
#                      ground_truth: torch.Tensor, confidence: float = 0.95) -> float:
#     """Fraction of truth values falling outside the credible interval."""
#     std = variance.sqrt()
#     z = 1.96  # 95% interval
#     lower, upper = mean - z * std, mean + z * std
#     coverage = ((ground_truth >= lower) & (ground_truth <= upper)).float().mean().item()
#     return abs(coverage - confidence)  # 0 is perfect calibration

# --- Planning Efficiency ---

def information_gain(previous_variance: torch.Tensor, current_variance: torch.Tensor) -> float:
    """Reduction in total posterior entropy from one step to the next."""
    return (previous_variance.sum() - current_variance.sum()).item()

def path_length(position_history: list) -> float:
    if len(position_history) < 2:
        return 0.0
    positions = np.array(position_history)
    return np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
