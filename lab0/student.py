"""Student-editable Lab 0 tasks.

Implement each public interface using NumPy or PyTorch as stated. The course tests
check behaviour and shapes, but the internal organization of your code is your choice.
"""
from __future__ import annotations
import numpy as np
import torch

def select_columns(x: np.ndarray, column_indices: list[int]) -> np.ndarray:
    """Return the requested columns while preserving the batch dimension."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f'expected a 2-D array (batch, features), got shape {x.shape}')
    return x[:, list(column_indices)]

def standardize_columns(x: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray) -> np.ndarray:
    """Apply column-wise standardization using the supplied training statistics.

    Raise ValueError for a zero standard deviation; do not modify the input."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    if np.any(standard_deviation == 0):
        raise ValueError('standard deviation contains zero; cannot standardize that column')
    return (x - mean) / standard_deviation

def linear_scores_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Calculate batched linear scores with shape (batch, outputs)."""
    x = np.asarray(x)
    weight = np.asarray(weight)
    bias = np.asarray(bias)
    if x.ndim != 2:
        raise ValueError(f'expected x with shape (batch, inputs), got {x.shape}')
    return x @ weight.T + bias

class LinearTransform:
    """Small Python class used to refresh constructors and callable objects."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        self.weight = np.array(weight, copy=True)
        self.bias = np.array(bias, copy=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return linear_scores_numpy(x, self.weight, self.bias)

def torch_linear_scores(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PyTorch equivalent of linear_scores_numpy."""
    if x.ndim != 2:
        raise ValueError(f'expected x with shape (batch, inputs), got {tuple(x.shape)}')
    return x @ weight.T + bias

def rearrange_nchw_to_nhwc(images: torch.Tensor) -> torch.Tensor:
    """Convert a batch from (N, C, H, W) to contiguous (N, H, W, C)."""
    if images.ndim != 4:
        raise ValueError(f'expected a 4-D (N, C, H, W) tensor, got {tuple(images.shape)}')
    return images.permute(0, 2, 3, 1).contiguous()

def accuracy_fraction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy as a fraction in the closed interval [0, 1]."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError(f'length mismatch: {y_true.size} labels vs {y_pred.size} predictions')
    if y_true.size == 0:
        raise ValueError('cannot compute accuracy of zero examples')
    return float(np.mean(y_true == y_pred))
