from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np

@dataclass(slots=True)
class Parameter:
    """Small NumPy parameter container used by Lab 4.

    Students do not implement registration or gradient clearing. They implement the
    mathematical operations that read and write these arrays.
    """
    data: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

class Module:

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for value in vars(self).values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

def numerical_gradient(
    objective,
    parameter: Parameter,
    *,
    epsilon: float=1e-06,
    indices: Iterable[tuple[int, ...]] | None=None,
) -> dict[tuple[int, ...], float]:
    """Estimate selected derivatives using centred finite differences."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    selected = list(indices) if indices is not None else list(np.ndindex(parameter.data.shape))
    estimates: dict[tuple[int, ...], float] = {}
    for index in selected:
        original = float(parameter.data[index])
        try:
            parameter.data[index] = original + epsilon
            plus = float(objective())
            parameter.data[index] = original - epsilon
            minus = float(objective())
        finally:
            parameter.data[index] = original
        estimates[index] = (plus - minus) / (2.0 * epsilon)
    return estimates

def relative_error(a: np.ndarray | float, b: np.ndarray | float, epsilon: float=1e-12) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denominator = np.maximum(epsilon, np.abs(a_arr) + np.abs(b_arr))
    return float(np.max(np.abs(a_arr - b_arr) / denominator))
