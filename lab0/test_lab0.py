from __future__ import annotations
import numpy as np
import pytest
import torch
from lab0.student import LinearTransform, accuracy_fraction, linear_scores_numpy, rearrange_nchw_to_nhwc, select_columns, standardize_columns, torch_linear_scores

def test_select_columns_preserves_batch_dimension() -> None:
    x = np.arange(20).reshape(5, 4)
    result = select_columns(x, [3, 1])
    np.testing.assert_array_equal(result, x[:, [3, 1]])
    assert result.shape == (5, 2)

def test_standardize_columns_uses_supplied_statistics() -> None:
    x = np.array([[1.0, 10.0], [3.0, 14.0]])
    mean = np.array([2.0, 12.0])
    std = np.array([1.0, 2.0])
    np.testing.assert_allclose(standardize_columns(x, mean, std), [[-1.0, -1.0], [1.0, 1.0]])

def test_linear_scores_numpy() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    weight = np.array([[2.0, -1.0], [0.5, 1.5]])
    bias = np.array([0.25, -0.5])
    expected = x @ weight.T + bias
    np.testing.assert_allclose(linear_scores_numpy(x, weight, bias), expected)

def test_linear_transform_class_matches_function() -> None:
    x = np.array([[1.0, 2.0]])
    weight = np.array([[2.0, -1.0]])
    bias = np.array([0.25])
    transform = LinearTransform(weight, bias)
    np.testing.assert_allclose(transform(x), linear_scores_numpy(x, weight, bias))

def test_torch_linear_scores_matches_numpy() -> None:
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    weight = torch.tensor([[2.0, -1.0], [0.5, 1.5]])
    bias = torch.tensor([0.25, -0.5])
    expected = x @ weight.T + bias
    torch.testing.assert_close(torch_linear_scores(x, weight, bias), expected)

def test_rearrange_is_nhwc_and_contiguous() -> None:
    images = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    result = rearrange_nchw_to_nhwc(images)
    assert result.shape == (2, 4, 5, 3)
    assert result.is_contiguous()
    torch.testing.assert_close(result[0, 2, 4], images[0, :, 2, 4])


def test_accuracy_fraction() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy_fraction(y_true, y_pred) == pytest.approx(0.75)
    assert 0.0 <= accuracy_fraction(y_true, y_pred) <= 1.0
