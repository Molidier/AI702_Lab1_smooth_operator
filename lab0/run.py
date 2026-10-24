"""Supplied Lab 0 baseline. Training internals are a black box until later labs."""
import numpy as np
import torch
from shared.common import Run, save_predictions
from shared.data import tabular
from shared import plots

def run(tasks, cfg, *, smoke=False, reference=False, stage='core'):
    out = Run(0, smoke=smoke, reference=reference, stage=stage, config=cfg)
    torch.manual_seed(cfg['seed'])
    data = tabular('breast_cancer', seed=cfg['seed'])
    x = torch.as_tensor(data['x_train'], dtype=torch.float32)
    y = torch.as_tensor(data['y_train'][:, None], dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    losses = []
    for _ in range(20 if smoke else 160):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(data['x_validation'], dtype=torch.float32)).squeeze(1)
        probabilities = logits.sigmoid().numpy()
        predicted = (probabilities >= 0.5).astype(int)
    accuracy = tasks.accuracy_fraction(data['y_validation'], predicted)
    reported = float(np.mean(data['y_validation'] == predicted)) * 100
    save_predictions(
        out.path / 'validation_predictions.csv',
        data['validation_ids'],
        data['y_validation'],
        predicted,
        np.c_[1 - probabilities, probabilities],
    )
    tiny = np.arange(12.0, dtype=float).reshape(4, 3)
    selected = tasks.select_columns(tiny, [0, 2])
    standardized = tasks.standardize_columns(selected, selected.mean(0), selected.std(0))
    weights = np.array([[0.2, 0.5], [-0.5, 0.1]])
    bias = np.array([0.2, -0.1])
    numpy_scores = tasks.LinearTransform(weights, bias)(standardized)
    tensor_scores = tasks.torch_linear_scores(
        torch.tensor(standardized),
        torch.tensor(weights),
        torch.tensor(bias),
    )
    rearranged = tasks.rearrange_nchw_to_nhwc(torch.arange(48.0).reshape(2, 3, 2, 4))
    original_layout = torch.arange(24.0).reshape(2, 3, 4)
    transposed_layout = original_layout.transpose(1, 2)
    try:
        transposed_layout.view(-1)
        view_message = 'view succeeded'
    except RuntimeError:
        view_message = 'view failed because this transposed tensor is non-contiguous'
    layout_report = {
        'original_shape': list(original_layout.shape),
        'transposed_shape': list(transposed_layout.shape),
        'transposed_is_contiguous': transposed_layout.is_contiguous(),
        'view_result': view_message,
        'reshape_shape': list(transposed_layout.reshape(-1).shape),
        'contiguous_view_shape': list(transposed_layout.contiguous().view(-1).shape),
    }
    # This is a demonstration of accumulation, not an implementation of backpropagation.
    # scalar = torch.tensor(2.0, requires_grad=True)
    # (scalar ** 2).backward()
    # first = scalar.grad.item()
    # (scalar ** 2).backward()
    # accumulated = scalar.grad.item()
    # scalar.grad.zero_()
    # (scalar ** 2).backward()
    # cleared = scalar.grad.item()
    out.arrays(
        'raw.npz',
        loss=losses,
        targets=data['y_validation'],
        predictions=predicted,
        probabilities=probabilities,
        selected=selected,
        standardized=standardized,
        numpy_scores=numpy_scores,
        torch_scores=tensor_scores.detach().numpy(),
    )
    plots.curves(
        out.path / 'loss.png',
        {'baseline': losses},
        title='Supplied baseline',
        ylabel='Binary cross entropy',
    )
    plots.confusion(out.path / 'confusion.png', data['y_validation'], predicted, data['class_names'])
    return out.finish(
        {
            'reported_accuracy': reported,
            'reported_units': 'fraction between 0 and 1',
            'student_recomputed_accuracy': accuracy,
            'examples': len(predicted),
        },
        dataset=data['name'],
        details={
            'note': 'One metric above is deliberately inconsistent. Use the CSV to explain why.',
            'array_tensor_max_difference': float(np.max(np.abs(numpy_scores - tensor_scores.numpy()))),
            'permuted_shape': list(rearranged.shape),
            'view_reshape_demonstration': layout_report,
            # 'quadratic_gradient': quadratic_gradient.tolist(),
            # 'gradient_accumulation': {
            #     'first': first,
            #     'second_without_clearing': accumulated,
            #     'after_clearing': cleared,
            # },
            'classes': data['class_names'],
            'educational_only': 'This is not a clinical model.',
        },
    )

def baseline_demo(cfg, *, smoke=False):
    """A completely provided run; it never imports student answers or counts as a submission."""
    out = Run(0, smoke=smoke, reference=True, stage='baseline', config=cfg)
    torch.manual_seed(cfg['seed'])
    data = tabular('breast_cancer', seed=cfg['seed'])
    x = torch.as_tensor(data['x_train'], dtype=torch.float32)
    y = torch.as_tensor(data['y_train'][:, None], dtype=torch.float32)
    model = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    history = []
    for _ in range(20 if smoke else 160):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        p = model(torch.as_tensor(data['x_validation'], dtype=torch.float32)).sigmoid().squeeze(1).numpy()
    yhat = (p >= 0.5).astype(int)
    save_predictions(
        out.path / 'validation_predictions.csv',
        data['validation_ids'],
        data['y_validation'],
        yhat,
        np.c_[1 - p, p],
    )
    plots.curves(
        out.path / 'loss.png',
        {'provided baseline': history},
        title='Provided baseline',
        ylabel='Loss',
    )
    return out.finish(
        {
            'reported_accuracy': 100 * float(np.mean(yhat == data['y_validation'])),
            'reported_units': 'fraction between 0 and 1',
        },
        dataset=data['name'],
        details={'task': 'The reporting unit is deliberately wrong. Find it from the CSV. No clinical use.'},
    )
