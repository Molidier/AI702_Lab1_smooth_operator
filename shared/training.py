"""Ordinary PyTorch loops supplied for Labs 5-7. No trainer framework or callbacks.

Students are shown this file in Lab 5, after implementing backpropagation in Lab 4.
Validation selects checkpoints. Only the staff holdout worker evaluates the final test split.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .common import save_json, save_predictions

class TrainingInstability(RuntimeError):
    """A measured nonfinite loss, with the completed epochs retained for diagnosis."""
    def __init__(self, history, epoch, examples_processed):
        super().__init__('Nonfinite training loss: inspect the learning rate and gradients')
        self.history = list(history)
        self.epoch = epoch
        self.examples_processed = examples_processed


def device_for(name='cpu'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA is not available; use --device cpu')
    if name == 'mps' and (not torch.backends.mps.is_available()):
        raise ValueError('MPS is not available; use --device cpu')
    return torch.device(name)

def mlp(input_shape, classes, *, hidden=96, dropout=0.0, batch_norm=False):
    modules = [nn.Flatten(), nn.Linear(int(np.prod(input_shape)), hidden)]
    if batch_norm:
        modules.append(nn.BatchNorm1d(hidden))
    modules.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, classes)])
    return nn.Sequential(*modules)

def loader(x, y, *, batch_size=64, seed=702, order='shuffle', drop_last=False):
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    if order == 'grouped':
        indices = torch.argsort(y, stable=True)
        x, y = (x[indices], y[indices])
    elif order not in ('shuffle', 'sequential'):
        raise ValueError('order must be shuffle, sequential, or grouped')
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=order == 'shuffle',
        generator=torch.Generator().manual_seed(seed),
        drop_last=drop_last,
        num_workers=0,
    )

def evaluate(model, data, *, split='validation', device='cpu', x_override=None, prepare=None):
    if prepare is None:
        model.eval()
    else:
        prepare(model)
        if model.training:
            raise ValueError('prepare_for_evaluation must put the model in evaluation mode')
    x = data[f'x_{split}'] if x_override is None else x_override
    y = data[f'y_{split}']
    logits = []
    with torch.no_grad():
        for bx, _ in loader(x, y, batch_size=128, order='sequential'):
            logits.append(model(bx.to(device)).detach().cpu())
    logits = torch.cat(logits)
    labels = torch.as_tensor(y, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    probabilities = logits.softmax(1).numpy()
    predictions = probabilities.argmax(1)
    per_class = [float(np.mean(predictions[y == c] == c)) if np.any(y == c) else None for c in range(len(data['class_names']))]
    return {
        'loss': loss,
        'accuracy': float(np.mean(predictions == y)),
        'predictions': predictions,
        'probabilities': probabilities,
        'logits': logits.numpy(),
        'per_class_accuracy': per_class,
    }

def activation_summary(model, x, *, device='cpu'):
    """Capture actual first-linear activations using a temporary forward hook."""
    layer = next((m for m in model.modules() if isinstance(m, nn.Linear)), None)
    if layer is None:
        return {}
    result = {}

    def record(_, __, output):
        value = output.detach()
        result.update(
            mean=float(value.mean()),
            standard_deviation=float(value.std(unbiased=False)),
            fraction_nonpositive=float((value <= 0).float().mean()),
        )
    handle = layer.register_forward_hook(record)
    try:
        model.eval()
        with torch.no_grad():
            model(torch.as_tensor(x[:64], dtype=torch.float32, device=device))
    finally:
        handle.remove()
    return result

def fit(
    model,
    data,
    output: Path,
    *,
    epochs=8,
    batch_size=64,
    lr=0.03,
    weight_decay=0.0,
    momentum=0.9,
    clip=None,
    patience=None,
    augment=False,
    order='shuffle',
    device='cpu',
    seed=702,
    prepare=None,
    checkpoint_selector=None,
    scheduler_step=None,
):
    """Save actual epoch checkpoints and restore the validation-selected model."""
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5) if scheduler_step else None
    has_bn = any((isinstance(layer, nn.BatchNorm1d) for layer in model.modules()))
    train_loader = loader(
        data['x_train'],
        data['y_train'],
        batch_size=batch_size,
        seed=seed,
        order=order,
        drop_last=has_bn and len(data['y_train']) % batch_size == 1,
    )
    history = []
    paths = []
    best = -float('inf')
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        norm_values = []
        for x, y in train_loader:
            x, y = (x.to(device), y.to(device))
            if augment and x.ndim == 4:
                flip = torch.rand(len(x), device=device) < 0.5
                x = x.clone()
                x[flip] = x[flip].flip(-1)
                # Padding uses the standardized training mean (zero), not wrapped pixels.
                padded = torch.nn.functional.pad(x, (2, 2, 2, 2))
                dy, dx = torch.randint(0, 5, (2,)).tolist()
                x = padded[:, :, dy:dy + x.shape[-2], dx:dx + x.shape[-1]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise TrainingInstability(history, epoch, seen)
            loss.backward()
            norm = torch.sqrt(sum((p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))
            norm_values.append(float(norm))
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
        train_result = evaluate(model, data, split='train', device=device, prepare=prepare)
        val_result = evaluate(model, data, split='validation', device=device, prepare=prepare)
        row = {
            'epoch': epoch,
            'training_loss': total_loss / seen,
            'train_eval_loss': train_result['loss'],
            'validation_loss': val_result['loss'],
            'train_accuracy': train_result['accuracy'],
            'validation_accuracy': val_result['accuracy'],
            'mean_gradient_norm': float(np.mean(norm_values)),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'examples_processed': seen,
            'activation': activation_summary(model, data['x_validation'], device=device),
        }
        history.append(row)
        file = checkpoints / f'epoch{epoch:02d}.pt'
        torch.save(
            {
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'epoch': epoch,
                'validation_accuracy': val_result['accuracy'],
                'split_fingerprint': data.get('split_fingerprint', ''),
            },
            file,
        )
        paths.append(str(file))
        if val_result['accuracy'] > best:
            best = val_result['accuracy']
            stale = 0
        else:
            stale += 1
        if scheduler:
            scheduler.step()
        if patience is not None and stale >= patience:
            break
    values = [r['validation_accuracy'] for r in history]
    selected = checkpoint_selector(values, paths) if checkpoint_selector else paths[int(np.argmax(values))]
    if selected not in paths or not Path(selected).is_file():
        raise ValueError('Checkpoint selector must return one of the saved epoch paths')
    saved = torch.load(selected, map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    restored = evaluate(model, data, split='validation', device=device, prepare=prepare)
    if abs(restored['accuracy'] - values[paths.index(selected)]) > 1e-07:
        raise RuntimeError('Reloaded checkpoint does not reproduce its validation score')
    save_json(output / 'history.json', history)
    save_json(
        output / 'selection.json',
        {
            'selected_file': str(Path(selected).relative_to(output)),
            'epoch': saved['epoch'],
            'validation_accuracy': restored['accuracy'],
            'elapsed_seconds': time.perf_counter() - start,
            'test_used_for_selection': False,
        },
    )
    save_predictions(
        output / 'validation_predictions.csv',
        data['validation_ids'],
        data['y_validation'],
        restored['predictions'],
        restored['probabilities'],
    )
    return {
        'model': model,
        'history': history,
        'selected_path': Path(selected),
        'selected_epoch': int(saved['epoch']),
        'validation': restored,
        'checkpoint_paths': paths,
    }
