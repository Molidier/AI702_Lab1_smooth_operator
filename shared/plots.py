"""Supplied Matplotlib functions. Every function saves one self-contained figure."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def curves(path, series, *, title, xlabel='Step', ylabel='Value', x=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, values in series.items():
        values = np.asarray(values, float)
        positions = np.arange(len(values)) if x is None else np.asarray(x)[:len(values)]
        ax.plot(positions, values, label=str(label))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if len(series):
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def matrix(path, values, title, *, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(np.asarray(values), aspect='auto')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def boundary(path, x, y, predict, *, names=None, title='Decision region', standardization=None):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Boundary plots need exactly two features')
    if standardization is not None:
        mean, scale = standardization
        mean, scale = (np.asarray(mean), np.asarray(scale))
        x = x * scale + mean
        supplied_predict = predict
        predict = lambda raw: supplied_predict((raw - mean) / scale)
    padding = 0.15 * np.maximum(np.ptp(x, axis=0), 1)
    low, high = (x.min(0) - padding, x.max(0) + padding)
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 150), np.linspace(low[1], high[1], 150))
    zz = np.asarray(predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.contourf(xx, yy, zz, levels=np.arange(np.min(zz) - 0.5, np.max(zz) + 1.5), alpha=0.25)
    for label in np.unique(y):
        mask = np.asarray(y) == label
        ax.scatter(x[mask, 0], x[mask, 1], s=18, label=str(label))
    ax.set(
        title=title,
        xlabel=(names or ['Feature 1', 'Feature 2'])[0],
        ylabel=(names or ['Feature 1', 'Feature 2'])[1],
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def boundary_history(path, x, y, weights, biases, *, names, standardization=None):
    """Optional slider: uses saved parameters, no browser server or CDN."""
    frames = path.parent / 'boundary_frames'
    frames.mkdir(exist_ok=True)
    chosen = np.unique(np.linspace(0, len(weights) - 1, min(12, len(weights)), dtype=int))
    names_out = []
    for i in chosen:
        name = f'epoch_{i:03d}.png'
        w, b = (np.asarray(weights[i]), biases[i])
        boundary(frames / name, x, y, lambda z, w=w, b=b: np.where(z @ w + b >= 0, 1, -1), names=names, title=f'Parameters after epoch {i}', standardization=standardization)
        names_out.append('boundary_frames/' + name)
    path.write_text(
        '<!doctype html><meta charset="utf-8"><title>Boundary history</title>\n<h1>Perceptron boundary history</h1><input id="slider" type="range" min="0" value="0"><p id="label"></p><img id="frame" style="max-width:95%">\n<script>const files=' + json.dumps(names_out) + '; const epochs=' + json.dumps(chosen.tolist()) + ";\nconst s=document.getElementById('slider');s.max=files.length-1;\nfunction show(){document.getElementById('frame').src=files[Number(s.value)];document.getElementById('label').textContent='Epoch '+epochs[Number(s.value)];}\ns.oninput=show;show();</script>",
        encoding='utf-8',
    )

def confusion(path, targets, predictions, classes):
    values = confusion_matrix(targets, predictions, labels=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(values)
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel='Predicted',
        ylabel='Actual',
        title='Confusion matrix',
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return values

def trajectories(path, runs, *, curvature=(1.0, 20.0), title='Optimizer paths'):
    fig, ax = plt.subplots(figsize=(7, 5))
    gx, gy = np.meshgrid(np.linspace(-5, 5, 180), np.linspace(-3, 3, 150))
    z = 0.5 * (curvature[0] * gx ** 2 + curvature[1] * gy ** 2)
    ax.contour(gx, gy, z, levels=18, alpha=0.35)
    for name, points in runs.items():
        points = np.asarray(points)
        ax.plot(points[:, 0], points[:, 1], '.-', label=name, markersize=3)
    ax.set(xlim=(-5, 5), ylim=(-3, 3), xlabel='Parameter 1', ylabel='Parameter 2', title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def image(path, values, *, title):
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1:] == (1,):
        array = array[..., 0]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(array)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

def error_examples(folder, x, y, pred, ids, *, classes, mean=None, scale=None, limit=4):
    indices = np.flatnonzero(np.asarray(y) != np.asarray(pred))[:limit]
    rows = []
    for number, i in enumerate(indices):
        values = np.asarray(x[i]).copy()
        if values.ndim == 3 and mean is not None:
            values = values * scale[0] + mean[0]
        if values.ndim == 1 and values.size == 64:
            values = values.reshape(8, 8)
        if values.ndim == 3:
            values = np.clip(values, 0, 1)
        name = f'error_{number}.png'
        image(folder / name, values, title=f'{ids[i]}: actual {classes[y[i]]}, model {classes[pred[i]]}')
        rows.append({
            'sample_id': str(ids[i]),
            'target': int(y[i]),
            'prediction': int(pred[i]),
            'figure': name,
        })
    return rows
