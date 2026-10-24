"""Supplied loaders: plain dictionaries of arrays and names, not a new dataset API.

Only CIFAR-10 uses torchvision. Its validation split comes from the official
training set. The official test set is held out from model selection.
"""
from __future__ import annotations
import hashlib
import io
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .common import digest
UCI = {
    'raisin': ('https://archive.ics.uci.edu/static/public/850/raisin.zip', 7),
    'banknote': ('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', 4),
    'dry_bean': ('https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip', 16),
}
PENGUINS = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'

def cache_dir() -> Path:
    root = Path(os.environ.get('AI702_DATA_DIR', Path.home() / '.cache' / 'ai702'))
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

def download(url: str, filename: str, *, maximum_bytes: int=15000000) -> Path:
    """Download once over HTTPS. Cache hashes detect later corruption, not provenance.

    The hash is recorded on the first successful download. Staff may distribute a
    verified cache for offline labs. No synthetic substitute is silently loaded.
    """
    target = cache_dir() / filename
    receipt = target.with_suffix(target.suffix + '.sha256')
    if target.is_file():
        if not receipt.is_file() or digest(target) != receipt.read_text().strip():
            raise RuntimeError(f'Cache checksum missing or changed: {target}. Re-download a trusted copy.')
        return target
    request = urllib.request.Request(url, headers={'User-Agent': 'AI702-teaching-labs/0.5'})
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as handle:
            total = 0
            while (chunk := response.read(65536)):
                total += len(chunk)
                if total > maximum_bytes:
                    raise RuntimeError('Downloaded file is unexpectedly large')
                handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError('Downloaded file is empty')
        temporary.replace(target)
        receipt.write_text(digest(target) + '\n')
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Cannot download {url}. Prepare the cache before the lab, set AI702_DATA_DIR to an existing cache, or use --smoke for an explicitly synthetic diagnostic run. {exc}') from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target

def _indices(y, seed, train_size=None, val_size=None, test_size=None):
    ids = np.arange(len(y), dtype=np.int64)
    trainval, test = train_test_split(ids, test_size=0.2, random_state=seed, stratify=y)
    train, val = train_test_split(trainval, test_size=0.2, random_state=seed + 1, stratify=y[trainval])
    return (
        _limit(train, y, train_size, seed + 2),
        _limit(val, y, val_size, seed + 3),
        _limit(test, y, test_size, seed + 4),
    )

def _limit(ids, y, size, seed):
    ids = np.asarray(ids, dtype=np.int64)
    if size is None or size >= len(ids):
        return ids
    if size < len(np.unique(y[ids])):
        raise ValueError('Subset size must allow every class to be represented')
    selected, _ = train_test_split(ids, train_size=int(size), stratify=y[ids], random_state=seed)
    return selected

def prepare(x, y, names, classes, *, seed=702, name='dataset', ids=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or (not np.isfinite(x).all()):
        raise ValueError('Expected finite feature matrix and one label per row')
    train, val, test = _indices(y, seed)
    scaler = StandardScaler().fit(x[train])
    identifiers = np.arange(len(y)) if ids is None else np.asarray(ids)
    result = {
        'name': name,
        'feature_names': list(names),
        'class_names': list(classes),
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'split_seed': seed,
        'synthetic': False,
    }
    for label, idx in [('train', train), ('validation', val), ('test', test)]:
        result[f'x_{label}'] = scaler.transform(x[idx])
        result[f'y_{label}'] = y[idx]
        result[f'{label}_ids'] = identifiers[idx]
    return result

def tabular(name: str, *, seed=702, smoke=False) -> dict:
    builtins = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'digits': datasets.load_digits,
        'breast_cancer': datasets.load_breast_cancer,
    }
    if name in builtins:
        d = builtins[name]()
        return prepare(
            d.data,
            d.target,
            list(d.feature_names),
            [str(x) for x in d.target_names],
            seed=seed,
            name=name,
        )
    if smoke:
        dimensions = {'raisin': 7, 'banknote': 4, 'dry_bean': 16, 'penguins': 4}
        classes = 7 if name == 'dry_bean' else 3 if name == 'penguins' else 2
        x, y = datasets.make_classification(
            n_samples=280,
            n_features=dimensions[name],
            n_informative=min(dimensions[name], 5),
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=classes,
            random_state=seed,
        )
        d = prepare(
            x,
            y,
            [f'fixture_feature_{i}' for i in range(x.shape[1])],
            [str(i) for i in range(classes)],
            seed=seed,
            name=f'SYNTHETIC {name}-shaped fixture',
        )
        d['synthetic'] = True
        return d
    if name == 'penguins':
        frame = pd.read_csv(download(PENGUINS, 'penguins.csv'))
        columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        cleaned = frame.dropna(subset=[*columns, 'species'])
        labels, classes = pd.factorize(cleaned['species'], sort=True)
        return prepare(
            cleaned[columns],
            labels,
            columns,
            classes,
            seed=seed,
            name='Palmer Penguins',
            ids=cleaned.index,
        )
    if name not in UCI:
        raise ValueError(f'Unknown dataset: {name}')
    url, n_features = UCI[name]
    frame = read_uci_archive(download(url, f'{name}.zip'), name)
    frame = frame.dropna()
    labels, classes = pd.factorize(frame.iloc[:, -1].astype(str), sort=True)
    x = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    if x.shape[1] != n_features:
        raise RuntimeError(f'{name}: expected {n_features} features, received {x.shape[1]}')
    return prepare(x, labels, frame.columns[:-1], classes, seed=seed, name=f'UCI {name}', ids=frame.index)

def read_uci_archive(path: Path, name: str) -> pd.DataFrame:
    """Read files in memory; never extract an untrusted ZIP into the filesystem."""
    candidates = []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.file_size > 30000000:
                raise RuntimeError('Unexpectedly large UCI archive member')
            suffix = Path(item.filename).suffix.lower()
            if suffix in {'.xlsx', '.csv', '.txt'}:
                candidates.append((item.filename, archive.read(item)))
            elif suffix == '.zip':
                with zipfile.ZipFile(io.BytesIO(archive.read(item))) as nested:
                    for member in nested.infolist():
                        if Path(member.filename).suffix.lower() in {'.xlsx', '.csv', '.txt'} and member.file_size <= 30000000:
                            candidates.append((member.filename, nested.read(member)))
    if name == 'banknote':
        matches = [(n, b) for n, b in candidates if 'data_banknote' in n.lower()]
        if not matches:
            raise RuntimeError('Banknote data file not found in archive')
        return pd.read_csv(
            io.BytesIO(matches[0][1]),
            header=None,
            names=['variance', 'skewness', 'curtosis', 'entropy', 'class'],
        )
    matches = [(n, b) for n, b in candidates if n.lower().endswith('.xlsx') and (not Path(n).name.startswith('~'))]
    if not matches:
        raise RuntimeError(f'{name} spreadsheet not found in archive')
    return pd.read_excel(io.BytesIO(matches[0][1]), engine='openpyxl')

def binary_view(data: dict, classes=(0, 1), columns=None, *, one_vs_rest=False) -> dict:
    out = {k: v for k, v in data.items() if not k.startswith(('x_', 'y_')) and (not k.endswith('_ids'))}
    columns = list(range(data['x_train'].shape[1])) if columns is None else list(columns)
    out['feature_names'] = [data['feature_names'][i] for i in columns]
    out['class_names'] = ['rest', str(data['class_names'][classes[0]])] if one_vs_rest else [str(data['class_names'][c]) for c in classes]
    for split in ('train', 'validation', 'test'):
        y = data[f'y_{split}']
        mask = np.ones(len(y), bool) if one_vs_rest else np.isin(y, classes)
        out[f'x_{split}'] = data[f'x_{split}'][mask][:, columns]
        out[f'y_{split}'] = (y[mask] == (classes[0] if one_vs_rest else classes[1])).astype(np.int64)
        out[f'{split}_ids'] = data[f'{split}_ids'][mask]
    return out

def cifar(*, seed=702, train_size=1600, validation_size=400, test_size=400, smoke=False) -> dict:
    """Identical Lab 6/7 split. Select rows before converting full images to float."""
    if smoke:
        rng = np.random.default_rng(seed)
        # Ten classes, RGB and 32x32 are retained, so CNN shape tests match real data.
        y_train = np.tile(np.arange(10), 16)
        y_test = np.tile(np.arange(10), 4)
        train_images = rng.integers(0, 256, (160, 32, 32, 3), dtype=np.uint8)
        test_images = rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8)
        train_size, validation_size, test_size = (100, 40, 30)
        names = [str(i) for i in range(10)]
    else:
        from torchvision.datasets import CIFAR10
        try:
            train_data = CIFAR10(str(cache_dir() / 'cifar10'), train=True, download=True)
            test_data = CIFAR10(str(cache_dir() / 'cifar10'), train=False, download=True)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            raise RuntimeError('CIFAR-10 unavailable. Download it once before the lab using `python lab.py data cifar10`, or run --smoke as an explicitly synthetic diagnostic.') from exc
        train_images = np.asarray(train_data.data)
        test_images = np.asarray(test_data.data)
        y_train = np.asarray(train_data.targets, dtype=np.int64)
        y_test = np.asarray(test_data.targets, dtype=np.int64)
        names = list(train_data.classes)
    all_ids = np.arange(len(y_train))
    train_pool, val_pool = train_test_split(all_ids, test_size=0.2, random_state=seed, stratify=y_train)
    tr = _limit(train_pool, y_train, train_size, seed + 1)
    va = _limit(val_pool, y_train, validation_size, seed + 2)
    te = _limit(np.arange(len(y_test)), y_test, test_size, seed + 3)

    def images(raw, ids):
        return np.transpose(raw[ids].astype(np.float32) / 255.0, (0, 3, 1, 2)).copy()
    xtr = images(train_images, tr)
    mean = xtr.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(xtr.std(axis=(0, 2, 3), keepdims=True), 1e-06)
    result = {
        'name': 'SYNTHETIC CIFAR-shaped fixture' if smoke else 'CIFAR-10',
        'synthetic': smoke,
        'class_names': names,
        'mean': mean,
        'scale': std,
        'split_seed': seed,
    }
    for label, x, y, ids, prefix in [
        ('train', xtr, y_train[tr], tr, 'train'),
        ('validation', images(train_images, va), y_train[va], va, 'train'),
        ('test', images(test_images, te), y_test[te], te, 'test'),
    ]:
        result[f'x_{label}'] = (x - mean) / std
        result[f'y_{label}'] = y
        result[f'{label}_ids'] = np.asarray([f'cifar-{prefix}-{i}' for i in ids])
    h = hashlib.sha256()
    for key in ('train_ids', 'validation_ids', 'test_ids', 'mean', 'scale'):
        value = result[key]
        h.update('\n'.join(value.tolist()).encode() if value.dtype.kind in 'US' else value.tobytes())
    result['split_fingerprint'] = h.hexdigest()
    return result

def image_change(x, condition, *, data=None, seed=702, shift=4):
    """Apply deterministic evaluation changes. Translation uses padding, not wraparound."""
    out = np.asarray(x, dtype=np.float32).copy()
    if condition == 'none':
        return out
    if condition == 'translation':
        if abs(shift) >= out.shape[-1]:
            raise ValueError('Shift must be smaller than image width')
        result = np.zeros_like(out)
        if shift > 0:
            result[:, :, :, shift:] = out[:, :, :, :-shift]
        elif shift < 0:
            result[:, :, :, :shift] = out[:, :, :, -shift:]
        else:
            result[:] = out
        return result
    if condition == 'occlusion':
        width = out.shape[-1] // 4
        start = out.shape[-1] // 2 - width // 2
        out[:, :, start:start + width, start:start + width] = 0
        return out
    if condition == 'noise':
        return out + np.random.default_rng(seed).normal(0, 0.25, out.shape).astype(np.float32)
    if condition == 'brightness':
        return out + 0.5
    if condition == 'contrast':
        return out * 0.5
    raise ValueError(f'Unknown image condition: {condition}')
