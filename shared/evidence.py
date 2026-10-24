"""Strict structural evidence checks. Local evidence never establishes authorship."""
from __future__ import annotations
import base64
import csv
import math
from pathlib import Path
import numpy as np
from .common import load_json, digest
from .integrity import checked_relative, public_key

COMMON = {'metrics.json', 'details.json', 'settings_used.json', 'notes_at_start.md', 'index.html'}
REQUIRED = {
    0: {'raw.npz'},
    1: {'raw.npz'},
    2: {'raw.npz'},
    3: {'binary_raw.npz', 'cross_entropy_raw.npz', 'l2_raw.npz'},
    4: {'tiny_forward_backward.npz', 'gradient_checks.json', 'training.npz'},
    5: {'surface_runs.json', 'gradient_samples.npz', 'optimizer_runs.json', 'batch_budget_comparison.json', 'ordering_run.json'},
    6: {'normalization_dropout.npz', 'evaluation.npz', 'split.json', 'baseline/selection.json', 'intervention/selection.json', 'baseline/training_condition.json', 'intervention/training_condition.json', 'baseline_for_lab7.pt'},
    7: {'scan_and_features.npz', 'receptive_field.json', 'split.json', 'cnn/selection.json', 'mlp_fallback/selection.json', 'mlp_fallback/comparison_protocol.json', 'shift_comparison.json'},
}


def predictions(lab, cfg):
    """file -> (metric path, accuracy or error, label view)."""
    if lab == 0:
        return {'validation_predictions.csv': ('student_recomputed_accuracy', 'accuracy', 'default')}
    if lab == 1:
        return {f'{name}_predictions.csv': (f'{name}_validation_accuracy', 'accuracy', 'binary') for name in ('linear', 'mlp')}
    if lab == 2:
        return {'validation_predictions.csv': ('validation_error_2_features', 'error', 'signed'),
                'all_features_predictions.csv': ('validation_error_all_features', 'error', 'signed')}
    if lab == 3:
        return {'binary_predictions.csv': ('binary_validation_accuracy', 'accuracy', 'binary'),
                **{f'{name}_predictions.csv': (f'{name}_validation_error', 'error', 'multiclass') for name in ('cross_entropy', 'l2')}}
    if lab == 4:
        return {'validation_predictions.csv': ('wine_validation_accuracy', 'accuracy', 'default')}
    if lab == 5:
        return {f'{name}_predictions.csv': (f'validation_accuracy.{name}', 'accuracy', 'default') for name in ('sgd', 'momentum', 'nesterov', cfg['adaptive'], 'decay')}
    if lab == 6:
        return {'validation_predictions.csv': ('selected_validation_accuracy', 'accuracy', 'default'),
                'shifted_validation_predictions.csv': ('shifted_validation_accuracy', 'accuracy', 'default'),
                'baseline/validation_predictions.csv': (None, 'accuracy', 'default'),
                'intervention/validation_predictions.csv': ('validation_accuracy', 'accuracy', 'default')}
    if lab == 7:
        tables = {f'{model.lower()}_{condition}_predictions.csv': (f'comparison.{model}.{condition}.accuracy', 'accuracy', 'default') for model in ('CNN', 'MLP') for condition in ('none', 'translation', 'occlusion')}
        tables.update({f'{model.lower()}_shift{shift}_predictions.csv': (None, 'accuracy', 'default') for model in ('CNN', 'MLP') for shift in sorted({0, 2, cfg['translation']})})
        tables.update({f'{folder}/validation_predictions.csv': (None, 'accuracy', 'default') for folder in ('cnn', 'mlp_fallback')})
        return tables
    raise ValueError('Unknown lab')


def metric_value(metrics, key):
    value = metrics
    for component in key.split('.'):
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid numerical metric: {key}')
    return value


def read_predictions(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'sample_id', 'target', 'prediction'}.issubset(fields):
            raise ValueError('Prediction CSV has invalid columns')
        rows = list(reader)
    if not rows or len(rows) > 100000 or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Prediction rows must be nonempty and have unique sample IDs')
    ids = [r['sample_id'] for r in rows]
    y = np.array([int(r['target']) for r in rows])
    pred = np.array([int(r['prediction']) for r in rows])
    scores = [f for f in fields if f.startswith('score_')]
    if scores:
        if scores != [f'score_{i}' for i in range(len(scores))]:
            raise ValueError('Invalid score columns')
        values = np.array([[float(r[f]) for f in scores] for r in rows])
        if (not np.isfinite(values).all() or np.any(pred < 0) or np.any(pred >= len(scores))
                or not np.all(values[np.arange(len(pred)), pred] == values.max(1))):
            raise ValueError('Nonfinite scores or predictions inconsistent with scores')
    return ids, y, pred


def check_evidence(path):
    errors = []
    try:
        # Check even ancestors named results, before reading anything.
        if any(p.is_symlink() for p in (path, *path.parents)):
            return ['Evidence path contains a symbolic link']
        record = load_json(path / 'run.json')
        lab, stage = record.get('lab'), record.get('stage')
        if record.get('schema_version') != 2 or type(lab) is not int or lab not in range(8) or stage not in ('core', 'transfer'):
            return ['Unsupported or incomplete assessed evidence schema']
        if not __import__('re').fullmatch(r'[0-9]{8}T[0-9]{6}-[0-9a-f]{8}', record.get('run_id', '')):
            errors.append('Missing or invalid run identifier')
        if any(type(record.get(k)) is not bool for k in ('smoke', 'reference', 'eligible_for_submission')):
            errors.append('Missing or invalid eligibility flags')
        elif record['eligible_for_submission'] != (not record['smoke'] and not record['reference']):
            errors.append('Contradictory eligibility flags')
        if not isinstance(record.get('dataset'), str) or not record['dataset']:
            errors.append('Missing dataset identity')
        if str(record.get('dataset', '')).startswith('SYNTHETIC') and not record.get('smoke'):
            errors.append('Synthetic dataset marked as real data')
        hashes = record.get('artifact_hashes', {})
        if not isinstance(hashes, dict) or not hashes:
            return errors + ['Missing artifact inventory']
        actual = set()
        for entry in path.rglob('*'):
            checked_relative(path, entry.relative_to(path).as_posix())
            if entry.is_file() and entry.relative_to(path).as_posix() != 'run.json':
                actual.add(entry.relative_to(path).as_posix())
        if set(hashes) != actual:
            errors.append('Artifact inventory is incomplete or contains nonexistent files')
        for name, expected in hashes.items():
            file = checked_relative(path, name)
            if not file.is_file() or file.stat().st_size > 256000000 or digest(file) != expected:
                errors.append(f'Changed or missing artifact: {name}')
        cfg = load_json(path / 'settings_used.json')
        metrics = load_json(path / 'metrics.json')
        tables = predictions(lab, cfg)
        required = COMMON | REQUIRED[lab] | set(tables)
        if stage == 'transfer':
            required |= {f'source/lab{lab}/transfer.json', f'source/lab{lab}/transfer.sig'}
            if lab == 4:
                required |= {'source/lab4/repair.py', 'transfer_diagnosis.json'}
        sources = record.get('source_hashes', {})
        code = f'staff/solutions/lab{lab}.py' if record.get('reference') else f'lab{lab}/student.py'
        required_sources = {code, f'lab{lab}/settings.json'}
        if stage == 'transfer':
            required_sources |= {f'lab{lab}/transfer.json', f'lab{lab}/transfer.sig'}
            if lab == 4:
                required_sources.add('lab4/repair.py')
        if set(sources) != required_sources:
            errors.append('Source inventory differs from required stage files')
        for name, expected in sources.items():
            file = checked_relative(path, 'source/' + name)
            required.add('source/' + name)
            if not file.is_file() or digest(file) != expected:
                errors.append(f'Source snapshot mismatch: {name}')
        if missing := required - set(hashes):
            errors.append('Missing required artifacts: ' + ', '.join(sorted(missing)))
        for name, (key, kind, _) in tables.items():
            ids, y, pred = read_predictions(path / name)
            accuracy = float(np.mean(y == pred))
            if key and abs(metric_value(metrics, key) - (1 - accuracy if kind == 'error' else accuracy)) > 1e-7:
                errors.append(f'Metric disagrees with predictions: {key}')
            if lab == 7 and key:
                counts = [float(np.mean(pred[y == c] == c)) if np.any(y == c) else None for c in range(10)]
                per_class = metrics['comparison'][key.split('.')[1]][key.split('.')[2]]['per_class_accuracy']
                if len(per_class) != len(counts) or any((a is None) != (b is None) or (a is not None and abs(a-b)>1e-7) for a,b in zip(per_class,counts)):
                    errors.append(f'Per-class accuracy mismatch: {name}')
            if lab == 0:
                # The one intentional inconsistency is specified, not a general exception.
                if abs(metric_value(metrics, 'reported_accuracy') - 100 * accuracy) > 1e-7 or metrics.get('reported_units') != 'fraction between 0 and 1':
                    errors.append('Lab 0 supplied unit exercise was altered')
        for folder in ({6: ('baseline', 'intervention'), 7: ('cnn', 'mlp_fallback')}.get(lab, ())):
            selection = load_json(path / folder / 'selection.json')
            history = load_json(path / folder / 'history.json')
            best = int(np.argmax([r['validation_accuracy'] for r in history])) + 1
            checkpoint = checked_relative(path, folder + '/' + selection['selected_file'])
            if checkpoint.relative_to(path).as_posix() not in hashes or selection['epoch'] != best:
                errors.append(f'Invalid selected checkpoint: {folder}')
            accuracy = float(np.mean(np.equal(*read_predictions(path / folder / 'validation_predictions.csv')[1:])))
            if abs(selection['validation_accuracy'] - accuracy) > 1e-7 or abs(history[best-1]['validation_accuracy'] - accuracy) > 1e-7:
                errors.append(f'Checkpoint selection metric mismatch: {folder}')
        if lab == 5:
            runs = load_json(path / 'optimizer_runs.json')
            budget = 5 if record['smoke'] else cfg['updates']
            for name in ('sgd', 'decay', 'momentum', 'nesterov', cfg['adaptive']):
                history = runs[name]['history']
                if len(history) != budget or runs[name]['updates'] != budget:
                    errors.append(f'Optimizer budget mismatch: {name}')
                if any(r.get('measurement_point') != ('lookahead' if name == 'nesterov' else 'before_update') for r in history):
                    errors.append(f'Optimizer measurement point missing: {name}')
        if lab == 6:
            selected = load_json(path / 'intervention/selection.json')['epoch']
            history = load_json(path / 'intervention/history.json')
            expected_gap = history[selected-1]['train_accuracy'] - metrics['validation_accuracy']
            if abs(metric_value(metrics, 'selected_epoch_generalization_gap') - expected_gap) > 1e-7:
                errors.append('Generalization gap inconsistent with selected checkpoint')
        if lab == 7:
            protocol = load_json(path / 'mlp_fallback/comparison_protocol.json')
            expected_cfg = {k: cfg[k] for k in ('seed', 'train_size', 'validation_size', 'test_size', 'device', 'epochs', 'batch_size', 'learning_rate')}
            expected_cfg.update(hidden_units=96, intervention='dropout', shift_condition='brightness')
            if protocol['training_config'] != expected_cfg:
                errors.append('MLP comparison training protocol differs from CNN assignment')
            shifts = load_json(path / 'shift_comparison.json')
            if shifts['shifts'] != sorted({0, 2, cfg['translation']}):
                errors.append('Wrong translation conditions')
            for model in ('MLP', 'CNN'):
                for i, shift in enumerate(shifts['shifts']):
                    _, y, pred = read_predictions(path / f'{model.lower()}_shift{shift}_predictions.csv')
                    if abs(shifts['accuracy'][model][i] - float(np.mean(y == pred))) > 1e-7:
                        errors.append('Translation metric differs from raw predictions')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(f'Invalid or missing evidence: {exc}')
    return errors


def verify_snapshot_transfer(snapshot, lab, release_root, trusted_key=None):
    path = checked_relative(snapshot, f'lab{lab}/transfer.json')
    raw = path.read_bytes()
    public_key(release_root, trusted_key).verify(base64.b64decode(path.with_suffix('.sig').read_bytes(), validate=True), raw)
    transfer = load_json(path)
    release = load_json(release_root / '.integrity/manifest.json')
    if transfer.get('course') != 'AI702' or transfer.get('lab') != lab or transfer.get('release') != release['release']:
        raise ValueError('Snapshot transfer belongs to another lab/release')
    return transfer


def trusted_labels(path, lab, cfg):
    """Rebuild IDs and labels from a staff-controlled cache. Never import submitted code."""
    from .data import tabular, binary_view, cifar
    name = {0: 'breast_cancer', 1: cfg.get('dataset'), 2: cfg.get('dataset'), 3: 'digits', 4: 'wine', 5: 'dry_bean', 6: 'CIFAR-10', 7: 'CIFAR-10'}[lab]
    d = cifar(**{k: cfg[k] for k in ('seed', 'train_size', 'validation_size', 'test_size')}) if lab >= 6 else tabular(name, seed=cfg['seed'])
    errors = []
    record = load_json(path / 'run.json')
    if record['dataset'] != d['name']:
        errors.append('Dataset identity differs from trusted dataset')
    for file, (_, _, view) in predictions(lab, cfg).items():
        source = d
        if view == 'binary':
            source = binary_view(d, classes=cfg['classes'] if lab == 1 else cfg['binary_classes'])
        ids, targets = source['validation_ids'], source['y_validation']
        if view == 'signed':
            targets = 2 * targets - 1
        if view == 'multiclass':
            classes = cfg['multiclass_classes']
            mask = np.isin(targets, classes)
            ids, targets = ids[mask], np.array([classes.index(int(y)) for y in targets[mask]])
        actual_ids, actual_targets, predicted = read_predictions(path / file)
        if actual_ids != [str(i) for i in ids] or not np.array_equal(actual_targets, targets):
            errors.append(f'Sample IDs/labels differ from trusted split: {file}')
        valid_labels = [-1, 1] if view == 'signed' else list(range(2 if view == 'binary' else len(cfg['multiclass_classes']) if view == 'multiclass' else len(d['class_names'])))
        if not np.isin(predicted, valid_labels).all():
            errors.append(f'Predicted class outside assigned task: {file}')
    if lab >= 6:
        split = load_json(path / 'split.json')
        for field in ('train_ids', 'validation_ids', 'test_ids'):
            if split.get(field) != d[field].tolist():
                errors.append(f'Trusted split mismatch: {field}')
        if split.get('fingerprint') != d['split_fingerprint']:
            errors.append('Split fingerprint mismatch')
    return errors
