"""Plain file and reporting helpers shared by the lab runners."""
from __future__ import annotations
import csv
import hashlib
import html
import importlib.util
import json
import os
import platform
import subprocess
import sys
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)

def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))

def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def student_module(number: int, reference: bool=False):
    path = ROOT / (f'staff/solutions/lab{number}.py' if reference else f'lab{number}/student.py')
    if not path.is_file():
        raise RuntimeError('Staff references are not included in student releases.')
    return load_module(path, f'ai702_lab{number}_' + ('reference' if reference else 'student'))

def settings(number: int, overrides: dict | None=None) -> dict:
    cfg = load_json(ROOT / f'lab{number}/settings.json')
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        cfg.update(overrides)
    return cfg

def git_state(root: Path=ROOT) -> dict:

    def run(*args):
        p = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain'))}

def save_predictions(path: Path, sample_ids, targets, predictions, scores=None) -> None:
    if not len(sample_ids) == len(targets) == len(predictions):
        raise ValueError('Prediction fields have different lengths')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        columns = ['sample_id', 'target', 'prediction']
        if scores is not None:
            columns += [f'score_{i}' for i in range(np.asarray(scores).shape[1])]
        writer.writerow(columns)
        for i in range(len(targets)):
            row = [sample_ids[i], int(targets[i]), int(predictions[i])]
            if scores is not None:
                row += list(np.asarray(scores)[i].astype(float))
            writer.writerow(row)

class Run:
    """One experiment output folder, with a source snapshot and a static HTML report."""

    def __init__(self, number: int, *, smoke=False, reference=False, stage='core', config=None):
        self.number = number
        self.smoke = bool(smoke)
        self.reference = bool(reference)
        self.stage = stage
        self.config = config or {}
        from .integrity import checked_relative
        if stage not in ('core', 'transfer', 'baseline', 'extension'):
            raise ValueError('Unknown run stage')
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
        self.path = checked_relative(ROOT, f'lab{number}/results/{stage}/{self.run_id}')
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = datetime.now(timezone.utc).isoformat()
        paths = [ROOT / f'lab{number}/student.py', ROOT / f'lab{number}/settings.json']
        if stage == 'transfer':
            paths.extend((ROOT / f'lab{number}' / name for name in ('transfer.json', 'transfer.sig', 'repair.py') if (ROOT / f'lab{number}' / name).is_file()))
        if stage in ('baseline', 'extension'):
            paths[0] = ROOT / f'lab{number}' / ('extension.py' if stage == 'extension' else 'run.py')
        elif reference:
            paths[0] = ROOT / f'staff/solutions/lab{number}.py'
        self.source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
        for source in paths:
            destination = self.path / 'source' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        save_json(self.path / 'settings_used.json', self.config)
        # Record the expectation as it existed before execution, without claiming a trusted timestamp.
        notes = ROOT / f'lab{number}/notes.md'
        if notes.is_file():
            (self.path / 'notes_at_start.md').write_text(notes.read_text(encoding='utf-8'), encoding='utf-8')

    def json(self, name, value):
        save_json(self.path / name, value)

    def arrays(self, name, **values):
        np.savez_compressed(self.path / name, **values)

    def finish(self, metrics: dict, *, dataset: str, details: dict | None=None) -> Path:
        self.json('metrics.json', metrics)
        if details is not None:
            self.json('details.json', details)
        title = f'AI702 Lab {self.number}: {self.stage}'
        files = sorted((p for p in self.path.rglob('*') if p.is_file()))
        images = '\n'.join((f'<figure><img src="{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in files if p.suffix == '.png' and p.parent == self.path))
        links = '<li><a href="run.json">run.json</a></li>\n' + '\n'.join((f'<li><a href="{html.escape(p.relative_to(self.path).as_posix())}">{html.escape(p.relative_to(self.path).as_posix())}</a></li>' for p in files))
        header = 'SUPPLIED EXAMPLE' if self.stage in ('baseline', 'extension') else 'STAFF REFERENCE' if self.reference else 'STUDENT IMPLEMENTATION'
        mode = 'SMOKE FIXTURES: not a graded dataset result' if self.smoke else dataset
        content = f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>\n<style>body{{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:1rem}}figure{{margin:1.5rem 0}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap}}a{{overflow-wrap:anywhere}}</style></head>\n<body><h1>{title}</h1><p>{header} | {html.escape(mode)}</p>\n<h2>Results</h2><pre>{html.escape(json.dumps(json_value(metrics), indent=2))}</pre>\n{images}<h2>Raw evidence</h2><ul>{links}</ul></body></html>'
        (self.path / 'index.html').write_text(content, encoding='utf-8')
        artifact_hashes = {str(p.relative_to(self.path)): digest(p) for p in self.path.rglob('*') if p.is_file() and p.name != 'run.json'}
        self.json(
            'run.json',
            {
                'schema_version': 2,
                'run_id': self.run_id,
                'lab': self.number,
                'stage': self.stage,
                'started_utc': self.started,
                'dataset': dataset,
                'smoke': self.smoke,
                'reference': self.reference,
                'eligible_for_submission': self.stage in ('core', 'transfer') and not (self.smoke or self.reference),
                'python': platform.python_version(),
                'git': git_state(),
                'source_hashes': self.source_hashes,
                'artifact_hashes': artifact_hashes,
                'note': 'Local evidence hashes detect changed files; they do not certify authorship or execution on a trusted machine.',
            },
        )
        save_json(self.path.parent / 'latest.json', {'run_id': self.run_id})
        return self.path / 'index.html'

def result_path(number: int, stage: str, run_id=None, *, root=None) -> Path:
    from .integrity import checked_relative
    root = ROOT if root is None else root
    base = checked_relative(root, f'lab{number}/results/{stage}')
    run_id = run_id or load_json(base / 'latest.json')['run_id']
    if not isinstance(run_id, str) or not __import__('re').fullmatch(r'[0-9]{8}T[0-9]{6}-[0-9a-f]{8}', run_id):
        raise ValueError('Invalid run identifier')
    return checked_relative(root, f'lab{number}/results/{stage}/{run_id}')


def check_evidence(path: Path) -> list[str]:
    from .evidence import check_evidence as check
    return check(path)
