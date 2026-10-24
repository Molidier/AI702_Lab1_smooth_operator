"""Signed release checks. These provide tamper evidence, not AI-use or authorship detection.

A trusted grader calls this module from its own installation with an external key.
A local script and repository key can be replaced together on a student-owned computer.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from .common import digest, load_json
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', '.releases', 'data', 'results', 'submissions'}
FORBIDDEN_NAMES = {
    'AGENTS.override.md',
    'CLAUDE.local.md',
    'conftest.py',
    'sitecustomize.py',
    'usercustomize.py',
    'pytest.ini',
    'tox.ini',
}
FORBIDDEN_DIRS = {'.claude', '.codex', '.cursor', '.windsurf'}
DANGEROUS_ENV = (
    'PYTHONPATH',
    'PYTHONSTARTUP',
    'PYTHONINSPECT',
    'PYTEST_ADDOPTS',
    'PYTEST_PLUGINS',
    'LD_PRELOAD',
    'DYLD_INSERT_LIBRARIES',
)

class IntegrityError(ValueError):
    pass

def public_key(root: Path, trusted_key: Path | None=None):
    path = trusted_key.resolve() if trusted_key else root / '.integrity/public_key.pem'
    if trusted_key and path.is_relative_to(root.resolve()):
        raise IntegrityError('Authoritative verification needs a public key outside the submitted repository')
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise IntegrityError('Expected an Ed25519 public key')
    return key

def checked_relative(root: Path, name: str) -> Path:
    if not isinstance(name, str) or '\\' in name or (not name) or name.startswith('/'):
        raise IntegrityError(f'Invalid manifest path: {name!r}')
    rel = Path(name)
    if '..' in rel.parts or name != rel.as_posix() or rel == Path('.'):
        raise IntegrityError('Path traversal in manifest')
    path = root / rel
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise IntegrityError(f'Symbolic link in submitted path: {name}')
    if not path.resolve().is_relative_to(root.resolve()):
        raise IntegrityError('File escapes the repository')
    return path

def verify(root: Path, trusted_key: Path | None=None, *, check_environment=True):
    root = root.resolve()
    key = public_key(root, trusted_key)
    manifest_file = checked_relative(root, '.integrity/manifest.json')
    checked_relative(root, '.integrity/manifest.sig')
    raw = manifest_file.read_bytes()
    try:
        key.verify(base64.b64decode((root / '.integrity/manifest.sig').read_bytes(), validate=True), raw)
    except (InvalidSignature, ValueError) as exc:
        raise IntegrityError('Release signature is invalid; restore the issued release') from exc
    manifest = json.loads(raw)
    if manifest.get('format') != 1 or manifest.get('course') != 'AI702':
        raise IntegrityError('Unknown manifest format or course')
    for name, expected in manifest['files'].items():
        path = checked_relative(root, name)
        if not path.is_file() or digest(path) != expected:
            raise IntegrityError(f'Protected file changed or missing: {name}')
    if check_environment:
        for name in DANGEROUS_ENV:
            if os.environ.get(name):
                raise IntegrityError(f'Unexpected {name}. Clear it for course commands; it can change Python or test execution.')
    expected_files = set(manifest['files'])
    for folder, dirs, names in os.walk(root, followlinks=False):
        base = Path(folder)
        if base == root and manifest.get('audience') == 'student' and ('staff' in dirs):
            raise IntegrityError('A student release cannot contain a staff directory')
        for entry in [*dirs, *names]:
            if (base / entry).is_symlink():
                raise IntegrityError(f'Symbolic link not allowed: {(base / entry).relative_to(root)}')
        dirs[:] = [d for d in dirs if d not in SKIP and (not (base == root and d == 'staff'))]
        for name in [*dirs, *names]:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if name in SKIP:
                continue
            if path.is_file() and path.suffix == '.pem' and (b'PRIVATE KEY' in path.read_bytes()):
                raise IntegrityError(f'Private key must not be in the repository: {relative}')
            if base == root and name.endswith(('.py', '.pth')) and (relative not in expected_files):
                raise IntegrityError(f'Unexpected root Python entry/import file: {relative}')
            if base == root and path.is_dir() and (name in {
                'numpy',
                'torch',
                'torchvision',
                'pytest',
                'cryptography',
                'sklearn',
                'site',
            }):
                raise IntegrityError(f'Import-shadowing directory not allowed: {relative}')
            if relative.startswith('.integrity/') and path.is_file() and (relative not in expected_files) and (name not in {'manifest.json', 'manifest.sig'}):
                raise IntegrityError(f'Unexpected integrity file: {relative}')
            if path.is_symlink():
                raise IntegrityError(f'Symbolic link not allowed: {relative}')
            if name in FORBIDDEN_NAMES or (path.is_dir() and name in FORBIDDEN_DIRS):
                raise IntegrityError(f'Unexpected override or execution hook: {relative}')
            if name in ('AGENTS.md', 'CLAUDE.md') and base != root:
                raise IntegrityError(f'Nested agent instructions not allowed: {relative}')
            if path.is_file() and (relative.startswith('shared/') or relative.startswith('.github/')) and (relative not in expected_files):
                raise IntegrityError(f'Unlisted file in protected folder: {relative}')
            if path.is_file() and len(path.relative_to(root).parts) == 2 and base.name.startswith('lab'):
                if base.name not in [f'lab{i}' for i in manifest['available_labs']]:
                    raise IntegrityError(f'Unreleased lab directory: {base.name}')
                if relative not in expected_files and name not in {
                    'student.py',
                    'settings.json',
                    'notes.md',
                    'transfer.json',
                    'transfer.sig',
                    'repair.py',
                    'submission.zip',
                }:
                    raise IntegrityError(f'Unexpected lab file: {relative}')
    for number in manifest['available_labs']:
        transfer = root / f'lab{number}/transfer.json'
        if transfer.exists():
            verify_transfer(root, number, trusted_key=trusted_key)
        repair = root / f'lab{number}/repair.py'
        if repair.exists() and (number != 4 or not transfer.exists()):
            raise IntegrityError('repair.py is allowed only for the released Lab 4 transfer')
    return {
        'release': manifest['release'],
        'available_labs': manifest['available_labs'],
        'audience': manifest.get('audience'),
        'checked_files': len(manifest['files']),
        'authority': 'external trusted key' if trusted_key else 'local repository key (feedback only)',
    }

def verify_transfer(root: Path, number: int, trusted_key=None):
    path = checked_relative(root, f'lab{number}/transfer.json')
    checked_relative(root, f'lab{number}/transfer.sig')
    if not path.is_file():
        raise IntegrityError('The transfer task has not been released yet')
    raw = path.read_bytes()
    try:
        public_key(root, trusted_key).verify(
            base64.b64decode(path.with_suffix('.sig').read_bytes(), validate=True),
            raw,
        )
    except (InvalidSignature, ValueError, OSError) as exc:
        raise IntegrityError('Transfer signature is invalid') from exc
    payload = json.loads(raw)
    if payload.get('lab') != number or payload.get('course') != 'AI702':
        raise IntegrityError('Transfer belongs to a different lab or course')
    if payload.get('release') != load_json(root / '.integrity/manifest.json')['release']:
        raise IntegrityError('Transfer belongs to another release; obtain a matching transfer')
    if type(payload.get('variant')) is not int or payload['variant'] < 0 or not isinstance(payload.get('settings'), dict):
        raise IntegrityError('Malformed transfer payload')
    return payload

def validate_settings(number, cfg):
    """Small, explicit settings checks; no dynamic Python expressions in configuration."""
    if not isinstance(cfg, dict):
        raise ValueError('settings.json must contain an object')
    expected = {
        0: 'seed device',
        1: 'seed dataset features classes hidden_units',
        2: 'seed dataset features max_epochs learning_rate label_noise animate',
        3: 'seed steps learning_rate binary_classes multiclass_classes',
        4: 'seed hidden_units steps learning_rate',
        5: 'seed surface_steps updates batch_size learning_rate example_budget adaptive order curvature device',
        6: 'seed train_size validation_size test_size hidden_units epochs batch_size learning_rate intervention shift_condition device failure_condition',
        7: 'seed train_size validation_size test_size epochs batch_size learning_rate base_channels translation device',
    }[number].split()
    if set(cfg) != set(expected):
        raise ValueError('Missing or unknown settings: ' + str(sorted(set(cfg) ^ set(expected))))
    import math
    for value in cfg.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Settings must be finite')
    if number == 2 and type(cfg['animate']) is not bool:
        raise ValueError('animate must be a boolean')
    if number == 6 and cfg['failure_condition'] not in ('standard', 'reduced_data', 'class_imbalance', 'high_learning_rate'):
        raise ValueError('Unknown failure condition')
    for name, value in cfg.items():
        if name in (
            'seed',
            'steps',
            'max_epochs',
            'updates',
            'surface_steps',
            'batch_size',
            'hidden_units',
            'epochs',
            'base_channels',
            'train_size',
            'validation_size',
            'test_size',
            'example_budget',
        ):
            if type(value) is not int or value < 1 or value > (2 ** 31 - 1 if name == 'seed' else 100000):
                raise ValueError(f'Invalid integer setting: {name}')
        if name in ('learning_rate', 'curvature') and (not isinstance(value, (int, float)) or not 0 < value < 1000):
            raise ValueError(f'Invalid positive setting: {name}')
    if cfg.get('device', 'cpu') not in ('cpu', 'auto', 'cuda', 'mps'):
        raise ValueError('Invalid device')
    if number == 1:
        if cfg['dataset'] not in ('iris', 'penguins'):
            raise ValueError('Lab 1 dataset must be iris or penguins')
        if len(cfg['features']) != 2 or len(cfg['classes']) != 2 or len(set(cfg['classes'])) != 2:
            raise ValueError('Select two features and two different classes')
        if any((type(i) is not int or i < 0 or i >= 4 for i in cfg['features'])):
            raise ValueError('Invalid feature index')
        if any((type(i) is not int or i < 0 or i >= 3 for i in cfg['classes'])):
            raise ValueError('Invalid class index')
    if number == 2:
        if cfg['dataset'] not in ('raisin', 'banknote'):
            raise ValueError('Lab 2 dataset must be raisin or banknote')
        n = 7 if cfg['dataset'] == 'raisin' else 4
        if len(cfg['features']) != 2 or any((type(i) is not int or not 0 <= i < n for i in cfg['features'])):
            raise ValueError('Invalid two-feature choice')
        if not 0 <= cfg['label_noise'] < 0.5:
            raise ValueError('label_noise must be in [0, .5)')
    if number == 3:
        for key in ('binary_classes', 'multiclass_classes'):
            values = cfg[key]
            if len(values) < 2 or len(set(values)) != len(values) or any((type(v) is not int or not 0 <= v <= 9 for v in values)):
                raise ValueError(f'Invalid {key}')
        if len(cfg['binary_classes']) != 2:
            raise ValueError('Select exactly two binary classes')
    if number == 5:
        if cfg['adaptive'] not in ('rmsprop', 'adam'):
            raise ValueError('Select rmsprop or adam')
        if cfg['order'] not in ('shuffle', 'grouped', 'sequential'):
            raise ValueError('Invalid sample order')
    if number == 6:
        if cfg['intervention'] not in (
            'dropout',
            'weight_decay',
            'batch_norm',
            'augmentation',
            'early_stopping',
            'gradient_clipping',
            'input_normalization',
            'initialization',
            'smaller_model',
        ):
            raise ValueError('Unknown intervention')
        if cfg['shift_condition'] not in ('brightness', 'contrast', 'noise', 'occlusion', 'translation'):
            raise ValueError('Unknown shift condition')
    if number == 7 and (type(cfg['translation']) is not int or not 0 < cfg['translation'] < 32):
        raise ValueError('Translation must be 1 to 31 pixels')
