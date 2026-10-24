#!/usr/bin/env python3
"""Small course launcher. Example: python lab.py run 4; python lab.py view 4."""
from __future__ import annotations
import argparse
import importlib
import json
import os
import subprocess
import sys
import webbrowser
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.common import load_json, settings, student_module, check_evidence, digest, result_path
from shared.integrity import verify, verify_transfer, validate_settings, IntegrityError

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('check', 'test', 'run', 'view', 'transfer', 'package', 'baseline', 'extension'):
        s = sub.add_parser(name)
        s.add_argument('lab', type=int, choices=range(8), nargs='?' if name == 'check' else None)
        s.add_argument('--key', type=Path, help='Authoritative public key outside this repository')
        if name in ('run', 'test'):
            s.add_argument(
                '--reference',
                action='store_true',
                help='Staff only: use the staff solution, visibly marked as reference',
            )
        if name == 'run':
            s.add_argument(
                '--smoke',
                action='store_true',
                help='Small synthetic diagnostic; not a graded dataset result',
            )
            s.add_argument('--stage', choices=('core', 'transfer'), default='core')
            s.add_argument('--device', choices=('cpu', 'auto', 'cuda', 'mps'))
        if name in ('view', 'package'):
            s.add_argument('--run-id', help='Select a previous completed run')
            s.add_argument('--stage', choices=('core', 'transfer', 'baseline', 'extension'), default='core')
        if name in ('baseline', 'extension'):
            s.add_argument('--smoke', action='store_true')
        if name == 'extension':
            s.add_argument('--fine-tune', action='store_true')
        if name == 'view':
            s.add_argument('--no-open', action='store_true')
    s = sub.add_parser('data')
    s.add_argument(
        'dataset',
        choices=[
            'iris',
            'digits',
            'wine',
            'breast_cancer',
            'raisin',
            'banknote',
            'dry_bean',
            'penguins',
            'cifar10',
        ],
    )
    return p

def test(number, reference=False):
    env = os.environ.copy()
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    for key in ('PYTHONPATH', 'PYTHONSTARTUP', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(key, None)
    if reference:
        if not (ROOT / 'staff/check.py').is_file():
            raise ValueError('Staff references are not distributed to students')
        command = [sys.executable, str(ROOT / 'staff/check.py'), '--lab', str(number)]
    else:
        command = [
            sys.executable,
            '-m',
            'pytest',
            '-q',
            '--import-mode=importlib',
            '-c',
            str(ROOT / 'pyproject.toml'),
            str(ROOT / f'lab{number}/test_lab{number}.py'),
        ]
    return subprocess.call(command, cwd=ROOT, env=env)

def package(number, stage, run_id=None, key=None):
    from shared.integrity import checked_relative
    if stage not in ('core', 'transfer'):
        raise ValueError('Only core and transfer runs are assessable')
    results = result_path(number, stage, run_id)
    errors = check_evidence(results)
    if errors:
        raise ValueError('; '.join(errors))
    record = load_json(results / 'run.json')
    if not record['eligible_for_submission']:
        raise ValueError('Smoke/reference runs are diagnostics and cannot be packaged as assessed results')
    if stage == 'transfer':
        # Verify the signed condition actually captured in this run.
        snapshot = results / 'source'
        from shared.evidence import verify_snapshot_transfer
        verify_snapshot_transfer(snapshot, number, ROOT, key)
    output = checked_relative(ROOT, f"lab{number}/submissions/{stage}-{record['run_id']}.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f'Package already exists: {output}. It is immutable. Select another run after revising work.')
    notes = checked_relative(ROOT, f'lab{number}/notes.md')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(results.rglob('*')):
            checked_relative(ROOT, path.relative_to(ROOT).as_posix())
            if path.is_file():
                z.write(path, f'lab{number}/results/{stage}/' + path.relative_to(results).as_posix())
        z.write(notes, f'lab{number}/notes_final.md')
    print(f'SHA256: {digest(output)}')
    print('Submit this stage ZIP and the matching stage commit to the LMS. Keep the core submission before editing for transfer.')
    return output

def main():
    args = parser().parse_args()
    try:
        report = verify(ROOT, getattr(args, 'key', None))
        if args.command == 'data':
            from shared.data import tabular, cifar
            d = cifar(train_size=100, validation_size=50, test_size=50) if args.dataset == 'cifar10' else tabular(args.dataset)
            print(f"Ready: {d['name']} (cache is outside the repository)")
            return 0
        if args.lab is not None and args.lab not in report['available_labs']:
            raise ValueError(f'Lab {args.lab} has not been released')
        if args.command == 'check':
            print(json.dumps(report, indent=2))
            return 0
        if args.command == 'test':
            return test(args.lab, args.reference)
        if args.command == 'baseline':
            if args.lab != 0:
                raise ValueError('The introductory baseline command is for Lab 0')
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            from lab0.run import baseline_demo
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                print(baseline_demo(settings(0), smoke=args.smoke))
            return 0
        if args.command == 'extension':
            if args.lab != 7:
                raise ValueError('The optional pretrained-network demonstration is in Lab 7')
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            from lab7.extension import run
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                print(run(settings(7), smoke=args.smoke, fine_tune=args.fine_tune))
            return 0
        if args.command == 'transfer':
            payload = verify_transfer(ROOT, args.lab, trusted_key=args.key)
            print(payload['task'])
            print('Settings for this condition:', payload['settings'])
            print(f'Run: python lab.py run {args.lab} --stage transfer')
            return 0
        if args.command == 'view':
            path = result_path(args.lab, args.stage, args.run_id) / 'index.html'
            if not path.is_file():
                raise ValueError('Run the lab first')
            print(path)
            if not args.no_open:
                webbrowser.open(path.as_uri())
            return 0
        if args.command == 'package':
            print(package(args.lab, args.stage, args.run_id, args.key))
            return 0
        if args.command == 'run':
            cfg = settings(args.lab)
            if args.device and 'device' in cfg:
                cfg['device'] = args.device
            if args.stage == 'transfer':
                payload = verify_transfer(ROOT, args.lab, trusted_key=args.key)
                unknown = set(payload['settings']) - set(cfg)
                if unknown:
                    raise ValueError(f'Transfer has unknown settings: {unknown}')
                cfg.update(payload['settings'])
            validate_settings(args.lab, cfg)
            tasks = student_module(args.lab, args.reference)
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            if args.reference:
                print('STAFF REFERENCE RUN: not student evidence')
            if args.smoke:
                print('SMOKE FIXTURE RUN: not real-data assessment evidence')
            runner = importlib.import_module(f'lab{args.lab}.run')
            from threadpoolctl import threadpool_limits
            # Tiny teaching matrices are slower with very large BLAS thread pools.
            with threadpool_limits(limits=4):
                result = runner.run(
                    tasks,
                    cfg,
                    smoke=args.smoke,
                    reference=args.reference,
                    stage=args.stage,
                )
            verify(ROOT, args.key)
            print(result)
            print(f'View: python lab.py view {args.lab} --stage {args.stage}')
            return 0
    except (
        IntegrityError,
        ValueError,
        RuntimeError,
        NotImplementedError,
        ImportError,
        OSError,
        AssertionError,
    ) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        if isinstance(exc, NotImplementedError):
            print(f'Complete lab{args.lab}/student.py, then run the public tests.', file=sys.stderr)
        return 2
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
