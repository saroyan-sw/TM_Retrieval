"""Reproducible public-data fitting and artifact-only held-out report regeneration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from tmbench.fit import run as fit_run, write_json
from tmbench.adapters.rice_phase import load
from tmbench.evaluation.metrics import evaluate
import numpy as np


DEFAULT_CONFIGS = {
    'prvamp': 'prvamp_i02_damped.json',
    'prgamp': 'prgamp_i04.json',
    'prvbem': 'prvbem_i05.json',
    'prsamp': 'prsamp_i06_fixed_variance.json',
    'gs': 'gs_i07_50.json',
    'wf': 'wf_i08_original.json',
}


def report(output, data_dir):
    provenance = json.loads((output / 'provenance.json').read_text())
    status = json.loads((output / 'status.json').read_text())
    dataset = load(data_dir, provenance['manifest'], provenance['policy'])
    if [r['detector_id'] for r in status['rows']] != dataset.train.detector_ids.tolist():
        raise ValueError('Saved detector ordering does not match dataset')
    estimates, hashes = [], {}
    for row in status['rows']:
        if 'estimate_file' in row:
            path = output / row['estimate_file']
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = np.load(path, allow_pickle=False)
            if int(saved['detector_id']) != row['detector_id']:
                raise ValueError('Artifact detector ID mismatch')
            a = saved['A_row']
            if a.shape != (dataset.train.X.shape[1],):
                raise ValueError('Artifact input dimension mismatch')
            estimates.append(a)
        else:
            estimates.append(None)
    statuses = [r['status'] for r in status['rows']]
    metrics = {name: evaluate(getattr(dataset, name), estimates, statuses)
               for name in ('validation', 'test')}
    result = dict(method=provenance.get('method', 'prvamp'), estimate_sha256=hashes,
                  metrics=metrics, note='No reference TM; no coefficient-error or published ranking claim')
    if result['method'] == 'wf':
        result['policy'] = provenance['config']['policy']
    write_json(output / 'evaluation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'report'])
    parser.add_argument('--method', default='prvamp', choices=DEFAULT_CONFIGS)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/rice_phase_slm_40x40.json'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw/phase_slm_40x40'))
    parser.add_argument('--policy', type=Path, default=Path('configs/public/rice_phase_slm_40x40.json'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.config is None:
        args.config = Path('configs/public') / DEFAULT_CONFIGS[args.method]
    start = time.perf_counter()
    exit_code = 0
    if args.action == 'run':
        exit_code = fit_run(args.manifest, args.data_dir, args.policy, args.config, args.output, method=args.method)
    result = report(args.output, args.data_dir)
    if args.action == 'run':
        write_json(args.output / 'pipeline_timing.json', dict(end_to_end_including_evaluation_seconds=time.perf_counter()-start))
    print(json.dumps({k: {n: v for n, v in m.items() if n != 'rows'} for k, m in result['metrics'].items()}, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
