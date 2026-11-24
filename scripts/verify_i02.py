"""Reload I02 estimates and independently check training diagnostics (no held-out scoring)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tmbench.adapters.rice_phase import load
from tmbench.fit import write_json


def verify(directory, archive_dir=None):
    status = json.loads((directory / 'status.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    dataset = load(Path('data/raw/phase_slm_40x40'), provenance['manifest'], provenance['policy'])
    assert status['status'] == 'complete'
    assert status['attempted_rows'] == len(dataset.train.detector_ids)
    assert [r['detector_id'] for r in status['rows']] == dataset.train.detector_ids.tolist()
    X = dataset.train.X.astype(float)
    Y = dataset.train.amplitudes
    errors = []
    estimates = np.full((len(status['rows']), X.shape[1]), np.nan + 1j * np.nan)
    for j, row in enumerate(status['rows']):
        if 'estimate_file' not in row:
            continue
        artifact = np.load(directory / row['estimate_file'], allow_pickle=False)
        estimate = artifact['A_row']
        assert int(artifact['detector_id']) == row['detector_id']
        assert estimate.shape == (X.shape[1],) and np.isfinite(estimate).all()
        estimates[j] = estimate
        # Evaluate using X directly, independently of the solver's SVD forward field.
        prediction = X @ estimate.real + 1j * (X @ estimate.imag)
        residual = float(np.linalg.norm(np.abs(prediction) - Y[:, j]) / np.linalg.norm(Y[:, j]))
        np.testing.assert_allclose(residual, row['training_relative_amplitude_residual'], rtol=1e-9, atol=1e-10)
        errors.append(abs(residual - row['training_relative_amplitude_residual']))
        history = json.loads((directory / f"row_{row['detector_id']}_history.json").read_text())
        assert len(history) == row['iterations']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(directory.iterdir()) if p.is_file()}
    residuals = [r['training_relative_amplitude_residual'] for r in status['rows'] if 'estimate_file' in r]
    archived = None
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f'{directory.name}_estimates.npz'
        np.savez_compressed(archive, A=estimates, detector_ids=dataset.train.detector_ids,
                            statuses=np.array([r['status'] for r in status['rows']]))
        archived = dict(path=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    return dict(run_directory=str(directory), source_commit=provenance['git_commit'],
                source_dirty=provenance['git_dirty'], config=provenance['config'],
                python=provenance['python'], numpy=provenance['numpy'], scipy=provenance['scipy'],
                threads=provenance['threads'], svd_count=status['svd_count'],
                prepare_seconds=status['prepare_seconds'], end_to_end_seconds=status['end_to_end_seconds'],
                peak_traced_bytes=status['peak_traced_bytes'], memory_note=status['memory_note'],
                attempted_rows=status['attempted_rows'], finite_estimate_rows=status['finite_estimate_rows'],
                converged_rows=status['converged_rows'],
                training_residual_min=float(min(residuals)) if residuals else None,
                training_residual_median=float(np.median(residuals)) if residuals else None,
                training_residual_max=float(max(residuals)) if residuals else None,
                reloaded_residual_max_difference=max(errors, default=0), rows=status['rows'],
                artifact_sha256=hashes, archived_estimates=archived)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/results/i02_summary.json'))
    parser.add_argument('--archive-estimates', action='store_true')
    args = parser.parse_args()
    reports = [verify(directory, args.output.parent if args.archive_estimates else None) for directory in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(note='Training diagnostics only; no held-out evaluation or accuracy claim', runs=reports))
    print(json.dumps([{k: v for k, v in report.items() if k not in ('rows', 'artifact_sha256')} for report in reports], indent=2))
