"""Verify saved coefficients and exact report replay; preserve compact checkpoint evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tmbench.adapters.rice_phase import load
from tmbench.fit import write_json
from tmbench.runner import report
from tmbench.solvers.prvamp.linear import matvec


def archive(directory, target, allow_incomplete=False):
    provenance = json.loads((directory / 'provenance.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'complete' and not allow_incomplete:
        raise ValueError('Run is not complete')
    original = (directory / 'evaluation.json').read_bytes()
    evaluation = report(directory, Path('data/raw/phase_slm_40x40'))
    if (directory / 'evaluation.json').read_bytes() != original:
        raise AssertionError('Regenerated evaluation differs')
    dataset = load(Path('data/raw/phase_slm_40x40'), provenance['manifest'], provenance['policy'])
    estimates = np.full((len(status['rows']), dataset.train.X.shape[1]), np.nan+1j*np.nan)
    discrepancies, warm_discrepancies = [], []
    X = dataset.train.X.astype(complex if np.iscomplexobj(dataset.train.X) else float)
    Y = dataset.train.amplitudes
    for index, row in enumerate(status['rows']):
        if 'estimate_file' not in row:
            continue
        saved = np.load(directory / row['estimate_file'], allow_pickle=False)
        estimates[index] = saved['A_row']
        if 'variance' in saved:
            assert saved['variance'].shape == saved['A_row'].shape
            assert np.isfinite(saved['variance']).all() and np.all(saved['variance'] > 0)
        predicted = np.abs(matvec(X, saved['A_row']))
        residual = float(np.linalg.norm(predicted-Y[:, index])/np.linalg.norm(Y[:, index]))
        np.testing.assert_allclose(residual, row['training_relative_amplitude_residual'], rtol=1e-9, atol=1e-10)
        discrepancies.append(abs(residual-row['training_relative_amplitude_residual']))
        if 'final_intensity_loss' in row:
            target_intensity = dataset.train.intensities[:, index].astype(float)
            loss = np.mean((predicted**2 - target_intensity)**2) / 2
            np.testing.assert_allclose(loss, row['final_intensity_loss'], rtol=1e-9, atol=1e-10)
        history = json.loads((directory / f"row_{row['detector_id']}_history.json").read_text())
        assert len(history) == row['iterations']
        if 'warm_start_file' in row:
            warm = np.load(directory / row['warm_start_file'], allow_pickle=False)
            assert int(warm['detector_id']) == row['detector_id']
            assert warm['A_row'].shape == warm['variance'].shape == saved['A_row'].shape
            assert np.isfinite(warm['variance']).all() and np.all(warm['variance'] > 0)
            warm_history = json.loads((directory / f"row_{row['detector_id']}_warm_history.json").read_text())
            assert len(warm_history) == row['warm_start_iterations']
            warm_residual = float(np.linalg.norm(np.abs(matvec(X, warm['A_row']))-Y[:, index])/np.linalg.norm(Y[:, index]))
            expected = warm_history[-1]['relative_amplitude_residual']
            np.testing.assert_allclose(warm_residual, expected, rtol=1e-9, atol=1e-10)
            warm_discrepancies.append(abs(warm_residual-expected))
            assert row['solve_seconds'] >= row['warm_start_seconds']+row['swept_seconds'] > 0
            rng = np.random.default_rng(row['sweep_seed'])
            for entry in history:
                np.testing.assert_array_equal(entry['coordinate_order'], rng.permutation(X.shape[1]))
    target.mkdir(parents=True, exist_ok=True)
    matrix = target / f'{directory.name}_estimates.npz'
    np.savez_compressed(matrix, A=estimates, detector_ids=dataset.train.detector_ids,
                        statuses=np.array([r['status'] for r in status['rows']]))
    result = dict(provenance=provenance, status=status, evaluation=evaluation,
                  complete_run=status['status'] == 'complete',
                  report_replay_byte_identical=True,
                  reloaded_training_residual_max_difference=max(discrepancies, default=0.),
                  warm_start_artifacts_verified=len(warm_discrepancies),
                  reloaded_warm_training_residual_max_difference=max(warm_discrepancies, default=0.),
                  archive_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(directory.iterdir()) if p.is_file()})
    for name in ('pilot', 'pipeline_timing'):
        path = directory / f'{name}.json'
        if path.exists():
            result[name] = json.loads(path.read_text())
    write_json(target / f'{directory.name}_summary.json', result)
    print(json.dumps(dict(run=directory.name, finite_rows=status['finite_estimate_rows'],
                         converged_rows=status['converged_rows'], replay='byte-identical',
                         held_out={k: v['relative_amplitude_residual'] for k, v in evaluation['metrics'].items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--target', type=Path, default=Path('docs/results'))
    parser.add_argument('--allow-incomplete', action='store_true', help='Preserve a stopped trial without calling it complete')
    args = parser.parse_args()
    archive(args.directory, args.target, args.allow_incomplete)
