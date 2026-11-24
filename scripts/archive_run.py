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


def archive(directory, target):
    provenance = json.loads((directory / 'provenance.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Run is not complete')
    original = (directory / 'evaluation.json').read_bytes()
    evaluation = report(directory, Path('data/raw/phase_slm_40x40'))
    if (directory / 'evaluation.json').read_bytes() != original:
        raise AssertionError('Regenerated evaluation differs')
    dataset = load(Path('data/raw/phase_slm_40x40'), provenance['manifest'], provenance['policy'])
    estimates = np.full((len(status['rows']), dataset.train.X.shape[1]), np.nan+1j*np.nan)
    discrepancies = []
    X, Y = dataset.train.X.astype(float), dataset.train.amplitudes
    for index, row in enumerate(status['rows']):
        if 'estimate_file' not in row:
            continue
        saved = np.load(directory / row['estimate_file'], allow_pickle=False)
        estimates[index] = saved['A_row']
        residual = float(np.linalg.norm(np.abs(matvec(X, saved['A_row']))-Y[:, index])/np.linalg.norm(Y[:, index]))
        np.testing.assert_allclose(residual, row['training_relative_amplitude_residual'], rtol=1e-9, atol=1e-10)
        discrepancies.append(abs(residual-row['training_relative_amplitude_residual']))
    target.mkdir(parents=True, exist_ok=True)
    matrix = target / f'{directory.name}_estimates.npz'
    np.savez_compressed(matrix, A=estimates, detector_ids=dataset.train.detector_ids,
                        statuses=np.array([r['status'] for r in status['rows']]))
    result = dict(provenance=provenance, status=status, evaluation=evaluation,
                  report_replay_byte_identical=True,
                  reloaded_training_residual_max_difference=max(discrepancies, default=0.),
                  archive_sha256=hashlib.sha256(matrix.read_bytes()).hexdigest(),
                  artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(directory.iterdir()) if p.is_file()})
    write_json(target / f'{directory.name}_summary.json', result)
    print(json.dumps(dict(run=directory.name, finite_rows=status['finite_estimate_rows'],
                         converged_rows=status['converged_rows'], replay='byte-identical',
                         held_out={k: v['relative_amplitude_residual'] for k, v in evaluation['metrics'].items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--target', type=Path, default=Path('docs/results'))
    args = parser.parse_args()
    archive(args.directory, args.target)
