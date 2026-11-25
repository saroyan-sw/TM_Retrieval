"""Fit and checkpoint Gerchberg–Saxton estimates on the frozen training split."""

import json
from pathlib import Path
import time

import numpy as np

from .fit import write_json
from .solvers.gs import GerchbergSaxton


def fit_rows(train, config, output):
    seed, budget, rcond = config['seed'], config['max_iter'], config['pinv_rcond']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(budget) is not int or budget < 1:
        raise ValueError('max_iter must be a positive integer')
    if not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError('pinv_rcond must be in [0, 1)')

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    summary = dict(status='running', method='gs', attempted_rows=len(records), rows=records,
                   training_shape=list(train.X.shape), solver_config=config,
                   initialization='unit total variance complex Gaussian coefficients; seed [seed, detector_id]',
                   stopping='fixed number of full projection cycles')
    with status_path.open('x', encoding='utf-8') as checkpoint:
        json.dump(summary, checkpoint)

    start = time.perf_counter()
    try:
        solver = GerchbergSaxton(train.X, rcond=rcond)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        for record in records:
            record.update(status='unsupported_design', error=str(error))
        summary.update(status='complete', prepare_seconds=time.perf_counter() - start,
                       finite_estimate_rows=0, converged_rows=0, finite_estimate_coverage=0.)
        write_json(status_path, summary)
        return summary
    summary.update(prepare_seconds=time.perf_counter() - start, pseudoinverse_count=1,
                   pseudoinverse_rank=solver.rank, pseudoinverse_cutoff=solver.cutoff)

    amplitudes = train.amplitudes
    for index, record in enumerate(records):
        start = time.perf_counter()
        row = record['detector_id']
        record.update(seed=[seed, row], status='running')
        write_json(status_path, summary)
        try:
            if not train.valid[:, index].all():
                record['status'] = 'unsupported_mask'
            else:
                y = amplitudes[:, index]
                result = solver.fit(y, max_iter=budget, seed=record['seed'])
                estimate_file = f'row_{row}.npz'
                np.savez(output / estimate_file, A_row=result.x, detector_id=row)
                write_json(output / f'row_{row}_history.json', result.history)
                residual = np.abs(result.z) - y
                record.update(status='max_iter_reached', converged=False, iterations=len(result.history),
                              estimate_file=estimate_file, training_squared_error=float(residual @ residual),
                              training_amplitude_energy=float(y @ y), training_sample_count=len(y),
                              training_relative_amplitude_residual=result.history[-1]['relative_amplitude_residual'])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter() - start
        write_json(status_path, summary)
        print(f"gs detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)
    finite = sum('estimate_file' in r for r in records)
    summary.update(status='complete', finite_estimate_rows=finite, converged_rows=0,
                   finite_estimate_coverage=finite / len(records))
    write_json(status_path, summary)
    return summary
