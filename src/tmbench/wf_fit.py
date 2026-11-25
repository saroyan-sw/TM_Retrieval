"""Fit and checkpoint intensity-domain Wirtinger Flow on training rows."""

import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from .fit import write_json
from .solvers.wf import WFConfig, WirtingerFlow


def fit_rows(train, config, output):
    seed = config['seed']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    settings = WFConfig(**{key: value for key, value in config.items() if key != 'seed'})

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    summary = dict(status='running', method='wf', policy=settings.policy,
                   attempted_rows=len(records), rows=records, solver_config=asdict(settings),
                   training_shape=list(train.X.shape), seed=seed,
                   objective='mean((abs(X @ a)**2 - training_intensity)**2) / 2',
                   initialization='seeded intensity-weighted spectral power iteration')
    with status_path.open('x', encoding='utf-8') as checkpoint:
        json.dump(summary, checkpoint)

    start = time.perf_counter()
    try:
        solver = WirtingerFlow(train.X)
    except ValueError as error:
        for record in records:
            record.update(status='unsupported_design', error=str(error))
        summary.update(status='complete', prepare_seconds=time.perf_counter() - start,
                       finite_estimate_rows=0, converged_rows=0, finite_estimate_coverage=0.)
        write_json(status_path, summary)
        return summary
    summary['prepare_seconds'] = time.perf_counter() - start

    for index, record in enumerate(records):
        start = time.perf_counter()
        row = record['detector_id']
        record.update(seed=[seed, row], status='running', policy=settings.policy)
        write_json(status_path, summary)
        try:
            if not train.valid[:, index].all():
                record['status'] = 'unsupported_mask'
            else:
                # Cast before the intensity loss to avoid integer overflow.
                intensities = train.intensities[:, index].astype(float)
                tick = time.perf_counter()
                try:
                    initial = solver.spectral_initialize(
                        intensities, iterations=settings.spectral_iterations, seed=record['seed'],
                    )
                finally:
                    record['initialization_seconds'] = time.perf_counter() - tick
                result = solver.fit(intensities, initial, settings)
                truth = np.sqrt(intensities)
                residual = np.abs(result.z) - truth
                squared_error = float(residual @ residual)
                energy = float(truth @ truth)
                estimate_file = f'row_{row}.npz'
                np.savez(output / estimate_file, A_row=result.x, detector_id=row)
                write_json(output / f'row_{row}_history.json', result.history)
                record.update(status=result.status, converged=result.status == 'converged',
                              iterations=len(result.history), estimate_file=estimate_file,
                              initial_intensity_loss=result.initial_loss, final_intensity_loss=result.final_loss,
                              training_squared_error=squared_error, training_amplitude_energy=energy,
                              training_sample_count=len(truth),
                              training_relative_amplitude_residual=float(np.sqrt(squared_error / energy)))
        except (ValueError, FloatingPointError) as error:
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter() - start
        write_json(status_path, summary)
        print(f"wf {settings.policy} detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)

    finite = sum('estimate_file' in row for row in records)
    summary.update(status='complete', finite_estimate_rows=finite,
                   converged_rows=sum(row.get('converged', False) for row in records),
                   finite_estimate_coverage=finite / len(records),
                   initialization_seconds=sum(row.get('initialization_seconds', 0.) for row in records))
    write_json(status_path, summary)
    return summary
