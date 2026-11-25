"""Train bounded PhaseMax with a saved spectral objective anchor per detector."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .fit import write_json
from .solvers._convex import BACKEND, BackendUnavailable, UnsupportedSize
from .solvers.phasemax import MAX_MODES, MAX_PATTERNS, PhaseMax, PhaseMaxConfig
from .solvers.wf import WirtingerFlow


def fit_rows(train, config, output):
    seed = config['seed']
    spectral_iterations = config['spectral_iterations']
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(spectral_iterations) is not int or spectral_iterations < 1:
        raise ValueError('spectral_iterations must be a positive integer')
    settings = PhaseMaxConfig(**{k: v for k, v in config.items() if k not in ('seed', 'spectral_iterations')})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'status.json'
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    summary = dict(status='running', method='phasemax', variant=settings.variant,
                   backend=BACKEND, backend_started=False, backend_warm_start=False,
                   anchor_policy='training_intensity_spectral', anchor_iterations=spectral_iterations,
                   anchor_seed=seed, attempted_rows=len(records), rows=records,
                   training_shape=list(train.X.shape), solver_config=asdict(settings),
                   limits=dict(max_patterns=MAX_PATTERNS, max_modes=MAX_MODES))
    with checkpoint.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle)

    start = time.perf_counter()
    try:
        solver = PhaseMax(train.X, settings)
        initializer = WirtingerFlow(solver.X)
    except (ValueError, BackendUnavailable) as error:
        status = 'unsupported_design'
        if isinstance(error, UnsupportedSize):
            status = 'unsupported_size'
        elif isinstance(error, BackendUnavailable):
            status = 'backend_unavailable'
        for record in records:
            record.update(status=status, reason=str(error))
        summary.update(status='complete', preparation_status=status, reason=str(error),
                       prepare_seconds=time.perf_counter() - start, finite_estimate_rows=0,
                       converged_rows=0, finite_estimate_coverage=0.)
        write_json(checkpoint, summary)
        print(f'phasemax: {status}: {error}', flush=True)
        return summary
    summary.update(prepare_seconds=time.perf_counter() - start, backend_versions=solver.backend_versions)

    for index, record in enumerate(records):
        start = time.perf_counter()
        row = record['detector_id']
        record['status'] = 'running'
        write_json(checkpoint, summary)
        try:
            if not train.valid[:, index].all():
                record['status'] = 'unsupported_mask'
            else:
                intensities = train.intensities[:, index]
                row_seed = [seed, row]
                record.update(anchor_seed=row_seed, anchor_status='running')
                tick = time.perf_counter()
                try:
                    anchor = initializer.spectral_initialize(intensities, iterations=spectral_iterations, seed=row_seed)
                    anchor /= np.linalg.norm(anchor)
                finally:
                    record['anchor_seconds'] = time.perf_counter() - tick
                anchor_file = f'row_{row}_anchor.npz'
                np.savez(output / anchor_file, anchor=anchor, detector_id=row)
                record.update(anchor_status='complete', anchor_file=anchor_file)
                write_json(checkpoint, summary)

                amplitudes = np.sqrt(intensities, dtype=float)
                result = solver.fit(amplitudes, anchor)
                summary['backend_started'] = True
                record.update(status=result.status, **result.diagnostics)
                if result.x is not None:
                    residual = np.abs(solver.X @ result.x) - amplitudes
                    energy = float(amplitudes @ amplitudes)
                    squared_error = float(residual @ residual)
                    estimate_file = f'row_{row}.npz'
                    np.savez(output / estimate_file, A_row=result.x, detector_id=row)
                    record.update(estimate_file=estimate_file, converged=result.status == 'optimal',
                                  training_squared_error=squared_error, training_amplitude_energy=energy,
                                  training_sample_count=len(amplitudes),
                                  training_relative_amplitude_residual=float(np.sqrt(squared_error / energy)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            if record.get('anchor_status') == 'running':
                record['anchor_status'] = 'failed'
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter() - start
        write_json(checkpoint, summary)
        print(f"phasemax detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)

    finite = sum('estimate_file' in row for row in records)
    summary.update(status='complete', finite_estimate_rows=finite,
                   converged_rows=sum(row.get('converged', False) for row in records),
                   finite_estimate_coverage=finite / len(records),
                   anchor_seconds=sum(row.get('anchor_seconds', 0.) for row in records))
    write_json(checkpoint, summary)
    return summary
