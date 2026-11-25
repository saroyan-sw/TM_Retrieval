"""PhaseLift training adapter with explicit coverage for skipped detectors."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .fit import write_json
from .solvers.phaselift import (
    BACKEND,
    MAX_MODES,
    MAX_PATTERNS,
    BackendUnavailable,
    PhaseLift,
    PhaseLiftConfig,
    UnsupportedSize,
)


def fit_rows(train, config, output):
    settings = PhaseLiftConfig(**config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'status.json'
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    patterns, modes = train.X.shape
    summary = dict(status='running', method='phaselift', variant=settings.variant,
                   backend=BACKEND, backend_started=False, attempted_rows=len(records), rows=records,
                   solver_config=asdict(settings), training_shape=[patterns, modes],
                   limits=dict(max_patterns=MAX_PATTERNS, max_modes=MAX_MODES),
                   dense_operator_bytes=16 * patterns * modes**2)
    with checkpoint.open('x', encoding='utf-8') as handle:
        json.dump(summary, handle)

    start = time.perf_counter()
    try:
        solver = PhaseLift(train.X, settings)
    except ValueError as error:
        return _skip(checkpoint, summary, start, error,
                     'unsupported_size' if isinstance(error, UnsupportedSize) else 'unsupported_design')
    except BackendUnavailable as error:
        return _skip(checkpoint, summary, start, error, 'backend_unavailable')
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
                I = train.intensities[:, index]
                result = solver.fit(I)
                summary['backend_started'] = True
                record.update(status=result.status, **result.diagnostics)
                if result.x is not None:
                    truth = np.sqrt(I, dtype=float)
                    residual = np.abs(solver.X @ result.x) - truth
                    energy = float(truth @ truth)
                    squared_error = float(residual @ residual)
                    name = f'row_{row}.npz'
                    np.savez(output / name, A_row=result.x, lifted=result.lifted, detector_id=row)
                    record.update(estimate_file=name, converged=result.status == 'optimal',
                                  training_squared_error=squared_error, training_amplitude_energy=energy,
                                  training_sample_count=len(truth),
                                  training_relative_amplitude_residual=float(np.sqrt(squared_error / energy)))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter() - start
        write_json(checkpoint, summary)
        print(f"phaselift detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)

    finite = sum('estimate_file' in row for row in records)
    summary.update(status='complete', finite_estimate_rows=finite,
                   converged_rows=sum(row.get('converged', False) for row in records),
                   finite_estimate_coverage=finite / len(records))
    write_json(checkpoint, summary)
    return summary


def _skip(checkpoint, summary, start, error, status):
    for record in summary['rows']:
        record.update(status=status, reason=str(error))
    summary.update(status='complete', preparation_status=status, reason=str(error),
                   prepare_seconds=time.perf_counter() - start, finite_estimate_rows=0,
                   converged_rows=0, finite_estimate_coverage=0.)
    write_json(checkpoint, summary)
    print(f'phaselift: {status}: {error}', flush=True)
    return summary
