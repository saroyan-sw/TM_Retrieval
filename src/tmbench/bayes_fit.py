"""Training-only row adapter for audited Bayesian baselines."""
from pathlib import Path
import time
import numpy as np

from .fit import write_json
from .solvers.bayes import Design, prgamp, prvbem
from .pilots import estimate_variances


def fit_rows(train, config, output, method):
    if method not in ('prgamp', 'prvbem'):
        raise ValueError('Unknown Bayesian solver')
    fraction = config['noise_fraction']
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('noise_fraction must be in (0, 1)')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists():
        raise FileExistsError('Choose a fresh output directory')
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    summary = dict(status='running', method=method, attempted_rows=len(records), rows=records,
                   solver_config=config, training_shape=list(train.X.shape),
                   likelihood='Rician amplitude; total circular complex field-noise variance',
                   mismatch='Quantized camera counts; field-noise likelihood is not a verified sensor model',
                   hyperparameter_policy='training energy with configured noise fraction; no held-out tuning')
    write_json(output / 'status.json', summary)
    start = time.perf_counter()
    design = Design(train.X, gram=method == 'prvbem' or bool(config.get('pilot')))
    summary['prepare_seconds'] = time.perf_counter()-start
    pilot = None
    if config.get('pilot'):
        pilot = estimate_variances(train, design, config['pilot'], lambda p: write_json(output / 'pilot.json', p))
        summary['pilot_seconds'] = pilot['seconds']
        summary['pilot_status'] = pilot['status']
        summary['hyperparameter_policy'] = 'training-only VBEM pilots; median normalized noise/prior; per-row energy rescaling'
    else:
        summary['pilot_seconds'] = 0.
    amplitudes = train.amplitudes
    for index, record in enumerate(records):
        start = time.perf_counter()
        row = record['detector_id']
        seed = [int(config['seed']), row]
        record.update(seed=seed, status='running')
        write_json(output / 'status.json', summary)
        try:
            if pilot is not None and pilot['status'] != 'complete':
                record['status'] = 'pilot_failed'
            elif not train.valid[:, index].all():
                record['status'] = 'unsupported_mask'
            else:
                y = amplitudes[:, index]
                energy = float(np.mean(y**2))
                noise, prior = fraction*energy, (1-fraction)*energy/design.probe_energy
                if pilot is not None:
                    noise, prior = pilot['noise_fraction']*energy, pilot['prior_factor']*energy/design.probe_energy
                record.update(initial_noise_variance=noise, initial_prior_variance=prior)
                solver = {'prgamp': prgamp, 'prvbem': prvbem}[method]
                result = solver(design, y, noise, prior, config, seed)
                estimate_file = f'row_{row}.npz'
                np.savez(output / estimate_file, A_row=result.x, variance=result.variance, detector_id=row)
                write_json(output / f'row_{row}_history.json', result.history)
                residual = np.abs(result.z)-y
                record.update(status=result.status, converged=result.converged, iterations=len(result.history),
                              estimate_file=estimate_file, noise_variance=result.noise_variance,
                              prior_variance=result.prior_variance,
                              training_squared_error=float(residual @ residual),
                              training_amplitude_energy=float(y @ y), training_sample_count=len(y),
                              training_relative_amplitude_residual=float(np.linalg.norm(residual)/np.linalg.norm(y)),
                              stabilized_messages=sum(h['stabilized'] for h in result.history), **result.metadata)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter()-start
        write_json(output / 'status.json', summary)
        print(f"{method} detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)
    summary.update(status='complete', finite_estimate_rows=sum('estimate_file' in r for r in records),
                   converged_rows=sum(r.get('converged', False) for r in records))
    summary['finite_estimate_coverage'] = summary['finite_estimate_rows']/len(records)
    write_json(output / 'status.json', summary)
    return summary
