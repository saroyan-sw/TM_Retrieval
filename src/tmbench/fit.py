"""I02 checkpoint: fit training rows, sharing X-only preparation and saving each row."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tracemalloc

import numpy as np
import scipy

from tmbench.adapters.rice_phase import load
from tmbench.solvers.prvamp import PrVAMP, VAMPConfig, SVDLinear


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def hyperparameters(X, amplitudes, noise_fraction):
    """Training energy moment under a zero-mean iid prior; noise fraction is assumed."""
    if not np.isfinite(noise_fraction) or not 0 < noise_fraction < 1:
        raise ValueError('noise_fraction must be between 0 and 1')
    energy = float(np.mean(np.square(amplitudes)))
    probe_energy = float(np.mean(np.sum(np.abs(np.asarray(X, dtype=np.complex128))**2, axis=1)))
    if not np.isfinite(energy) or energy <= 0 or probe_energy <= 0:
        raise ValueError('Positive finite training and probe energies required')
    noise = noise_fraction * energy
    return dict(noise_variance=noise, prior_variance=(energy - noise) / probe_energy)


def fit_rows(train, config, output):
    """Accept only a training split. Invalid masks get explicit unsupported statuses."""
    cfg = VAMPConfig(**{k: v for k, v in config.items() if k not in ('seed', 'noise_fraction')})
    fraction = config['noise_fraction']
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('noise_fraction must be between 0 and 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'status.json').exists():
        raise FileExistsError('Choose a fresh output directory; existing checkpoints are preserved')
    records = [dict(detector_id=int(row), status='pending') for row in train.detector_ids]
    summary = dict(status='running', attempted_rows=len(records), rows=records,
                   solver_config=asdict(cfg), seed=config['seed'], noise_fraction=fraction,
                   likelihood='Rician amplitude; total circular complex field-noise variance',
                   mismatch='Quantized camera counts; Rician field noise is not a verified sensor model',
                   hyperparameter_policy='noise = fixed fraction * mean training intensity; prior = (mean intensity - noise) / mean probe energy; no validation/test tuning')
    write_json(output / 'status.json', summary)
    start = time.perf_counter()
    linear = SVDLinear(train.X)
    summary['prepare_seconds'] = time.perf_counter() - start
    summary['svd_count'] = 1
    summary['training_shape'] = list(train.X.shape)
    summary['singular_value_range'] = [float(linear.s.min()), float(linear.s.max())]
    amplitudes = train.amplitudes
    # Compute shared probe energy once (outside per-row timings).
    probe_energy = float(np.sum(linear.s**2) / linear.P)
    for index, record in enumerate(records):
        start = time.perf_counter()
        row = record['detector_id']
        seed = [int(config['seed']), row]
        record['seed'] = seed
        record['status'] = 'running'
        write_json(output / 'status.json', summary)
        try:
            if not train.valid[:, index].all():
                record['status'] = 'unsupported_mask'
            else:
                y = amplitudes[:, index]
                energy = float(np.mean(y**2))
                params = dict(noise_variance=fraction * energy,
                              prior_variance=(1 - fraction) * energy / probe_energy)
                record.update(params)
                model = PrVAMP(linear, config=cfg, **params)
                result = model.fit(y, seed=seed)
                if not np.isfinite(result.x).all():
                    raise FloatingPointError('Nonfinite estimate')
                estimate_file = f'row_{row}.npz'
                np.savez(output / estimate_file, A_row=result.x, detector_id=row)
                write_json(output / f'row_{row}_history.json', result.history)
                residual = np.abs(result.z) - y
                record.update(status=result.status, converged=result.converged,
                              iterations=len(result.history), estimate_file=estimate_file,
                              training_squared_error=float(residual @ residual),
                              training_amplitude_energy=float(y @ y),
                              training_sample_count=len(y),
                              training_relative_amplitude_residual=float(np.linalg.norm(residual) / np.linalg.norm(y)),
                              stabilized_messages=sum(h['stabilized'] for h in result.history))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['solve_seconds'] = time.perf_counter() - start
        write_json(output / 'status.json', summary)
        print(f"detector {row}: {record['status']} ({record['solve_seconds']:.2f}s)", flush=True)
    summary['status'] = 'complete'
    summary['finite_estimate_rows'] = sum('estimate_file' in r for r in records)
    summary['converged_rows'] = sum(r.get('converged', False) for r in records)
    summary['finite_estimate_coverage'] = summary['finite_estimate_rows'] / len(records)
    write_json(output / 'status.json', summary)
    return summary


def run(manifest_path, data_dir, policy_path, config_path, output, method='prvamp'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    start = time.perf_counter()
    tracemalloc.start()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    dataset = load(data_dir, manifest, policy)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'inspection.json', dataset.report)
    def git(*args):
        return subprocess.check_output(['git', '-c', f'safe.directory={Path.cwd().as_posix()}', *args], text=True).strip()
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    platform=platform.platform(), device='CPU', git_commit=git('rev-parse', 'HEAD'),
                    git_dirty=bool(git('status', '--porcelain')),
                    threads={k: os.getenv(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                    config=config, manifest=manifest, policy=policy, method=method)
    write_json(output / 'provenance.json', versions)
    if method == 'prvamp':
        summary = fit_rows(dataset.train, config, output)
    else:
        from tmbench.bayes_fit import fit_rows as fit_bayes
        summary = fit_bayes(dataset.train, config, output, method)
    summary['end_to_end_seconds'] = time.perf_counter() - start
    summary['peak_traced_bytes'] = tracemalloc.get_traced_memory()[1]
    summary['memory_note'] = 'Python/NumPy traced allocation peak, not process peak RSS or native BLAS workspace'
    tracemalloc.stop()
    write_json(output / 'status.json', summary)
    return 0 if summary['finite_estimate_rows'] == summary['attempted_rows'] else 1
