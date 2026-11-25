"""Training-only variance pilots; never accepts a dataset or held-out split."""
import time
import hashlib
import numpy as np
from .solvers.bayes import prvbem


def estimate_variances(train, design, config, checkpoint=None):
    """Use the first fixed detector IDs, without score-based selection or fallback."""
    count = config['rows']
    if not isinstance(count, int) or not 1 <= count <= len(train.detector_ids):
        raise ValueError('Pilot row count must be within fixed detector selection')
    start = time.perf_counter()
    amplitudes = train.amplitudes
    cfg = dict(max_iter=config['max_iter'], min_iter=config.get('min_iter', 1),
               tol=config.get('tol', 1e-7), damping=1., learn_noise=True, learn_prior=True)
    records = []
    summary = dict(status='running', selection='first fixed detector IDs; no score-based selection',
                   attempted_rows=count, rows=records, config=config,
                   training_sample_count=len(train.source_indices),
                   training_source_indices_sha256=hashlib.sha256(np.asarray(train.source_indices, dtype='<i8').tobytes()).hexdigest(),
                   aggregation='median noise/mean_intensity and median prior*mean_probe_energy/mean_intensity',
                   failure_policy='any failed/unsupported pilot makes pilot unavailable; no hidden fallback')
    for index in range(count):
        row, tick = int(train.detector_ids[index]), time.perf_counter()
        record = dict(detector_id=row, seed=[int(config['seed']), row, 501])
        try:
            if not train.valid[:, index].all():
                raise ValueError('Pilot has unsupported mask')
            y = amplitudes[:, index]
            energy = float(np.mean(y**2))
            fraction = config.get('initial_noise_fraction', .01)
            if not 0 < fraction < 1:
                raise ValueError('Invalid initial pilot noise fraction')
            result = prvbem(design, y, fraction*energy, (1-fraction)*energy/design.probe_energy, cfg, record['seed'])
            record.update(status=result.status, iterations=len(result.history), converged=result.converged,
                          noise_variance=result.noise_variance, prior_variance=result.prior_variance,
                          noise_fraction=result.noise_variance/energy,
                          prior_factor=result.prior_variance*design.probe_energy/energy,
                          history=result.history)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            record.update(status='failed', error=str(error))
        record['seconds'] = time.perf_counter()-tick
        records.append(record)
        summary['seconds'] = time.perf_counter()-start
        if checkpoint:
            checkpoint(summary)
    if any(r['status'] == 'failed' for r in records):
        summary['status'] = 'failed'
    else:
        summary.update(status='complete', noise_fraction=float(np.median([r['noise_fraction'] for r in records])),
                       prior_factor=float(np.median([r['prior_factor'] for r in records])),
                       converged_rows=sum(r['converged'] for r in records))
    summary['seconds'] = time.perf_counter()-start
    if checkpoint:
        checkpoint(summary)
    return summary
