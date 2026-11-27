"""Run or summarize the checkpointed ICCP simulation and measured Rice campaigns."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from tmbench.adapters.rice_phase import load
from tmbench.benchmark import AMP_METHODS, PAPER_ITERATIONS, fit, method_family, prepare
from tmbench.benchmark import prediction_metrics, training_variances
from tmbench.fit import write_json
from tmbench.solvers.bayes import Design, prvbem
from tmbench.solvers.prvamp.linear import matvec
from tmbench.solvers.prvamp.utils import complex_normal, phase_nmse


METHODS = tuple(PAPER_ITERATIONS)
MATCHED = tuple(method for method in METHODS if method not in ('gs100', 'wf500'))
RICE_MANIFEST = Path('data/manifests/rice_phase_slm_40x40.json')
RICE_POLICY = Path('configs/public/rice_phase_slm_40x40.json')
RICE_DATA = Path('data/raw/phase_slm_40x40')
BLAS_ENV = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cases(cfg, suite):
    p_per_n = cfg['patterns_per_mode']
    n = cfg['oversampling_modes']
    if suite == 'timing':
        for modes in cfg['timing_modes']:
            for trial in range(cfg['timing_trials']):
                yield (f'timing_gaussian_{modes}', 'gaussian', modes, p_per_n, 0., trial, 40 + modes, False)
        return
    for ratio in cfg['oversampling_ratios']:
        for trial in range(cfg['sweep_trials']):
            yield (f'ratio_gaussian_{ratio:02}', 'gaussian', n, ratio, 0., trial, 1, False)
    for regime, seed in [('amplitude', 2), ('phase', 3)]:
        for noise in cfg['noise_levels']:
            for trial in range(cfg['sweep_trials']):
                yield (f'noise_{regime}_{noise:g}', regime, n, p_per_n, noise, trial, seed, False)
    for regime, seed in [('gaussian', 4), ('amplitude', 5), ('phase', 6)]:
        for trial in range(cfg['image_trials']):
            yield (f'image_{regime}', regime, cfg['image_modes'], p_per_n, 0., trial, seed, True)


def image_target(modes):
    size = int(np.sqrt(modes))
    if size**2 != modes:
        raise ValueError('Image cases need a square number of modes')
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    radius = np.hypot(x, y)
    outline = (radius >= .72) & (radius <= .87)
    eyes = (np.hypot(np.abs(x)-.27, y+.26) <= .10)
    mouth = (np.abs(np.hypot(x, y+.05)-.46) <= .06) & (y > .17) & (np.abs(x) < .45)
    image = (outline | eyes | mouth).astype(float).ravel()
    return image / np.sqrt(np.mean(image**2))


def simulate(case, cfg):
    _, regime, modes, ratio, sigma, trial, group, image = case
    seed = [cfg['seed'], group, trial]
    streams = np.random.SeedSequence(seed).spawn(3)
    truth = image_target(modes) if image else complex_normal(np.random.default_rng(streams[0]), modes)
    rng = np.random.default_rng(streams[1])
    # Generate the largest paired design before slicing so ratios share a prefix.
    p = modes * max(cfg['patterns_per_mode'], ratio)
    if regime == 'gaussian':
        X = complex_normal(rng, (p, modes))
    else:
        X = rng.integers(0, 2, (p, modes), dtype=np.int8).astype(float)
        if regime == 'phase':
            X = 2*X - 1
    X = X[:modes*ratio]
    field = X @ truth
    y = np.abs(field + complex_normal(np.random.default_rng(streams[2]), len(X), sigma**2))
    variance = max(sigma**2, 1e-12 * np.mean(np.abs(field)**2))
    return X, y, truth, (float(variance), float(np.mean(np.abs(truth)**2))), seed


def write_arrays(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def peak_working_set():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (key, ctypes.c_size_t) for key in ('peak', 'current', 'peak_paged', 'paged',
                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')
            ]

        counter = Counters(size=ctypes.sizeof(Counters))
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.WinDLL('psapi').GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.size):
            return counter.peak
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if platform.system() == 'Darwin' else 1024))


def timed_prepare(family, X):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        model, error = prepare(family, X), None
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        model, error = None, str(exc)
    return model, dict(prepare_cpu_seconds=time.process_time()-cpu,
                       prepare_wall_seconds=time.perf_counter()-wall), error


def attempt(method, model, y, seed, variances, args, setup_error):
    if setup_error is not None:
        return None, dict(status='unsupported_design', reason=setup_error, converged=False, iterations=0)
    start, cpu = time.perf_counter(), time.process_time()
    try:
        result = fit(method, model, y, seed, variances, **args)
        return result.x, result.metrics
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return None, dict(status='failed', reason=str(error), converged=False, iterations=0,
                          solve_cpu_seconds=time.process_time()-cpu, solve_wall_seconds=time.perf_counter()-start)


def simulation_methods(case, cfg):
    study = case[0]
    methods = [(method, 'oracle') for method in METHODS]
    if study == f"ratio_gaussian_{cfg['patterns_per_mode']:02}":
        methods.extend((method, 'training_energy') for method in METHODS if method in AMP_METHODS)
    return methods


def run_simulation(case, cfg, output):
    study, _, modes, ratio, sigma, trial, _, image = case
    folder = output / 'simulation' / study
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{trial:02}.json'
    if path.exists():
        return 0
    X, y, truth, oracle, seed = simulate(case, cfg)
    estimated = training_variances(X, y, cfg['assumed_training_noise_fraction'])
    records, estimates = [], {}
    previous_family = None
    for method, policy in simulation_methods(case, cfg):
        family = method_family(method)
        if family != previous_family:
            if previous_family is not None:
                del model
            model, costs, error = timed_prepare(family, X)
        previous_family = family
        variances = oracle if policy == 'oracle' else estimated
        options = dict(simulated=True, max_iter=PAPER_ITERATIONS[method],
                       cpu_seconds=cfg['simulation_cpu_limit_seconds'])
        x, result = attempt(method, model, y, [*seed, 12], variances, options, error)
        record = dict(method=method, variance_policy=policy if method in AMP_METHODS else 'none',
                      **costs, **result, finite=x is not None)
        if method in AMP_METHODS:
            record.update(initial_noise_variance=variances[0], initial_prior_variance=variances[1])
        if x is not None:
            record['phase_nmse'] = phase_nmse(x, truth)
            if np.isrealobj(X):
                record['phase_conjugation_nmse'] = phase_nmse(x, truth, allow_conjugation=True)
            estimates[f'{method}_{policy}'] = x
        records.append(record)
    if estimates:
        write_arrays(path.with_suffix('.npz'), **estimates)
    write_json(path, dict(study=study, trial=trial, seed=seed, modes=modes, patterns=modes*ratio,
                          sigma=sigma, image=image, worker_peak_rss_bytes=peak_working_set(), records=records))
    return 1


def make_rice_cache(cfg, output):
    cache = output / 'rice_data.npz'
    metadata = output / 'rice_pilot.json'
    if cache.exists() and metadata.exists():
        saved = read_json(metadata)
        if hashlib.sha256(cache.read_bytes()).hexdigest() != saved['cache_sha256']:
            raise ValueError('Saved Rice data cache hash mismatch')
        return saved
    manifest, policy = read_json(RICE_MANIFEST), read_json(RICE_POLICY)
    start = time.perf_counter()
    dataset = load(RICE_DATA, manifest, policy)
    load_seconds = time.perf_counter()-start
    train = dataset.train
    arrays = dict(detector_ids=train.detector_ids)
    for name in ('train', 'validation', 'test'):
        split = getattr(dataset, name)
        arrays.update({f'{name}_X': split.X, f'{name}_Y': split.amplitudes,
                       f'{name}_valid': split.valid, f'{name}_indices': split.source_indices})
    write_arrays(cache, **arrays)
    wall, cpu = time.perf_counter(), time.process_time()
    design = Design(train.X, gram=True)
    prepare_cpu = time.process_time()-cpu
    y = arrays['train_Y'][:, 0]
    noise, prior = training_variances(design.X, y, cfg['assumed_training_noise_fraction'])
    detector = int(train.detector_ids[0])
    result = prvbem(design, y, noise, prior, dict(max_iter=50, min_iter=50, learn_noise=True,
                                                 learn_prior=True), [cfg['seed'], detector, 501])
    report = dict(detector=detector, estimate_norm=float(np.linalg.norm(result.x)),
                  training_relative_residual=result.history[-1]['relative_amplitude_residual'],
                  initial_noise=noise, initial_prior=prior, noise_variance=result.noise_variance,
                  prior_variance=result.prior_variance, status=result.status, iterations=len(result.history),
                  prepare_cpu_seconds=prepare_cpu, pilot_and_preparation_cpu_seconds=time.process_time()-cpu,
                  pilot_and_preparation_wall_seconds=time.perf_counter()-wall, load_wall_seconds=load_seconds,
                  cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(), source_manifest=manifest,
                  source_policy=policy, split_manifest_sha256=hashlib.sha256(
                      Path(policy['frozen_manifest']).read_bytes()).hexdigest())
    write_json(metadata, report)
    return report


def run_rice(family, indices, cfg, output, pilot):
    with np.load(output/'rice_data.npz', allow_pickle=False) as data:
        ids = data['detector_ids']
        entries = [
            (method, arm, index, output/'rice'/arm/method/f'{int(ids[index])}.json')
            for method in METHODS if method_family(method) == family
            for arm in ('nominal', 'matched') if arm == 'nominal' or method in MATCHED
            for index in indices
        ]
        pending = [item for item in entries if not item[-1].exists()]
        if not pending:
            return 0
        train, validation, test = (
            (data[f'{split}_X'].astype(float), data[f'{split}_Y'], data[f'{split}_valid'])
            for split in ('train', 'validation', 'test')
        )
    model, costs, error = timed_prepare(family, train[0])
    fixed = (pilot['noise_variance'], pilot['prior_variance'])
    for method, arm, index, path in pending:
        detector = int(ids[index])
        y = train[1][:, index]
        local = training_variances(train[0], y, cfg['assumed_training_noise_fraction'])
        variances = fixed if method in AMP_METHODS and method != 'prvbem' else local
        options = dict(
            simulated=False,
            max_iter=PAPER_ITERATIONS[method] if arm == 'nominal' else cfg['rice_matched_max_iterations'],
            cpu_seconds=cfg[f'rice_{arm}_cpu_limit_seconds'],
        )
        setup_error = error if train[2][:, index].all() else 'Unsupported training observation mask'
        seed = [cfg['seed'], detector, 12]
        x, result = attempt(method, model, y, seed, variances, options, setup_error)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(method=method, arm=arm, detector=detector, seed=seed, finite=x is not None,
                      work_group=f'{family}_{indices[0]:02}', **costs, **result)
        if method in AMP_METHODS:
            record.update(initial_noise_variance=variances[0], initial_prior_variance=variances[1])
        if x is not None:
            for split, values in [('train', train), ('validation', validation), ('test', test)]:
                X, observed, valid = values
                record[split] = prediction_metrics(X, x, observed[:, index], valid[:, index])
            write_arrays(path.with_suffix('.npz'), A_row=x)
        record['worker_peak_rss_bytes'] = peak_working_set()
        write_json(path, record)
    return len(pending)


def run_jobs(jobs, workers, label):
    finished = written = 0
    if not jobs:
        return
    if workers == 1:
        for function, args in jobs:
            written += function(*args)
            finished += 1
            if finished == len(jobs) or finished % 10 == 0:
                print(f'{label}: {finished}/{len(jobs)} jobs, {written} new records', flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(function, *args) for function, args in jobs]
        for future in as_completed(pending):
            written += future.result()
            finished += 1
            if finished == len(jobs) or finished % 10 == 0:
                print(f'{label}: {finished}/{len(jobs)} jobs, {written} new records', flush=True)


def statistics(values):
    if not values:
        return None
    lo, median, hi = np.quantile(values, [.25, .5, .75])
    return dict(median=float(median), q25=float(lo), q75=float(hi))


def summarize_group(rows, attempted):
    group = dict(attempted=attempted, recorded=len(rows), finite=sum(record['finite'] for record in rows),
                 converged=sum(record['converged'] for record in rows),
                 statuses=dict(Counter(record['status'] for record in rows)))
    group['finite_coverage'] = group['finite']/attempted
    fields = ('phase_nmse', 'phase_conjugation_nmse', 'solve_cpu_seconds', 'solve_wall_seconds',
              'initialization_cpu_seconds', 'prepare_cpu_seconds', 'constraint_violation', 'iterations')
    for key in fields:
        stats = statistics([record[key] for record in rows if record.get(key) is not None])
        if stats is not None:
            group[key] = stats
    memory = [record['worker_peak_rss_bytes'] for record in rows if record.get('worker_peak_rss_bytes')]
    if memory:
        group['max_worker_peak_rss_bytes'] = max(memory)
    for split in ('train', 'validation', 'test'):
        valid = [record[split] for record in rows if record.get(split)]
        if not valid:
            continue
        sse, energy, count = (sum(record[key] for record in valid) for key in ('squared_error', 'energy', 'count'))
        group[split] = dict(evaluated=len(valid), observations=count, amplitude_mse=sse/count,
                            relative_residual=float(np.sqrt(sse/energy)) if energy else None)
    images = [(record['phase_nmse'], record['trial']) for record in rows
              if record.get('image') and record.get('phase_nmse') is not None]
    if images:
        images.sort()
        group['median_error_image_trial'] = images[(len(images)-1)//2][1]
    return group


def summarize(cfg, output, destination=None, trials_archive=None):
    expected = Counter()
    measured_expected = {
        (arm, method): len(read_json(RICE_POLICY)['detector_ids'])
        for arm in ('nominal', 'matched') for method in (METHODS if arm == 'nominal' else MATCHED)
    }
    for suite in ('figures', 'timing'):
        for case in cases(cfg, suite):
            for method, variance in simulation_methods(case, cfg):
                expected[(case[0], method, variance if method in AMP_METHODS else 'none')] += 1
    simulated, measured, raw_records = defaultdict(list), defaultdict(list), []
    for path in sorted((output/'simulation').glob('*/*.json')):
        trial = read_json(path)
        for row in trial.pop('records'):
            record = dict(trial, **row)
            simulated[(record['study'], record['method'], record['variance_policy'])].append(record)
            raw_records.append(record)
    for path in sorted((output/'rice').glob('*/*/*.json')):
        record = read_json(path)
        measured[(record['arm'], record['method'])].append(record)
        raw_records.append(record)
    result = dict(
        metadata=read_json(output/'provenance.json'),
        simulation=[dict(study=study, method=method, variance_policy=variance,
                         **summarize_group(simulated[(study, method, variance)], count))
                    for (study, method, variance), count in sorted(expected.items())],
        rice=[dict(arm=arm, method=method, **summarize_group(measured[(arm, method)], count))
              for (arm, method), count in sorted(measured_expected.items())],
    )
    pilot = output/'rice_pilot.json'
    if pilot.exists():
        result['rice_pilot'] = read_json(pilot)
    result['completion'] = {
        name: {key: sum(record[key] for record in result[name]) for key in ('attempted', 'recorded', 'finite')}
        for name in ('simulation', 'rice')
    }
    print(json.dumps(result['completion'], indent=2))
    if destination:
        write_json(destination, result)
    if trials_archive:
        with trials_archive.open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as archive:
                for record in raw_records:
                    archive.write((json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode())
    return result


def provenance(cfg, output):
    paths = sorted(Path('src/tmbench').rglob('*.py')) + [
        Path(__file__).resolve(), RICE_MANIFEST, RICE_POLICY, Path(read_json(RICE_POLICY)['frozen_manifest']),
    ]
    source = hashlib.sha256()
    for path in paths:
        source.update(hashlib.sha256(path.read_bytes()).digest())
    signature = dict(config=cfg, source_sha256=source.hexdigest(),
                     platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                     python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                     thread_settings={name: os.environ.get(name) for name in BLAS_ENV},
                     memory_measure='Maximum OS-reported working set over each worker lifetime')
    path = output/'provenance.json'
    if path.exists() and read_json(path) != signature:
        raise ValueError('Benchmark provenance changed. Use a fresh output directory.')
    if not path.exists():
        write_json(path, signature)


def simulation_jobs(cfg, suite, output):
    planned = list(cases(cfg, suite))
    # Large image trials begin first so the long cases overlap the sweep.
    planned.sort(key=lambda case: (not case[-1], -case[3], case[0], case[5]))
    return [(run_simulation, (case, cfg, output)) for case in planned
            if not (output/'simulation'/case[0]/f'{case[5]:02}.json').exists()]


def rice_jobs(cfg, output):
    pilot = make_rice_cache(cfg, output)
    detectors = len(pilot['source_policy']['detector_ids'])
    chunk = cfg['rice_detector_chunk_size']
    by_family = ('phaselift', 'prsamp', 'prvamp', 'prgamp', 'phasemax', 'gs', 'wf', 'prvbem')
    return [(run_rice, (family, list(range(first, min(first+chunk, detectors))), cfg, output, pilot))
            for first in range(0, detectors, chunk) for family in by_family]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('all', 'figures', 'rice', 'timing', 'report'))
    parser.add_argument('--config', type=Path, default=Path('configs/benchmarks/iccp2017_i12.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.action == 'timing' and args.workers != 1:
        parser.error('Timing requires one worker. Other suites require a positive worker count.')
    cfg = read_json(args.config)
    output = args.output
    if args.action == 'report':
        if read_json(output/'provenance.json')['config'] != cfg:
            parser.error('Report configuration differs from the run configuration')
        summarize(cfg, output, args.summary, args.archive)
        return
    if any(os.environ.get(key) != '1' for key in BLAS_ENV):
        parser.error('Set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and OMP_NUM_THREADS to 1')
    output.mkdir(parents=True, exist_ok=True)
    provenance(cfg, output)
    if args.action == 'all':
        run_jobs(simulation_jobs(cfg, 'timing', output), 1, 'timing')
        jobs = rice_jobs(cfg, output) + simulation_jobs(cfg, 'figures', output)
    elif args.action in ('figures', 'timing'):
        jobs = simulation_jobs(cfg, args.action, output)
    else:
        jobs = rice_jobs(cfg, output)
    run_jobs(jobs, args.workers, args.action)
    summarize(cfg, output, args.summary, args.archive)


if __name__ == '__main__':
    main()
