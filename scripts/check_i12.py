"""Independently check I12 coverage, stored coefficients, and reported errors."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from run_i12 import AMP_METHODS, MATCHED, METHODS, cases, image_target, read_json, simulation_methods


def coefficient_error(x, target):
    total = np.vdot(target, target).real
    return max(0., float((np.vdot(x, x).real + total - 2*abs(np.vdot(x, target))) / total))


def check_simulations(output, cfg):
    count = 0
    for suite in ('timing', 'figures'):
        for case in cases(cfg, suite):
            study, regime, modes, ratio, sigma, trial, group, image = case
            path = output/'simulation'/study/f'{trial:02}.json'
            record = read_json(path)
            seed = [cfg['seed'], group, trial]
            assert record['seed'] == seed and record['sigma'] == sigma
            assert record['modes'] == modes and record['patterns'] == modes * ratio
            targets = np.random.SeedSequence(seed).spawn(3)
            rng = np.random.default_rng(targets[0])
            target = image_target(modes) if image else (
                rng.standard_normal(modes) + 1j*rng.standard_normal(modes)
            ) / np.sqrt(2)
            expected = simulation_methods(case, cfg)
            assert len(expected) == len(record['records'])
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as estimates:
                existing = set()
                for (method, variance), result in zip(expected, record['records']):
                    key = f'{method}_{variance}'
                    assert result['method'] == method
                    assert result['variance_policy'] == (variance if method in AMP_METHODS else 'none')
                    assert result['finite'] == (key in estimates)
                    if result['finite']:
                        existing.add(key)
                        x = estimates[key]
                        assert x.shape == (modes,) and np.isfinite(x).all()
                        error = coefficient_error(x, target)
                        np.testing.assert_allclose(error, result['phase_nmse'], rtol=1e-9, atol=1e-10)
                        if regime != 'gaussian':
                            ambiguity = min(error, coefficient_error(x.conj(), target))
                            np.testing.assert_allclose(ambiguity, result['phase_conjugation_nmse'],
                                                       rtol=1e-9, atol=1e-10)
                    count += 1
                assert set(estimates.files) == existing
    return count


def check_rice(output, pilot, estimates_destination=None):
    count = 0
    matrices, keys, statuses = [], [], []
    path = output/'rice_data.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pilot['cache_sha256']
    with np.load(path, allow_pickle=False) as data:
        ids = data['detector_ids']
        assert ids.tolist() == pilot['source_policy']['detector_ids']
        assert pilot['detector'] == ids[0]
        for arm, methods in [('nominal', METHODS), ('matched', MATCHED)]:
            for method in methods:
                records, estimates, evaluated_indices = [], [], []
                saved_statuses = []
                for index, detector in enumerate(ids):
                    path = output/'rice'/arm/method/f'{int(detector)}.json'
                    record = read_json(path)
                    assert record['detector'] == detector and record['method'] == method and record['arm'] == arm
                    count += 1
                    saved_statuses.append(record['status'])
                    if not record['finite']:
                        assert not path.with_suffix('.npz').exists()
                        continue
                    with np.load(path.with_suffix('.npz'), allow_pickle=False) as matrix:
                        x = matrix['A_row']
                    assert x.shape == (data['train_X'].shape[1],) and np.isfinite(x).all()
                    estimates.append(x)
                    records.append(record)
                    evaluated_indices.append(index)
                if estimates_destination is not None:
                    matrix = np.full((len(ids), data['train_X'].shape[1]), np.nan + 1j*np.nan)
                    if estimates:
                        matrix[evaluated_indices] = estimates
                    matrices.append(matrix)
                    keys.append((arm, method))
                    statuses.append(saved_statuses)
                if not estimates:
                    continue
                rows = np.stack(estimates)
                for name in ('train', 'validation', 'test'):
                    X = data[f'{name}_X'].astype(float)
                    predicted = np.abs(X @ rows.real.T + 1j*(X @ rows.imag.T))
                    observed = data[f'{name}_Y'][:, evaluated_indices]
                    masks = data[f'{name}_valid'][:, evaluated_indices]
                    for column, record in enumerate(records):
                        used = masks[:, column]
                        target = observed[used, column]
                        residual = predicted[used, column] - target
                        report = record[name]
                        assert report['count'] == len(target)
                        np.testing.assert_allclose(report['energy'], target @ target, rtol=1e-12)
                        np.testing.assert_allclose(report['squared_error'], residual @ residual,
                                                   rtol=1e-9, atol=1e-10)
    if estimates_destination is not None:
        with estimates_destination.open('wb') as handle:
            np.savez_compressed(handle, A=np.stack(matrices), detector_ids=ids,
                                arms=np.array([arm for arm, _ in keys]),
                                methods=np.array([method for _, method in keys]), statuses=np.array(statuses))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rice-estimates', type=Path)
    args = parser.parse_args()
    cfg = read_json(args.output/'provenance.json')['config']
    simulated = check_simulations(args.output, cfg)
    measured = check_rice(args.output, read_json(args.output/'rice_pilot.json'), args.rice_estimates)
    print(f'Verified coverage and errors for {simulated} simulated fits and {measured} measured detector fits.')


if __name__ == '__main__':
    main()
