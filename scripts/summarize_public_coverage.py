"""Rebuild the I03–I10 public coverage manifest from verified run archives."""

from collections import Counter
import json
from pathlib import Path

from tmbench.fit import write_json


CHECKPOINT_RUNS = {
    'prvamp': ('i03_prvamp',),
    'prgamp': ('i04_prgamp',),
    'prvbem': ('i05_prvbem',),
    'prsamp': ('i06_prsamp', 'i06_prsamp_fixed'),
    'gs': ('i07_gs50', 'i07_gs100'),
    'wf': ('i08_wf_original', 'i08_wf_adaptive'),
    'phaselift': ('i09_phaselift',),
    'phasemax': ('i10_phasemax',),
}


def summarize(directory):
    method_coverage = []
    dataset = None
    for method, names in CHECKPOINT_RUNS.items():
        runs = []
        for name in names:
            path = directory / f'{name}_summary.json'
            archive = json.loads(path.read_text(encoding='utf-8'))
            provenance = archive['provenance']
            status, evaluation = archive['status'], archive['evaluation']
            signature = (provenance['manifest'], provenance['policy'], status['training_shape'])
            if dataset is None:
                dataset = signature
            if signature != dataset or evaluation['method'] != method:
                raise ValueError(f'{name}: unexpected dataset or method')
            if not archive['report_replay_byte_identical']:
                raise ValueError(f'{name}: report replay was not verified')
            detector_ids = [row['detector_id'] for row in status['rows']]
            if detector_ids != provenance['policy']['detector_ids'] or len(detector_ids) != status['attempted_rows']:
                raise ValueError(f'{name}: attempted detectors do not match the frozen selection')
            finite = sum('estimate_file' in row for row in status['rows'])
            if finite != status['finite_estimate_rows']:
                raise ValueError(f'{name}: estimate count does not match row artifacts')
            coverage = {}
            for split in ('validation', 'test'):
                result = evaluation['metrics'][split]
                if result['attempted_rows'] != len(detector_ids):
                    raise ValueError(f'{name}: {split} denominator does not match detector selection')
                coverage[split] = dict(evaluated_rows=result['evaluated_rows'], coverage=result['coverage'])
            runs.append(dict(run=name, summary_file=path.name, status=status['status'],
                             attempted_rows=len(detector_ids), finite_estimate_rows=finite,
                             converged_rows=status['converged_rows'],
                             row_status_counts=dict(sorted(Counter(row['status'] for row in status['rows']).items())),
                             held_out=coverage, reason=status.get('reason') or status.get('stop_reason')))
        method_coverage.append(dict(method=method, runs=runs))

    manifest, policy, training_shape = dataset
    return dict(dataset_id=manifest['dataset_id'], frozen_split=policy['frozen_manifest'],
                training_shape=training_shape, detector_ids=policy['detector_ids'],
                scope='Archived held-out runs and size rejections from I03 through I10',
                methods=method_coverage)


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent.parent / 'docs' / 'results'
    write_json(directory / 'public_method_coverage.json', summarize(directory))
