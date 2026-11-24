"""Run project checks from the command line."""

import argparse
import json
import os
from pathlib import Path

DEFAULT_MANIFEST = Path("data/manifests/rice_phase_slm_40x40.json")
DEFAULT_DATA_DIR = Path("data/raw/phase_slm_40x40")


def check_files(manifest_path: Path, data_dir: Path) -> int:
    """Report missing or empty calibration files listed in the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    any_problem = False
    for split, pair in manifest["calibration_pairs"].items():
        for role, filename in pair.items():
            path = data_dir / filename
            if not path.is_file():
                print(f"{split} {role}: missing {path}")
                any_problem = True
                continue
            size = path.stat().st_size
            if size == 0:
                print(f"{split} {role}: empty {path}")
                any_problem = True
                continue
            print(f"{split} {role}: {path} ({size:,} bytes)")
    return int(any_problem)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tmbench")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser('fit-prvamp', help='Fit fixed detector rows using training measurements only')
    fit.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    fit.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    fit.add_argument('--split-policy', type=Path, default=Path('configs/public/rice_phase_slm_40x40.json'))
    fit.add_argument('--config', type=Path, default=Path('configs/public/prvamp_i02.json'))
    fit.add_argument('--output', type=Path, default=Path('runs/prvamp_i02'))
    check = commands.add_parser("check-files", help="Check that calibration files are present")
    inspect = commands.add_parser(
        "inspect-data", help="Load Rice calibration pairs and write an inspection report"
    )
    for command in (check, inspect):
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    inspect.add_argument(
        "--split-policy", type=Path, default=Path("configs/public/rice_phase_slm_40x40.json")
    )
    inspect.add_argument(
        "--report", type=Path, default=Path("runs/rice_phase_slm_40x40/inspection.json")
    )
    args = parser.parse_args()

    try:
        if args.command == 'fit-prvamp':
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
                os.environ.setdefault(key, '4')
            from tmbench.fit import run
            return run(args.manifest, args.data_dir, args.split_policy, args.config, args.output)
        if args.command == "check-files":
            return check_files(args.manifest, args.data_dir)
        if check_files(args.manifest, args.data_dir):
            return 1
        return inspect_data(args.manifest, args.data_dir, args.split_policy, args.report)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, MemoryError) as error:
        detail = str(error) or type(error).__name__
        parser.exit(2, f"Error reading calibration data: {detail}\n")


def inspect_data(manifest_path: Path, data_dir: Path, policy_path: Path, report_path: Path) -> int:
    """Verify the calibration pairs and save provenance and source indices."""
    from tmbench.adapters.rice_phase import load

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    report = load(data_dir, manifest, policy).report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pairing: {report['pairing_check']}")
    for name, split in report["splits"].items():
        print(
            f"{name}: X {tuple(split['pattern_shape'])}, "
            f"I {tuple(split['intensity_shape'])}, "
            f"{split['valid_observations']:,} valid observations"
        )
    print(f"Training rows excluded for test overlap: {len(report['training_rows_excluded_for_test_overlap'])}")
    print(f"Inspection report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
