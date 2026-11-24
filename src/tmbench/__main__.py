"""Run project checks from the command line."""

import argparse
import json
from pathlib import Path


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
    check = commands.add_parser("check-files", help="Check that calibration files are present")
    check.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/rice_phase_slm_40x40.json"),
    )
    check.add_argument("--data-dir", type=Path, default=Path("data/raw/phase_slm_40x40"))
    args = parser.parse_args()

    try:
        return check_files(args.manifest, args.data_dir)
    except (OSError, ValueError, KeyError, AttributeError) as error:
        parser.exit(2, f"Error checking calibration files: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
