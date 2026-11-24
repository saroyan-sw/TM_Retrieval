"""Load the paired PhaseSLM calibration measurements from the Rice release."""

import hashlib
import math
from pathlib import Path

import numpy as np

from tmbench.data import CalibrationDataset, CalibrationSplit


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _sample_axis(shape: tuple, features: int, axis: int | None) -> int:
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Expected a nonempty 2D array, got {shape}")
    if axis is not None:
        if axis not in (0, 1) or shape[1 - axis] != features:
            raise ValueError(f"Sample axis {axis} does not match {shape} and {features} features")
        return axis
    axes = [i for i in (0, 1) if shape[1 - i] == features]
    if len(axes) != 1:
        raise ValueError(f"Cannot infer the sample axis for {shape} with {features} features")
    return axes[0]


def _select_variable(arrays: dict, path: Path, spec: dict, features: int) -> str:
    explicit = spec.get("variable")
    if explicit is not None:
        if explicit not in arrays:
            raise ValueError(f"Variable {explicit!r} is missing in {path.name}")
        return explicit
    for name in (path.stem, path.stem.rsplit("_", 1)[0]):
        if name in arrays:
            return name
    candidates = [
        name for name, shape in arrays.items() if len(shape) == 2 and features in shape
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify a unique array in {path.name}. Available arrays: {arrays}. "
            "Set its variable in file_layout."
        )
    return candidates[0]


def _read_mat(
    path: Path, spec: dict, features: int, detector_ids: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """Read one vector-valued array and return it in sample-major order."""
    with path.open("rb") as stream:
        header = stream.read(128)
    hdf5 = header.startswith((b"MATLAB 7.3 MAT-file", b"\x89HDF\r\n\x1a\n"))

    if hdf5:
        try:
            import h5py
        except ImportError as error:
            raise ValueError("MATLAB v7.3 requires the hdf5 extra. Install tmbench[hdf5].") from error

        with h5py.File(path, "r") as handle:
            shapes = {
                name: array.shape
                for name, array in handle.items()
                if isinstance(array, h5py.Dataset)
            }
            name = _select_variable(shapes, path, spec, features)
            axis = _sample_axis(shapes[name], features, spec.get("sample_axis"))
            dataset = handle[name]
            if detector_ids is None:
                values = dataset[...]
            elif axis == 0:
                values = dataset[:, detector_ids]
            else:
                values = dataset[detector_ids, :]
        stored_dtype = str(values.dtype)
        if set(values.dtype.names or ()) == {"real", "imag"}:
            values = values["real"] + 1j * values["imag"]
    else:
        try:
            from scipy.io import loadmat, whosmat
        except ImportError as error:
            raise ValueError("Classic MAT files require scipy. Install the project dependencies.") from error

        shapes = {name: shape for name, shape, _ in whosmat(path, appendmat=False)}
        name = _select_variable(shapes, path, spec, features)
        axis = _sample_axis(shapes[name], features, spec.get("sample_axis"))
        values = loadmat(path, appendmat=False, variable_names=[name])[name]
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected a dense array for {name!r} in {path.name}")
        stored_dtype = str(values.dtype)
        if detector_ids is not None:
            values = np.take(values, detector_ids, axis=1 - axis)

    info = {
        "filename": path.name,
        "format": "hdf5_mat" if hdf5 else "classic_mat",
        "variable": name,
        "stored_shape": list(shapes[name]),
        "stored_dtype": stored_dtype,
        "sample_axis_in_storage": axis,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    return values if axis == 0 else values.T, info


def _decode_patterns(values: np.ndarray, encoding: str) -> np.ndarray:
    if values.dtype.kind not in "biufc" or not np.isfinite(values).all():
        raise ValueError("SLM patterns must contain finite numeric entries")
    if encoding == "signed_binary_field":
        if np.any(np.abs(values.imag) > 1e-7) or not np.all(
            np.isclose(np.abs(values.real), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError("Expected signed binary fields (-1/+1). Verify the released encoding.")
        return np.where(values.real > 0, np.int8(1), np.int8(-1))
    if encoding == "binary_phase_0_pi":
        if values.dtype.kind == "c" or not np.isin(values, (0, 1)).all():
            raise ValueError("Expected binary phase codes (0/1 for 0/pi)")
        return np.where(values == 0, np.int8(1), np.int8(-1))
    raise ValueError(f"Unsupported pattern encoding {encoding!r}")


def _load_pair(
    split: str, manifest: dict, data_dir: Path, detector_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    pair = manifest["calibration_pairs"][split]
    layout = manifest.get("file_layout", {})
    patterns, pattern_info = _read_mat(
        data_dir / pair["patterns"],
        layout.get(pair["patterns"], {}),
        manifest["input_modes"],
    )
    observations, observation_info = _read_mat(
        data_dir / pair["observations"],
        layout.get(pair["observations"], {}),
        manifest["detector_count"],
        detector_ids,
    )
    if len(patterns) != len(observations):
        raise ValueError(
            f"{split} pairing failed: {len(patterns)} patterns and {len(observations)} camera frames"
        )
    if observations.dtype.kind not in "iuf":
        raise ValueError("Camera intensities must be real numeric values")
    encoding = manifest["pattern_encoding"]
    return _decode_patterns(patterns, encoding), observations, {
        "patterns": pattern_info,
        "observations": observation_info,
        "paired_count": len(patterns),
    }


def _probe_ids(patterns: np.ndarray) -> list[bytes]:
    # Binary fields differing by a global sign produce identical intensities.
    return [np.packbits(probe == probe[0]).tobytes() for probe in patterns]


def _split_training(
    train_patterns: np.ndarray, test_patterns: np.ndarray, policy: dict
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    test_keys = set(_probe_ids(test_patterns))
    groups: dict[bytes, list[int]] = {}
    excluded = []
    for index, key in enumerate(_probe_ids(train_patterns)):
        if key in test_keys:
            excluded.append(index)
        else:
            groups.setdefault(key, []).append(index)
    fraction = policy["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if len(groups) < 2:
        raise ValueError("Need at least two distinct training probes after testing for overlap")
    count = min(len(groups) - 1, math.ceil(fraction * len(groups)))
    order = np.random.default_rng(policy["seed"]).permutation(len(groups))
    members = list(groups.values())
    validation = sorted(index for group in order[:count] for index in members[group])
    training = sorted(index for group in order[count:] for index in members[group])
    return np.asarray(training), np.asarray(validation), excluded


def _select(
    X: np.ndarray, intensities: np.ndarray, indices: np.ndarray, detector_ids: np.ndarray
) -> CalibrationSplit:
    selected = intensities[indices]
    valid = np.isfinite(selected) & (selected >= 0)
    return CalibrationSplit(X[indices], selected, valid, indices, detector_ids)


def load(data_dir: Path, manifest: dict, split_policy: dict) -> CalibrationDataset:
    """Load the Rice release and resolve its predeclared split and detector policy."""
    if split_policy["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Split policy and dataset manifest refer to different datasets")
    if manifest["observation_domain"] != "intensity":
        raise ValueError("The Rice adapter requires a manifest declaring intensity observations")
    detector_ids = np.asarray(split_policy["detector_ids"])
    if (
        detector_ids.ndim != 1
        or len(detector_ids) == 0
        or detector_ids.dtype.kind not in "iu"
        or detector_ids[0] < 0
        or detector_ids[-1] >= manifest["detector_count"]
        or np.any(detector_ids[1:] <= detector_ids[:-1])
    ):
        raise ValueError("Detector IDs must be sorted, unique, and within the detector range")

    train_X, train_I, train_info = _load_pair("train", manifest, data_dir, detector_ids)
    test_X, test_I, test_info = _load_pair("test", manifest, data_dir, detector_ids)
    train_ids, validation_ids, excluded = _split_training(train_X, test_X, split_policy)
    train = _select(train_X, train_I, train_ids, detector_ids)
    validation = _select(train_X, train_I, validation_ids, detector_ids)
    test = _select(test_X, test_I, np.arange(len(test_X)), detector_ids)
    report = {
        "dataset_id": manifest["dataset_id"],
        "source": manifest["source"],
        "observation_domain": manifest["observation_domain"],
        "intensity_units": manifest.get("intensity_units"),
        "release_preprocessing": manifest.get("release_preprocessing"),
        "pattern_encoding": manifest["pattern_encoding"],
        "files": {"train": train_info, "test": test_info},
        "pairing_check": "pattern and camera sample counts agree in release order",
        "detector_ids": detector_ids.tolist(),
        "split_policy": split_policy,
        "training_rows_excluded_for_test_overlap": excluded,
        "splits": {
            name: {
                "source_indices": split.source_indices.tolist(),
                "pattern_shape": list(split.X.shape),
                "intensity_shape": list(split.intensities.shape),
                "valid_observations": int(split.valid.sum()),
                "negative_observations": int(np.count_nonzero(
                    np.isfinite(split.intensities) & (split.intensities < 0)
                )),
                "nonfinite_observations": int(np.count_nonzero(~np.isfinite(split.intensities))),
            }
            for name, split in (("train", train), ("validation", validation), ("test", test))
        },
    }
    return CalibrationDataset(train, validation, test, report)
