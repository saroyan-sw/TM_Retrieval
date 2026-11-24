"""Exercise MAT orientation and held-out grouping with small calibration fixtures."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tmbench.__main__ import inspect_data
from tmbench.adapters.rice_phase import _decode_patterns, _sample_axis, _split_training, load


TRAIN_PATTERNS = np.array([
    [1, 1, 1], [-1, -1, -1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
])
TEST_PATTERNS = np.array([[-1, 1, 1]])
MANIFEST = {
    "dataset_id": "rice_fixture",
    "source": {},
    "input_modes": 3,
    "detector_count": 4,
    "pattern_encoding": "signed_binary_field",
    "observation_domain": "intensity",
    "calibration_pairs": {
        "train": {"patterns": "XH_train.mat", "observations": "YH_squared_train.mat"},
        "test": {"patterns": "XH_test.mat", "observations": "YH_squared_test.mat"},
    },
}
POLICY = {
    "dataset_id": "rice_fixture",
    "validation_fraction": 0.25,
    "seed": 42,
    "detector_ids": [1, 3],
}


class SplitTests(unittest.TestCase):
    def test_global_sign_and_test_overlap_stay_out_of_fitting(self):
        train, validation, excluded = _split_training(TRAIN_PATTERNS, TEST_PATTERNS, POLICY)
        self.assertEqual(excluded, [4])
        self.assertEqual(set(train) | set(validation), {0, 1, 2, 3})
        self.assertEqual(0 in train, 1 in train)
        self.assertTrue(set(train).isdisjoint(validation))
        repeated_train, repeated_validation, _ = _split_training(
            TRAIN_PATTERNS, TEST_PATTERNS, POLICY
        )
        np.testing.assert_array_equal(train, repeated_train)
        np.testing.assert_array_equal(validation, repeated_validation)

    def test_ambiguous_storage_and_unverified_codes_require_explicit_settings(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            _sample_axis((3, 3), 3, None)
        self.assertEqual(_sample_axis((3, 3), 3, 1), 1)
        with self.assertRaisesRegex(ValueError, "Verify the released encoding"):
            _decode_patterns(np.array([[0, 1]]), "signed_binary_field")
        np.testing.assert_array_equal(
            _decode_patterns(np.array([[0, 1]]), "binary_phase_0_pi"), [[1, -1]]
        )


def write_pairs(directory, write):
    train_intensities = np.arange(20, dtype=float).reshape(5, 4)
    test_intensities = np.array([[25.0, -1.0, 36.0, np.nan]])
    write(directory / "XH_train.mat", "XH", TRAIN_PATTERNS)
    write(directory / "YH_squared_train.mat", "YH_squared", train_intensities.T)
    write(directory / "XH_test.mat", "XH_test", TEST_PATTERNS.T)
    write(directory / "YH_squared_test.mat", "YH_squared_test", test_intensities)
    return train_intensities


class MatTests(unittest.TestCase):
    def assert_pairing(self, dataset, train_intensities):
        for split in (dataset.train, dataset.validation):
            np.testing.assert_array_equal(split.X, TRAIN_PATTERNS[split.source_indices])
            np.testing.assert_array_equal(
                split.intensities, train_intensities[split.source_indices][:, [1, 3]]
            )
            np.testing.assert_allclose(split.amplitudes, np.sqrt(split.intensities))
        np.testing.assert_array_equal(dataset.test.X, TEST_PATTERNS)
        self.assertEqual(dataset.test.intensities[0, 0], -1)
        self.assertTrue(np.isnan(dataset.test.intensities[0, 1]))
        self.assertFalse(dataset.test.valid.any())
        self.assertTrue(np.isnan(dataset.test.amplitudes).all())

    def test_dataset_assembly_preserves_source_rows_and_selected_observations(self):
        train_I = np.arange(20, dtype=float).reshape(5, 4)
        files = {
            "XH_train.mat": TRAIN_PATTERNS,
            "XH_test.mat": TEST_PATTERNS,
            "YH_squared_train.mat": train_I,
            "YH_squared_test.mat": np.array([[25.0, -1.0, 36.0, np.nan]]),
        }

        def read(path, spec, features, selected=None):
            values = files[path.name]
            self.assertEqual(values.shape[1], features)
            return (values if selected is None else values[:, selected]), {}

        with patch("tmbench.adapters.rice_phase._read_mat", side_effect=read):
            dataset = load(Path(), MANIFEST, POLICY)
        self.assert_pairing(dataset, train_I)
        self.assertEqual(dataset.report["training_rows_excluded_for_test_overlap"], [4])
        self.assertEqual(dataset.report["splits"]["test"]["valid_observations"], 0)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy required")
    def test_classic_mat_and_inspection_report(self):
        from scipy.io import savemat

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train_I = write_pairs(directory, lambda path, name, data: savemat(path, {name: data}))
            dataset = load(directory, MANIFEST, POLICY)
            self.assert_pairing(dataset, train_I)
            info = dataset.report["files"]["train"]["observations"]
            self.assertEqual((info["variable"], info["sample_axis_in_storage"]), ("YH_squared", 1))
            self.assertEqual(len(info["sha256"]), 64)
            for name, data in (("manifest", MANIFEST), ("policy", POLICY)):
                (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
            report = directory / "results" / "inspection.json"
            self.assertEqual(
                inspect_data(directory / "manifest.json", directory, directory / "policy.json", report),
                0,
            )
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), dataset.report)
            savemat(directory / "YH_squared_test.mat", {"YH_squared_test": np.ones((2, 4))})
            with self.assertRaisesRegex(ValueError, "test pairing failed"):
                load(directory, MANIFEST, POLICY)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py required")
    def test_hdf5_mat_detector_slices_and_compound_field(self):
        import h5py

        def write(path, name, values):
            if name == "XH":
                complex_values = np.empty(values.shape, dtype=[("real", "f8"), ("imag", "f8")])
                complex_values["real"] = values
                complex_values["imag"] = 0
                values = complex_values
            with h5py.File(path, "w", userblock_size=512) as handle:
                handle[name] = values.T
            with path.open("r+b") as stream:
                stream.write(b"MATLAB 7.3 MAT-file")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train_I = write_pairs(directory, write)
            dataset = load(directory, MANIFEST, POLICY)
            self.assert_pairing(dataset, train_I)
            self.assertEqual(dataset.report["files"]["train"]["patterns"]["format"], "hdf5_mat")
