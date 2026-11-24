import numpy as np
import pytest
from tmbench.data import CalibrationSplit
from tmbench.evaluation.metrics import evaluate


def test_complex_plain_transpose_masks_and_failed_denominator():
    X = np.array([[1j, 1], [1, -1j]])
    a = np.array([1+2j, 2-1j])
    y = np.abs(X @ a)
    split = CalibrationSplit(X, np.column_stack([y**2, y**2]), np.array([[True, True], [False, True]]), np.arange(2), np.array([10, 11]))
    result = evaluate(split, [a, None], ['max_iter_reached', 'failed'])
    assert result['coverage'] == .5
    assert result['valid_count'] == 1
    assert result['amplitude_mse'] == pytest.approx(0)
    assert result['rows'][1]['solver_status'] == 'failed'
