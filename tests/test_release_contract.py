import numpy as np
import pytest
from tmbench.adapters.rice_phase import _read_mat, _select


def test_checksum_rejects_modified_release(tmp_path):
    from scipy.io import savemat
    path = tmp_path / 'X.mat'
    savemat(path, {'X': np.ones((5, 3))})
    with pytest.raises(ValueError, match='Checksum mismatch'):
        _read_mat(path, {'sha256': '0' * 64}, 3)


def test_saturation_preserves_counts_but_masks_amplitudes():
    intensity = np.array([[0, 254, 255, -1, np.nan]])
    data = _select(np.ones((1, 2)), intensity, np.array([0]), np.arange(5), 255)
    np.testing.assert_array_equal(data.valid, [[True, True, False, False, False]])
    assert data.intensities[0, 2] == 255
    assert data.amplitudes[0, 0] == 0
    assert np.isnan(data.amplitudes[0, 2:]).all()
