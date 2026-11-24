"""Sample-major calibration data shared by dataset adapters and solvers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CalibrationSplit:
    """Known probes and measured intensities for one split.

    The axes of `X` are patterns and input modes. The axes of `intensities`
    and `valid` are patterns and selected detector outputs. Source indices
    and detector IDs refer to zero-based indices in the original release.
    """

    X: NDArray
    intensities: NDArray
    valid: NDArray[np.bool_]
    source_indices: NDArray
    detector_ids: NDArray

    @property
    def amplitudes(self) -> NDArray[np.float64]:
        """Return square-root amplitudes with NaN for invalid measurements."""
        amplitudes = np.full(self.intensities.shape, np.nan)
        np.sqrt(self.intensities, out=amplitudes, where=self.valid, dtype=np.float64)
        return amplitudes


@dataclass(frozen=True)
class CalibrationDataset:
    """Loaded splits and their reproducibility report."""

    train: CalibrationSplit
    validation: CalibrationSplit
    test: CalibrationSplit
    report: dict
