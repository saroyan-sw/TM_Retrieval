"""Simulation and phase-aware diagnostics; no ground truth enters the solver."""

import numpy as np


def complex_normal(rng, shape, variance=1.0):
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative.")
    return np.sqrt(variance / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def align_phase(estimate, reference):
    """Align a single vector by a unit-modulus factor; never rescale it."""
    estimate = np.asarray(estimate, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have matching shapes.")
    overlap = np.vdot(estimate, reference)
    rotation = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
    return rotation * estimate


def phase_nmse(estimate, reference, *, allow_conjugation=False):
    """NMSE up to global phase; optionally account for REAL-probe ambiguity."""
    reference = np.asarray(reference, dtype=np.complex128)
    energy = float(np.linalg.norm(reference)**2)
    if energy == 0:
        raise ValueError("Reference must have nonzero energy.")
    estimate = np.asarray(estimate, dtype=np.complex128)
    error = np.linalg.norm(align_phase(estimate, reference) - reference)**2 / energy
    if allow_conjugation:
        conjugate_error = np.linalg.norm(align_phase(estimate.conj(), reference) - reference)**2 / energy
        error = min(error, conjugate_error)
    return float(error)


def amplitude_residual(phi, x, y):
    denominator = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
    return float(np.linalg.norm(np.abs(phi @ x) - y) / denominator)
