"""Blocks 3 and 4: two outputs of ONE shared Gaussian linear belief."""

from dataclasses import dataclass
import numpy as np
from .moments import positive_scalar


def matvec(matrix, vector):
    """Avoid copying an entire real matrix to complex for each product."""
    if not np.iscomplexobj(matrix) and np.iscomplexobj(vector):
        return matrix @ vector.real + 1j * (matrix @ vector.imag)
    return matrix @ vector


@dataclass
class LinearEstimate:
    x: np.ndarray
    z: np.ndarray
    variance_x: float
    variance_z: float
    alpha: float
    beta: float


class SVDLinear:
    """Cache one economy SVD and reuse it across iterations and TM rows.

    phi has shape (P, N). Handles tall, wide and rank-deficient matrices.
    No dense posterior covariance or matrix inverse is formed in solve().
    """

    def __init__(self, phi):
        phi = np.asarray(phi, dtype=np.complex128 if np.iscomplexobj(phi) else np.float64)
        if phi.ndim != 2 or min(phi.shape) == 0 or not np.all(np.isfinite(phi)):
            raise ValueError("phi must be a finite, nonempty 2D matrix.")
        if not np.any(phi):
            raise ValueError("phi must contain at least one nonzero entry.")
        self.phi = phi.copy()
        self.P, self.N = phi.shape
        self.U, self.s, self.Vh = np.linalg.svd(phi, full_matrices=False)
        self.Uh = self.U.conj().T
        self.V = self.Vh.conj().T

    def forward(self, x):
        return matvec(self.phi, x)

    def solve(self, r2, gamma2, p2, tau2):
        gamma2 = positive_scalar(gamma2, "gamma2")
        tau2 = positive_scalar(tau2, "tau2")
        r2 = np.asarray(r2, dtype=np.complex128)
        p2 = np.asarray(p2, dtype=np.complex128)
        if r2.shape != (self.N,) or p2.shape != (self.P,):
            raise ValueError("r2 must have shape (N,) and p2 shape (P,).")
        if not np.all(np.isfinite(r2)) or not np.all(np.isfinite(p2)):
            raise ValueError("Linear message centers must be finite.")
        d = gamma2 + tau2 * self.s**2
        r_rotated = matvec(self.Vh, r2)
        p_rotated = matvec(self.Uh, p2)
        x_rotated = (gamma2 * r_rotated + tau2 * self.s * p_rotated) / d
        x = matvec(self.V, x_rotated)
        # For P < N the economy Vh omits N-P nullspace directions.
        # Their mean stays r2 and their variance stays 1/gamma2.
        missing = self.N - self.s.size
        if missing:
            x += r2 - matvec(self.V, r_rotated)
        z = matvec(self.U, self.s * x_rotated)
        variance_x = float((np.sum(1.0 / d) + missing / gamma2) / self.N)
        variance_z = float(np.sum(self.s**2 / d) / self.P)
        alpha = gamma2 * variance_x
        beta = tau2 * variance_z
        return LinearEstimate(x, z, variance_x, variance_z, alpha, beta)

    def dense_covariances(self, gamma2, tau2):
        """Teaching/debugging only: explicitly form Cx and Cz for small examples."""
        gamma2 = positive_scalar(gamma2, "gamma2")
        tau2 = positive_scalar(tau2, "tau2")
        precision = gamma2 * np.eye(self.N) + tau2 * self.phi.conj().T @ self.phi
        Cx = np.linalg.solve(precision, np.eye(self.N))
        return Cx, self.phi @ Cx @ self.phi.conj().T
