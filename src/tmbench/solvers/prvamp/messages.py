"""Gaussian information subtraction and explicit numerical safeguards."""

from dataclasses import dataclass
import numpy as np
from .moments import positive_scalar


@dataclass
class Message:
    center: np.ndarray
    precision: float
    sensitivity_used: float
    stabilized: bool


def extrinsic(mean, incoming_center, incoming_precision, sensitivity,
              *, stabilize=False, epsilon=1e-8,
              precision_min=1e-12, precision_max=1e12):
    """Divide the moment-matched Gaussian by its incoming Gaussian.

    Exact when 0 < sensitivity < 1, unless explicit stabilization is active.
    stabilize=True clips sensitivity and precision to keep a usable message.
    This is an engineering approximation; Message.stabilized reports it.
    Clipping is never performed inside the exact moment functions.
    """
    precision = positive_scalar(incoming_precision, "incoming_precision")
    mean = np.asarray(mean, dtype=np.complex128)
    incoming = np.asarray(incoming_center, dtype=np.complex128)
    if mean.shape != incoming.shape or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(incoming)):
        raise ValueError("mean and incoming_center must have the same finite shape.")
    a = float(sensitivity)
    if not np.isfinite(a):
        raise FloatingPointError("Nonfinite sensitivity.")
    if not 0 < epsilon < 0.5 or not 0 < precision_min < precision_max:
        raise ValueError("Invalid stabilization bounds.")
    if not stabilize and not 0 < a < 1:
        raise ValueError("Positive Gaussian quotient requires 0 < sensitivity < 1.")
    used = float(np.clip(a, epsilon, 1.0 - epsilon)) if stabilize else a
    outgoing = precision * (1.0 - used) / used
    bounded = float(np.clip(outgoing, precision_min, precision_max)) if stabilize else outgoing
    center = (mean - used * incoming) / (1.0 - used)
    if not np.isfinite(bounded) or not np.all(np.isfinite(center)):
        raise FloatingPointError("Nonfinite extrinsic message; check scale and hyperparameters.")
    return Message(center, bounded, used, used != a or bounded != outgoing)


def damp_message(old_center, old_precision, new_center, new_precision, step):
    """Geometric damping: interpolate Gaussian natural parameters.

    step=1 accepts the new message; step=0.5 takes a halfway information step.
    This convention is stated explicitly, rather than inferred from a paper's
    damping number. All damping in this project uses this convention.
    """
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1].")
    positive_scalar(old_precision, "old_precision")
    positive_scalar(new_precision, "new_precision")
    precision = (1 - step) * old_precision + step * new_precision
    information = ((1 - step) * old_precision * old_center
                   + step * new_precision * new_center)
    return information / precision, float(precision)
