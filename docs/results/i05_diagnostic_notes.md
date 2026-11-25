# Training-only variance-collapse diagnostic

The saved i05_training_burn_in_diagnostic.json is an unsuccessful exploratory
check, separate from the reproducible I05 benchmark run. It used the same frozen
training split and first detector (8208), seed [20261124, 8208, 501], assumed
initial noise 0.01*mean_intensity and prior 0.99*mean_intensity/probe_energy.

Its temporary variant of `prvbem` at source commit 96a783b changed only the
schedule: both `learn_noise` and `learn_prior` updates required `iteration > 10`,
and stopping was prohibited before iteration 11. It ran 30 iterations with
min_iter=10, tol=1e-7 and damping=1. The temporary variance_burn_in config field
in that JSON described this experimental variant; it is **not a supported option
of the released solver**. All coordinate, phase, variance and bound equations
were unchanged. To reproduce the diagnostic, apply those two EM gates and the
stopping gate to an isolated copy of that function and use the recorded inputs.

Training amplitude residual rose from approximately 0.42 at the end of the
fixed-variance interval to 0.72 at iteration 30. The schedule therefore did not
resolve the high-noise solution and was removed. No validation or test response
was loaded into this diagnostic function or used to select its setting.

The separate original I06 pilot trial was intentionally interrupted. It retains
five completed estimates, one interrupted row and 26 unrun rows in its full
32-row denominator. Completed-row, pilot and preparation timings are available;
an exact end-to-end time for the interrupted process is not claimed. I05 and
this initial I06 trial overlapped on the shared computer, so their wall times
are execution diagnostics, not a controlled speed comparison.
