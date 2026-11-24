# I02 source and mathematical audit

Source: user-supplied prVAMP_TM_Retrieval_Python.zip, SHA256
3f9629f73d21dc1e8d523a3f66424bfb6d2cdbf4b960871b36b0e7a22eaef3f6.
Imported six modules from prvamp_project/prvamp and both original test modules.
The archive identifies itself as a teaching implementation reconstructed from
the user's derivations, not a bit-for-bit original MATLAB port. No license file
was included; do not infer an upstream open-source license.

Local verification reruns the supplied independent Cartesian integration of
Rician moments, finite-difference complex divergence, direct dense linear solve
and covariance trace comparisons, nullspace handling, natural-parameter damping,
complex transpose convention, stabilization, reproducibility and recovery checks.
Small numerical recovery fixtures are tests, not public benchmark observations.

All variances are total complex variances E|u-Eu|^2. Each real component has half
that variance. Rician output moments use scaled Bessel functions. Sensitivity
above one is possible and the supplied stabilization clips it; stabilization
counts are retained. The Gaussian prior's extrinsic factor is exactly the prior.
The rectangular trace identity N*alpha2 + P*beta2 = N is independently tested.
Damping is the fraction of the new Gaussian natural parameters, not the old.

Adaptations: PrVAMP accepts a prepared SVDLinear object so per-row hyperparameters
do not cause repeated decompositions. Real probes keep float64 factors and use
two real matrix-vector products for complex vectors; this avoids full complex
matrix copies on every iteration. Adjoint factors are cached. Training residuals
use the already-computed linear field (mathematically X@x). A parity test compares
real-factor and complex-factor iteration results. No update equations changed.

The sample-major adapter fits one output column at a time and returns a complex
TM row with all input modes. It takes only the training split. The fixed config
uses 100 iterations, random initialization, seed [20261124, detector_id], damping
0.5 and tolerance 1e-7. Each row uses zero prior mean, noise variance equal to
0.01 times mean training intensity, and prior variance (mean training intensity
minus noise variance)/mean probe energy. The 0.01 fraction is a predeclared
assumption, not an estimated camera noise level. The energy-based prior is a
training-only moment heuristic, not a validated empirical-Bayes optimum.

Released uint8 camera intensities do not establish a circular Gaussian field
noise model. This likelihood mismatch is explicit in every status report.
Validation/test intensities are not used for hyperparameters, initialization,
fitting, restart selection or stopping. Held-out evaluation remains I03.

Missing-data masks are explicitly unsupported by this first adapter; affected
rows remain in the attempted denominator. Each row writes its estimate, history,
status, parameters, timing and training diagnostic immediately. Pending/running
statuses survive interruptions. Existing run directories cannot be overwritten.
Iteration-limit estimates are kept but never labelled converged. Convergence
does not establish predictive quality or a correct TM.
