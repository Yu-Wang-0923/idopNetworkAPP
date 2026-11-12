# Dynamic solver: idopECG(4)

Source: user-supplied `idopECG(4).zip`, `idopECG/src/idopECG.py` (Yu Wang).
The numerical functions are in `idopnetwork.network.ecg_solver`. Plot imports,
CLI execution and automatic filesystem writes are excluded from web execution.
An explicit `output_path` still writes the original tables for library users.

The Dynamic construction route now uses the complete SWT-smoothed record.
Sliding-window LASSO standardizes both predictors and response, sets alpha to
`alpha_ratio * alpha_max`, and retains support frequencies strictly greater than
the threshold. Defaults: ratio 0.1, window `max(4, N//5)`, stride `max(1, window//4)`,
threshold 0.9. These windows select edges; they do not split the fit.

Self terms use an integrated Fourier basis, with order automatically set from
a 10 Hz bandwidth and record duration. Cross terms use centered Legendre
polynomials P1–P3 of the source signal. Interaction contributions are allocated
first by sequential orthogonalization and Shapley averaging over source orders;
the self term then fits the remaining response. Ridge defaults to 1e-6.

Predictions, observations and cumulative effects share the complete physical
time grid and original first-sample initial value. An explicit `time` column,
or an index named `time`, supplies seconds; otherwise the sampling rate is used.
There is no best-window selection, cross-window averaging or discarded tail.

The native network table retains nonnegative energy-share weights and independent
signs from rate/source regression slopes. For the application's signed network
and GLMY interfaces, adjacency stores `sign * weight`. The raw table and all
coefficients, self/cross effects, reconstruction, diagnostics, orthogonalization,
settings, support sets and selection frequencies are exported under `dynamic/`.
Static ASGL and static LASSO routes retain their existing definitions.

Old dynamic session results must be recomputed after this update. Historical
window-average tests are replaced by full-record/additive-identity, source-order,
physical-time, energy-share/sign, empty-support and invalid-input tests. Local
differential verification against the supplied source compares every result
table exactly. The user-requested 12-channel, 100 Hz ECG example is bundled as
`idopnetwork_app/data/ecg_100hz.csv`, with a time-in-seconds index added for the
Curve Fitting uploader; the original twelve signal columns are retained.
