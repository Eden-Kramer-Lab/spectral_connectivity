# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**The Granger and transfer-function measures changed direction.** `result[...,
i, j]` from the `Connectivity` methods of the spectral Granger family, directed
transfer function, directed coherence, (generalized) partial directed coherence,
and direct directed transfer function is now `i -> j` (2.x returned `j -> i`).
For pairwise and subset spectral Granger prediction, the directed measures the
2.x wrapper accepted, `multitaper_connectivity(...).sel(source="a", target="b")`
is now `a -> b` (2.x returned `b -> a`). Indexing code written for 2.x still
runs but reads the opposite direction; see the first item of the
[migration guide](https://spectral-connectivity.readthedocs.io/en/latest/migration.html). `phase_slope_index`, `group_delay`, and `delay` were already source
first; do not swap them. Until 3.2 the `Connectivity` methods
that existed in 2.x (pairwise and subset spectral Granger, directed transfer
function, directed coherence, (generalized) partial directed coherence, and
direct directed transfer function) emit a `DirectedOrientationWarning` as a
reminder, as do `multitaper_connectivity` and `connectivity_to_xarray` for
pairwise and subset spectral Granger prediction, once per session for each
measure; silence it with `warnings.filterwarnings("ignore",
category=spectral_connectivity.DirectedOrientationWarning)`.
`Connectivity.power()` is one-sided: its bins between 0 Hz and Nyquist are twice
the 2.x values. Until 3.2 it emits a `SidednessWarning` (once per session)
when called without `one_sided` on two-sided coefficients; the wrappers' power
results changed the same way without a warning (see the
[migration guide](https://spectral-connectivity.readthedocs.io/en/latest/migration.html)).

This release includes corrected numerical definitions and therefore requires a
major version bump. Recompute affected results rather than comparing them
directly with results from 2.x.

### Migration guide

The migration guide for code written against 2.x is
[its own page in the documentation](https://spectral-connectivity.readthedocs.io/en/latest/migration.html)
(`docs/migration.md` in the repository). The `DirectedOrientationWarning` and
`SidednessWarning` messages link to that page.

### Added

- `multitaper_connectivity` and `connectivity_to_xarray` take `dtype`,
  `minimum_phase_tolerance`, and `minimum_phase_max_iterations`, as
  `fourier_connectivity` and `Connectivity.from_transform` do: the working
  precision and the Wilson factorization's tolerance and iteration cap for
  the directed measures. In `connectivity_kwargs` or `transform_kwargs` they
  raise `ValueError` naming the argument to use.
- `Connectivity.time_reversed_conditional_spectral_granger_prediction()` and
  `Connectivity.time_reversed_blockwise_spectral_granger_prediction(group_labels)`,
  also available in the wrappers: conditional and blockwise spectral Granger of
  the time-reversed data (the transposed cross-spectral matrix), the
  counterparts of `time_reversed_spectral_granger_prediction` for checking
  whether a direction reverses under time reversal (Winkler et al., 2016).
- `partial_coherence(shrinkage=...)` shrinks each cross-spectral matrix
  toward its diagonal before inverting it, multiplying every coherency by
  `1 - lambda`: a fixed `lambda` in [0, 1], or `"auto"`, the Ledoit-Wolf /
  Schäfer-Strimmer intensity estimated for each time window and frequency.
  `Connectivity.partial_coherence_shrinkage()` returns those intensities, for
  reporting. The default, `None`, does not shrink; the wrappers take it in
  `connectivity_kwargs`. Use `"auto"` with at most about 2.5 trials x tapers
  per signal; with more it can raise the error (see the `partial_coherence`
  Notes).
- `adjust_for_multiple_comparisons(..., axis=...)`: the axes whose tests form
  one family, as in `scipy.stats.false_discovery_control`, so each slice
  along the other axes (each time window, say) is corrected on its own. The
  default, `axis=None`, still pools every p-value into one family.
- `group_labels` of `multitaper_connectivity` and `fourier_connectivity` can be
  the name of a coordinate along the input DataArray's signal dimension, e.g.
  `group_labels="region"`, whose values then label the signals. A string
  given with a plain array, or to `connectivity_to_xarray`, now raises
  `TypeError` saying so, where it raised a shape `ValueError`.
- `Connectivity.power(one_sided=False)` returns the two-sided power spectral
  density at the non-negative frequencies (no bin doubled), which is what
  `power()` returned in 2.x and what formulas written for a two-sided
  spectrum take. The default is the one-sided density, as before in 3.0. It
  raises for one-sided coefficients, whose two-sided scale is not known, and
  the labelled results of the wrappers stay one-sided: they refuse
  `connectivity_kwargs={"one_sided": False}` and point to the method.
- `DirectedOrientationWarning`, a temporary `UserWarning` (planned removal in
  3.2) emitted by the seven `Connectivity` methods whose arrays were target
  first in 2.x, by `Connectivity.jackknife` of those measures, and by
  `multitaper_connectivity` and `connectivity_to_xarray` when they compute
  pairwise or subset spectral Granger prediction (the 2.x
  wrapper rejected the transfer-function measures). It points at the calling
  line and appears once per Python session for each measure, and for each
  way of computing it: once for calls through `multitaper_connectivity` or
  `connectivity_to_xarray` (about `sel(source, target)`) and once for calls
  to the `Connectivity` method (about the array's axes), so a script that
  uses both sees both, once each. `warnings.simplefilter("always",
  spectral_connectivity.DirectedOrientationWarning)` shows it at every call,
  and any other filter on it (such as `"error"`) applies as usual. A call
  made while it is ignored does not count as shown, so removing the filter
  shows it once; a call turned into an error counts, since the exception
  carried the message.
- `SidednessWarning`, a temporary `UserWarning` (planned removal in 3.2)
  emitted by `Connectivity.power()` called without `one_sided` on two-sided
  coefficients, and once by `jackknife("power")` for its full estimate: since
  3.0 the bins between 0 Hz and Nyquist are twice the 2.x values. It is also
  emitted by `Connectivity.global_coherence()` on two-sided coefficients,
  which since 3.0 returns only the non-negative frequencies. Each notice
  appears once per Python session, like `DirectedOrientationWarning`. Pass
  `one_sided=True` to `power` (or `False` for the 2.x values) or silence
  both with `warnings.filterwarnings("ignore",
  category=spectral_connectivity.SidednessWarning)`. The wrappers'
  results do not emit it. Both warnings' messages say that they are shown
  once per session and give the `simplefilter("always", ...)` call that shows
  them at every call.
- Public simulators with planted, known structure in
  `spectral_connectivity.simulate`: `simulate_lagged_broadband` (noisy copies of
  one white-noise source, each delayed by a whole number of samples given as
  `lags_samples`, for a known lead/lag; convert a delay in seconds with
  `round(delay * sampling_frequency)`) and `simulate_shared_oscillation` (one
  sinusoid, its `frequency` and `sampling_frequency` in Hz, shared by several
  signals with per-signal amplitude, phase offset, and noise, optionally with a
  random phase per trial). Like `simulate_MVAR`, they return
  `(time, trial, signal)` arrays with `n_trials=1` by default;
  `simulate_lagged_broadband(..., n_trials=None)` returns `(time, signal)`.
  `simulate_shared_oscillation` takes its required `amplitudes` (and its other
  per-signal settings) by keyword. Both raise `ValueError` for a sample or
  trial count that is not a positive integer.
  The simulated-examples tutorial builds every case
  with them and asserts what each plot shows, so executing it checks the plots.
- Spectral primitives and pairwise measures: one-sided
  `cross_spectral_density`, signed `imaginary_coherency`, `partial_coherence`,
  corrected imaginary PLV, and directed PLI.
- Multivariate/group measures: maximized imaginary coherency (MIC),
  multivariate interaction measure (MIM), conditional spectral Granger
  (the Chen, Bressler & Ding 2006 frequency decomposition, computed from one
  full-system and one reduced-system factorization per source), blockwise
  spectral Granger (one factorization per group pair), and time-reversed
  spectral Granger.
- Exact complex `canonical_coherency` (Vidaurre CaCoh) performs phase
  optimisation and component deflation and returns component scores, spatial
  filters, patterns, connections, and group membership. The new
  `maximized_imaginary_coherency_components` exposes the same rich result schema
  for multiple MIC singular components; the historical score-only APIs remain
  available unchanged. In the wrappers' results each connection names its
  `connection_source_group` and `connection_target_group`, and filters and
  patterns have a `side` axis, `"source"` or `"target"`, over a `signal` axis.
  `canonical_coherency` is a different estimator from the historical
  `canonical_coherence` (the squared canonical correlation), not its square
  root: its magnitude is the unsquared, phase-optimized coherence. Both are vectorized over the time/frequency axes on the
  active backend (GPU-capable), with the CaCoh phase search done as a batched
  grid-and-Newton optimization rather than a per-bin loop.
- The tuning settings of the new measures are keyword-only: `rank`,
  `regularization`, and `n_components` of `canonical_coherency`,
  `maximized_imaginary_coherency(_components)`,
  `multivariate_interaction_measure`, and `partial_coherence`, and the new
  `max_workspace_elements` of `global_coherence`. `rank` is an estimation rank
  (it restricts the whitening subspace, so it changes the scores);
  `n_components` and `global_coherence`'s released `max_rank`, which keeps its
  keyword and position, only count the components returned.
  `canonical_coherence` keeps its released signature and gains no tuning
  setting.
- `Connectivity.clear_cache()` releases the intermediates cached for reuse
  across measures (cross-spectral matrix, power, phase-lag moments, and the
  minimum-phase factorization with the Granger quantities derived from it) so
  memory-constrained runs can keep an instance without holding them.
- `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet` transforms. Each
  takes `time_series` and `sampling_frequency` positionally and every other
  setting by keyword (e.g. `MorletWavelet(x, fs, frequencies=[4, 8, 16])`);
  `Multitaper` keeps its released positional order. Morlet
  output is explicitly one-sided and rejects Wilson-factorized directed
  measures, which require a full two-sided spectrum. Morlet coefficients are
  scaled so that `power()` is the one-sided PSD in signal²/Hz, on the same
  scale as `Multitaper` (FieldTrip's convention; MNE's wavelet power is
  `2 * sampling_frequency` times larger). `ShortTimeFourierTransform`'s `repr`
  and `summarize_parameters()` describe its Hann window and equivalent noise
  bandwidth. `ShortTimeFourierTransform` and `Welch` expose
  `equivalent_noise_bandwidth`, computed from the actual window (so a
  two-sample window gives `sampling_frequency`).
  `ShortTimeFourierTransform` is not a `Multitaper` subclass, so it has none of
  the DPSS-only settings (`frequency_resolution`, `time_halfbandwidth_product`,
  `n_tapers`, `tapers`, taper weighting). `Welch` exposes its resolved settings (segment duration and step,
  overlap, detrending, start time, FFT length, workers) as read-only
  properties.
- Morlet transforms now expose strict time-frequency edge validity and support
  constant/reflect/edge convolution padding, keep/NaN/trim edge policies,
  adjacent-frequency smoothing, and boxcar or Hann time-frequency kernels.
  Connectivity expectations consume the local weights directly, and xarray
  results carry the `valid_time_frequency` mask alongside their labeled time
  and frequency coordinates.
- `SpectralTransform`, a public, runtime-checkable `typing.Protocol` for the
  transforms `Connectivity.from_transform` and `Connectivity.from_multitaper`
  accept: `fft()`, `frequencies`, and `time` are required, and the optional
  `is_one_sided`, `observation_weights`, `observations_are_independent`, and
  `time_bins_are_independent` attributes fall back to their defaults when
  absent, so a custom transform needs no new attributes. Both constructors are
  annotated with it, so mypy now checks custom transforms passed to them, and a
  `Connectivity` subclass overriding `from_multitaper` must widen its argument
  to `SpectralTransform`.
- `Connectivity` rejects input that gave silently wrong results in 2.x:
  frequencies not uniformly spaced in `numpy.fft.fftfreq` order without
  `is_one_sided=True` (2.x read `rfft` output as a two-sided spectrum and
  returned only its first half; `fourier_connectivity` infers a one-sided
  spectrum from a non-negative grid instead), and real-valued Fourier
  coefficients, for which imaginary coherence, the phase-lag index, the
  weighted phase-lag index, and the phase slope index were exactly 0 and the
  coherence phase only 0 or pi. `Connectivity`, `Connectivity.from_transform`,
  and `fourier_connectivity` reject `is_one_sided`,
  `observations_are_independent`, and `time_bins_are_independent` values that
  are not booleans (e.g. `"False"`, `None`, 0, or a transform's flag written as
  a method), which `bool()` would misread; NumPy bools and 0-d boolean arrays
  are accepted. `from_transform` does not take an `AttributeError` raised
  inside a capability property for the attribute's absence.
- Multitaper `taper_weighting` supports historical uniform weighting,
  eigenvalue weighting, and Thomson adaptive frequency/signal-specific
  weighting. Adaptive weighting compares the periodogram against the process
  noise on a matched power-spectral-density scale and warns if the iteration
  does not converge.
- `fourier_connectivity` accepts externally computed Fourier coefficients as
  NumPy arrays or labeled DataArrays. It takes `fourier_coefficients`,
  `frequencies`, `time`, `method`, and `signal_names` positionally and every
  other setting (`squeeze`, `connectivity_kwargs`, `is_one_sided`, ...) by
  keyword. Functional measures support one-sided
  coefficients (inferred from a labeled non-negative frequency coordinate or
  declared with `is_one_sided=True`), while directed factorization still
  requires a full two-sided spectrum. High-level results can be cropped,
  decimated, or reduced into named frequency bands; phase uses a circular mean
  and integration is restricted to spectral densities. `frequency_range` and
  each band of `frequency_bands` include both endpoints. Bin `f` lies on bound
  `b` when `abs(f - b) <= 1e-6 * gap(f) + 2 * (ulp(f) + ulp(b))`, with `gap(f)`
  the distance to its nearest neighbouring bin and `ulp` each value's unit in
  the last place in its own stored precision: the allowances the uniform-grid
  check uses; a bound farther from every bin is an ordinary edge between bins.
  A decimal bound names a bin only when the bin's stored value has drifted
  from that decimal by at most one part in a million of its spacing (plus
  rounding): float32 coordinates, `0.1 * k`, `np.linspace`, `rfftfreq`, and
  `np.arange(1.0, 100.0, 0.1)` stay within that, while
  `np.arange(1000.0, 1100.0, 0.001)` (drift up to 2.4e-6 of its spacing) and
  float32 cumulative sums do not; take bounds from the coordinate for those.
  The standalone
  `frequency_band_reduce(result, frequency_bands, frequency_reduction=...)`
  applies the same band reduction, with the wrappers' keywords, to an
  already-computed labeled result. Datasets holding
  spatial filters, patterns, or global-coherence vectors are rejected, since
  their sign and phase are arbitrary per frequency; reduce the score variable
  instead.
- `Connectivity.jackknife` and `jackknife_confidence_interval` provide
  leave-one-trial/taper bias correction, standard errors, and confidence
  intervals with automatic variance-stabilizing transformations: log for power,
  `atanh(sqrt(.))` for magnitude-squared coherence (`transformation=
  "fisher_squared"`), and circular for phase.
- Measures that a single observation fixes whatever the data warn there:
  coherency and everything derived from it, the phase-locking value and its
  corrected imaginary part, and the canonical coherency (magnitude forced to
  1); the phase-lag index and its weighted and directed forms (the sign of one
  imaginary cross-spectrum); the phase slope index; partial coherence; and
  global coherence, the maximized imaginary coherency, and the multivariate
  interaction measure (a rank-one cross-spectral matrix). Set `smoothing_time`
  on `MorletWavelet` or provide multiple trials/tapers. The bias-corrected
  phase measures, `delay`, and `group_delay` raise there instead, the
  spectral Granger and transfer-function measures return `NaN` with their own
  warning, and `canonical_coherence` warns whenever a group pair has more
  signals than observations. `Welch` warns when its default segment length
  is shorter than 0.25 s, reporting the resulting equivalent noise bandwidth.
  `fourier_connectivity` rejects directed
  measures on unlabeled coefficients whose two-sidedness cannot be verified.
- `multitaper_connectivity` now accepts the directed-transfer-function family
  (`directed_transfer_function`, `directed_coherence`,
  `partial_directed_coherence`, `generalized_partial_directed_coherence`,
  `direct_directed_transfer_function`) by name. They are opt-in (not in the
  default set) and, like the spectral Granger measures, are oriented so
  `sel(source=a, target=b)` is the influence from `a` to `b`.
- `multitaper_connectivity` names the common multitaper settings
  (`time_window_step`, `time_halfbandwidth_product`, `n_tapers`, and
  `detrend_type`, keyword-only, with `Multitaper`'s defaults, beside the
  released `sampling_frequency` and `time_window_duration`) and takes every
  other `Multitaper` setting in a `transform_kwargs` mapping, as measure
  settings go in `connectivity_kwargs`. The released positional order is
  unchanged. A misspelled key in either mapping raises `TypeError`; a common
  setting inside `transform_kwargs` raises `ValueError`, so a setting cannot be
  given two values. `connectivity_to_xarray` gains `connectivity_kwargs`, a
  keyword-only `group_labels` (also accepted in the mapping, not both), and a
  keyword-only `frequency_range` that, as in the other wrappers, the interval
  measures consume and every other measure is cropped to (it is rejected
  inside `connectivity_kwargs`). Every wrapper checks the name and route of
  every setting, and that `group_labels` is given once and only with a group
  measure, before computing anything. The one such check that comes later is
  `connectivity_to_xarray`'s missing-`group_labels` error, raised after the
  transform's FFT because it quotes the signal count. Provenance does not
  depend on the route that supplied a setting.
- `list_measures()` enumerates every wrapper-supported `method` name with its
  output category, a one-line description sourced from the `Connectivity`
  method docstring, and whether it is a default and/or directed measure;
  filter with `category`, `default_only`, and `directed`. An unknown `method`
  passed to `multitaper_connectivity` raises a `ValueError` that suggests the
  closest measure names and points to `list_measures()`, instead of 2.x's bare
  `AttributeError`; `fourier_connectivity` does the same.
- A doctested `docs/cookbook.md` collects short, copy-pasteable recipes for the
  most common tasks (functional and directed connectivity, reading the labeled
  output, frequency bands, and bringing your own Fourier coefficients).
- `MeasureInfo` records what each measure's values mean: `long_name`, `units`,
  `value_range`, `is_complex`, the result's `dims`, an `interpretation`, the
  `sign_convention` of measures whose sign or angle says which signal leads,
  the `diagonal_policy` (and `diagonal_value`) of its self-connections, and,
  for the coherency family, a `quantity` (`magnitude`, `squared_magnitude`,
  `signed`, `phase`, or `complex`) and `formula`. The measure table in
  `docs/CONNECTIVITY_METRIC_RANGES.md`, with its Diagonal column and a formula
  table, is generated from it. Its repr shows `name`, `category`,
  `value_range` (written as in the table), `is_directed` when true, and the
  one-line `description`, so `list_measures()` prints one line per measure;
  every field is still an attribute, and `interpretation` says how to read
  the values.
- Each measure variable of a wrapper result records how to read it in
  NetCDF-safe attributes from the same registry: `interpretation`,
  `sign_convention`, `value_range_lower`/`value_range_upper` (an unbounded side
  is omitted; complex measures add `value_range_applies_to="magnitude"`),
  `quantity` and `formula` for the coherency family, `diagonal_policy`
  (`masked_nan`, `undefined_nan`, `constant`, `rounding_noise`, `estimated`, or
  `not_applicable`, defined in the policy table of
  `docs/CONNECTIVITY_METRIC_RANGES.md`) with `diagonal_value` beside a nominal
  value, and `nan_for_zero_power_signal`. `group_delay_slope` and
  `group_delay_r_value` carry their own records. Recording the
  policy changed no diagonal value (the diagonals that differ from 2.x are in
  the migration guide); the cookbook shows how to mask or fill diagonals
  uniformly. Band reduction keeps these attributes true of the band values:
  `interpretation` and `formula` name the reduction (mean, circular mean,
  complex mean, or integral) that made each value, and the range, sign
  convention, `quantity`, and `diagonal_policy`, which hold of the band values,
  are kept.
- The docstring of each of the 35 `Connectivity` measures has a runnable
  example (the deprecated `coherence_magnitude` alias has none), and every
  directed measure states its array orientation.
- A doctested guide for AI coding assistants (`docs/llm_guide.md`) covers the
  workflow, direction conventions, parameter choice, and common pitfalls; the
  documentation site serves an `llms.txt` index, and the package docstring
  (`help(spectral_connectivity)`) points to both.
- A multi-measure `multitaper_connectivity` `Dataset` now carries the shared
  provenance (package, version, transform, backend, expectation type, observation
  counts, and transform settings) as top-level `Dataset.attrs`, not only on each
  variable.
- Result schema version 1 uses common known facts across Multitaper, STFT,
  Welch, Morlet, and external Fourier inputs, lowercase `backend="cpu"`/`"gpu"`,
  and one canonical `transform_parameters_json` record separating estimator
  and execution settings. Optional structured settings use JSON `null`.
  Durations in it are the ones used, in whole samples: a `MorletWavelet`'s
  `smoothing_time` and `smoothing_step` are its realized smoothing window and
  step (`smoothing_time=0.123` at 100 Hz after decimation records 0.12).
  Only the 15 scalar `mt_*` attributes released in 2.0.1 remain, as deprecated
  compatibility copies until 4.0. They are stored as scalars NetCDF can save:
  `mt_is_low_bias` as 0/1, `mt_start_time` and `mt_sampling_frequency` as
  floats, and a `None` setting as the string `"None"`. Multitaper's `frequency_resolution`, its
  helper names, and the helper's returned key remain supported.
- Results distinguish qualified estimator `spectral_bandwidth` from uniform
  returned-grid `frequency_bin_spacing`, refreshed after crop/decimation and
  omitted after band reduction or on singleton/irregular grids. Normalized
  external grids keep `cycles/sample` units and record no physical sampling
  rate; an external observation axis not named or declared as trials records no
  `n_trials`. Scalar recording coordinates keep their attrs; coordinates along
  averaged dimensions, such as per-trial labels, are not carried over, and one
  named like a coordinate or variable of the result is dropped with a warning
  (band reduction likewise discards existing `band`, `band_lower`,
  `band_upper`, and `valid_time_band` coordinates). Fourier coefficients
  without a time dimension use a numeric scalar coordinate named `time` or
  `times`, as left by `isel(time=0)`, as the time of their single window (a
  scalar named `window`, `windows`, or `time_window` stays an identifier: it
  may be a window's length or count). A DataArray frequency
  coordinate may spell `Hz` as `Hertz`, `cps`, `1/s`, `cycles/s`, `s-1`,
  `s^-1`, or `s**-1`; other units, such as `kHz`, are rejected and an empty
  `units` is treated as unstated. `connectivity_to_xarray` accepts any
  `SpectralTransform`; one that is not built in records its class name as
  `transform`, and its result is labeled in cycles/sample or window indices
  when it leaves `frequencies` or `time` as `None`. The shared `group_membership` variable of the component
  measures carries only the shared provenance.
- The xarray interfaces now expose every built-in nonstandard result contract:
  group-pair matrices, exact CaCoh/rich MIC components with filters, patterns,
  and membership, delay candidates, global-coherence scores/vectors,
  group-delay quantities, and frequency-reduced phase-slope matrices.
- Wrapper results carry descriptive time/frequency metadata plus NetCDF-safe
  measure, package-version, backend, expectation, transform, and method-argument
  provenance. Scalar method arguments are stored under `arg_<key>`; structured
  or non-finite values are stored as canonical, JSON-normalized data under
  `arg_<key>_json` (and `None` serializes as `"null"`), with the full mapping in
  `measure_kwargs_json`. Non-string mapping keys use a collision-safe tagged
  item representation. Attributes on an input `xarray.DataArray` are carried
  through in one canonical `input_attrs_json` record, preserving arbitrary keys
  without collisions or invalid NetCDF attribute names.
- `multitaper_connectivity` accepts an `xarray.DataArray`, infers common
  time/trial/signal dimension names, and transposes them into the numerical
  core's order. **This changes the DataArray contract from position-driven to
  name-driven:** domain-specific names must be assigned with `time_dim`,
  `trial_dim`, and `signal_dim`, and ambiguous dimensions raise instead of
  falling back to axis position; when a single unrecognized dimension is
  assigned to the one remaining role by elimination, a warning names the assumed
  mapping. Numeric time coordinates are treated as
  elapsed seconds, numeric sample coordinates as sample numbers, and are used
  for output window times. `sampling_frequency` is now optional for a DataArray:
  when omitted it is inferred from a numeric elapsed-seconds `time` coordinate
  with sufficient numerical precision (a `sample` index cannot supply one);
  low-precision or large-offset coordinates require an explicit rate. When given,
  the rate is validated against the index. It remains required for array input.
  Datetime, timedelta, and object-valued time coordinates are not yet supported
  and must first be converted to numeric elapsed seconds.
  Unless `signal_names` is supplied explicitly, the signal index is preserved
  as the result's `source` / `target` coordinates without coercing its type.
  Missing, duplicate, nested, and structured signal labels are rejected;
  integer labels must fit the signed 32-bit range required for portable NetCDF3
  serialization. Labels that cannot be inferred still produce a warning.
  Dask-backed DataArrays are rejected with a materialization hint.
- `DEFAULT_METHODS` is exported at package level.
- Independent analytic-oracle, failure-mode, backend-boundary, serialization,
  minimum-dependency, artifact, doctest, and notebook checks cover the corrected
  behavior.
- `observations_are_independent` on every transform and on `Connectivity`
  (forwarded by `from_transform`): `Multitaper` and `ShortTimeFourierTransform`
  report `True`, `Welch` reports `True` only up to 50% segment overlap, and
  `MorletWavelet` reports `False` whenever a smoothing neighborhood is collected
  on the observation axis. The finite-sample bias corrections (debiased
  PLI/wPLI, PPC), the zero-coherence null used by `delay`/`group_delay`, and
  `Connectivity.jackknife` consume it: the former warn once per call, and
  `jackknife` refuses leave-one-taper-out replicates on correlated observations
  and names the transforms whose observations are independent (`Welch` with
  `segment_overlap <= 0.5`, `Multitaper`, `MorletWavelet` without
  `smoothing_time` on at least 3 trials). Leave-one-trial-out
  (`expectation_type="trials"`) remains allowed but keeps the correlated axis
  unaveraged, so it measures something else. Forced through, a jackknife over
  a Morlet transform's smoothed samples gave 95% intervals that covered 15-40%
  in simulation. A companion
  `time_bins_are_independent` flag (also forwarded by `from_transform`) covers
  expectations that average over time: `Multitaper` and
  `ShortTimeFourierTransform` report `False` when windows overlap by more than
  half, and `MorletWavelet` when successive time bins are closer than four
  wavelet standard deviations (always, without decimation). Those measures then
  warn for `"time"`-averaging expectations only: pairwise phase consistency of
  independent white noise averaged over 90%-overlapping STFT or multitaper
  windows, or over unsmoothed Morlet samples, is biased by +0.002 to +0.006 in
  simulation. The warning is new; the bias itself is not removed.
- `MorletWavelet` warns when a single-trial `smoothing_time` is shorter than
  four wavelet standard deviations (`4 * n_cycles / (2 pi f)` at the widest
  wavelet), where the smoothed samples are so correlated that normalized
  measures are forced toward 1 even for independent signals; the docstring
  states the remedy. With multiple trials the expectation also averages
  independent realizations, so no warning is emitted.
- A `sampling_frequency` that is not a number (e.g. the string `"1000"` or
  `True`) raises a `TypeError` naming the argument and showing how to pass it,
  from every transform and from the `xarray.DataArray` input path.
- `multitaper_connectivity` and `fourier_connectivity` take a named
  `group_labels=...` argument and pass it only to the group measures
  (`canonical_coherence`, `canonical_coherency`,
  `maximized_imaginary_coherency`, `multivariate_interaction_measure`,
  `blockwise_spectral_granger_prediction`,
  `maximized_imaginary_coherency_components`), so one call can mix group and
  pairwise measures; `connectivity_kwargs={"group_labels": ...}` is accepted
  too. A group measure without labels, labels with no group measure, or labels
  given both ways raise an actionable `ValueError`.
- Regression coverage: an emulated CuPy-like backend that rejects host/device
  mixing, Parseval oracles for one-sided `power` and for band integration, a
  nitime oracle for adaptive multitaper power, invariance of CaCoh components to
  within-group mixing, and wrapper-level orientation checks for every directed
  measure.
- `suggest_parameters` warns when keeping the requested frequency resolution
  with at least three time windows leaves a single taper, and says how much
  longer the signal or how much coarser the resolution must be for two.
- Planted-structure and cross-package checks for measures that previously had
  only property tests: the cross-spectral density recovers a planted amplitude
  and phase, MIC/MIM find a lagged between-group source and reject a zero-lag
  one, blockwise Granger with two-channel blocks matches Geweke's closed form,
  and partial coherence matches the inverse-spectrum oracle and removes a
  mediated link. Coherence and CSD agree with nitime's non-adaptive multitaper
  estimate (`taper_weighting="eigen"`), and eleven measures agree with a
  recorded mne-connectivity 0.9 reference. The seven phase measures (PLV,
  ciPLV, PPC, PLI, dPLI, wPLI, debiased wPLI²) agree in the single-taper
  setting: mne-connectivity combines each epoch's tapers before the phase step
  while this package treats every trial x taper as an observation, so with
  several tapers the two estimators differ by design.

### Changed

- matplotlib is no longer a required dependency; the package never imported
  it. Install it to run the README example and the tutorials, for example with
  the new `plot` extra: `pip install "spectral-connectivity[plot]"`.
- `Connectivity.global_coherence` returns only the non-negative frequencies,
  labelled by `Connectivity.frequencies`, like every other measure; 2.x
  returned every bin of the two-sided transform, labelled by
  `all_frequencies`. For a real signal the negative bins repeat the
  non-negative ones (the same fractions, conjugated vectors), and the
  non-negative bins are unchanged, so code that sliced
  `[..., :len(conn.frequencies), :]` gets the same values. Until 3.2 a call on
  two-sided coefficients emits a `SidednessWarning` once per session.
  The results of `multitaper_connectivity` and `connectivity_to_xarray` were
  already trimmed so; they are unchanged and do not warn. Since the negative
  bins are no longer decomposed, NaN or Inf coefficients confined to them no
  longer raise the `ValueError` described under Fixed.
- `Connectivity.global_coherence` follows `expectation_type` for the settings
  that average the tapers: it decomposes the cross-spectral matrix averaged
  over the same axes as the other measures, with the kept axes leading the
  result in the same order. The default, `"trials_tapers"`, gives the same
  values as before (each window's trials and tapers); `"time_trials_tapers"`
  pools every window, trial, and taper into one value per frequency, the
  estimand of Chronux `CrossSpecMatc`'s `Ctot` on non-overlapping windows;
  `"time_tapers"` gives one value per trial; and `"tapers"` one per window and
  trial. Observation weights weight the pooled columns as before. `"time"`,
  `"trials"`, and `"time_trials"`, which keep the tapers as an output axis,
  raise `ValueError` naming the supported settings. 2.x ignored
  `expectation_type` and pooled each window's trials and tapers for every
  setting. The wrappers keep `"trials_tapers"`; the `global_coherence`
  docstring and the cookbook show how to pool time windows with
  `Connectivity.from_transform(transform,
  expectation_type="time_trials_tapers")`, and how that differs from the
  median-over-windows estimator of Purdon et al. (2013).
- `Connectivity.global_coherence` warns when it pools fewer observations than
  signals (`n_observations < n_signals`), not only for a single observation:
  the pooled matrix then has rank at most `n_observations`, so the value is at
  least `1 / n_observations` whatever the data and cannot reach the
  `1 / n_signals` of independent signals. The message keeps the
  single-observation case (forced to 1), says to compare values only at
  matched numbers of observations and signals, and offers only what raises
  the count. When it pools time windows the transform reports as correlated
  (`time_bins_are_independent=False`, such as `Multitaper` windows
  overlapping by more than half), it warns, as the other measures that count
  observations do, that they add fewer effective observations than their
  count.
- `minimum_phase_decomposition`'s `max_iterations` defaults to 500; 2.0.1
  defaulted to 60. Near-singular cross-spectra (highly correlated channels)
  can need several hundred Wilson iterations, and the loop still stops as soon
  as every sub-spectrum converges. Pass `max_iterations=60` for the 2.0.1 cap.
  `tolerance` and `max_iterations` keep their positions and are accepted by
  position or keyword.
- On the GPU, the default FFT length (and so the default frequency grid) of
  `Multitaper`, `ShortTimeFourierTransform`, and `Welch` now follows SciPy's
  `next_fast_len`, as on the CPU, so the same data gives the same frequencies
  on both backends. CuPy's rule differed for 8258 of the lengths from 64 to
  19999: SciPy treats 11 as a fast factor, so 3601 samples padded to 3630 on
  the CPU but 3645 on the GPU. CPU results are unchanged. cuFFT is no slower at
  SciPy's lengths (median raw-FFT time ratio 1.00 over 25 sampled lengths).
  Pass `n_fft_samples` to choose a length explicitly.
- Every directed `Connectivity` array is indexed `[..., source, target]`:
  `result[..., i, j]` is the influence `i -> j`, the same order as the
  wrapper's `sel(source=i, target=j)` and the lead/lag measures. In 2.x
  pairwise spectral Granger and its subset form, the directed transfer
  function, directed coherence, (generalized) partial directed coherence, and
  the direct directed transfer function returned `[..., target, source]`.
  `np.swapaxes(old, -1, -2)` gives stored 2.x arrays the new orientation. It
  reproduces the 3.0 values of pairwise Granger, DTF, PDC, and generalized PDC
  only where the FFT length was odd and there were at least as many
  observations as signals: on an even FFT length every directed value changed
  (see Fixed), directed coherence and the direct DTF changed in value
  regardless, and with fewer observations than signals the transfer-function
  measures are now `NaN` (see the migration guide).
- The array backend is selected once, in `spectral_connectivity._backend`,
  instead of in each module: an unrecognized `SPECTRAL_CONNECTIVITY_ENABLE_GPU`
  value now warns (2.x silently used the CPU), and the "Using CPU/GPU" message
  is logged once at import rather than once per module. `Multitaper`'s
  `fft_workers` and the xarray `backend` attribute follow the imported backend
  instead of re-reading the variable after import.
- xarray measure results carry `long_name` and known `units` (spectral
  densities in `(<input units>)^2/Hz` when the input states its units; omitted
  when unknown),
  `band_lower`/`band_upper` coordinates after band reduction, and an input
  DataArray's per-signal coordinates, with their attrs, as
  `source_<name>`/`target_<name>`. The package's coordinates (including band
  names and edges and component connection labels) carry a `long_name`.
  `frequency_band_reduce` takes `circular=` and infers circular averaging
  from `units="rad"`. Large array input attributes are summarized by shape.

- `canonical_coherency` fixes the sign of each spatial filter by the dominant
  coefficient of its pattern, so the canonical phase is not ambiguous by
  pi; single-channel groups reproduce the conjugate pairwise coherency.
- `phase_slope_index`, `delay`, and `group_delay` require a uniformly spaced
  frequency grid (every step within one part in a million of the mean step,
  allowing for the storage rounding of a float32 coordinate; the same test
  decides whether a result records `frequency_bin_spacing` and whether
  two-sided frequencies are in FFT order), and
  `delay`/`group_delay` reject non-uniform observation
  weights, because their adjacent-bin combinations and coherence null assume
  equal spacing and equally weighted observations (wavelet grids and Hann
  smoothing violate these silently).
- `Connectivity.jackknife` requires at least three observations, rejects
  diagnostics, alternate constructors, and structured (component) results with
  a clear error, and documents the supported explicit `fisher_squared`
  transformation.
- `frequency_band_reduce` treats a band containing any NaN bin as undefined for
  both reductions and records per-band
  validity in a `valid_time_band` coordinate when the input carries
  `valid_time_frequency`.
- `fourier_connectivity` warns when neither a frequency coordinate nor
  `is_one_sided` is given and a two-sided spectrum is assumed, and rejects a
  lone negative-frequency bin because it cannot be a complete FFT spectrum.
- Group measures require at least two groups and a non-missing label per signal
  (in 2.x `canonical_coherence` with one group raised an unrelated
  `ValueError: need at least one array to stack`), and
  `Multitaper` reports non-positive or non-finite sampling rates as
  "sampling_frequency must be finite and positive".
- `global_coherence`, `power`, `phase_slope_index`, `delay`, and
  `Multitaper.time` now follow the definitions in the migration table.
- Public connectivity methods reject single-signal inputs with an actionable
  error; `power` and `cross_spectral_density` remain available for one signal.
- Bias-corrected phase measures require at least two observations.
- `Multitaper` parameters and owned input snapshots are immutable after
  construction. Array accessors return detached copies so cached calculations
  cannot become stale through external mutation.
- `multitaper_connectivity(method=None)` uses the stable, exported
  `DEFAULT_METHODS` allowlist. Every measure in `list_measures()` is available
  by name. Every directed measure is opt-in by name,
  pairwise spectral Granger included: it factorizes the spectrum and costs
  more than the rest of the default set together.
- Multi-component `canonical_coherency` deflates in whitened space, so
  components beyond the first are uncorrelated within each group and invariant
  to invertible within-group mixing. The phase search refines every local
  maximum of a 74-point coarse phase grid, so near-equal lobes of the objective
  yield a wrong canonical phase only when two lie within about `2 * pi / 74` of
  each other.
- `Connectivity.jackknife` and `jackknife_confidence_interval` use a Student t
  critical value with `n_observations - 1` degrees of freedom (Thomson & Chave
  1991), not the normal quantile, which under-covers at small `n` (Monte Carlo
  coverage of a Gaussian mean at five observations: 0.95, against 0.88).
  `"auto"` selects `fisher_squared` for `partial_coherence` and `fisher` for
  `phase_locking_value` and `imaginary_coherence`, whose lower bounds and
  bias-corrected estimates are clamped at 0. Circular bounds are wrapped to
  `(-pi, pi]` (lower above upper means the interval crosses `+/-pi`); an
  interval whose half-width reaches `pi` is reported as `(-pi, pi)` with a
  warning. The docstrings state the intervals' measured limits: they are not
  tests of nonzero coherence (at zero true coherence the `fisher_squared`
  interval excludes 0 in 11-15% of datasets with five or more observations, 6%
  with three and 9% with four; use `statistics.coherence_significance_pvalue`,
  the exact zero-coherence null), PLV intervals under-cover at high PLV, and
  circular intervals under-cover at weak coherence. The Fisher saturation
  warning counts every estimate the transform pins at 1 (to machine precision,
  not only exact 1s) and ignores `phase_locking_value`'s diagonal, which is 1
  by definition, so it does not fire on every PLV jackknife.
- `partial_coherence` warns when there are fewer observations than signals
  (the rank-deficient estimate is forced to 1 or dominated by the null space).
- `multitaper_connectivity` rejects a DataArray with a `frequency`/`freq`/`band`
  dimension, since its input is time-domain signals (2.x read a DataArray by
  axis position, whatever its dimension names).
- `start_time` must be a single finite value for `Multitaper`,
  `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet`. 2.x `Multitaper`
  accepted `NaN` and, for a single window, a per-trial array, which became the
  `time` axis. A single-element array such as `time[0]` of a column
  time axis is accepted as one start time.
- `phase_slope_index`, `delay`, and `group_delay` take
  `frequency_range=(low, high)`, which includes both endpoints, to choose the
  bins they consume. It is validated: two finite values, lower <= upper, and
  at least one frequency bin selected. Their xarray results record the
  requested bounds (`requested_frequency_lower`/`_upper`, omitted when every
  bin is used), the lowest and highest bins in the interval
  (`selected_frequency_lower`/`_upper`, before any `frequency_resolution`
  subsampling, which can make `phase_slope_index` sum only some of them), and
  `frequency_interval_closed` (`"both"`, or `"neither"` for the deprecated
  keyword) on each of their variables. A Morlet result's `valid_time` is `True`
  only where every bin of the interval has full support, so with
  `frequency_resolution` it can be stricter than the bins
  `phase_slope_index` actually sums.
- `multitaper_connectivity` and `fourier_connectivity` pass their
  `frequency_range` to `delay`, `group_delay`, and `phase_slope_index`, so the
  wrapper and a direct `Connectivity` call consume the same bins. Every other
  measure is computed from the full spectrum (the Wilson factorization sees the
  whole two-sided spectrum) and cropped to the same bins, so a Dataset mixing
  them shares one frequency axis. `frequency_bands`
  and `frequency_decimation` reduce the frequency-resolved variables, `delay`
  included, and leave `phase_slope_index` and `group_delay` as computed over
  `frequency_range`; requesting either with only those two measures raises.
  `frequency_range` inside `connectivity_kwargs` is rejected in favor of the
  wrapper argument. The range is selected once per call, on the full grid, and
  every variable it applied to, cropped or consuming, records the same
  `requested_frequency_*`, `selected_frequency_*`, and
  `frequency_interval_closed` attributes; decimation and band reduction keep
  them and record themselves.
- The `taper_weighting="adaptive"` documentation states that the Thomson
  weights are applied per signal, so coherence between signals with different
  spectra is shrunk by the cosine similarity of their weight vectors and cannot
  reach 1; a joint pairwise normalization is not implemented.
- Tooling: the CI quality job installs from `uv.lock` and pre-commit pins the
  same ruff version; `examples/` is excluded from ruff; the sdist ships the docs
  files the test suite reads; `CLAUDE.md` and `CONTRIBUTING.md` state Python
  3.10-3.14 and the `uv` commands CI actually runs. The stale
  `docs/NOTEBOOK_SNAPSHOT_TESTS.md` plan is removed.
- README and tutorial lead with `multitaper_connectivity`; the DataArray
  contract moved to a cookbook recipe.
- `Connectivity.group_delay` and `Connectivity.delay` document that a pair
  without significant coherence is NaN (for independent signals,
  `group_delay()` returns NaN delay, slope, and r), which means no delay was
  estimated rather than a delay of zero, and how `significance_threshold`
  selects the frequencies: an exact zero-coherence test at each bin,
  Benjamini-Hochberg corrected at that false discovery rate over every bin,
  pair, and window of the call, keeping each pair's largest contiguous run of
  at least 3 independent significant bins. The values are unchanged.
- `coherence_significance_pvalue` and `coherence_fisher_z_transform` say that
  they take coherency (`Connectivity.coherency()`) or its magnitude, not
  `magnitude_squared_coherence()`. Both square or read the absolute value of
  their input, so magnitude-squared coherence gives wrong p-values or z-scores
  without an error: for independent noise no interior frequency had
  `p < 0.05`, where `coherency()` gives about the nominal 5%. A real-valued input
  is still accepted, since coherency magnitudes and coherency from real
  coefficients are real too. The values are unchanged.
- The `multitaper_connectivity` and `Multitaper` docstrings state that trials
  are the middle axis of a 3-D time series, `(n_time_samples, n_trials,
  n_signals)`, with an example: a `(500, 3, 40)` array is read as 3 trials of
  40 signals, and `np.swapaxes(x, 1, 2)` reads it as 40 trials of 3. The
  wrapper's docstring named the axes `(n_times, n_trials, n_channels)`. An
  array in the other order cannot be detected, so it is analyzed with trials
  and signals exchanged.
- The examples of `coherence_significance_pvalue` and
  `power_confidence_intervals` start from a time-resolved `Connectivity`:
  they pass its `n_observations`, reshape `real_valued_frequencies` to
  broadcast against the coherency or power (so 0 Hz and Nyquist get their own
  null), and, for the p-values, select one pair before
  `adjust_for_multiple_comparisons`.
- The `Connectivity` docstring has a table of the leading axes each
  `expectation_type` keeps and the shape it gives, and its descriptions of
  `"trials"` and `"tapers"` say that they keep the time-window axis too.
- The `Connectivity` docstring says what each group measure returns and how
  to unpack it: `(values, labels)` from `canonical_coherence` and the other
  scalar group measures, a `MultivariateConnectivityResult` from
  `canonical_coherency` and `maximized_imaginary_coherency_components`, and
  two arrays from `global_coherence`. `global_coherence` names its second
  return value `vectors`, and its Notes say how to read the value: the floor
  and bias set by the numbers of observations and signals, effective
  observations, pooling time, channel scale and reference, the mean rather
  than Purdon et al.'s median over windows, 0 Hz and Nyquist, and the absence
  of a significance test.
- `Multitaper.frequencies` (and `ShortTimeFourierTransform`'s and `Welch`'s)
  says that it labels the two-sided FFT bins of `fft()` and that
  `Connectivity.frequencies`, the non-negative bins, labels results; the
  `Multitaper` attributes give `fft` and `frequencies` their `n_fft_samples`
  length.
- The Returns section of `multitaper_connectivity` says that one `method`
  string gives a DataArray and the default a Dataset of `DEFAULT_METHODS`,
  and that without `time_window_duration` `time` has length 1 (drop it with
  `isel(time=0)` or `squeeze("time")` to plot).
- `spectral_connectivity.statistics` defines `__all__`: its functions,
  `JackknifeResult`, and the `MULTIPLE_COMPARISONS` mapping (now documented in
  the API reference). `from spectral_connectivity.statistics import *` no
  longer also imports the module's own imports (`np`, `scipy`, `warnings`,
  `dataclass`, `cast`, ...); import those yourself.

### Removed

- `blocks` of `Connectivity` and `Connectivity.from_multitaper` (released in
  2.0.1). It was meant only to chunk the cross-spectral computation, and memory
  is now bounded automatically. In 2.x it also changed the four phase-lag-index
  measures (see the migration guide): `blocks=1` returned `phase_lag_index` and
  `weighted_phase_lag_index` with the opposite sign, and larger values raised
  `IndexError` or returned constants. Every other measure agreed with and
  without it to rounding. A call that still passes it, by keyword or in its
  released position, raises a `TypeError` that names `blocks`, says to drop it,
  and says to recompute phase-lag-index results obtained with it, before any
  computation. It is not kept as a deprecated alias.
- `dpss_windows(interp_from=..., interp_kind=...)`: the tapers come from
  SciPy's exact solver, so there is nothing to interpolate.
- `tridisolve` and `tridi_inverse_iteration` from
  `spectral_connectivity.transforms`, public helpers of the 2.x DPSS solver;
  use `scipy.linalg.solve_banded` or `scipy.linalg.eigh_tridiagonal`.
- `partial_directed_coherence(keep_cupy=...)`: public measures return NumPy
  arrays.
- `Multitaper`'s default `sampling_frequency=1000`: the rate is required.
- `spectral_connectivity.connectivity.EXPECTATION` (a dict of averaging
  functions), replaced by `EXPECTATION_AXES` (the axes each
  `expectation_type` averages), and the module's incidental re-imports of
  `minimum_phase_decomposition`, `coherence_fisher_z_transform`, and
  `get_normal_distribution_p_values`; import them from
  `spectral_connectivity.minimum_phase_decomposition` and
  `spectral_connectivity.statistics`.
- The `mt_summarize_parameters` result attribute, which held a bound method
  rather than a value.

### Deprecated

Each deprecated spelling keeps its 2.x meaning through 3.x and gives the same
values as its replacement (where the replacement's values differ from 2.x, the
entry says so), warns with a `DeprecationWarning` that names its replacement, and
is removed in 4.0.

- `coherence_magnitude`, as the `Connectivity` method and as a `method` string
  of `multitaper_connectivity`, `fourier_connectivity`, `connectivity_to_xarray`
  and `Connectivity.jackknife`: use `magnitude_squared_coherence`, the same
  calculation under a name that says it is squared (as SciPy's
  `scipy.signal.coherence` documents). It is the default result variable, and
  the name (not the old one) that appears in `DEFAULT_METHODS`,
  `list_measures()`, and the measure table; the old name is not listed or
  computed by default. A result requested
  by the old name keeps the variable name `coherence_magnitude` and records
  `measure="coherence_magnitude"` (the name asked for), so 2.x selectors keep
  working and a saved file says which name produced it; its values and other
  attributes are those of `magnitude_squared_coherence`. Requesting both names
  in one call raises `ValueError`. The wrappers warn only for a call that
  succeeds, so a call rejected for another reason raises only its error.
- `Connectivity.from_multitaper`: use `Connectivity.from_transform`, the
  canonical constructor for any spectral transform. The old name takes the
  same arguments (including its `multitaper_instance` keyword) and builds the
  same instance; it warns once, at the line that called it. Neither name calls
  the other: a subclass overriding `from_multitaper` keeps working when called
  by that name and warns at its own call to `super().from_multitaper`, but
  `from_transform` does not go through such an override, and
  `Sub.from_multitaper` does not go through an override of `from_transform`.
  The package itself uses only `from_transform`.
- Advanced settings passed to `multitaper_connectivity` or
  `connectivity_to_xarray` as plain keyword arguments (the 2.x `**kwargs`
  forwarding): use `transform_kwargs` for `Multitaper` settings and
  `connectivity_kwargs` for measure settings. The plain keywords still reach
  the same place, with the same result and provenance, and warn once per
  call, at the caller, naming the mapping, only when the call succeeds (a
  rejected call raises its error without the warning). The common settings
  are named arguments and never warn. A setting given both in the mapping and
  as a plain keyword raises `ValueError`, and a name the transform or measure
  does not take, in either mapping or as a plain keyword, raises `TypeError`
  before any computation. A plain `frequencies_of_interest=...` to
  `connectivity_to_xarray` gets only its own deprecation, which names the
  `frequency_range` that replaces it.
- `power_confidence_intervals(n_tapers=...)`: use `n_observations=...`. The
  count is every observation averaged into the estimate (tapers x trials), not
  the taper count, so the old name invited intervals far too wide for
  multi-trial averages. The argument keeps its first position, so positional
  calls such as `power_confidence_intervals(25, power, 0.95)` keep working
  and do not warn; `n_tapers` is accepted by keyword only and gives the same
  interval as `n_observations`. `power` keeps its name and position, and the
  confidence level its position (next entry). Giving both names raises
  `ValueError`. Neither form reproduces 2.x's numbers: both return the
  corrected interval (see Fixed), and the interval 2.0.1 returned for `ci=c`
  is 3.0's `confidence_level=2c-1` (for `c >= 0.75`; 3.0 accepts levels in
  `[0.5, 1)`).
- `power_confidence_intervals(ci=...)`: use `confidence_level=...`, the name
  `Connectivity.jackknife` and `jackknife_confidence_interval` use. The level
  keeps its third position, so positional calls such as
  `power_confidence_intervals(25, power, 0.95)` keep working and do not warn;
  `ci` is accepted by keyword only and gives the same interval as
  `confidence_level`, which, like it, must lie in `[0.5, 1)` (the error
  names the argument given). Giving both names raises `ValueError`, even
  with `confidence_level=0.95`, and a call rejected for another reason
  raises without the warning. The interval is the corrected one (previous
  entry).
- `suggest_parameters(desired_freq_resolution=...)`: use
  `frequency_resolution=...`, the name `Multitaper` and
  `estimate_frequency_resolution` use (and the key of the returned
  suggestion). The argument keeps its third position, so positional calls
  keep working and do not warn; `desired_freq_resolution` is accepted by
  keyword only and gives the same suggestion. Giving both names raises
  `ValueError`, and a call rejected for another reason raises without the
  warning. `signal_duration` and `desired_n_tapers` keep their names.
- `frequencies_of_interest` of `delay`, `group_delay`, and
  `phase_slope_index`, by keyword or as the first positional argument: use the
  keyword-only `frequency_range`. The deprecated form excludes bins on both
  edges and, for every band of two values with a bin inside it, selects the
  bins 2.x selected, including on float32 grids and for open-ended bands such
  as `(10, np.inf)`: it evaluates 2.x's `band[0] < frequencies < band[1]` on
  the band as given. Its results record `frequency_interval_closed="neither"`,
  and an infinite bound records no `requested_frequency_*` for that side. Its
  warning names the finite `frequency_range` that reproduces the same bins,
  e.g. on a 2 Hz grid `frequencies_of_interest=(10, 50)` is
  `frequency_range=(12.0, 48.0)`. Unlike 2.x it is validated: a band that is
  not exactly two values (2.x silently used the first two of a longer one), a
  NaN bound, lower >= upper, or a band with no bin strictly inside (one beyond
  Nyquist, between two adjacent bins, or `(5, 5)`) raises `ValueError`. For a
  band with no bin inside, 2.x returned an empty `delay` frequency axis, while
  `phase_slope_index` failed with an unrelated `IndexError` and `group_delay`
  with an unrelated `ValueError` (a zero-size reduction). Giving both
  spellings raises `ValueError`.

### Fixed

Entries marked *Unreleased* fix defects that existed only in development
builds after 2.0.1, or in features 2.0.1 did not have: nothing computed with
2.x is affected by them.

- *Unreleased:* `power_bias` and `power_variance` accept an array of counts
  with a scalar `real_valued` (the default), returning one value per count as
  2.0.1 did; once `real_valued` was added they raised converting the array to
  a float.
- Every observation count of the statistics helpers is checked the same way,
  so the same bad count gets the same message, naming the argument, the
  counts it takes, and `Connectivity.n_observations`: `coherence_bias`,
  `power_bias`, `power_variance`, `coherence_significance_pvalue`,
  `power_confidence_intervals`, and both counts of
  `coherence_fisher_z_transform` and `power_fisher_z_transform`. A count the
  formula cannot use raises a `ValueError`, as `coherence_significance_pvalue`
  already did: `coherence_bias` needs at least 2 (it returned -0.5 for 0 and
  divided by zero for 1), and the power helpers at least 1 (0 hit the
  digamma and trigamma poles). A non-finite or fractional count raises too,
  naming the offending values of an array of counts. A boolean, a string, or
  None raises a `TypeError`: `power_bias(True)` and
  `power_confidence_intervals(True)` read `True` as 1 observation,
  `coherence_bias(True)` divided by zero, and
  `coherence_significance_pvalue(c, True)` raised a `ValueError` about
  needing 2. `n_obs2` may be 0 (a one-sample test) as before, and the
  messages say so ("n_obs2 must be a finite integer >= 2, or 0 for a
  one-sample test, got 1."). An array of counts, one per estimate (as an
  array or a list), is still accepted by `coherence_bias`, `power_bias`, and
  `power_variance`, and now by both counts of the Fisher z-transforms and the
  count of `power_confidence_intervals`, where 0 in `n_obs2` means no second
  sample for that estimate; all but `coherence_fisher_z_transform`'s `n_obs1`
  raised NumPy's "truth value of an array is ambiguous" for one.
  `coherence_significance_pvalue` takes a single count, and says so for an
  array, because its real-valued bins use one count for the call.
- *Unreleased:* `simulate_shared_oscillation` raises a `ValueError` naming
  the frequency and the Nyquist frequency when `frequency` is negative or not
  below Nyquist, and validates `sampling_frequency` as the transforms do.
  `simulate_shared_oscillation(300, 500, 1000, 5, amplitudes=[1, 1])` returned
  the 200 Hz sinusoid that 300 Hz aliases to at 500 Hz without a sign.
- `simulate_MVAR` warns when its coefficients are unstable (the spectral
  radius of their companion matrix is at least 1), giving the radius (naming
  a radius of 1 a unit root) and a scaling that brings it to 0.9 for any
  number of lags (`A_k` times `c**k` scales the radius by `c`):
  `simulate_MVAR(np.array([[[1.5, 0], [0, 1.5]]]), n_time_samples=200)`
  returned values near 1e52 without a sign. Malformed input raises a
  `ValueError` naming the expected shape and the one received, where NumPy
  failed with an unrelated message: coefficients that are not
  `(n_lags, n_signals, n_signals)` (2-D, or non-square), a `noise_covariance`
  of another size than the signals, a count that is not a positive integer
  (`n_trials=2.5`), or a negative `n_burnin_samples`. Complex coefficients,
  which it cannot simulate (NumPy failed with a casting error), raise a
  `TypeError`.
- *Unreleased:* `multitaper_connectivity` warns when it infers the sampling
  rate of a DataArray from a time coordinate that may be in milliseconds: when
  the inferred rate is at most 1 Hz (unless the coordinate's `units`
  attribute says seconds), or when that attribute says milliseconds. The
  coordinate is read as elapsed seconds, so 500 Hz data on a millisecond axis
  gave 0.5 Hz and a frequency axis ending at 0.25 Hz without a sign, and 1 kHz
  data gave 1 Hz. The warning gives the inferred rate and the conversion,
  `da = da.assign_coords({'time': (da['time'] / 1000).assign_attrs(units='s')})`;
  for slow recordings such as fMRI, passing `sampling_frequency` silences it.
  A rate passed explicitly never warns, and a mismatch between it and a
  coordinate 1000 times too coarse asks whether the coordinate is in
  milliseconds. A millisecond axis of data sampled faster than 1 kHz infers a
  rate above 1 Hz and is not detected unless the coordinate's `units`
  attribute says milliseconds: set `units`, or pass `sampling_frequency`.
- *Unreleased:* `multitaper_connectivity` accepts a nested list as 2-D
  `(time, signal)` input, and `fourier_connectivity` one as 3-D or 4-D
  coefficients, as they do the same array. Both read the layout before
  converting, so a list raised a `ValueError` asking for 3-D input (or for 3
  to 5 dimensions) although the array it describes has a supported shape. A
  ragged nested list, or one of strings, given to `multitaper_connectivity`
  raises an error naming the expected numeric array instead of NumPy's
  inhomogeneous-shape or ufunc message.
- `Multitaper` and `multitaper_connectivity` warn when low-bias filtering
  (`is_low_bias=True`, the default) uses fewer DPSS tapers than requested, and
  every taper count they report is the number used. Only about `2 * NW - 1`
  tapers have a spectral concentration above 0.9, so `n_tapers=50` at the
  default NW=3 on a 1000-sample window used 5, giving the same power as
  `n_tapers=5`, while `Multitaper.n_tapers`, its `repr`,
  `summarize_parameters()`, and the results' `mt_n_tapers` and
  `transform_parameters_json` said 50 (only `n_observations` counted 5). The
  warning names both counts and the remedies (a larger
  `time_halfbandwidth_product`, or `is_low_bias=False`); the request is kept as
  `requested_n_tapers` in `transform_parameters_json`. Saved results differ
  accordingly: older files hold the request in `mt_n_tapers` and
  `estimator.n_tapers` and have no `requested_n_tapers`; new files hold the
  count used there and the `n_tapers` argument in `requested_n_tapers` (`null`
  when it was not given). To compare the two, match an old `n_tapers` with a
  new `requested_n_tapers` and check `n_observations`. The DPSS tapers are now
  generated when the `Multitaper` is constructed, so a window too short for
  them, too long for the recording, or of zero samples raises there rather
  than at the first `fft()` (see the migration guide).
- *Unreleased:* `coherence_rate_adjustment` given `Connectivity.power()`
  returned a factor too close to 1: 0.81 where the coherence of a thinned
  spike train fell to 0.71 of the original. `power()` became one-sided in 3.0
  while the function read its spectrum as the two-sided density of the
  reference. A new keyword, `is_one_sided`, says which density is given.
  Without it the spectrum is taken as one-sided and a warning says so, since
  the two cannot be told apart from their values.
- `coherence_rate_adjustment` with `homogeneous_poisson_noise` non-zero takes
  the thinning fraction from the second condition's rate less the added
  spikes, `(firing_rate_condition2 - homogeneous_poisson_noise) /
  firing_rate_condition1`. In the model of Aoi et al. (2015) the second
  condition's intensity is `alpha * lambda(t) + beta`, with mean rate
  `alpha * mu + beta`. The function used `firing_rate_condition2 /
  firing_rate_condition1`, which is `alpha` only when no spikes are added, so
  for the documented argument the factor was too large: 0.90, 0.81 and 0.74
  where the thinned process gives 0.69, 0.54 and 0.46 in the case run. A
  noise rate that is negative, not finite, or not below the second rate raises
  a `ValueError`. A 2.x call that passed the kept-spike rate still runs, so a
  call with a non-zero noise rate warns that the argument's meaning changed,
  until 4.0; `warnings.filterwarnings("ignore", message="coherence_rate_adjustment,
  changed in")` silences it.
- *Unreleased:* `Connectivity`, `Connectivity.from_transform`, and
  `from_multitaper` reject a `dtype` that is not complex with a `TypeError`
  before any FFT. Once `blocks` was removed, `dtype=None` (for instance 2.x's
  default `blocks` passed in its old position, where `dtype` now is) was read
  as float64 and silently discarded the imaginary part of every coefficient,
  as did a real or integer dtype, and an integer count failed only when a
  measure first used it (`power` did not). 2.x read `dtype=None` as its
  complex128 default.
- Every directed measure (the spectral Granger family,
  `directed_transfer_function`, `directed_coherence`,
  `partial_directed_coherence`, `generalized_partial_directed_coherence`,
  `direct_directed_transfer_function`) keeps the imaginary part of the
  minimum-phase factor's lag-0 coefficient, and so of the innovation
  covariance, which is Hermitian. 2.x and earlier 3.0 builds discarded it,
  which is harmless for a real process (it is rounding-level there; results
  for real signals change by at most 1e-15) but not for a complex one, whose
  spectrum is not conjugate-symmetric: complex white noise with innovations
  correlated by `0.6j`, which has no lagged dependence, gave pairwise Granger
  0.45 and DTF 0.26 in one direction. Pairwise, conditional and blockwise
  Granger of complex processes now match an independent state-space
  calculation.
- The paper-examples tutorial displays Ding et al.'s (2006) models as
  published: Example 2a showed the lag-two `x2 -> x1` term that only Example
  2b has, and Example 3 showed the `x4 -> x5` coefficient as `+0.25 sqrt(2)`
  rather than `-0.25 sqrt(2)`. The simulated models were already the published
  ones.
- `direct_directed_transfer_function` and its `list_measures` entry no
  longer describe it as direct influence only. It is zero for an influence
  relayed through observed signals, but its partial-coherence factor is also
  nonzero between two signals that drive a common observed signal: in
  `0 -> 1 -> 2` with `0 -> 3 <- 2`, which has no `0 -> 2` coefficient, dDTF
  `0 -> 2` is positive at every frequency (summing to 1/18 for unit couplings)
  while PDC and conditional Granger are zero. The values are unchanged.
- Inference at real-valued Fourier coefficients, the 0 Hz bin and (for an
  even FFT length) the Nyquist bin of a real signal, where each observation
  carries one degree of freedom rather than two. `power_confidence_intervals`,
  `power_bias`, `power_variance`, and `power_fisher_z_transform` take a
  keyword `real_valued`, a boolean mask broadcast against the estimates, and
  use `n` rather than `2 n` chi-squared degrees of freedom where it is set:
  with five observations the 95% power interval had covered 83%, and a
  one-sample log-power z had variance 2.2 instead of 1. The new
  `Connectivity.real_valued_frequencies` gives the mask for
  `Connectivity.frequencies` (prefer it to comparing frequencies with
  `sampling_frequency / 2`, which rounding can defeat at Nyquist), and a mask
  that is not boolean or does not broadcast raises. `group_delay` and `delay`
  apply it themselves, testing those bins against the real zero-coherence
  null, so their results at 0 Hz and Nyquist can change.
- *Unreleased:* `coherence_significance_pvalue` takes the same `real_valued`
  mask and uses the real null `Beta(1/2, (n - 1) / 2)` for squared coherence
  there: with five observations the complex null had rejected 10% of
  independent pairs at a nominal 5%. `coherence_fisher_z_transform` and
  `coherence_bias` keep their complex-coefficient terms and document that
  they are not calibrated at those bins (9-18% rejection of equal coherences
  at a nominal 5% in simulation), pointing to the exact zero-coherence test.
- *Unreleased:* `Multitaper.observations_are_independent` is `False` for
  custom tapers that are not mutually orthogonal, whose eigencoefficients are
  correlated even for white noise, so the measures that count observations
  warn and the jackknife refuses to leave out tapers. It was always `True`:
  four copies of one taper made `n_observations` 4 and pairwise phase
  consistency of independent channels 1, with no warning. DPSS and orthogonal
  custom tapers are unaffected.
- *Unreleased:* An observation with weight 0 contributes nothing to any
  weighted expectation even when its coefficients are not finite: power, the
  cross-spectrum and the measures built on them, phase locking, the phase-lag
  family, canonical coherence and global coherence equal what omitting the
  observation gives. `0 * NaN` made them NaN, even with every other
  observation finite and positively weighted.
- *Unreleased:* `corrected_imaginary_phase_locking_value` forms its PLV in at
  least complex128, so a small consistent lag of complex64 coefficients gives
  1, as the definition does; a lag of 1e-4 rad rounded `Re(PLV)` to 1 and gave
  0. The result keeps the precision of `dtype`.
- `debiased_squared_phase_lag_index` is the mean of `s_k s_l` over pairs of
  distinct observations, for `s` the sign of the imaginary cross-spectrum. 2.x
  and earlier 3.0 builds used Vinck's `(n * PLI**2 - 1) / (n - 1)`, which
  assumes every sign is +/-1: an observation whose imaginary cross-spectrum is
  exactly 0 (an in-phase or all-zero trial or taper among lagged ones) has sign
  0, and the shortcut then subtracted too much (signs `[0, +1, +1]` gave 1/6
  instead of 1/3). Results without such observations are unchanged.
- The Wilson factorization behind every directed measure (the spectral Granger
  family, `directed_transfer_function`, `directed_coherence`,
  `partial_directed_coherence`, `generalized_partial_directed_coherence`,
  `direct_directed_transfer_function`) reproduces the spectrum on an
  even-length FFT grid. There the lag `n_fft_samples / 2` is its own negative,
  so half of it is causal, as with lag 0; the causal projection dropped it, and
  the iteration converged to a factor `G` with `G⁻¹ S G⁻ᴴ = I + (-1)ᵏ C` at bin
  `k` instead of `I`. For spectra estimated from simulated data the factor
  missed the spectrum by 0.2-10% of its largest entry (most for short windows,
  few observations, and highly correlated channels) and now does so by about
  1e-9. Odd-length grids were not affected, and 2.x has the same defect. A
  simulation study compared the old and new factor in 42 settings of two to
  five signals, 40 datasets each; its error is the RMS difference, over
  frequencies and signal pairs, between a measure and the same measure from a
  factorization of the same spectral estimate on a four times finer grid. The
  new factor was better in 37 settings and worse in 5. In the 37, the error of
  spectral Granger, DTF, and PDC fell in at least 33 of the 40 datasets, by
  40-50% for the median setting, and the error of DTF and PDC against the
  generating model's true values fell by about 10%. The 5 are 50-sample windows
  of a sharply resonant signal (pole radius 0.99), where every estimate is
  poor: there the error of spectral Granger (four settings) or of DTF and PDC
  (one) rose by 0.01-0.14, in the measures' own units. The old factor also
  returned `NaN` for up to 1% of spectral Granger values, and the new one for
  none. Directed measures from a window shorter than the slowest correlation in
  the data remain unreliable, and their values depend slightly on the FFT
  length, which zero-padding removes without making them more accurate for
  well-resolved data; the Notes of
  `spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`
  give the sizes.
- `Multitaper`, `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet`
  refuse an `xarray.DataArray` whose dimension names put time, trials, or
  signals on another axis than their positional `(time, trial, signal)`. They
  read a labeled input by position, so 2.x `Multitaper` transformed, say, a
  `(trial, time, signal)` array along its trial axis and returned a plausible
  result of the wrong quantity (or, with too few trials to taper, raised an
  opaque `select_range` error). The `ValueError` names each misplaced
  dimension and, when the names determine it, the transpose to apply;
  names that say nothing of a role
  (`dim_0`, `session`) are still read by position, and
  `multitaper_connectivity` assigns axes from dimension names itself.
- `Multitaper` reports the number of tapers used: the column count for
  custom tapers (2.x reported `n_tapers` if given, else the count NW
  implies); for DPSS the requested count less any that low-bias filtering
  dropped (see the low-bias entry above).
- `Multitaper` rejects an impossible `n_tapers` when it is constructed: zero,
  negative, larger than the window, a boolean, or disagreeing with the columns
  of supplied `tapers`, or not a whole number. 2.x accepted these: it failed
  later inside `fft()` with an opaque SciPy `select_range` error, used `True`
  as one taper, used the supplied `tapers` while reporting the disagreeing
  count, or truncated a fractional count while reporting it as given.
- `canonical_coherence` overstated the coherence of a group containing a dead,
  duplicated, or (numerically) collinear channel, by more the fewer the
  observations (0.4-0.7 with 6, 0.25-0.35 with 20, 0.06-0.07 with 100 in the
  cases run): the
  SVD-based whitening turned each zero singular direction into an arbitrary
  unit-norm null-space direction, whose value depended on the LAPACK build.
  After each channel is scaled to unit norm (so the value does not depend on
  channel units), a direction with singular value at most `eps * max(n, m)` of
  the group's largest (the usual numerical-rank threshold) is now dropped, so
  the value equals that of the group's independent channels, and a warning names
  the group. Weak but real directions above that threshold are kept, so the
  value is unchanged by invertible within-group mixing up to condition number
  1e8 (tested). A group with no power at a bin (every signal zero)
  now gives NaN there, with a warning, as coherency does, instead of an
  arbitrary value. The whitening runs at complex128 or better, so complex64
  coefficients give the same value as complex128. Likewise a numerically
  zero-power component on the thin path of `global_coherence` now has a zero
  vector instead of an arbitrary one.
- On the GPU, the Wilson factorization behind spectral Granger, DTF, PDC, and
  directed coherence, and the whitening in `canonical_coherence`, crashed with
  `CUSOLVERError: CUSOLVER_STATUS_INVALID_VALUE` once a decomposition held more
  than about 2**21 2-by-2 matrices (about 1M for 32 signals; windows x
  frequencies x pairs, e.g. subset pairwise Granger over all 120 pairs of 16
  signals for 30 windows of 1000-sample FFTs, or canonical coherence over 4000
  windows x 551 frequencies). CuPy's batched
  `eigh` (cuSOLVER `syevjBatched`) rejects such batches: the largest that
  worked was 2,078,879 matrices for 2 signals and 1,016,319 for 32 (CuPy 14.2,
  CUDA 12.9). These eigendecompositions now run in chunks well below that
  limit, which gives the same result as one call.
  NumPy results are unchanged.
- With CuPy 14, importing the package on the GPU emitted a `FutureWarning` from
  CuPy's own `cupyx.scipy.signal` (`cupyx.jit.rawkernel is experimental`),
  which failed `pytest -W error` runs, including the GPU release gate. That
  warning is now suppressed at the import.
- `fourier_connectivity` raised `TypeError: Implicit conversion to a NumPy
  array` when given the device (CuPy) `frequencies` or `time` that
  `Multitaper` returns on the GPU; they are now moved to the host.
- On the GPU, complex64 Fourier coefficients gave cross-spectral matrices with
  single-precision error (~1e-7 relative) although they are documented to be
  accumulated at the requested `dtype` (complex128 by default): CuPy's
  `matmul(..., dtype=)` multiplies in the input precision and casts only the
  product. The operands are now cast first, as NumPy does. NumPy results are
  unchanged.
- `minimum_phase_decomposition` raised `TypeError` on the GPU for a NumPy
  cross-spectral matrix; it now accepts host arrays (the result is a CuPy
  array on the GPU).
- *Unreleased:* `group_delay` and `delay` unwrapped the coherence phase before
  excluding undefined bins, so one bin without a defined phase (e.g. a
  zero-power DC bin) made every later frequency NaN; the unwrapping now skips
  such bins. An undefined bin inside the band still splits it: only the
  longer contiguous run of significant bins is used.
- `group_delay`'s `r_value` for the reverse pair `[..., j, i]` had the forward
  pair's sign although the slope is negated; it is now `-r_value[..., i, j]`
  (in 2.x visible only with exactly two observations; with more, `r_value`
  was `NaN` off the diagonal, see the migration guide).
- `delay`'s reverse-pair candidates were the negated forward candidates in the
  same order, so reverse candidate `k` held the value for `-k`; every
  candidate now follows `(phase + 2 pi k) / (2 pi f)` for its own pair.
- The paper tutorial's Dhamala 2a/2b panels plotted `[..., 0, 1]` under the
  title "x1 -> x2", which in 2.x was the x2 -> x1 entry. The re-executed
  figures are now labeled correctly.
- *Unreleased:* Warnings from `Connectivity` measures, the spectral Granger
  kernels, and the Wilson factorization now name the user's line. Several
  pointed inside the package instead (e.g. `magnitude_squared_coherence`'s
  single-observation warning, and every Granger NaN-pair warning reached
  through `multitaper_connectivity`), which also defeated per-module warning
  filters.
- The input warnings raised while `multitaper_connectivity` builds its
  `Multitaper` (NaN or infinite input, likely transposed data, gaps between
  windows, an unusually large `time_halfbandwidth_product`) now name the user's
  line instead of the `Multitaper(...)` call inside `wrapper.py`. Their
  categories and messages are unchanged. Every warning a transform issues
  itself now names the user's line (a NumPy or SciPy `RuntimeWarning` from
  non-finite input still names a library line).
- `global_coherence` is reproducible run to run on the CPU backend. In 2.x
  every call with `max_rank < n_signals - 1` used SciPy's `svds` from a random
  start, so the values varied at rounding level and the component vectors'
  phase was arbitrary between runs.
- 2.x results could not be written to NetCDF by any engine
  (`mt_summarize_parameters` held a bound method). Every real-valued result
  now writes with the SciPy, netCDF4, and h5netcdf engines (boolean attributes
  are stored as 0/1). The complex results (`coherency`,
  `cross_spectral_density`, `canonical_coherency`, `global_coherence`) do not
  write with the SciPy engine, which cannot store complex values, nor with
  netCDF4 unless `auto_complex=True` is passed; h5netcdf writes them.
- *Unreleased:* `frequency_band_reduce`'s integral covers the whole band
  (off-grid edges, one-bin bands, additive adjacent bands) instead of only the
  span between its outer bins; default `fourier_connectivity` coordinates are
  labeled as normalized frequency and window index, and datetime time is
  rejected; a frequency-reduced measure's band no longer leaks onto other
  Dataset variables, and crop, decimation and band provenance is recorded on
  the variables it applies to.
- `canonical_coherence` warns when a group pair has more signals than
  trial x taper observations; its value is then forced to 1 for any data and
  was previously returned silently.
- *Unreleased:* Directed measures use at least complex128 working precision,
  so complex64 inputs can satisfy the Wilson factorization's default
  tolerance.
- *Unreleased:* Subset spectral Granger restores `NaN` on the global
  self-Granger diagonal.
- *Unreleased:* `partial_coherence` loads each signal's power by
  `regularization` times itself before inverting, instead of every signal by
  `regularization` times the matrix's RMS entry, which perturbed a signal far
  weaker than the others by far more than `regularization` of its own size.
  It inverts the coherency matrix, so it is invariant under rescaling the
  signals to rounding; before, signal amplitudes
  from 1e-6 to 1e6 moved the default result by up to 0.98 from that of the
  same data at one scale (simulated dense 8-signal VARs with 10 to 100
  observations). For signals of comparable power the default result changed
  by about 1e-11 or less; with fewer observations than signals, where it
  warns, the loading sets the estimate and it changed by up to 0.45. There,
  without shrinkage, the estimate is sensitive to rounding (perturbing the
  data by one unit in the last place moved it by up to 4e-4), and rescaling
  moves it by about as much.
- Wilson initialization and solves isolate rank-deficient sub-spectra instead of
  poisoning a whole batch. A unit whose Cholesky start fails starts from the
  Cholesky factor of `n_signals * I`, which is deterministic; healthy units
  keep their own starts. In 2.x one rank-deficient window gave every window a
  random start, which changed the healthy windows' values (by 6e-4 in one
  case) and made them differ from run to run.
- Wilson convergence is relative to the scale of the data, and non-converged
  units return `NaN` with a targeted warning. 2.x used an absolute threshold,
  so its Granger and transfer-function values changed with the data's units:
  by 1e-4 to 0.1 for data at 1e-6 of unit variance (microvolt signals stored
  in volts) in the cases run, and by more below that. Recompute directed
  results from small-amplitude data.
- The Wilson factorization behind every directed measure starts from a
  factor covariant under rescaling the signals (`U` with `U Uᵀ = R0`, not
  `chol(R0)ᵀ`, for the zero-lag matrix `R0`), so the factor of `D S D` is
  `D G`. From the old start it could converge, without a warning and with
  `G Gᴴ` within 1e-9 of `S`, to a factor that is not minimum phase, and every
  directed measure was then wrong: DTF and PDC were 0.3-0.8 off for 9 of 400
  random VARs of comparable-power signals (now none; at most 5e-9), and
  pairwise Granger 6.5 off for a chain of nearly integrating signals (now
  3e-7). 2.x has the same start and defect. Recompute directed results.
- As preconditioning, every signal is scaled to unit mean power for the
  factorization and the factor unscaled afterwards. Pairwise Granger of raw
  and standardized multitaper estimates of that chain differed by 2.7-5.8 at
  their largest in five simulations (3.7-5.9 in 2.x) and now agree to within
  the convergence tolerance (at most 7e-7). Signals of comparable power move by
  much less than their estimation error: at most 7e-3 in the regression
  examples, under a sixth of their RMS error against the generating models,
  which is unchanged to 0.1%. A signal with no power is factored as an
  independent white signal at rounding-level power, so its window needs no
  fallback start; on CuPy, whose Cholesky returns NaN rather than raising,
  a window with a dead channel was NaN and now factors as on NumPy.
- The inverses behind the directed measures (of the factor's lag-0
  coefficient, the transfer function, the reduced transfer function, and
  the noise covariance or its target block) load each diagonal entry by
  1e-12 of its own size instead of 1e-12 of the matrix's RMS entry, which
  perturbed a signal far weaker than the others by far more (PDC 0.9 off for
  signal powers from 5e-5 to 3e12). The spectral Granger family, directed
  coherence, and gPDC are now invariant under rescaling the signals, and
  DTF, PDC, and dDTF covariant.
- The factorization warns, as a best-effort check, when a converged factor's
  determinant winds around zero (so the factor is not minimum phase); the
  spectral Granger measures warn once per call, naming the signal pairs. The
  winding is counted from the determinant's unit phase where it turns by less
  than `π/2` between bins, and a nonzero count is recounted on grids 2 to 16
  times finer, so that steps a coarse grid aliases do not warn. It catches
  most wrong factors but not all: in sweeps of 800 random dense VARs factored
  from the old start it caught 41 of the 44 that converged to a wrong factor
  (the three it missed were off by 2e-4 to 3e-2), and it flagged none of the
  800 factors from the new start, nothing in the test suite, and none of
  1440 windows of zero-padded multitaper estimates. It cannot judge a
  determinant whose phase turns by more than `π/2` between bins, and a
  factor that fails to be minimum phase without its determinant winding
  countably is not caught; that includes a grid too coarse for the
  spectrum, where four coincident resonators at radius 0.995 factor 0.8 off
  with no warning on 4096 bins (and correctly on 16384).
- The Wilson iteration builds each update from a square root of the
  cross-spectrum, so it is Hermitian positive semidefinite by construction.
  Sub-spectra with near-duplicate channels (condition numbers from about 1e10)
  now converge. 2.x stopped at its iteration cap there and returned the
  unconverged factor, so directed measures differ from 2.x, by more the worse
  the conditioning (1e-6 near 1e10, and 1e-4 to 0.5 at 1e12-1e14 in the cases
  run). Other near-singular spectra, such as one channel a lagged copy of
  another, can still fail and return `NaN` with the warning, as an indefinite
  cross-spectrum does.
- Directed-measure regularization is scale-invariant, directed coherence uses
  the correct source-axis noise variance, and it warns when correlated
  innovations materially violate its diagonal-covariance assumption.
- GPU boundaries for `group_delay`, `delay`, statistics, metadata, and public
  returns now use explicit device-to-host conversion.
- Global coherence avoids overflow or underflow for extreme input scales.
  (`power` returns float32 for complex64 Fourier coefficients passed directly
  to `Connectivity`, as in 2.x; `Multitaper` uses float64 tapers, so its
  coefficients and power are double precision whatever the input dtype.)
- On a pair with a dead channel, `debiased_squared_phase_lag_index` is 0
  (2.x: `-1/(n-1)`) and `debiased_squared_weighted_phase_lag_index` is 0 (2.x:
  `NaN`); the phase-locking value and pairwise phase consistency stay `NaN`
  there and say so in a `UserWarning`. The documented finite-sample ranges of
  the phase-locking and phase-lag measures are corrected.
- Group delay and delay use the exact zero-coherence significance distribution.
  In 2.x their one-sample Fisher z-scores were `NaN` for three or more
  observations (`coherence_fisher_z_transform` used its `n_obs2=0` default as a
  sample size), so `group_delay` returned `NaN` delays and slopes and `delay`
  only the constants `2πk` between signals; the one-sample z-score now has no
  second-sample terms. Undefined p-values are excluded consistently from BH and
  Bonferroni families.
- Statistical helpers validate confidence levels, spectra, firing rates,
  p-values, correction methods, and, in the z-transforms and
  `power_confidence_intervals`, observation counts (`coherence_bias`,
  `power_bias`, and `power_variance` still accept any count). Power confidence
  intervals and log-power bias/variance use the corrected formulas: in 2.x a
  nominal 95% interval covered 90%, and two-sample power z-scores were 1.4 to
  1.6 times too large (a nominal 5% test rejected 17-19% under the null). Every
  one-sample `power_fisher_z_transform` call was `NaN` in 2.x, with `spectrum2`
  omitted (its default was 0) or given, because `n_obs2=0` was used as a sample
  size; `spectrum2` now defaults to 1.0 and the one-sample z has no
  second-sample terms.
- Many invalid inputs now raise an error naming the parameter, where 2.x
  failed inside SciPy or NumPy or accepted them: at construction for the taper
  count, `start_time`, a non-finite or non-numeric `sampling_frequency`, the
  DPSS arguments, and coordinate arrays; when `fft()` runs for an FFT size
  below the window, a window longer than the data, and a zero-length window or
  step. A `NaN` duration or a fractional sample count still fails there with
  NumPy's own message.
- The wrapper accepts documented 2-D `(time, channels)` input and produces
  NetCDF-serializable metadata. A multi-measure request no longer drops a
  measure that raises `NotImplementedError`: 2.x returned the other measures
  without it.
- The documentation build installs the current checkout on Read the Docs,
  confines generated sources to ignored directories, and the introductory
  tutorial no longer uses the removed `blocks` argument.
- The xarray wrapper now labels pairwise and subset spectral Granger
  prediction so that `sel(source=a, target=b)` is the influence *from* `a` *to*
  `b`; previously the `source`/`target` axes were transposed, silently
  returning the reverse direction. Recompute any Granger results obtained
  through `multitaper_connectivity`. The underlying
  `Connectivity` arrays now use the same source-first order (see Changed).
- The wrapper rejects an empty `method` list, duplicate `signal_names`, and
  requests where no method yields a compatible result, instead of silently
  returning an empty or mislabeled `Dataset`.
- `multitaper_connectivity(..., squeeze=True)` keeps the selected `source` and
  `target` as scalar coordinates instead of dropping them, so the squeezed
  `(time, frequency)` result still records which pair (and, for directed
  measures, which direction) it represents. `squeeze=True` is now honored only
  for single-method (DataArray) requests -- because those scalar coordinates
  are Dataset-wide, applying them in a multi-measure Dataset would collide with
  a sibling `power` variable's `source` dimension -- and is ignored, with a
  warning, for multi-measure requests. It is a no-op for `power`.
- Importing the package no longer changes NumPy's global floating-point warning
  state, and backend reporting now reflects the backend actually imported.
- `simulate_MVAR` with one signal and several trials raised `ValueError` in
  2.x; it now returns `(n_time, n_trials, 1)`.
- *Unreleased:* `frequency_band_reduce(frequency_reduction="integral")`
  half-counted the DC and Nyquist bins of a one-sided spectrum; integrating
  `power` over every bin now equals `sum(power) * spacing` (Parseval), so a
  band that includes DC on undetrended data is no longer short by half the DC
  cell. The Nyquist bin is the bin at the result's recorded `nyquist_frequency`
  (or a 2.x result's `mt_nyquist_frequency`), recorded as half the sampling
  rate wherever that is known, so an odd-length FFT's last bin, which lies
  below Nyquist, is integrated as an ordinary bin, and a band ending at a
  `frequency_range` or `frequency_decimation` crop edge is not integrated as if
  its last bin were Nyquist (which added 3-15% to, e.g., a 4-8 Hz band cropped
  at 8 Hz). An integrated band is labeled `"Band power"` or `"Band
  cross-power"` in `(units)^2` rather than keeping the density's `"Power
  spectral density"` label and `(units)^2/Hz`.
- *Unreleased:* Band integrals keep the Nyquist bin of a frequency coordinate
  that does not hold the Nyquist frequency exactly. The last bin is at
  `nyquist_frequency` by the rule that places a bin on a band's bound, so a
  coordinate stored as float32 (as netCDF often does) keeps it on NumPy 1.x too
  (NumPy 2 already compared the two in float32), and `fourier_connectivity`
  reads a two-sided grid's rate from the bin farthest from 0 Hz (the Nyquist
  bin of an even-length grid), in the grid's own precision (float32 at least),
  instead of multiplying the first bin's rounding by the FFT length, so float32
  and rounded FFT-order grids record the right `sampling_frequency` (exactly
  for an even length, to within a unit in the grid's last place for an odd
  one). A band ending on the Nyquist bin of such a grid was short by half that
  bin's cell.
- *Unreleased:* `frequency_band_reduce` integrates each variable of a Dataset
  with that variable's recorded Nyquist frequency. A Dataset assembled from
  result variables has no attributes of its own, so a cropped grid's last bin
  was taken for the Nyquist bin and a band ending there came out high by half
  that bin's cell (4-10% of a 4-8 Hz band on 0.5 Hz bins, about 50% of a band
  one bin wide).
- *Unreleased:* With `frequency_range`, `multitaper_connectivity` and
  `fourier_connectivity` integrate a band on the uncropped grid, so a band
  inside the range gives the same value with and without the range even when
  its edge falls between bins (without `frequency_decimation`, whose kept bins
  start at the first bin in the range). Cropping first dropped the part of the
  band lying in the cell of a bin just outside the range (up to half a bin's
  cell at each end: 2.5% of a 20 Hz band per end on 1 Hz bins). A band reaching
  past the range is cut at the range's bounds, or at the outer edge of the
  first or last selected bin's cell where that lies further out.
  `frequency_band_reduce` cannot recover that part from a result that was
  cropped first, by `frequency_range` or `sel`: integrate before cropping, or
  crop at least one bin wider than the bands. That is the only route for
  `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet` results, so leave
  `frequency_range` of `connectivity_to_xarray` unset for them.
- `suggest_parameters` no longer loses a taper to floating-point error (in
  Python-float or float64 arguments) when the resolution and the shortened
  window put `2 * NW` on a whole number (30 Hz over a 0.3 s record suggested NW
  1.4999999999999998 and one taper; now 1.5 and two). Its other suggestions
  keep their 2.0.1 values exactly.
- A window step given in seconds is rounded to the nearest sample instead of
  truncated (`time_window_step=0.57` at 100 Hz gave 56 samples).
  `Multitaper.time_window_step` and `time_window_duration` (and the
  `mt_time_window_step`/`mt_time_window_duration` result attributes) report the
  rounded step and window actually used, and `frequency_resolution` (with
  `mt_frequency_resolution`) follows that window, rather than the request
  (`time_window_step=0.01` at 250 Hz reported 0.01 s while stepping 2 samples,
  0.008 s). Giving both a duration and a sample count for the window or the
  step now raises `ValueError` when they disagree; previously the value in
  seconds silently won for both (`time_window_step=0.2` with
  `n_time_samples_per_step=7` at 100 Hz stepped 20 samples).
- `weighted_phase_lag_index` collapsed to about 0 for small-amplitude inputs
  because of an absolute epsilon guard: at 1 kHz it began to fail for data with
  a standard deviation of 1e-6 and was about 1e-17, whatever the true value, by
  3e-7 (MEG in tesla, for example). `debiased_squared_phase_lag_index` returned
  `-1/(n-1)` on the diagonal and at the purely real DC and Nyquist bins, and up
  to 1 for a zero-lag pair, instead of 0. The phase-lag family is now invariant
  to the signal's amplitude scale (`debiased_squared_weighted_phase_lag_index`
  still returns `NaN` when its squared moment under- or overflows, at data
  scales beyond about 1e-80 or 1e80). "No phase lag" is judged relative to the
  pair's power, so a channel paired with a scaled copy of itself (zero-lag
  coupling, whose imaginary cross-spectrum is rounding noise rather than
  exactly 0) gives 0 for both, and 0 for `phase_lag_index` (2.x: up to +-1 with
  a few observations, 0.3 with 100) and 0.5 for `directed_phase_lag_index`
  rather than a direction read from the sign of that noise.
  `debiased_squared_weighted_phase_lag_index` is 0 there too, instead of a
  ratio of rounding errors (anywhere in [-1, 1] with a few observations, about
  0.2 with 100) or NaN (0 / 0 on the diagonal and at the purely real DC and
  Nyquist bins).
- *Unreleased:* `fourier_connectivity(is_one_sided=False)` on unlabeled
  coefficients honors the declaration and runs the two-sided-only measures,
  which the default `is_one_sided=None` does not. Because one-sided (e.g.
  `rfft`) coefficients declared two-sided give silently wrong Granger values,
  those measures warn when the declared coefficients are not
  conjugate-symmetric (as the FFT of real-valued signals is), which is also
  expected for complex-valued signals.
- *Unreleased:* DataArray time coordinates are judged uniform step by step:
  each step must be within a quarter of the sampling interval, and the axis may
  drift from the regular grid by at most a quarter interval plus one part in
  1e6 of the elapsed time. Axes built with `np.cumsum` or `np.linspace` (hours
  long, at any start offset) and timestamps jittered by a small fraction of an
  interval are therefore accepted with or without `sampling_frequency`, while a
  single dropped or duplicated sample, or an explicit rate that disagrees with
  the axis, is still rejected.
- CuPy backend: `Connectivity` moves host coefficients and frequencies to the
  device, `canonical_coherence` indexes with device group indices, `time` stays
  a host array after coefficient reassignment, and `MorletWavelet` accepts
  device arrays for `frequencies` and `n_cycles`, so `fourier_connectivity` and
  `canonical_coherence` run on the GPU.
- *Unreleased:* `conditional_` and `blockwise_spectral_granger_prediction`
  returned float32 under `dtype=complex64`; they now return float64 like the
  other Granger variants. Every spectral Granger measure (pairwise,
  time-reversed, subset, conditional, blockwise) warns once, naming the `source
  -> target` pairs (0-based signal indices, or group labels) that are `NaN` at
  every frequency because their factorization was singular or did not converge
  or the target has no power, as with a duplicated or dead channel. The
  factorization's pair-less "did not converge" warning is no longer repeated
  for these measures' own factorizations (conditional Granger's full model,
  cached and shared with DTF and PDC, keeps it), and the advice names
  `minimum_phase_max_iterations` instead of a regularization parameter no
  Granger method has.
- `coherence_fisher_z_transform` accepts scalar coherency, and
  `power_confidence_intervals` documents that its count is the number of
  averaged observations (tapers x trials), not the taper count, and names it
  `n_observations` (see Deprecated).
- The simulated-examples tutorial no longer stores a local absolute path in a
  warning output, and `help(spectral_connectivity)` links both the AI-assistant
  guide and `llms.txt`.
- The paper-examples tutorial simulates 20 s trials for Wen example 2, where it
  simulated 5 s (as it has since 2.x). The model's five signals resonate within
  0.6 Hz of each other near 40 Hz, and with 5 s trials the tapers blended them:
  the partial directed coherence showed an influence of x3 on x5, 0.19 at 40
  Hz, that the model does not have, and pairwise spectral Granger from x4 to
  x5 read 6.5 where the model has 4.3. Every example of that tutorial is now
  tested against the true values of the model it simulates
  (`tests/test_tutorial_paper_examples.py`).
- *Unreleased:* `multitaper_connectivity` given one signal for a measure that
  relates signals says that the time series has 1 signal and the measure needs
  at least 2, before any FFT, where it raised `Connectivity`'s message about
  `fourier_coefficients`, an argument the wrapper's caller never passes. It
  points to `method='power'` for one signal and to keeping the signal axis
  when selecting a channel (`x[:, [i]]`, not `x[:, i]`). The message names
  at most three of the requested measures and counts the rest.
  `connectivity_to_xarray` given a transform of one signal says, likewise,
  that the transform has 1 signal (after the FFT, since a custom transform
  may give its signal count only through its coefficients).
- A time series containing NaN or Inf gives one warning instead of three or
  more. `multitaper_connectivity` warned about the time series, then about
  `fourier_coefficients` (an argument its caller never passes), then with
  NumPy `RuntimeWarning`s such as "invalid value encountered in divide";
  `Connectivity.from_transform` after `Multitaper`,
  `ShortTimeFourierTransform`, or `Welch` gave the first two, and the
  transforms' `fft()` a `RuntimeWarning` too. In 2.0.1,
  `multitaper_connectivity` gave the time-series warning and the
  `fourier_coefficients` warning (for `coherence_magnitude` in the case run),
  each a dozen lines of generic causes and suggestions. The remaining warning
  gives the
  number of such values and the trials and signals holding them, says that
  they propagate to the outputs (NaN power for the affected signals in the
  windows containing them, NaN connectivity for every pair involving those
  signals, or for every pair with measures that model all signals jointly,
  while `canonical_coherence` and `global_coherence` raise a `ValueError`),
  and says to interpolate the samples or drop the trials or signals, with
  `np.argwhere(~np.isfinite(x))` to find them (the indices it gives are
  0-based). The input is still
  accepted. `Connectivity` built directly from coefficients, or from another
  transform, still warns about NaN or Inf in `fourier_coefficients`, and its
  docstring no longer says they must be finite. `Connectivity` measures
  called after `from_transform` can still emit NumPy `RuntimeWarning`s on
  such values; `multitaper_connectivity` does not.
- The spectral Granger and transfer-function measures computed from NaN or
  Inf coefficients no longer add the warnings of their failed factorization,
  which blamed duplicated or collinear channels for the NaN already warned
  about: the Cholesky fallback, the Wilson non-convergence, the "NaN at every
  frequency" pair list, and the conditional and block Granger
  innovation-spectrum warnings. `pairwise_spectral_granger_prediction` gave
  four of them and DTF two, besides the NaN warning, in the case run; any
  path to `Connectivity` (the wrapper, `from_transform`, or coefficients
  passed directly) now gives the NaN warning alone, plus the
  `DirectedOrientationWarning` reminder. Finite coefficients keep every
  warning. The Cholesky fallback is no longer also logged at WARNING level,
  which printed it to stderr a second time. The warnings are dropped with
  `warnings.catch_warnings`, here and for the `fourier_coefficients` warning
  after a transform's NaN warning, which changes the process-wide warning
  filters while the measure runs: with NaN or Inf input it is not
  thread-safe (another thread's matching warnings are dropped meanwhile),
  and a warning already shown once may be shown again (not the 3.x migration
  notices, which keep their own once-per-session record). Finite input never
  changes the filters.
- `canonical_coherence` and `global_coherence` computed from NaN or Inf
  coefficients raise a `ValueError` saying the measure cannot be computed
  because the coefficients contain NaN or Inf, and to interpolate or remove
  those samples or drop the affected trials or signals; they raised LAPACK's
  `LinAlgError` ("SVD did not converge", "Eigenvalues did not converge"),
  which is kept as the cause. The same `ValueError` is raised when the
  decomposition returns NaN for such coefficients rather than raising, as
  CuPy's do (simulated here; no GPU run), unless zero observation weights
  omit them. `canonical_coherency`,
  `maximized_imaginary_coherency` (and its components), and
  `multivariate_interaction_measure` already returned NaN there, and still
  do. `global_coherence` no longer emits NumPy's "invalid value encountered
  in divide" `RuntimeWarning` before that error. Its per-bin path (used when
  both the number of signals and of observations exceed 64) raised ARPACK's
  `ArpackError` there instead, with LAPACK printing "ZLASCL ... illegal
  value" to stderr; it skips such bins and raises the same `ValueError`.
  NaN or Inf coefficients that zero weights omit do not raise: a bin of
  `canonical_coherence` or `global_coherence` left with no weighted
  observation is NaN, as for finite coefficients (with `global_coherence`'s
  zero-power warning).
- `Connectivity.group_delay` and `Connectivity.delay` raise a `ValueError`
  naming `significance_threshold` when it is not a number strictly between 0
  and 1 (e.g. 0.05), before any computation. 2.0.1 accepted any value: 1.0
  counted every bin as significant and 0 none, without a sign; development
  builds raised from deep in the statistics module about an `alpha` the
  caller never passed.
- `Multitaper`, `ShortTimeFourierTransform`, `Welch`, and
  `multitaper_connectivity` reject an unknown `detrend_type` (e.g.
  `"quadratic"`) when constructed, with a `ValueError` naming `detrend_type`
  and listing `'constant'`, `'linear'`, and `None`. It was accepted and failed
  at the first `fft()` with `detrend`'s message about its `type` argument
  (`Example: detrend(data, type='linear')`), as `Multitaper` did in 2.0.1
  (`ShortTimeFourierTransform` and `Welch` are new in 3.0). Code that catches
  this error should wrap the constructor.
- A transposed time series, e.g. `multitaper_connectivity(x.T,
  sampling_frequency=500)` for a `(1000, 3)` recording, raises a `ValueError`
  that says so: "Your array has 3 time samples and 1000 signals; expected
  (n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals). It
  looks transposed: put time on the first axis, e.g. x.T." The window error
  it replaces, which advised a longer window or a smaller
  `time_halfbandwidth_product`, is its `__cause__`; the transposition hint
  was a separate `UserWarning`. 2.0.1, which took only 3-D input, gave that
  warning and then SciPy's "select_range out of bounds" for
  `multitaper_connectivity(x.T[:, np.newaxis, :], 500)`. This applies to
  `Multitaper` (which names only its 3-D layout) and
  `ShortTimeFourierTransform` too, when an array with more signals than time
  samples is too short for its window (the window is longer than the
  recording, or too short for the DPSS tapers). Every other error on such an
  array (a bad `start_time`, say) keeps its own message, and such an array
  whose window fits still only warns. `Multitaper` now emits its warnings
  about the time series (transposed, NaN or Inf) after every check has
  passed, so a call that raises does not warn first, as
  `ShortTimeFourierTransform` already did.
- `multitaper_connectivity` and `fourier_connectivity` given a `method` that
  is neither a measure name nor a list of names (`method=3`, or a list holding
  a number) raise a `TypeError` saying what `method` may be, giving what was
  received, and pointing to `list_measures()`. They raised a bare
  `TypeError: 'int' object is not iterable`, or an `AttributeError` for a
  list holding a non-string.
- *Unreleased:* `Connectivity.jackknife` errors say how to fix the call. An
  unknown measure (`jackknife("coherence")`) gets the wrapper's suggestions
  ("Did you mean 'magnitude_squared_coherence', ...?") and a pointer to
  `list_measures()`, where the message said only that `method` must name a
  public connectivity measure. A `method` that is not a string
  (`jackknife(3)`) raises the wrappers' `TypeError` ("method must be a measure
  name (str), got int 3. ..."), where it raised an `AttributeError` from
  `str.startswith`. An invalid `confidence_level`
  (`confidence_level=95`) is echoed with "e.g. 0.95 for a 95% interval" and
  rejected before the leave-one-out estimates are computed, here and in
  `jackknife_confidence_interval`; a NumPy scalar or 0-d array is accepted as
  before. An `expectation_type` it cannot use names
  the ones it can (`'trials_tapers'`, the default, `'trials'`, or
  `'tapers'`) and the one the `Connectivity` has.
- `multitaper_connectivity`, `Multitaper`, `ShortTimeFourierTransform`,
  `Welch`, `MorletWavelet`, `suggest_parameters`, and
  `estimate_frequency_resolution` answer another package's name for the sampling
  rate (`fs`, `Fs`, `sfreq`, `srate`, `sampling_rate`, `sample_rate`) with
  "Did you mean sampling_frequency (in Hz)?", where the `TypeError` for the
  unexpected keyword suggested nothing. `transform_kwargs={"fs": ...}` gets
  the same hint. Given beside `sampling_frequency`, the alias is answered
  with "pass only sampling_frequency (in Hz)". `fourier_connectivity`, which
  takes no sampling rate, keeps Python's `TypeError` for `fs=`.
- The GPU messages of `get_compute_backend()` and of the import-time GPU
  check say to install `pip install "spectral-connectivity[gpu]"` (the extra
  that pins the CUDA 12 build, `cupy-cuda12x`), or `conda install -c
  conda-forge cupy`; they said `pip install cupy`, which builds CuPy from
  source. With CuPy installed and `SPECTRAL_CONNECTIVITY_ENABLE_GPU` set only
  after `spectral_connectivity` was imported, `get_compute_backend()` says
  that the variable must be set before the import and to restart the
  interpreter, where it said to install CuPy and set the variable, both
  already done.
- `canonical_coherence`, the magnitude of `canonical_coherency`, and
  `global_coherence` stay within [0, 1]. Where they reach 1 (fewer
  observations than signals, or a duplicated or collinear channel) rounding
  left them up to 4e-15, 1.2e-14, and 7e-16 above 1
  (`canonical_coherence` was 1.000000000000003 for white noise with 4 trials,
  8 signals in two groups, and 1 taper). Only values within 1e-10 above 1
  are set to 1: a value further above it would be an error rather than
  rounding, and is left visible. A sweep of every measure whose range is
  [0, 1] over such inputs found no other value above 1, except the magnitude
  of the complex `coherency` by up to 3 ulps (7e-16), the precision of the
  magnitude itself.
- `Multitaper`, `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet`
  given a ragged nested list, or values that are not numbers (strings,
  `None`), raise an error naming the `(n_time_samples, n_trials, n_signals)`
  array they take and how to fix it, as `multitaper_connectivity` does,
  where they raised NumPy's "inhomogeneous shape" message, a `ufunc
  'isfinite'` error for strings (`MorletWavelet` accepted an array of
  strings), or, for `Welch(None)`, one about its segment length.
  `Welch(None)` now raises a `TypeError`, not that `ValueError`.
  `fourier_connectivity` names its coefficient layouts for a ragged list
  too. An object-dtype array holding numbers is read as those numbers by all
  four transforms and both wrappers, as `MorletWavelet` read it; `Multitaper`,
  `ShortTimeFourierTransform`, and `Welch` raised NumPy's `isfinite` error
  for one.
- `multitaper_connectivity` given an array of 1 or 4 or more dimensions says
  so in its own terms: "time_series has 1 dimension, shape (200,).
  multitaper_connectivity needs (n_time_samples, n_signals) or
  (n_time_samples, n_trials, n_signals). For one signal use x[:, np.newaxis]
  with method='power'." It raised `Multitaper`'s "Expected 3D array" error,
  which points to `prepare_time_series`, a helper for `Multitaper`'s 3-D
  input.
- `Multitaper` given an `n_tapers` that is not a number (`n_tapers="5"`, or a
  list) raises a `TypeError` naming `n_tapers` and the integer it takes,
  where it raised NumPy's `ufunc 'isfinite'` error. A NumPy integer or a 0-d
  array is still accepted, and `n_tapers=True` keeps its `ValueError`.

### Performance

- On the GPU, each Wilson iteration synchronizes the device once (the
  convergence check) instead of twice: the lower-triangle indices of the causal
  projection, whose CuPy implementation synchronizes, are computed once per
  factorization. Directed transfer function on 239 windows x 250 frequencies x
  8 signals takes 0.096 s instead of 0.102 s (best of 7); pairwise Granger over
  all 28 pairs of the same data is unchanged within noise (0.91 s).
- Reduced cross-spectral matrices use a batched matrix multiplication rather
  than materializing the observation-level signal-by-signal outer product.
- Phase locking uses unit-normalized coefficients with the same batched
  reduction. Phase-lag measures use bounded signal-row tiles and cache only
  reduced moments.
- The phase-lag-index family forms its observation-level tiles in reused
  buffers. Without observation weights and with at least 64 averaged
  observations per window (`PHASE_LAG_ALL_MOMENTS_MIN_OBSERVATIONS`), the first
  phase-lag call reduces all four moments the family needs from one pass and
  retains them (up to four `(n_time_windows, n_frequencies, n_signals,
  n_signals)` arrays) until `clear_cache()`. In that regime, on a 2000-sample,
  100-trial, 32-signal multitaper spectrum (500 observations; CPU,
  `benchmarks/bench_default_measures.py`), the four phase-lag measures together
  take 1.4 s instead of 3.9 s, the ten default measures with pairwise Granger
  3.5 s instead of 5.9 s, and a single phase-lag measure is 1.1-1.9x faster
  with a lower peak (741 -> 536 MB of traced allocation). With fewer observations (e.g. a
  time-resolved single-trial spectrum with 3 tapers) or with observation
  weights, only the requested moments are computed, because there the extra
  moments cost more than re-forming the tiles; on 1199 windows x 3 tapers x
  32 signals a single phase-lag measure is 1.1-1.3x faster than before with an
  unchanged peak, and a lone weighted measure on a Hann `MorletWavelet` with
  `smoothing_time` about 1.2-1.4x faster. Outputs are unchanged: every
  default measure is bit-identical except
  `debiased_squared_weighted_phase_lag_index`, which differs only by
  summation-order rounding in the squared moment (<= 1.4e-17 on the 250- and
  500-observation benchmark cases; up to 2.5e-13 with three observations,
  where the debiasing denominator is ill-conditioned).
- Compact subset spectral-Granger factors only the requested 2-by-2 spectra.
- Global coherence uses chunked batched eigendecomposition/SVD for modest
  decomposition dimensions and retains a per-bin sparse fallback for large
  square problems. `max_workspace_elements` controls its working-set target.
- Group delay regression and significant-frequency cluster selection are
  vectorized and processed in bounded chunks.
- Wilson iteration reduces GPU synchronization, DPSS delegates to SciPy's
  compiled solver, detrending delegates to SciPy/CuPy, and sliding windows use
  `sliding_window_view`.
- Multi-measure wrapper calls build one `Connectivity` and reuse its cached
  power, cross-spectrum, directed factors, and phase-lag moments.
- `Connectivity.from_transform` adopts the transform's fresh FFT output
  without a redundant full-size copy. CPU FFT parallelism is available through
  the opt-in `fft_workers` argument.
- `MorletWavelet.fft` fills one preallocated coefficient array instead of
  stacking per-frequency results, removing one full copy of the coefficients
  from peak memory (about 3.1x to 2.1x the output size without decimation or
  overlapping smoothing).
- Coherence and phase-lag-index measures no longer retain the pairwise power
  normalizer `sqrt(P_i P_j)`; it is recomputed from the cached power on each
  call, cutting what they retain by a third relative to the preceding
  development build. (2.x retained nothing after a coherence call; 3.0 keeps
  its intermediates, 34 MB for 32 signals and 1001 frequencies, until
  `clear_cache()`.)
- The Wilson minimum-phase factorization behind the spectral Granger, DTF,
  and PDC families is 1.8-2.9x faster in benchmarks: it factors the
  cross-spectrum once and solves once per iteration instead of twice, and for
  real-valued signals iterates on the non-negative frequencies with real FFTs.
  End to end, pairwise spectral Granger for 32 signals is about 2.3x faster,
  conditional spectral Granger for 24 signals about 1.7x, and DTF for 100
  windows of 8 signals about 1.9x. For well-conditioned spectra,
  results agree with the previous implementation to rounding (absolute
  differences of 3e-15 or less on values up to 1 in those benchmarks; relative
  differences reach 3e-10 on values near 1e-11).
- The measure set that was then the default, which included pairwise spectral
  Granger (now opt-in), is about 2x faster on top of that. Every 2-signal
  Wilson factorization (pairwise and subset spectral Granger, and 2-signal
  DTF/PDC) solves its 2x2 systems in closed form; the phase-lag-index family
  reduces only the non-negative frequencies, forms the imaginary cross-spectrum
  in real arithmetic from contiguous copies of the coefficients, and, when its workspace is split into several signal
  tiles, skips each tile's pairs with earlier signals and fills them by
  symmetry; coherency, imaginary coherence, and PLV/PPC normalize only the
  frequencies they return; and the transfer function and noise covariance read
  the factor's lag-0 coefficient as a frequency mean instead of a full inverse
  FFT. `multitaper_connectivity` with that set is about 2.2x faster
  for 32 signals x 50 trials, 2.7x for 64 signals x 100 trials, and 1.5x for 16
  signals in 117 sliding windows, where pairwise Granger still dominates.
- On the GPU, `Multitaper` (and `ShortTimeFourierTransform`/`Welch`, which share
  its FFT) transforms real-valued data with a real FFT and writes the negative
  frequencies as the exact conjugate mirror. cuFFT's complex FFT of real input
  was conjugate-symmetric only to rounding (~1e-16), so the Wilson
  factorization's exact symmetry check never passed on the GPU and every
  spectral Granger, DTF, and PDC computation iterated on the full two-sided
  spectrum; it now takes the half-spectrum path, as on the CPU. On one shared
  GPU, `directed_transfer_function` is 1.7x faster (0.161 -> 0.093 s for 30
  windows x 10 trials x 5 tapers x 1000 frequencies x 16 signals; 0.095 ->
  0.054 s for 100 trials x 5 tapers x 2000 frequencies x 32 signals) and
  `Multitaper.fft` 1.1-1.2x faster; pairwise spectral Granger is unchanged
  within timing noise. GPU outputs change at rounding level (<= 1.8e-14) and
  stay as close to the CPU's; the Nyquist bin is now exactly real, so
  `coherence_phase` there is 0 or +pi as on the CPU, never a rounding-dependent
  -pi. Complex-valued input keeps the full FFT. The CPU keeps SciPy's complex
  FFT, which is already exactly symmetric for real input and was faster than a
  real FFT plus mirror (0.31 vs 0.35 s on the first size); its outputs are
  bit-identical.
- `canonical_coherence` and the thin path of `global_coherence` (fewer
  trial x taper estimates than signals) no longer call `svd` on matrices with a
  dimension above 32, which CuPy decomposes one bin at a time. Canonical
  coherence whitens each group with the pseudo-inverse square root of the
  smaller Gram matrix (`eigh`) at complex128, applied twice so the second pass
  restores SVD accuracy; bins where that whitening drops a direction (singular
  value below about 1e-7 of the largest, which the Gram matrix cannot resolve)
  are recomputed with an SVD (of the small R factor of a QR decomposition, which
  CuPy batches for groups of up to 32 signals). A group with a dead or
  duplicated channel takes that path at every bin: on the benchmark case below
  1.21 s instead of 0.32 s on the GPU and 1.73 s instead of 1.51 s on the CPU,
  and for two groups of 40 signals (`(10, 10, 5, 500, 80)`) 14.6 s instead of
  5.5 s on the GPU and 5.0 s instead of 3.6 s on the CPU; global coherence
  diagonalizes the `(n_estimates, n_estimates)` Gram matrix and maps its
  eigenvectors to the left singular vectors. Best of 3-5 interleaved runs:
  canonical coherence on `(30 windows, 10 trials, 5 tapers, 1000 frequencies,
  16 signals)` in four groups of four takes 0.32 s instead of 37.8 s on the GPU
  (A100) and 1.34 s instead of 1.85 s on the CPU; global coherence
  (`max_rank=3`, 64 signals) takes 0.12 s instead of 22.5 s on the GPU and
  1.03 s instead of 1.59 s on the CPU for 100 windows x 5 tapers x 500
  frequencies, and 1.13 s instead of 38.3 s (GPU) and 3.85 s instead of 7.38 s
  (CPU) for 20 windows x 40 tapers. Results agree with the SVD to rounding
  (relative differences <= 5e-14 here; vectors equal up to phase) except for
  rank-deficient groups (see Fixed).
- On the GPU, pairwise, time-reversed, and blockwise spectral Granger factor
  many signal (or same-sized group) pairs per Wilson call instead of one pair
  at a time. One pair's factorization there is bound by kernel launches and
  synchronizations rather than arithmetic. Chunks hold at most
  `GRANGER_GPU_BATCH_MAX_WORKSPACE_ELEMENTS` (2**22) spectral elements, which
  bounds working memory and keeps CuPy's batched eigendecomposition under its
  batch limit (a single pair larger than that is factored on its own). Subset spectral Granger factors in the same bounded chunks, and
  its per-pair scatter, a boolean-mask assignment in the predictive power, and
  conditional Granger's per-pair warning check no longer synchronize the
  device. Measured on one GPU (best of 3): pairwise spectral Granger in
  `benchmarks/bench_default_measures.py` takes 0.36 s instead of 12.8 s for
  2000 samples x 100 trials x 32 signals, and 0.076 s instead of 1.05 s for 10
  windows x 50 trials x 8 signals. For 30 windows x 10 trials x 5 tapers x 1000
  FFT bins x 16 signals, pairwise and time-reversed take 1.6 s instead of
  3.4-4.5 s, and blockwise (8 groups of 2) 0.7 s instead of 1.1-1.4 s.
  Conditional Granger, dominated by its 15-signal reduced factorizations, is
  unchanged. Each pair still converges on its own, so results match the
  pair-by-pair factorization to rounding (<= 3.6e-15 in the tests). On the
  CPU, pairwise, time-reversed, and blockwise Granger keep one pair per call,
  which measured faster than batching there (subset Granger factors all its
  pairs in one call, as before). Conditional Granger's degenerate-bins warning
  is now issued once per call instead of once per pair.

## [2.0.1] - 2026-05-12

### Fixed

- Broadcast identity matrix to batch dimensions of the LHS before calling `xp.linalg.solve` in `_estimate_transfer_function` and `_MVAR_Fourier_coefficients`, fixing CuPy batched-solve shape mismatch crashes that affected `directed_transfer_function`, `partial_directed_coherence`, `generalized_partial_directed_coherence`, and `direct_directed_transfer_function` on GPU.

## [2.0.0] - 2025-10-27

### BREAKING CHANGES

#### 3D Input Requirement for Multitaper Class

**BREAKING CHANGE**: `Multitaper` now requires 3D input arrays with explicit `(n_time_samples, n_trials, n_signals)` shape. Previously ambiguous 2D arrays now raise informative `ValueError`.

- **Migration Required**: Use the new `prepare_time_series()` helper function to convert 1D/2D arrays to 3D format
- **Why**: Eliminates dangerous ambiguity where `(n_time, n)` could mean either:
  - `(n_time, 1 trial, n signals)` OR
  - `(n_time, n trials, 1 signal)`

  Silent misinterpretation produces scientifically incorrect results.

**Migration Example**:
```python
# Before (ambiguous)
mt = Multitaper(eeg_data, sampling_frequency=1000)

# After (explicit)
from spectral_connectivity.transforms import prepare_time_series

eeg_3d = prepare_time_series(eeg_data, axis="signals")  # or axis='trials'
mt = Multitaper(eeg_3d, sampling_frequency=1000)
```

#### Nyquist Frequency Now Included for Even-Length FFTs

**BREAKING CHANGE**: Corrected frequency bin indexing to include Nyquist frequency for even-length FFTs

- **Impact on results**:
  - Even-length FFTs (N=1024): Now return 513 frequencies instead of 512 (adds Nyquist bin)
  - Odd-length FFTs (N=1023): No change (still return 512 frequencies)
- **Affected functions**: All connectivity measures using `@_non_negative_frequencies` decorator
- **Scientific justification**: The Nyquist frequency (sampling_rate/2) represents valid spectral information that should not be discarded
- See CHANGELOG "Unreleased" section for complete details and migration guidance

#### Development Tooling Changes

- **BREAKING**: Migrated from `black` to `ruff format` for code formatting
- **BREAKING**: Migrated from `flake8`/`pydocstyle` to `ruff` for linting
- Development workflows now use `ruff format` and `ruff check` commands
- All formatting/linting configurations moved to `pyproject.toml` under `[tool.ruff]`
- Developers must update their tooling: `pip install ruff` (100x faster than previous tools)

### Added
- `prepare_time_series()` helper function for safe dimension handling:
  - Converts 1D/2D arrays to required 3D format
  - Explicit `axis` parameter to clarify dimension meaning
  - Prevents ambiguous dimension interpretation
  - **Required for migration to v2.0.0**
- Comprehensive notebook snapshot testing with Syrupy:
  - 27 snapshot tests covering all tutorial notebook examples
  - Tests validate output shapes, data types, and numerical properties
  - Ensures tutorial examples remain accurate across releases
  - Automatically catches breaking changes in example code
  - Uses Syrupy's snapshot testing for efficient test maintenance
- **Test infrastructure** (`tests/conftest.py`):
  - Added pytest fixture to reset numpy random state before each test
  - Ensures consistent test reproducibility with fixed seed (42)
  - Helps prevent test ordering issues from shared random state
  - Auto-runs before every test without explicit fixture declaration
- **Physical constants with scientific rationale** (`spectral_connectivity/transforms.py`):
  - `MIN_EIGENVALUE_THRESHOLD = 0.9`: Minimum eigenvalue for low-bias tapers
    - Documents that 90% of taper energy must be in main lobe to reduce spectral leakage
    - Reference: Thomson (1982), "Spectrum estimation and harmonic analysis"
  - `TAPER_MULTIPLIER = 2.0`: Multiplier for calculating number of tapers
    - Documents the theoretical basis: ~2*NW orthogonal Slepian sequences are well-concentrated
    - Reference: Slepian (1978), "Prolate spheroidal wave functions"
  - All magic numbers replaced with named constants throughout codebase
  - Docstrings updated to reference constants while preserving readability
- Comprehensive test suite for advanced connectivity measures (`tests/test_advanced_connectivity.py`):
  - 18 test methods covering `canonical_coherence()`, `global_coherence()`, and `group_delay()`
  - Tests validate output shapes, value ranges, and mathematical properties (symmetry, antisymmetry)
  - Integration tests for complete Multitaper → Connectivity → advanced measures workflow
  - Tests handle edge cases: different group sizes, single signals, non-contiguous labels
  - Graceful handling of group_delay edge case when no frequencies are significant
- Parameter helper functions for guided multitaper analysis:
  - `estimate_frequency_resolution()`: Calculate frequency resolution from parameters
  - `estimate_n_tapers()`: Calculate number of tapers from time-halfbandwidth product
  - `suggest_parameters()`: Get parameter recommendations for your data
  - `MultitaperParameters` TypedDict for type-safe parameter handling
- `summarize_parameters()` method to `Multitaper` class:
  - Human-readable summary of all analysis parameters
  - Shows computed values (n_tapers, frequency_resolution, n_windows)
  - Displays overlap percentage for windowing
  - Formatted for terminal/notebook output
- Enhanced `time_halfbandwidth_product` docstring with formulas and practical guidance:
  - Mathematical relationship to frequency resolution and n_tapers
  - Typical values (NW=2,3,4,5+) with trade-off explanations
  - Examples for achieving target resolutions (1 Hz, 5 Hz, 10 Hz)
  - Cross-references to helper functions
- Comprehensive test suite for parameter helpers (`test_parameter_helpers.py`):
  - 22 tests covering all helper functions
  - Domain-specific tests (EEG, LFP typical parameters)
  - Edge cases (conflicting parameters, impossible resolutions)
  - Consistency checks with actual `Multitaper` behavior

### Changed
- **Complete type hint coverage** (`spectral_connectivity/transforms.py`, `spectral_connectivity/connectivity.py`):
  - Added type hints to ALL 28 previously untyped functions:
    - `transforms._make_tapers()`: Added parameter and return types
    - `connectivity.py`: 27 functions including all connectivity measures and helper functions
  - Used `TYPE_CHECKING` block to avoid circular imports for Multitaper type
  - Added `Literal` types for `multiple_comparisons_method` parameter
  - Fixed `Optional`/`None` handling in `_get_independent_frequency_step()` and `_bandpass()`
  - Fixed phase_lag_index return type (extract real part from complex result)
  - Enabled `disallow_untyped_defs = true` for transforms and connectivity modules
  - All 9 modules now have strict type checking enabled
  - 100% of public API is now fully type-annotated
- **Test coverage improvements**:
  - Overall coverage increased from 85% to **88%**
  - `connectivity.py` coverage improved from 71% to **93%**
  - Fixed snapshot tests after Nyquist frequency bin correction
  - Updated test assertions to use correct frequency bin calculation: `N//2 + 1` instead of `(N+1)//2`
- Improved method discovery in `multitaper_connectivity()` wrapper:
  - Replaced `dir()` with `inspect.getmembers(predicate=inspect.isfunction)` for type-safe method filtering
  - Automatically excludes properties and classmethods (more robust)
  - Renamed `bad_methods` to `excluded_methods` with clear categorization
  - Changed from list to set for O(1) membership testing
  - Added test `test_method_discovery_with_inspect()` to verify behavior
- Documented design decision for `adjust_for_multiple_comparisons()`:
  - Replaced TODO comment with clear explanation of current behavior
  - Explained why axis parameter is not implemented (standard approach treats all p-values as single family)
  - Left open for future enhancement if needed

### Removed
- All TODO comments from codebase (2 resolved)

### Changed
- **Development tooling modernization**:
  - Migrated from `black` to `ruff format` for code formatting (100x faster)
  - Migrated from `flake8`/`pydocstyle` to `ruff check` for linting
  - Updated GitHub Actions CI to use `ruff format --check` and `ruff check`
  - Updated [CLAUDE.md](CLAUDE.md) with new ruff commands
  - All formatting and linting now unified under single fast tool
- **Code quality improvements**:
  - Applied `ruff format` to all source files (10 files reformatted)
  - Fixed 19 auto-fixable ruff linting issues
  - All source code and tests now pass `ruff check` with zero warnings
  - Type hints improved with better union handling for `detrend()` function
- Updated tutorial notebooks to use `prepare_time_series()` for 3D input
- Improved random number generation in tests for better isolation

### Fixed
- **MyPy type annotation error** in `detrend()` function (`transforms.py:1867-1876`):
  - Fixed union type handling for `bp` parameter (now `int | list[int] | NDArray[np.integer]`)
  - Added support for list input in addition to int and ndarray
  - Eliminated unreachable code warnings from MyPy
  - All 9 source files now pass strict mypy type checking

#### **Nyquist Frequency Fix (from earlier release)**

- **Critical bug fix**: Corrected frequency bin indexing to include Nyquist frequency for even-length FFTs
  - **Affected code**: Changed frequency indexing from `(N+1)//2` to `N//2 + 1` in three locations:
    - `_non_negative_frequencies` decorator (line 107)
    - `canonical_coherence` method (line 638)
    - `_estimate_spectral_granger_prediction` function (line 2108)
  - **Impact on results**:
    - Even-length FFTs (N=1024): Now return 513 frequencies instead of 512 (adds Nyquist bin)
    - Odd-length FFTs (N=1023): No change (still return 512 frequencies)
  - **Affected functions**: All connectivity measures using `@_non_negative_frequencies` decorator:
    - `coherency()`, `coherence_magnitude()`, `coherence_phase()`, `imaginary_coherence()`
    - `phase_lag_index()`, `weighted_phase_lag_index()`, `debiased_squared_phase_lag_index()`
    - `debiased_squared_weighted_phase_lag_index()`, `phase_locking_value()`, `pairwise_phase_consistency()`
    - `power()`, all Granger causality measures (DTF, PDC, etc.)
  - **Scientific justification**: The Nyquist frequency (sampling_rate/2) represents the highest frequency
    that can be unambiguously represented in sampled data. For even-length FFTs, this frequency should be
    included once (not in negative frequencies). Excluding it discards valid spectral information and
    violates standard FFT conventions (numpy.fft.rfft, scipy.fft.rfft).
  - **Migration impact**:
    - New analyses: More accurate (includes previously missing frequency information)
    - Comparing to old results: Array shapes differ by 1 in frequency dimension for even-length FFTs
    - Published results: Document which version was used in methods section
  - **Tests added**:
    - `test_nyquist_bin_even_n()`: Validates N=1024 produces 513 frequencies
    - `test_nyquist_bin_odd_n()`: Validates N=1023 produces 512 frequencies
  - **Example**: With 1000 Hz sampling and 1024-sample FFT:
    - Old (incorrect): 512 bins, missing 500 Hz (Nyquist)
    - New (correct): 513 bins, includes 500 Hz (Nyquist)
  - See PR #71 for detailed analysis and discussion

- **Tikhonov regularization for MVAR matrix inversion stability**:
  - Replaced direct matrix inverse (`xp.linalg.inv()`) with Tikhonov-regularized solve in MVAR computations
  - Prevents `LinAlgError` exceptions when computing Granger causality with near-singular matrices
  - Affected functions: `_MVAR_Fourier_coefficients` property and `_estimate_transfer_function` function
  - Uses scale-aware regularization: λ = `TIKHONOV_REGULARIZATION_FACTOR` × mean(||H||²)
  - Added module-level constant `TIKHONOV_REGULARIZATION_FACTOR = 1e-12` for consistency
  - Solves `(H + λI)x = I` instead of computing `inv(H)` for better numerical stability
  - Added stress test `test_mvar_regularized_inverse_near_singular()` validating near-singular cases
  - All Granger causality measures now handle highly correlated signals gracefully
  - See PR #72 for detailed numerical analysis

- **Numerical stability improvements for coherence calculations**:
  - Replaced zero-clamping with epsilon-clamping in `coherency()` to prevent division by zero
  - Changed from `norm[norm == 0] = xp.nan` to `norm = xp.maximum(norm, xp.finfo(norm.dtype).eps)`
  - Added bounds clipping to `coherence_magnitude()` to ensure values stay in [0, 1] range
  - Added bounds clipping to `imaginary_coherence()` with epsilon protection
  - Prevents numerical artifacts from floating-point precision issues
  - More graceful degradation for low-power signals compared to NaN propagation
  - Added comprehensive test suite (`test_coherence_bounds.py`) with 3 tests covering edge cases
  - See PR #62 for detailed analysis

- CHANGELOG.md to track version changes following Keep a Changelog format
- Ruff linter configuration for faster, more comprehensive Python linting
- Enhanced package metadata with additional project URLs (Changelog, Source Code, Issue Tracker)
- Modern unified CI/CD workflow (`release.yml`) with automated PyPI publishing
- Support for Python 3.13
- Comprehensive parameter validation to `Multitaper` class:
  - Validates `sampling_frequency > 0` with domain-specific examples (EEG, LFP, fMRI)
  - Validates `time_halfbandwidth_product >= 1` with physical meaning explanation
  - Validates `time_window_duration > 0` (when provided) with frequency resolution formula
  - Validates `time_window_step > 0` (when provided) with overlap guidance
  - Warns when `time_halfbandwidth_product > 10` (unusually large, performance impact)
  - Warns when `time_window_step > time_window_duration` (creates data gaps)
  - Warns when data appears transposed (`n_time < n_signals`)
  - Warns when input contains NaN or Inf values with recovery suggestions
- Input shape validation to `Connectivity` class:
  - Requires 5D `fourier_coefficients` with clear error messages
  - Validates minimum 2 signals for connectivity analysis
  - Warns on NaN/Inf values in Fourier coefficients
- `prepare_time_series()` helper function for safe dimension handling:
  - Converts 1D/2D arrays to required 3D format
  - Explicit `axis` parameter to clarify dimension meaning
  - Prevents ambiguous dimension interpretation
- Enhanced error messages following WHAT/WHY/HOW pattern throughout
- 3D input requirement for `Multitaper` class to eliminate dimension ambiguity
- Intelligent error suggestion for `expectation_type` parameter:
  - Detects wrong word order (e.g., "tapers_trials" instead of "trials_tapers")
  - Suggests correct ordering with helpful explanation
  - Lists all valid options with most common choice highlighted
- Improved `detrend()` function error messages:
  - Clear explanation of linear vs constant detrending
  - Examples with domain-specific terminology (DC offset, best-fit line)
  - Actionable guidance for parameter selection
- Enhanced breakpoint validation in `detrend()`:
  - Shows specific invalid breakpoint values
  - Displays valid range based on actual data dimensions
  - Includes user's original input for easy debugging
- Comprehensive test suite for error message quality (`test_error_messages.py`)
- GPU status utility function `get_compute_backend()`:
  - Returns dict with backend type, GPU availability, device name, and helpful message
  - Shows actual GPU model name (e.g., "NVIDIA Tesla V100-SXM2-16GB") instead of compute capability
- Comprehensive test suite for block-wise computation:
  - 5 new tests validating correctness, symmetry, edge cases, and memory reduction
  - Empirical memory measurement showing 73% reduction for n_signals=50, blocks=5
  - Tests cover different expectation types and block configurations
- Enhanced documentation for `blocks` parameter in Connectivity class:
  - "When to use" and "When NOT to use" guidance with specific thresholds
  - Quick decision guide based on n_signals
  - Memory-speed tradeoff quantified (70-80% memory reduction, <10% speed penalty)
  - GPU VRAM considerations documented
  - Detects CuPy availability without side effects (uses `importlib.util.find_spec`)
  - Provides 4 different message variants for different GPU configurations
  - Includes comprehensive NumPy-style docstring with 3 usage examples
  - Example return value documented in docstring
  - Located in new `spectral_connectivity.utils` module
- Enhanced GPU device logging in `transforms.py` and `connectivity.py`:
  - Now shows actual GPU model name in log messages
  - Graceful fallback to compute capability if name unavailable
- Comprehensive GPU documentation in README.md:
  - 130+ line GPU Acceleration section with setup, troubleshooting, and usage guidance
  - 3 setup methods documented (shell, Python script, Jupyter notebook)
  - Verification steps included in all setup examples
  - Simplified CuPy installation instructions (conda recommended first)
  - Troubleshooting guide with 4 common issues and solutions
  - Clear explanation of import timing requirement (why "before importing" matters)
  - Example outputs shown for all code samples
  - Kernel restart guidance for Jupyter notebook users
  - Guidance on when GPU acceleration is beneficial
- Comprehensive test suite for GPU backend (`tests/test_gpu.py`):
  - 13 test methods covering all GPU configuration scenarios
  - Tests for CPU mode (default and explicit)
  - Tests for GPU mode (with and without CuPy available)
  - Validation of return value structure and types
  - Mock-based testing to avoid CuPy dependency
  - All tests pass (11 passed, 1 skipped when CuPy unavailable)

### Changed
- **BREAKING**: Minimum Python version raised from 3.9 to 3.10
- Migrated from flake8 to ruff for linting (100x faster, replaces flake8, isort, pydocstyle)
- Updated dependency pins: numpy>=1.24, scipy>=1.10, xarray>=2023.1, matplotlib>=3.7
- Improved mypy configuration with stricter type checking and per-module overrides
- Updated development documentation (CLAUDE.md, CONTRIBUTING.md) to reflect current tooling
- Expanded test matrix to Python 3.10, 3.11, 3.12, 3.13
- Consolidated CI workflows: removed redundant PR-test.yml and linting.yml in favor of release.yml
- Simplified CI from conda-based to pip-based installation (faster builds)
- Enhanced black configuration to target Python 3.10-3.13
- Updated ReadTheDocs to use Python 3.10

### Fixed
- Outdated release instructions in CONTRIBUTING.md (removed setup.py references)
- Deprecation warning in `minimum_phase_decomposition.py`: Changed `xp.linalg.linalg.LinAlgError` to `xp.linalg.LinAlgError` for compatibility with NumPy 2.0+
- **Critical bug in block-wise computation**: Fixed missing diagonal elements in cross-spectral matrix when using `blocks` parameter (changed `k=1` to `k=0` in `triu_indices`)
- **Critical bug in block-wise computation**: Fixed incorrect Hermitian symmetry in blocked cross-spectral matrix computation (added conjugate when filling transpose positions)

## [1.1.2] - 2023-10-17

### Added
- Conda packaging support with conda-recipe directory
- CLAUDE.md with development commands and architecture documentation
- Pinned coverage reporter version in CI workflow to avoid bugs

### Changed
- Updated module docstrings for clarity and context
- Updated README with Contributing and License sections

### Fixed
- Linting issues resolved

## [1.1.1] - 2023-09-15

### Changed
- Switch build system from setuptools to Hatch
- Add py.typed marker for type hint support
- Update and reorganize dependencies in environment.yml

### Fixed
- Resolve n_time_samples_per_window property logic error in transforms module
- Resolve mypy Optional[int] vs int return type errors
- Correct _fix_taper_sign return type annotation

## [1.1.0] - 2023-08-20

### Added
- GPU request guard feature to safely handle CUDA availability
- Complete audit of connectivity metric ranges documentation
- ValueError raised when window size parameters are unset

### Changed
- Updated GitHub Actions to latest versions
- Improved type hints throughout codebase

## [1.0.4] - 2023-03-15

### Fixed
- Bug fixes in connectivity calculations
- Improved numerical stability

## [1.0.3] - 2023-02-10

### Changed
- Performance improvements
- Documentation updates

## [1.0.2] - 2023-01-20

### Fixed
- Minor bug fixes
- Test coverage improvements

## [1.0.1] - 2022-12-15

### Fixed
- Package distribution fixes
- Documentation corrections

## [1.0.0] - 2022-12-01

### Added
- First stable release
- Full implementation of 15+ connectivity measures
- GPU acceleration support via CuPy
- Comprehensive test suite
- Complete documentation on ReadTheDocs

### Changed
- API stabilized for 1.0 release
- Performance optimizations

## [0.2.7] - 2022-06-15

### Added
- Additional connectivity measures
- Improved caching strategy

### Changed
- API improvements and refinements

## [0.2.6] - 2022-03-10

### Added
- Initial GPU support
- More connectivity measures

### Changed
- Refactored core architecture
- Improved documentation

---

[Unreleased]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.2...v2.0.0
[1.1.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.7...v1.0.0
[0.2.7]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/releases/tag/v0.2.6
