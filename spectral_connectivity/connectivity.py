"""Compute metrics for relating signals in the frequency domain."""

import warnings
import weakref
from collections.abc import Callable
from functools import cached_property, partial, wraps
from itertools import combinations
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

from spectral_connectivity.minimum_phase_decomposition import (
    minimum_phase_decomposition,
)
from spectral_connectivity.statistics import (
    adjust_for_multiple_comparisons,
    coherence_significance_pvalue,
)
from spectral_connectivity.utils import freeze_readonly, is_gpu_enabled, to_numpy

if TYPE_CHECKING:
    from spectral_connectivity.transforms import Multitaper

logger = getLogger(__name__)

if is_gpu_enabled():
    try:
        import cupy as xp
        from cupyx.scipy.fft import ifft
        from cupyx.scipy.sparse.linalg import svds

        # Log GPU device information
        try:
            device = xp.cuda.Device()
            # Try to get the actual GPU model name first
            try:
                device_name = xp.cuda.runtime.getDeviceProperties(device.id)[
                    "name"
                ].decode()
                device_name = device_name.strip("\x00")
            except Exception:
                # Fallback to compute capability
                compute_cap = device.compute_capability
                device_name = (
                    f"GPU (Compute Capability {compute_cap[0]}.{compute_cap[1]})"
                )
            logger.info(f"Using GPU for spectral_connectivity on {device_name}")
        except Exception:
            logger.info("Using GPU for spectral_connectivity...")
    except ImportError as exc:
        raise RuntimeError(
            "GPU support was explicitly requested via SPECTRAL_CONNECTIVITY_ENABLE_GPU='true', "
            "but CuPy is not installed. Please install CuPy with: "
            "'pip install cupy' or 'conda install cupy'"
        ) from exc
else:
    logger.info("Using CPU for spectral_connectivity...")
    import numpy as xp
    from scipy.fft import ifft
    from scipy.sparse.linalg import svds

EXPECTATION_AXES = {
    "time": (0,),
    "trials": (1,),
    "tapers": (2,),
    "time_trials": (0, 1),
    "time_tapers": (0, 2),
    "trials_tapers": (1, 2),
    "time_trials_tapers": (0, 1, 2),
}
EXPECTATION = {
    name: partial(xp.mean, axis=axes) for name, axes in EXPECTATION_AXES.items()
}

# Tikhonov regularization factor for stabilizing matrix inversions
# Used to prevent numerical instability with near-singular matrices
TIKHONOV_REGULARIZATION_FACTOR = 1e-12

# global_coherence computes, per time-frequency bin, the strongest components of
# the (n_signals, n_estimates) coefficient matrix. When the decomposition
# dimension min(n_signals, n_estimates) is modest these are found with a single
# batched decomposition over all bins (eigh of the cross-spectral matrix when
# n_estimates >= n_signals, otherwise the economy SVD of the thin matrix),
# replacing a Python loop over bins and its per-bin device syncs on GPU. Above
# this dimension the per-bin path is used (it finds only the requested top
# components via svds when max_rank is small), where forming every component of
# a large matrix would be wasteful.
GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS = 64
# Bin-chunk element cap for the batched path: peak memory scales with
# chunk * (n_signals * n_estimates + min(n_signals, n_estimates)**2), so cap the
# element count to keep it bounded regardless of the number of bins.
GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS = 16_000_000
# Peak workspace cap for the phase-lag-index family's observation-level signal
# tiles. The final reduced signal-by-signal result is unavoidable, but the large
# trial/taper/time-resolved outer product is never materialized in full.
PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS = 16_000_000

# Largest relative gap between the diagonal-noise-power denominator that
# ``directed_coherence`` uses and the true power spectral density, tolerated
# before it warns that its diagonal-noise-covariance assumption is violated (see
# the note in that method). A dimension-aware criterion: unlike a pairwise
# correlation threshold, it catches many weakly-but-jointly correlated sources
# whose cross-power still omits a large fraction of the true power. Non-parametric
# estimation of a truly diagonal covariance leaves a small discrepancy from
# finite-sample noise, so the threshold sits above that floor: only a material
# omission (>= 10% of the true power) triggers the warning.
DIRECTED_COHERENCE_DISCREPANCY_TOLERANCE = 0.1

# Preserves a helper's input dtype (real vs complex) in its return annotation.
_NumberT = TypeVar("_NumberT", bound=np.number)


def _asnumpy(connectivity_measure: Callable) -> Callable:
    """Transform cupy array to numpy array.

    If cupy is not installed, then return original.

    Parameters
    ----------
    connectivity_measure : callable
        Connectivity measure function to wrap.

    Returns
    -------
    callable
        Wrapped function that converts output to numpy.

    """

    @wraps(connectivity_measure)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        measure = connectivity_measure(*args, **kwargs)
        if measure is not None:
            return to_numpy(measure)
        else:
            return None

    return wrapper


def _ignore_nan_propagation_warnings(connectivity_measure: Callable) -> Callable:
    """Suppress NumPy invalid/divide warnings from expected NaN propagation.

    The directed measures (DTF, PDC and relatives) normalize the transfer
    function or MVAR coefficients by an inflow/outflow sum. When the Wilson
    minimum-phase decomposition fails to converge those inputs are NaN (already
    warned about at decomposition time), and the normalization then emits
    ``invalid value encountered in divide``. Scope the suppression to these
    measures rather than silencing NumPy globally; well-conditioned inputs
    produce no NaN and are unaffected.

    Parameters
    ----------
    connectivity_measure : callable
        Connectivity measure method to wrap.

    Returns
    -------
    callable
        Wrapped method whose numpy arithmetic runs under a scoped errstate.
    """

    @wraps(connectivity_measure)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with np.errstate(invalid="ignore", divide="ignore"):
            return connectivity_measure(*args, **kwargs)

    return wrapper


def _non_negative_frequencies(axis: int) -> Callable:
    """Remove the negative frequencies.

    Parameters
    ----------
    axis : int
        Axis along which to remove negative frequencies.

    Returns
    -------
    callable
        Decorator function.

    """

    def decorator(connectivity_measure: Callable) -> Callable:
        @wraps(connectivity_measure)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            measure = connectivity_measure(*args, **kwargs)
            if measure is not None:
                n_frequencies = measure.shape[axis]
                non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
                return xp.take(measure, indices=non_neg_index, axis=axis)
            else:
                return None

        return wrapper

    return decorator


def _freeze_writeable_chain(array: NDArray) -> None:
    """Mark ``array`` and every array in its ``.base`` chain read-only (NumPy).

    A NumPy view shares its buffer with its ``.base``, so freezing only the
    outer array leaves that buffer writable through the base. ``Multitaper.fft``
    returns a ``swapaxes`` view whose base *is* the writable owning array, so the
    whole chain must be frozen for the adoption path to be safe. Backends without
    a settable ``writeable`` flag (e.g. CuPy) raise, which is ignored: adoption
    there still relies on the array being unshared, and the getter returns a
    fresh copy on such backends anyway. ``.base`` chains are acyclic, so the walk
    terminates at the owning array.
    """
    obj: NDArray | None = array
    while obj is not None:
        try:
            obj.flags.writeable = False
        except (AttributeError, ValueError):
            pass
        obj = getattr(obj, "base", None)


class Connectivity:
    """
    Compute functional and directed connectivity measures from spectral data.

    This class provides a comprehensive suite of connectivity analysis methods
    based on cross-spectral matrices derived from Fourier-transformed time series.
    Methods range from basic coherence to advanced Granger causality measures.

    Parameters
    ----------
    fourier_coefficients : NDArray[complexfloating], shape (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals)
        Complex-valued Fourier coefficients from spectral analysis. Must be
        two-sided (positive and negative frequencies) for Granger methods.
        Usually obtained from multitaper or other spectral estimation methods.
        **Validation**: Must be 5-dimensional with at least 2 signals and
        contain only finite values (no NaN/Inf).
    expectation_type : {"trials_tapers", "trials", "tapers", "time",
        "time_trials", "time_tapers", "time_trials_tapers"},
        default="trials_tapers"
        Specifies how to average the cross-spectral matrix:
        - "trials_tapers": average over trials and tapers (most common)
        - "trials": average over trials only (keep taper dimension)
        - "tapers": average over tapers only (keep trial dimension)
        - "time": average over time windows
        - combinations: average over multiple dimensions
    frequencies : NDArray[floating], shape (n_frequencies,), optional
        Frequency values in Hz corresponding to FFT bins. If None, uses
        normalized frequencies.
    time : NDArray[floating], shape (n_time_windows,), optional
        Time values in seconds for each time window. If None, uses indices.
    dtype : np.dtype, default=complex128
        Data type for internal computations. Should match input precision.
    minimum_phase_tolerance : float, default=1e-8
        Relative convergence tolerance for the Wilson minimum-phase
        factorization used by the directed measures (spectral Granger, DTF,
        PDC, and relatives).
    minimum_phase_max_iterations : int, default=500
        Maximum Wilson iterations. Near-singular cross-spectral matrices (highly
        correlated channels) can need several hundred iterations; if the
        directed measures return NaN with a non-convergence warning, increase
        this value. The factorization returns early once every sub-spectrum has
        converged, so a large ceiling is cheap for well-conditioned data.

    Attributes
    ----------
    n_observations : int
        Effective number of independent observations after averaging,
        used for statistical inference.

    See Also
    --------
    spectral_connectivity.transforms.Multitaper : Produce the Fourier
        coefficients this class consumes.
    spectral_connectivity.wrapper.multitaper_connectivity : High-level interface
        returning labeled xarray results.

    Notes
    -----
    Expensive intermediates (the minimum-phase factor, transfer function, noise
    covariance, and MVAR coefficients) are cached on first access. Reassigning
    ``fourier_coefficients`` or ``expectation_type`` automatically invalidates
    these caches, so reusing an instance for new data is safe (constructing a
    new instance is still the clearer choice).

    The class supports both CPU (NumPy) and GPU (CuPy) computation depending
    on the SPECTRAL_CONNECTIVITY_ENABLE_GPU environment variable. For Granger
    causality measures, minimum phase decomposition [1]_ is used to estimate
    transfer functions and noise covariances non-parametrically.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.
    .. [2] Bastos, A. M., & Schoffelen, J. M. (2016). A tutorial review of
           functional connectivity analysis methods and their interpretational
           pitfalls. Frontiers in systems neuroscience, 9, 175.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_connectivity import Connectivity
    >>> rng = np.random.default_rng(0)
    >>> n_times, n_trials, n_tapers, n_freqs, n_signals = 50, 10, 5, 100, 2
    >>> # Create complex coefficients with coherence injected at frequency bin 10
    >>> phase_diff = np.pi / 4  # 45 degree phase difference
    >>> coeffs = (
    ...     rng.standard_normal((n_times, n_trials, n_tapers, n_freqs, n_signals))
    ...     + 1j
    ...     * rng.standard_normal((n_times, n_trials, n_tapers, n_freqs, n_signals))
    ... )
    >>> coeffs[:, :, :, 10, 1] = coeffs[:, :, :, 10, 0] * np.exp(1j * phase_diff)
    >>> conn = Connectivity(coeffs, expectation_type="trials_tapers")
    >>> coherence = conn.coherence_magnitude()
    >>> coherence.shape  # (n_times, non-negative freqs, n_signals, n_signals)
    (50, 51, 2, 2)
    >>> print(f"Peak coherence: {np.max(coherence[:, 10, 0, 1]):.3f}")
    Peak coherence: 1.000
    """

    def __init__(
        self,
        fourier_coefficients: NDArray[np.complexfloating],
        expectation_type: str = "trials_tapers",
        frequencies: NDArray[np.floating] | None = None,
        time: NDArray[np.floating] | None = None,
        dtype: np.dtype = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
        *,
        _adopt_fourier_coefficients: bool = False,
    ) -> None:
        # fourier_coefficients and expectation_type are validated in their
        # property setters (below), which also clear the cached intermediates so
        # reassigning either on an existing instance cannot serve stale results.
        # _adopt_fourier_coefficients is a private fast path for from_multitaper:
        # the fft() output is unshared, so it is frozen in place instead of
        # copied, avoiding a transient 2x peak of the largest array. It must not
        # be set for a caller-owned array (see _adopt_fourier_coefficients).
        if _adopt_fourier_coefficients:
            self._adopt_fourier_coefficients(fourier_coefficients)
        else:
            self.fourier_coefficients = fourier_coefficients
        self.expectation_type = expectation_type
        # Wilson minimum-phase factorization controls, used by the directed
        # measures. Near-singular cross-spectral matrices can need more than the
        # default iterations to converge; exposing these lets callers recover
        # (the non-convergence warning advises increasing max_iterations).
        self._minimum_phase_tolerance = minimum_phase_tolerance
        self._minimum_phase_max_iterations = minimum_phase_max_iterations
        # Fill documented defaults when coordinates are omitted: normalized
        # (sampling-frequency-1) FFT frequencies and integer time-window indices.
        # Otherwise coordinate-dependent methods (delay, group_delay,
        # canonical_coherence) would dereference None.
        n_fft_samples = self._fourier_coefficients.shape[-2]
        n_time_windows = self._fourier_coefficients.shape[0]

        # Supplied coordinates must be 1-D, finite, and match the data geometry.
        # A wrong shape (e.g. (n, 1)) or length would silently misalign/drop
        # frequency bins (phase_slope_index) or crash with a broadcasting error
        # (delay, group_delay, canonical_coherence); non-finite coordinates
        # would propagate NaN into the frequency step and delays.
        def _validate_coordinate(name: str, coord: Any, expected_length: int) -> None:
            arr = xp.asarray(coord)
            if arr.ndim != 1:
                raise ValueError(
                    f"{name} must be a 1-D array, got shape {tuple(arr.shape)}."
                )
            if arr.shape[0] != expected_length:
                raise ValueError(
                    f"{name} must have length {expected_length}, got {arr.shape[0]}."
                )
            if not bool(xp.all(xp.isfinite(arr))):
                raise ValueError(f"{name} must contain only finite values.")

        if frequencies is not None:
            _validate_coordinate("frequencies", frequencies, n_fft_samples)
        if time is not None:
            _validate_coordinate("time", time, n_time_windows)
        if frequencies is None:
            frequencies = xp.fft.fftfreq(n_fft_samples)
        if time is None:
            time = xp.arange(n_time_windows)
        self._frequencies = frequencies
        self._dtype = dtype
        # Set by from_multitaper to a weakref to the source Multitaper, so
        # consumers can verify provenance by identity (see from_multitaper).
        # None for a directly-constructed instance, whose source is unknown.
        self._source_multitaper: weakref.ref | None = None
        # Snapshot of the source Multitaper's parameters at build time (see
        # from_multitaper); None for a directly-constructed instance.
        self._source_parameters: dict[str, Any] | None = None
        try:
            self.time = xp.asnumpy(time)
        except AttributeError:
            self.time = time

    def __getstate__(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return picklable state, dropping the transient provenance weakref.

        ``from_multitaper`` stores a ``weakref`` to its source ``Multitaper`` in
        ``_source_multitaper``; weakrefs cannot be pickled, so an instance built
        that way would otherwise fail to serialize (a directly constructed one
        has ``None`` there and pickles fine). Provenance is a live-object link
        that cannot survive serialization anyway, so clear it: an unpickled
        instance behaves like a directly constructed one (usable everywhere
        except as an identity-verified injected ``connectivity=`` argument).

        The state is returned as a ``(dict_state, slot_state)`` pair (paired with
        :meth:`__setstate__`) so attributes declared through a subclass'
        ``__slots__`` are preserved — returning only ``__dict__`` would silently
        drop them on pickle / ``copy.copy`` / ``copy.deepcopy``. The base class
        keeps its data in ``__dict__``; the slot walk covers slotted subclasses.
        """
        dict_state = dict(getattr(self, "__dict__", {}))
        dict_state["_source_multitaper"] = None
        slot_state: dict[str, Any] = {}
        for klass in type(self).__mro__:
            slots = getattr(klass, "__slots__", ())
            # ``__slots__`` may be a single string (Python treats it as one slot);
            # iterating it directly would walk characters, so normalize it.
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in ("__dict__", "__weakref__"):
                    continue
                # Private slot names are name-mangled (``__x`` -> ``_Class__x``);
                # the descriptor lives under the mangled name.
                if slot.startswith("__") and not slot.endswith("__"):
                    slot = f"_{klass.__name__.lstrip('_')}{slot}"
                try:
                    slot_state[slot] = getattr(self, slot)
                except AttributeError:
                    pass  # an unset slot has no value to serialize
        return dict_state, slot_state

    def __setstate__(
        self, state: tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]
    ) -> None:
        """Restore state from :meth:`__getstate__` or a legacy plain dict.

        Pickles written before this class defined ``__getstate__`` (e.g. from an
        earlier release) contain a plain ``__dict__`` rather than the
        ``(dict, slots)`` pair, so accept both forms; the provenance fields that
        did not exist then are initialized to ``None``.
        """
        if isinstance(state, tuple):
            dict_state, slot_state = state
        else:
            dict_state, slot_state = state, {}  # legacy plain-dict pickle
        if dict_state:
            self.__dict__.update(dict_state)
        for name, value in (slot_state or {}).items():
            setattr(self, name, value)
        # Legacy pickles predate the provenance fields; ensure they exist.
        self.__dict__.setdefault("_source_multitaper", None)
        self.__dict__.setdefault("_source_parameters", None)

    # Cached quantities that depend on fourier_coefficients / expectation_type
    # and must be invalidated when either changes (see the setters below).
    _CACHED_INTERMEDIATES = (
        "_power",
        "_cached_reduced_cross_spectral_matrix",
        "_imaginary_moment_cache",
        "_minimum_phase_factor",
        "_transfer_function",
        "_noise_covariance",
        "_MVAR_Fourier_coefficients",
    )

    def _clear_cached_intermediates(self) -> None:
        """Drop cached_property results that depend on the inputs."""
        for name in self._CACHED_INTERMEDIATES:
            self.__dict__.pop(name, None)

    @property
    def fourier_coefficients(self) -> NDArray[np.complexfloating]:
        """Multitaper Fourier coefficients.

        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        Stored as an immutable snapshot so the cached intermediates (power,
        cross-spectrum, directed factors) cannot be silently made stale by
        in-place edits. This accessor returns an independent, **read-only copy**:
        an in-place edit (``c.fourier_coefficients[...] = x``) raises rather than
        silently vanishing, and because the returned array is a copy (it owns its
        data, so ``.base`` is ``None``) it is fully disconnected from the snapshot
        — even re-enabling its ``writeable`` flag only affects the caller's throw-
        away copy, never the instance. (A read-only *view* would not be safe: its
        owning base is reachable through ``.base``, whose ``writeable`` flag a
        caller can re-enable and then mutate the snapshot behind the caches.) On
        backends without a settable ``writeable`` flag (e.g. CuPy) the copy is
        returned writable, which is harmless because it is still independent. To
        change the data, assign a new array (which clears the caches) rather than
        mutating the returned copy. Internal computations read
        ``self._fourier_coefficients`` directly, so the copy is paid only on
        explicit external access, not on the hot paths.

        ``from_multitaper`` skips the *construction-time* copy: the
        ``Multitaper.fft()`` output is freshly built and unshared, so it is frozen
        in place (the whole ``.base`` chain, since ``fft()`` returns a view)
        rather than duplicated, avoiding a transient doubling of the largest
        array. That is independent of this getter, which still copies on access.
        """
        # Hand out an independent, read-only copy. copy() severs any link to the
        # snapshot (the result owns its data, base is None), so a read-only *copy*
        # is safe even if the caller re-enables its writeable flag -- unlike a
        # read-only view, whose owning base is reachable and re-enable-able. The
        # read-only flag makes an in-place edit raise loudly instead of silently
        # vanishing against a discarded copy. On CuPy the freeze is a no-op; the
        # copy is still independent, so a write to it is harmless.
        return freeze_readonly(self._fourier_coefficients.copy())

    @fourier_coefficients.setter
    def fourier_coefficients(self, value: NDArray[np.complexfloating]) -> None:
        # Public assignment always copies defensively: the caller may keep its
        # array and later mutate it in place, which would silently invalidate the
        # cached intermediates.
        self._set_fourier_coefficients(value, adopt=False)
        # Replacing the data breaks any provenance link recorded by
        # from_multitaper: the instance no longer holds the source transform's
        # coefficients, so it must not still claim that source or its parameters.
        self._source_multitaper = None
        self._source_parameters = None

    def _adopt_fourier_coefficients(self, value: NDArray[np.complexfloating]) -> None:
        """Take ownership of a freshly produced, unshared array without copying.

        Used only by ``from_multitaper``, where ``value`` is the
        ``Multitaper.fft()`` output and is referenced nowhere else. This avoids a
        full copy of the largest array in the pipeline -- a transient ~2x peak on
        every construction, which can push a memory-constrained GPU into OOM.

        This is deliberately private and has no public ``copy=False`` surface: it
        is safe only when the caller guarantees the array *and its writable NumPy
        base buffer* are unshared, which ``from_multitaper`` controls.
        """
        self._set_fourier_coefficients(value, adopt=True)

    def _set_fourier_coefficients(
        self, value: NDArray[np.complexfloating], *, adopt: bool
    ) -> None:
        if value.ndim != 5:
            raise ValueError(
                f"fourier_coefficients must be 5-dimensional, got {value.ndim}D array.\n"
                f"Expected shape: (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)\n"
                f"Got shape: {value.shape}\n\n"
                f"If you have time series data, use the Multitaper class to transform it:\n"
                f"  from spectral_connectivity import Multitaper\n"
                f"  m = Multitaper(time_series, sampling_frequency=your_fs, ...)\n"
                f"  fourier_coefficients = m.fft()"
            )
        # Power spectral density can be computed on single signals, but
        # connectivity metrics require >= 2 signals; that is validated per-method
        # in _validate_multiple_signals.
        if not xp.all(xp.isfinite(value)):
            warnings.warn(
                "fourier_coefficients contains NaN or Inf values. This may indicate:\n"
                "  - NaN/Inf in your input time series data\n"
                "  - Issues with windowing parameters (e.g., window too short)\n"
                "  - Numerical instability in preprocessing\n\n"
                "Suggestions:\n"
                "  - Check your input data for NaN/Inf values\n"
                "  - Consider interpolating missing data points\n"
                "  - Review artifact removal procedures\n"
                "  - Verify time_window_duration and time_halfbandwidth_product parameters",
                UserWarning,
                stacklevel=2,
            )
        # Own the coefficients as an immutable snapshot: the cached intermediates
        # (_power, the reduced cross-spectrum, and the directed-measure factors)
        # assume the coefficients change only through the setter, which clears
        # them. Marking the snapshot read-only turns an in-place edit via the
        # getter into a clear error rather than silently stale results.
        if adopt:
            # `value` is unshared (Multitaper.fft output) but is a swapaxes VIEW
            # whose base buffer is writable; freeze the whole base chain, not just
            # the outer view, or the data stays reachable and mutable through
            # `.base`. No copy -- this is the memory-saving path.
            _freeze_writeable_chain(value)
            owned = value
        else:
            # copy(order="K") keeps the caller's array's memory layout, so
            # downstream matmuls see the same strides and results are unchanged to
            # the bit (a plain C-order copy would perturb the BLAS summation order
            # by ~1e-16). CuPy may not support the writeable flag; the copy alone
            # still decouples the instance from later mutation of the caller's
            # array there.
            owned = freeze_readonly(value.copy(order="K"))
        value = owned
        self._fourier_coefficients = value
        self._clear_cached_intermediates()
        # On reassignment (not initial construction), a change in the number of
        # FFT bins or time windows invalidates the stored frequency/time
        # coordinates. Reset them to geometry-matching defaults so
        # coordinate-dependent methods stay consistent -- stale coordinates would
        # otherwise silently drop or misalign bins (e.g. phase_slope_index) or
        # raise (group_delay). Warn because any user-supplied coordinates (e.g.
        # Hz frequencies from from_multitaper) are discarded.
        n_fft_samples = value.shape[-2]
        n_time_windows = value.shape[0]
        frequencies_stale = (
            getattr(self, "_frequencies", None) is not None
            and len(self._frequencies) != n_fft_samples
        )
        time_stale = (
            getattr(self, "time", None) is not None and len(self.time) != n_time_windows
        )
        if frequencies_stale or time_stale:
            warnings.warn(
                "Reassigning fourier_coefficients changed the FFT/time geometry; "
                "the frequency/time coordinates were reset to defaults "
                "(normalized frequencies / integer indices). Construct a new "
                "Connectivity if you need specific coordinates for the new data.",
                UserWarning,
                stacklevel=2,
            )
            if frequencies_stale:
                self._frequencies = xp.fft.fftfreq(n_fft_samples)
            if time_stale:
                self.time = xp.arange(n_time_windows)

    @property
    def expectation_type(self) -> str:
        """Which dimensions the cross-spectral matrix is averaged over.

        Reassigning clears cached directed-connectivity intermediates.
        """
        return self._expectation_type

    @expectation_type.setter
    def expectation_type(self, value: str) -> None:
        if value not in EXPECTATION:
            # Detect the common mistake of the right words in the wrong order.
            words = set(value.split("_"))
            valid_words = {"time", "trials", "tapers"}
            suggestion = None
            if words.issubset(valid_words):
                for valid_key in EXPECTATION:
                    if set(valid_key.split("_")) == words:
                        suggestion = valid_key
                        break

            error_msg = (
                f"Invalid expectation_type '{value}' is not supported.\n"
                f"This parameter controls which dimensions to average over when computing "
                f"the cross-spectral matrix.\n"
            )
            if suggestion:
                error_msg += (
                    f"\nDid you mean '{suggestion}'? "
                    f"(The words must be in a specific order)\n"
                )
            error_msg += "\nValid options are:\n"
            for key in sorted(EXPECTATION.keys()):
                error_msg += f"  - '{key}'\n"
            error_msg += (
                "\nMost common: 'trials_tapers' (average over both trials and tapers)"
            )
            raise ValueError(error_msg)

        self._expectation_type = value
        self._clear_cached_intermediates()

    @classmethod
    def from_multitaper(
        cls,
        multitaper_instance: "Multitaper",
        expectation_type: str = "trials_tapers",
        dtype: Any = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
    ) -> "Connectivity":
        """Construct connectivity class using a multitaper instance.

        Parameters
        ----------
        multitaper_instance : Multitaper
            Instance of Multitaper class.
        expectation_type : str, default="trials_tapers"
            How to average the cross-spectral matrix.
        dtype : np.dtype, default=complex128
            Data type for computations.
        minimum_phase_tolerance : float, default=1e-8
            Relative convergence tolerance for the Wilson minimum-phase
            factorization used by the directed measures.
        minimum_phase_max_iterations : int, default=500
            Maximum Wilson iterations. Increase for near-singular cross-spectral
            matrices (highly correlated channels) that fail to converge.

        Returns
        -------
        Connectivity
            New Connectivity instance.

        """
        init_kwargs: dict[str, Any] = {
            "fourier_coefficients": multitaper_instance.fft(),
            "expectation_type": expectation_type,
            "time": multitaper_instance.time,
            "frequencies": multitaper_instance.frequencies,
            "dtype": dtype,
            "minimum_phase_tolerance": minimum_phase_tolerance,
            "minimum_phase_max_iterations": minimum_phase_max_iterations,
        }
        # fft() returns a freshly built, unshared array, so adopt it in place
        # instead of copying (see Connectivity._adopt_fourier_coefficients). Only
        # pass the private keyword when the subclass has not overridden __init__:
        # an overriding subclass need not accept it, and passing it would raise
        # TypeError. Such a subclass falls back to the defensive-copy path.
        if cls.__init__ is Connectivity.__init__:
            init_kwargs["_adopt_fourier_coefficients"] = True
        instance = cls(**init_kwargs)
        # Record a verifiable link to the source transform so consumers that take
        # data from this instance but metadata/coordinates from a Multitaper
        # (e.g. wrapper.connectivity_to_xarray) can confirm the two describe the
        # same data -- geometry alone (channel count, frequency grid, time bins)
        # cannot, since two different recordings can share it. A weakref avoids
        # extending the lifetime of the source Multitaper (and its raw
        # time_series); a dropped reference simply fails the identity check.
        instance._source_multitaper = weakref.ref(multitaper_instance)
        # Snapshot the source transform's parameters as they were at build time.
        # Multitaper configurations are immutable, and retaining the snapshot here
        # also keeps metadata self-contained after the source object is released.
        instance._source_parameters = multitaper_instance._provenance_metadata()
        return instance

    def _validate_multiple_signals(self) -> None:
        """Raise if fewer than two signals are present.

        Connectivity measures quantify relationships between pairs of signals
        and are undefined for a single signal (they would otherwise return an
        all-NaN array with no error). ``power()`` is exempt because power
        spectral density is well-defined for one signal.
        """
        n_signals = self._fourier_coefficients.shape[-1]
        if n_signals < 2:
            raise ValueError(
                f"Connectivity measures require at least 2 signals, but "
                f"fourier_coefficients has {n_signals} signal "
                f"(shape[-1] == {n_signals}).\n"
                f"Connectivity quantifies relationships between pairs of "
                f"signals; for a single signal use power() instead.\n"
                f"If you sliced to one channel, keep the signal axis, e.g. "
                f"fourier_coefficients[..., [channel_index]] rather than "
                f"fourier_coefficients[..., channel_index]."
            )

    def _validate_debiasing_observations(self, measure: str) -> None:
        """Raise if a bias-corrected measure has too few observations.

        The debiased phase-lag index and pairwise phase consistency divide by
        ``n_observations - 1`` (respectively ``n_observations ** 2 -
        n_observations``), which is zero for a single observation and would
        otherwise return silent inf/NaN.
        """
        n_observations = self.n_observations
        if n_observations < 2:
            raise ValueError(
                f"{measure} requires at least 2 observations "
                f"(n_observations == n_trials * n_tapers), but got "
                f"{n_observations}. This bias correction divides by a factor "
                f"that is zero when n_observations < 2, so it is undefined for a "
                f"single observation. Use more trials/tapers, or the "
                f"non-debiased measure (phase_lag_index / phase_locking_value)."
            )

    def _require_multiple_frequencies(self, measure: str) -> None:
        """Raise if fewer than two frequency bins are available.

        ``delay``, ``group_delay`` and ``phase_slope_index`` read
        ``frequencies[1] - frequencies[0]`` to get the frequency step; with a
        single frequency bin that indexing would raise a raw ``IndexError``
        instead of a clear message.
        """
        frequencies = self.frequencies
        n_frequencies = 0 if frequencies is None else len(frequencies)
        if n_frequencies < 2:
            raise ValueError(
                f"{measure} requires at least 2 frequency bins, but the data has "
                f"{n_frequencies}. Use a longer FFT (larger n_fft_samples / "
                f"n_time_samples_per_window)."
            )

    @property
    @_asnumpy
    def frequencies(self) -> NDArray[np.floating] | None:
        """Return non-negative frequencies of the transform.

        Returns
        -------
        NDArray[floating], shape (n_frequencies,)
            Non-negative frequency values.

        """
        if self._frequencies is not None:
            # Extract non-negative frequencies (first N//2 + 1 for even N, (N+1)//2 for odd N)
            n_frequencies = len(self._frequencies)
            non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
            freqs = xp.take(self._frequencies, indices=non_neg_index, axis=0)

            # fftfreq returns negative Nyquist for even N, fix the sign
            if len(freqs) > 0 and freqs[-1] < 0:
                freqs = freqs.copy()  # Don't modify the original
                freqs[-1] = abs(freqs[-1])
            return freqs
        return None

    @property
    @_asnumpy
    def all_frequencies(self) -> NDArray[np.floating] | None:
        """Return positive and negative frequencies of the transform.

        Returns
        -------
        NDArray[floating], shape (n_frequencies,)
            All frequency values including negative frequencies.

        """
        if self._frequencies is not None:
            return self._frequencies
        return None

    @cached_property
    def _power(self) -> NDArray[np.floating]:
        # Cached per instance (invalidated when fourier_coefficients /
        # expectation_type change; see _CACHED_INTERMEDIATES). coherency reads it
        # twice and several measures each read it, so caching avoids recomputing
        # the expectation. Consumers treat it as read-only.
        return self._expectation(
            self._fourier_coefficients * self._fourier_coefficients.conjugate()
        ).real

    @property
    def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Return the complex-valued linear association between fourier coefficients.

        Returns
        -------
        cross_spectral_matrix : array
            Shape (n_time_windows, n_trials, n_tapers, n_fft_samples,
            n_signals, n_signals). Complex cross-spectral matrix.

        """
        fourier_coefficients = self._fourier_coefficients[..., xp.newaxis]
        return _complex_inner_product(
            fourier_coefficients, fourier_coefficients, dtype=self._dtype
        )

    def _expectation_cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Expected cross-spectral matrix, reduced over the averaged observations.

        Validates that at least two signals are present, then returns the cached
        reduced cross-spectral matrix -- a single batched matmul over the
        averaged time/trials/tapers axes (see ``_reduced_cross_spectral_matrix``)
        rather than a per-observation outer product.

        Returns
        -------
        array, shape (..., n_frequencies, n_signals, n_signals)
            Expected cross-spectral matrix.
        """
        self._validate_multiple_signals()
        return self._cached_reduced_cross_spectral_matrix

    def _reduced_cross_spectral_matrix(
        self, fourier_coefficients: NDArray[np.complexfloating] | None = None
    ) -> NDArray[np.complexfloating]:
        """Expected cross-spectral matrix via a single batched matmul.

        Numerically equivalent (to floating-point tolerance) to
        ``self._expectation(self._cross_spectral_matrix)``, but contracts the
        averaged observation axes (any subset of time/trials/tapers, taken from
        the active ``expectation_type``) directly instead of materializing the
        full ``(..., n_signals, n_signals)`` outer product for every
        observation. For the default ``trials_tapers`` expectation this replaces
        a large intermediate with a small result and is markedly faster.

        Parameters
        ----------
        fourier_coefficients : array, optional
            Coefficients of shape
            ``(n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)``
            to reduce; defaults to this instance's coefficients.
            ``phase_locking_value`` passes unit-normalized coefficients so the
            same batched matmul yields its normalized cross-spectrum.

        Returns
        -------
        array, shape (..., n_frequencies, n_signals, n_signals)
            Expected cross-spectral matrix. The leading axes are whichever of
            time/trials/tapers are *not* averaged, matching the shape produced
            by the equivalent expectation over the full outer product.

        """
        if fourier_coefficients is None:
            fourier_coefficients = self._fourier_coefficients
        average_axes = self._expectation_axes

        signal_axis = fourier_coefficients.ndim - 1
        frequency_axis = signal_axis - 1
        kept_axes = [axis for axis in range(frequency_axis) if axis not in average_axes]
        # Reorder to (kept leading axes..., frequency, averaged axes..., signals)
        # so the averaged axes collapse into a single observation axis adjacent
        # to signals, ready for a batched matmul.
        order = [*kept_axes, frequency_axis, *average_axes, signal_axis]
        observations = xp.transpose(fourier_coefficients, order)

        n_observations = int(
            np.prod([fourier_coefficients.shape[axis] for axis in average_axes])
        )
        n_signals = fourier_coefficients.shape[signal_axis]
        observations = observations.reshape(
            (*observations.shape[: len(kept_axes) + 1], n_observations, n_signals)
        )
        # cross_spectral_matrix[..., i, j] = mean_obs f_i * conj(f_j), matching
        # _complex_inner_product's convention, then averaged over observations.
        cross_spectral_matrix = xp.matmul(
            xp.swapaxes(observations, -1, -2),
            xp.conj(observations),
            dtype=self._dtype,
        )
        return cross_spectral_matrix / n_observations

    @cached_property
    def _cached_reduced_cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Cache the reduced expected cross-spectral matrix.

        This is the reduced result of ``_expectation_cross_spectral_matrix()``.
        It is reused within a measure (``coherency`` divides it by power) and
        across measures that share one ``Connectivity`` instance
        (``coherence_magnitude``, ``coherence_phase``, ``imaginary_coherence``,
        pairwise spectral Granger). Only the reduced ``(..., n_signals,
        n_signals)`` form is cached, never the observation-resolved
        ``_cross_spectral_matrix``. Invalidated with the other cached
        intermediates when the inputs change; consumers treat it as read-only.
        """
        return self._reduced_cross_spectral_matrix()

    def _subset_cross_spectral_matrix(
        self, pairs: list | NDArray[np.integer]
    ) -> NDArray[np.complexfloating]:
        """Compute compact observation-level spectra for channel pairs.

        Parameters
        ----------
        pairs : array_like
            Pairs of channel indices.

        Returns
        -------
        array, shape (..., n_pairs, n_frequencies, 2, 2)
            One compact 2-by-2 cross-spectral matrix per requested pair. The
            trial/taper/time observation axes remain present until the caller
            applies its configured expectation.

        """
        pair_indices = xp.asarray(pairs, dtype=int)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pairs must have shape (n_pairs, 2).")
        if pair_indices.size == 0:
            raise ValueError("pairs must contain at least one signal pair.")
        n_signals = self._fourier_coefficients.shape[-1]
        if bool(xp.any(pair_indices < 0)) or bool(xp.any(pair_indices >= n_signals)):
            raise IndexError(f"pair indices must be between 0 and {n_signals - 1}.")
        # Advanced-index the signal axis into (..., frequency, pair, 2), then put
        # pair before frequency so frequency remains axis -3 as required by the
        # Wilson factorization after the observation expectation is applied.
        coefficients = self._fourier_coefficients[..., pair_indices]
        coefficients = xp.moveaxis(coefficients, -2, -3).astype(self._dtype, copy=False)
        coefficients = coefficients[..., xp.newaxis]
        return _complex_inner_product(coefficients, coefficients, dtype=self._dtype)

    # These quantities feed every directed-connectivity measure and are
    # expensive to compute (the minimum-phase decomposition in particular), so
    # they are cached per instance. The validated input setters clear these
    # cached properties before accepting a replacement.
    @cached_property
    def _minimum_phase_factor(self) -> NDArray[np.complexfloating]:
        return minimum_phase_decomposition(
            self._expectation_cross_spectral_matrix(),
            tolerance=self._minimum_phase_tolerance,
            max_iterations=self._minimum_phase_max_iterations,
        )

    @cached_property
    @_non_negative_frequencies(axis=-3)
    def _transfer_function(self) -> NDArray[np.complexfloating]:
        return _estimate_transfer_function(self._minimum_phase_factor)

    @cached_property
    def _noise_covariance(self) -> NDArray[np.floating]:
        return _estimate_noise_covariance(self._minimum_phase_factor)

    @cached_property
    def _MVAR_Fourier_coefficients(self) -> NDArray[np.complexfloating]:
        return _regularized_inverse(self._transfer_function)

    @property
    def _expectation(self) -> Callable:
        return EXPECTATION[self.expectation_type]

    @property
    def _expectation_axes(self) -> tuple[int, ...]:
        """Observation axes reduced by the configured expectation."""
        return EXPECTATION_AXES[self.expectation_type]

    @property
    def n_observations(self) -> int:
        """Return number of observations.

        Returns
        -------
        int
            Effective number of independent observations after averaging.

        """
        return int(
            np.prod(
                [
                    self._fourier_coefficients.shape[axis]
                    for axis in self._expectation_axes
                ]
            )
        )

    @_asnumpy
    def power(self) -> NDArray[np.floating]:
        """Return the one-sided power spectral density of the signal.

        Only the non-negative frequencies are returned, with the interior
        positive-frequency bins doubled so that integrating the returned
        spectrum over frequency recovers the full signal power (the negative
        frequencies of a real signal carry equal power). The DC bin, and the
        Nyquist bin for an even FFT length, are not doubled.

        Returns
        -------
        NDArray[floating]
            One-sided power spectral density for non-negative frequencies.

        Notes
        -----
        **Range**: [0, ∞). Power spectral density is always non-negative
        with no finite upper bound.

        """
        power = self._power
        n_fft_samples = power.shape[-2]
        one_sided = power[..., : n_fft_samples // 2 + 1, :]

        # Double the interior positive-frequency bins so the one-sided PSD
        # integrates to the same total power as the two-sided spectrum. DC (bin
        # 0) is unique; the Nyquist bin (present only for an even FFT length) is
        # also unique, so neither is doubled. Match the spectrum's dtype so a
        # float32 (complex64) request is not silently upcast to float64.
        scale = xp.full((one_sided.shape[-2],), 2.0, dtype=one_sided.dtype)
        scale[0] = 1.0
        if n_fft_samples % 2 == 0:
            scale[-1] = 1.0
        # scale is 1-D over frequency (axis -2); add a trailing axis to broadcast
        # across signals.
        return one_sided * scale[:, xp.newaxis]

    @_non_negative_frequencies(axis=-3)
    def coherency(self) -> NDArray[np.complexfloating]:
        """Return the complex-valued linear association between time series.

        Computed in the frequency domain.

        Returns
        -------
        complex_coherency : array, shape (..., n_fft_samples, n_signals, n_signals)
            Complex coherency between all signal pairs.

        Notes
        -----
        **Range**: Magnitude |C_{xy}(f)| ∈ [0, 1]; phase ∈ [−π, π].
        Values lie in the unit disk of the complex plane.

        """
        norm = xp.sqrt(
            self._power[..., :, xp.newaxis] * self._power[..., xp.newaxis, :]
        )
        complex_coherencey = _divide_masking_zero_denominator(
            self._expectation_cross_spectral_matrix(),
            norm,
            "Some signals have (near-)zero power, so coherency is undefined "
            "for those pairs and is returned as NaN. This usually indicates "
            "a flat/dead channel or all-zero input.",
        )
        n_signals = self._fourier_coefficients.shape[-1]
        diagonal_ind = xp.arange(0, n_signals)
        complex_coherencey[..., diagonal_ind, diagonal_ind] = xp.nan
        return complex_coherencey

    @_asnumpy
    def coherence_phase(self) -> NDArray[np.floating]:
        """Return the phase angle of the complex coherency.

        Returns
        -------
        phase : array, shape (..., n_fft_samples, n_signals, n_signals)
            Phase angles in radians.

        Notes
        -----
        **Range**: [−π, π]. Phase angles in radians for complex coherency.

        """
        return xp.angle(self.coherency())

    @_asnumpy
    def coherence_magnitude(self) -> NDArray[np.floating]:
        """Return the magnitude squared of the complex coherency.

        Note that the squared modulus of coherency (originally a complex quantity)
        is the magnitude-squared coherence (i.e., the normalized, real component
        of coherency). This value should be bounded by 0 and 1.

        Returns
        -------
        magnitude : array, shape (..., n_fft_samples, n_signals, n_signals)
            Magnitude-squared coherence values.

        Notes
        -----
        **Range**: [0, 1]. Implementation may produce tiny numerical excursions
        beyond bounds due to floating-point precision.

        References
        ----------
        .. [1] Hansson-Sandsten M (2011) Cross-spectrum and coherence function
               estimation using time-delayed Thomson multitapers. In: 2011 IEEE
               International Conference on Acoustics, Speech and Signal
               Processing (ICASSP), pp 4240–4243.

        """
        magnitude = _squared_magnitude(self.coherency())
        return xp.clip(magnitude, 0, 1)

    @_asnumpy
    @_non_negative_frequencies(axis=-3)
    def imaginary_coherence(self) -> NDArray[np.floating]:
        """Return the normalized imaginary component of the cross-spectrum.

        Projects the cross-spectrum onto the imaginary axis to mitigate the
        effect of volume-conducted dependencies. Assumes volume-conducted
        sources arrive at sensors at the same time, resulting in
        a cross-spectrum with phase angle of 0 (perfectly in-phase) or π
        (anti-phase) if the sensors are on opposite sides of a dipole
        source. With the imaginary coherence, in-phase and anti-phase
        associations are set to zero.

        Returns
        -------
        imaginary_coherence_magnitude : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Imaginary coherence magnitudes.

        Notes
        -----
        **Range**: [0, 1]. Magnitude version of imaginary part of coherency.
        Raw imaginary component ranges in [-1, 1].

        References
        ----------
        .. [1] Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., and
               Hallett, M. (2004). Identifying true brain interaction from
               EEG data using the imaginary part of coherency. Clinical
               Neurophysiology 115, 2292-2307.

        """
        denominator = xp.sqrt(
            self._power[..., :, xp.newaxis] * self._power[..., xp.newaxis, :]
        )
        imaginary_coh = xp.abs(
            _divide_masking_zero_denominator(
                self._expectation_cross_spectral_matrix().imag,
                denominator,
                "Some signals have (near-)zero power, so imaginary coherence is "
                "undefined for those pairs and is returned as NaN. This usually "
                "indicates a flat/dead channel or all-zero input.",
            )
        )
        # abs()/clip() leave the NaN-masked zero-power entries as NaN.
        return xp.clip(imaginary_coh, 0, 1)

    def canonical_coherence(
        self, group_labels: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Find the maximal coherence between each combination of groups.

        The canonical coherence finds two sets of weights such that the
        coherence between the linear combination of group1 and the linear
        combination of group2 is maximized.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Links each signal to a group.

        Returns
        -------
        canonical_coherence : array
            Shape (n_time_samples, n_fft_samples, n_groups, n_groups).
            The maximal coherence for each group pair.
        labels : array, shape (n_groups,)
            The sorted unique group labels that correspond to `n_groups`.

        Notes
        -----
        **Range**: [0, 1]. Maximal coherence values are bounded like
        coherence magnitude.

        References
        ----------
        .. [1] Stephen, E.P. (2015). Characterizing dynamically evolving
               functional networks in humans with application to speech.
               Boston University.

        """
        self._validate_multiple_signals()
        labels = np.unique(group_labels)
        n_frequencies = self._fourier_coefficients.shape[-2]
        non_negative_frequencies = xp.arange(0, n_frequencies // 2 + 1)
        fourier_coefficients = self._fourier_coefficients[
            ..., non_negative_frequencies, :
        ]
        normalized_fourier_coefficients = [
            _normalize_fourier_coefficients(
                fourier_coefficients[..., xp.isin(group_labels, label)]
            )
            for label in labels
        ]

        n_groups = len(labels)
        new_shape = (self.time.size, self.frequencies.size, n_groups, n_groups)
        magnitude = _squared_magnitude(
            xp.stack(
                [
                    _estimate_canonical_coherence(
                        fourier_coefficients1, fourier_coefficients2
                    )
                    for fourier_coefficients1, fourier_coefficients2 in combinations(
                        normalized_fourier_coefficients, 2
                    )
                ],
                axis=-1,
            )
        )

        canonical_coherence_magnitude = xp.full(new_shape, xp.nan)
        group_combination_ind = xp.array(list(combinations(xp.arange(n_groups), 2)))
        canonical_coherence_magnitude[
            ..., group_combination_ind[:, 0], group_combination_ind[:, 1]
        ] = magnitude
        canonical_coherence_magnitude[
            ..., group_combination_ind[:, 1], group_combination_ind[:, 0]
        ] = magnitude

        try:
            return xp.asnumpy(canonical_coherence_magnitude), xp.asnumpy(labels)
        except AttributeError:
            return canonical_coherence_magnitude, labels

    def global_coherence(
        self,
        max_rank: int = 1,
        max_workspace_elements: int = GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
    ) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
        """Find linear combinations that capture the most coherent power.

        The linear combinations of signals that capture the most coherent
        power at each frequency and time window.

        This is a frequency domain analog of PCA over signals at a given
        frequency/time window.

        Parameters
        ----------
        max_rank : int, default=1
            The number of components to keep (like the number of PC dimensions).
        max_workspace_elements : int, default=16_000_000
            Approximate working-set target, in array elements, for the batched
            decomposition: frequency bins are processed in chunks sized so the
            main intermediates stay near this many complex elements (the default
            ~16M ≈ 256 MB of complex128). It is a soft target, not a hard memory
            cap — it counts the dominant per-bin intermediates, not the outputs or
            LAPACK's internal workspace, and it never goes below one bin per chunk,
            so actual peak memory is somewhat higher. Lower it to reduce peak
            memory on a constrained CPU or GPU (at the cost of more, smaller
            chunks); the default favors speed and does not change the result.
            Ignored on the per-bin fallback path used for a large decomposition
            dimension.

        Returns
        -------
        global_coherence : ndarray
            Shape (n_time_windows, n_fft_samples, n_components).
            The fraction of total coherent power captured by each component
            (eigenvalue of the cross-spectral matrix divided by the sum of all
            eigenvalues), ordered strongest component first.
        unnormalized_global_coherence : ndarray
            Shape (n_time_windows, n_fft_samples, n_signals, n_components).
            The global coherence vectors (left singular vectors).

        Notes
        -----
        **Range**: [0, 1]. Each value is the fraction of total coherent power
        in that component, so the measure is scale-invariant and the components
        sum to at most 1.

        **Algorithm**: when the number of estimates (``n_trials * n_tapers``)
        is at least ``n_signals`` and ``n_signals`` is small
        (``<= 64``), the components are obtained from an eigendecomposition of
        the ``(n_signals, n_signals)`` cross-spectral matrix ``A @ Aᴴ`` rather
        than a singular value decomposition of ``A``. This is substantially
        faster but squares the condition number, so for a nearly rank-deficient
        cross-spectral matrix (near-duplicate channels) the *weakest* returned
        components (large ``max_rank``) may lose relative precision. The dominant
        component(s) — the usual use of this measure — are unaffected. A thin
        matrix (fewer estimates than signals) uses the economy SVD directly.

        References
        ----------
        .. [1] Cimenser, A., Purdon, P.L., Pierce, E.T., Walsh, J.L.,
               Salazar-Gomez, A.F., Harrell, P.G., Tavares-Stoeckel, C.,
               Habeeb, K., and Brown, E.N. (2011). Tracking brain states under
               general anesthesia by using global coherence analysis.
               Proceedings of the National Academy of Sciences 108, 8832–8837.

        """
        self._validate_multiple_signals()
        (
            n_time_windows,
            n_trials,
            n_tapers,
            n_fft_samples,
            n_signals,
        ) = self._fourier_coefficients.shape

        # A rank-r decomposition of the (n_signals, n_trials * n_tapers)
        # coefficient matrix has at most min(n_signals, n_trials * n_tapers)
        # non-trivial components. Requesting more than that would crash svds or
        # (in the dense branch) broadcast a single component into duplicates, so
        # clamp the requested rank to what is realizable.
        max_available_rank = min(n_signals, n_trials * n_tapers)
        if max_rank > max_available_rank:
            warnings.warn(
                f"max_rank={max_rank} exceeds the number of available "
                f"global-coherence components "
                f"(min(n_signals, n_trials * n_tapers) = {max_available_rank}); "
                f"clamping to {max_available_rank}.",
                UserWarning,
                stacklevel=2,
            )
            max_rank = max_available_rank

        # The batched decomposition works on min(n_signals, n_estimates)-square
        # matrices, so gate on that dimension (not n_signals alone): a thin
        # matrix with few estimates is cheap even for many signals, while a large
        # square matrix is better served by the per-bin svds fallback.
        n_estimates = n_trials * n_tapers
        # Must be a genuine positive integer: it is a floor-divided into a chunk
        # size, so a float (NaN/inf included) or bool would either pick a
        # nonsensical chunk or blow up later inside range(). bool is an int
        # subclass, so reject it explicitly.
        if (
            isinstance(max_workspace_elements, bool)
            or not isinstance(max_workspace_elements, (int, np.integer))
            or max_workspace_elements < 1
        ):
            raise ValueError(
                f"max_workspace_elements must be a positive integer (e.g. "
                f"1_000_000), got {max_workspace_elements!r}. It bounds the memory "
                f"budget, in array elements, for global_coherence's batched "
                f"decomposition; lower it to reduce peak memory."
            )
        if min(n_signals, n_estimates) <= GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS:
            global_coherence, unnormalized_global_coherence = _batched_global_coherence(
                self._fourier_coefficients, max_rank, max_workspace_elements
            )
        else:
            # A user who tuned max_workspace_elements for memory gets no effect
            # here (the per-bin path decomposes one bin at a time); note it so the
            # setting having no effect is discoverable, without warning on the
            # common default-valued call.
            if max_workspace_elements != GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS:
                logger.debug(
                    "global_coherence: max_workspace_elements=%d is ignored on "
                    "the per-bin fallback path used when "
                    "min(n_signals, n_estimates)=%d > %d.",
                    max_workspace_elements,
                    min(n_signals, n_estimates),
                    GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS,
                )
            # Per-bin fallback for a large decomposition dimension, where forming
            # every component is wasteful and svds (used when max_rank is small)
            # finds only the top ones requested.
            # S - singular values
            global_coherence = xp.zeros((n_time_windows, n_fft_samples, max_rank))
            # U - rotation
            unnormalized_global_coherence = xp.zeros(
                (n_time_windows, n_fft_samples, n_signals, max_rank),
                dtype=xp.complex128,
            )

            for time_ind in range(n_time_windows):
                for freq_ind in range(n_fft_samples):
                    # reshape to (n_signals, n_trials * n_tapers)
                    fourier_coefficients = (
                        self._fourier_coefficients[time_ind, :, :, freq_ind, :]
                        .reshape((n_trials * n_tapers, n_signals))
                        .T
                    )

                    (
                        global_coherence[time_ind, freq_ind],
                        unnormalized_global_coherence[time_ind, freq_ind],
                    ) = _estimate_global_coherence(
                        fourier_coefficients, max_rank=max_rank
                    )

        if xp.any(xp.isnan(global_coherence)):
            warnings.warn(
                "Some time-frequency bins have (near-)zero total power, so "
                "global coherence is undefined there and is returned as NaN. "
                "This usually indicates a flat/dead channel or all-zero input.",
                UserWarning,
                stacklevel=2,
            )

        try:
            return xp.asnumpy(global_coherence), xp.asnumpy(
                unnormalized_global_coherence
            )
        except AttributeError:
            return global_coherence, unnormalized_global_coherence

    @_non_negative_frequencies(axis=-3)
    def _phase_locking_value(self) -> NDArray[np.complexfloating]:
        # Normalize each Fourier coefficient to unit magnitude, then reuse the
        # batched reduced cross-spectral matmul: because
        #   (z_i conj(z_j)) / |z_i conj(z_j)| = (z_i / |z_i|) conj(z_j / |z_j|),
        # the mean over observations of the normalized per-observation
        # cross-spectrum equals the reduced cross-spectral matrix of the
        # unit-normalized coefficients -- with no per-observation outer product,
        # so peak memory is O(observations * signals) rather than
        # O(observations * signals**2). Kept on the active namespace (xp); the
        # public ``phase_locking_value`` wrapper converts to NumPy.
        self._validate_multiple_signals()
        # Normalize at the computation dtype (``self._dtype``, complex128 by
        # default): the previous materialized path formed the outer product at
        # that dtype, so normalizing complex64 inputs at their own precision here
        # would let float32 rounding push the unit magnitudes -- and thus the
        # averaged PLV/PPC -- slightly past 1. copy=False avoids a copy when the
        # dtype already matches (the division below allocates a fresh array).
        coefficients = self._fourier_coefficients.astype(self._dtype, copy=False)
        magnitude = xp.abs(coefficients)
        zero_magnitude = magnitude == 0
        if bool(xp.any(zero_magnitude)):
            warnings.warn(
                "Some cross-spectrum entries have zero magnitude (e.g. a "
                "flat/dead channel or all-zero input at a taper/trial), so "
                "the phase-locking normalization z / |z| is undefined there "
                "and is returned as NaN.",
                UserWarning,
                stacklevel=2,
            )
        # z / |z| is undefined where |z| == 0; divide under a scoped errstate and
        # set those coefficients to NaN explicitly (rather than leaking a
        # RuntimeWarning). A NaN coefficient at any observation makes every pair
        # involving it reduce to NaN, matching the previous per-observation path
        # where a zero-magnitude cross-spectrum entry became NaN before averaging.
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = coefficients / magnitude
        normalized[zero_magnitude] = xp.nan
        return self._reduced_cross_spectral_matrix(normalized)

    @_asnumpy
    def phase_locking_value(self) -> NDArray[np.floating]:
        """Return the cross-spectrum with power scaled to magnitude 1.

        The phase locking value attempts to mitigate power differences
        between realizations (tapers or trials) by treating all values of
        the cross-spectrum as the same power. This has the effect of
        downweighting high power realizations and upweighting low power
        realizations.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals, n_signals)
            Phase locking values between all signal pairs.

        Notes
        -----
        **Range**: [0, 1]. 0 indicates random phases; 1 indicates
        constant phase difference.

        References
        ----------
        .. [1] Lachaux, J.-P., Rodriguez, E., Martinerie, J., Varela, F.J.,
               and others (1999). Measuring phase synchrony in brain
               signals. Human Brain Mapping 8, 194-208.

        """
        # Clip to the documented [0, 1] range: |mean of unit-magnitude entries|
        # is <= 1 mathematically, but floating-point rounding can leave it a few
        # ulp above 1 (matching the bounds clipping coherence_magnitude applies).
        return xp.clip(xp.abs(self._phase_locking_value()), 0.0, 1.0)

    def _imaginary_cross_spectrum_moments(
        self, *keys: str
    ) -> tuple[NDArray[np.floating], ...]:
        """Reduced moments of the per-observation imaginary cross-spectrum.

        The phase-lag-index family (``phase_lag_index``,
        ``weighted_phase_lag_index``, ``debiased_squared_phase_lag_index``, and
        ``debiased_squared_weighted_phase_lag_index``) each average a function --
        ``sign``, identity, ``abs`` or square -- of the imaginary part of the
        per-observation cross-spectral matrix, with the diagonal zeroed. This
        returns the requested reduced moments, computing (and caching) any not
        already available from signal-row tiles of the observation-level
        cross-spectrum.

        Computing only the requested keys keeps a single-measure call
        (e.g. ``phase_lag_index`` needs only ``"sign"``) from doing the other
        measures' reductions or retaining their moments. Each tile is reduced
        immediately, avoiding an observation-resolved ``n_signals**2``
        intermediate. The reduced ``n_signals**2`` outputs are unavoidable. The
        cached moments are invalidated with the other cached intermediates and
        are treated as read-only (callers copy before any in-place edit).

        Parameters
        ----------
        *keys : str
            Any of ``"sign"``, ``"imaginary"``, ``"absolute"``, ``"squared"`` for
            ``E[sign(Im)]``, ``E[Im]``, ``E[|Im|]`` and ``E[Im**2]``.

        Returns
        -------
        tuple of arrays, each shape (..., n_frequencies, n_signals, n_signals)
            The requested moments, in the order of ``keys``.
        """
        self._validate_multiple_signals()
        cache = self.__dict__.setdefault("_imaginary_moment_cache", {})
        missing = [key for key in keys if key not in cache]
        if missing:
            valid_keys = {"sign", "imaginary", "absolute", "squared"}
            unknown = set(missing) - valid_keys
            if unknown:
                raise ValueError(f"unknown imaginary moment key(s): {sorted(unknown)}")

            coefficients = self._fourier_coefficients.astype(self._dtype, copy=False)
            n_signals = coefficients.shape[-1]
            kept_observation_axes = [
                axis for axis in range(3) if axis not in self._expectation_axes
            ]
            result_shape = (
                *[coefficients.shape[axis] for axis in kept_observation_axes],
                coefficients.shape[-2],
                n_signals,
                n_signals,
            )
            real_dtype = coefficients.real.dtype
            for key in missing:
                cache[key] = xp.empty(result_shape, dtype=real_dtype)

            observation_frequency_elements = int(np.prod(coefficients.shape[:-1]))
            elements_per_source = max(1, observation_frequency_elements * n_signals)
            signals_per_block = max(
                1,
                min(
                    n_signals,
                    PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS // elements_per_source,
                ),
            )

            # Nonlinear sign/abs/square transforms prevent contracting the
            # observation axes before the outer product. Form only a source-row
            # tile at a time, reduce it immediately, and write the small result.
            all_coefficients = coefficients[..., xp.newaxis]
            for start in range(0, n_signals, signals_per_block):
                stop = min(n_signals, start + signals_per_block)
                source_coefficients = coefficients[..., start:stop, xp.newaxis]
                imaginary = _complex_inner_product(
                    source_coefficients,
                    all_coefficients,
                    dtype=self._dtype,
                ).imag
                local_diagonal = xp.arange(stop - start)
                global_diagonal = xp.arange(start, stop)
                imaginary[..., local_diagonal, global_diagonal] = 0

                for key in missing:
                    if key == "sign":
                        moment = xp.sign(imaginary)
                    elif key == "imaginary":
                        moment = imaginary
                    elif key == "absolute":
                        moment = xp.abs(imaginary)
                    else:  # key == "squared"; unknown keys were rejected above
                        moment = imaginary**2
                    cache[key][..., start:stop, :] = self._expectation(moment)
        return tuple(cache[key] for key in keys)

    @_asnumpy
    @_non_negative_frequencies(axis=-3)
    def phase_lag_index(self) -> NDArray[np.floating]:
        """Return non-parametric synchrony measure mitigating power differences.

        A non-parametric synchrony measure designed to mitigate power
        differences between realizations (tapers, trials) and
        volume-conduction.

        The phase lag index is the average sign of the imaginary
        component of the cross-spectrum. The imaginary component sets
        in-phase or anti-phase signals to zero and the sign scales it to
        have the same magnitude regardless of phase.

        Note that this is the signed version of the phase lag index. In order
        to obtain the unsigned version, as in [1], take the absolute value
        of this quantity.

        Returns
        -------
        phase_lag_index : array, shape (..., n_fft_samples, n_signals, n_signals)
            Phase lag index values for all signal pairs.

        Notes
        -----
        **Range**: [-1, 1] (signed version). For unsigned version (as in [1]),
        take absolute value to get range [0, 1].

        References
        ----------
        .. [1] Stam, C.J., Nolte, G., and Daffertshofer, A. (2007). Phase
               lag index: Assessment of functional connectivity from multi
               channel EEG and MEG with diminished bias from common
               sources. Human Brain Mapping 28, 1178-1193.

        """

        # E[sign(Im)] of the cross-spectrum (real-valued); copy so the returned
        # array is disconnected from the cached moment.
        (mean_sign,) = self._imaginary_cross_spectrum_moments("sign")
        return mean_sign.real.copy()

    @_asnumpy
    @_non_negative_frequencies(-3)
    def weighted_phase_lag_index(self) -> NDArray[np.floating]:
        """Return weighted average of phase lag index using imaginary coherency magnitudes.

        Weighted average of the phase lag index using the imaginary
        coherency magnitudes as weights.

        Note that this is the signed version of the weighted phase lag index
        (mirroring :meth:`phase_lag_index`). In order to obtain the unsigned
        version, as in [1], take the absolute value of this quantity.

        Returns
        -------
        weighted_phase_lag_index : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Weighted phase lag index values.

        Notes
        -----
        **Range**: [-1, 1] (signed version). For the unsigned version (as in
        [1]), take the absolute value to get range [0, 1]. The sign depends on
        the ordering of the signal pair (wpli[i, j] = -wpli[j, i]).

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        """

        mean_imaginary, mean_absolute = self._imaginary_cross_spectrum_moments(
            "imaginary", "absolute"
        )
        # Copy before the in-place zero-weight guard so the cached moment is not
        # mutated.
        weights = mean_absolute.copy()
        weights[weights < xp.finfo(float).eps] = 1
        return mean_imaginary / weights

    @_asnumpy
    def debiased_squared_phase_lag_index(self) -> NDArray[np.floating]:
        """Return square of phase lag index corrected for positive bias.

        The square of the phase lag index corrected for the positive
        bias induced by using the magnitude of the complex cross-spectrum.

        Returns
        -------
        phase_lag_index : array, shape (..., n_fft_samples, n_signals, n_signals)
            Debiased squared phase lag index values.

        Notes
        -----
        **Range**: [-1 / (n_observations - 1), 1]. The unbiased finite-sample
        estimate can be negative when the observed phase consistency is below
        its null bias; negative values do not represent negative coupling.

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        """
        self._validate_debiasing_observations("debiased_squared_phase_lag_index")
        n_observations = self.n_observations
        return (n_observations * self.phase_lag_index() ** 2 - 1.0) / (
            n_observations - 1.0
        )

    @_asnumpy
    @_non_negative_frequencies(-3)
    def debiased_squared_weighted_phase_lag_index(self) -> NDArray[np.floating]:
        """Return square of weighted phase lag index corrected for bias.

        The square of the weighted phase lag index corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        weighted_phase_lag_index : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Debiased squared weighted phase lag index values.

        Notes
        -----
        **Range**: [-1, 1]. The debiased finite-sample estimate can be negative
        when the signed cross-products are dominated by inconsistent phase lags;
        negative values do not represent negative coupling.

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        """
        self._validate_debiasing_observations(
            "debiased_squared_weighted_phase_lag_index"
        )
        n_observations = self.n_observations
        mean_imaginary, mean_squared, mean_absolute = (
            self._imaginary_cross_spectrum_moments("imaginary", "squared", "absolute")
        )
        # Each product is a fresh array, so the cached moments are not mutated.
        imaginary_csm_sum = mean_imaginary * n_observations
        squared_imaginary_csm_sum = mean_squared * n_observations
        imaginary_csm_magnitude_sum = mean_absolute * n_observations
        weights = imaginary_csm_magnitude_sum**2 - squared_imaginary_csm_sum
        weights[weights == 0] = xp.nan

        return (imaginary_csm_sum**2 - squared_imaginary_csm_sum) / weights

    @_asnumpy
    def pairwise_phase_consistency(self) -> NDArray[np.floating]:
        """Return square of phase locking value corrected for bias.

        The square of the phase locking value corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals, n_signals)
            Pairwise phase consistency values.

        Notes
        -----
        **Range**: [-1 / (n_observations - 1), 1]. The unbiased finite-sample
        estimate can be negative when phase consistency is below its null bias;
        negative values do not represent negative coupling.

        References
        ----------
        .. [1] Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., and
               Pennartz, C.M.A. (2010). The pairwise phase consistency: A
               bias-free measure of rhythmic neuronal synchronization.
               NeuroImage 51, 112-122.

        """
        self._validate_debiasing_observations("pairwise_phase_consistency")
        n_observations = self.n_observations
        plv_sum = self._phase_locking_value() * n_observations
        ppc = (plv_sum * plv_sum.conjugate() - n_observations) / (
            n_observations**2 - n_observations
        )
        return ppc.real

    @_asnumpy
    def pairwise_spectral_granger_prediction(self) -> NDArray[np.floating]:
        """Return amount of power at a node explained by other nodes.

        The amount of power at a node in a frequency explained by (is
        predictive of) the power at other nodes.

        Also known as spectral granger causality.

        Returns
        -------
        array
            Spectral Granger prediction values.

        Notes
        -----
        **Range**: [0, ∞). Non-negative values with no finite upper bound.
        Output [i,j] corresponds to causal influence j → i.

        References
        ----------
        .. [1] Geweke, J. (1982). Measurement of Linear Dependence and
               Feedback Between Multiple Time Series. Journal of the
               American Statistical Association 77, 304.

        """
        csm = self._expectation_cross_spectral_matrix()
        n_signals = csm.shape[-1]
        pairs = combinations(range(n_signals), 2)
        total_power = self._power
        return _estimate_spectral_granger_prediction(
            total_power,
            csm,
            pairs,  # type: ignore[arg-type]
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )

    @_asnumpy
    def subset_pairwise_spectral_granger_prediction(
        self, pairs: list | NDArray[np.integer]
    ) -> NDArray[np.floating]:
        """Return predictive power for a subset of signal pairs.

        Parameters
        ----------
        pairs : array_like
            Pairs of signal indices.

        Returns
        -------
        array
            Spectral Granger prediction for specified pairs.

        """
        pairs = np.array(pairs)
        pair_csm = self._expectation(self._subset_cross_spectral_matrix(pairs))
        return _estimate_subset_spectral_granger_prediction(
            self._power,
            pair_csm,
            pairs,
            n_signals=self._fourier_coefficients.shape[-1],
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )

    def conditional_spectral_granger_prediction(self) -> None:
        """Raise NotImplementedError for conditional spectral Granger prediction.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.

        """
        raise NotImplementedError

    def blockwise_spectral_granger_prediction(self) -> None:
        """Raise NotImplementedError for blockwise spectral Granger prediction.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.

        """
        raise NotImplementedError

    @_ignore_nan_propagation_warnings
    @_asnumpy
    def directed_transfer_function(self) -> NDArray[np.floating]:
        """Return transfer function coupling strength normalized by inflow.

        The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        Characterizes the direct and indirect coupling to a node.

        Returns
        -------
        directed_transfer_function : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Directed transfer function values.

        Notes
        -----
        **Range**: [0, 1] (normalized). Represents proportion of inflow
        via transfer function.

        References
        ----------
        .. [1] Kaminski, M., and Blinowska, K.J. (1991). A new method of
               the description of the information flow in the brain
               structures. Biological Cybernetics 65, 203-210.

        """
        return _squared_magnitude(
            self._transfer_function / _total_inflow(self._transfer_function)
        )

    @_ignore_nan_propagation_warnings
    @_asnumpy
    def directed_coherence(self) -> NDArray[np.floating]:
        """Return the squared directed coherence (noise-weighted DTF).

        Like the directed transfer function, but the noise variance weights
        **both** the numerator and the inflow normalization. The returned value
        is the squared directed coherence
        ``nv_j |H_ij|^2 / sum_k nv_k |H_ik|^2``, where ``nv`` is the per-signal
        innovation (noise) variance and ``H`` is the transfer function; it sums
        to 1 over sources ``j`` for each target ``i``.

        Returns
        -------
        directed_coherence : array, shape (..., n_fft_samples, n_signals, n_signals)
            Squared directed coherence values.

        Notes
        -----
        **Range**: [0, 1]. Normalized directional connectivity measure.

        **Assumption**: This measure follows Baccala et al. (1998), which
        assumes the MVAR innovations are uncorrelated (a diagonal noise
        covariance). The denominator then equals the signal's power spectral
        density ``S_ii = sum_k nv_k |H_ik|^2``. When the estimated innovation
        covariance has non-negligible off-diagonal terms (common for
        non-parametrically estimated MVARs), the true PSD
        ``S_ii = (H Cov H^H)_ii`` also contains cross-power between correlated
        sources that this diagonal formula omits, so the values are approximate.
        A ``UserWarning`` is emitted in that case.

        References
        ----------
        .. [1] Baccala, L., Sameshima, K., Ballester, G., Do Valle, A., and
               Timo-Iaria, C. (1998). Studying the interaction between
               brain structures via directed coherence and Granger
               causality. Applied Signal Processing 5, 40.

        """
        # Directed coherence normalizes the noise-weighted inflow over sources
        # (axis -1), so the per-source noise variance must vary along that axis.
        # The squared measure is nv_j |H_ij|^2 / sum_k nv_k |H_ik|^2, which sums
        # to 1 over sources like the directed transfer function. This uses only
        # the diagonal of the noise covariance, which equals the PSD denominator
        # only when the innovations are uncorrelated; warn when the omitted
        # cross-power is a material fraction of the true PSD.
        if (
            _max_psd_discrepancy(self._transfer_function, self._noise_covariance)
            > DIRECTED_COHERENCE_DISCREPANCY_TOLERANCE
        ):
            warnings.warn(
                "directed_coherence assumes uncorrelated MVAR innovations (a "
                "diagonal noise covariance), but the estimated innovation "
                "covariance has non-negligible off-diagonal terms: the diagonal "
                "normalization omits a material fraction of the true power "
                "spectral density (cross-power between correlated sources). "
                "Interpret the values as approximate, or use "
                "partial_directed_coherence instead.",
                UserWarning,
                stacklevel=2,
            )
        noise_variance = _get_noise_variance(self._noise_covariance, axis=-1)
        return (
            noise_variance
            * _squared_magnitude(self._transfer_function)
            / _total_inflow(self._transfer_function, noise_variance) ** 2
        )

    def _partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return device-native PDC for reuse by other device-native measures."""
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients
            / _total_outflow(self._MVAR_Fourier_coefficients)
        )

    @_ignore_nan_propagation_warnings
    @_asnumpy
    def partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return transfer function coupling strength normalized by outflow.

        The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        Returns
        -------
        partial_directed_coherence : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Partial directed coherence values.

        Notes
        -----
        **Range**: [0, 1]. Normalized direct coupling measure.

        References
        ----------
        .. [1] Baccala, L.A., and Sameshima, K. (2001). Partial directed
               coherence: a new concept in neural structure determination.
               Biological Cybernetics 84, 463-474.

        """
        return self._partial_directed_coherence()

    @_ignore_nan_propagation_warnings
    @_asnumpy
    def generalized_partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return generalized partial directed coherence.

        The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        The generalized partial directed coherence scales the relative
        strength of coupling by the noise variance.

        Returns
        -------
        generalized_partial_directed_coherence : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Generalized partial directed coherence values.

        Notes
        -----
        **Range**: [0, 1]. Normalized, scaled by noise variance.

        References
        ----------
        .. [1] Baccala, L.A., Sameshima, K., and Takahashi, D.Y. (2007).
               Generalized partial directed coherence. In Digital Signal
               Processing, 2007 15th International Conference on, (IEEE),
               pp. 163-166.

        """
        noise_variance = _get_noise_variance(self._noise_covariance)
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients
            / xp.sqrt(noise_variance)
            / _total_outflow(self._MVAR_Fourier_coefficients, noise_variance)
        )

    @_ignore_nan_propagation_warnings
    @_asnumpy
    def direct_directed_transfer_function(self) -> NDArray[np.floating]:
        """Return combination of directed transfer function and partial coherence.

        A combination of the directed transfer function estimate of
        directional influence between signals and the partial coherence's
        accounting for the influence of other signals.

        Returns
        -------
        direct_directed_transfer_function : array
            Shape (..., n_fft_samples, n_signals, n_signals).
            Direct directed transfer function values.

        Notes
        -----
        **Range**: [0, 1]. Normalized combination of DTF and partial coherence.

        References
        ----------
        .. [1] Korzeniewska, A., Manczak, M., Kaminski,
               M., Blinowska, K.J., and Kasicki, S. (2003). Determination
               of information flow direction among brain structures by a
               modified directed transfer function (dDTF) method.
               Journal of Neuroscience Methods 125, 195-207.

        """
        full_frequency_DTF = self._transfer_function / _total_inflow(
            self._transfer_function, axis=(-1, -3)
        )
        return xp.abs(full_frequency_DTF) * xp.sqrt(self._partial_directed_coherence())

    def group_delay(
        self,
        frequencies_of_interest: NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
        significance_threshold: float = 0.05,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Return the average time-delay of a broadband signal.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Frequency band of interest.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.
        significance_threshold : float, default=0.05
            P-value threshold for significance.

        Returns
        -------
        delay : array, shape (..., n_signals, n_signals)
            Time delays between signal pairs.
        slope : array, shape (..., n_signals, n_signals)
            Slope of phase vs frequency.
        r_value : array, shape (..., n_signals, n_signals)
            Correlation coefficient of linear fit.

        Notes
        -----
        **Range**: (−∞, ∞). Time delays can be positive or negative.

        References
        ----------
        .. [1] Gotman, J. (1983). Measurement of small time differences
               between EEG channels: method and application to epileptic
               seizure propagation. Electroencephalography and Clinical
               Neurophysiology 56, 501-514.

        """
        frequencies = self.frequencies
        self._require_multiple_frequencies("group_delay")
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution
        )
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest
        )
        # Statistical inference and masked regression below are NumPy operations.
        # Make the GPU-to-host boundary explicit before passing data into them;
        # NumPy deliberately refuses implicit conversion of CuPy arrays.
        bandpassed_coherency = to_numpy(bandpassed_coherency)
        bandpassed_frequencies = to_numpy(bandpassed_frequencies)

        n_signals = bandpassed_coherency.shape[-1]
        signal_combination_ind = np.asarray(list(combinations(np.arange(n_signals), 2)))
        bandpassed_coherency = bandpassed_coherency[
            ..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]
        ]

        is_significant = _find_significant_frequencies(
            bandpassed_coherency,
            self.n_observations,
            independent_frequency_step,
            significance_threshold=significance_threshold,
        )
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-2),
            mask=~is_significant,
        )

        # Vectorized masked linear regression of the unwrapped phase on
        # frequency, per (batch, signal pair), replacing a per-slice scipy
        # ``linregress`` call (``apply_along_axis`` invokes it once per slice).
        # Uses the closed-form ordinary-least-squares slope and Pearson r from
        # centered masked sums over the frequency axis (-2); where at least two
        # distinct significant frequencies remain these match ``linregress`` to
        # floating-point tolerance, and degenerate slices (fewer than two, or a
        # single distinct frequency) yield NaN as ``linregress`` does.
        is_valid = ~np.ma.getmaskarray(coherence_phase)
        # Replace masked entries with a finite value before the arithmetic: the
        # masked phase can be NaN (e.g. a zero-power bin's coherency angle), and
        # ``0 * NaN`` is NaN, which would poison the whole slice's sums even
        # though the valid mask already excludes those entries.
        phase = np.where(is_valid, np.ma.getdata(coherence_phase), 0.0)
        frequency = np.asarray(bandpassed_frequencies, dtype=float).reshape(-1, 1)
        axis = -2
        count = is_valid.sum(axis, keepdims=True)
        safe_count = np.where(count == 0, 1, count)
        # Center x and y on their per-slice means before summing squares, so the
        # variance does not catastrophically cancel when the absolute
        # frequencies are large relative to their spacing (raw-moment
        # ``count * sum_xx - sum_x**2`` loses all precision there).
        mean_x = (is_valid * frequency).sum(axis, keepdims=True) / safe_count
        mean_y = (is_valid * phase).sum(axis, keepdims=True) / safe_count
        centered_x = frequency - mean_x
        centered_y = phase - mean_y
        sum_xx = (is_valid * centered_x * centered_x).sum(axis, keepdims=True)
        sum_yy = (is_valid * centered_y * centered_y).sum(axis, keepdims=True)
        sum_xy = (is_valid * centered_x * centered_y).sum(axis, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            pair_slope = (sum_xy / sum_xx)[..., 0, :]
            pair_r_value = (sum_xy / np.sqrt(sum_xx * sum_yy))[..., 0, :]
        # Guard against |r| drifting just past 1 from rounding.
        pair_r_value = np.clip(pair_r_value, -1.0, 1.0)

        new_shape = (*bandpassed_coherency.shape[:-2], n_signals, n_signals)
        slope = np.full(new_shape, np.nan)
        slope[..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]] = (
            pair_slope
        )
        slope[
            ..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]
        ] = -pair_slope

        delay = slope / (2 * np.pi)

        r_value = np.ones(new_shape)
        r_value[..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]] = (
            pair_r_value
        )
        r_value[..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]] = (
            pair_r_value
        )

        return delay, slope, r_value

    @_asnumpy
    def delay(
        self,
        frequencies_of_interest: NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
        significance_threshold: float = 0.05,
        n_range: int = 3,
    ) -> NDArray[np.floating]:
        """Find a range of possible delays from the coherence phase.

        The delay (and phase) at each frequency is indistinguishable from
        2π phase jumps, but we can look at a range of possible delays
        and see which one is most likely.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Frequency band of interest.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.
        significance_threshold : float, default=0.05
            P-value threshold for significance.
        n_range : int, default=3
            Number of phases to consider.

        Returns
        -------
        possible_delays : array
            Shape (..., n_frequencies, (n_range * 2) + 1, n_signals, n_signals).
            Array of possible time delays in seconds. The true delay is the
            candidate that is consistent (frequency-independent) across the
            band. Frequencies without significant coherence, and the 0 Hz (DC)
            bin, are undefined and returned as NaN.

        """
        frequencies = self.frequencies
        self._require_multiple_frequencies("delay")
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution
        )
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest
        )
        # Delay masking is NumPy-only. Transfer once at this boundary instead of
        # mixing NumPy masks with device arrays throughout the calculation.
        bandpassed_coherency = to_numpy(bandpassed_coherency)
        bandpassed_frequencies = to_numpy(bandpassed_frequencies)
        n_signals = bandpassed_coherency.shape[-1]
        signal_combination_ind = np.asarray(list(combinations(np.arange(n_signals), 2)))
        bandpassed_coherency = bandpassed_coherency[
            ..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]
        ]

        is_significant = _find_significant_frequencies(
            bandpassed_coherency,
            self.n_observations,
            independent_frequency_step,
            significance_threshold=significance_threshold,
        )
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-2), mask=~is_significant
        )
        possible_range = 2 * np.pi * np.arange(-n_range, n_range + 1)
        # Convert phase to a time delay: tau = (phase + 2*pi*k) / (2*pi*f). The
        # 2*pi*k terms resolve the phase-wrapping ambiguity. Dividing only by
        # 2*pi (omitting f) would return cycles, not seconds, making a constant
        # physical delay appear frequency-dependent.
        cycles = np.rollaxis(
            (possible_range + coherence_phase[..., np.newaxis]) / (2 * np.pi), -1, -2
        )
        # cycles has shape (..., n_frequencies, n_candidates, n_pairs); divide by
        # the frequency along the n_frequencies axis (-3). DC (f == 0) has no
        # defined delay and becomes NaN.
        frequency = bandpassed_frequencies[:, np.newaxis, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            delays = cycles / frequency
        delays[..., bandpassed_frequencies == 0, :, :] = np.nan
        # Fill non-significant frequencies (masked) with NaN rather than the
        # masked array's underlying 0.0, so a non-significant bin is not read as
        # a genuine zero-lag delay. This matches the DC handling above.
        delays = np.ma.filled(delays, np.nan)
        new_shape = (
            *bandpassed_coherency.shape[:-1],
            len(possible_range),
            n_signals,
            n_signals,
        )
        possible_delays = np.full(new_shape, np.nan)
        possible_delays[
            ..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]
        ] = delays
        possible_delays[
            ..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]
        ] = -delays

        return possible_delays

    @_asnumpy
    def phase_slope_index(
        self,
        frequencies_of_interest: NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
    ) -> NDArray[np.floating]:
        """Return weighted average of slopes projected onto imaginary axis.

        The phase slope index sums the product of the coherency at adjacent
        frequencies, ``conj(C(f)) * C(f + df)``, over the band and projects the
        result onto the imaginary axis to avoid volume-conduction effects
        (Nolte et al. 2008). The magnitude of the coherency at each frequency
        therefore weights the contribution of that frequency step.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Frequency band of interest.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.

        Returns
        -------
        phase_slope_index : array, shape (..., n_signals, n_signals)
            Phase slope index values.

        Notes
        -----
        **Range**: (−∞, ∞). Signed directional measure with no bounds.

        References
        ----------
        .. [1] Nolte, G., Ziehe, A., Nikulin, V.V., Schlogl, A., Kramer,
               N., Brismar, T., and Muller, K.-R. (2008). Robustly
               Estimating the Flow Direction of Information in Complex
               Physical Systems. Physical Review Letters 100.

        """
        frequencies = self.frequencies
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest
        )

        self._require_multiple_frequencies("phase_slope_index")
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution
        )
        frequency_index = xp.arange(
            0, bandpassed_frequencies.shape[0], independent_frequency_step
        )
        bandpassed_coherency = bandpassed_coherency[..., frequency_index, :, :]

        # The phase slope index needs at least two frequency bins to form an
        # adjacent-frequency product. With fewer, the sum below would be an empty
        # sum that NumPy reports as 0, which is indistinguishable from a genuine
        # "no directionality" result.
        n_band_frequencies = bandpassed_coherency.shape[-3]
        if n_band_frequencies < 2:
            raise ValueError(
                f"phase_slope_index needs at least 2 frequency bins in the band "
                f"after subsampling, but got {n_band_frequencies}. Widen "
                f"frequencies_of_interest, or decrease frequency_resolution so "
                f"more than one independent frequency remains."
            )

        # Nolte et al. (2008): sum conj(C(f)) * C(f + df) over adjacent
        # (independent) frequency bins, then take the imaginary part. The
        # frequency axis is -3 (the two trailing axes are the signal pair).
        adjacent_product = (
            xp.conj(bandpassed_coherency[..., :-1, :, :])
            * bandpassed_coherency[..., 1:, :, :]
        ).sum(axis=-3)
        return adjacent_product.imag


def _estimate_noise_covariance(
    minimum_phase: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Estimate noise covariance non-parametrically from minimum phase factor.

    Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the noise covariance
    of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples, n_signals, n_signals)
        The matrix square root of a cross spectral matrix.

    Returns
    -------
    noise_covariance : array, shape (n_time_windows, n_signals, n_signals)
        The noise covariance of a MVAR model.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    """
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    return _complex_inner_product(
        inverse_fourier_coefficients[..., 0, :, :],
        inverse_fourier_coefficients[..., 0, :, :],
    ).real


def _divide_masking_zero_denominator(
    numerator: NDArray[_NumberT],
    denominator: NDArray[np.floating],
    message: str,
) -> NDArray[_NumberT]:
    """Divide, returning NaN where the denominator is (near-)zero.

    The coherence measures are undefined where a signal has (near-)zero power.
    Rather than dividing by a floored epsilon (which yields spuriously large
    values), this emits a single ``UserWarning`` (``message``) when any
    denominator entry is at or below the dtype's smallest positive value,
    divides against a safe (1.0-substituted) denominator, and sets those entries
    to NaN.

    Parameters
    ----------
    numerator : NDArray
        Numerator array (real or complex).
    denominator : NDArray[floating]
        Non-negative denominator, broadcastable against ``numerator``.
    message : str
        Warning text emitted once if any denominator entry is (near-)zero.

    Returns
    -------
    NDArray
        ``numerator / denominator`` with (near-)zero-denominator entries NaN.
    """
    zero = denominator <= xp.finfo(denominator.dtype).tiny
    if xp.any(zero):
        warnings.warn(message, UserWarning, stacklevel=3)
    safe = xp.where(zero, xp.asarray(1.0, dtype=denominator.dtype), denominator)
    result = numerator / safe
    result[zero] = xp.nan
    return result


def _regularized_inverse(
    matrix: NDArray[_NumberT],
) -> NDArray[_NumberT]:
    """Return the Tikhonov-regularized inverse of a batched matrix.

    Solves ``(M + λI) X = I`` instead of inverting ``M`` directly. The diagonal
    loading ``λ = TIKHONOV_REGULARIZATION_FACTOR * sqrt(mean(|M|^2))`` is scaled
    per batched matrix (over the last two axes), so windows/frequencies with very
    different power are each conditioned appropriately. Using the RMS magnitude
    (amplitude units) rather than the mean square keeps ``λ`` in the same units
    as ``M``: adding ``λI`` is dimensionally consistent and the regularized
    inverse is scale-covariant (rescaling ``M`` by ``c`` rescales ``λ`` by ``c``,
    leaving downstream connectivity measures invariant).

    Parameters
    ----------
    matrix : NDArray, shape (..., n_signals, n_signals)
        Batched (real or complex) matrices to invert.

    Returns
    -------
    NDArray, shape (..., n_signals, n_signals)
        Regularized inverse of each batched matrix.
    """
    lam = TIKHONOV_REGULARIZATION_FACTOR * xp.sqrt(
        xp.mean(xp.real(xp.conj(matrix) * matrix), axis=(-2, -1), keepdims=True)
    )
    identity = xp.eye(matrix.shape[-1], dtype=matrix.dtype)
    # Broadcast identity to the batch dimensions so CuPy's batched solve accepts
    # the RHS shape (NumPy tolerates the mismatch; CuPy does not).
    identity_batched = xp.broadcast_to(identity, matrix.shape)
    return xp.linalg.solve(matrix + lam * identity_batched, identity_batched)


def _estimate_transfer_function(
    minimum_phase: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Estimate transfer function non-parametrically from minimum phase factor.

    Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the transfer
    function of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples, n_signals, n_signals)
        The matrix square root of a cross spectral matrix.

    Returns
    -------
    transfer_function : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_signals).
        The transfer function of a MVAR model.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    """
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    H_0 = inverse_fourier_coefficients[..., 0:1, :, :]
    return xp.matmul(minimum_phase, _regularized_inverse(H_0))


def _estimate_predictive_power(
    total_power: NDArray[np.floating],
    rotated_covariance: NDArray[np.floating],
    transfer_function: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Estimate predictive power from total power and transfer function.

    Parameters
    ----------
    total_power : array_like
        Total power of signals.
    rotated_covariance : array_like
        Rotated noise covariance matrix.
    transfer_function : array_like
        Transfer function matrix.

    Returns
    -------
    array_like
        Predictive power values.

    """
    intrinsic_power = total_power[..., xp.newaxis] - rotated_covariance[
        ..., xp.newaxis, :, :
    ] * _squared_magnitude(transfer_function)
    intrinsic_power[intrinsic_power == 0] = xp.finfo(float).eps
    # A near-singular rotation can drive intrinsic_power negative; log() then
    # yields NaN, which is deliberately masked out below. Scope the warning
    # suppression to this operation rather than silencing it process-wide.
    with np.errstate(invalid="ignore", divide="ignore"):
        predictive_power = xp.log(total_power[..., xp.newaxis]) - xp.log(
            intrinsic_power
        )
    predictive_power[predictive_power <= 0] = xp.nan
    return predictive_power


def _squared_magnitude(x: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """Return squared magnitude of complex array.

    Parameters
    ----------
    x : array_like
        Complex input array.

    Returns
    -------
    array_like
        Squared magnitude values.

    """
    return xp.abs(x) ** 2


def _complex_inner_product(
    a: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    dtype: np.dtype = xp.complex128,
) -> NDArray[np.complexfloating]:
    """Measure orthogonality (similarity) of complex arrays.

    Measures the orthogonality (similarity) of complex arrays in
    the last two dimensions.

    Parameters
    ----------
    a, b : array_like
        Complex input arrays.
    dtype : np.dtype, default=complex128
        Data type for computation.

    Returns
    -------
    array_like
        Complex inner product.

    """
    return xp.matmul(a, _conjugate_transpose(b), dtype=dtype)


def _remove_instantaneous_causality(
    noise_covariance: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Remove instantaneous causality effects from noise covariance.

    Rotates the noise covariance so that the effect of instantaneous
    signals (like those caused by volume conduction) are removed.

    x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    y -> x: var(y) - (cov(x,y) ** 2 / var(x))

    Parameters
    ----------
    noise_covariance : array, shape (..., n_signals, n_signals)
        Input noise covariance matrix.

    Returns
    -------
    rotated_noise_covariance : array, shape (..., n_signals, n_signals)
        The noise covariance without the instantaneous causality effects.

    """
    variance = xp.diagonal(noise_covariance, axis1=-1, axis2=-2)[..., xp.newaxis]
    return variance.swapaxes(-1, -2) - noise_covariance**2 / variance


def _set_diagonal_to_zero(
    x: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Set diagonal of the last two dimensions to zero.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    array_like
        Array with diagonal elements set to zero.

    """
    n_signals = x.shape[-1]
    diagonal_index = xp.diag_indices(n_signals)
    x[..., diagonal_index[0], diagonal_index[1]] = 0
    return x


def _total_inflow(
    transfer_function: NDArray[np.complexfloating],
    noise_variance: float | NDArray[np.floating] = 1.0,
    axis: int | tuple[int, ...] = -1,
) -> NDArray[np.floating]:
    """Measure effect of incoming signals onto a node via sum of squares.

    Parameters
    ----------
    transfer_function : array_like
        Transfer function matrix.
    noise_variance : float or array_like, default=1.0
        Noise variance values.
    axis : int, default=-1
        Axis for summation.

    Returns
    -------
    array_like
        Total inflow values.

    """
    return xp.sqrt(
        xp.sum(
            noise_variance * _squared_magnitude(transfer_function),
            keepdims=True,
            axis=axis,
        )
    )


def _get_noise_variance(
    noise_covariance: NDArray[np.floating],
    axis: int = -2,
) -> NDArray[np.floating]:
    """Extract noise variance and broadcast it along the requested signal axis.

    The transfer function / MVAR coefficients use the convention
    ``[..., n_fft, target, source]`` (axis -2 is the target/row, axis -1 is the
    source/column). Different directed measures weight by the noise variance
    along different axes, so the diagonal of the covariance must be broadcast to
    match: partial-directed-coherence-family measures sum over the target axis
    (-2), while directed coherence sums over the source axis (-1).

    Parameters
    ----------
    noise_covariance : array, shape (..., n_signals, n_signals)
        Noise covariance matrix.
    axis : int, default=-2
        Signal axis the noise variance should vary along, either -2 (target) or
        -1 (source). A leading ``newaxis`` is always inserted for the frequency
        axis.

    Returns
    -------
    noise_variance : array
        Diagonal elements (noise variances) reshaped to broadcast against the
        transfer function along `axis`.

    """
    noise_variance = xp.diagonal(noise_covariance, axis1=-1, axis2=-2)
    if axis == -2:
        return noise_variance[..., xp.newaxis, :, xp.newaxis]
    elif axis == -1:
        return noise_variance[..., xp.newaxis, xp.newaxis, :]
    else:
        raise ValueError(f"axis must be -2 (target) or -1 (source), got {axis}")


def _max_psd_discrepancy(
    transfer_function: NDArray[np.complexfloating],
    noise_covariance: NDArray[np.floating],
) -> float:
    """Return the largest relative gap between the diagonal-noise PSD and the true PSD.

    ``directed_coherence`` normalizes each target ``i`` by the diagonal-only power
    ``D_i = sum_k Cov_kk |H_ik|^2``, which equals the true power spectral density
    ``S_ii = (H Cov H^H)_ii`` only when the innovations are uncorrelated. This
    returns the largest relative gap ``|S_ii - D_i| / S_ii`` over all targets,
    frequencies, and batch elements -- a dimension-aware measure of how much
    cross-power the diagonal formula omits. Unlike a pairwise-correlation
    criterion, it flags many weakly-but-jointly correlated sources whose cross
    terms still omit a large fraction of the true power.

    Parameters
    ----------
    transfer_function : array, shape (..., n_fft_samples, n_signals, n_signals)
        MVAR transfer function ``H`` (axis -2 target, axis -1 source).
    noise_covariance : array, shape (..., n_signals, n_signals)
        Estimated MVAR innovation covariance ``Cov``.

    Returns
    -------
    float
        Maximum relative PSD discrepancy, or 0.0 when it cannot be assessed
        (single signal, or all entries non-finite).

    """
    if transfer_function.shape[-1] < 2:
        return 0.0
    H = transfer_function
    # True PSD S_ii = (H Cov H^H)_ii = sum_l (H Cov)_il conj(H_il). Broadcast the
    # frequency-independent covariance over the frequency axis.
    covariance = noise_covariance[..., xp.newaxis, :, :]
    true_psd = xp.real(xp.sum(xp.matmul(H, covariance) * xp.conj(H), axis=-1))
    # Diagonal-only power D_i = sum_k Cov_kk |H_ik|^2.
    noise_variance = xp.real(xp.diagonal(noise_covariance, axis1=-1, axis2=-2))
    diagonal_psd = xp.sum(
        noise_variance[..., xp.newaxis, xp.newaxis, :] * _squared_magnitude(H),
        axis=-1,
    )
    # The true PSD is non-negative in exact arithmetic (diagonal of a PSD
    # matrix); clamp roundoff-negative values to 0 so a near-zero true power with
    # nonzero diagonal power reads as an infinite (not a large-negative) gap.
    true_psd = xp.maximum(true_psd, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = xp.abs(true_psd - diagonal_psd) / true_psd
    # Keep +inf (true power 0 but diagonal power > 0 -- the diagonal formula is
    # infinitely wrong there, the strongest reason to warn); drop only 0/0 NaN
    # (no power at all, so nothing to normalize against).
    relative = relative[~xp.isnan(relative)]
    if relative.size == 0:
        return 0.0
    return float(xp.max(relative))


def _total_outflow(
    MVAR_Fourier_coefficients: NDArray[np.complexfloating],
    noise_variance: float | NDArray[np.floating] = 1.0,
) -> NDArray[np.floating]:
    """Measure effect of outgoing signals on the node via sum of squares.

    Parameters
    ----------
    MVAR_Fourier_coefficients : array_like
        MVAR Fourier coefficients.
    noise_variance : float or array_like, default=1.0
        Noise variance values.

    Returns
    -------
    array_like
        Total outflow values.

    """
    return xp.sqrt(
        xp.sum(
            _squared_magnitude(MVAR_Fourier_coefficients) / noise_variance,
            keepdims=True,
            axis=-2,
        )
    )


def _reshape(
    fourier_coefficients: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Combine trials and tapers dimensions and move to last axis.

    Combine trials and tapers dimensions and move the combined dimension
    to the last axis position.

    Parameters
    ----------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        Input Fourier coefficients.

    Returns
    -------
    fourier_coefficients : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Reshaped Fourier coefficients.

    """
    (n_time_windows, _, _, n_fft_samples, n_signals) = fourier_coefficients.shape
    new_shape = (n_time_windows, -1, n_fft_samples, n_signals)
    return xp.moveaxis(fourier_coefficients.reshape(new_shape), 1, -1)


def _normalize_fourier_coefficients(
    fourier_coefficients: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Normalize fourier coefficients by power within group.

    Parameters
    ----------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        Input Fourier coefficients.

    Returns
    -------
    normalized_fourier_coefficients : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Normalized Fourier coefficients.

    """
    U, _, V_transpose = xp.linalg.svd(
        _reshape(fourier_coefficients), full_matrices=False
    )
    return xp.matmul(U, V_transpose)


def _estimate_canonical_coherence(
    normalized_fourier_coefficients1: NDArray[np.complexfloating],
    normalized_fourier_coefficients2: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Find maximum complex correlation between groups of signals.

    Find the maximum complex correlation between groups of signals
    at each time and frequency.

    Parameters
    ----------
    normalized_fourier_coefficients1 : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        First group of normalized coefficients.
    normalized_fourier_coefficients2 : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Second group of normalized coefficients.

    Returns
    -------
    canonical_coherence : array, shape (n_time_windows, n_fft_samples)
        Canonical coherence values.

    """
    group_cross_spectrum = _complex_inner_product(
        normalized_fourier_coefficients1, normalized_fourier_coefficients2
    )
    return xp.linalg.svd(group_cross_spectrum, full_matrices=False, compute_uv=False)[
        ..., 0
    ]


def _bandpass(
    data: NDArray[np.complexfloating],
    frequencies: NDArray[np.floating],
    frequencies_of_interest: NDArray[np.floating] | None,
    axis: int = -3,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """Filter data matrix along axis for frequencies of interest.

    Filters the data matrix along an axis given a maximum and minimum
    frequency of interest.

    Parameters
    ----------
    data : array, shape (..., n_fft_samples, ...)
        Input data array.
    frequencies : array, shape (n_fft_samples,)
        Frequency values.
    frequencies_of_interest : array-like, shape (2,)
        Min and max frequencies of interest.
    axis : int, default=-3
        Axis along which to filter.

    Returns
    -------
    filtered_data : array
        Filtered data.
    filtered_frequencies : array
        Corresponding filtered frequencies.

    """
    if frequencies_of_interest is None:
        return data, frequencies
    frequency_index = (frequencies_of_interest[0] < frequencies) & (
        frequencies < frequencies_of_interest[1]
    )
    return (
        xp.take(data, frequency_index.nonzero()[0], axis=axis),
        frequencies[frequency_index],
    )


def _get_independent_frequency_step(
    frequency_difference: float, frequency_resolution: float | None
) -> int:
    """Find number of points for statistically independent frequencies.

    Find the number of points of a frequency axis such that they
    are statistically independent given a frequency resolution.

    Parameters
    ----------
    frequency_difference : float
        The distance between two frequency points.
    frequency_resolution : float | None
        The ability to resolve frequency points. If None, returns 1.

    Returns
    -------
    frequency_step : int
        The number of points required so that two
        frequency points are statistically independent.

    """
    if frequency_resolution is None:
        return 1
    if not np.isfinite(frequency_resolution) or frequency_resolution <= 0:
        raise ValueError(
            f"frequency_resolution must be a finite positive number when "
            f"provided, got {frequency_resolution}."
        )
    return int(xp.ceil(frequency_resolution / frequency_difference))


def _find_largest_significant_group(
    is_significant: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Find the largest cluster of significant values over frequencies.

    If frequency value is significant and its neighbor in the next frequency
    is also a significant value, then they are part of the same cluster.

    If there are two clusters of the same size, the first one encountered
    is the significant cluster. All other significant values are set to
    false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_largest : bool array

    """
    labeled, _ = label(is_significant)
    label_groups, label_counts = np.unique(labeled, return_counts=True)

    if not np.all(label_groups == 0):
        label_counts[0] = 0
        max_group = label_groups[np.argmax(label_counts)]
        return labeled == max_group
    else:
        return np.zeros(is_significant.shape, dtype=bool)


def _get_independent_frequencies(
    is_significant: NDArray[np.bool_], frequency_step: int
) -> NDArray[np.bool_]:
    """Set non-distinguishable points to false based on frequency step.

    Given a `frequency_step` that determines the distance to the next
    significant point, sets non-distinguishable points to false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_independent : bool array

    """
    index = is_significant.nonzero()[0]
    independent_index = index[0 : len(index) : frequency_step]
    return xp.isin(np.arange(0, len(is_significant)), independent_index)


def _find_largest_independent_group(
    is_significant: NDArray[np.bool_], frequency_step: int, min_group_size: int = 3
) -> NDArray[np.bool_]:
    """Find the largest significant cluster and return independent points.

    Find the largest significant cluster of frequency points and
    return the independent frequency points of that cluster.

    .. note::
        Retained as the scalar reference implementation for the vectorized
        ``_select_largest_independent_cluster`` used in production; it is not
        called on the hot path, but pins that vectorization's behavior in
        ``test_largest_independent_group_vectorized_matches_reference``.

    Parameters
    ----------
    is_significant : bool array
    frequency_step : int
        The number of points between each independent frequency step
    min_group_size : int
        The minimum number of points for a group to be considered

    Returns
    -------
    is_significant : bool array

    """
    is_significant = _find_largest_significant_group(is_significant)
    is_significant = _get_independent_frequencies(is_significant, frequency_step)
    if sum(is_significant) < min_group_size:
        is_significant[:] = False
    return is_significant


# Element cap (rows * n_frequencies) for one chunk of the significant-frequency
# selector. The per-slice int32 run-length temporaries dominate its memory, so
# processing the flattened signal-pair slices in chunks keeps peak usage bounded
# regardless of the number of slices.
_SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS = 2_000_000


def _select_largest_independent_cluster(
    block: NDArray[np.bool_], frequency_step: int, min_group_size: int
) -> NDArray[np.bool_]:
    """Largest independent significant cluster per row (frequency on last axis).

    ``block`` has shape ``(n_rows, n_frequencies)``. See
    ``_largest_independent_group_along_frequency`` for the selection rule.
    """
    n_frequencies = block.shape[-1]
    # run_length[r, f] = length of the contiguous True-run ending at f (0 where
    # False): a cumulative count that resets at each False (running count minus
    # its value at the most recent False). int32 suffices (runs <= n_frequencies)
    # and halves the temporaries relative to the default int64.
    cumulative = np.cumsum(block, axis=-1, dtype=np.int32)
    run_length = cumulative - np.maximum.accumulate(
        np.where(block, np.int32(0), cumulative), axis=-1
    )

    # Largest run per row; run_length only reaches this value at a run's end, so
    # the first index attaining it is the end of the first largest cluster
    # (matching the "first cluster on ties" rule).
    max_size = run_length.max(axis=-1, keepdims=True)
    end_index = np.expand_dims(np.argmax(run_length == max_size, axis=-1), -1)
    start_index = end_index - max_size + 1

    frequency_index = np.arange(n_frequencies)
    # The largest cluster is contiguous [start_index, end_index]; max_size == 0
    # means no significant frequency, giving an all-False row.
    in_largest_cluster = (
        (frequency_index >= start_index)
        & (frequency_index <= end_index)
        & (max_size > 0)
    )
    # Independent points are start_index, start_index + frequency_step, ...
    independent = in_largest_cluster & (
        (frequency_index - start_index) % frequency_step == 0
    )
    count = independent.sum(axis=-1, keepdims=True)
    return independent & (count >= min_group_size)


def _largest_independent_group_along_frequency(
    is_significant: NDArray[np.bool_], frequency_step: int, min_group_size: int
) -> NDArray[np.bool_]:
    """Vectorized ``_find_largest_independent_group`` over the frequency axis (-2).

    For every slice along axis -2, keep the largest contiguous cluster of
    significant frequencies (the first cluster on ties), subsample it every
    ``frequency_step`` points, and drop the slice to all-False if fewer than
    ``min_group_size`` independent points remain. Equivalent to applying
    ``_find_largest_independent_group`` per slice, but computed for all slices at
    once instead of via ``np.apply_along_axis`` (one Python call per slice). The
    slices are processed in bounded chunks so peak memory stays independent of
    their number.

    Parameters
    ----------
    is_significant : bool array, shape (..., n_frequencies, n_signal_pairs)
    frequency_step : int
        Spacing (in points) between retained independent frequencies.
    min_group_size : int
        Minimum number of independent points for a cluster to be kept.

    Returns
    -------
    bool array, same shape as ``is_significant``.
    """
    axis = -2
    n_frequencies = is_significant.shape[axis]
    if is_significant.size == 0:
        # An empty frequency band (or any zero-length axis) has no cluster to
        # select; return an all-False array of the same (empty) shape.
        return np.zeros(is_significant.shape, dtype=bool)
    # Move frequency to the last axis and flatten the rest so the signal-pair
    # slices can be processed in chunks. asarray(copy=False) avoids a needless
    # copy of an already-boolean input; the reshape of the moved (non-contiguous)
    # view then makes a single bool copy.
    moved = np.moveaxis(np.asarray(is_significant, dtype=bool), axis, -1)
    flattened = moved.reshape(-1, n_frequencies)

    result = np.empty_like(flattened)
    chunk = max(1, _SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS // max(1, n_frequencies))
    for start in range(0, flattened.shape[0], chunk):
        block = flattened[start : start + chunk]
        result[start : start + chunk] = _select_largest_independent_cluster(
            block, frequency_step, min_group_size
        )
    return np.moveaxis(result.reshape(moved.shape), -1, axis)


def _find_significant_frequencies(
    coherency: NDArray[np.complexfloating],
    n_obs: int,
    frequency_step: int = 1,
    significance_threshold: float = 0.05,
    min_group_size: int = 3,
    multiple_comparisons_method: Literal[
        "Benjamini_Hochberg_procedure", "Bonferroni_correction"
    ] = "Benjamini_Hochberg_procedure",
) -> NDArray[np.bool_]:
    """Determine the largest significant cluster along the frequency axis.

    This function uses the exact zero-coherence null distribution to determine
    the p-values and adjusts for multiple comparisons using the
    `multiple_comparisons_method`. Only independent frequencies are
    returned and there must be at least `min_group_size` frequency
    points for the cluster to be returned. If there are several significant
    groups, then only the largest group is returned.

    Parameters
    ----------
    coherency : array, shape (..., n_frequencies, n_signals, n_signals)
        The complex coherency between signals.
    n_obs : int
        The number of observations used to estimate the coherency.
    frequency_step : int
        The number of points between each independent frequency step
    significance_threshold : float
        The threshold for a p-value to be considered significant.
    min_group_size : int
        The minimum number of independent frequency points for
    multiple_comparisons_method : 'Benjamini_Hochberg_procedure' |
        'Bonferroni_correction'
        Procedure used to correct for multiple comparisons.

    Returns
    -------
    is_significant : bool array, shape (..., n_frequencies,
                                        n_signal_combintaions)

    """
    # Test each frequency's coherence against zero using the exact
    # magnitude-squared-coherence null distribution. The Fisher z-transform is
    # miscalibrated at the zero boundary and over-rejects the null ~3-4x.
    p_values = coherence_significance_pvalue(coherency, n_obs)
    is_significant = adjust_for_multiple_comparisons(
        p_values, alpha=significance_threshold, method=multiple_comparisons_method
    )
    return _largest_independent_group_along_frequency(
        is_significant, frequency_step, min_group_size
    )


def _conjugate_transpose(x: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Conjugate transpose of the last two dimensions of array x."""
    return x.swapaxes(-1, -2).conjugate()


def _global_coherence_components(
    block: NDArray[np.complexfloating], max_rank: int, use_eigh: bool
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Global-coherence fractions and vectors for one chunk of bins.

    Parameters
    ----------
    block : ndarray, shape (n_bins, n_signals, n_estimates)
        Per-bin coefficient matrices for this chunk.
    max_rank : int
        Number of strongest components to return.
    use_eigh : bool
        If True (``n_estimates >= n_signals``), diagonalize the
        ``(n_signals, n_signals)`` cross-spectral matrix with ``eigh``. If False
        (a *thin* matrix), use the economy SVD, which computes only the
        ``n_estimates`` non-trivial components rather than a large rank-deficient
        cross-spectral matrix.

    Returns
    -------
    fractions : ndarray, shape (n_bins, max_rank)
        Fraction of total coherent power per component, strongest first, in
        [0, 1]; NaN for a (near-)zero-power bin.
    vectors : ndarray, shape (n_bins, n_signals, max_rank)
        Global-coherence vectors (left singular vectors).
    """
    # Rescale each bin by its max magnitude first: the coherence fraction is
    # invariant to this, but summing squares of extreme-magnitude coefficients
    # would under/overflow to a false zero/inf (see the per-bin path). A
    # genuinely zero-power bin is flagged and returned as NaN.
    max_magnitude = xp.max(xp.abs(block), axis=(-2, -1), keepdims=True)
    is_zero_power = max_magnitude == 0
    scaled = block / xp.where(is_zero_power, 1, max_magnitude)
    # total_power is the squared Frobenius norm (== sum of squared singular
    # values); using it as the denominator keeps each fraction in [0, 1] exactly.
    total_power = xp.sum(xp.abs(scaled) ** 2, axis=(-2, -1))

    if use_eigh:
        # Eigenvalues of the Hermitian PSD cross-spectral matrix are the squared
        # singular values; eigenvectors are the left singular vectors.
        cross_spectral_matrix = xp.matmul(scaled, _conjugate_transpose(scaled))
        eigenvalues, eigenvectors = xp.linalg.eigh(cross_spectral_matrix)
        # eigh returns ascending order; take the strongest components first.
        component_power = xp.flip(eigenvalues, axis=-1)[..., :max_rank]
        vectors = xp.flip(eigenvectors, axis=-1)[..., :max_rank]
    else:
        # Thin matrix: the economy SVD returns descending singular values and the
        # left singular vectors directly, computing only n_estimates components.
        left_vectors, singular_values, _ = xp.linalg.svd(scaled, full_matrices=False)
        component_power = singular_values[..., :max_rank] ** 2
        vectors = left_vectors[..., :max_rank]

    safe_total = xp.where(total_power == 0, 1, total_power)
    # The cross-spectral matrix is PSD by construction (``scaled @ scaledᴴ``), so
    # any negative eigenvalue is round-off in eigh and is clipped to 0. (The SVD
    # path returns squared singular values, which are non-negative already, so
    # the clip is a no-op there.)
    fractions = xp.clip(component_power, 0.0, None) / safe_total[..., xp.newaxis]

    undefined = is_zero_power[..., 0, 0]
    fractions = xp.where(undefined[:, xp.newaxis], xp.nan, fractions)
    vectors = xp.where(undefined[:, xp.newaxis, xp.newaxis], xp.nan, vectors)
    return fractions, vectors


def _batched_global_coherence(
    fourier_coefficients: NDArray[np.complexfloating],
    max_rank: int,
    max_workspace_elements: int = GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Global coherence for all time-frequency bins, batched over bins.

    Global coherence is defined as the eigenvalues of the per-bin cross-spectral
    matrix, normalized by their total (Cimenser et al. 2011). Diagonalizing all
    bins at once replaces the Python loop over bins and its per-bin device syncs.
    The eigenvalues equal the squared singular values used by the per-bin path,
    so the returned fractions match it to floating-point tolerance. The vectors
    are only defined up to a per-component phase where the components are
    distinct, and up to an arbitrary unitary rotation/permutation within any set
    of repeated (degenerate) components, so they need not match the per-bin path.

    Bins are processed in chunks taken from the original tensor (only each chunk
    is rearranged, never the whole array), and the chunk size is derived from the
    actual per-bin working set so peak memory stays bounded on both CPU and GPU.

    Parameters
    ----------
    fourier_coefficients : ndarray,
        shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    max_rank : int
        Number of strongest components to return.

    Returns
    -------
    global_coherence : ndarray, shape (n_time_windows, n_fft_samples, max_rank)
    vectors : ndarray,
        shape (n_time_windows, n_fft_samples, n_signals, max_rank)
    """
    n_time, n_trials, n_tapers, n_fft, n_signals = fourier_coefficients.shape
    n_estimates = n_trials * n_tapers

    global_coherence = xp.empty((n_time, n_fft, max_rank))
    vectors = xp.empty((n_time, n_fft, n_signals, max_rank), dtype=xp.complex128)

    # Size the frequency chunk from the per-bin peak working set so memory stays
    # bounded regardless of the number of bins. Several coefficient-sized arrays
    # are live at once (the rearranged block, its rescaled copy, the conjugate
    # transpose fed to matmul, and the SVD's U/Vh factors on the thin path), plus
    # the decomposition of a min(n_signals, n_estimates)-square matrix and its
    # vectors; the 4x / 2x factors approximate that simultaneous footprint.
    decomposition_dim = min(n_signals, n_estimates)
    per_bin_elements = 4 * n_signals * n_estimates + 2 * decomposition_dim**2
    chunk = max(1, max_workspace_elements // per_bin_elements)
    use_eigh = n_estimates >= n_signals

    for time_ind in range(n_time):
        # View of this time slice as (n_fft, n_signals, n_trials, n_tapers); only
        # a chunk of frequencies is materialized (reshaped) at a time, so the
        # full-size transpose/copy of the tensor is never formed.
        time_slice = fourier_coefficients[time_ind].transpose(2, 3, 0, 1)
        for freq_start in range(0, n_fft, chunk):
            freq_stop = min(freq_start + chunk, n_fft)
            block = time_slice[freq_start:freq_stop].reshape(
                freq_stop - freq_start, n_signals, n_estimates
            )
            fractions, block_vectors = _global_coherence_components(
                block, max_rank, use_eigh
            )
            global_coherence[time_ind, freq_start:freq_stop] = fractions
            vectors[time_ind, freq_start:freq_stop] = block_vectors
    return global_coherence, vectors


def _estimate_global_coherence(
    fourier_coefficients: NDArray[np.complexfloating], max_rank: int = 1
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Estimate global coherence.

    Parameters
    ----------
    fourier_coefficients : ndarray, shape (n_signals, n_trials * n_tapers)
        The fourier coefficients for a given frequency across all channels
    max_rank : float, optional
        The maximum number of singular values to keep

    Returns
    -------
    global_coherence : ndarray, shape (max_rank,)
        The fraction of total coherent power per component (squared singular
        value divided by the sum of all squared singular values), in [0, 1],
        strongest component first. NaN for a (near-)zero-power bin.
    unnormalized_global_coherence : ndarray, shape (n_signals, max_rank)
        The global coherence vectors (left singular vectors)

    """
    n_signals, n_estimates = fourier_coefficients.shape
    # The coefficient matrix has at most min(n_signals, n_estimates) non-trivial
    # singular values, and svds requires a rank strictly below that minimum.
    n_components = min(n_signals, n_estimates)

    # Global coherence is the fraction of total coherent power in each component:
    # the eigenvalue of the cross-spectral matrix divided by the sum of all
    # eigenvalues (Cimenser et al. 2011). The eigenvalues are the squared
    # singular values of the coefficient matrix (up to the shared 1/n_estimates
    # factor), and their sum equals the squared Frobenius norm, so normalizing
    # by it makes the measure scale-invariant and bounded in [0, 1].
    #
    # Rescale by the maximum coefficient magnitude first. The fraction is
    # invariant to this rescaling, but computing the sum of squares directly on
    # extreme-magnitude coefficients would underflow (e.g. ~1e-200 -> 0, a false
    # zero-power bin) or overflow (~1e200 -> inf) and return NaN.
    max_magnitude = float(xp.max(xp.abs(fourier_coefficients)))
    if max_magnitude == 0:
        # Genuinely zero-power bin (e.g. a dead/flat channel): global coherence
        # is 0/0 and undefined. Return NaN rather than silently substituting 0,
        # mirroring coherency() / imaginary_coherence(); the caller warns once.
        return (
            xp.full(max_rank, xp.nan),
            xp.full((n_signals, max_rank), xp.nan, dtype=xp.complex128),
        )
    scaled_coefficients = fourier_coefficients / max_magnitude
    total_power = float(xp.sum(xp.abs(scaled_coefficients) ** 2))

    if max_rank >= n_components - 1:
        unnormalized_global_coherence, singular_values, _ = xp.linalg.svd(
            scaled_coefficients, full_matrices=False
        )
        global_coherence = singular_values[:max_rank] ** 2 / total_power
        unnormalized_global_coherence = unnormalized_global_coherence[:, :max_rank]
    else:
        unnormalized_global_coherence, singular_values, _ = svds(
            scaled_coefficients, max_rank
        )
        # svds does not guarantee the order of the returned singular values, so
        # sort strongest-first explicitly (rather than assuming ascending) and
        # apply the same ordering to the vectors, matching the dense (svd)
        # branch.
        order = xp.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        unnormalized_global_coherence = unnormalized_global_coherence[:, order]
        global_coherence = singular_values**2 / total_power

    return global_coherence, unnormalized_global_coherence


def _estimate_spectral_granger_prediction(
    total_power: NDArray[np.floating],
    csm: NDArray[np.complexfloating],
    pairs: list | NDArray[np.integer],
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> NDArray[np.floating]:
    """
    Estimate spectral granger causality.

    Parameters
    ----------
    total_power : ndarray, shape (..., n_frequencies, n_signals)
        The total power of the signals.
    csm : ndarray, shape (..., n_frequencies, n_signals, n_signals)
        The cross spectral matrix of the signals.
    pairs : list of tuples
        The pairs of signals to estimate the spectral granger
        causality for.

    Returns
    -------
    predictive_power : ndarray, shape (..., n_frequencies, n_signals, n_signals)
        The spectral granger causality of the signals.
    """
    n_frequencies = total_power.shape[-2]
    non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
    total_power = xp.take(total_power, indices=non_neg_index, axis=-2)

    n_frequencies = csm.shape[-3]
    new_shape = list(csm.shape)
    new_shape[-3] = non_neg_index.size
    predictive_power = xp.full(new_shape, xp.nan)

    for pair_indices in pairs:
        pair_indices = xp.array(pair_indices)[:, xp.newaxis]
        try:
            minimum_phase_factor = minimum_phase_decomposition(
                csm[..., pair_indices, pair_indices.T],
                tolerance=minimum_phase_tolerance,
                max_iterations=minimum_phase_max_iterations,
            )
            transfer_function = _estimate_transfer_function(minimum_phase_factor)[
                ..., non_neg_index, :, :
            ]
            rotated_covariance = _remove_instantaneous_causality(
                _estimate_noise_covariance(minimum_phase_factor)
            )
            predictive_power[..., pair_indices, pair_indices.T] = (
                _estimate_predictive_power(
                    total_power[..., pair_indices[:, 0]],
                    rotated_covariance,
                    transfer_function,
                )
            )
        except np.linalg.LinAlgError:
            predictive_power[..., pair_indices, pair_indices.T] = xp.nan

    n_signals = csm.shape[-1]
    diagonal_ind = xp.diag_indices(n_signals)
    predictive_power[..., diagonal_ind[0], diagonal_ind[1]] = xp.nan

    return predictive_power


def _estimate_subset_spectral_granger_prediction(
    total_power: NDArray[np.floating],
    pair_csm: NDArray[np.complexfloating],
    pairs: NDArray[np.integer],
    n_signals: int,
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> NDArray[np.floating]:
    """Estimate selected pairwise Granger values from compact 2-by-2 spectra.

    ``pair_csm`` has shape ``(..., n_pairs, n_frequencies, 2, 2)``. Keeping the
    pair axis as a batch dimension avoids allocating a full signal-by-signal CSM
    with uninitialized entries merely to consume its requested 2-by-2 slices.
    """
    pair_indices = xp.asarray(pairs, dtype=int)
    n_frequencies = total_power.shape[-2]
    non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
    one_sided_power = xp.take(total_power, indices=non_neg_index, axis=-2)

    # Gather the two powers for every pair, then move pair before frequency to
    # match pair_csm's (..., pair, frequency, 2) batch layout.
    pair_power = one_sided_power[..., pair_indices]
    pair_power = xp.moveaxis(pair_power, -2, -3)

    minimum_phase_factor = minimum_phase_decomposition(
        pair_csm,
        tolerance=minimum_phase_tolerance,
        max_iterations=minimum_phase_max_iterations,
    )
    transfer_function = _estimate_transfer_function(minimum_phase_factor)[
        ..., non_neg_index, :, :
    ]
    rotated_covariance = _remove_instantaneous_causality(
        _estimate_noise_covariance(minimum_phase_factor)
    )
    pair_predictive_power = _estimate_predictive_power(
        pair_power,
        rotated_covariance,
        transfer_function,
    )

    output_shape = (*one_sided_power.shape, n_signals)
    predictive_power = xp.full(output_shape, xp.nan)
    for pair_number, pair in enumerate(pair_indices):
        matrix_indices = pair[:, xp.newaxis]
        predictive_power[..., matrix_indices, matrix_indices.T] = xp.take(
            pair_predictive_power, pair_number, axis=-4
        )
    diagonal_indices = xp.diag_indices(n_signals)
    predictive_power[..., diagonal_indices[0], diagonal_indices[1]] = xp.nan
    return predictive_power
