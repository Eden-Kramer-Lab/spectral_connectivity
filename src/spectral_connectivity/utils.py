"""Utility functions for spectral_connectivity package."""

import contextlib
import difflib
import functools
import inspect
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

GPU_ENV_VAR = "SPECTRAL_CONNECTIVITY_ENABLE_GPU"
_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off", ""})

# NumPy and CuPy intentionally share the same runtime API throughout the
# package. ``numpy.typing.NDArray`` would incorrectly promise that a value is
# always host-backed, so backend-facing interfaces use this explicit alias. The
# type checker analyzes the NumPy code path (CuPy mirrors its API), so there
# the alias is a NumPy array.
if TYPE_CHECKING:
    BackendArray: TypeAlias = NDArray[Any]
else:
    BackendArray: TypeAlias = Any
_ArrayT = TypeVar("_ArrayT")
_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_positive_integer(value: Any, minimum: int = 1) -> bool:
    """Whether ``value`` is an integer (not a bool) of at least ``minimum``.

    ``bool`` is an ``int`` subclass, so it is rejected explicitly; NumPy
    integer scalars are accepted.
    """
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, np.integer))
        and bool(value >= minimum)
    )


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Return ``random_state`` if it is a Generator, else a Generator seeded by it."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _real_scalar(value: Any) -> float | None:
    """``value`` as a Python ``float`` if it is one real number, else ``None``.

    Any real scalar counts, including a NumPy scalar or a 0-d array such as
    ``dataset["fs"].values``; booleans (dtype kind ``"b"``), strings and arrays
    with dimensions do not. Shared by the validators of positive scalar
    settings so they accept and refuse the same values.
    """
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        return None
    return float(array)


def _validate_sampling_frequency(sampling_frequency: Any) -> float:
    """Validate a sampling rate and return it as a Python ``float``.

    Shared by every transform and by the DataArray input path so each gives the
    same guidance rather than a bare one-line message or a raw ``isfinite``
    ``TypeError``. Any real scalar is accepted, including a NumPy scalar or a
    0-d array such as ``dataset["fs"].values``; booleans (dtype kind ``"b"``),
    strings and arrays with dimensions are rejected.

    Parameters
    ----------
    sampling_frequency : Any
        The rate in samples per second.

    Returns
    -------
    float
        The rate, normalized so downstream ``float`` annotations hold.

    Raises
    ------
    TypeError
        If the rate is not a real scalar number.
    ValueError
        If the rate is not finite and positive.
    """
    rate = _real_scalar(sampling_frequency)
    if rate is None:
        if isinstance(sampling_frequency, str):
            how = "Pass it as a number, e.g. sampling_frequency=1000 rather than '1000'."
        else:
            how = "Pass it as a single number, e.g. sampling_frequency=1000."
        msg = (
            "sampling_frequency must be a number (samples per second), got "
            f"{type(sampling_frequency).__name__} {sampling_frequency!r}.\n"
            "\n"
            "It labels the frequency axis and scales power, so it cannot be "
            "inferred from a plain array.\n"
            "\n"
            f"{how}"
        )
        raise TypeError(msg)
    if not np.isfinite(rate) or rate <= 0:
        msg = (
            f"sampling_frequency must be finite and positive, got "
            f"{sampling_frequency!r}.\n"
            "\n"
            "The sampling frequency is the rate at which your data was collected.\n"
            "Common values:\n"
            "  - EEG: 250-1000 Hz\n"
            "  - LFP/ephys: 1000-30000 Hz\n"
            "  - fMRI: 0.5-2 Hz (1/TR)\n"
            "\n"
            "Check your data acquisition settings or metadata."
        )
        raise ValueError(msg)
    return rate


def to_numpy(array: Any) -> NDArray[Any]:
    """Return an array on the host without implicit device conversion.

    CuPy arrays expose ``get()`` and deliberately reject ``np.asarray``. They
    are recognized by also exposing the CUDA array interface, since host
    array-likes such as a pandas Series have an unrelated ``get`` method.
    NumPy arrays and ordinary array-likes pass directly through ``np.asarray``.
    """
    get = getattr(array, "get", None)
    if callable(get) and hasattr(type(array), "__cuda_array_interface__"):
        return np.asarray(get())
    return np.asarray(array)


# Frame filenames share __file__'s (unresolved) form, so compare against that.
_PACKAGE_DIRECTORY = str(Path(__file__).parent) + os.sep
# Standard-library frames the package's own decorators and cached properties
# (functools.cached_property, functools.wraps, contextlib) insert between
# package frames; they are never the user's call.
_TRANSPARENT_FILES = frozenset({functools.__file__, contextlib.__file__})


def stacklevel_outside_package() -> int:
    """``stacklevel`` that attributes a warning to the first caller outside the package.

    Pass it directly as ``warnings.warn(..., stacklevel=stacklevel_outside_package())``
    in the function that warns. Unlike a hand-counted level, it stays correct
    however many package frames (measure, helpers, kernels, and the standard
    library machinery of cached properties and decorators) sit between the
    user's call and the warning.
    """
    frame: FrameType | None = sys._getframe(1)  # the function calling warnings.warn
    level = 1
    while frame is not None and (
        frame.f_code.co_filename.startswith(_PACKAGE_DIRECTORY)
        or frame.f_code.co_filename in _TRANSPARENT_FILES
    ):
        frame = frame.f_back
        level += 1
    return level


def close_names(name: str, valid: Iterable[str], limit: int = 3) -> list[str]:
    """The names in ``valid`` that ``name`` most likely misspells.

    ``difflib``'s close matches, then valid names that start with ``name``
    (so a truncated ``n_fft`` finds ``n_fft_samples``), without repeats.
    """
    candidates = list(valid)
    matches = difflib.get_close_matches(name, candidates, n=limit)
    if len(name) >= 3:
        matches += [c for c in candidates if c.startswith(name) and c not in matches]
    return matches[:limit]


def did_you_mean(names: Iterable[str]) -> str:
    """``" Did you mean 'a', 'b'?"`` for ``names`` (repeats dropped), or ``""``."""
    unique = list(dict.fromkeys(names))
    return f" Did you mean {', '.join(map(repr, unique))}?" if unique else ""


# Names other packages give the sampling rate, which no string metric relates
# to ``sampling_frequency``.
SAMPLING_FREQUENCY_ALIASES = frozenset(
    {"fs", "Fs", "sfreq", "srate", "sampling_rate", "sample_rate"}
)


def sampling_frequency_hint(sampling_frequency_given: bool) -> str:
    """What to say after rejecting a sampling-rate alias such as ``fs``: its
    name, or, when ``sampling_frequency`` was passed as well, to drop the alias."""
    if sampling_frequency_given:
        return " sampling_frequency was given too; pass only sampling_frequency (in Hz)."
    return " Did you mean sampling_frequency (in Hz)?"


def reject_sampling_frequency_aliases(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Make a function, or a transform's ``__init__``, that takes
    ``sampling_frequency`` answer ``fs=...`` with the parameter's name.

    Python's own ``TypeError`` for an unexpected keyword names no alternative;
    this one adds :func:`sampling_frequency_hint`, and names the class for an
    ``__init__``. Other calls pass through, and the signature is unchanged for
    ``inspect``.
    """
    parameters = list(inspect.signature(function).parameters)
    is_method = parameters[0] == "self"
    # sampling_frequency was given if passed by name or reached by position.
    position = parameters.index("sampling_frequency")

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        alias = next((name for name in kwargs if name in SAMPLING_FREQUENCY_ALIASES), None)
        if alias is not None:
            given = "sampling_frequency" in kwargs or len(args) > position
            name = type(args[0]).__name__ if is_method else function.__name__
            msg = (
                f"{name}() got an unexpected keyword "
                f"argument {alias!r}.{sampling_frequency_hint(given)}"
            )
            raise TypeError(msg)
        return function(*args, **kwargs)

    return wrapper


# How to even out a ragged nested-list time series (see numeric_array).
TIME_SERIES_RAGGED_FIX = (
    "Give every signal (and trial) the same number of samples, trimming or "
    "padding them, then pass the result."
)


def numeric_array(
    value: Any,
    name: str,
    layout: str,
    ragged_fix: str,
    dtype_example: str = "float",
) -> BackendArray:
    """Convert array-like input (a nested list, say) to an array of numbers.

    Converting first lets the caller's shape checks read the shape a list
    describes. A ragged list, or values that are not numbers, raise naming
    ``name`` and the ``layout`` it must have rather than with NumPy's message;
    ``ragged_fix`` says how to even out the rows. An array (NumPy, CuPy, or an
    ``xarray.DataArray``) is returned as it is, so the caller makes the one
    device copy and can still read a DataArray's dimension names. An
    array-like whose ``dtype`` NumPy cannot read is converted like a list, and
    an object array holding numbers to an array of those numbers.
    """
    data: BackendArray = value
    try:
        np.dtype(value.dtype)
    except (AttributeError, TypeError):
        try:
            data = np.asarray(value)
        except ValueError as error:
            msg = (
                f"{name} must be a numeric array of shape {layout}, but NumPy "
                "cannot make it one array: its nested lists have rows of "
                f"different lengths. {ragged_fix}"
            )
            raise ValueError(msg) from error
    if data.dtype.kind == "O":
        # An object array of numbers is read as the numbers it holds, with the
        # dtype NumPy infers for them (float, or complex for coefficients).
        with contextlib.suppress(ValueError):
            converted = np.asarray(np.asarray(data).tolist())
            if converted.dtype.kind in "biufc":
                data = converted
    if data.dtype.kind not in "biufc":
        received = (
            f"{type(value).__name__} {value!r}"
            if data.ndim == 0
            else f"values of dtype {data.dtype}"
        )
        msg = (
            f"{name} must be a numeric array of shape {layout}, got {received}, "
            f"not numbers. Convert them first, e.g. np.asarray({name}, "
            f"dtype={dtype_example})."
        )
        raise TypeError(msg)
    return data


def warn_deprecated(old: str, new: str, outcome: str = "gives the same values") -> None:
    """Warn that ``old`` is deprecated, at the first caller outside the package.

    The template of the deprecated measure names, ``from_multitaper``,
    ``power_confidence_intervals(n_tapers=, ci=)``, and the wrappers' plain-keyword
    settings: ``"<old> is deprecated and will be removed in 4.0; use <new>,
    which <outcome>."``. The ``frequencies_of_interest`` warning, which gives an
    exact replacement interval, is composed separately. ``old`` and
    ``new`` are written as the caller would write them (``"method='x'"``,
    ``"Connectivity.from_transform()"``).
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in 4.0; use {new}, which {outcome}.",
        DeprecationWarning,
        stacklevel=stacklevel_outside_package(),
    )


def both_spellings_message(function: str, new: str, old: str) -> str:
    """The ``ValueError`` message for a call given a setting under both names."""
    return f"{function} was given both {new} and its deprecated name {old}; pass only {new}."


_issued_migration_messages: set[str] = set()
# The filters in effect at the call. Under Python 3.14's context-aware warnings
# (the default on free-threaded builds) those a catch_warnings block sets, as
# pytest.warns does, are not in warnings.filters; 3.14's _get_filters has them.
_current_warning_filters: Callable[[], Sequence[Any]] = getattr(
    warnings, "_get_filters", lambda: warnings.filters
)


def warn_migration(message: str, category: type[UserWarning]) -> None:
    """Issue a migration notice once per session, unless a warnings filter applies.

    Each message (the ``SidednessWarning`` of ``power`` and of
    ``global_coherence``, one ``DirectedOrientationWarning`` per measure) is
    issued until the user has seen it once. Later it is issued again only if
    a warnings filter in effect could match it, so a filter the user set
    (``-W``, ``PYTHONWARNINGS``, ``warnings.simplefilter("always")`` or
    ``"error"``, ``pytest.warns``) still decides; Python's own default
    filters match no ``UserWarning``.
    Module and line restrictions are not checked, so a filter limited to a
    module counts as matching.

    A call whose first matching filter is ``"ignore"`` does not count as
    seen, so a notice silenced and later un-silenced still appears once. A
    call turned into an error does count: the exception carried the message.

    Python's ``"once"`` action would not do: it forgets what it showed
    whenever any code changes the filters, as ``warnings.catch_warnings``
    does, and a filter the package registered at import is dropped when the
    import runs inside ``catch_warnings`` (as pytest collects test modules).
    """
    action = next(
        (
            filter_action
            for filter_action, pattern, filter_category, _, _ in _current_warning_filters()
            if issubclass(category, filter_category)
            and (pattern is None or pattern.match(message))
        ),
        None,
    )
    # Unlocked check-then-add: under threads, a notice can at worst show twice.
    if message in _issued_migration_messages and action is None:
        return
    if action != "ignore":
        _issued_migration_messages.add(message)
    warnings.warn(message, category, stacklevel=stacklevel_outside_package())


def mark_readonly_if_supported(array: _ArrayT) -> _ArrayT:
    """Try to mark ``array`` read-only in place and return it.

    Setting ``writeable = False`` makes an accidental in-place edit raise loudly
    rather than silently corrupt shared/cached state. Array backends without a
    settable writeable flag (e.g. CuPy) are left as-is; callers rely on an
    already-detached copy for ownership there, not on the flag.
    """
    backend_array: Any = array
    with contextlib.suppress(AttributeError, ValueError):
        backend_array.flags.writeable = False
    return array


def mark_readonly_chain_if_supported(array: _ArrayT) -> _ArrayT:
    """Try to mark an array and every NumPy ``.base`` view read-only.

    Freezing only the outer view does not protect its owning base buffer. Array
    backends without a writable flag or a ``.base`` chain are left unchanged.
    """
    current: Any = array
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        mark_readonly_if_supported(current)
        current = getattr(current, "base", None)
    return array


# How to install the GPU dependencies: the package's gpu extra pins the CuPy
# build for CUDA 12 (cupy-cuda12x), where `pip install cupy` builds from source.
GPU_INSTALL_HINT = (
    'pip install "spectral-connectivity[gpu]" (or with conda: '
    "conda install -c conda-forge cupy)"
)


def gpu_request_error_message() -> str:
    """Error text for a GPU request (see ``GPU_ENV_VAR``) without CuPy installed."""
    return (
        f"GPU support was explicitly requested via "
        f"{GPU_ENV_VAR}={os.environ.get(GPU_ENV_VAR, '')!r}, but CuPy is not installed. "
        f"Install it with: {GPU_INSTALL_HINT}. Or unset {GPU_ENV_VAR} to use the CPU."
    )


def cupy_device_name(cp: Any) -> str:
    """Name of CuPy's current device, or its compute capability if unnamed.

    Raises whatever ``cp.cuda.Device()`` raises when no device is usable.
    """
    device = cp.cuda.Device()
    try:
        name: str = cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode()
        return name.strip("\x00")
    except Exception:
        major, minor = device.compute_capability
        return f"GPU (Compute Capability {major}.{minor})"


def is_gpu_enabled() -> bool:
    """Return whether GPU acceleration was requested via the environment.

    The ``SPECTRAL_CONNECTIVITY_ENABLE_GPU`` environment variable is parsed
    case-insensitively: ``"true"``, ``"1"``, ``"yes"``, and ``"on"`` all enable
    GPU. If the variable is set to an unrecognized value the package falls back
    to CPU and emits a ``UserWarning`` (previously such values were silently
    ignored, e.g. ``"True"`` would run on CPU with no indication).

    Returns
    -------
    bool
        True if GPU acceleration was requested.
    """
    raw = os.environ.get(GPU_ENV_VAR)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    warnings.warn(
        f"{GPU_ENV_VAR} is set to {raw!r}, which is not a recognized value; "
        f"falling back to CPU. Set it to one of {sorted(_TRUE_VALUES)} to enable "
        f"GPU acceleration.",
        UserWarning,
        stacklevel=2,
    )
    return False


def get_compute_backend() -> dict[str, Any]:
    """Get information about the current compute backend (CPU or GPU).

    Returns a dictionary with detailed information about whether GPU
    acceleration is enabled and available, which backend is being used,
    and device-specific information.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:
        - backend : str
            Either "cpu" or "gpu" indicating which backend is currently active
        - gpu_enabled : bool
            True if GPU was requested via SPECTRAL_CONNECTIVITY_ENABLE_GPU='true'
        - gpu_available : bool
            True if CuPy is installed and can be imported
        - device_name : str
            Name of the compute device (e.g., "CPU", "NVIDIA Tesla V100-SXM2-16GB")
        - message : str
            Human-readable message explaining the current configuration

        Example return value when GPU is active:
            {
                'backend': 'gpu',
                'gpu_enabled': True,
                'gpu_available': True,
                'device_name': 'NVIDIA Tesla V100-SXM2-16GB',
                'message': 'Using GPU backend with CuPy on NVIDIA Tesla V100-SXM2-16GB.'
            }

    Examples
    --------
    Inspect the active compute backend (the exact ``message`` and
    ``device_name`` depend on the machine, so only the structure is shown here):

    >>> import spectral_connectivity
    >>> backend = spectral_connectivity.get_compute_backend()
    >>> backend["backend"] in ("cpu", "gpu")
    True
    >>> sorted(backend)
    ['backend', 'device_name', 'gpu_available', 'gpu_enabled', 'message']
    >>> isinstance(backend["message"], str)
    True

    In a script, branch on the backend before a heavy computation::

        backend = spectral_connectivity.get_compute_backend()
        if backend["backend"] == "gpu":
            print(f"GPU acceleration enabled on {backend['device_name']}")
        else:
            print("Running on CPU - consider enabling GPU for large datasets")

    Notes
    -----
    This function does not change the backend configuration; it only reports
    the current state. The backend is determined when the package modules are
    first imported, based on the SPECTRAL_CONNECTIVITY_ENABLE_GPU environment
    variable.

    To enable GPU acceleration, set the environment variable before importing:

    - In shell: ``export SPECTRAL_CONNECTIVITY_ENABLE_GPU=true``
    - In Python: ``os.environ['SPECTRAL_CONNECTIVITY_ENABLE_GPU'] = 'true'``
    - In notebook: ``%env SPECTRAL_CONNECTIVITY_ENABLE_GPU=true``

    See Also
    --------
    Multitaper : Uses the configured compute backend for FFT operations
    Connectivity : Uses the configured compute backend for connectivity calculations
    """
    # Check if GPU was requested via environment variable
    gpu_enabled = is_gpu_enabled()

    # Check if CuPy is available (either already imported or can be imported)
    cupy_available = False
    device_name = "CPU"

    # First check if cupy is already imported
    if "cupy" in sys.modules:
        cupy_available = True
        try:
            import cupy as cp

            try:
                device_name = cupy_device_name(cp)
            except Exception:
                device_name = "GPU"
        except Exception:
            cupy_available = False

    # If cupy not already imported, check if it can be imported
    # (but don't actually import it to avoid side effects)
    if not cupy_available:
        try:
            # Use __import__ to check availability without fully importing
            import importlib.util

            spec = importlib.util.find_spec("cupy")
            cupy_available = spec is not None
        except (ImportError, ValueError, AttributeError):
            cupy_available = False

    # The backend actually imported, which the environment variable may no
    # longer match. Imported here because _backend imports this module.
    from spectral_connectivity._backend import ON_GPU

    backend = "gpu" if ON_GPU else "cpu"

    # Generate helpful message
    if backend == "gpu":
        message = f"Using GPU backend with CuPy on {device_name}."
    elif gpu_enabled and not cupy_available:
        message = (
            "GPU acceleration was requested (SPECTRAL_CONNECTIVITY_ENABLE_GPU='true') "
            f"but CuPy is not installed. Install it with: {GPU_INSTALL_HINT}, then "
            "restart the Python interpreter (the variable is read when "
            "spectral_connectivity is imported). "
            "Currently using CPU backend with NumPy."
        )
    elif gpu_enabled and cupy_available:
        # Requested and installed, yet the CPU backend was imported: the
        # variable was set after spectral_connectivity was imported.
        message = (
            "Using CPU backend with NumPy. GPU acceleration was requested "
            "(SPECTRAL_CONNECTIVITY_ENABLE_GPU='true') and CuPy is installed, but "
            "the variable was set after spectral_connectivity was imported, and "
            "the backend is chosen at import. Restart the Python interpreter (in "
            "a notebook, restart the kernel) and set the variable before "
            "`import spectral_connectivity`."
        )
    elif not gpu_enabled and cupy_available:
        message = (
            "Using CPU backend with NumPy. CuPy is installed and GPU acceleration "
            "is available. To enable GPU, set environment variable: "
            "SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before importing spectral_connectivity. "
            "See documentation for details."
        )
    else:  # CPU mode, cupy not available
        message = (
            "Using CPU backend with NumPy. To enable GPU acceleration:\n"
            f"  1. Install CuPy: {GPU_INSTALL_HINT}\n"
            "  2. Set environment variable SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before importing\n"
            "See documentation for detailed setup instructions."
        )

    return {
        "backend": backend,
        "gpu_enabled": gpu_enabled,
        "gpu_available": cupy_available,
        "device_name": device_name,
        "message": message,
    }


def _validated_flag(name: str, value: Any) -> bool:
    """Return ``value`` as a bool, or raise unless it is a boolean.

    Accepts Python and NumPy bools and 0-d boolean arrays (e.g. a flag computed
    as ``xp.all(...)``). ``bool()`` would read a method or the string
    ``"False"`` as True and ``None`` or 0 as False, silently flipping the
    setting (for the spectrum flags, how the spectrum is treated).
    """
    is_boolean_scalar = getattr(value, "ndim", None) == 0 and (
        getattr(getattr(value, "dtype", None), "kind", None) == "b"
    )
    if not (isinstance(value, bool) or is_boolean_scalar):
        msg = (
            f"{name} must be a bool, got {type(value).__name__} ({value!r}); it is "
            "not guessed from a truth value, so pass True or False (for a "
            "transform's attribute, a bool attribute or property, not a method)."
        )
        raise TypeError(msg)
    return bool(value)


# The VAR model x(t) = sum_k A_k x(t - k) + e(t) of simulate_MVAR and
# VARSpectrum, with coefficients[k - 1] = A_k.

#: Appended to the shape error for 2-D coefficients, a single lag's matrix.
VAR_SINGLE_LAG_HINT = "; for a single lag, pass coefficients[np.newaxis]"


def reject_complex_var(array: Any, name: str, reason: str) -> None:
    """Raise ``TypeError`` if ``array`` (a VAR's ``name``) is complex; ``reason``
    says why the caller takes real values only."""
    if np.iscomplexobj(array):
        msg = f"{name} must be real, got dtype {array.dtype}. {reason}"
        raise TypeError(msg)


def var_companion_matrices(coefficients: NDArray[np.floating]) -> NDArray[np.float64]:
    """Companion matrices ``[[A_1 ... A_p], [I 0]]`` of VAR coefficients.

    ``coefficients`` is ``(..., n_lags, n_signals, n_signals)``; the result is
    ``(..., n_lags * n_signals, n_lags * n_signals)``, the transition matrix
    of the state ``[x(t - 1), ..., x(t - n_lags)]``.
    """
    *batch, n_lags, n_signals, _ = coefficients.shape
    dimension = n_lags * n_signals
    companion = np.zeros((*batch, dimension, dimension))
    if n_lags == 0:
        return companion
    companion[..., n_signals:, :-n_signals] = np.eye(dimension - n_signals)
    companion[..., :n_signals, :] = np.concatenate(
        [coefficients[..., lag, :, :] for lag in range(n_lags)], axis=-1
    )
    return companion


def var_spectral_radius(companion: NDArray[np.floating]) -> NDArray[np.float64]:
    """Largest eigenvalue modulus of each companion matrix, ``(...)``; 0 for no lags.

    The VAR is stable when it is below 1 (see :func:`var_is_unstable`).
    """
    return np.abs(np.linalg.eigvals(companion)).max(axis=-1, initial=0.0)


def var_is_unstable(spectral_radius: Any) -> NDArray[np.bool_]:
    """Whether a spectral radius is above 1, or within a relative 1e-12 of it
    (a unit root): the process then has no stationary state."""
    radius = np.asarray(spectral_radius)
    return (radius > 1) | np.isclose(radius, 1, rtol=1e-12, atol=0)


def var_stabilizing_advice(spectral_radius: float) -> str:
    """How to scale unstable coefficients to a spectral radius of 0.9."""
    # Scaling A_k by c**k scales every companion eigenvalue by c.
    return (
        "Scale the coefficients until the radius is below 1: multiplying each "
        "lag's matrix A_k (coefficients[k - 1]) by c**k multiplies the radius by "
        "c, so coefficients * c ** np.arange(1, n_lags + 1)[:, np.newaxis, "
        f"np.newaxis] with c = {0.9 / spectral_radius:.6g} gives a radius of 0.9."
    )


# Provenance is copied onto every result variable, so an array in it (an
# input attribute, a VAR model) larger than this is recorded by shape and
# dtype instead of by value.
MAX_PROVENANCE_ARRAY_SIZE = 100


def summarized_if_large(value: Any) -> Any:
    """Replace an array with more than ``MAX_PROVENANCE_ARRAY_SIZE`` elements
    by ``{"summarized_array": {"shape": ..., "dtype": ...}}``."""
    if isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value)
        if array.size > MAX_PROVENANCE_ARRAY_SIZE:
            return {
                "summarized_array": {"shape": list(array.shape), "dtype": str(array.dtype)}
            }
    return value
