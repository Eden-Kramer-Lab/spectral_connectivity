"""Exercise ``Connectivity`` under an emulated device (CuPy-like) array namespace.

CuPy is not installed in CI, so the host/device boundary is checked with a
proxy-array namespace built on NumPy that reproduces the CuPy behaviours that
break naive NumPy code:

* ``np.asarray(device_array)`` raises ``TypeError`` (CuPy refuses implicit
  device->host conversion); ``.get()`` returns a NumPy copy.
* Namespace functions and array operators reject bare ``numpy.ndarray``
  operands ("Unsupported type"), except the explicit converters
  (``asarray``/``array``/...). Python scalars, NumPy scalars, and NumPy index
  arrays (which CuPy converts itself) are accepted.
* ``isin`` follows CuPy's implementation (``element.ravel()`` then a device
  kernel), so a list or NumPy operand fails as it does on CuPy.
* ufuncs reject the ``where=`` keyword; ``flags.writeable`` is not settable
  and ``base`` is ``None``.

Under real NumPy every case below is a no-op, which is why the emulation is
needed to keep these regressions from returning.
"""

import operator
import types

import numpy as np
import pytest
import scipy.fft

from spectral_connectivity import Connectivity, minimum_phase_decomposition, transforms
from spectral_connectivity import connectivity as connectivity_module

_CONVERSION_MESSAGE = (
    "Implicit conversion to a NumPy array is not allowed. "
    "Please use `.get()` to construct a NumPy array explicitly."
)
_CONVERTERS = frozenset(
    {"asarray", "array", "asanyarray", "ascontiguousarray", "asfortranarray"}
)
# ndarray methods whose array-like arguments CuPy converts from host itself.
_HOST_TOLERANT_METHODS = frozenset({"take", "compress", "put"})


def _unwrap(obj, strict):
    if isinstance(obj, _DeviceArray):
        return obj._array
    if isinstance(obj, np.ndarray):
        if strict:
            msg = f"Unsupported type {type(obj)}"
            raise TypeError(msg)
        return obj
    if isinstance(obj, tuple):
        return tuple(_unwrap(item, strict) for item in obj)
    if isinstance(obj, list):
        return [_unwrap(item, strict) for item in obj]
    if isinstance(obj, dict):
        return {key: _unwrap(value, strict) for key, value in obj.items()}
    return obj


def _wrap(obj):
    if isinstance(obj, _DeviceArray):
        return obj
    if isinstance(obj, (np.ndarray, np.generic)):
        return _DeviceArray(obj)
    if isinstance(obj, tuple):
        return tuple(_wrap(item) for item in obj)
    if isinstance(obj, list):
        return [_wrap(item) for item in obj]
    return obj


class _Flags:
    """CuPy's flags expose only contiguity/owndata; nothing is settable."""

    def __setattr__(self, name, value):
        msg = f"attribute '{name}' of 'cupy._core.flags.Flags' objects is not writable"
        raise AttributeError(msg)


def _binary(op, reflected=False):
    def method(self, other):
        other_value = _unwrap(other, strict=True)
        if reflected:
            return _wrap(op(other_value, self._array))
        return _wrap(op(self._array, other_value))

    return method


class _DeviceArray:
    """Stand-in for ``cupy.ndarray`` wrapping a NumPy array."""

    __slots__ = ("_array",)
    __array_ufunc__ = None  # NumPy operators defer to our reflected operators
    __hash__ = None

    def __init__(self, array):
        object.__setattr__(self, "_array", np.asarray(array))

    def __array__(self, *args, **kwargs):
        raise TypeError(_CONVERSION_MESSAGE)

    def get(self):
        return self._array.copy()

    @property
    def __cuda_array_interface__(self):
        # Like cupy.ndarray; the backend identifies device arrays by it.
        return {"shape": self._array.shape, "version": 3}

    @property
    def flags(self):
        return _Flags()

    @property
    def base(self):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        attribute = getattr(self._array, name)
        if callable(attribute):
            strict = name not in _HOST_TOLERANT_METHODS

            def method(*args, **kwargs):
                return _wrap(attribute(*_unwrap(args, strict), **_unwrap(kwargs, strict)))

            return method
        return _wrap(attribute)

    def __getitem__(self, key):
        return _wrap(self._array[_unwrap(key, strict=False)])

    def __setitem__(self, key, value):
        self._array[_unwrap(key, strict=False)] = _unwrap(value, strict=False)

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        for item in self._array:
            yield _wrap(item)

    def __bool__(self):
        return bool(self._array)

    def __float__(self):
        return float(self._array)

    def __int__(self):
        return int(self._array)

    def __complex__(self):
        return complex(self._array)

    def __repr__(self):
        return f"_DeviceArray({self._array!r})"

    __add__ = _binary(operator.add)
    __radd__ = _binary(operator.add, reflected=True)
    __sub__ = _binary(operator.sub)
    __rsub__ = _binary(operator.sub, reflected=True)
    __mul__ = _binary(operator.mul)
    __rmul__ = _binary(operator.mul, reflected=True)
    __truediv__ = _binary(operator.truediv)
    __rtruediv__ = _binary(operator.truediv, reflected=True)
    __pow__ = _binary(operator.pow)
    __rpow__ = _binary(operator.pow, reflected=True)
    __matmul__ = _binary(operator.matmul)
    __rmatmul__ = _binary(operator.matmul, reflected=True)
    __and__ = _binary(operator.and_)
    __rand__ = _binary(operator.and_, reflected=True)
    __or__ = _binary(operator.or_)
    __ror__ = _binary(operator.or_, reflected=True)
    __lt__ = _binary(operator.lt)
    __le__ = _binary(operator.le)
    __gt__ = _binary(operator.gt)
    __ge__ = _binary(operator.ge)
    __eq__ = _binary(operator.eq)
    __ne__ = _binary(operator.ne)
    __neg__ = lambda self: _wrap(-self._array)  # noqa: E731
    __abs__ = lambda self: _wrap(abs(self._array))  # noqa: E731
    __invert__ = lambda self: _wrap(~self._array)  # noqa: E731


def _wrap_function(function, name):
    strict = name not in _CONVERTERS
    is_ufunc = isinstance(function, np.ufunc)

    def wrapped(*args, **kwargs):
        if is_ufunc and "where" in kwargs:
            msg = f"{name}() got an unexpected keyword argument 'where'"
            raise TypeError(msg)
        return _wrap(function(*_unwrap(args, strict), **_unwrap(kwargs, strict)))

    return wrapped


def _device_isin(element, test_elements, assume_unique=False, invert=False):
    # cupy/_logic/truth.py: ``element.ravel()`` (AttributeError for a list),
    # then an ElementwiseKernel that rejects non-cupy operands.
    flat_element = element.ravel()
    flat_test = test_elements.ravel()
    for operand in (flat_element, flat_test):
        if not isinstance(operand, _DeviceArray):
            msg = f"Unsupported type {type(operand)}"
            raise TypeError(msg)
    return _wrap(np.isin(flat_element._array, flat_test._array, invert=invert)).reshape(
        element.shape
    )


def _device_take(a, indices, axis=None, out=None):
    # cupy.take dispatches to ``a.take``; a host ``a`` reaches NumPy, which
    # cannot convert device indices.
    return a.take(indices, axis, out)


class _DeviceNamespace:
    """``cupy``-like namespace view over a NumPy module."""

    ndarray = _DeviceArray

    def __init__(self, module, name):
        self._module = module
        self.__name__ = name

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "isin":
            return _device_isin
        if name == "take":
            return _device_take
        attribute = getattr(self._module, name)
        if isinstance(attribute, types.ModuleType):
            return _DeviceNamespace(attribute, f"{self.__name__}.{name}")
        if isinstance(attribute, type):
            return attribute  # dtypes, finfo, LinAlgError
        if callable(attribute):
            return _wrap_function(attribute, name)
        return attribute  # pi, nan, newaxis


@pytest.fixture
def xp(monkeypatch):
    """Swap every package module's ``xp`` for the device emulation."""
    namespace = _DeviceNamespace(np, "cupy")
    for module in (connectivity_module, transforms, minimum_phase_decomposition):
        if module.xp is not np:
            pytest.skip("the emulation replaces the NumPy backend only")
        monkeypatch.setattr(module, "xp", namespace)
    return namespace


def _coefficients(rng, shape=(1, 4, 3, 16, 3)):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_emulation_reproduces_the_cupy_restrictions(xp):
    device = xp.asarray(np.ones(3))
    assert isinstance(device, xp.ndarray)
    with pytest.raises(TypeError, match="Implicit conversion"):
        np.asarray(device)
    with pytest.raises(TypeError, match="Unsupported type"):
        xp.isfinite(np.ones(3))
    with pytest.raises(TypeError, match="Unsupported type"):
        device + np.ones(3)
    with pytest.raises(AttributeError):
        xp.isin([0, 1], device)
    with pytest.raises(TypeError, match="Unsupported type"):
        xp.isin(np.arange(3), device)
    with pytest.raises(TypeError, match="where"):
        xp.divide(device, device, where=device > 0)
    assert isinstance(device.get(), np.ndarray)
    assert isinstance(xp.sum(device), xp.ndarray)  # reductions stay on device


def test_host_coefficients_are_moved_to_the_device(xp):
    """The first device operation on the coefficients must not see a bare
    NumPy array: ``Connectivity(host_array)`` has to convert it first."""
    connectivity = Connectivity(_coefficients(np.random.default_rng(0)))
    assert isinstance(connectivity._fourier_coefficients, xp.ndarray)
    power = connectivity.power()
    coherence = connectivity.coherence_magnitude()
    assert isinstance(power, np.ndarray)
    assert np.isfinite(power).all()
    assert isinstance(coherence, np.ndarray)
    assert np.isfinite(coherence[..., 0, 1]).all()


def test_host_frequencies_are_moved_to_the_device(xp):
    """Host ``frequencies`` with device coefficients: the stored coordinate must
    live on the device, or ``frequencies`` indexes a NumPy array with a device
    index."""
    coefficients = _coefficients(np.random.default_rng(1))
    host_frequencies = np.fft.fftfreq(16, d=1 / 100.0)
    connectivity = Connectivity(xp.asarray(coefficients), frequencies=host_frequencies)
    np.testing.assert_array_equal(connectivity.frequencies, np.abs(host_frequencies[:9]))
    np.testing.assert_array_equal(connectivity.all_frequencies, host_frequencies)


@pytest.mark.parametrize("labels", [["a", "a", "b"], np.array([0, 0, 1])])
def test_canonical_coherence_accepts_host_group_labels(xp, labels):
    """Group membership is resolved on the host; the device coefficients must
    be indexed with device index arrays, not ``xp.isin`` over host labels."""
    connectivity = Connectivity(xp.asarray(_coefficients(np.random.default_rng(2))))
    values, unique_labels = connectivity.canonical_coherence(labels)
    assert isinstance(values, np.ndarray)
    assert np.isfinite(values[..., 0, 1]).all()
    np.testing.assert_array_equal(unique_labels, np.unique(labels))


def test_reassigned_coefficients_keep_the_time_coordinate_on_the_host(xp):
    """``time`` is a host coordinate at construction; the geometry reset on
    reassignment must keep it on the host, so ``np.asarray(conn.time)`` works."""
    rng = np.random.default_rng(3)
    connectivity = Connectivity(xp.asarray(_coefficients(rng)))
    with pytest.warns(UserWarning, match="changed the FFT/time geometry"):
        connectivity.fourier_coefficients = xp.asarray(_coefficients(rng, (2, 4, 3, 8, 3)))
    np.testing.assert_array_equal(np.asarray(connectivity.time), [0, 1])
    assert connectivity.frequencies.shape == (5,)


def test_wilson_factorization_of_real_signals_runs_on_the_device(xp, monkeypatch):
    """Directed measures on real-valued signals stay on the device.

    Their cross-spectra are conjugate-symmetric, so the Wilson factorization
    iterates on the non-negative frequencies with real FFTs. CuPy supplies those
    transforms from ``cupyx.scipy.fft``; route the module's SciPy imports
    through the emulation so a host array reaching them fails here.
    """
    rng = np.random.default_rng(4)
    coefficients = scipy.fft.fft(rng.standard_normal((1, 6, 3, 32, 3)), axis=-2)
    cross_spectrum = Connectivity(coefficients)._expectation_cross_spectral_matrix()
    assert minimum_phase_decomposition._is_conjugate_symmetric(cross_spectrum)

    for name in ("fft", "ifft", "rfft", "irfft"):
        monkeypatch.setattr(
            minimum_phase_decomposition, name, _wrap_function(getattr(scipy.fft, name), name)
        )

    device_granger = Connectivity(coefficients).pairwise_spectral_granger_prediction()
    monkeypatch.undo()
    host_granger = Connectivity(coefficients).pairwise_spectral_granger_prediction()

    assert isinstance(device_granger, np.ndarray)
    np.testing.assert_array_equal(device_granger, host_granger)
