"""State, cache invalidation, ownership, and serialization contracts."""

import copy
import pickle
import warnings
from functools import cached_property

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity import Multitaper
from spectral_connectivity.connectivity import Connectivity


def test_transfer_function_is_cached():
    """Expensive directed-connectivity intermediates are cached per instance."""
    rng = np.random.default_rng(1)
    fourier = rng.standard_normal((1, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (1, 3, 2, 8, 2)
    )
    c = Connectivity(fourier_coefficients=fourier)
    assert c._minimum_phase_factor is c._minimum_phase_factor
    assert c._transfer_function is c._transfer_function
    assert c._noise_covariance is c._noise_covariance


def test_changing_inputs_clears_cached_intermediates():
    """Reassigning fourier_coefficients / expectation_type must not serve stale cache."""
    rng = np.random.default_rng(2)
    fourier = rng.standard_normal((1, 4, 2, 8, 2)) + 1j * rng.standard_normal(
        (1, 4, 2, 8, 2)
    )
    c = Connectivity(fourier_coefficients=fourier, expectation_type="trials_tapers")
    transfer_default = c._transfer_function

    # "tapers" retains the trials dimension, so the recomputed transfer function
    # has a different shape -- proving the cache was invalidated, not reused.
    c.expectation_type = "tapers"
    transfer_tapers = c._transfer_function
    assert transfer_tapers.shape != transfer_default.shape

    # Reassigning fourier_coefficients also clears the cache.
    minimum_phase = c._minimum_phase_factor
    c.fourier_coefficients = fourier
    assert c._minimum_phase_factor is not minimum_phase


def test_power_and_cross_spectrum_caches_invalidate():
    """Cached _power / reduced cross-spectrum must not serve stale results.

    Both are cached per instance for reuse across measures; reassigning the
    inputs must drop them so a later access recomputes from the new data.
    """
    rng = np.random.default_rng(5)
    shape = (2, 4, 3, 8, 2)
    fourier = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    c = Connectivity(fourier_coefficients=fourier, expectation_type="trials_tapers")

    # Populate the caches: coherency reads power + reduced CSM; a phase-lag-index
    # measure populates the shared imaginary-cross-spectrum moments.
    c.coherency()
    c.weighted_phase_lag_index()
    assert "_power" in c.__dict__
    assert "_cached_reduced_cross_spectral_matrix" in c.__dict__
    assert "_imaginary_moment_cache" in c.__dict__
    power_before = c._power
    csm_before = c._cached_reduced_cross_spectral_matrix
    wpli_before = c.weighted_phase_lag_index()

    # Reassigning to *different* data must recompute, not serve the stale cache.
    other = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    c.fourier_coefficients = other
    assert "_power" not in c.__dict__
    assert "_cached_reduced_cross_spectral_matrix" not in c.__dict__
    assert "_imaginary_moment_cache" not in c.__dict__
    assert not np.allclose(c._power, power_before)
    assert not np.allclose(c._cached_reduced_cross_spectral_matrix, csm_before)
    # The fused phase-lag-index family recomputes from the new data and matches a
    # fresh instance (a stale moment cache would fail this).
    fresh_wpli = Connectivity(
        fourier_coefficients=other, expectation_type="trials_tapers"
    ).weighted_phase_lag_index()
    assert not np.allclose(c.weighted_phase_lag_index(), wpli_before)
    np.testing.assert_array_equal(c.weighted_phase_lag_index(), fresh_wpli)

    # Changing expectation_type also invalidates (and changes the averaged shape).
    power_trials_tapers = c._power
    c.expectation_type = "tapers"  # retains the trials axis
    assert "_power" not in c.__dict__
    assert c._power.shape != power_trials_tapers.shape


def test_subclass_cached_property_is_invalidated_automatically():
    """New dependent caches need no entry in a parallel name registry."""

    class ExtendedConnectivity(Connectivity):
        @cached_property
        def custom_cached_shape(self):
            return self._power.shape

    coefficients = np.ones((2, 3, 2, 8, 2), dtype=complex)
    connectivity = ExtendedConnectivity(coefficients)
    default_shape = connectivity.custom_cached_shape

    connectivity.expectation_type = "tapers"

    assert "custom_cached_shape" not in connectivity.__dict__
    assert connectivity.custom_cached_shape != default_shape


def test_fourier_coefficients_are_an_immutable_snapshot():
    """In-place edits must not silently bypass cache invalidation.

    The cached power / cross-spectrum assume the coefficients change only via
    assignment (which clears the caches). The constructor therefore stores a
    private snapshot, and the accessor never hands out a writable alias of it:
    mutating the caller's original array, or the array the property returns, must
    not serve stale scientific results.
    """
    rng = np.random.default_rng(9)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    c = Connectivity(fourier_coefficients=fourier)
    power_before = c.power().copy()
    coherence_before = c.coherence_magnitude().copy()

    # Mutating the caller's original array must not reach the warmed instance.
    fourier[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    np.testing.assert_array_equal(c.coherence_magnitude(), coherence_before)

    # The getter returns an independent, read-only copy: an in-place edit raises
    # loudly rather than silently vanishing against a discarded object.
    returned = c.fourier_coefficients
    assert returned.base is None  # an owning copy, not a view of the snapshot
    assert returned.flags.writeable is False
    with pytest.raises(ValueError):
        returned[...] = 0.0
    # Even re-enabling writeability (permitted on an owning copy) and mutating it
    # cannot reach the instance: the copy shares no buffer with the snapshot.
    returned.flags.writeable = True
    returned[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    np.testing.assert_array_equal(c.coherence_magnitude(), coherence_before)


def test_fourier_coefficients_getter_returns_fresh_independent_copy():
    """Each getter call returns a distinct copy disconnected from the snapshot."""
    rng = np.random.default_rng(11)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )

    c = Connectivity(fourier_coefficients=fourier)
    power_before = c.power().copy()

    first = c.fourier_coefficients
    second = c.fourier_coefficients
    # Distinct objects, neither aliasing the backing snapshot.
    assert first is not second
    assert first is not c._fourier_coefficients
    assert first.base is None and second.base is None
    np.testing.assert_array_equal(first, c._fourier_coefficients)

    # Re-enable and mutate one copy; the other copy, the snapshot, and the cache
    # are all untouched (independent buffers).
    first.flags.writeable = True
    first[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)
    assert not np.array_equal(first, second)


def test_direct_construction_copies_and_is_isolated_from_caller_mutation():
    """Connectivity(fc) must defensively copy: mutating fc cannot reach it."""
    rng = np.random.default_rng(21)
    shape = (1, 4, 2, 8, 3)
    fourier = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex128
    )
    c = Connectivity(fourier_coefficients=fourier)
    # The stored snapshot owns its data (a copy), not a view of the caller's array.
    assert c._fourier_coefficients.base is None
    power_before = c.power().copy()
    fourier[...] = 0.0
    np.testing.assert_array_equal(c.power(), power_before)


def test_from_multitaper_adopts_without_copying():
    """from_multitaper takes ownership of fft() output without a full copy.

    Multitaper.fft() returns a swapaxes view that is referenced nowhere else, so
    Connectivity freezes it in place rather than copying the largest array in the
    pipeline. The stored array is that view (``base`` is not None), and the entire
    base chain is frozen so the backing buffer cannot be mutated.
    """
    rng = np.random.default_rng(22)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)

    stored = c._fourier_coefficients
    # Adopted (a view of fft() output), not an owning copy.
    assert stored.base is not None

    # The whole base chain is read-only, so the writable backing buffer of the
    # swapaxes view is unreachable for mutation.
    obj = stored
    while obj is not None:
        assert obj.flags.writeable is False
        obj = obj.base

    # Freezing only the outer view would leave the deepest base writable; confirm
    # the deepest base cannot be mutated.
    base = stored
    while base.base is not None:
        base = base.base
    with pytest.raises(ValueError):
        base[(0,) * base.ndim] = 0.0


@mark.parametrize(
    "make",
    [
        lambda m: Connectivity.from_multitaper(m),  # adoption path (frozen view)
        lambda m: Connectivity(np.asarray(m.fft())),  # defensive-copy path
    ],
    ids=["adopt", "copy"],
)
def test_getter_copy_defeats_base_reenable_attack(make):
    """The getter must not expose any writable path to the internal snapshot.

    A read-only view would still let a caller reach the owning base through
    ``.base``, re-enable its ``writeable`` flag (NumPy allows this on an array
    that owns its data), then re-enable and mutate the view -- corrupting the
    snapshot behind the warmed caches. The getter therefore returns an
    independent copy; walking to the root base and re-enabling it must not reach
    ``c._fourier_coefficients``.
    """
    rng = np.random.default_rng(23)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = make(m)
    internal_before = np.asarray(c._fourier_coefficients).copy()
    power_before = c.power().copy()

    returned = c.fourier_coefficients
    assert returned.base is None  # an owning copy: no reachable snapshot base

    # Even performing the full attack on the returned copy's own root base leaves
    # the instance untouched, because the copy shares no buffer with the snapshot.
    root = returned
    while root.base is not None:
        root = root.base
    root.flags.writeable = True
    returned.flags.writeable = True
    returned[...] = 0.0

    np.testing.assert_array_equal(c._fourier_coefficients, internal_before)
    np.testing.assert_array_equal(c.power(), power_before)


def test_from_multitaper_adoption_matches_copy_numerically():
    """Adoption must not change results vs. the defensive-copy path."""
    rng = np.random.default_rng(24)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    adopted = Connectivity.from_multitaper(m)
    copied = Connectivity(np.asarray(m.fft()))  # forces the copy path

    np.testing.assert_array_equal(adopted.power(), copied.power())
    for measure in ("coherence_magnitude", "imaginary_coherence"):
        a = getattr(adopted, measure)()
        b = getattr(copied, measure)()
        # equal including matching NaN positions (the coherence diagonal)
        assert np.array_equal(a, b, equal_nan=True)


def test_from_multitaper_adoption_invalidates_cache_on_reassignment():
    """Reassigning fourier_coefficients on an adopted instance clears caches."""
    rng = np.random.default_rng(25)
    m = Multitaper(
        rng.standard_normal((400, 6, 3)),
        sampling_frequency=400,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)
    _ = c.power()
    assert "_power" in c.__dict__  # cached
    c.fourier_coefficients = np.asarray(m.fft()) * 2.0
    assert "_power" not in c.__dict__  # cleared
    # New power reflects the reassigned (4x) data.
    np.testing.assert_allclose(c.power(), 4.0 * Connectivity.from_multitaper(m).power())


def test_reassigning_different_geometry_resets_coordinates():
    """A geometry change on reassignment must reset (not silently keep) coords.

    Reassigning 8 -> 10 FFT bins previously left 5 frequency coordinates for 6
    power bins, so phase_slope_index silently dropped a bin and group_delay
    could fail. The coordinates are now reset to the new geometry with a warning.
    """
    rng = np.random.default_rng(0)
    fc8 = rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 2)
    )
    conn = Connectivity(fourier_coefficients=fc8)
    assert conn.frequencies.size == 8 // 2 + 1
    assert conn.time.size == 2

    fc10 = rng.standard_normal((3, 3, 2, 10, 2)) + 1j * rng.standard_normal(
        (3, 3, 2, 10, 2)
    )
    with pytest.warns(UserWarning, match="changed the FFT/time geometry"):
        conn.fourier_coefficients = fc10
    assert conn.frequencies.size == 10 // 2 + 1
    assert conn.time.size == 3
    # Coordinate-dependent methods are now consistent (no silent bin drop).
    assert conn.group_delay()[0].shape[-2:] == (2, 2)


def test_reassigning_same_geometry_keeps_coordinates_without_warning():
    """Same-geometry reassignment (the reuse pattern) must not warn or reset."""
    rng = np.random.default_rng(0)
    fc = rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 2)
    )
    conn = Connectivity(fourier_coefficients=fc)
    freqs_before = conn.frequencies.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no geometry warning
        conn.fourier_coefficients = fc
    np.testing.assert_array_equal(conn.frequencies, freqs_before)


def test_reassigning_one_sided_geometry_keeps_nonnegative_frequencies():
    rng = np.random.default_rng(26)
    coefficients = rng.standard_normal((2, 3, 1, 5, 2)) + 1j * rng.standard_normal(
        (2, 3, 1, 5, 2)
    )
    conn = Connectivity(coefficients, is_one_sided=True)

    replacement = rng.standard_normal((2, 3, 1, 7, 2)) + 1j * rng.standard_normal(
        (2, 3, 1, 7, 2)
    )
    with pytest.warns(UserWarning, match="changed the FFT/time geometry"):
        conn.fourier_coefficients = replacement

    np.testing.assert_array_equal(conn.frequencies, np.linspace(0.0, 0.5, 7))
    assert conn.is_one_sided


def test_reassigning_incompatible_geometry_clears_observation_weights():
    rng = np.random.default_rng(27)
    coefficients = rng.standard_normal((2, 3, 1, 5, 2)) + 1j * rng.standard_normal(
        (2, 3, 1, 5, 2)
    )
    weights = np.ones((*coefficients.shape[:-1], 1))
    conn = Connectivity(
        coefficients,
        is_one_sided=True,
        observation_weights=weights,
    )

    replacement = rng.standard_normal((2, 3, 1, 7, 2)) + 1j * rng.standard_normal(
        (2, 3, 1, 7, 2)
    )
    with pytest.warns(UserWarning, match="observation weights"):
        conn.fourier_coefficients = replacement

    assert conn.observation_weights is None
    assert conn.power().shape == (2, 7, 2)


def test_connectivity_rejects_mismatched_coordinate_lengths():
    """Supplied frequencies/time must match the data geometry."""
    fc = np.zeros((3, 2, 1, 8, 2), dtype=complex)  # n_time=3, n_fft=8
    with pytest.raises(ValueError, match="frequencies must have length"):
        Connectivity(fourier_coefficients=fc, frequencies=np.arange(4))
    with pytest.raises(ValueError, match="time must have length"):
        Connectivity(fourier_coefficients=fc, time=np.arange(5))


def test_connectivity_rejects_non_1d_or_nonfinite_coordinates():
    """Coordinates must be exactly 1-D and finite, not merely the right length."""
    fc = np.zeros((3, 2, 1, 8, 2), dtype=complex)  # n_time=3, n_fft=8
    # (n, 1) frequency array has the right len but wrong shape.
    with pytest.raises(ValueError, match=r"frequencies must be a 1-D array"):
        Connectivity(fourier_coefficients=fc, frequencies=np.zeros((8, 1)))
    with pytest.raises(ValueError, match=r"time must be a 1-D array"):
        Connectivity(fourier_coefficients=fc, time=np.zeros((3, 1)))
    # Non-finite coordinate.
    bad_freqs = np.linspace(0, 1, 8)
    bad_freqs[2] = np.nan
    with pytest.raises(ValueError, match="frequencies must contain only finite"):
        Connectivity(fourier_coefficients=fc, frequencies=bad_freqs)


def test_from_multitaper_connectivity_is_picklable():
    """A Connectivity built from Multitaper round-trips with standard pickle."""

    rng = np.random.default_rng(0)
    m = Multitaper(rng.standard_normal((128, 2, 3)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m)

    restored = pickle.loads(pickle.dumps(conn))
    np.testing.assert_allclose(
        restored.coherence_magnitude(), conn.coherence_magnitude()
    )


class _SlottedConnectivity(Connectivity):
    """Module-level slotted subclass so pickle can reference it by qualified name."""

    __slots__ = ("extra_metadata",)


class _StringSlottedConnectivity(Connectivity):
    """A subclass whose ``__slots__`` is a bare string (a single slot name)."""

    __slots__ = "extra_metadata"


@mark.parametrize("subclass", [_SlottedConnectivity, _StringSlottedConnectivity])
def test_pickle_and_copy_preserve_subclass_slots(subclass):
    """Python\'s default state handling preserves subclass slots."""

    rng = np.random.default_rng(1)
    m = Multitaper(rng.standard_normal((128, 2, 3)), sampling_frequency=500)
    conn = subclass.from_multitaper(m)
    conn.extra_metadata = {"subject": "s1"}

    for clone in (
        pickle.loads(pickle.dumps(conn)),
        copy.copy(conn),
        copy.deepcopy(conn),
    ):
        assert clone.extra_metadata == {"subject": "s1"}
        np.testing.assert_allclose(
            clone.coherence_magnitude(), conn.coherence_magnitude()
        )
