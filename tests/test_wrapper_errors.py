import numpy as np
import pytest

from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    multitaper_connectivity,
)


def test_unsupported_method_error_message():
    """Test that unsupported methods provide actionable error messages."""
    # Create test data
    n_time_samples, n_trials, n_signals = 100, 5, 2
    time_series = np.random.random((n_time_samples, n_trials, n_signals))

    m = Multitaper(
        time_series=time_series,
        sampling_frequency=1000,
        time_window_duration=0.1,
    )

    # Test directed method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="directed_coherence")

    error_msg = str(exc_info.value)
    assert "is not supported by the xarray interface" in error_msg
    assert "Connectivity class directly" in error_msg
    assert "from spectral_connectivity import Connectivity" in error_msg
    assert "conn = Connectivity.from_multitaper(m)" in error_msg
    assert "result = conn.directed_coherence()" in error_msg

    # Test canonical_coherence method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="canonical_coherence")

    error_msg = str(exc_info.value)
    assert "canonical_coherence" in error_msg
    assert "result = conn.canonical_coherence()" in error_msg

    # Test group_delay method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="group_delay")

    error_msg = str(exc_info.value)
    assert "group_delay" in error_msg
    assert "result = conn.group_delay()" in error_msg

    # global_coherence returns a tuple; the wrapper must give the friendly
    # message rather than a cryptic xarray "coords is not dict-like" error.
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="global_coherence")
    error_msg = str(exc_info.value)
    assert "is not supported by the xarray interface" in error_msg
    assert "result = conn.global_coherence()" in error_msg


def test_time_halfbandwidth_product_kwarg_passthrough():
    """The documented Multitaper kwarg name works through the wrapper."""
    rng = np.random.default_rng(0)
    time_series = rng.random((200, 5, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=1000,
        method="coherence_magnitude",
        time_halfbandwidth_product=3,
    )
    assert result.name == "coherence_magnitude"
    # The (misspelled) name from the old docstring must not silently work.
    with pytest.raises(TypeError):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method="coherence_magnitude",
            time_bandwidth_product=3,
        )


def test_injected_connectivity_mismatch_raises():
    """A Connectivity not built from ``m`` is rejected, not silently mislabeled.

    ``connectivity_to_xarray`` takes results/coordinates from the injected
    ``connectivity`` but metadata from ``m``; a mismatched pair (e.g. a
    different sampling frequency) would otherwise produce output whose frequency
    axis and ``mt_sampling_frequency`` attribute disagree.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 3, 4))
    m_500 = Multitaper(time_series, sampling_frequency=500)
    m_1000 = Multitaper(time_series, sampling_frequency=1000)
    connectivity_500 = Connectivity.from_multitaper(m_500)

    # Consistent instance is accepted.
    connectivity_to_xarray(m_500, "coherence_magnitude", _connectivity=connectivity_500)

    # Different sampling frequency -> different frequency grid -> rejected.
    with pytest.raises(ValueError, match="not built from this `Multitaper`"):
        connectivity_to_xarray(
            m_1000, "coherence_magnitude", _connectivity=connectivity_500
        )

    # Different channel count -> rejected.
    m_two_signals = Multitaper(time_series[..., :2], sampling_frequency=500)
    with pytest.raises(ValueError, match="n_signals"):
        connectivity_to_xarray(
            m_two_signals, "coherence_magnitude", _connectivity=connectivity_500
        )

    # Different number of time windows (same frequency grid and channel count)
    # -> the `time` branch of the validator rejects it.
    longer_series = rng.standard_normal((768, 3, 4))
    m_short = Multitaper(time_series, sampling_frequency=500, time_window_duration=0.2)
    m_long = Multitaper(longer_series, sampling_frequency=500, time_window_duration=0.2)
    connectivity_short = Connectivity.from_multitaper(m_short)
    with pytest.raises(ValueError, match="time"):
        connectivity_to_xarray(
            m_long, "coherence_magnitude", _connectivity=connectivity_short
        )


def test_injected_connectivity_same_geometry_different_data_rejected():
    """Identical geometry but a different recording must be rejected.

    Geometry (channel count, frequency grid, time bins) cannot establish
    provenance: two different datasets with the same sampling frequency, window,
    and channel count share it. Without an identity check the result would carry
    the injected instance's connectivity with `m`'s metadata — a silent
    mislabeling. Provenance is verified by the source recorded in
    `from_multitaper`, so a Connectivity built from a *different* Multitaper of
    the same shape is rejected even though every geometry check passes.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(0)
    data_a = rng.standard_normal((512, 3, 4))
    data_b = rng.standard_normal((512, 3, 4))  # same shape, different data
    m_a = Multitaper(data_a, sampling_frequency=500)
    m_b = Multitaper(data_b, sampling_frequency=500)
    conn_a = Connectivity.from_multitaper(m_a)

    # Sanity: the two transforms are geometrically identical (the old geometry-
    # only validator would have accepted this mismatch).
    assert np.array_equal(m_a.frequencies, m_b.frequencies)
    assert np.array_equal(m_a.time, m_b.time)

    with pytest.raises(ValueError, match="cannot be verified to come from"):
        connectivity_to_xarray(m_b, "coherence_magnitude", _connectivity=conn_a)


def test_injected_connectivity_reassigned_coefficients_rejected():
    """Reassigning coefficients clears the provenance link, so injection fails.

    After `conn = from_multitaper(m)`, assigning `conn.fourier_coefficients`
    replaces the data; the instance no longer provably holds `m`'s coefficients,
    so it must not be accepted with `m` even though the geometry still matches.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(1)
    m = Multitaper(rng.standard_normal((512, 3, 4)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m)
    # Accepted before reassignment (provenance intact).
    connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)
    # Reassign (even to a fresh transform of the same data): link is cleared.
    conn.fourier_coefficients = m.fft()
    with pytest.raises(ValueError, match="cannot be verified to come from"):
        connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)


def test_injected_connectivity_mutated_coordinate_rejected():
    """Mutating a public coordinate after from_multitaper is caught by geometry.

    Identity alone would still accept the instance (its recorded source is still
    `m`), but the output would take the mutated `time` coordinate while the
    metadata comes from `m` — a silent disagreement. Geometry is validated
    regardless of the provenance link, so this is rejected.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((512, 3, 4)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m)
    # Provenance link is intact (same source object), but shift the coordinate.
    conn.time = conn.time + 1.0
    with pytest.raises(ValueError, match="time"):
        connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)


@pytest.mark.parametrize(
    "attr,value",
    [
        ("detrend_type", "linear"),
        ("time_halfbandwidth_product", 5),
    ],
)
def test_multitaper_source_parameter_change_is_prevented(attr, value):
    """A source transform cannot diverge from its coefficients after construction."""
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(4)
    m = Multitaper(
        rng.standard_normal((512, 3, 3)),
        sampling_frequency=500,
        time_halfbandwidth_product=3,
    )
    conn = Connectivity.from_multitaper(m)
    with pytest.raises(AttributeError, match="immutable after construction"):
        setattr(m, attr, value)
    # The failed mutation leaves the source and its coefficient snapshot aligned.
    connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)
    # Retain the provenance defense for legacy objects/subclasses that bypass the
    # public immutability guard.
    object.__setattr__(m, attr, value)
    with pytest.raises(ValueError, match="was modified after"):
        connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)


def test_injected_connectivity_nondefault_expectation_type_rejected():
    """Only the default expectation_type fits the fixed xarray layout.

    `from_multitaper(m, expectation_type="time_trials_tapers")` is provably from
    `m`, but it averages the time axis, so its result does not fit the
    (time, frequency, source, target) layout. Require an actionable error rather
    than a cryptic xarray dimension-mismatch.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(2)
    m = Multitaper(rng.standard_normal((512, 3, 4)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m, expectation_type="time_trials_tapers")
    with pytest.raises(ValueError, match="expectation_type='trials_tapers'"):
        connectivity_to_xarray(m, "coherence_magnitude", _connectivity=conn)


def test_connectivity_injection_is_not_public():
    """The Connectivity-sharing hook is internal (keyword-only ``_connectivity``).

    It was a public `connectivity=` argument tying results to a live, mutable
    Multitaper — a provenance footgun. It is now internal, so the public call
    cannot inject a (potentially stale) instance; reuse a transform by calling
    the Connectivity methods directly or via multitaper_connectivity([...]).
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(5)
    m = Multitaper(rng.standard_normal((256, 3, 3)), sampling_frequency=500)
    conn = Connectivity.from_multitaper(m)
    with pytest.raises(TypeError):
        connectivity_to_xarray(m, "coherence_magnitude", connectivity=conn)
