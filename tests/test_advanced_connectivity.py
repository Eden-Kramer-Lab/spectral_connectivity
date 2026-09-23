"""Tests for advanced connectivity measures: canonical_coherence, global_coherence, group_delay."""

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper


@pytest.mark.parametrize("method", ["group_delay", "delay"])
def test_delay_methods_explicitly_transfer_device_results_to_host(method, monkeypatch):
    """NumPy-only delay code must cross the device boundary through ``get``."""
    import spectral_connectivity.connectivity as connectivity_module

    class DeviceLike:
        """Minimal CuPy-like object that rejects implicit NumPy conversion."""

        def __init__(self, array):
            self._array = np.asarray(array)

        def __array__(self, *args, **kwargs):
            msg = "implicit device-to-host conversion is forbidden"
            raise TypeError(msg)

        def get(self):
            return self._array.copy()

    rng = np.random.default_rng(11)
    shape = (1, 4, 3, 16, 2)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    conn = Connectivity(coefficients)
    original_bandpass = connectivity_module._bandpass

    def device_bandpass(*args, **kwargs):
        data, frequencies = original_bandpass(*args, **kwargs)
        return DeviceLike(data), DeviceLike(frequencies)

    monkeypatch.setattr(connectivity_module, "_bandpass", device_bandpass)
    result = getattr(conn, method)()
    arrays = result if isinstance(result, tuple) else (result,)
    assert all(isinstance(array, np.ndarray) for array in arrays)


def _observations(fourier_coefficients):
    """Stack trials and tapers as observations.

    (n_time, n_trials, n_tapers, n_freq, n_signals) ->
    (n_time, n_freq, n_signals, n_trials * n_tapers).
    """
    n_time, n_trials, n_tapers, n_freq, n_signals = fourier_coefficients.shape
    stacked = fourier_coefficients.reshape(n_time, n_trials * n_tapers, n_freq, n_signals)
    return np.moveaxis(stacked, 1, -1)


def _cross_spectral_matrix(fourier_coefficients):
    """Per-bin (uncentered) cross-spectral matrix, shape (n_time, n_freq, n, n)."""
    x = _observations(fourier_coefficients)
    return x @ x.conj().swapaxes(-1, -2)


def _inverse_sqrt(hermitian):
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return (
        eigenvectors * eigenvalues[..., np.newaxis, :] ** -0.5
    ) @ eigenvectors.conj().swapaxes(-1, -2)


def _canonical_coherence_oracle(fourier_coefficients, group1, group2):
    """Squared largest canonical correlation between two groups of signals.

    Whitens the between-group cross-spectrum,
    ``K = S11^(-1/2) S12 S22^(-1/2)``, and returns the square of its largest
    singular value at every (time, frequency) bin, shape (n_time, n_freq).
    """
    csm = _cross_spectral_matrix(fourier_coefficients)
    s11 = csm[..., group1, :][..., :, group1]
    s12 = csm[..., group1, :][..., :, group2]
    s22 = csm[..., group2, :][..., :, group2]
    whitened = _inverse_sqrt(s11) @ s12 @ _inverse_sqrt(s22)
    return np.linalg.svd(whitened, compute_uv=False)[..., 0] ** 2


def _global_coherence_oracle(fourier_coefficients):
    """Eigenvalues of the per-bin cross-spectral matrix over its trace, descending.

    Shape (n_time, n_freq, n_signals).
    """
    csm = _cross_spectral_matrix(fourier_coefficients)
    eigenvalues = np.linalg.eigvalsh(csm)[..., ::-1]
    return eigenvalues / np.trace(csm, axis1=-2, axis2=-1).real[..., np.newaxis]


def _lagged_broadband(rng, lags, noise_levels, n_time, n_trials):
    """Signals that share one broadband source, each delayed by ``lags`` samples.

    Signal ``k`` is the common white-noise source circularly delayed by
    ``lags[k]`` samples plus independent noise of standard deviation
    ``noise_levels[k]``. Shape (n_time, n_trials, n_signals).
    """
    common = rng.standard_normal((n_time, n_trials))
    return np.stack(
        [
            np.roll(common, lag, axis=0) + noise * rng.standard_normal((n_time, n_trials))
            for lag, noise in zip(lags, noise_levels, strict=True)
        ],
        axis=-1,
    )


class TestCanonicalCoherence:
    """Test canonical_coherence() method."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Set up RNG for each test method."""
        self.rng = np.random.default_rng(42)

    def test_canonical_coherence_basic(self):
        """Canonical coherence detects a between-group shared source, not within.

        Both groups carry a common 20 Hz oscillation (random phase per trial),
        so the groups are coherent at 20 Hz. Group 2 additionally carries a
        private 40 Hz oscillation: its members are mutually coherent at 40 Hz,
        but group 1 has nothing there, so the *between-group* canonical
        coherence at 40 Hz must stay at the noise level.
        """
        n_time = 200
        n_trials = 10
        sampling_frequency = 500
        time = np.arange(n_time)[:, np.newaxis] / sampling_frequency

        shared = np.sin(2 * np.pi * 20 * time + self.rng.uniform(0, 2 * np.pi, n_trials))
        private = np.sin(2 * np.pi * 40 * time + self.rng.uniform(0, 2 * np.pi, n_trials))
        group1 = [
            shared + 0.5 * self.rng.standard_normal((n_time, n_trials)) for _ in range(3)
        ]
        group2 = [
            shared + private + 0.5 * self.rng.standard_normal((n_time, n_trials))
            for _ in range(3)
        ]
        # Shape (n_time, n_trials, n_signals)
        time_series = np.stack(group1 + group2, axis=-1)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)
        group_labels = np.array([0, 0, 0, 1, 1, 1])
        canonical_coh, labels = conn.canonical_coherence(group_labels)

        # (n_time_windows, n_non_negative_freqs, n_groups, n_groups)
        n_non_negative_freqs = m.frequencies.size // 2 + 1
        assert canonical_coh.shape == (m.time.size, n_non_negative_freqs, 2, 2)
        np.testing.assert_array_equal(labels, [0, 1])

        # Within-group entries are undefined; the between-group entry is symmetric.
        assert np.all(np.isnan(canonical_coh[..., 0, 0]))
        assert np.all(np.isnan(canonical_coh[..., 1, 1]))
        np.testing.assert_array_equal(canonical_coh[..., 0, 1], canonical_coh[..., 1, 0])

        # Values equal the squared largest canonical correlation.
        fourier_coefficients = m.fft()[..., :n_non_negative_freqs, :]
        np.testing.assert_allclose(
            canonical_coh[..., 0, 1],
            _canonical_coherence_oracle(fourier_coefficients, [0, 1, 2], [3, 4, 5]),
            rtol=1e-10,
            atol=1e-12,
        )

        frequencies = m.frequencies[:n_non_negative_freqs]
        at_20_hz = canonical_coh[0, np.argmin(np.abs(frequencies - 20)), 0, 1]
        at_40_hz = canonical_coh[0, np.argmin(np.abs(frequencies - 40)), 0, 1]
        # Beyond the 20 Hz peak's multitaper bandwidth (W = NW / T = 5 Hz).
        off_peak = canonical_coh[0, np.abs(frequencies - 20) > 10, 0, 1]
        assert at_20_hz > 0.95
        assert at_40_hz < 0.5
        assert off_peak.max() < 0.5
        # Group 2 is internally coherent at 40 Hz, so the low between-group
        # value there is not simply an absence of signal.
        coherence_at_40_hz = conn.coherence_magnitude()[0, np.argmin(np.abs(frequencies - 40))]
        assert coherence_at_40_hz[3, 4] > 0.9

    @pytest.mark.parametrize(
        "group_labels",
        [
            [0, 0, 1, 1, 1, 1, 2, 2, 2],  # unequal group sizes
            [0, 0, 1, 1],  # exactly two groups
            [10, 10, 20, 20, 30, 30],  # non-contiguous labels
            [2, 0, 1],  # single signal per group, unsorted labels
        ],
        ids=["unequal_sizes", "two_groups", "non_contiguous", "single_signal_per_group"],
    )
    def test_canonical_coherence_matches_canonical_correlation(self, group_labels):
        """Every group pair equals the squared largest canonical correlation.

        Covers shapes, sorted labels, undefined (NaN) diagonals, symmetry, the
        [0, 1] range, and the values themselves against an in-test oracle.
        """
        group_labels = np.asarray(group_labels)
        n_time, n_trials = 50, 20
        time_series = self.rng.standard_normal((n_time, n_trials, group_labels.size))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=1,
        )
        conn = Connectivity.from_multitaper(m)

        canonical_coh, labels = conn.canonical_coherence(group_labels)

        expected_labels = np.unique(group_labels)
        n_groups = expected_labels.size
        n_non_negative_freqs = m.frequencies.size // 2 + 1
        assert canonical_coh.shape == (m.time.size, n_non_negative_freqs, n_groups, n_groups)
        np.testing.assert_array_equal(labels, expected_labels)

        fourier_coefficients = m.fft()[..., :n_non_negative_freqs, :]
        for i, label_i in enumerate(expected_labels):
            assert np.all(np.isnan(canonical_coh[..., i, i]))
            for j, label_j in enumerate(expected_labels):
                if i == j:
                    continue
                expected = _canonical_coherence_oracle(
                    fourier_coefficients,
                    np.flatnonzero(group_labels == label_i),
                    np.flatnonzero(group_labels == label_j),
                )
                np.testing.assert_allclose(
                    canonical_coh[..., i, j], expected, rtol=1e-10, atol=1e-12
                )

        off_diagonal = canonical_coh[..., ~np.eye(n_groups, dtype=bool)]
        assert np.all(off_diagonal >= 0)
        assert np.all(off_diagonal <= 1 + 1e-12)

    def test_canonical_coherence_single_signal_groups_equal_coherence(self):
        """With one signal per group, canonical coherence is the pairwise coherence.

        The only linear combination of a single signal is the signal itself, so
        the maximal between-group coherence is exactly ``coherence_magnitude``.
        """
        time_series = self.rng.standard_normal((50, 5, 3))
        conn = Connectivity.from_multitaper(
            Multitaper(
                time_series=time_series, sampling_frequency=200, time_halfbandwidth_product=1
            )
        )
        canonical_coh, _ = conn.canonical_coherence(np.array([0, 1, 2]))
        np.testing.assert_allclose(
            canonical_coh, conn.coherence_magnitude(), rtol=0, atol=1e-12, equal_nan=True
        )


class TestGlobalCoherence:
    """Test global_coherence() method."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Set up RNG for each test method."""
        self.rng = np.random.default_rng(42)

    def test_global_coherence_basic(self):
        """Global coherence peaks where all signals share a common oscillation.

        Every signal carries the same 30 Hz oscillation (random phase per trial)
        plus independent noise, so the leading component captures nearly all of
        the power at 30 Hz but only a noise-level fraction elsewhere.
        """
        n_time = 200
        n_trials = 10
        n_signals = 5
        sampling_frequency = 500
        time = np.arange(n_time)[:, np.newaxis] / sampling_frequency

        common = np.sin(2 * np.pi * 30 * time + self.rng.uniform(0, 2 * np.pi, n_trials))
        time_series = np.stack(
            [
                (0.5 + 0.2 * i) * common + 0.3 * self.rng.standard_normal((n_time, n_trials))
                for i in range(n_signals)
            ],
            axis=-1,
        )  # (n_time, n_trials, n_signals)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)
        global_coh, global_coh_vectors = conn.global_coherence(max_rank=1)

        assert global_coh.shape == (m.time.size, m.frequencies.size, 1)
        assert global_coh_vectors.shape == (m.time.size, m.frequencies.size, n_signals, 1)
        np.testing.assert_allclose(
            global_coh, _global_coherence_oracle(m.fft())[..., :1], rtol=1e-10, atol=1e-12
        )

        frequencies = np.abs(m.frequencies)
        at_target = global_coh[0, np.argmin(np.abs(frequencies - 30)), 0]
        # Beyond the peak's multitaper bandwidth (W = NW / T = 5 Hz).
        off_peak = global_coh[0, np.abs(frequencies - 30) > 10, 0]
        assert at_target > 0.95
        assert off_peak.max() < 0.6

    def test_global_coherence_multiple_components(self):
        """max_rank > 1 returns the leading eigenvalue fractions, strongest first."""
        n_signals = 6
        time_series = self.rng.standard_normal((100, 10, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=500,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        max_rank = 3
        global_coh, global_coh_vectors = conn.global_coherence(max_rank=max_rank)

        assert global_coh.shape == (m.time.size, m.frequencies.size, max_rank)
        assert global_coh_vectors.shape == (
            m.time.size,
            m.frequencies.size,
            n_signals,
            max_rank,
        )
        np.testing.assert_allclose(
            global_coh,
            _global_coherence_oracle(m.fft())[..., :max_rank],
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.all(np.diff(global_coh, axis=-1) <= 1e-12)

    def test_global_coherence_is_scale_invariant(self):
        """Global coherence is an eigenvalue fraction and must not depend on gain.

        Scaling the input signals by a constant must leave the (normalized)
        global coherence unchanged.
        """
        time_series = self.rng.standard_normal((80, 8, 5))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        m_scaled = Multitaper(
            time_series=10.0 * time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        gc, _ = Connectivity.from_multitaper(m).global_coherence(max_rank=1)
        gc_scaled, _ = Connectivity.from_multitaper(m_scaled).global_coherence(max_rank=1)
        np.testing.assert_allclose(gc, gc_scaled, rtol=1e-6)

    @pytest.mark.parametrize("path", ["batched", "per_bin"])
    def test_global_coherence_all_components_bounded_and_scale_invariant(
        self, path, monkeypatch
    ):
        """All components (max_rank >= n_signals - 1) match the oracle and sum to 1.

        The per-bin path takes its dense-SVD branch here; the batched path is
        the default eigendecomposition. Forcing the per-bin fallback (normally
        used only above ``GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS``) exercises it
        at a small size.
        """
        import spectral_connectivity.connectivity as connectivity_module

        if path == "per_bin":
            monkeypatch.setattr(
                connectivity_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
            )
        n_signals = 3
        time_series = self.rng.standard_normal((80, 8, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        m_scaled = Multitaper(
            time_series=7.0 * time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        gc, _ = Connectivity.from_multitaper(m).global_coherence(max_rank=n_signals)
        gc_scaled, _ = Connectivity.from_multitaper(m_scaled).global_coherence(
            max_rank=n_signals
        )
        np.testing.assert_allclose(
            gc, _global_coherence_oracle(m.fft()), rtol=1e-10, atol=1e-12
        )
        # The fractions of every component sum to the whole (trace) power.
        np.testing.assert_allclose(gc.sum(axis=-1), 1.0, rtol=1e-12)
        np.testing.assert_allclose(gc, gc_scaled, rtol=1e-6)

    @pytest.mark.parametrize("path", ["batched", "per_bin"])
    def test_global_coherence_bounded_and_descending(self, path, monkeypatch):
        """Leading components match the oracle, strongest first, on both paths.

        max_rank=3 < n_signals - 1 sends the forced per-bin path through its
        truncated ``svds`` branch, which must return components strongest-first
        like the batched eigendecomposition.
        """
        import spectral_connectivity.connectivity as connectivity_module

        if path == "per_bin":
            monkeypatch.setattr(
                connectivity_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
            )
        n_signals = 6
        time_series = self.rng.standard_normal((80, 8, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        gc, _ = Connectivity.from_multitaper(m).global_coherence(max_rank=3)
        np.testing.assert_allclose(
            gc, _global_coherence_oracle(m.fft())[..., :3], rtol=1e-8, atol=1e-12
        )
        assert np.all(gc >= 0)
        assert np.all(gc <= 1.0 + 1e-12)
        assert np.all(gc[..., 0] >= gc[..., 1])
        assert np.all(gc[..., 1] >= gc[..., 2])

    def test_global_coherence_max_rank_edge_cases(self):
        """max_rank = n_signals and n_signals - 1 return that many components."""
        n_signals = 4
        time_series = self.rng.standard_normal((50, 5, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=1,
        )
        conn = Connectivity.from_multitaper(m)
        expected = _global_coherence_oracle(m.fft())

        for max_rank in (n_signals, n_signals - 1):
            global_coh, _ = conn.global_coherence(max_rank=max_rank)
            assert global_coh.shape == (m.time.size, m.frequencies.size, max_rank)
            np.testing.assert_allclose(
                global_coh, expected[..., :max_rank], rtol=1e-10, atol=1e-12
            )

    def test_global_coherence_two_signals_closed_form(self):
        """For two signals the leading fraction has a closed form.

        With per-bin cross-spectral matrix ``[[a, c], [c*, b]]`` the largest
        eigenvalue is ``(a + b) / 2 + sqrt(((a - b) / 2) ** 2 + |c| ** 2)``, so
        global coherence is that over the trace ``a + b``.
        """
        time_series = self.rng.standard_normal((50, 5, 2))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=1,
        )
        global_coh, _ = Connectivity.from_multitaper(m).global_coherence(max_rank=1)

        x = _observations(m.fft())
        a = np.sum(np.abs(x[..., 0, :]) ** 2, axis=-1)
        b = np.sum(np.abs(x[..., 1, :]) ** 2, axis=-1)
        c = np.sum(x[..., 0, :] * x[..., 1, :].conj(), axis=-1)
        largest = (a + b) / 2 + np.sqrt(((a - b) / 2) ** 2 + np.abs(c) ** 2)

        assert global_coh.shape == (m.time.size, m.frequencies.size, 1)
        np.testing.assert_allclose(global_coh[..., 0], largest / (a + b), rtol=1e-10)

    def test_global_coherence_values_are_squared_singular_values(self):
        """Global coherence is the squared singular values of the coefficient matrix.

        Per bin, the fraction for component k is ``s_k**2 / sum(s**2)`` where
        ``s`` are the singular values of the (n_signals, n_trials * n_tapers)
        matrix of Fourier coefficients.
        """
        time_series = self.rng.standard_normal((20, 3, 3))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=100,
            time_halfbandwidth_product=1,
        )
        global_coh, global_coh_vectors = Connectivity.from_multitaper(m).global_coherence(
            max_rank=2
        )

        singular_values = np.linalg.svd(_observations(m.fft()), compute_uv=False)
        power = singular_values**2
        expected = power / power.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(global_coh, expected[..., :2], rtol=1e-10, atol=1e-12)
        assert np.iscomplexobj(global_coh_vectors)

    def test_global_coherence_zero_power_bin_warns_and_nans(self):
        """A dead (zero-power) channel must yield NaN with a warning, not 0."""
        # Build fourier coefficients with an all-zero bin across every channel.
        rng = np.random.default_rng(0)
        shape = (1, 8, 1, 4, 3)  # (time, trials, tapers, freq, signals)
        fc = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fc[:, :, :, 2, :] = 0.0  # zero total power at frequency bin 2
        conn = Connectivity(fourier_coefficients=fc)
        with pytest.warns(UserWarning, match="zero total power"):
            gc, _ = conn.global_coherence(max_rank=1)
        assert np.all(np.isnan(gc[:, 2, :]))
        assert np.all(np.isfinite(gc[:, [0, 1, 3], :]))

    @pytest.mark.parametrize("path", ["batched", "per_bin"])
    def test_global_coherence_single_estimate_is_rank_one(self, path, monkeypatch):
        """One trial/taper (n_estimates=1) must not crash and puts all power in one.

        A single estimate gives a rank-one cross-spectral matrix, so the leading
        component holds the whole power (global coherence exactly 1). Checked on
        the batched thin-SVD path and on the forced per-bin fallback.
        """
        import spectral_connectivity.connectivity as connectivity_module

        if path == "per_bin":
            monkeypatch.setattr(
                connectivity_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
            )
        rng = np.random.default_rng(0)
        # (n_time, n_trials, n_tapers, n_fft, n_signals) with n_trials*n_tapers = 1
        fc = rng.standard_normal((1, 1, 1, 4, 3)) + 1j * rng.standard_normal((1, 1, 1, 4, 3))
        gc, _ = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=1)
        assert gc.shape == (1, 4, 1)
        np.testing.assert_allclose(gc, 1.0, rtol=1e-12)

    def test_global_coherence_over_requested_rank_is_clamped(self):
        """Requesting more components than exist clamps (no duplicate broadcast)."""
        rng = np.random.default_rng(2)
        # n_signals=3, n_estimates = n_trials * n_tapers = 8 -> at most 3 components
        fc = rng.standard_normal((1, 4, 2, 4, 3)) + 1j * rng.standard_normal((1, 4, 2, 4, 3))
        with pytest.warns(UserWarning, match="clamping"):
            gc, vectors = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=5)
        assert gc.shape[-1] == 3  # min(n_signals, n_estimates)
        assert vectors.shape[-1] == 3
        # Components are distinct, not one value broadcast into several.
        assert np.unique(np.round(gc[0, 0], 8)).size == 3

    def test_global_coherence_is_stable_at_extreme_magnitudes(self):
        """Extreme coefficient magnitudes must not underflow/overflow to NaN."""
        rng = np.random.default_rng(0)
        fc = rng.standard_normal((1, 8, 1, 4, 4)) + 1j * rng.standard_normal((1, 8, 1, 4, 4))
        base, _ = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=1)
        for scale in (1e-200, 1e200):
            gc, _ = Connectivity(fourier_coefficients=fc * scale).global_coherence(max_rank=1)
            assert np.all(np.isfinite(gc))
            np.testing.assert_allclose(gc, base, rtol=1e-6)


class TestGroupDelay:
    """Test group_delay() method."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Set up RNG for each test method."""
        self.rng = np.random.default_rng(42)

    def test_group_delay_basic(self):
        """Group delay recovers a known broadband time delay between two signals.

        Group delay is the slope of the coherence phase across frequency, so it
        is only well-defined when two signals are coherent across a *band*. We
        build a broadband common source and delay it by a known integer number
        of samples; the recovered signed delay must match the imposed lag.

        Regression test: previously this path returned all-NaN z-scores (the
        one-sample Fisher z-transform evaluated ``coherence_bias(0)``), so no
        frequency was ever significant and ``group_delay`` raised a
        ``zero-size array`` ``ValueError`` regardless of the data.
        """
        n_time = 2000
        n_trials = 30
        sampling_frequency = 500  # Hz
        lag_samples = 10  # signal2 lags signal1 by 10 samples == 20 ms
        expected_delay = lag_samples / sampling_frequency

        # Broadband common source shared by both signals, plus independent noise.
        common = self.rng.standard_normal((n_time, n_trials))
        signal1 = common + 0.2 * self.rng.standard_normal((n_time, n_trials))
        signal2 = np.roll(common, lag_samples, axis=0) + 0.2 * self.rng.standard_normal(
            (n_time, n_trials)
        )

        # Shape (n_time, n_trials, n_signals)
        time_series = np.stack([signal1, signal2], axis=-1)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_multitaper(m)

        # Standard (not relaxed) significance threshold: real coherence must be
        # detected for this to return anything.
        delay, slope, r_value = conn.group_delay(
            frequencies_of_interest=[20, 80],
            frequency_resolution=2.0,
            significance_threshold=0.05,
        )

        expected_shape = (m.time.size, 2, 2)
        assert delay.shape == expected_shape
        assert slope.shape == expected_shape
        assert r_value.shape == expected_shape

        # Diagonal (self-delay) is undefined.
        assert np.all(np.isnan(delay[..., 0, 0]))
        assert np.all(np.isnan(delay[..., 1, 1]))

        # A genuine, near-linear phase-frequency relationship must be recovered,
        # with the sign of the lead: signal 0 leads, so [0, 1] > 0 and [1, 0] < 0.
        np.testing.assert_allclose(delay[0, 0, 1], expected_delay, rtol=0, atol=5e-4)
        np.testing.assert_allclose(delay[0, 1, 0], -expected_delay, rtol=0, atol=5e-4)
        assert np.abs(r_value[0, 0, 1]) > 0.9

    def test_group_delay_without_frequency_band(self):
        """With no frequencies_of_interest, the whole band recovers signed lags.

        ``delay[..., i, j]`` is positive when signal ``i`` leads signal ``j``,
        i.e. ``(lags[j] - lags[i]) / sampling_frequency``.
        """
        sampling_frequency = 200
        lags = np.array([0, 3, 8])
        time_series = _lagged_broadband(self.rng, lags, [0.3, 0.3, 0.3], 200, 5)
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        delay, slope, r_value = conn.group_delay()

        n_signals = lags.size
        expected_shape = (m.time.size, n_signals, n_signals)
        assert delay.shape == expected_shape
        assert slope.shape == expected_shape
        assert r_value.shape == expected_shape
        for i in range(n_signals):
            assert np.all(np.isnan(delay[..., i, i]))

        off_diagonal = ~np.eye(n_signals, dtype=bool)
        expected_delay = (lags[np.newaxis, :] - lags[:, np.newaxis]) / sampling_frequency
        np.testing.assert_allclose(
            delay[0][off_diagonal], expected_delay[off_diagonal], rtol=0, atol=1.25e-3
        )

    def test_group_delay_with_frequency_resolution(self):
        """Different frequency resolutions both recover the signed lags.

        The resolution sets the independent-frequency step used to select the
        significant band, so the two fits differ slightly but must both match
        the imposed lags.
        """
        sampling_frequency = 200
        lags = np.array([0, 3, 8])
        time_series = _lagged_broadband(self.rng, lags, [0.3, 0.3, 0.3], 200, 5)
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        delay1, _slope1, _r_value1 = conn.group_delay(
            frequencies_of_interest=[10, 90], frequency_resolution=5.0
        )
        delay2, _slope2, _r_value2 = conn.group_delay(
            frequencies_of_interest=[10, 90], frequency_resolution=10.0
        )

        n_signals = lags.size
        expected_shape = (m.time.size, n_signals, n_signals)
        assert delay1.shape == expected_shape
        assert delay2.shape == expected_shape

        off_diagonal = ~np.eye(n_signals, dtype=bool)
        expected_delay = (lags[np.newaxis, :] - lags[:, np.newaxis]) / sampling_frequency
        for delay in (delay1, delay2):
            np.testing.assert_allclose(
                delay[0][off_diagonal], expected_delay[off_diagonal], rtol=0, atol=1.25e-3
            )
        # The resolution is honored: it changes which frequencies enter the fit.
        assert not np.allclose(delay1[0][off_diagonal], delay2[0][off_diagonal])

    def test_group_delay_significance_threshold(self):
        """A stricter threshold drops the weakly coherent pairs, keeps the strong one.

        Signals 0 and 1 are strongly coherent; signal 2 is buried in noise, so
        its pairs are significant at 0.05 but not at 1e-6. The stricter
        threshold's defined (finite) set must be a strict subset of the looser
        one's.
        """
        sampling_frequency = 200
        lags = np.array([0, 3, 8])
        time_series = _lagged_broadband(self.rng, lags, [0.3, 0.3, 0.8], 200, 5)
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        delay_strict, _, _ = conn.group_delay(
            frequencies_of_interest=[10, 90],
            frequency_resolution=5.0,
            significance_threshold=1e-6,
        )
        delay_loose, _, _ = conn.group_delay(
            frequencies_of_interest=[10, 90],
            frequency_resolution=5.0,
            significance_threshold=0.05,
        )

        off_diagonal = ~np.eye(lags.size, dtype=bool)
        finite_strict = np.isfinite(delay_strict[0])
        finite_loose = np.isfinite(delay_loose[0])
        # Loose: every pair has a significant band.
        np.testing.assert_array_equal(finite_loose, off_diagonal)
        # Strict: only the strongly coherent pair (0, 1) survives.
        expected_strict = np.zeros_like(off_diagonal)
        expected_strict[0, 1] = expected_strict[1, 0] = True
        np.testing.assert_array_equal(finite_strict, expected_strict)
        np.testing.assert_allclose(
            delay_strict[0, 0, 1], (lags[1] - lags[0]) / sampling_frequency, atol=1.25e-3
        )

    def test_group_delay_output_ranges(self):
        """Every off-diagonal output is defined, with r in [-1, 1] and a near-linear fit."""
        lags = np.array([0, 3, 8])
        time_series = _lagged_broadband(self.rng, lags, [0.3, 0.3, 0.3], 200, 5)
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        delay, slope, r_value = conn.group_delay(
            frequencies_of_interest=[10, 90], frequency_resolution=5.0
        )

        n_signals = lags.size
        off_diagonal = ~np.eye(n_signals, dtype=bool)
        # All n_signals * (n_signals - 1) off-diagonal entries are finite.
        assert np.isfinite(delay[..., off_diagonal]).all()
        assert np.isfinite(slope[..., off_diagonal]).all()
        assert np.isfinite(r_value[..., off_diagonal]).all()
        assert np.all(np.isnan(delay[..., ~off_diagonal]))
        # r is a correlation coefficient, and a pure delay fits a line closely.
        assert np.all(np.abs(r_value) <= 1)
        assert np.all(np.abs(r_value[..., off_diagonal]) > 0.99)
        # delay is the phase slope in cycles: slope / (2 * pi).
        np.testing.assert_allclose(
            delay[..., off_diagonal], slope[..., off_diagonal] / (2 * np.pi), rtol=1e-12
        )

    def test_group_delay_zero_power_bin_does_not_poison_slice(self):
        """A masked (zero-power) frequency must not NaN-poison a pair's fit.

        A zero-power frequency makes that bin's coherency NaN (0/0), which is
        masked out. The vectorized regression must exclude it rather than let
        ``0 * NaN`` propagate NaN through the whole slice's sums, so a pair that
        is coherent at the remaining frequencies still gets a finite slope/r.
        """
        n_time, n_trials, n_tapers, n_fft, n_signals = 1, 20, 1, 16, 3
        delay = 0.3
        bins = np.arange(n_fft)
        fourier = np.zeros((n_time, n_trials, n_tapers, n_fft, n_signals), dtype=complex)
        for trial in range(n_trials):
            base = self.rng.standard_normal(n_fft) + 1j * self.rng.standard_normal(n_fft)
            fourier[0, trial, 0, :, 0] = base
            # Signal 1 is signal 0 with a linear phase ramp (a broadband delay).
            fourier[0, trial, 0, :, 1] = base * np.exp(-1j * 2 * np.pi * bins * delay / n_fft)
            fourier[0, trial, 0, :, 2] = self.rng.standard_normal(
                n_fft
            ) + 1j * self.rng.standard_normal(n_fft)
        fourier[..., 5, :] = 0.0  # a zero-power frequency -> NaN coherency, masked

        with pytest.warns(UserWarning, match="zero power"):
            _delay, slope, r_value = Connectivity(fourier_coefficients=fourier).group_delay()

        # The coherent pair (0, 1) keeps a finite fit despite the masked bin.
        assert np.isfinite(slope[..., 0, 1]).all()
        assert np.isfinite(r_value[..., 0, 1]).all()

    def test_group_delay_stable_for_large_frequency_offset(self):
        """Group delay stays accurate when frequencies dwarf their spacing.

        With a large absolute-frequency offset, the naive raw-moment variance
        (``count * sum_xx - sum_x**2``) cancels to exactly zero in float64 and
        the slope blows up to inf/NaN. The mean-centered computation must still
        recover the injected phase slope. This is the case the centering exists
        for; comparing to ``linregress`` (which also centers) would not exercise
        it.
        """
        n_time, n_trials, n_tapers, n_fft, n_signals = 1, 30, 1, 32, 3
        expected_slope = -0.7
        # Frequency labels offset far from zero relative to their 0.001 spacing.
        frequencies = 1e6 + np.arange(n_fft) * 0.001
        # group_delay regresses over the non-negative frequencies; the raw-moment
        # variance genuinely cancels to zero on that grid in float64 (the old
        # failure the centering fixes).
        band = frequencies[: n_fft // 2 + 1]
        assert len(band) * (band @ band) - band.sum() ** 2 == 0.0

        fourier = np.zeros((n_time, n_trials, n_tapers, n_fft, n_signals), dtype=complex)
        for trial in range(n_trials):
            base = self.rng.standard_normal(n_fft) + 1j * self.rng.standard_normal(n_fft)
            fourier[0, trial, 0, :, 0] = base
            # coherency(0, 1) phase == angle(f0 * conj(f1)) == slope * (f - f0).
            fourier[0, trial, 0, :, 1] = base * np.exp(
                -1j * expected_slope * (frequencies - frequencies[0])
            )
            fourier[0, trial, 0, :, 2] = self.rng.standard_normal(
                n_fft
            ) + 1j * self.rng.standard_normal(n_fft)

        _delay, slope, r_value = Connectivity(
            fourier_coefficients=fourier, frequencies=frequencies
        ).group_delay()

        assert np.isfinite(slope[..., 0, 1]).all()
        np.testing.assert_allclose(slope[..., 0, 1], expected_slope, atol=1e-9)
        np.testing.assert_allclose(np.abs(r_value[..., 0, 1]), 1.0, atol=1e-9)

    def test_group_delay_matches_scipy_linregress(self):
        """The vectorized masked regression matches a per-slice scipy linregress.

        group_delay fits the unwrapped coherence phase against frequency per
        (time, signal pair) with vectorized masked sums. This must reproduce the
        slope and r-value that ``scipy.stats.mstats.linregress`` returns slice by
        slice, including NaN for slices with fewer than two significant
        frequencies.
        """
        from itertools import combinations

        from scipy.stats.mstats import linregress

        n_time, n_trials, n_signals, sampling_frequency = 400, 10, 4, 250
        time = np.arange(n_time) / sampling_frequency
        base = np.sin(2 * np.pi * 20 * time)
        signals = []
        for shift in range(n_signals):
            signal = np.roll(base, shift) + 0.2 * self.rng.standard_normal(n_time)
            trials = signal[:, np.newaxis] + 0.1 * self.rng.standard_normal((n_time, n_trials))
            signals.append(trials)
        time_series = np.stack(signals, axis=-1)  # (n_time, n_trials, n_signals)

        conn = Connectivity.from_multitaper(
            Multitaper(
                time_series=time_series,
                sampling_frequency=sampling_frequency,
                time_window_duration=0.4,
                time_halfbandwidth_product=3,
            )
        )
        _delay, slope, r_value = conn.group_delay()

        # Recompute a reference with the previous per-slice approach.
        frequencies = conn.frequencies
        from spectral_connectivity.connectivity import (
            _bandpass,
            _find_significant_frequencies,
            _get_independent_frequency_step,
        )

        step = _get_independent_frequency_step(frequencies[1] - frequencies[0], None)
        bandpassed, band_frequencies = _bandpass(conn.coherency(), frequencies, None)
        pairs = np.asarray(list(combinations(range(n_signals), 2)))
        bandpassed = bandpassed[..., pairs[:, 0], pairs[:, 1]]
        significant = _find_significant_frequencies(
            bandpassed, conn.n_observations, step, significance_threshold=0.05
        )
        phase = np.ma.masked_array(np.unwrap(np.angle(bandpassed), axis=-2), mask=~significant)

        reference_slope = np.full(slope.shape, np.nan)
        reference_r = np.ones(r_value.shape)
        it = np.ndindex(phase.shape[:-2])
        for lead in it:
            for pair_index, (i, j) in enumerate(pairs):
                fit = linregress(band_frequencies, y=phase[(*lead, slice(None), pair_index)])
                reference_slope[(*lead, i, j)] = fit[0]
                reference_slope[(*lead, j, i)] = -fit[0]
                reference_r[(*lead, i, j)] = fit[2]
                reference_r[(*lead, j, i)] = fit[2]

        np.testing.assert_array_equal(np.isnan(slope), np.isnan(reference_slope))
        np.testing.assert_allclose(
            slope, reference_slope, rtol=1e-9, atol=1e-11, equal_nan=True
        )
        np.testing.assert_allclose(r_value, reference_r, rtol=1e-9, atol=1e-11, equal_nan=True)

    def test_group_delay_antisymmetry(self):
        """Group delay is antisymmetric: delay[i, j] == -delay[j, i]."""
        lags = np.array([0, 3, 8])
        time_series = _lagged_broadband(self.rng, lags, [0.3, 0.3, 0.3], 200, 10)
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_multitaper(m)

        delay, _slope, _r_value = conn.group_delay(
            frequencies_of_interest=[10, 90], frequency_resolution=5.0
        )

        off_diagonal = ~np.eye(lags.size, dtype=bool)
        assert np.isfinite(delay[..., off_diagonal]).all()
        np.testing.assert_allclose(delay, -np.swapaxes(delay, -1, -2), rtol=0, atol=1e-12)


class TestAdvancedConnectivityIntegration:
    """Integration tests for advanced connectivity measures."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        """Set up RNG for each test method."""
        self.rng = np.random.default_rng(42)

    def test_multitaper_to_connectivity_to_advanced_measures(self):
        """Multitaper -> Connectivity -> advanced measures on a known lagged source.

        All six signals share one broadband source, each delayed by a known
        number of samples, so every measure has a known target: canonical and
        global coherence match their oracles and are high across the band, and
        group delay recovers the signed pairwise lags.
        """
        n_time = 200
        n_trials = 10
        sampling_frequency = 500
        lags = np.array([0, 2, 4, 6, 8, 10])
        n_signals = lags.size
        time_series = _lagged_broadband(self.rng, lags, [0.3] * n_signals, n_time, n_trials)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_multitaper(m)
        fourier_coefficients = m.fft()

        # Canonical coherence
        group_labels = np.array([0, 0, 0, 1, 1, 1])
        canonical_coh, _labels = conn.canonical_coherence(group_labels)
        n_non_negative_freqs = m.frequencies.size // 2 + 1
        assert canonical_coh.shape == (m.time.size, n_non_negative_freqs, 2, 2)
        np.testing.assert_allclose(
            canonical_coh[..., 0, 1],
            _canonical_coherence_oracle(
                fourier_coefficients[..., :n_non_negative_freqs, :], [0, 1, 2], [3, 4, 5]
            ),
            rtol=1e-10,
            atol=1e-12,
        )
        assert np.all(canonical_coh[..., 0, 1] > 0.8)

        # Global coherence
        global_coh, global_coh_vectors = conn.global_coherence(max_rank=2)
        assert global_coh.shape == (m.time.size, m.frequencies.size, 2)
        assert global_coh_vectors.shape == (
            m.time.size,
            m.frequencies.size,
            n_signals,
            2,
        )
        np.testing.assert_allclose(
            global_coh,
            _global_coherence_oracle(fourier_coefficients)[..., :2],
            rtol=1e-10,
            atol=1e-12,
        )
        # One shared source: the leading component dominates at low frequencies,
        # where the lags barely rotate the phases.
        low = np.abs(m.frequencies) <= 50
        assert np.all(global_coh[:, low, 0] > 0.8)

        # Group delay recovers the signed lags: [i, j] = (lags[j] - lags[i]) / fs.
        delay, slope, r_value = conn.group_delay(
            frequencies_of_interest=[10, 100],
            frequency_resolution=5.0,
            significance_threshold=0.05,
        )
        assert delay.shape == (m.time.size, n_signals, n_signals)
        assert slope.shape == (m.time.size, n_signals, n_signals)
        assert r_value.shape == (m.time.size, n_signals, n_signals)
        off_diagonal = ~np.eye(n_signals, dtype=bool)
        expected_delay = (lags[np.newaxis, :] - lags[:, np.newaxis]) / sampling_frequency
        np.testing.assert_allclose(
            delay[0][off_diagonal], expected_delay[off_diagonal], rtol=0, atol=5e-4
        )

    def test_advanced_measures_consistency(self):
        """Advanced measures reduce to basic ones in their degenerate cases.

        With one signal per group, canonical coherence is exactly the pairwise
        coherence magnitude; with every component kept, the global-coherence
        fractions partition the total power (sum to 1).
        """
        n_signals = 4
        time_series = self.rng.standard_normal((100, 10, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=500,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_multitaper(m)

        coherence = conn.coherence_magnitude()
        canonical_coh, _ = conn.canonical_coherence(np.arange(n_signals))
        np.testing.assert_allclose(
            canonical_coh, coherence, rtol=0, atol=1e-12, equal_nan=True
        )

        global_coh, _ = conn.global_coherence(max_rank=n_signals)
        np.testing.assert_allclose(global_coh.sum(axis=-1), 1.0, rtol=1e-12)
        assert np.all(global_coh >= 0)


class TestDelay:
    """Test the delay() method returns a time delay, not phase cycles."""

    @pytest.fixture(autouse=True)
    def setup_rng(self):
        self.rng = np.random.default_rng(3)

    def test_delay_recovers_frequency_independent_time_delay(self):
        """A constant physical delay must be frequency-independent (in seconds).

        delay() previously returned phase/(2*pi) (cycles), which scales with
        frequency for a fixed physical delay. After dividing by frequency the
        zero-wrap candidate is ~constant across the band and equals the signed
        lag: signal 0 leads, so [0, 1] is +lag/fs and [1, 0] is -lag/fs.
        """
        n_time = 2000
        n_trials = 30
        sampling_frequency = 500  # Hz
        lag_samples = 2  # small so phase does not wrap over the analysis band
        expected_delay = lag_samples / sampling_frequency  # 0.004 s

        common = self.rng.standard_normal((n_time, n_trials))
        signal1 = common + 0.2 * self.rng.standard_normal((n_time, n_trials))
        signal2 = np.roll(common, lag_samples, axis=0) + 0.2 * self.rng.standard_normal(
            (n_time, n_trials)
        )
        time_series = np.stack([signal1, signal2], axis=-1)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_multitaper(m)
        n_range = 3
        possible_delays = conn.delay(
            frequencies_of_interest=[20, 80],
            frequency_resolution=2.0,
            significance_threshold=0.05,
            n_range=n_range,
        )
        # Shape (n_time, n_freq, 2*n_range+1, n_signals, n_signals).
        # The zero-wrap candidate is the middle one (k=0).
        zero_wrap = possible_delays[0, :, n_range, 0, 1]
        # Non-significant frequencies (and DC) are NaN, not a spurious 0.0.
        assert np.isnan(zero_wrap).any()
        is_estimated = np.isfinite(zero_wrap)
        assert is_estimated.sum() >= 3
        # Frequency-independent (constant) and equal to the signed lag/fs.
        np.testing.assert_allclose(zero_wrap[is_estimated], expected_delay, rtol=0, atol=5e-4)
        reverse = possible_delays[0, :, n_range, 1, 0]
        np.testing.assert_allclose(reverse[is_estimated], -expected_delay, rtol=0, atol=5e-4)
        assert np.std(zero_wrap[is_estimated]) < 1e-3
