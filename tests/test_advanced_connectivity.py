"""Tests for advanced connectivity measures: canonical_coherence, global_coherence, group_delay."""

import re
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, SidednessWarning
from spectral_connectivity.simulate import (
    simulate_lagged_broadband,
    simulate_shared_oscillation,
)
from tests._backend_helpers import to_host


@pytest.mark.parametrize("method", ["group_delay", "delay"])
def test_delay_methods_explicitly_transfer_device_results_to_host(method, monkeypatch):
    """NumPy-only delay code must cross the device boundary through ``get``."""
    import spectral_connectivity.connectivity as connectivity_module

    class DeviceLike:
        """Minimal CuPy-like object that rejects implicit NumPy conversion."""

        def __init__(self, array):
            self._array = to_host(array)

        def __array__(self, *args, **kwargs):
            msg = "implicit device-to-host conversion is forbidden"
            raise TypeError(msg)

        def get(self):
            return self._array.copy()

        @property
        def __cuda_array_interface__(self):
            # Like cupy.ndarray: marks this as a device array.
            return {"version": 3}

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


def _nonnegative_bins(fourier_coefficients):
    """The non-negative frequency bins of two-sided coefficients in FFT order."""
    return fourier_coefficients[..., : fourier_coefficients.shape[-2] // 2 + 1, :]


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

        # Shape (n_time, n_trials, n_signals): every signal carries 20 Hz plus
        # noise; only group 2 (signals 3-5) carries the 40 Hz rhythm.
        shared = simulate_shared_oscillation(
            20,
            sampling_frequency,
            n_time,
            n_trials,
            amplitudes=[1.0] * 6,
            noise_levels=0.5,
            random_state=self.rng,
        )
        private = simulate_shared_oscillation(
            40,
            sampling_frequency,
            n_time,
            n_trials,
            amplitudes=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            random_state=self.rng,
        )
        time_series = shared + private

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)
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
        fourier_coefficients = to_host(m.fft())[..., :n_non_negative_freqs, :]
        np.testing.assert_allclose(
            canonical_coh[..., 0, 1],
            _canonical_coherence_oracle(fourier_coefficients, [0, 1, 2], [3, 4, 5]),
            rtol=1e-10,
            atol=1e-12,
        )

        frequencies = to_host(m.frequencies)[:n_non_negative_freqs]
        at_20_hz = canonical_coh[0, np.argmin(np.abs(frequencies - 20)), 0, 1]
        at_40_hz = canonical_coh[0, np.argmin(np.abs(frequencies - 40)), 0, 1]
        # Beyond the 20 Hz peak's multitaper bandwidth (W = NW / T = 5 Hz).
        off_peak = canonical_coh[0, np.abs(frequencies - 20) > 10, 0, 1]
        assert at_20_hz > 0.95
        assert at_40_hz < 0.5
        assert off_peak.max() < 0.5
        # Group 2 is internally coherent at 40 Hz, so the low between-group
        # value there is not simply an absence of signal.
        coherence_at_40_hz = conn.magnitude_squared_coherence()[
            0, np.argmin(np.abs(frequencies - 40))
        ]
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
        conn = Connectivity.from_transform(m)

        canonical_coh, labels = conn.canonical_coherence(group_labels)

        expected_labels = np.unique(group_labels)
        n_groups = expected_labels.size
        n_non_negative_freqs = m.frequencies.size // 2 + 1
        assert canonical_coh.shape == (m.time.size, n_non_negative_freqs, n_groups, n_groups)
        np.testing.assert_array_equal(labels, expected_labels)

        fourier_coefficients = to_host(m.fft())[..., :n_non_negative_freqs, :]
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
        the maximal between-group coherence is exactly ``magnitude_squared_coherence``.
        """
        time_series = self.rng.standard_normal((50, 5, 3))
        conn = Connectivity.from_transform(
            Multitaper(
                time_series=time_series, sampling_frequency=200, time_halfbandwidth_product=1
            )
        )
        canonical_coh, _ = conn.canonical_coherence(np.array([0, 1, 2]))
        np.testing.assert_allclose(
            canonical_coh,
            conn.magnitude_squared_coherence(),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        )

    @pytest.mark.parametrize("extra", ["duplicate", "scaled_duplicate", "dead"])
    def test_canonical_coherence_ignores_rank_deficient_channel(self, extra):
        """A dead or duplicated channel adds no direction to its group.

        The group's canonical coherence must equal that of its independent
        channels alone (an SVD-based whitening would instead give the null
        direction an arbitrary unit-norm vector, which can inflate the value).
        """
        shape = (2, 6, 3, 9, 4)  # 18 observations
        coefficients = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        coefficients[..., 2:] += 0.7 * coefficients[..., :2]
        added = {
            "duplicate": coefficients[..., :1],
            "scaled_duplicate": (2 - 1j) * coefficients[..., :1],
            "dead": np.zeros_like(coefficients[..., :1]),
        }[extra]
        with_extra = np.concatenate([coefficients, added], axis=-1)

        expected, _ = Connectivity(coefficients).canonical_coherence(np.array([0, 0, 1, 1]))
        with pytest.warns(UserWarning, match=r"group\(s\) \[0\] contain linearly dependent"):
            result, _ = Connectivity(with_extra).canonical_coherence(np.array([0, 0, 1, 1, 0]))
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12, equal_nan=True)

    @pytest.mark.parametrize(
        ("n_signals", "n_trials", "condition"),
        [(4, 10, 1.0), (4, 10, 1e5), (5, 1, 1.0)],  # last: more signals than observations
        ids=["wide", "ill_conditioned", "tall"],
    )
    def test_canonical_coherence_matches_svd_whitening(self, n_signals, n_trials, condition):
        """Canonical coherence equals the SVD (polar-factor) computation.

        Covers an ill-conditioned group, where whitening through the Gram
        matrix alone loses accuracy, and a group with more signals than
        trial x taper observations, where the smaller Gram matrix is used.
        """
        import warnings

        shape = (2, n_trials, 3, 7, n_signals + 2)
        coefficients = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        # Signal 1 differs from signal 0 by a 1/condition-sized component.
        coefficients[..., 1] = coefficients[..., 0] + coefficients[..., 1] / condition
        coefficients[..., n_signals:] += 0.5 * coefficients[..., :2]

        def polar(x):
            left, _, right_h = np.linalg.svd(x, full_matrices=False)
            return left @ right_h

        observations = _observations(coefficients)
        cross = polar(observations[..., :n_signals, :]) @ polar(
            observations[..., n_signals:, :]
        ).conj().swapaxes(-1, -2)
        expected = np.linalg.svd(cross, compute_uv=False)[..., 0] ** 2

        labels = np.array([0] * n_signals + [1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # tall: forced-to-1 warning
            result, _ = Connectivity(coefficients).canonical_coherence(labels)
        n_freq = result.shape[1]
        np.testing.assert_allclose(result[..., 0, 1], expected[:, :n_freq], rtol=1e-9)

    @pytest.mark.parametrize("scale", [1e-9, 1e9])
    def test_canonical_coherence_is_invariant_to_channel_scale(self, scale):
        """Rescaling one channel (e.g. mixed units, V vs nV) leaves the group's
        row space, and so its canonical coherence, unchanged: a small but
        independent channel must not be dropped as rank-deficient."""
        shape = (2, 10, 3, 7, 4)
        coefficients = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        coefficients[..., 2:] += 0.5 * coefficients[..., :2]
        labels = np.array([0, 0, 1, 1])
        expected, _ = Connectivity(coefficients).canonical_coherence(labels)

        rescaled = coefficients.copy()
        rescaled[..., 1] *= scale
        result, _ = Connectivity(rescaled).canonical_coherence(labels)
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12, equal_nan=True)

    @pytest.mark.parametrize("n_signals", [2, 16])
    def test_canonical_coherence_is_invariant_to_ill_conditioned_mixing(self, n_signals):
        """Invertible mixing leaves a group's row space, and so its canonical
        coherence, unchanged, however ill-conditioned the mixing.

        Regression: whitening through the Gram matrix alone dropped directions
        with singular value below about 1e-7 of the largest, so a mixing of
        condition number 1e8 changed the value by up to 0.66.
        """
        shape = (2, 3 * n_signals, 1, 6)
        signals = self.rng.standard_normal(
            (*shape, n_signals)
        ) + 1j * self.rng.standard_normal((*shape, n_signals))
        other = 0.6 * signals[..., :2] + self.rng.standard_normal((*shape, 2))
        rotation1, _ = np.linalg.qr(self.rng.standard_normal((n_signals, n_signals)))
        rotation2, _ = np.linalg.qr(self.rng.standard_normal((n_signals, n_signals)))
        mixing = rotation1 @ np.diag(np.logspace(0, -8, n_signals)) @ rotation2
        labels = np.array([0] * n_signals + [1, 1])

        expected, _ = Connectivity(np.concatenate([signals, other], -1)).canonical_coherence(
            labels
        )
        result, _ = Connectivity(
            np.concatenate([signals @ mixing.T, other], -1)
        ).canonical_coherence(labels)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6, equal_nan=True)

    def test_complex64_canonical_coherence_keeps_near_collinear_channels(self):
        """complex64 coefficients are whitened at complex128.

        Regression: whitening through the Gram matrix squares the condition
        number, so at complex64 precision a real but nearly collinear channel
        (here 1e-4 of its partner) fell below the rank threshold and the value
        collapsed to ~0 instead of ~0.92.
        """
        shape = (1, 50, 5, 6, 1)
        x, y, w, noise = (
            self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
            for _ in range(4)
        )
        coefficients = np.concatenate([x, x + 1e-4 * y, y + 0.3 * noise, w], axis=-1)
        labels = np.array([0, 0, 1, 1])
        expected, _ = Connectivity(coefficients).canonical_coherence(labels)
        result, _ = Connectivity(coefficients.astype(np.complex64)).canonical_coherence(labels)
        np.testing.assert_allclose(result, expected, atol=1e-3, equal_nan=True)
        assert np.nanmin(expected[..., 0, 1]) > 0.5

    def test_canonical_coherence_of_a_powerless_group_is_nan(self):
        """A group whose signals are all zero has 0/0 canonical coherence: NaN
        with a warning naming the group, like coherency, not a confident 0."""
        shape = (1, 8, 3, 6, 4)
        coefficients = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        coefficients[..., 3] = 0
        with pytest.warns(UserWarning, match=r"group\(s\) \['c'\] have no power"):
            result, _ = Connectivity(coefficients).canonical_coherence(
                np.array(["a", "a", "b", "c"])
            )
        assert np.isnan(result[..., :, 2]).all()
        assert np.isnan(result[..., 2, :]).all()
        assert np.isfinite(result[..., 0, 1]).all()

    def test_canonical_coherence_with_chunked_eigh_matches(self, monkeypatch):
        """The whitening eigh is chunked (CuPy's batched eigh fails on very
        large batches); chunked and single-call results are identical."""
        from spectral_connectivity import _array_utils

        shape = (2, 6, 3, 9, 6)
        coefficients = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        labels = np.array([0, 0, 0, 1, 1, 1])
        single, _ = Connectivity(coefficients).canonical_coherence(labels)
        # Chunks of 4 of the 18 bins per group: a partial last chunk.
        monkeypatch.setattr(_array_utils, "_eigh_chunk_size", lambda n_signals: 4)
        chunked, _ = Connectivity(coefficients).canonical_coherence(labels)
        np.testing.assert_array_equal(chunked, single)


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

        time_series = simulate_shared_oscillation(
            30,
            sampling_frequency,
            n_time,
            n_trials,
            amplitudes=0.5 + 0.2 * np.arange(n_signals),
            noise_levels=0.3,
            random_state=self.rng,
        )  # (n_time, n_trials, n_signals)

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)
        global_coh, global_coh_vectors = conn.global_coherence(max_rank=1)

        assert global_coh.shape == (m.time.size, m.frequencies.size // 2 + 1, 1)
        assert global_coh_vectors.shape == (
            m.time.size,
            m.frequencies.size // 2 + 1,
            n_signals,
            1,
        )
        np.testing.assert_allclose(
            global_coh,
            _global_coherence_oracle(_nonnegative_bins(to_host(m.fft())))[..., :1],
            rtol=1e-10,
            atol=1e-12,
        )

        frequencies = conn.frequencies
        at_target = global_coh[0, np.argmin(np.abs(frequencies - 30)), 0]
        # Beyond the peak's multitaper bandwidth (W = NW / T = 5 Hz).
        off_peak = global_coh[0, np.abs(frequencies - 30) > 10, 0]
        assert at_target > 0.95
        assert off_peak.max() < 0.6

    @pytest.mark.parametrize("n_time_samples", [100, 101], ids=["even_fft", "odd_fft"])
    def test_global_coherence_nonnegative_bins_lose_nothing(self, n_time_samples):
        """Only the non-negative bins are returned, and they hold everything.

        For a real signal S(-f) = conj(S(f)), so the negative bins of the
        two-sided decomposition (all that 2.x returned) have the same fractions
        as their positive partners and conjugated vectors. The non-negative
        bins themselves are bit-for-bit those of the two-sided decomposition.
        """
        from spectral_connectivity._multivariate import (
            GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
            _global_coherence,
        )

        n_signals = 4
        m = Multitaper(
            time_series=self.rng.standard_normal((n_time_samples, 6, n_signals)),
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)
        fractions, vectors = conn.global_coherence(max_rank=n_signals)
        full_fractions, full_vectors = (
            to_host(values)
            for values in _global_coherence(
                m.fft(), None, n_signals, GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS
            )
        )

        n_frequencies = conn.frequencies.size
        n_fft_samples = full_fractions.shape[1]
        assert fractions.shape == (conn.time.size, n_frequencies, n_signals)
        assert vectors.shape == (conn.time.size, n_frequencies, n_signals, n_signals)
        np.testing.assert_array_equal(fractions, full_fractions[:, :n_frequencies])
        np.testing.assert_array_equal(vectors, full_vectors[:, :n_frequencies])

        # Bin k (0 < k < n_fft_samples - k) pairs with bin n_fft_samples - k.
        positive = np.arange(1, n_fft_samples - n_frequencies + 1)
        negative = n_fft_samples - positive
        np.testing.assert_allclose(
            full_fractions[:, negative], fractions[:, positive], rtol=1e-10, atol=1e-14
        )
        # conj(v) up to a phase: |<conj(v(f)), v(-f)>| = 1 for unit vectors.
        overlap = np.abs(np.sum(vectors[:, positive] * full_vectors[:, negative], axis=-2))
        np.testing.assert_allclose(overlap, 1.0, rtol=1e-8)

    def test_global_coherence_of_one_sided_coefficients_keeps_every_bin(self):
        """One-sided coefficients hold only non-negative bins: all are returned."""
        coefficients = self.rng.standard_normal(
            (1, 6, 2, 5, 3)
        ) + 1j * self.rng.standard_normal((1, 6, 2, 5, 3))
        conn = Connectivity(coefficients, is_one_sided=True)
        fractions, vectors = conn.global_coherence(max_rank=2)
        assert fractions.shape == (1, 5, 2)
        assert vectors.shape == (1, 5, 3, 2)
        np.testing.assert_allclose(
            fractions, _global_coherence_oracle(coefficients)[..., :2], rtol=1e-10, atol=1e-12
        )

    def test_global_coherence_multiple_components(self):
        """max_rank > 1 returns the leading eigenvalue fractions, strongest first."""
        n_signals = 6
        time_series = self.rng.standard_normal((100, 10, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=500,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)

        max_rank = 3
        global_coh, global_coh_vectors = conn.global_coherence(max_rank=max_rank)

        assert global_coh.shape == (m.time.size, m.frequencies.size // 2 + 1, max_rank)
        assert global_coh_vectors.shape == (
            m.time.size,
            m.frequencies.size // 2 + 1,
            n_signals,
            max_rank,
        )
        np.testing.assert_allclose(
            global_coh,
            _global_coherence_oracle(_nonnegative_bins(to_host(m.fft())))[..., :max_rank],
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
        gc, _ = Connectivity.from_transform(m).global_coherence(max_rank=1)
        gc_scaled, _ = Connectivity.from_transform(m_scaled).global_coherence(max_rank=1)
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
        import spectral_connectivity._multivariate as multivariate_module

        if path == "per_bin":
            monkeypatch.setattr(
                multivariate_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
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
        gc, _ = Connectivity.from_transform(m).global_coherence(max_rank=n_signals)
        gc_scaled, _ = Connectivity.from_transform(m_scaled).global_coherence(
            max_rank=n_signals
        )
        np.testing.assert_allclose(
            gc,
            _global_coherence_oracle(_nonnegative_bins(to_host(m.fft()))),
            rtol=1e-10,
            atol=1e-12,
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
        import spectral_connectivity._multivariate as multivariate_module

        if path == "per_bin":
            monkeypatch.setattr(
                multivariate_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
            )
        n_signals = 6
        time_series = self.rng.standard_normal((80, 8, n_signals))
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        gc, _ = Connectivity.from_transform(m).global_coherence(max_rank=3)
        np.testing.assert_allclose(
            gc,
            _global_coherence_oracle(_nonnegative_bins(to_host(m.fft())))[..., :3],
            rtol=1e-8,
            atol=1e-12,
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
        conn = Connectivity.from_transform(m)
        expected = _global_coherence_oracle(_nonnegative_bins(to_host(m.fft())))

        for max_rank in (n_signals, n_signals - 1):
            global_coh, _ = conn.global_coherence(max_rank=max_rank)
            assert global_coh.shape == (m.time.size, m.frequencies.size // 2 + 1, max_rank)
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
        global_coh, _ = Connectivity.from_transform(m).global_coherence(max_rank=1)

        x = _observations(_nonnegative_bins(to_host(m.fft())))
        a = np.sum(np.abs(x[..., 0, :]) ** 2, axis=-1)
        b = np.sum(np.abs(x[..., 1, :]) ** 2, axis=-1)
        c = np.sum(x[..., 0, :] * x[..., 1, :].conj(), axis=-1)
        largest = (a + b) / 2 + np.sqrt(((a - b) / 2) ** 2 + np.abs(c) ** 2)

        assert global_coh.shape == (m.time.size, m.frequencies.size // 2 + 1, 1)
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
        global_coh, global_coh_vectors = Connectivity.from_transform(m).global_coherence(
            max_rank=2
        )

        singular_values = np.linalg.svd(
            _observations(_nonnegative_bins(to_host(m.fft()))), compute_uv=False
        )
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
        assert np.all(np.isnan(gc[:, 2, :]))  # bin 2 of 4 is Nyquist
        assert np.all(np.isfinite(gc[:, [0, 1], :]))

    @pytest.mark.parametrize("path", ["batched", "per_bin"])
    def test_global_coherence_single_estimate_is_rank_one(self, path, monkeypatch):
        """One trial/taper (n_estimates=1) must not crash and puts all power in one.

        A single estimate gives a rank-one cross-spectral matrix, so the leading
        component holds the whole power (global coherence exactly 1, which is
        announced). Checked on the batched thin-SVD path and on the forced
        per-bin fallback.
        """
        import spectral_connectivity._multivariate as multivariate_module

        if path == "per_bin":
            monkeypatch.setattr(
                multivariate_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0
            )
        rng = np.random.default_rng(0)
        # (n_time, n_trials, n_tapers, n_fft, n_signals) with n_trials*n_tapers = 1
        fc = rng.standard_normal((1, 1, 1, 4, 3)) + 1j * rng.standard_normal((1, 1, 1, 4, 3))
        with pytest.warns(UserWarning, match="global_coherence is computed from a single"):
            gc, _ = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=1)
        assert gc.shape == (1, 3, 1)  # the non-negative bins of 4
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


# The expectation_type settings global coherence supports (they average the
# tapers), with the observation axes each keeps ahead of the frequency axis.
_GLOBAL_COHERENCE_KEPT_AXES = {
    "trials_tapers": "t",
    "time_trials_tapers": "",
    "time_tapers": "r",
    "tapers": "tr",
}


def _pooled_cross_spectral_matrix(fourier_coefficients, kept_axes, weights=None):
    """Mean of x x^H over the observation axes not in ``kept_axes``, by einsum.

    ``fourier_coefficients`` has shape (n_time, n_trials, n_tapers, n_freq,
    n_signals), lettered ``t``, ``r``, ``k``; ``weights``, shape
    (n_time, n_trials, n_tapers, n_freq, 1), weight the mean. Returns shape
    (*kept axes, n_freq, n_signals, n_signals).
    """
    weights = np.ones(fourier_coefficients.shape[:-1]) if weights is None else weights[..., 0]
    total = np.einsum(
        f"trkf,trkfa,trkfb->{kept_axes}fab",
        weights,
        fourier_coefficients,
        fourier_coefficients.conj(),
    )
    return total / np.einsum(f"trkf->{kept_axes}f", weights)[..., np.newaxis, np.newaxis]


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
@pytest.mark.parametrize("path", ["batched", "per_bin"])
@pytest.mark.parametrize("expectation_type", sorted(_GLOBAL_COHERENCE_KEPT_AXES))
def test_global_coherence_decomposes_the_matrix_its_expectation_averages(
    expectation_type, path, weighted, monkeypatch
):
    """For each supported expectation_type, the fractions and vectors are the
    eigendecomposition of the cross-spectral matrix averaged over the same
    axes (with the same observation weights), and the kept axes lead the
    result in the order the other measures keep them."""
    import spectral_connectivity._multivariate as multivariate_module

    if path == "per_bin":
        monkeypatch.setattr(multivariate_module, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0)
    rng = np.random.default_rng(31)
    shape = (3, 4, 6, 6, 6)  # (time, trials, tapers, two-sided frequency, signals)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    weights = rng.uniform(0.2, 1.0, (*shape[:-1], 1)) if weighted else None
    conn = Connectivity(
        coefficients, expectation_type=expectation_type, observation_weights=weights
    )
    # Two of six components: the per-bin path takes its truncated svds branch.
    fractions, vectors = conn.global_coherence(max_rank=2)

    kept_axes = _GLOBAL_COHERENCE_KEPT_AXES[expectation_type]
    kept_shape = tuple(shape["trk".index(axis)] for axis in kept_axes)
    n_frequencies = conn.frequencies.size
    assert fractions.shape == (*kept_shape, n_frequencies, 2)
    assert vectors.shape == (*kept_shape, n_frequencies, shape[-1], 2)
    assert fractions.shape[:-1] == conn.magnitude_squared_coherence().shape[:-2]

    expected = _pooled_cross_spectral_matrix(
        _nonnegative_bins(coefficients),
        kept_axes,
        None if weights is None else _nonnegative_bins(weights),
    )
    eigenvalues, eigenvectors = np.linalg.eigh(expected)
    trace = np.trace(expected, axis1=-2, axis2=-1).real[..., np.newaxis]
    np.testing.assert_allclose(
        fractions, eigenvalues[..., ::-1][..., :2] / trace, rtol=1e-10, atol=1e-12
    )
    # The same vectors up to a phase: |<u, v>| = 1 for unit vectors.
    leading = eigenvectors[..., ::-1][..., :2]
    overlap = np.abs(np.sum(leading.conj() * vectors, axis=-2))
    np.testing.assert_allclose(overlap, 1.0, rtol=1e-8)


def test_time_trials_tapers_global_coherence_is_the_chronux_estimand():
    """On non-overlapping windows, "time_trials_tapers" estimates what Chronux
    ``CrossSpecMatc`` returns as ``Ctot``: the cross-spectral matrix ``Sc``
    averaged with equal weight over every taper, trial and window (Chronux
    adds the taper mean of ``conj(J_k) J_l`` for each window and trial and
    divides by ``Nwins * Ntr``), then ``s(1) / sum(s)`` from its singular
    values. ``Sc`` is the conjugate of the matrix decomposed here, so the
    values are the same."""
    time_series = np.random.default_rng(32).standard_normal((1000, 3, 5))
    m = Multitaper(
        time_series,
        sampling_frequency=500,
        time_halfbandwidth_product=3,
        time_window_duration=0.4,
        time_window_step=0.4,
    )
    conn = Connectivity.from_transform(m, expectation_type="time_trials_tapers")
    assert conn.time_bins_are_independent  # the windows do not overlap
    fractions, _ = conn.global_coherence()

    coefficients = _nonnegative_bins(to_host(m.fft()))  # (window, trial, taper, f, signal)
    n_windows, n_trials = coefficients.shape[:2]
    assert n_windows == 5
    summed = np.zeros((*coefficients.shape[3:], coefficients.shape[-1]), dtype=complex)
    for window in range(n_windows):
        for trial in range(n_trials):
            tapered = coefficients[window, trial]  # (taper, frequency, signal)
            summed += np.mean(
                tapered.conj()[..., :, np.newaxis] * tapered[..., np.newaxis, :], 0
            )
    singular_values = np.linalg.svd(summed / (n_windows * n_trials), compute_uv=False)
    np.testing.assert_allclose(
        fractions[..., 0], singular_values[..., 0] / singular_values.sum(axis=-1), rtol=1e-10
    )


@pytest.mark.parametrize("expectation_type", ["time", "trials", "time_trials"])
def test_global_coherence_rejects_an_expectation_that_keeps_the_tapers(expectation_type):
    """A per-taper global coherence has no established meaning, so the
    settings that keep tapers as an output axis raise, naming the supported
    ones and how to rebuild; the rebuilt instance computes it."""
    rng = np.random.default_rng(33)
    coefficients = rng.standard_normal((2, 4, 3, 8, 3)) + 1j * rng.standard_normal(
        (2, 4, 3, 8, 3)
    )
    conn = Connectivity(coefficients, expectation_type=expectation_type)
    with pytest.raises(
        ValueError, match=r"^global_coherence needs the tapers averaged"
    ) as info:
        conn.global_coherence()
    message = str(info.value)
    assert f"expectation_type={expectation_type!r} keeps them as an output axis" in message
    assert (
        "Use one of 'trials_tapers' (the default), 'tapers', 'time_tapers', "
        "'time_trials_tapers'"
    ) in message
    assert (
        "Connectivity.from_transform(transform, expectation_type='trials_tapers')" in message
    )
    conn.expectation_type = "trials_tapers"
    fractions, _ = conn.global_coherence()
    assert fractions.shape == (2, 5, 1)


@pytest.mark.parametrize(
    ("expectation_type", "shape", "pooled"),
    [
        ("trials_tapers", (2, 3, 2, 8, 8), "trials x tapers"),
        ("tapers", (2, 3, 5, 8, 8), "tapers"),
        ("time_tapers", (1, 3, 5, 8, 8), "time x tapers"),
    ],
)
def test_global_coherence_warns_when_fewer_observations_than_signals(
    expectation_type, shape, pooled
):
    """With n pooled observations for p > n signals the matrix has rank at
    most n, so global coherence is at least 1/n whatever the data and cannot
    reach the 1/p of independent signals. That warns, naming n, p and the
    floor, and offers only what raises n."""
    rng = np.random.default_rng(34)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    conn = Connectivity(coefficients, expectation_type=expectation_type)
    n_observations, n_signals = conn.n_observations, conn.n_signals
    assert 1 < n_observations < n_signals
    with pytest.warns(UserWarning, match="fewer observations than signals") as record:
        fractions, _ = conn.global_coherence(max_rank=1)
    (message,) = [
        str(warning.message)
        for warning in record
        if "fewer observations than signals" in str(warning.message)
    ]
    assert (
        f"pools n_observations == {n_observations} observations (the averaged {pooled}) "
        f"for {n_signals} signals"
    ) in message
    assert f"at least 1/{n_observations} = {1 / n_observations:.3g}" in message
    assert f"1/n_signals = {1 / n_signals:.3g}" in message
    assert "Compare values only at matched numbers of observations and signals." in message
    assert "fewer signals" in message
    # Short sentences: none of the statements runs past 30 words (the last,
    # the list of remedies, is exempt).
    *statements, _ = re.split(r"(?<=[a-z0-9)])\. (?=[A-Z])", message)
    assert len(statements) >= 4
    assert max(len(statement.split()) for statement in statements) <= 30
    assert ("more trials" in message) == ("trials" in expectation_type)
    assert ("more time windows" in message) == ("time" in expectation_type)
    # The floor the message states holds for independent noise.
    assert np.all(fractions >= 1 / n_observations - 1e-12)


def test_labelled_global_coherence_with_few_observations_names_the_pooling_route():
    """The wrappers fix expectation_type to "trials_tapers", so their small-n
    warning names the route that pools time windows instead of an
    expectation_type the caller cannot pass, and only when there are windows
    to pool."""
    from spectral_connectivity import multitaper_connectivity

    time_series = np.random.default_rng(37).standard_normal((1000, 1, 8))

    def labelled_message(**window):
        with pytest.warns(UserWarning, match="fewer observations than signals") as record:
            multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                method="global_coherence",
                time_halfbandwidth_product=2,
                **window,
            )
        return next(
            str(w.message) for w in record if "fewer observations" in str(w.message)
        )

    route = (
        "Connectivity.from_transform(transform, "
        "expectation_type='time_trials_tapers').global_coherence() to pool the "
        "time windows"
    )
    message = labelled_message(time_window_duration=0.4, time_window_step=0.4)
    assert route in message
    assert "an expectation_type that averages more axes" not in message
    assert "more trials" in message
    assert route not in labelled_message()  # one window: nothing to pool

    # The route the message names works and pools the five windows.
    multitaper = Multitaper(
        time_series,
        sampling_frequency=500,
        time_halfbandwidth_product=2,
        time_window_duration=0.4,
        time_window_step=0.4,
    )
    pooled = Connectivity.from_transform(multitaper, expectation_type="time_trials_tapers")
    assert pooled.n_observations == 5 * multitaper.n_tapers
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", category=SidednessWarning)
        fractions, _ = pooled.global_coherence()
    assert fractions.shape == (pooled.frequencies.size, 1)


def test_global_coherence_with_as_many_observations_as_signals_does_not_warn():
    """n = p no longer bounds the value away from 1/p, and the default use,
    many trials of few signals, stays silent."""
    rng = np.random.default_rng(35)
    for shape in [(1, 4, 2, 8, 8), (1, 50, 3, 8, 4)]:
        coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", category=SidednessWarning)
            Connectivity(coefficients).global_coherence()


@pytest.mark.parametrize("expectation_type", ["time_trials_tapers", "time_tapers"])
def test_global_coherence_warns_when_it_pools_overlapping_windows(expectation_type):
    """Windows overlapping by more than half share data, so pooling them adds
    fewer effective observations than their count; that warns as the other
    measures that count observations do. Keeping the windows, or pooling
    windows that do not overlap, is silent."""
    time_series = np.random.default_rng(36).standard_normal((1000, 3, 3))

    def connectivity(time_window_step, expectation):
        multitaper = Multitaper(
            time_series,
            sampling_frequency=500,
            time_halfbandwidth_product=3,
            time_window_duration=0.4,
            time_window_step=time_window_step,
        )
        return Connectivity.from_transform(multitaper, expectation_type=expectation)

    overlapping = connectivity(0.1, expectation_type)
    assert not overlapping.time_bins_are_independent
    with pytest.warns(UserWarning, match="global_coherence assumes independent") as record:
        overlapping.global_coherence()
    (message,) = [
        str(warning.message)
        for warning in record
        if "assumes independent" in str(warning.message)
    ]
    assert f"expectation_type={expectation_type!r} averages over time bins" in message
    assert "time_bins_are_independent is False" in message
    assert f"n_observations == {overlapping.n_observations} overstates" in message

    for silent in (connectivity(0.1, "trials_tapers"), connectivity(0.4, expectation_type)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", category=SidednessWarning)
            silent.global_coherence()


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
        conn = Connectivity.from_transform(m)

        # Standard (not relaxed) significance threshold: real coherence must be
        # detected for this to return anything.
        delay, slope, r_value = conn.group_delay(
            frequency_range=(20, 80),
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
        """With no frequency_range, the whole band recovers signed lags.

        ``delay[..., i, j]`` is positive when signal ``i`` leads signal ``j``,
        i.e. ``(lags[j] - lags[i]) / sampling_frequency``.
        """
        sampling_frequency = 200
        lags = np.array([0, 3, 8])
        time_series = simulate_lagged_broadband(
            lags, [0.3, 0.3, 0.3], 200, 5, random_state=self.rng
        )
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)

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
        time_series = simulate_lagged_broadband(
            lags, [0.3, 0.3, 0.3], 200, 5, random_state=self.rng
        )
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)

        delay1, _slope1, _r_value1 = conn.group_delay(
            frequency_range=(10, 90), frequency_resolution=5.0
        )
        delay2, _slope2, _r_value2 = conn.group_delay(
            frequency_range=(10, 90), frequency_resolution=10.0
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
        time_series = simulate_lagged_broadband(
            lags, [0.3, 0.3, 0.8], 200, 5, random_state=self.rng
        )
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)

        delay_strict, _, _ = conn.group_delay(
            frequency_range=(10, 90),
            frequency_resolution=5.0,
            significance_threshold=1e-6,
        )
        delay_loose, _, _ = conn.group_delay(
            frequency_range=(10, 90),
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
        time_series = simulate_lagged_broadband(
            lags, [0.3, 0.3, 0.3], 200, 5, random_state=self.rng
        )
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=2,
        )
        conn = Connectivity.from_transform(m)

        delay, slope, r_value = conn.group_delay(
            frequency_range=(10, 90), frequency_resolution=5.0
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

    def test_pair_without_significant_coherence_is_nan(self):
        """For independent noise no frequency's coherence survives the
        Benjamini-Hochberg-corrected zero-coherence test, so, as documented,
        the pair's delay, slope, and r are NaN in group_delay and every
        candidate in delay is NaN."""
        noise = self.rng.standard_normal((1000, 10, 2))
        conn = Connectivity.from_transform(Multitaper(noise, sampling_frequency=500))
        delay, slope, r_value = conn.group_delay()
        for output in (delay, slope, r_value):
            assert np.isnan(output[..., 0, 1]).all()
            assert np.isnan(output[..., 1, 0]).all()
        assert np.isnan(conn.delay(frequency_range=(5, 50))[..., 0, 1]).all()

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
        n_time, n_trials, n_tapers, n_fft, n_signals = 1, 30, 1, 17, 3
        expected_slope = -0.7
        # One-sided frequency labels offset far from zero relative to their
        # 0.001 spacing (a two-sided spectrum must be in FFT order).
        frequencies = 1e6 + np.arange(n_fft) * 0.001
        # The raw-moment variance genuinely cancels to zero on that grid in
        # float64 (the old failure the centering fixes).
        assert n_fft * (frequencies @ frequencies) - frequencies.sum() ** 2 == 0.0

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
            fourier_coefficients=fourier, frequencies=frequencies, is_one_sided=True
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

        conn = Connectivity.from_transform(
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
            _find_significant_frequencies,
            _get_independent_frequency_step,
        )

        step = _get_independent_frequency_step(frequencies[1] - frequencies[0], None)
        bandpassed, band_frequencies = conn.coherency(), frequencies
        pairs = np.asarray(list(combinations(range(n_signals), 2)))
        bandpassed = bandpassed[..., pairs[:, 0], pairs[:, 1]]
        # The same significance filter as group_delay, which tests the bins
        # whose coefficients are all real (0 Hz and Nyquist) against the real null.
        significant = _find_significant_frequencies(
            bandpassed,
            conn.n_observations,
            step,
            significance_threshold=0.05,
            real_valued=conn.real_valued_frequencies[:, np.newaxis],
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
                # linregress on the reverse pair's phase, -phase, gives -r.
                reference_r[(*lead, j, i)] = -fit[2]

        np.testing.assert_array_equal(np.isnan(slope), np.isnan(reference_slope))
        np.testing.assert_allclose(
            slope, reference_slope, rtol=1e-9, atol=1e-11, equal_nan=True
        )
        np.testing.assert_allclose(r_value, reference_r, rtol=1e-9, atol=1e-11, equal_nan=True)

    def test_group_delay_antisymmetry(self):
        """Group delay is antisymmetric: delay[i, j] == -delay[j, i]."""
        lags = np.array([0, 3, 8])
        time_series = simulate_lagged_broadband(
            lags, [0.3, 0.3, 0.3], 200, 10, random_state=self.rng
        )
        m = Multitaper(
            time_series=time_series,
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_transform(m)

        delay, _slope, _r_value = conn.group_delay(
            frequency_range=(10, 90), frequency_resolution=5.0
        )

        off_diagonal = ~np.eye(lags.size, dtype=bool)
        assert np.isfinite(delay[..., off_diagonal]).all()
        np.testing.assert_allclose(delay, -np.swapaxes(delay, -1, -2), rtol=0, atol=1e-12)

    def test_group_delay_slope_and_r_value_flip_with_the_pair_order(self):
        """Reversing a pair negates its phase, so both the slope and the
        phase-frequency correlation change sign: a decreasing phase has r < 0."""
        time_series = simulate_lagged_broadband(
            [0, 3, 8], [0.3, 0.3, 0.3], 200, 10, random_state=self.rng
        )
        conn = Connectivity.from_transform(
            Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)
        )
        _delay, slope, r_value = conn.group_delay(frequency_range=(10, 90))

        off_diagonal = ~np.eye(3, dtype=bool)
        assert (np.abs(r_value[..., off_diagonal]) > 0.9).all()  # premise: linear phase
        np.testing.assert_array_equal(slope, -np.swapaxes(slope, -1, -2))
        np.testing.assert_array_equal(
            r_value[..., off_diagonal], -np.swapaxes(r_value, -1, -2)[..., off_diagonal]
        )

    def test_undefined_frequency_bin_does_not_invalidate_later_bins(self):
        """A bin with no defined phase (here a zeroed DC bin, 0/0 coherency)
        must not NaN every later bin through the phase unwrapping."""
        sampling_frequency = 1000
        time_series = simulate_lagged_broadband(
            [0, 1], 0.3, n_time_samples=1000, n_trials=30, random_state=self.rng
        )
        multitaper = Multitaper(
            time_series, sampling_frequency=sampling_frequency, time_halfbandwidth_product=3
        )
        coefficients = to_host(multitaper.fft()).copy()
        coefficients[..., 0, :] = 0  # the DC bin of every signal
        frequencies = np.fft.fftfreq(coefficients.shape[-2], 1 / sampling_frequency)
        conn = Connectivity(coefficients, frequencies=frequencies)

        with pytest.warns(UserWarning, match="zero power"):
            delay, _slope, _r_value = conn.group_delay()
        # Signal 0 leads signal 1 by one sample, 1 ms.
        assert abs(delay[0, 0, 1] - 1 / sampling_frequency) < 1e-4

        with pytest.warns(UserWarning, match="zero power"):
            possible_delays = conn.delay(frequency_range=(20, 200), n_range=1)
        zero_wrap = possible_delays[0, :, 1, 0, 1]
        assert np.isfinite(zero_wrap).mean() > 0.9
        assert abs(np.nanmedian(zero_wrap) - 1 / sampling_frequency) < 1e-4


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
        time_series = simulate_lagged_broadband(
            lags, [0.3] * n_signals, n_time, n_trials, random_state=self.rng
        )

        m = Multitaper(
            time_series=time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=3,
        )
        conn = Connectivity.from_transform(m)
        fourier_coefficients = to_host(m.fft())

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
        assert global_coh.shape == (m.time.size, n_non_negative_freqs, 2)
        assert global_coh_vectors.shape == (
            m.time.size,
            n_non_negative_freqs,
            n_signals,
            2,
        )
        np.testing.assert_allclose(
            global_coh,
            _global_coherence_oracle(fourier_coefficients[..., :n_non_negative_freqs, :])[
                ..., :2
            ],
            rtol=1e-10,
            atol=1e-12,
        )
        # One shared source: the leading component dominates at low frequencies,
        # where the lags barely rotate the phases.
        low = conn.frequencies <= 50
        assert np.all(global_coh[:, low, 0] > 0.8)

        # Group delay recovers the signed lags: [i, j] = (lags[j] - lags[i]) / fs.
        delay, slope, r_value = conn.group_delay(
            frequency_range=(10, 100),
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
        conn = Connectivity.from_transform(m)

        coherence = conn.magnitude_squared_coherence()
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

    def test_reverse_delay_candidates_follow_the_formula(self):
        """Candidate k of the reverse pair is (-phase + 2 pi k) / (2 pi f), which
        is minus the forward pair's candidate -k."""
        time_series = simulate_lagged_broadband(
            [0, 3], 0.3, n_time_samples=1000, n_trials=20, random_state=self.rng
        )
        conn = Connectivity.from_transform(
            Multitaper(time_series, sampling_frequency=500, time_halfbandwidth_product=3)
        )
        n_range = 2
        delays = conn.delay(frequency_range=(20, 100), n_range=n_range)
        forward, reverse = delays[..., 0, 1], delays[..., 1, 0]
        assert np.isfinite(forward).any()
        np.testing.assert_array_equal(reverse, -forward[..., ::-1])
        # Spelled out for k = +1: the reverse pair's candidate adds one cycle.
        frequencies = conn.frequencies[(conn.frequencies >= 20) & (conn.frequencies <= 100)]
        reverse_phase = -forward[0, :, n_range] * 2 * np.pi * frequencies
        np.testing.assert_allclose(
            reverse[0, :, n_range + 1],
            (reverse_phase + 2 * np.pi) / (2 * np.pi * frequencies),
            rtol=1e-12,
        )

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
        conn = Connectivity.from_transform(m)
        n_range = 3
        possible_delays = conn.delay(
            frequency_range=(20, 80),
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


def test_global_coherence_thin_rank_deficient_bin_has_zero_power_component():
    """Thin path (fewer estimates than signals) with a repeated estimate.

    Duplicating one trial makes every bin's coefficient matrix rank 3 of 4
    estimates, so the weakest component carries (numerically) zero power: its
    fraction is ~0 (not NaN) and, having no resolvable direction, its vector
    is zero, while the strong components still match the singular value
    decomposition and have unit-norm vectors.
    """
    rng = np.random.default_rng(3)
    shape = (1, 3, 1, 6, 8)  # 3 estimates for 8 signals
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    coefficients = np.concatenate([coefficients, coefficients[:, :1]], axis=1)  # 4 estimates
    with pytest.warns(UserWarning, match="fewer observations than signals"):
        fractions, vectors = Connectivity(coefficients).global_coherence(max_rank=4)

    observations = _observations(_nonnegative_bins(coefficients))
    singular_values = np.linalg.svd(observations, compute_uv=False)
    expected = singular_values**2 / np.sum(singular_values**2, axis=-1, keepdims=True)
    np.testing.assert_allclose(fractions, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(vectors[..., 3], 0)
    np.testing.assert_allclose(np.linalg.norm(vectors[..., :3], axis=-2), 1.0, rtol=1e-12)
    # Directions, not just norms: each strong vector is the SVD's left singular
    # vector up to a phase (|<u, v>| = 1 for unit vectors).
    left_vectors = np.linalg.svd(observations, full_matrices=False)[0]
    overlap = np.abs(np.sum(left_vectors[..., :3].conj() * vectors[..., :3], axis=-2))
    np.testing.assert_allclose(overlap, 1.0, rtol=1e-10)
