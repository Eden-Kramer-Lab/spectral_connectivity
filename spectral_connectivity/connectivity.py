from functools import lru_cache, partial, wraps
from inspect import signature
from itertools import combinations

import numpy as np
from scipy.fftpack import ifft
from scipy.ndimage import label
from scipy.sparse.linalg import svds
from scipy.stats.mstats import linregress

from .minimum_phase_decomposition import minimum_phase_decomposition
from .statistics import (adjust_for_multiple_comparisons, coherence_bias,
                         fisher_z_transform, get_normal_distribution_p_values)

EXPECTATION = {
    'time': partial(np.mean, axis=0),
    'trials': partial(np.mean, axis=1),
    'tapers': partial(np.mean, axis=2),
    'time_trials': partial(np.mean, axis=(0, 1)),
    'time_tapers': partial(np.mean, axis=(0, 2)),
    'trials_tapers': partial(np.mean, axis=(1, 2)),
    'time_trials_tapers': partial(np.mean, axis=(1, 2, 3)),
}


def non_negative_frequencies(axis):
    '''Decorator that removes the negative frequencies.'''
    def decorator(connectivity_measure):
        @wraps(connectivity_measure)
        def wrapper(*args, **kwargs):
            measure = connectivity_measure(*args, **kwargs)
            if measure is not None:
                n_frequencies = measure.shape[axis]
                non_neg_index = np.arange(0, (n_frequencies + 1) // 2)
                return np.take(measure, indices=non_neg_index, axis=axis)
            else:
                return None
        return wrapper
    return decorator


class Connectivity:
    '''Computes brain connectivity measures based on the cross spectral
    matrix.

    Spectral granger methods that require estimation of transfer function
    and noise covariance use minimum phase decomposition [1] to decompose
    the cross spectral matrix into square roots, which then can be used to
    non-parametrically estimate the transfer function and noise covariance.

    Attributes
    ----------
    fourier_coefficients : array, shape (n_time_windows, n_trials,
                                         n_tapers, n_fft_samples,
                                         n_signals)
        The compex-valued coefficients from a fourier transform. Note that
        this is expected to be the two-sided fourier coefficients
        (both the positive and negative lags). This is needed for the
        Granger-based methods to work.
    expectation_type : ('trials_tapers' | 'trials' | 'tapers')
        How to average the cross spectral matrix. 'trials_tapers' averages
        over the trials and tapers dimensions. 'trials' only averages over
        the trials dimensions (leaving tapers) and 'tapers' only averages
        over tapers (leaving trials).
    frequencies : array, shape (n_fft_samples,)
    time : array, shape (n_time_windows,)

    Methods
    -------
    coherency
    coherence_magnitude
    coherence_phase
    canonical_coherence
    imaginary_coherence
    phase_locking_value
    phase_lag_index
    weighted_phase_lag_index
    debiased_squared_phase_lag_index
    debiased_squared_weighted_phase_lag_index
    pairwise_phase_consistency
    directed_transfer_function
    directed_coherence
    partial_directed_coherence
    generalized_partial_directed_coherence
    direct_directed_transfer_function
    group_delay
    phase_lag_index
    pairwise_spectral_granger_prediction
    conditional_spectral_granger_prediction (Not implemented)
    blockwise_spectral_granger_prediction (Not implemented)

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    '''

    def __init__(self, fourier_coefficients,
                 expectation_type='trials_tapers', frequencies=None,
                 time=None):
        self.fourier_coefficients = fourier_coefficients
        self.expectation_type = expectation_type
        self._frequencies = frequencies
        self.time = time

    @classmethod
    def from_multitaper(cls, multitaper_instance,
                        expectation_type='trials_tapers'):
        '''Construct connectivity class using a multitaper instance'''
        return cls(
            fourier_coefficients=multitaper_instance.fft(),
            expectation_type=expectation_type,
            time=multitaper_instance.time,
            frequencies=multitaper_instance.frequencies
        )

    @property
    @non_negative_frequencies(axis=0)
    def frequencies(self):
        if self._frequencies is not None:
            return self._frequencies

    @property
    @lru_cache(maxsize=1)
    def _power(self):
        return self._expectation(
            self.fourier_coefficients *
            self.fourier_coefficients.conjugate()).real

    @property
    @lru_cache(maxsize=1)
    def _cross_spectral_matrix(self):
        '''The complex-valued linear association between fourier
        coefficients at each frequency.

        Returns
        -------
        cross_spectral_matrix : array, shape (n_time_windows, n_trials,
                                              n_tapers, n_fft_samples,
                                              n_signals, n_signals)

        '''
        fourier_coefficients = self.fourier_coefficients[..., np.newaxis]
        return _complex_inner_product(fourier_coefficients,
                                      fourier_coefficients)

    @property
    @lru_cache(maxsize=1)
    def _minimum_phase_factor(self):
        return minimum_phase_decomposition(
            self._expectation(self._cross_spectral_matrix))

    @property
    @lru_cache(maxsize=1)
    @non_negative_frequencies(axis=-3)
    def _transfer_function(self):
        return _estimate_transfer_function(self._minimum_phase_factor)

    @property
    @lru_cache(maxsize=1)
    def _noise_covariance(self):
        return _estimate_noise_covariance(self._minimum_phase_factor)

    @property
    @lru_cache(maxsize=1)
    def _MVAR_Fourier_coefficients(self):
        return np.linalg.inv(self._transfer_function)

    @property
    def _expectation(self):
        return EXPECTATION[self.expectation_type]

    @property
    def n_observations(self):
        axes = signature(self._expectation).parameters['axis'].default
        if isinstance(axes, int):
            return self.fourier_coefficients.shape[axes]
        else:
            return np.prod(
                [self.fourier_coefficients.shape[axis]
                 for axis in axes])

    @non_negative_frequencies(axis=-2)
    def power(self):
        return self._power

    @non_negative_frequencies(axis=-3)
    def coherency(self):
        '''The complex-valued linear association between time series in the
         frequency domain.

         Returns
         -------
         complex_coherency : array, shape (..., n_fft_samples, n_signals,
                                           n_signals)

         '''
        norm = np.sqrt(self._power[..., :, np.newaxis] *
                       self._power[..., np.newaxis, :])
        norm[norm == 0] = np.nan
        complex_coherencey = (
            self._expectation(self._cross_spectral_matrix) / norm)
        n_signals = self.fourier_coefficients.shape[-1]
        diagonal_ind = np.arange(0, n_signals)
        complex_coherencey[..., diagonal_ind, diagonal_ind] = np.nan
        return complex_coherencey

    def coherence_phase(self):
        '''The phase angle of the complex coherency.

        Returns
        -------
        phase : array, shape (..., n_fft_samples, n_signals, n_signals)

        '''
        return np.angle(self.coherency())

    def coherence_magnitude(self):
        '''The magnitude of the complex coherency.

        Note that this is not the magnitude squared coherence.

        Returns
        -------
        magnitude : array, shape (..., n_fft_samples, n_signals, n_signals)

        '''
        return _squared_magnitude(self.coherency())

    @non_negative_frequencies(axis=-3)
    def imaginary_coherence(self):
        '''The normalized imaginary component of the cross-spectrum.

        Projects the cross-spectrum onto the imaginary axis to mitigate the
        effect of volume-conducted dependencies. Assumes volume-conducted
        sources arrive at sensors at the same time, resulting in
        a cross-spectrum with phase angle of 0 (perfectly in-phase) or \pi
        (anti-phase) if the sensors are on opposite sides of a dipole
        source. With the imaginary coherence, in-phase and anti-phase
        associations are set to zero.

        Returns
        -------
        imaginary_coherence_magnitude : array, shape (..., n_fft_samples,
                                                      n_signals, n_signals)

        References
        ----------
        .. [1] Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., and
               Hallett, M. (2004). Identifying true brain interaction from
               EEG data using the imaginary part of coherency. Clinical
               Neurophysiology 115, 2292-2307.

        '''
        return np.abs(
            self._expectation(self._cross_spectral_matrix).imag /
            np.sqrt(self._power[..., :, np.newaxis] *
                    self._power[..., np.newaxis, :]))

    def canonical_coherence(self, group_labels):
        '''Finds the maximal coherence between each combination of groups.

        The canonical coherence finds two sets of weights such that the
        coherence between the linear combination of group1 and the linear
        combination of group2 is maximized.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Links each signal to a group.

        Returns
        -------
        canonical_coherence : array, shape (n_time_samples, n_fft_samples,
                                            n_groups, n_groups)
            The maximimal coherence for each group pair
        labels : array, shape (n_groups,)
            The sorted unique group labels that correspond to `n_groups`

        References
        ----------
        .. [1] Stephen, E.P. (2015). Characterizing dynamically evolving
               functional networks in humans with application to speech.
               Boston University.

        '''
        labels = np.unique(group_labels)
        n_frequencies = self.fourier_coefficients.shape[-2]
        non_negative_frequencies = np.arange(0, (n_frequencies + 1) // 2)
        fourier_coefficients = self.fourier_coefficients[
            ..., non_negative_frequencies, :]
        normalized_fourier_coefficients = [
            _normalize_fourier_coefficients(
                fourier_coefficients[..., np.in1d(group_labels, label)])
            for label in labels]

        n_groups = len(labels)
        new_shape = (self.time.size, self.frequencies.size, n_groups,
                     n_groups)
        magnitude = _squared_magnitude(np.stack([
            _estimate_canonical_coherence(
                fourier_coefficients1, fourier_coefficients2)
            for fourier_coefficients1, fourier_coefficients2
            in combinations(normalized_fourier_coefficients, 2)
        ], axis=-1))

        canonical_coherence_magnitude = np.full(new_shape, np.nan)
        group_combination_ind = np.array(
            list(combinations(np.arange(n_groups), 2)))
        canonical_coherence_magnitude[
            ..., group_combination_ind[:, 0],
            group_combination_ind[:, 1]] = magnitude
        canonical_coherence_magnitude[
            ..., group_combination_ind[:, 1],
            group_combination_ind[:, 0]] = magnitude

        return canonical_coherence_magnitude, labels

    def global_coherence(self, max_rank=1):
        '''The linear combinations of signals that capture the most coherent
        power at each frequency and time window.

        This is a frequency domain analog of PCA over signals at a given
        frequency/time window.

        Parameters
        ----------
        max_rank : int, optional
            The number of components to keep (like the number of PC dimensions)

        Returns
        -------
        global_coherence : ndarray, shape (n_time_windows,
                                           n_fft_samples,
                                           n_components)
            The vector of global coherences (square of the singular values)
        unnormalized_global_coherence : ndarray, shape (n_time_windows,
                                                        n_fft_samples,
                                                        n_signals,
                                                        n_components)
            The (unnormalized) global coherence vectors

        References
        ----------
        .. [1] Cimenser, A., Purdon, P.L., Pierce, E.T., Walsh, J.L.,
               Salazar-Gomez, A.F., Harrell, P.G., Tavares-Stoeckel, C.,
               Habeeb, K., and Brown, E.N. (2011). Tracking brain states under
               general anesthesia by using global coherence analysis.
               Proceedings of the National Academy of Sciences 108, 8832–8837.

        '''
        (n_time_windows, n_trials, n_tapers,
         n_fft_samples, n_signals) = self.fourier_coefficients.shape

        # S - singular values
        global_coherence = np.zeros((n_time_windows, n_fft_samples, max_rank))
        # U - rotation
        unnormalized_global_coherence = np.zeros(
            (n_time_windows, n_fft_samples, n_signals, max_rank),
            dtype=np.complex128)

        for time_ind in range(n_time_windows):
            for freq_ind in range(n_fft_samples):
                # reshape to (n_signals, n_trials * n_tapers)
                fourier_coefficients = (
                    self.fourier_coefficients[time_ind, :, :, freq_ind, :]
                    .reshape((n_trials * n_tapers, n_signals))
                    .T
                )

                (global_coherence[time_ind, freq_ind],
                 unnormalized_global_coherence[time_ind, freq_ind]
                 ) = _estimate_global_coherence(fourier_coefficients,
                                                max_rank=max_rank)

        return global_coherence, unnormalized_global_coherence

    @non_negative_frequencies(axis=-3)
    def phase_locking_value(self):
        '''The cross-spectrum with the power for each signal scaled to
        a magnitude of 1.

        The phase locking value attempts to mitigate power differences
        between realizations (tapers or trials) by treating all values of
        the cross-spectrum as the same power. This has the effect of
        downweighting high power realizations and upweighting low power
        realizations.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals,
                                            n_signals)

        References
        ----------
        .. [1] Lachaux, J.-P., Rodriguez, E., Martinerie, J., Varela, F.J.,
               and others (1999). Measuring phase synchrony in brain
               signals. Human Brain Mapping 8, 194-208.

        '''
        return self._expectation(
            self._cross_spectral_matrix /
            np.abs(self._cross_spectral_matrix))

    @non_negative_frequencies(axis=-3)
    def phase_lag_index(self):
        '''A non-parametric synchrony measure designed to mitigate power
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
        phase_lag_index : array, shape (..., n_fft_samples, n_signals,
                                        n_signals)

        References
        ----------
        .. [1] Stam, C.J., Nolte, G., and Daffertshofer, A. (2007). Phase
               lag index: Assessment of functional connectivity from multi
               channel EEG and MEG with diminished bias from common
               sources. Human Brain Mapping 28, 1178-1193.

        '''
        return self._expectation(np.sign(self._cross_spectral_matrix.imag))

    @non_negative_frequencies(-3)
    def weighted_phase_lag_index(self):
        '''Weighted average of the phase lag index using the imaginary
        coherency magnitudes as weights.

        Returns
        -------
        weighted_phase_lag_index : array, shape (..., n_fft_samples,
                                                 n_signals, n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        weights = self._expectation(
            np.abs(self._cross_spectral_matrix.imag))
        weights[weights < np.finfo(float).eps] = 1
        return self._expectation(
            self._cross_spectral_matrix.imag) / weights

    def debiased_squared_phase_lag_index(self):
        '''The square of the phase lag index corrected for the positive
        bias induced by using the magnitude of the complex cross-spectrum.

        Returns
        -------
        phase_lag_index : array, shape (..., n_fft_samples, n_signals,
                                        n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        n_observations = self.n_observations
        return ((n_observations * self.phase_lag_index() ** 2 - 1.0) /
                (n_observations - 1.0))

    @non_negative_frequencies(-3)
    def debiased_squared_weighted_phase_lag_index(self):
        '''The square of the weighted phase lag index corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        weighted_phase_lag_index : array, shape (..., n_fft_samples,
                                                 n_signals, n_signals)

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        '''
        n_observations = self.n_observations
        imaginary_cross_spectral_matrix_sum = self._expectation(
            self._cross_spectral_matrix.imag) * n_observations
        squared_imaginary_cross_spectral_matrix_sum = self._expectation(
            self._cross_spectral_matrix.imag ** 2) * n_observations
        imaginary_cross_spectral_matrix_magnitude_sum = self._expectation(
            np.abs(self._cross_spectral_matrix.imag)) * n_observations
        weights = (imaginary_cross_spectral_matrix_magnitude_sum ** 2 -
                   squared_imaginary_cross_spectral_matrix_sum)
        weights[weights == 0] = np.nan
        return (imaginary_cross_spectral_matrix_sum ** 2 -
                squared_imaginary_cross_spectral_matrix_sum) / weights

    def pairwise_phase_consistency(self):
        '''The square of the phase locking value corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        phase_locking_value : array, shape (..., n_fft_samples, n_signals,
                                            n_signals)

        References
        ----------
        .. [1] Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., and
               Pennartz, C.M.A. (2010). The pairwise phase consistency: A
               bias-free measure of rhythmic neuronal synchronization.
               NeuroImage 51, 112-122.

        '''
        n_observations = self.n_observations
        plv_sum = self.phase_locking_value() * n_observations
        ppc = ((plv_sum * plv_sum.conjugate() - n_observations) /
               (n_observations ** 2 - n_observations))
        return ppc.real

    def pairwise_spectral_granger_prediction(self):
        '''The amount of power at a node in a frequency explained by (is
        predictive of) the power at other nodes.

        Also known as spectral granger causality.

        References
        ----------
        .. [1] Geweke, J. (1982). Measurement of Linear Dependence and
               Feedback Between Multiple Time Series. Journal of the
               American Statistical Association 77, 304.

        '''
        cross_spectral_matrix = self._expectation(
            self._cross_spectral_matrix)
        n_signals = cross_spectral_matrix.shape[-1]
        total_power = self.power()
        n_frequencies = cross_spectral_matrix.shape[-3]
        non_neg_index = np.arange(0, (n_frequencies + 1) // 2)
        new_shape = list(cross_spectral_matrix.shape)
        new_shape[-3] = non_neg_index.size
        predictive_power = np.empty(new_shape)

        for pair_indices in combinations(range(n_signals), 2):
            pair_indices = np.array(pair_indices)[:, np.newaxis]
            try:
                minimum_phase_factor = (
                    minimum_phase_decomposition(
                        cross_spectral_matrix[
                            ..., pair_indices, pair_indices.T]))
                transfer_function = _estimate_transfer_function(
                    minimum_phase_factor)[..., non_neg_index, :, :]
                rotated_covariance = _remove_instantaneous_causality(
                    _estimate_noise_covariance(minimum_phase_factor))
                predictive_power[..., pair_indices, pair_indices.T] = (
                    _estimate_predictive_power(
                        total_power[..., pair_indices[:, 0]],
                        rotated_covariance, transfer_function))
            except np.linalg.LinAlgError:
                predictive_power[
                    ..., pair_indices, pair_indices.T] = np.nan

        diagonal_ind = np.diag_indices(n_signals)
        predictive_power[..., diagonal_ind[0], diagonal_ind[1]] = np.nan

        return predictive_power

    def conditional_spectral_granger_prediction(self):
        raise NotImplementedError

    def blockwise_spectral_granger_prediction(self):
        raise NotImplementedError

    def directed_transfer_function(self):
        '''The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        Characterizes the direct and indirect coupling to a node.

        Returns
        -------
        directed_transfer_function : array, shape (..., n_fft_samples,
                                                   n_signals, n_signals)

        References
        ----------
        .. [1] Kaminski, M., and Blinowska, K.J. (1991). A new method of
               the description of the information flow in the brain
               structures. Biological Cybernetics 65, 203-210.

        '''
        return _squared_magnitude(
            self._transfer_function /
            _total_inflow(self._transfer_function))

    def directed_coherence(self):
        '''The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        This measure is the same as the directed transfer function but the
        signal inflow is scaled by the noise variance.

        Returns
        -------
        directed_coherence : array, shape (..., n_fft_samples,
                                           n_signals, n_signals)

        References
        ----------
        .. [1] Baccala, L., Sameshima, K., Ballester, G., Do Valle, A., and
               Timo-Iaria, C. (1998). Studying the interaction between
               brain structures via directed coherence and Granger
               causality. Applied Signal Processing 5, 40.

        '''
        noise_variance = _get_noise_variance(self._noise_covariance)
        return (np.sqrt(noise_variance) *
                _squared_magnitude(self._transfer_function) /
                _total_inflow(self._transfer_function, noise_variance))

    def partial_directed_coherence(self):
        '''The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        Returns
        -------
        partial_directed_coherence : array, shape (..., n_fft_samples,
                                                   n_signals, n_signals)

        References
        ----------
        .. [1] Baccala, L.A., and Sameshima, K. (2001). Partial directed
               coherence: a new concept in neural structure determination.
               Biological Cybernetics 84, 463-474.

        '''
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients /
            _total_outflow(self._MVAR_Fourier_coefficients))

    def generalized_partial_directed_coherence(self):
        '''The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        The generalized partial directed coherence scales the relative
        strength of coupling by the noise variance.

        Returns
        -------
        generalized_partial_directed_coherence : array,
                                                 shape (..., n_fft_samples,
                                                        n_signals,
                                                        n_signals)

        References
        ----------
        .. [1] Baccala, L.A., Sameshima, K., and Takahashi, D.Y. (2007).
               Generalized partial directed coherence. In Digital Signal
               Processing, 2007 15th International Conference on, (IEEE),
               pp. 163-166.

        '''
        noise_variance = _get_noise_variance(self._noise_covariance)
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients /
            np.sqrt(noise_variance) / _total_outflow(
                self._MVAR_Fourier_coefficients, noise_variance))

    def direct_directed_transfer_function(self):
        '''A combination of the directed transfer function estimate of
        directional influence between signals and the partial coherence's
        accounting for the influence of other signals.

        Returns
        -------
        direct_directed_transfer_function : array, shape
                                            (..., n_fft_samples,
                                             n_signals, n_signals)

        References
        ----------
        .. [1] Korzeniewska, A., Manczak, M., Kaminski,
               M., Blinowska, K.J., and Kasicki, S. (2003). Determination
               of information flow direction among brain structures by a
               modified directed transfer function (dDTF) method.
               Journal of Neuroscience Methods 125, 195-207.

        '''
        full_frequency_DTF = (
            self._transfer_function /
            _total_inflow(self._transfer_function, axis=(-1, -3)))
        return (np.abs(full_frequency_DTF) *
                np.sqrt(self.partial_directed_coherence()))

    def group_delay(self, frequencies_of_interest=None,
                    frequency_resolution=None,
                    significance_threshold=0.05):
        '''The average time-delay of a broadband signal.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,)
        frequencies : array-like, shape (n_fft_samples,)
        frequency_resolution : float

        Returns
        -------
        delay : array, shape (..., n_signals, n_signals)
        slope : array, shape (..., n_signals, n_signals)
        r_value : array, shape (..., n_signals, n_signals)

        References
        ----------
        .. [1] Gotman, J. (1983). Measurement of small time differences
               between EEG channels: method and application to epileptic
               seizure propagation. Electroencephalography and Clinical
               Neurophysiology 56, 501-514.

        '''
        frequencies = self.frequencies
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)
        bias = coherence_bias(self.n_observations)

        n_signals = bandpassed_coherency.shape[-1]
        signal_combination_ind = np.array(
            list(combinations(np.arange(n_signals), 2)))
        bandpassed_coherency = bandpassed_coherency[
            ..., signal_combination_ind[:, 0],
            signal_combination_ind[:, 1]]

        is_significant = _find_significant_frequencies(
            bandpassed_coherency, bias, independent_frequency_step,
            significance_threshold=significance_threshold)
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-2),
            mask=~is_significant)

        def _linear_regression(response):
            return linregress(bandpassed_frequencies, y=response)

        regression_results = np.ma.apply_along_axis(
            _linear_regression, -2, coherence_phase)
        new_shape = (
            *bandpassed_coherency.shape[:-2], n_signals, n_signals)
        slope = np.full(new_shape, np.nan)
        slope[..., signal_combination_ind[:, 0],
              signal_combination_ind[:, 1]] = np.array(
            regression_results[..., 0, :], dtype=np.float)
        slope[..., signal_combination_ind[:, 1],
              signal_combination_ind[:, 0]] = -1 * np.array(
            regression_results[..., 0, :], dtype=np.float)

        delay = slope / (2 * np.pi)

        r_value = np.ones(new_shape)
        r_value[..., signal_combination_ind[:, 0],
                signal_combination_ind[:, 1]] = np.array(
            regression_results[..., 2, :], dtype=np.float)
        r_value[..., signal_combination_ind[:, 1],
                signal_combination_ind[:, 0]] = np.array(
            regression_results[..., 2, :], dtype=np.float)
        return delay, slope, r_value

    def delay(self, frequencies_of_interest=None,
              frequency_resolution=None,
              significance_threshold=0.05, n_range=3):
        '''Find a range of possible delays from the coherence phase.

        The delay (and phase) at each frequency is indistinguishable from
        2 \pi phase jumps, but we can look at a range of possible delays
        and see which one is most likely.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,)
        frequencies : array-like, shape (n_fft_samples,)
        frequency_resolution : float
        n_range : int
            Number of phases to consider.

        Returns
        -------
        possible_delays : array, shape (..., n_frequencies,
                                        (n_range * 2) + 1, n_signals,
                                        n_signals)

        '''
        frequencies = self.frequencies
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)
        bias = coherence_bias(self.n_observations)
        n_signals = bandpassed_coherency.shape[-1]
        signal_combination_ind = np.array(
            list(combinations(np.arange(n_signals), 2)))
        bandpassed_coherency = bandpassed_coherency[
            ..., signal_combination_ind[:, 0],
            signal_combination_ind[:, 1]]

        is_significant = _find_significant_frequencies(
            bandpassed_coherency, bias, independent_frequency_step,
            significance_threshold=significance_threshold)
        coherence_phase = np.ma.masked_array(
            np.unwrap(np.angle(bandpassed_coherency), axis=-2),
            mask=~is_significant)
        possible_range = 2 * np.pi * np.arange(-n_range, n_range + 1)
        delays = np.rollaxis((
            possible_range + coherence_phase[..., np.newaxis]) /
            (2 * np.pi), -1, -2)
        new_shape = (
            *bandpassed_coherency.shape[:-1], len(possible_range),
            n_signals, n_signals)
        possible_delays = np.full(new_shape, np.nan)
        possible_delays[..., signal_combination_ind[:, 0],
                        signal_combination_ind[:, 1]] = delays
        possible_delays[..., signal_combination_ind[:, 1],
                        signal_combination_ind[:, 0]] = -delays

        return possible_delays

    def phase_slope_index(self, frequencies_of_interest=None,
                          frequency_resolution=None):
        '''The weighted average of slopes of a broadband signal projected
        onto the imaginary axis.

        The phase slope index finds the complex weighted average of the
        coherency between frequencies where the weights correspond to the
        magnitude of the coherency at that frequency. This is projected
        on to the imaginary axis to avoid volume conduction effects.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,)
        frequencies : array-like, shape (n_fft_samples,)
        frequency_resolution : float

        Returns
        -------
        phase_slope_index : array, shape (..., n_signals, n_signals)

        References
        ----------
        .. [1] Nolte, G., Ziehe, A., Nikulin, V.V., Schlogl, A., Kramer,
               N., Brismar, T., and Muller, K.-R. (2008). Robustly
               Estimating the Flow Direction of Information in Complex
               Physical Systems. Physical Review Letters 100.

        '''
        frequencies = self.frequencies
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self.coherency(), frequencies, frequencies_of_interest)

        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution)
        frequency_index = np.arange(0, bandpassed_frequencies.shape[0],
                                    independent_frequency_step)
        bandpassed_coherency = bandpassed_coherency[
            ..., frequency_index, :, :]

        return _inner_combination(bandpassed_coherency).imag


def _inner_combination(data, axis=-3):
    '''Takes the inner product of all possible pairs of a
    dimension without regard to order (combinations)'''
    combination_index = np.array(
        list(combinations(range(data.shape[axis]), 2)))
    combination_slice1 = np.take(data, combination_index[:, 0], axis)
    combination_slice2 = np.take(data, combination_index[:, 1], axis)
    return (combination_slice1.conjugate() * combination_slice2).sum(
        axis=axis)


def _estimate_noise_covariance(minimum_phase):
    '''Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the noise covariance
    of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples,
                                  n_signals, n_signals)
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

    '''
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    return _complex_inner_product(
        inverse_fourier_coefficients[..., 0, :, :],
        inverse_fourier_coefficients[..., 0, :, :]).real


def _estimate_transfer_function(minimum_phase):
    '''Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the transfer
    function of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples,
                                  n_signals, n_signals)
        The matrix square root of a cross spectral matrix.

    Returns
    -------
    transfer_function : array, shape (n_time_windows, n_fft_samples,
                                      n_signals, n_signals)
        The transfer function of a MVAR model.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    '''
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    return np.matmul(
        minimum_phase,
        np.linalg.inv(inverse_fourier_coefficients[..., 0:1, :, :]))


def _estimate_predictive_power(total_power, rotated_covariance,
                               transfer_function):
    intrinsic_power = (total_power[..., np.newaxis] -
                       rotated_covariance[..., np.newaxis, :, :] *
                       _squared_magnitude(transfer_function))
    intrinsic_power[intrinsic_power == 0] = np.finfo(float).eps
    predictive_power = (
        np.log(total_power[..., np.newaxis]) - np.log(intrinsic_power))
    predictive_power[predictive_power <= 0] = np.nan
    return predictive_power


def _squared_magnitude(x):
    return np.abs(x) ** 2


def _complex_inner_product(a, b):
    '''Measures the orthogonality (similarity) of complex arrays in
    the last two dimensions.'''
    return np.matmul(a, _conjugate_transpose(b))


def _remove_instantaneous_causality(noise_covariance):
    '''Rotates the noise covariance so that the effect of instantaneous
    signals (like those caused by volume conduction) are removed.

    x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    y -> x: var(y) - (cov(x,y) ** 2 / var(x))

    Parameters
    ----------
    noise_covariance : array, shape (..., n_signals, n_signals)

    Returns
    -------
    rotated_noise_covariance : array, shape (..., n_signals, n_signals)
        The noise covariance without the instantaneous causality effects.

    '''
    variance = np.diagonal(noise_covariance, axis1=-1,
                           axis2=-2)[..., np.newaxis]
    return (variance.swapaxes(-1, -2) - noise_covariance ** 2 / variance)


def _set_diagonal_to_zero(x):
    '''Sets the diaginal of the last two dimensions to zero.'''
    n_signals = x.shape[-1]
    diagonal_index = np.diag_indices(n_signals)
    x[..., diagonal_index[0], diagonal_index[1]] = 0
    return x


def _total_inflow(transfer_function, noise_variance=1.0, axis=-1):
    '''Measures the effect of incoming signals onto a node via sum of
    squares.'''
    return np.sqrt(np.sum(
        noise_variance * _squared_magnitude(transfer_function),
        keepdims=True, axis=axis))


def _get_noise_variance(noise_covariance):
    '''Extracts the noise variance from the noise covariance matrix.'''
    return np.diagonal(noise_covariance, axis1=-1, axis2=-2)[
        ..., np.newaxis, :, np.newaxis]


def _total_outflow(MVAR_Fourier_coefficients, noise_variance=1.0):
    '''Measures the effect of outgoing signals on the node via
    sum of squares.'''
    return np.sqrt(np.sum(
        _squared_magnitude(MVAR_Fourier_coefficients) / noise_variance,
        keepdims=True, axis=-2))


def _reshape(fourier_coefficients):
    '''Combine trials and tapers dimensions and move the combined dimension
    to the last axis position.

    Parameters
    ----------
    fourier_coefficients : array, shape (n_time_windows, n_trials,
                                         n_tapers, n_fft_samples,
                                         n_signals)

    Returns
    -------
    fourier_coefficients : array, shape (n_time_windows, n_fft_samples,
                                         n_signals, n_trials * n_tapers)

    '''
    (n_time_windows, _, _, n_fft_samples,
     n_signals) = fourier_coefficients.shape
    new_shape = (n_time_windows, -1, n_fft_samples, n_signals)
    return np.moveaxis(fourier_coefficients.reshape(new_shape), 1, -1)


def _normalize_fourier_coefficients(fourier_coefficients):
    '''Normalizes a group of fourier coefficients by power within group

    Parameters
    ----------
    fourier_coefficients : array, shape (n_time_windows, n_trials,
                                         n_tapers, n_fft_samples,
                                         n_signals)

    Returns
    -------
    normalized_fourier_coefficients : array, shape (n_time_windows,
                                                    n_fft_samples,
                                                    n_signals,
                                                    n_trials * n_tapers)
    '''
    U, _, V_transpose = np.linalg.svd(
        _reshape(fourier_coefficients), full_matrices=False)
    return np.matmul(U, V_transpose)


def _estimate_canonical_coherence(normalized_fourier_coefficients1,
                                  normalized_fourier_coefficients2):
    '''Finds the maximum complex correlation between groups of signals
    at each time and frequency.

    Parameters
    ----------
    normalized_fourier_coefficients1 : array, shape (n_time_windows,
                                                     n_fft_samples,
                                                     n_signals,
                                                     n_trials * n_tapers)
    normalized_fourier_coefficients2 : array, shape (n_time_windows,
                                                     n_fft_samples,
                                                     n_signals,
                                                     n_trials * n_tapers)

    Returns
    -------
    canonical_coherence : array, shape (n_time_windows, n_fft_samples)

    '''
    group_cross_spectrum = _complex_inner_product(
        normalized_fourier_coefficients1, normalized_fourier_coefficients2)
    return np.linalg.svd(group_cross_spectrum,
                         full_matrices=False, compute_uv=False)[..., 0]


def _bandpass(data, frequencies, frequencies_of_interest, axis=-3):
    '''Filters the data matrix along an axis given a maximum and minimum
    frequency of interest.

    Parameters
    ----------
    data : array, shape (..., n_fft_samples, ...)
    frequencies : array, shape (n_fft_samples,)
    frequencies_of_interest : array-like, shape (2,)

    Returns
    -------
    filtered_data : array
    filtered_frequencies : array

    '''
    frequency_index = ((frequencies_of_interest[0] < frequencies) &
                       (frequencies < frequencies_of_interest[1]))
    return (np.take(data, frequency_index.nonzero()[0], axis=axis),
            frequencies[frequency_index])


def _get_independent_frequency_step(frequency_difference,
                                    frequency_resolution):
    '''Find the number of points of a frequency axis such that they
    are statistically independent given a frequency resolution.


    Parameters
    ----------
    frequency_difference : float
        The distance between two frequency points
    frequency_resolution : float
        The ability to resolve frequency points

    Returns
    -------
    frequency_step : int
        The number of points required so that two
        frequency points are statistically independent.

    '''
    return int(np.ceil(frequency_resolution / frequency_difference))


def _find_largest_significant_group(is_significant):
    '''Finds the largest cluster of significant values over frequencies.

    If frequency value is signficant and its neighbor in the next frequency
    is also a signficant value, then they are part of the same cluster.

    If there are two clusters of the same size, the first one encountered
    is the signficant cluster. All other signficant values are set to
    false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_largest : bool array

    '''
    labeled, _ = label(is_significant)
    label_groups, label_counts = np.unique(labeled, return_counts=True)

    if not np.all(label_groups == 0):
        label_counts[0] = 0
        max_group = label_groups[np.argmax(label_counts)]
        return labeled == max_group
    else:
        return np.zeros(is_significant.shape, dtype=bool)


def _get_independent_frequencies(is_significant, frequency_step):
    '''Given a `frequency_step` that determines the distance to the next
    signficant point, sets non-distinguishable points to false.

    Parameters
    ----------
    is_significant : bool array

    Returns
    -------
    is_significant_independent : bool array

    '''
    index = is_significant.nonzero()[0]
    independent_index = index[0:len(index):frequency_step]
    return np.in1d(np.arange(0, len(is_significant)), independent_index)


def _find_largest_independent_group(is_significant, frequency_step,
                                    min_group_size=3):
    '''Finds the largest signficant cluster of frequency points and
    returns the indpendent frequency points of that cluster

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

    '''
    is_significant = _find_largest_significant_group(is_significant)
    is_significant = _get_independent_frequencies(
        is_significant, frequency_step)
    if sum(is_significant) < min_group_size:
        is_significant[:] = False
    return is_significant


def _find_significant_frequencies(
    coherency, bias, frequency_step=1, significance_threshold=0.05,
    min_group_size=3,
        multiple_comparisons_method='Benjamini_Hochberg_procedure'):
    '''Determines the largest significant cluster along the frequency axis.

    This function uses the fisher z-transform to determine the p-values and
    adjusts for multiple comparisons using the
    `multiple_comparisons_method`. Only independent frequencies are
    returned and there must be at least `min_group_size` frequency
    points for the cluster to be returned. If there are several significant
    groups, then only the largest group is returned.

    Parameters
    ----------
    coherency : array, shape (..., n_frequencies, n_signals, n_signals)
        The complex coherency between signals.
    bias : float
        Bias from the number of indpendent estimates of the frequency
        transform.
    frequency_step : int
        The number of points between each independent frequency step
    significance_threshold : float
        The threshold for a p-value to be considered signficant.
    min_group_size : int
        The minimum number of independent frequency points for
    multiple_comparisons_method : 'Benjamini_Hochberg_procedure' |
                                  'Bonferroni_correction'
        Procedure used to correct for multiple comparisons.

    Returns
    -------
    is_significant : bool array, shape (..., n_frequencies,
                                        n_signal_combintaions)

    '''
    z_coherence = fisher_z_transform(coherency, bias)
    p_values = get_normal_distribution_p_values(z_coherence)
    is_significant = adjust_for_multiple_comparisons(
        p_values, alpha=significance_threshold)
    return np.apply_along_axis(_find_largest_independent_group, -2,
                               is_significant, frequency_step,
                               min_group_size)


def _conjugate_transpose(x):
    '''Conjugate transpose of the last two dimensions of array x'''
    return x.swapaxes(-1, -2).conjugate()


def _estimate_global_coherence(fourier_coefficients, max_rank=1):
    """Estimate global coherence

    Parameters
    ----------
    fourier_coefficients : ndarray, shape (n_signals, n_trials * n_tapers)
        The fourier coefficients for a given frequency across all channels
    max_rank : float, optional
        The maximum number of singular values to keep

    Returns
    -------
    global_coherence : ndarray, shape (max_rank,)
        The vector of global coherences (square of the singular values)
    unnormalized_global_coherence : ndarray, shape (n_signals, max_rank)
        The (unnormalized) global coherence vectors

    """
    n_signals, n_estimates = fourier_coefficients.shape

    if max_rank >= n_signals - 1:
        unnormalized_global_coherence, global_coherence, _ = np.linalg.svd(
            fourier_coefficients, full_matrices=False)
        global_coherence = global_coherence[:max_rank]**2 / n_estimates
        unnormalized_global_coherence = unnormalized_global_coherence[:, :max_rank]  # noqa
    else:
        unnormalized_global_coherence, global_coherence, _ = svds(
            fourier_coefficients, max_rank)
        global_coherence = global_coherence**2 / n_estimates

    return global_coherence, unnormalized_global_coherence
