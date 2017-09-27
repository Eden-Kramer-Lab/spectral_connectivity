import numpy as np
from scipy.stats import norm

np.seterr(invalid='ignore')


def Benjamini_Hochberg_procedure(p_values, alpha=0.05):
    '''Corrects for multiple comparisons and returns the significant
    p-values by controlling the false discovery rate at level `alpha`
    using the Benjamani-Hochberg procedure.
    Parameters
    ----------
    p_values : array_like
    alpha : float, optional
        The expected proportion of false positive tests.
    Returns
    -------
    is_significant : boolean nd-array
        A boolean array the same shape as `p_values` indicating whether the
        null hypothesis has been rejected (True) or failed to reject
        (False).
    '''
    p_values = np.array(p_values)
    threshold_line = np.linspace(0, alpha, num=p_values.size + 1,
                                 endpoint=True)[1:]
    sorted_p_values = np.sort(p_values.flatten())
    try:
        threshold_ind = np.max(
            np.where(sorted_p_values <= threshold_line)[0])
        threshold = sorted_p_values[threshold_ind]
    except ValueError:  # There are no values below threshold
        threshold = -1
    return p_values <= threshold


def Bonferroni_correction(p_values, alpha=0.05):
    p_values = np.asarray(p_values)
    return p_values <= alpha / p_values.size


MULTIPLE_COMPARISONS = dict(
    Benjamini_Hochberg_procedure=Benjamini_Hochberg_procedure,
    Bonferroni_correction=Bonferroni_correction
)


def adjust_for_multiple_comparisons(p_values, alpha=0.05,
                                    method='Benjamini_Hochberg_procedure'):
    '''Corrects for multiple comparisons and returns the significant
    p-values.

    Parameters
    ----------
    p_values : array_like
    alpha : float, optional
        The expected proportion of false positive tests.
    method : string, optional
        Name of the method to use to correct for multiple comparisons.
        Options are "Benjamini_Hochberg_procedure", "Bonferroni_correction"
    Returns
    -------
    is_significant : boolean nd-array
        A boolean array the same shape as `p_values` indicating whether the
        null hypothesis has been rejected (True) or failed to reject
        (False).

    '''
    # TODO: add axis keyword?
    return MULTIPLE_COMPARISONS[method](p_values, alpha=alpha)


def fisher_z_transform(coherency1, bias1, coherency2=0, bias2=0):
    '''Transform the coherence magnitude to an approximately normal
    distribution.

    If two coherencies are provided, then the function returns the
    fisher z transform of the difference of the coherencies with
    `coherency1` - `coherency2`.

    Parameters
    ----------
    coherency1 : complex array
        The complex coherency between signals
    bias1 : float
        The bias from independent estimates of the frequency domain.
    coherency2 : complex array, optional
    bias2 : float, optional
        The bias from independent estimates of the frequency domain.

    Returns
    -------
    fisher_z_transform : real array
        Either the difference from 0 mean or, if another coherency is
        provided, the difference from that coherency.

    '''
    coherence_magnitude1 = np.abs(coherency1)
    coherence_magnitude1[coherence_magnitude1 >= 1] = (
        1 - np.finfo(float).eps)

    coherence_magnitude2 = np.array(np.abs(coherency2))
    coherence_magnitude2[coherence_magnitude2 >= 1] = (
        1 - np.finfo(float).eps)

    z1 = np.arctanh(coherence_magnitude1) - bias1
    z2 = np.arctanh(coherence_magnitude2) - bias2
    return (z1 - z2) / np.sqrt(bias1 + bias2)


def get_normal_distribution_p_values(data, mean=0, std_deviation=1):
    '''Given data, returns the probability the data was generated from
    a normal distribution with `mean` and `std_deviation`
    '''
    return 1 - norm.cdf(data, loc=mean, scale=std_deviation)


def coherence_bias(n_observations):
    ''' Enochson and Goodman (1965)
        Bokil et al. (2007)
    '''
    degrees_of_freedom = 2 * n_observations
    return 1.0 / (degrees_of_freedom - 2)
