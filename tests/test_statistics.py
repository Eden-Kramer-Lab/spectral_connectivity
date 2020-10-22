import numpy as np
from pytest import mark
from spectral_connectivity.statistics import (Benjamini_Hochberg_procedure,
                                              Bonferroni_correction,
                                              coherence_bias,
                                              fisher_z_transform,
                                              get_normal_distribution_p_values)


def test_get_normal_distribution_p_values():
    # approximate 97.5 percentile of the standard normal distribution
    zscore = 1.95996
    assert np.allclose(get_normal_distribution_p_values(zscore), 0.025)


def test_fisher_z_transform():
    coherency = 0.5 * np.exp(1j * np.pi / 2) * np.ones((2, 2))
    bias1, bias2 = 3, 6
    expected_difference_z = np.ones((2, 2))
    assert np.allclose(
        fisher_z_transform(
            coherency, bias1, coherency2=coherency, bias2=bias2),
        expected_difference_z)


@mark.parametrize(
    'p_values, expected_is_significant',
    [(np.ones((10, 2)), np.zeros((10, 2), dtype=bool)),
     (np.zeros((10, 2)), np.ones((10, 2), dtype=bool)),
     (np.array([0.03, 0.01, 0.04, 0.06]),
      np.array([False, True, False, False])),
     (np.array([0.03, 0.01, 0.04, 0.05]),
      np.ones((4,)))])
def test_Benjamini_Hochberg_procedure(p_values, expected_is_significant):
    alpha = 0.05
    assert np.allclose(
        Benjamini_Hochberg_procedure(p_values, alpha),
        expected_is_significant)


@mark.parametrize(
    'p_values, expected_is_significant',
    [(np.ones((10, 2)), np.zeros((10, 2), dtype=bool)),
     (np.zeros((10, 2)), np.ones((10, 2), dtype=bool)),
     (np.array([0.03, 0.01, 0.04, 0.06]),
      np.array([False, True, False, False])),
     (np.array([0.03, 0.01, 0.04, 0.05]),
      np.array([False, True, False, False]))])
def test_Bonferroni_correction(p_values, expected_is_significant):
    alpha = 0.05
    assert np.allclose(
        Bonferroni_correction(p_values, alpha),
        expected_is_significant)


def test_coherence_bias():
    n_observations = 10
    expected_bias = 1.0 / 18
    assert coherence_bias(n_observations) == expected_bias
