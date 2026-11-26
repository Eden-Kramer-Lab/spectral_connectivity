"""The Monte Carlo acceptance rule shared by the tests (see CLAUDE.md).

A Monte Carlo mean is accepted when it lies within five standard errors of its
expectation, the standard error being the replicates' sample standard
deviation over ``sqrt(n_replicates)``. The mean of an unbiased estimator falls
outside that limit with probability 6e-7.
"""

import numpy as np
from numpy.typing import NDArray


def replicate_mean_and_limit(replicates: NDArray[np.floating]) -> tuple[float, float]:
    """Mean of independent replicates and five standard errors of that mean.

    Parameters
    ----------
    replicates : NDArray[floating], shape (n_replicates,)
        One value per independent replicate.

    Returns
    -------
    mean, limit : float
    """
    standard_error = replicates.std(ddof=1) / np.sqrt(replicates.size)
    return float(replicates.mean()), float(5 * standard_error)
