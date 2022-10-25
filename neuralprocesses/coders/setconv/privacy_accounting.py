import numpy as np
from scipy import special
import scipy.optimize as optim

def delta(epsilon, sens_per_sigma):
    """Compute delta for given epsilon and sensitivity per noise standard deviation for the Gaussian mechanism.

    Args:
        epsilon (float)
        sens_per_sigma (float): Sensitivity per noise standard deviation.

    Returns:
        float: Delta
    """
    if sens_per_sigma <= 0:
        return 0
    mu = sens_per_sigma**2 / 2
    term1 = special.erfc((epsilon - mu) / np.sqrt(mu) / 2)
    term2 = np.exp(epsilon) * special.erfc((epsilon + mu) / np.sqrt(mu) / 2)
    return 0.5 * (term1 - term2)

def find_sens_per_sigma(epsilon, delta_bound, upper_bound=20):
    """Find the required sensitivity per noise standard deviation for (epsilon, delta)-DP with Gaussian mechanism.

    Args:
        epsilon (float)
        delta_bound (float)
        upper_bound (float, optional): Upper bound guess on sensitivity per sigma. Defaults to 20.

    Returns:
        float: The required sensitivity per noise standard deviation.
    """
    return optim.brentq(lambda sens_per_sigma: delta(epsilon, sens_per_sigma) - delta_bound, 0, upper_bound)

def sigma(epsilon, delta_bound, sensitivity_squared, sensitivity_per_sigma_upper_bound=20):
    """Find the required noise standard deviation for the Gaussian mechanism with (epsilon, delta)-DP.

    Args:
        epsilon (float)
        delta_bound (float)
        sensitivity_squared (float): Square of the sensitivity of the Gaussian mechanism.
        sensitivity_per_sigma_upper_bound (float, optional): Guess for an upper bound on sensitivity / sigma. Defaults to 20.

    Returns:
        float: The required noise standard deviation.
    """
    return np.sqrt(sensitivity_squared) / find_sens_per_sigma(epsilon, delta_bound, sensitivity_per_sigma_upper_bound)
