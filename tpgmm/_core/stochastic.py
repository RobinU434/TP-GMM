from numpy import ndarray
import numpy as np


def multivariate_gauss_cdf(data: ndarray, mean: ndarray, cov: ndarray) -> ndarray:
    """Computes the probability of the given data points if they are in a given multivariate Gaussian distribution.

    Args:
        data (ndarray): Data array with shape (num_points, num_features).
        mean (ndarray): Mean array with shape (num_features).
        cov (ndarray): Covariance matrix with shape (num_features, num_features).

    Returns:
        ndarray: Array containing the probabilities for each data point, with shape (num_points).
    """
    num_features = len(mean)
    diff = data - mean
    a = diff @ np.linalg.inv(cov)
    diag = np.einsum("ij,ji->i", a, diff.T)
    return np.exp(-0.5 * diag) / np.sqrt(np.power(2 * np.pi, num_features) * np.linalg.det(cov))
