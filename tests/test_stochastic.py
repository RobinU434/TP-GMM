import numpy as np

from tpgmm._core.stochastic import multivariate_gauss_cdf


class TestMultivariateGaussCdf:
    def test_known_distribution(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        mean = np.array([2, 3])
        cov = np.array([[1, 0], [0, 1]])
        result = multivariate_gauss_cdf(data, mean, cov)
        expected = [0.05854983152431917, 0.05854983152431917, 1.9641280346397437e-05]
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestBackwardCompatImport:
    """Verify backward-compatible import from tpgmm.utils.stochastic still works."""

    def test_import_from_utils(self):
        from tpgmm.utils.stochastic import multivariate_gauss_cdf
        assert callable(multivariate_gauss_cdf)