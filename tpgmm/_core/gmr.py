from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from numpy import ndarray

from tpgmm._core.learning_modules import RegressionModel
from tpgmm._core.arrays import get_subarray


class BaseGMR(RegressionModel):
    """Abstract base class for Gaussian Mixture Regression.

    This class defines the common interface and documentation for all GMR backend
    implementations (NumPy, PyTorch, JAX).

    It fits a Gaussian mixture regression on a given Gaussian Mixture model or a
    Task-Parameterized Gaussian Mixture model. Used equations are:
    \\f[
        \\mathcal{P}(\\phi_t^\\mathcal{O}|\\phi_t^\\mathcal{I}) \\sim \\sum_{i=1}^K h_i(\\phi_t^\\mathcal{I}) \\mathcal{N}\\left(\\hat{\\mu}_t^\\mathcal{O}(\\phi_t^\\mathcal{I}), \\hat{\\Sigma}_t^\\mathcal{O}\\right)
    \\f]
    \\f[
        \\hat{\\mu}_i^\\mathcal{O}(\\phi_t^\\mathcal{I}) = \\mu_i^\\mathcal{O} + \\Sigma_i^\\mathcal{OI}\\Sigma_i^{\\mathcal{I}, -1}(\\phi_t^\\mathcal{I} - \\mu_i^\\mathcal{I})
    \\f]
    \\f[
        \\hat{\\Sigma}_t^\\mathcal{O} = \\Sigma_i^\\mathcal{O} - \\Sigma_i^\\mathcal{OI}\\Sigma_i^{\\mathcal{I}, -1}\\Sigma_i^\\mathcal{OI}
    \\f]
    \\f[
        h_i(\\phi_t^\\mathcal{I}) = \\frac{\\pi_i \\mathcal{N}(\\phi_t^\\mathcal{I} \\mid \\mu_i^\\mathcal{I}, \\Sigma_i^\\mathcal{I})}{\\sum_k^K \\pi_k \\mathcal{N}\\mathcal(\\phi_t^\\mathcal{I} \\mid \\mu_k^\\mathcal{I}, \\Sigma_k^\\mathcal{I})}
    \\f]

    Example:
        >>> trajectories = load_trajectories()  # shape: (num_reference_frames, num_points, 4)
        >>> tpgmm = TPGMM(n_components=5)
        >>> tpgmm.fit(trajectories)
        >>> gmr = GaussianMixtureRegression(weights=tpgmm.weights_, means=tpgmm.means_[0], covariances=tpgmm.covariances_[0], input_idx=[3])
        >>> gmr.fit(trajectory[0])
    """

    def __init__(
        self,
        weights,
        means,
        covariances,
        input_idx: Iterable[int],
    ) -> None:
        self.tpgmm_means_ = means
        self.tpgmm_covariances_ = covariances
        self.gmm_weights = weights
        (
            self.num_frames,
            self.num_components,
            self.num_features,
        ) = self.tpgmm_means_.shape

        self.input_idx = input_idx

    @abstractmethod
    def fit(self, translation, rotation_matrix) -> None:
        """Turns the task-parameterized GMM into a single GMM.

        Performs equation (5) and (6) from the Calinon paper.

        Args:
            translation: Translation matrix. Shape (num_frames, num_output_features).
            rotation_matrix: Rotation matrix. Shape (num_frames, num_output_features, num_output_features).
        """
        ...

    @abstractmethod
    def predict(self, input_data) -> Tuple:
        """Predict output distribution for each input data point.

        Args:
            input_data: Shape (num_points, num_input_features).

        Returns:
            Tuple: (mu, cov) - mu shape (num_points, num_output_features),
                   cov shape (num_points, num_output_features, num_output_features).
        """
        ...

    def _sort_by_input(self, data: ndarray, axes: Iterable[int] = (0,)) -> ndarray:
        """Sort model parameters in feature space: input features first, then output features."""
        sort_index = [*self.input_idx, *self.output_idx]
        sort_index = [sort_index for _ in axes]
        return get_subarray(data, axes, sort_index)

    def _revoke_sort_by_input(self, data: ndarray, axes: Iterable[int] = (0,)) -> ndarray:
        """Revoke the self._sort_by_input reordering."""
        sort_index = np.empty(self.num_features, dtype=int)
        sort_index[self.input_idx] = range(self.num_input_features)
        sort_index[self.output_idx] = range(self.num_input_features, self.num_features)
        sort_index = [sort_index.tolist() for _ in axes]
        return get_subarray(data, axes, sort_index)

    def _tile_mean(self, mean: ndarray) -> Tuple[ndarray, ndarray]:
        """Tile mean into input and output components."""
        return get_subarray(mean, axes=[-1], indices=[self.input_idx]), np.delete(
            mean, self.input_idx, -1
        )

    def _tile_covariance(self, cov_mat: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Tile covariance matrix into input, input-output, output-input, and output blocks."""
        feature_index = [-2, -1]
        cov_i = get_subarray(cov_mat, feature_index, [self.input_idx, self.input_idx])
        cov_io = get_subarray(cov_mat, feature_index, [self.input_idx, self.output_idx])
        cov_oi = get_subarray(cov_mat, feature_index, [self.output_idx, self.input_idx])
        cov_o = get_subarray(cov_mat, feature_index, [self.output_idx, self.output_idx])
        return cov_i, cov_io, cov_oi, cov_o

    @property
    def num_input_features(self) -> int:
        """Get number of input features."""
        return len(self.input_idx)

    @property
    def num_output_features(self) -> int:
        """Get number of output features."""
        return self.num_features - self.num_input_features

    @property
    def output_idx(self) -> List[int]:
        """Get indices of output features (all features not in input_idx)."""
        return np.setdiff1d(np.array(range(self.num_features)), self.input_idx).tolist()

    @property
    def config(self) -> Dict[str, Any]:
        return {}
