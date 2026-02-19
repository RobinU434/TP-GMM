import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from numpy import ndarray

from tpgmm._core.gmr import BaseGMR
from tpgmm._core.stochastic import multivariate_gauss_cdf
from tpgmm._core.arrays import identity_like


class GaussianMixtureRegression(BaseGMR):
    """NumPy implementation of Gaussian Mixture Regression.

    See :class:`tpgmm._core.gmr.BaseGMR` for full documentation.
    """

    def __init__(
        self,
        weights: ndarray,
        means: ndarray,
        covariances: ndarray,
        input_idx: Iterable[int],
    ) -> None:
        super().__init__(weights, means, covariances, input_idx)

        # parameters from equation 5 and 6
        self.xi_: ndarray  # shape: (num_components, num_features)
        self.sigma_: ndarray  # shape: (num_components, num_features, num_features)

    @classmethod
    def from_tpgmm(
        cls: "GaussianMixtureRegression", tpgmm, input_idx: Iterable[int]
    ) -> "GaussianMixtureRegression":
        """Create a GaussianMixtureRegression instance from a fitted TPGMM model.

        Args:
            tpgmm: A fitted TPGMM model with weights_, means_, and covariances_ attributes.
            input_idx: Indices of input features for the regression.

        Returns:
            GaussianMixtureRegression: A new GMR instance initialized from the TPGMM parameters.
        """
        result = cls(
            weights=tpgmm.weights_,
            means=tpgmm.means_,
            covariances=tpgmm.covariances_,
            input_idx=input_idx,
        )
        return result

    def _equation_5(self, translation: ndarray, rotation_matrix: ndarray):
        """Transform means and covariances into task frames (Calinon Eq. 5).

        Args:
            translation: Translation vectors. Shape (num_frames, num_features).
            rotation_matrix: Rotation matrices. Shape (num_frames, num_features, num_features).

        Returns:
            Tuple[ndarray, ndarray]: Transformed means and covariances per frame.
        """
        sorted_means = self._sort_by_input(
            self.tpgmm_means_,
            axes=[-1],
        )
        sorted_covariances = self._sort_by_input(
            self.tpgmm_covariances_,
            axes=[-2, -1],
        )
        # i: num_frames, k, l: num_features, j: num_components
        xi_hat_ = np.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)
        # broadcast translation (num_frames, num_features) -> (num_frames, num_components, num_features)
        translation = np.tile(translation[:, None, :], (1, xi_hat_.shape[1], 1))
        xi_hat_ = xi_hat_ + translation
        # i: num_frames, k, l, h: num_features, j: num_components
        sigma_hat_ = np.einsum("ikl,ijlh->ijkh", rotation_matrix, sorted_covariances)
        sigma_hat_ = np.einsum(
            "ijkh,ihl->ijkl", sigma_hat_, rotation_matrix.swapaxes(-2, -1)
        )

        return xi_hat_, sigma_hat_

    def _equation_6(self, xi_hat_: ndarray, sigma_hat_: ndarray):
        """Combine frame-specific parameters into a single GMM (Calinon Eq. 6).

        Args:
            xi_hat_: Frame-specific means. Shape (num_frames, num_components, num_features).
            sigma_hat_: Frame-specific covariances. Shape (num_frames, num_components, num_features, num_features).

        Returns:
            Tuple[ndarray, ndarray]: Combined means and covariances.
        """
        sigma_hat_inv = np.linalg.inv(sigma_hat_)
        # shape: (num_components, num_features, num_features)
        sigma_hat = np.linalg.inv(np.sum(sigma_hat_inv, axis=0))

        # shape (num_frames, num_components, num_features)
        xi_hat = np.einsum("ijkl,ijl->ijk", sigma_hat_inv, xi_hat_)
        # shape (num_components, num_features)
        xi_hat = np.sum(xi_hat, axis=0)
        # shape (num_components, num_features)
        xi_hat = np.einsum("jkl,jl->jk", sigma_hat, xi_hat)

        return xi_hat, sigma_hat

    def fit(self, translation: ndarray, rotation_matrix: ndarray) -> None:
        """Turns the task_parameterized gaussian mixture model into a single gaussian mixture model

        function is performing equation (5) and (6) from calinon paper
        Args:
            translation (ndarray): translation matrix for translating into desired frames. Shape (num_frames, num_output_features)
            rotation_matrix (ndarray): rotation matrix for rotating into desired frames. Shape (num_frames, num_output_features, num_output_features)
        """
        rotation_matrix, translation = self._pad(rotation_matrix, translation)

        xi_hat, sigma_hat = self._equation_5(translation, rotation_matrix)
        xi_hat, sigma_hat = self._equation_6(xi_hat, sigma_hat + identity_like(sigma_hat) * 1e-15)

        # rearange into original feature order
        xi_hat = self._revoke_sort_by_input(
            xi_hat,
            axes=[
                -1,
            ],
        )
        sigma_hat = self._revoke_sort_by_input(sigma_hat, axes=[-2, -1])

        # store in self
        self.xi_ = xi_hat  # shape (num_components, num_features)
        self.sigma_ = sigma_hat  # shape: (num_components, num_features, num_features)

    def predict(self, input_data: ndarray) -> Tuple[ndarray, ndarray]:
        """this function is inspired by formula 13 in Calinon paper
        it creates for each given data point its own parameterized gaussian distribution.

        Args:
            data (ndarray): Shape: (num_points, num_input_features)

        Returns:
            Tuple[ndarray, ndarray] : mu: shape -> (num_points, num_output_features), cov: shape (num_points, num_output_features, num_output_features)
        """
        try:
            self.xi_
            self.sigma_
        except AttributeError:
            logging.error(
                "Not possible to predict trajectory because model was not fit on pick and place frames"
            )
            return np.zeros((len(input_data), self.num_output_features)), np.zeros(
                (len(input_data), self.num_output_features, self.num_output_features)
            )

        n_points = len(input_data)

        h = self._h(input_data)

        # MEAN
        mu_hat_out_ = self._mu_hat_out(input_data)
        # swap axis to: [num_output_features, num_points, num_components]
        mu_hat_out_ = mu_hat_out_.transpose((2, 0, 1))
        # weighted sum over all clusters
        mu_hat_out_ = (h * mu_hat_out_).sum(axis=-1)
        # swap dimensions back to: [num_points, num_output_features]
        mu_hat_out_ = mu_hat_out_.transpose((1, 0))

        # COVARIANCE MATRICES
        sigma_hat_out_ = self._sigma_hat_out()
        # bumb up shape to: [num_points, num_components, num_output_features, num_output_features]
        sigma_hat_out_ = np.expand_dims(sigma_hat_out_, axis=0)
        sigma_hat_out_ = np.repeat(sigma_hat_out_, n_points, axis=0)
        # swap dims to: [num_output_features, num_output_features, num_points, num_components]
        sigma_hat_out_ = sigma_hat_out_.transpose((2, 3, 0, 1))
        # weighted sum over all clusters
        sigma_hat_out_ = (sigma_hat_out_ * h).sum(axis=-1)
        # swap dims back to: [num_points, num_output_features, num_output_features]
        sigma_hat_out_ = sigma_hat_out_.transpose((2, 0, 1))

        return mu_hat_out_, sigma_hat_out_

    def _h(self, data: ndarray) -> ndarray:
        """Compute component responsibilities for input data points.

        Args:
            data: Input data. Shape (num_points, num_input_features).

        Returns:
            ndarray: Responsibility weights. Shape (num_points, num_components).
        """
        probabilities = []
        for component_input_mean, component_input_covariance in zip(
            self._tile_mean(self.xi_)[0], self._tile_covariance(self.sigma_)[0]
        ):
            probabilities.append(
                multivariate_gauss_cdf(
                    data, component_input_mean, component_input_covariance
                )
            )
        probabilities = np.stack(probabilities).T

        weighted_probs = probabilities * self.gmm_weights

        cluster_probs = (weighted_probs.T / np.sum(weighted_probs, axis=1)).T

        return cluster_probs

    def _mu_hat_out(self, input_data: ndarray) -> ndarray:
        """Compute conditional output means for each component given input data.

        Args:
            input_data: Input data. Shape (num_points, num_input_features).

        Returns:
            ndarray: Conditional means. Shape (num_points, num_components, num_output_features).
        """
        input_data = np.expand_dims(input_data, axis=1)
        input_data = np.tile(input_data, [1, self.num_components, 1])

        cov_i, _, cov_oi, _ = self._tile_covariance(self.sigma_)
        mean_input, mean_output = self._tile_mean(self.xi_)

        centered_points = input_data - mean_input
        cluster_mats = cov_oi @ np.linalg.inv(cov_i)

        mu_hat = np.einsum("ikh,jih->jik", cluster_mats, centered_points)

        mu_hat = mu_hat + mean_output
        return mu_hat

    def _sigma_hat_out(self) -> ndarray:
        """Compute conditional output covariance for each component.

        Returns:
            ndarray: Conditional covariances. Shape (num_components, num_output_features, num_output_features).
        """
        cov_i, cov_io, cov_oi, cov_o = self._tile_covariance(self.sigma_)
        return cov_o - cov_oi @ np.linalg.inv(cov_i) @ cov_io

    def _pad(
        self, rotation_matrix: ndarray, translation: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Pad rotation and translation matrices to include input features as identity block.

        Extends the rotation matrix with an identity block for input features and
        zero-pads the translation vector for input feature dimensions.

        Args:
            rotation_matrix: Rotation matrix for output features. Shape (num_frames, num_output_features, num_output_features).
            translation: Translation vector for output features. Shape (num_frames, num_output_features).

        Returns:
            Tuple[ndarray, ndarray]: Padded rotation matrix and translation vector with full feature dimensions.
        """
        num_frames, num_output_features, _ = rotation_matrix.shape
        identity = np.eye(self.num_input_features)
        identity = np.repeat(identity[None], num_frames, axis=0)

        zeros_io = np.zeros((num_frames, self.num_input_features, num_output_features))
        zeros_oi = zeros_io.swapaxes(-1, -2)
        padded_rot_mat = np.block([[identity, zeros_io], [zeros_oi, rotation_matrix]])

        zeros_o = np.zeros((num_frames, self.num_input_features))
        padded_translation = np.concatenate([zeros_o, translation], axis=-1)

        return padded_rot_mat, padded_translation
