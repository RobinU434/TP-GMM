import logging
from typing import Iterable, Tuple

import jax.numpy as jnp
from jax import Array
import numpy as np

from tpgmm._core.gmr import BaseGMR
from tpgmm._core.arrays import identity_like


class GaussianMixtureRegression(BaseGMR):
    """JAX implementation of Gaussian Mixture Regression.

    See :class:`tpgmm._core.gmr.BaseGMR` for full documentation.
    """

    def __init__(
        self,
        weights,
        means,
        covariances,
        input_idx: Iterable[int],
    ) -> None:
        weights_np = np.asarray(weights)
        means_np = np.asarray(means)
        covariances_np = np.asarray(covariances)
        super().__init__(weights_np, means_np, covariances_np, input_idx)

        self.xi_: Array
        self.sigma_: Array

    @classmethod
    def from_tpgmm(
        cls, tpgmm, input_idx: Iterable[int]
    ) -> "GaussianMixtureRegression":
        """Create a GaussianMixtureRegression instance from a fitted TPGMM model.

        Args:
            tpgmm: A fitted TPGMM model with weights_, means_, and covariances_ attributes.
            input_idx: Indices of input features for the regression.

        Returns:
            GaussianMixtureRegression: A new GMR instance initialized from the TPGMM parameters.
        """
        return cls(
            weights=np.asarray(tpgmm.weights_),
            means=np.asarray(tpgmm.means_),
            covariances=np.asarray(tpgmm.covariances_),
            input_idx=input_idx,
        )

    def _equation_5(self, translation: Array, rotation_matrix: Array):
        """Transform means and covariances into task frames (Calinon Eq. 5).

        Args:
            translation: Translation vectors. Shape (num_frames, num_features).
            rotation_matrix: Rotation matrices. Shape (num_frames, num_features, num_features).

        Returns:
            Tuple[Array, Array]: Transformed means and covariances per frame.
        """
        sorted_means = self._sort_by_input(self.tpgmm_means_, axes=[-1])
        sorted_covariances = self._sort_by_input(self.tpgmm_covariances_, axes=[-2, -1])
        sorted_means = jnp.asarray(sorted_means)
        sorted_covariances = jnp.asarray(sorted_covariances)

        xi_hat_ = jnp.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)
        translation_tiled = jnp.tile(translation[:, None, :], (1, xi_hat_.shape[1], 1))
        xi_hat_ = xi_hat_ + translation_tiled

        sigma_hat_ = jnp.einsum("ikl,ijlh->ijkh", rotation_matrix, sorted_covariances)
        sigma_hat_ = jnp.einsum(
            "ijkh,ihl->ijkl", sigma_hat_, jnp.swapaxes(rotation_matrix, -2, -1)
        )
        return xi_hat_, sigma_hat_

    def _equation_6(self, xi_hat_: Array, sigma_hat_: Array):
        """Combine frame-specific parameters into a single GMM (Calinon Eq. 6).

        Args:
            xi_hat_: Frame-specific means. Shape (num_frames, num_components, num_features).
            sigma_hat_: Frame-specific covariances. Shape (num_frames, num_components, num_features, num_features).

        Returns:
            Tuple[Array, Array]: Combined means and covariances.
        """
        sigma_hat_inv = jnp.linalg.inv(sigma_hat_)
        sigma_hat = jnp.linalg.inv(jnp.sum(sigma_hat_inv, axis=0))

        xi_hat = jnp.einsum("ijkl,ijl->ijk", sigma_hat_inv, xi_hat_)
        xi_hat = jnp.sum(xi_hat, axis=0)
        xi_hat = jnp.einsum("jkl,jl->jk", sigma_hat, xi_hat)
        return xi_hat, sigma_hat

    def fit(self, translation, rotation_matrix) -> None:
        """Turns the task_parameterized GMM into a single GMM.

        Args:
            translation: Shape (num_frames, num_output_features).
            rotation_matrix: Shape (num_frames, num_output_features, num_output_features).
        """
        rotation_matrix_np, translation_np = self._pad(
            np.asarray(rotation_matrix), np.asarray(translation)
        )
        rotation_matrix_j = jnp.asarray(rotation_matrix_np)
        translation_j = jnp.asarray(translation_np)

        xi_hat, sigma_hat = self._equation_5(translation_j, rotation_matrix_j)
        reg = jnp.eye(sigma_hat.shape[-1]) * 1e-15
        xi_hat, sigma_hat = self._equation_6(xi_hat, sigma_hat + reg)

        xi_hat_np = np.asarray(xi_hat)
        sigma_hat_np = np.asarray(sigma_hat)
        xi_hat_np = self._revoke_sort_by_input(xi_hat_np, axes=[-1])
        sigma_hat_np = self._revoke_sort_by_input(sigma_hat_np, axes=[-2, -1])

        self.xi_ = jnp.asarray(xi_hat_np)
        self.sigma_ = jnp.asarray(sigma_hat_np)

    def predict(self, input_data) -> Tuple[Array, Array]:
        """Predict output distribution for each input data point.

        Args:
            input_data: Shape (num_points, num_input_features).

        Returns:
            Tuple[Array, Array]: mu (num_points, num_output_features),
                cov (num_points, num_output_features, num_output_features).
        """
        try:
            self.xi_
            self.sigma_
        except AttributeError:
            logging.error(
                "Not possible to predict trajectory because model was not fit"
            )
            n = len(input_data)
            return jnp.zeros((n, self.num_output_features)), jnp.zeros(
                (n, self.num_output_features, self.num_output_features)
            )

        input_data = jnp.asarray(input_data)
        n_points = len(input_data)

        h = self._h(input_data)

        # MEAN
        mu_hat_out_ = self._mu_hat_out(input_data)
        mu_hat_out_ = jnp.transpose(mu_hat_out_, (2, 0, 1))
        mu_hat_out_ = (h * mu_hat_out_).sum(axis=-1)
        mu_hat_out_ = jnp.transpose(mu_hat_out_, (1, 0))

        # COVARIANCE
        sigma_hat_out_ = self._sigma_hat_out()
        sigma_hat_out_ = jnp.expand_dims(sigma_hat_out_, axis=0)
        sigma_hat_out_ = jnp.tile(sigma_hat_out_, (n_points, 1, 1, 1))
        sigma_hat_out_ = jnp.transpose(sigma_hat_out_, (2, 3, 0, 1))
        sigma_hat_out_ = (sigma_hat_out_ * h).sum(axis=-1)
        sigma_hat_out_ = jnp.transpose(sigma_hat_out_, (2, 0, 1))

        return mu_hat_out_, sigma_hat_out_

    def _h(self, data: Array) -> Array:
        """Compute component responsibilities for input data points.

        Args:
            data: Input data. Shape (num_points, num_input_features).

        Returns:
            Array: Responsibility weights. Shape (num_points, num_components).
        """
        xi_np = np.asarray(self.xi_)
        sigma_np = np.asarray(self.sigma_)
        mean_input_np, _ = self._tile_mean(xi_np)
        cov_i_np, _, _, _ = self._tile_covariance(sigma_np)

        data_np = np.asarray(data)
        probabilities = []
        for comp_mean, comp_cov in zip(mean_input_np, cov_i_np):
            diff = data_np - comp_mean
            a = diff @ np.linalg.inv(comp_cov)
            diag = np.einsum("ij,ji->i", a, diff.T)
            num_features = len(comp_mean)
            prob = np.exp(-0.5 * diag) / np.sqrt(
                np.power(2 * np.pi, num_features) * np.linalg.det(comp_cov)
            )
            probabilities.append(prob)
        probabilities = jnp.array(np.stack(probabilities).T)

        gmm_weights = jnp.asarray(self.gmm_weights)
        weighted_probs = probabilities * gmm_weights
        cluster_probs = (weighted_probs.T / jnp.sum(weighted_probs, axis=1)).T
        return cluster_probs

    def _mu_hat_out(self, input_data: Array) -> Array:
        """Compute conditional output means for each component given input data.

        Args:
            input_data: Input data. Shape (num_points, num_input_features).

        Returns:
            Array: Conditional means. Shape (num_points, num_components, num_output_features).
        """
        input_data = jnp.expand_dims(input_data, axis=1)
        input_data = jnp.tile(input_data, (1, self.num_components, 1))

        xi_np = np.asarray(self.xi_)
        sigma_np = np.asarray(self.sigma_)
        cov_i_np, _, cov_oi_np, _ = self._tile_covariance(sigma_np)
        mean_input_np, mean_output_np = self._tile_mean(xi_np)

        cov_i = jnp.asarray(cov_i_np)
        cov_oi = jnp.asarray(cov_oi_np)
        mean_input = jnp.asarray(mean_input_np)
        mean_output = jnp.asarray(mean_output_np)

        centered_points = input_data - mean_input
        cluster_mats = cov_oi @ jnp.linalg.inv(cov_i)
        mu_hat = jnp.einsum("ikh,jih->jik", cluster_mats, centered_points)
        mu_hat = mu_hat + mean_output
        return mu_hat

    def _sigma_hat_out(self) -> Array:
        """Compute conditional output covariance for each component.

        Returns:
            Array: Conditional covariances. Shape (num_components, num_output_features, num_output_features).
        """
        sigma_np = np.asarray(self.sigma_)
        cov_i_np, cov_io_np, cov_oi_np, cov_o_np = self._tile_covariance(sigma_np)
        cov_i = jnp.asarray(cov_i_np)
        cov_io = jnp.asarray(cov_io_np)
        cov_oi = jnp.asarray(cov_oi_np)
        cov_o = jnp.asarray(cov_o_np)
        return cov_o - cov_oi @ jnp.linalg.inv(cov_i) @ cov_io

    def _pad(self, rotation_matrix, translation):
        """Pad rotation and translation to include input features."""
        rotation_matrix = np.asarray(rotation_matrix)
        translation = np.asarray(translation)

        num_frames, num_output_features, _ = rotation_matrix.shape
        identity = np.eye(self.num_input_features)
        identity = np.repeat(identity[None], num_frames, axis=0)

        zeros_io = np.zeros((num_frames, self.num_input_features, num_output_features))
        zeros_oi = zeros_io.swapaxes(-1, -2)
        padded_rot_mat = np.block([[identity, zeros_io], [zeros_oi, rotation_matrix]])

        zeros_o = np.zeros((num_frames, self.num_input_features))
        padded_translation = np.concatenate([zeros_o, translation], axis=-1)

        return padded_rot_mat, padded_translation
