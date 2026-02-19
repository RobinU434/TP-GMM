import logging
from typing import Iterable, Tuple

import torch
from torch import Tensor
import numpy as np

from tpgmm._core.gmr import BaseGMR
from tpgmm._core.arrays import identity_like


class GaussianMixtureRegression(BaseGMR):
    """PyTorch implementation of Gaussian Mixture Regression.

    See :class:`tpgmm._core.gmr.BaseGMR` for full documentation.
    """

    def __init__(
        self,
        weights: Tensor,
        means: Tensor,
        covariances: Tensor,
        input_idx: Iterable[int],
    ) -> None:
        # Convert to numpy for base class (uses numpy internally for indexing)
        if isinstance(weights, Tensor):
            weights_np = weights.detach().numpy()
            means_np = means.detach().numpy()
            covariances_np = covariances.detach().numpy()
        else:
            weights_np = np.asarray(weights)
            means_np = np.asarray(means)
            covariances_np = np.asarray(covariances)
        super().__init__(weights_np, means_np, covariances_np, input_idx)
        # Store torch versions
        self._weights_torch = torch.as_tensor(weights_np, dtype=torch.float32)
        self._means_torch = torch.as_tensor(means_np, dtype=torch.float32)
        self._covariances_torch = torch.as_tensor(covariances_np, dtype=torch.float32)

        self.xi_: Tensor
        self.sigma_: Tensor

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
            weights=tpgmm.weights_,
            means=tpgmm.means_,
            covariances=tpgmm.covariances_,
            input_idx=input_idx,
        )

    def _equation_5(self, translation: Tensor, rotation_matrix: Tensor):
        """Transform means and covariances into task frames (Calinon Eq. 5).

        Args:
            translation: Translation vectors. Shape (num_frames, num_features).
            rotation_matrix: Rotation matrices. Shape (num_frames, num_features, num_features).

        Returns:
            Tuple[Tensor, Tensor]: Transformed means and covariances per frame.
        """
        sorted_means = self._sort_by_input(self.tpgmm_means_, axes=[-1])
        sorted_covariances = self._sort_by_input(self.tpgmm_covariances_, axes=[-2, -1])
        sorted_means = torch.as_tensor(sorted_means, dtype=torch.float32)
        sorted_covariances = torch.as_tensor(sorted_covariances, dtype=torch.float32)

        xi_hat_ = torch.einsum("ikl,ijl->ijk", rotation_matrix, sorted_means)
        translation_tiled = translation[:, None, :].expand(-1, xi_hat_.shape[1], -1)
        xi_hat_ = xi_hat_ + translation_tiled

        sigma_hat_ = torch.einsum("ikl,ijlh->ijkh", rotation_matrix, sorted_covariances)
        sigma_hat_ = torch.einsum(
            "ijkh,ihl->ijkl", sigma_hat_, rotation_matrix.transpose(-2, -1)
        )
        return xi_hat_, sigma_hat_

    def _equation_6(self, xi_hat_: Tensor, sigma_hat_: Tensor):
        """Combine frame-specific parameters into a single GMM (Calinon Eq. 6).

        Args:
            xi_hat_: Frame-specific means. Shape (num_frames, num_components, num_features).
            sigma_hat_: Frame-specific covariances. Shape (num_frames, num_components, num_features, num_features).

        Returns:
            Tuple[Tensor, Tensor]: Combined means and covariances.
        """
        sigma_hat_inv = torch.linalg.inv(sigma_hat_)
        sigma_hat = torch.linalg.inv(torch.sum(sigma_hat_inv, dim=0))

        xi_hat = torch.einsum("ijkl,ijl->ijk", sigma_hat_inv, xi_hat_)
        xi_hat = torch.sum(xi_hat, dim=0)
        xi_hat = torch.einsum("jkl,jl->jk", sigma_hat, xi_hat)
        return xi_hat, sigma_hat

    def fit(self, translation: Tensor, rotation_matrix: Tensor) -> None:
        """Turns the task_parameterized GMM into a single GMM.

        Args:
            translation (Tensor): Shape (num_frames, num_output_features).
            rotation_matrix (Tensor): Shape (num_frames, num_output_features, num_output_features).
        """
        rotation_matrix, translation = self._pad(rotation_matrix, translation)
        rotation_matrix = torch.as_tensor(rotation_matrix, dtype=torch.float32)
        translation = torch.as_tensor(translation, dtype=torch.float32)

        xi_hat, sigma_hat = self._equation_5(translation, rotation_matrix)
        reg = torch.eye(sigma_hat.shape[-1]).expand_as(sigma_hat) * 1e-15
        xi_hat, sigma_hat = self._equation_6(xi_hat, sigma_hat + reg)

        # rearrange into original feature order (uses numpy via base class)
        xi_hat_np = xi_hat.detach().numpy()
        sigma_hat_np = sigma_hat.detach().numpy()
        xi_hat_np = self._revoke_sort_by_input(xi_hat_np, axes=[-1])
        sigma_hat_np = self._revoke_sort_by_input(sigma_hat_np, axes=[-2, -1])

        self.xi_ = torch.from_numpy(xi_hat_np)
        self.sigma_ = torch.from_numpy(sigma_hat_np)

    def predict(self, input_data: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict output distribution for each input data point.

        Args:
            input_data (Tensor): Shape (num_points, num_input_features).

        Returns:
            Tuple[Tensor, Tensor]: mu (num_points, num_output_features),
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
            return torch.zeros(n, self.num_output_features), torch.zeros(
                n, self.num_output_features, self.num_output_features
            )

        if not isinstance(input_data, Tensor):
            input_data = torch.as_tensor(input_data, dtype=torch.float32)

        n_points = len(input_data)
        h = self._h(input_data)

        # MEAN
        mu_hat_out_ = self._mu_hat_out(input_data)
        mu_hat_out_ = mu_hat_out_.permute(2, 0, 1)
        mu_hat_out_ = (h * mu_hat_out_).sum(dim=-1)
        mu_hat_out_ = mu_hat_out_.permute(1, 0)

        # COVARIANCE
        sigma_hat_out_ = self._sigma_hat_out()
        sigma_hat_out_ = sigma_hat_out_.unsqueeze(0).expand(n_points, -1, -1, -1)
        sigma_hat_out_ = sigma_hat_out_.permute(2, 3, 0, 1)
        sigma_hat_out_ = (sigma_hat_out_ * h).sum(dim=-1)
        sigma_hat_out_ = sigma_hat_out_.permute(2, 0, 1)

        return mu_hat_out_, sigma_hat_out_

    def _h(self, data: Tensor) -> Tensor:
        """Compute component responsibilities for input data points.

        Args:
            data: Input data. Shape (num_points, num_input_features).

        Returns:
            Tensor: Responsibility weights. Shape (num_points, num_components).
        """
        xi_np = self.xi_.detach().numpy()
        sigma_np = self.sigma_.detach().numpy()
        mean_input_np, _ = self._tile_mean(xi_np)
        cov_i_np, _, _, _ = self._tile_covariance(sigma_np)

        probabilities = []
        for comp_mean, comp_cov in zip(mean_input_np, cov_i_np):
            diff = data.detach().numpy() - comp_mean
            a = diff @ np.linalg.inv(comp_cov)
            diag = np.einsum("ij,ji->i", a, diff.T)
            num_features = len(comp_mean)
            prob = np.exp(-0.5 * diag) / np.sqrt(
                np.power(2 * np.pi, num_features) * np.linalg.det(comp_cov)
            )
            probabilities.append(prob)
        probabilities = torch.from_numpy(np.stack(probabilities).T).float()

        weighted_probs = probabilities * self._weights_torch
        cluster_probs = (weighted_probs.T / torch.sum(weighted_probs, dim=1)).T
        return cluster_probs

    def _mu_hat_out(self, input_data: Tensor) -> Tensor:
        """Compute conditional output means for each component given input data.

        Args:
            input_data: Input data. Shape (num_points, num_input_features).

        Returns:
            Tensor: Conditional means. Shape (num_points, num_components, num_output_features).
        """
        input_data = input_data.unsqueeze(1).expand(-1, self.num_components, -1)

        xi_np = self.xi_.detach().numpy()
        sigma_np = self.sigma_.detach().numpy()
        cov_i_np, _, cov_oi_np, _ = self._tile_covariance(sigma_np)
        mean_input_np, mean_output_np = self._tile_mean(xi_np)

        cov_i = torch.from_numpy(cov_i_np).float()
        cov_oi = torch.from_numpy(cov_oi_np).float()
        mean_input = torch.from_numpy(mean_input_np).float()
        mean_output = torch.from_numpy(mean_output_np).float()

        centered_points = input_data - mean_input
        cluster_mats = cov_oi @ torch.linalg.inv(cov_i)
        mu_hat = torch.einsum("ikh,jih->jik", cluster_mats, centered_points)
        mu_hat = mu_hat + mean_output
        return mu_hat

    def _sigma_hat_out(self) -> Tensor:
        """Compute conditional output covariance for each component.

        Returns:
            Tensor: Conditional covariances. Shape (num_components, num_output_features, num_output_features).
        """
        sigma_np = self.sigma_.detach().numpy()
        cov_i_np, cov_io_np, cov_oi_np, cov_o_np = self._tile_covariance(sigma_np)
        cov_i = torch.from_numpy(cov_i_np).float()
        cov_io = torch.from_numpy(cov_io_np).float()
        cov_oi = torch.from_numpy(cov_oi_np).float()
        cov_o = torch.from_numpy(cov_o_np).float()
        return cov_o - cov_oi @ torch.linalg.inv(cov_i) @ cov_io

    def _pad(self, rotation_matrix, translation):
        """Pad rotation and translation to include input features (identity block)."""
        if isinstance(rotation_matrix, Tensor):
            rotation_matrix = rotation_matrix.detach().numpy()
        if isinstance(translation, Tensor):
            translation = translation.detach().numpy()

        num_frames, num_output_features, _ = rotation_matrix.shape
        identity = np.eye(self.num_input_features)
        identity = np.repeat(identity[None], num_frames, axis=0)

        zeros_io = np.zeros((num_frames, self.num_input_features, num_output_features))
        zeros_oi = zeros_io.swapaxes(-1, -2)
        padded_rot_mat = np.block([[identity, zeros_io], [zeros_oi, rotation_matrix]])

        zeros_o = np.zeros((num_frames, self.num_input_features))
        padded_translation = np.concatenate([zeros_o, translation], axis=-1)

        return padded_rot_mat, padded_translation
