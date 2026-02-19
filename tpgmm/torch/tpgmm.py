from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from sklearn import metrics
from sklearn.cluster import KMeans

from tpgmm._core.tpgmm import BaseTPGMM


class TPGMM(BaseTPGMM):
    """PyTorch implementation of the Task Parameterized Gaussian Mixture Model.

    See :class:`tpgmm._core.tpgmm.BaseTPGMM` for full documentation.
    """

    def __init__(
        self,
        n_components: int,
        threshold: float = 1e-7,
        max_iter: int = 100,
        min_iter: int = 5,
        weights_init: Tensor = None,
        means_init: Tensor = None,
        reg_factor: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            n_components=n_components,
            threshold=threshold,
            max_iter=max_iter,
            min_iter=min_iter,
            weights_init=weights_init,
            means_init=means_init,
            reg_factor=reg_factor,
            verbose=verbose,
        )

        self._k_means_algo = KMeans(
            n_clusters=self._n_components, init="k-means++", n_init="auto"
        )
        self._cov_reg_matrix = None

    def fit(self, X: Tensor) -> None:
        # perform k-means clustering
        if self._verbose:
            print("Started KMeans clustering")
        self.means_, self.covariances_ = self._k_means(X)
        self._cov_reg_matrix = torch.eye(self.covariances_.shape[-1]) * self._reg_factor

        if self._verbose:
            print("finished KMeans clustering")

        # init weights with uniform probability
        if self.weights_ is None:
            self.weights_ = torch.ones(self._n_components) / self._n_components

        if self._verbose:
            print("Start expectation maximization")

        probabilities = self.gauss_pdf(X)
        log_likelihood = self._log_likelihood(probabilities)
        for epoch_idx in range(self._max_iter):
            # Expectation
            h = self._update_h(probabilities)

            # Maximization
            self._update_weights(h)
            self._update_mean(X, h)
            self._update_covariances_(X, h)

            # update probabilities and log likelihood
            probabilities = self.gauss_pdf(X)
            updated_log_likelihood = self._log_likelihood(probabilities)

            # Logging
            difference = updated_log_likelihood - log_likelihood
            if torch.isnan(torch.tensor(difference)):
                print(Warning("improvement is nan. Abort fit"))
                return False

            if self._verbose:
                print(
                    f"Log likelihood: {updated_log_likelihood} improvement {difference}"
                )

            # break if threshold is reached
            if (
                difference < self._threshold and epoch_idx >= self._min_iter
            ) or epoch_idx > self._max_iter:
                break

            log_likelihood = updated_log_likelihood

    def predict(self, X: Tensor) -> Tensor:
        probabilities = self.predict_proba(X)
        labels = torch.argmax(probabilities, dim=1)
        return labels

    def predict_proba(self, X: Tensor) -> Tensor:
        frame_probs = self.gauss_pdf(X)
        probabilities = torch.prod(frame_probs, dim=0).T
        return probabilities

    def silhouette_score(self, X: Tensor) -> float:
        labels = self.predict(X)
        scores = np.empty(X.shape[0])
        for frame_idx in range(X.shape[0]):
            scores[frame_idx] = metrics.silhouette_score(X[frame_idx].numpy(), labels.numpy())
        weights = self.weights_[:, None].repeat(1, X.shape[0])
        weighted_sum = (weights @ scores) / (self.weights_ * X.shape[0])
        return weighted_sum.mean()

    def score(self, X: Tensor) -> float:
        probabilities = self.gauss_pdf(X)
        score = self._log_likelihood(probabilities)
        return score

    def bic(self, X: Tensor) -> float:
        num_points = X.shape[1]
        log_likelihood = self.score(X)
        bic = -2 * log_likelihood + np.log(num_points) * self._num_params()
        return bic

    def aic(self, X: Tensor) -> float:
        log_likelihood = self.score(X).numpy()
        bic = -2 * log_likelihood + 2 * self._num_params()
        return bic

    def gauss_pdf(self, X: Tensor) -> Tensor:
        covariances = self.covariances_ + self._cov_reg_matrix
        # X: (F, N, D) -> (F, 1, N, D);  means: (F, K, D) -> (F, K, 1, D)
        diff = (X[:, None, :, :] - self.means_[:, :, None, :]).float()  # (F, K, N, D)
        cov_inv = torch.linalg.inv(covariances).float()  # (F, K, D, D)
        mahal = torch.einsum("fknd,fkde,fkne->fkn", diff, cov_inv, diff)
        D = X.shape[-1]
        det = torch.linalg.det(covariances)  # (F, K)
        norm = torch.sqrt((2 * torch.pi) ** D * det)  # (F, K)
        return torch.exp(-0.5 * mahal) / norm[:, :, None]

    @property
    def config(self) -> Dict[str, Any]:
        config = {
            "max_iter": self._max_iter,
            "min_iter": self._min_iter,
            "threshold": self._threshold,
            "reg_factor": self._reg_factor,
        }

        config = {**config, **super().config}
        return config

    def _num_params(self) -> int:
        """Calculate the number of free parameters in the model.

        Returns:
            int: Total number of free parameters (means + covariances + weights).
        """
        num_frames = 2
        num_mean_params = self._n_components * num_frames
        num_cov_params = num_frames * self._n_components * (self._n_components + 1) // 2
        num_weight_params = self._n_components - 1
        num_params = num_mean_params + num_cov_params + num_weight_params
        return num_params
    
    def _k_means(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Initialize means and covariances using K-Means clustering.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).

        Returns:
            Tuple[Tensor, Tensor]: Initial means (num_frames, n_components, num_features)
                and covariances (num_frames, n_components, num_features, num_features).
        """
        num_frames, _, num_features = X.shape
        means = torch.empty((num_frames, self._n_components, num_features))
        covariances = torch.empty(
            (num_frames, self._n_components, num_features, num_features)
        )
        for frame_idx, frame_data in enumerate(X):
            self._k_means_algo.fit(frame_data.detach().numpy())
            means[frame_idx] = torch.from_numpy(self._k_means_algo.cluster_centers_)
            for cluster_idx in range(self._n_components):
                data_idx = np.argwhere(
                    self._k_means_algo.labels_ == cluster_idx
                ).squeeze()
                covariances[frame_idx, cluster_idx] = torch.from_numpy(
                    np.cov(frame_data[data_idx].T)
                )

        reg_matrix = torch.eye(covariances.shape[-1]) * self._reg_factor
        covariances = covariances + reg_matrix
        return means, covariances

    def _update_h(self, probabilities: Tensor) -> Tensor:
        """E-step: compute component responsibilities from probability densities.

        Args:
            probabilities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            Tensor: Responsibilities. Shape (n_components, num_points).
        """
        cluster_probs = torch.prod(probabilities, dim=0)
        numerator = (self.weights_ * cluster_probs.T).T
        denominator = torch.sum(numerator, dim=0)
        h = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))
        return h

    def _update_weights(self, h: Tensor) -> None:
        """M-step: update component weights from responsibilities.

        Args:
            h: Responsibilities. Shape (n_components, num_points).
        """
        self.weights_ = torch.mean(h, dim=1)

    def _update_mean(self, X: Tensor, h: Tensor) -> None:
        """M-step: update component means from data and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        num_frames, _, num_features = X.shape
        X = X[:, None].repeat(1, self._n_components, 1, 1)
        h = h[None, ..., None].repeat(num_frames, 1, 1, num_features)
        
        numerator = torch.sum(h * X, dim=2)
        denominator = torch.sum(h, dim=2)
        means = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))
        self.means_ = means

    def _update_covariances_(self, X: Tensor, h: Tensor) -> None:
        """M-step: update component covariances from data, means, and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        x_centered = X[..., None, :] - self.means_[:, None]
        prod = torch.einsum("ijkh,ijkl,kj->ikhl", x_centered.float(), x_centered.float(), h)
        denom = h.sum(dim=1)[None, :, None, None]
        cov = torch.where(denom != 0, prod / denom, torch.zeros_like(prod))
        self.covariances_ = cov

    def _log_likelihood(self, densities: Tensor) -> float:
        """Compute the log-likelihood of the data given current model parameters.

        Args:
            densities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            float: Log-likelihood value.
        """
        densities = torch.prod(densities, dim=0)
        weighted_sum = self.weights_ @ densities
        weighted_sum += torch.ones_like(weighted_sum) * 1e-18 
        ll = torch.sum(torch.log(weighted_sum)).item()
        return ll
