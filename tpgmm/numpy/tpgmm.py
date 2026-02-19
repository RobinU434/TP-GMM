from typing import Any, Dict, Tuple

import numpy as np
from numpy import ndarray
from sklearn import metrics
from sklearn.cluster import KMeans

from tpgmm._core.tpgmm import BaseTPGMM
from tpgmm._core.arrays import identity_like


class TPGMM(BaseTPGMM):
    """NumPy implementation of the Task Parameterized Gaussian Mixture Model.

    See :class:`tpgmm._core.tpgmm.BaseTPGMM` for full documentation.
    """

    def __init__(
        self,
        n_components: int,
        threshold: float = 1e-7,
        max_iter: int = 100,
        min_iter: int = 5,
        weights_init: ndarray = None,
        means_init: ndarray = None,
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

    def fit(self, X: ndarray) -> None:
        if self._verbose:
            print("Started KMeans clustering")
        self.means_, self.covariances_ = self._k_means(X)
        self._cov_reg_matrix = identity_like(self.covariances_) * self._reg_factor

        if self._verbose:
            print("finished KMeans clustering")

        if self.weights_ is None:
            self.weights_ = np.ones(self._n_components) / self._n_components

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

            difference = updated_log_likelihood - log_likelihood
            if np.isnan(difference):
                print(Warning("improvement is nan. Abort fit"))
                return False

            if self._verbose:
                print(
                    f"Log likelihood: {updated_log_likelihood} improvement {difference}"
                )

            if (
                difference < self._threshold and epoch_idx >= self._min_iter
            ) or epoch_idx > self._max_iter:
                break

            log_likelihood = updated_log_likelihood

    def predict(self, X: ndarray) -> ndarray:
        probabilities = self.predict_proba(X)
        labels = np.argmax(probabilities, axis=1)
        return labels

    def predict_proba(self, X: ndarray) -> ndarray:
        frame_probs = self.gauss_pdf(X)
        probabilities = np.prod(frame_probs, axis=0).T
        return probabilities

    def silhouette_score(self, X: ndarray) -> float:
        labels = self.predict(X)
        scores = np.empty(X.shape[0])
        for frame_idx in range(X.shape[0]):
            scores[frame_idx] = metrics.silhouette_score(X[frame_idx], labels)
        weights = np.tile(self.weights_[:, None], (1, X.shape[0]))
        weighted_sum = (weights @ scores) / (self.weights_ * X.shape[0])
        return weighted_sum.mean()

    def score(self, X: ndarray) -> float:
        probabilities = self.gauss_pdf(X)
        score = self._log_likelihood(probabilities)
        return score

    def bic(self, X: ndarray) -> float:
        num_points = X.shape[1]
        log_likelihood = self.score(X)
        bic = -2 * log_likelihood + np.log(num_points) * self._num_params()
        return bic

    def aic(self, X: ndarray) -> float:
        log_likelihood = self.score(X)
        aic = -2 * log_likelihood + 2 * self._num_params()
        return aic

    def gauss_pdf(self, X: ndarray) -> ndarray:
        covariances = self.covariances_ + self._cov_reg_matrix
        # X: (F, N, D) -> (F, 1, N, D);  means: (F, K, D) -> (F, K, 1, D)
        diff = X[:, None, :, :] - self.means_[:, :, None, :]  # (F, K, N, D)
        cov_inv = np.linalg.inv(covariances)  # (F, K, D, D)
        mahal = np.einsum("fknd,fkde,fkne->fkn", diff, cov_inv, diff)
        D = X.shape[-1]
        det = np.linalg.det(covariances)  # (F, K)
        norm = np.sqrt((2 * np.pi) ** D * det)  # (F, K)
        return np.exp(-0.5 * mahal) / norm[:, :, None]

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
        return num_mean_params + num_cov_params + num_weight_params

    def _k_means(self, X: ndarray) -> Tuple[ndarray, ndarray]:
        """Initialize means and covariances using K-Means clustering.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).

        Returns:
            Tuple[ndarray, ndarray]: Initial means (num_frames, n_components, num_features)
                and covariances (num_frames, n_components, num_features, num_features).
        """
        num_frames, _, num_features = X.shape
        means = np.empty((num_frames, self._n_components, num_features))
        covariances = np.empty(
            (num_frames, self._n_components, num_features, num_features)
        )
        for frame_idx, frame_data in enumerate(X):
            self._k_means_algo.fit(frame_data)
            means[frame_idx] = self._k_means_algo.cluster_centers_
            for cluster_idx in range(self._n_components):
                data_idx = np.argwhere(
                    self._k_means_algo.labels_ == cluster_idx
                ).squeeze()
                covariances[frame_idx, cluster_idx] = np.cov(frame_data[data_idx].T)

        reg_matrix = identity_like(covariances) * self._reg_factor
        covariances += reg_matrix
        return means, covariances

    def _update_h(self, probabilities: ndarray) -> ndarray:
        """E-step: compute component responsibilities from probability densities.

        Args:
            probabilities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            ndarray: Responsibilities. Shape (n_components, num_points).
        """
        cluster_probs = np.prod(probabilities, axis=0)
        numerator = (self.weights_ * cluster_probs.T).T
        denominator = np.sum(numerator, axis=0)
        h = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        return h

    def _update_weights(self, h: ndarray) -> None:
        """M-step: update component weights from responsibilities.

        Args:
            h: Responsibilities. Shape (n_components, num_points).
        """
        self.weights_ = np.mean(h, axis=1)

    def _update_mean(self, X: ndarray, h: ndarray) -> None:
        """M-step: update component means from data and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        num_frames, _, num_features = X.shape
        X_expanded = np.tile(X[:, None], (1, self._n_components, 1, 1))
        h_expanded = np.tile(h[None, ..., None], (num_frames, 1, 1, num_features))

        numerator = np.sum(h_expanded * X_expanded, axis=2)
        denominator = np.sum(h_expanded, axis=2)
        means = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        self.means_ = means

    def _update_covariances_(self, X: ndarray, h: ndarray) -> None:
        """M-step: update component covariances from data, means, and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        x_centered = X[..., None, :] - self.means_[:, None]
        prod = np.einsum("ijkh,ijkl,kj->ikhl", x_centered, x_centered, h)
        denom = h.sum(axis=1)[None, :, None, None]
        cov = np.divide(prod, denom, out=np.zeros_like(prod), where=denom != 0)
        self.covariances_ = cov

    def _log_likelihood(self, densities: ndarray) -> float:
        """Compute the log-likelihood of the data given current model parameters.

        Args:
            densities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            float: Log-likelihood value.
        """
        densities = np.prod(densities, axis=0)
        weighted_sum = self.weights_ @ densities
        weighted_sum += np.ones_like(weighted_sum) * 1e-18
        ll = np.sum(np.log(weighted_sum)).item()
        return ll
