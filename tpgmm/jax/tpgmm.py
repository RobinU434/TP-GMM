from typing import Any, Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

from tpgmm._core.tpgmm import BaseTPGMM


@jax.jit
def _gauss_pdf_jit(X: Array, means: Array, covariances: Array) -> Array:
    """JIT-compiled Gaussian PDF computation."""
    diff = X[:, None, :, :] - means[:, :, None, :]
    cov_inv = jnp.linalg.inv(covariances)
    mahal = jnp.einsum("fknd,fkde,fkne->fkn", diff, cov_inv, diff)
    D = X.shape[-1]
    det = jnp.linalg.det(covariances)
    norm = jnp.sqrt((2 * jnp.pi) ** D * det)
    return jnp.exp(-0.5 * mahal) / norm[:, :, None]


@partial(jax.jit, static_argnums=(5,))
def _em_step_jit(
    X: Array,
    means: Array,
    covariances: Array,
    cov_reg_matrix: Array,
    weights: Array,
    n_components: int,
) -> Tuple[Array, Array, Array, Array, float]:
    """JIT-compiled single EM iteration returning updated parameters."""
    # gauss_pdf
    reg_cov = covariances + cov_reg_matrix
    probs = _gauss_pdf_jit(X, means, reg_cov)

    # E-step: update h
    cluster_probs = jnp.prod(probs, axis=0)
    numerator = (weights * cluster_probs.T).T
    denominator = jnp.sum(numerator, axis=0)
    h = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(numerator))

    # M-step: update weights
    new_weights = jnp.mean(h, axis=1)

    # M-step: update means
    num_frames, _, num_features = X.shape
    X_expanded = jnp.tile(X[:, None], (1, n_components, 1, 1))
    h_expanded = jnp.tile(h[None, ..., None], (num_frames, 1, 1, num_features))
    mean_num = jnp.sum(h_expanded * X_expanded, axis=2)
    mean_den = jnp.sum(h_expanded, axis=2)
    new_means = jnp.where(mean_den != 0, mean_num / mean_den, jnp.zeros_like(mean_num))

    # M-step: update covariances
    x_centered = X[..., None, :] - new_means[:, None]
    prod = jnp.einsum("ijkh,ijkl,kj->ikhl", x_centered, x_centered, h)
    h_denom = h.sum(axis=1)[None, :, None, None]
    new_cov = jnp.where(h_denom != 0, prod / h_denom, jnp.zeros_like(prod))

    # log-likelihood
    new_reg_cov = new_cov + cov_reg_matrix
    new_probs = _gauss_pdf_jit(X, new_means, new_reg_cov)
    densities = jnp.prod(new_probs, axis=0)
    weighted_sum = new_weights @ densities + 1e-18
    ll = jnp.sum(jnp.log(weighted_sum))

    return new_weights, new_means, new_cov, new_probs, ll


class TPGMM(BaseTPGMM):
    """JAX implementation of the Task Parameterized Gaussian Mixture Model.

    See :class:`tpgmm._core.tpgmm.BaseTPGMM` for full documentation.
    """

    def __init__(
        self,
        n_components: int,
        threshold: float = 1e-7,
        max_iter: int = 100,
        min_iter: int = 5,
        weights_init: Array = None,
        means_init: Array = None,
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

    def fit(self, X: Array) -> None:
        X = jnp.asarray(X)
        if self._verbose:
            print("Started KMeans clustering")
        self.means_, self.covariances_ = self._k_means(X)
        self._cov_reg_matrix = jnp.eye(self.covariances_.shape[-1]) * self._reg_factor
        # broadcast to covariances shape
        self._cov_reg_matrix = jnp.broadcast_to(
            self._cov_reg_matrix, self.covariances_.shape
        )

        if self._verbose:
            print("finished KMeans clustering")

        if self.weights_ is None:
            self.weights_ = jnp.ones(self._n_components) / self._n_components

        if self._verbose:
            print("Start expectation maximization")

        # Warm up JIT on first call
        probabilities = self.gauss_pdf(X)
        log_likelihood = self._log_likelihood(probabilities)

        for epoch_idx in range(self._max_iter):
            weights, means, covariances, probabilities, updated_log_likelihood = (
                _em_step_jit(
                    X,
                    self.means_,
                    self.covariances_,
                    self._cov_reg_matrix,
                    self.weights_,
                    self._n_components,
                )
            )

            self.weights_ = weights
            self.means_ = means
            self.covariances_ = covariances
            # Ensure reg matrix matches new covariance shape
            self._cov_reg_matrix = jnp.broadcast_to(
                jnp.eye(covariances.shape[-1]) * self._reg_factor,
                covariances.shape,
            )

            updated_log_likelihood = updated_log_likelihood.item()

            difference = updated_log_likelihood - log_likelihood
            if jnp.isnan(difference):
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

    def predict(self, X: Array) -> Array:
        probabilities = self.predict_proba(X)
        labels = jnp.argmax(probabilities, axis=1)
        return labels

    def predict_proba(self, X: Array) -> Array:
        frame_probs = self.gauss_pdf(X)
        probabilities = jnp.prod(frame_probs, axis=0).T
        return probabilities

    def silhouette_score(self, X: Array) -> float:
        labels = self.predict(X)
        scores = np.empty(X.shape[0])
        for frame_idx in range(X.shape[0]):
            scores[frame_idx] = metrics.silhouette_score(
                np.asarray(X[frame_idx]), np.asarray(labels)
            )
        weights = np.tile(np.asarray(self.weights_)[:, None], (1, X.shape[0]))
        weighted_sum = (weights @ scores) / (np.asarray(self.weights_) * X.shape[0])
        return weighted_sum.mean()

    def score(self, X: Array) -> float:
        probabilities = self.gauss_pdf(X)
        return self._log_likelihood(probabilities)

    def bic(self, X: Array) -> float:
        num_points = X.shape[1]
        log_likelihood = self.score(X)
        return -2 * log_likelihood + jnp.log(num_points) * self._num_params()

    def aic(self, X: Array) -> float:
        log_likelihood = self.score(X)
        return -2 * log_likelihood + 2 * self._num_params()

    def gauss_pdf(self, X: Array) -> Array:
        covariances = self.covariances_ + self._cov_reg_matrix
        return _gauss_pdf_jit(X, self.means_, covariances)

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

    def _k_means(self, X: Array) -> Tuple[Array, Array]:
        """Initialize means and covariances using K-Means clustering.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).

        Returns:
            Tuple[Array, Array]: Initial means (num_frames, n_components, num_features)
                and covariances (num_frames, n_components, num_features, num_features).
        """
        num_frames, _, num_features = X.shape
        means_list = []
        cov_list = []
        for frame_idx in range(num_frames):
            frame_data = np.asarray(X[frame_idx])
            self._k_means_algo.fit(frame_data)
            means_list.append(self._k_means_algo.cluster_centers_)
            frame_covs = []
            for cluster_idx in range(self._n_components):
                data_idx = np.argwhere(
                    self._k_means_algo.labels_ == cluster_idx
                ).squeeze()
                frame_covs.append(np.cov(frame_data[data_idx].T))
            cov_list.append(np.stack(frame_covs))
        means = jnp.array(np.stack(means_list))
        covariances = jnp.array(np.stack(cov_list))
        reg_matrix = jnp.eye(num_features) * self._reg_factor
        covariances = covariances + reg_matrix
        return means, covariances

    def _update_h(self, probabilities: Array) -> Array:
        """E-step: compute component responsibilities from probability densities.

        Args:
            probabilities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            Array: Responsibilities. Shape (n_components, num_points).
        """
        cluster_probs = jnp.prod(probabilities, axis=0)
        numerator = (self.weights_ * cluster_probs.T).T
        denominator = jnp.sum(numerator, axis=0)
        h = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(numerator))
        return h

    def _update_weights(self, h: Array) -> None:
        """M-step: update component weights from responsibilities.

        Args:
            h: Responsibilities. Shape (n_components, num_points).
        """
        self.weights_ = jnp.mean(h, axis=1)

    def _update_mean(self, X: Array, h: Array) -> None:
        """M-step: update component means from data and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        num_frames, _, num_features = X.shape
        X_expanded = jnp.tile(X[:, None], (1, self._n_components, 1, 1))
        h_expanded = jnp.tile(h[None, ..., None], (num_frames, 1, 1, num_features))

        numerator = jnp.sum(h_expanded * X_expanded, axis=2)
        denominator = jnp.sum(h_expanded, axis=2)
        means = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(numerator))
        self.means_ = means

    def _update_covariances_(self, X: Array, h: Array) -> None:
        """M-step: update component covariances from data, means, and responsibilities.

        Args:
            X: Data tensor. Shape (num_frames, num_points, num_features).
            h: Responsibilities. Shape (n_components, num_points).
        """
        x_centered = X[..., None, :] - self.means_[:, None]
        prod = jnp.einsum("ijkh,ijkl,kj->ikhl", x_centered, x_centered, h)
        h_denom = h.sum(axis=1)[None, :, None, None]
        cov = jnp.where(h_denom != 0, prod / h_denom, jnp.zeros_like(prod))
        self.covariances_ = cov

    def _log_likelihood(self, densities: Array) -> float:
        """Compute the log-likelihood of the data given current model parameters.

        Args:
            densities: Gaussian PDF values. Shape (num_frames, n_components, num_points).

        Returns:
            float: Log-likelihood value.
        """
        densities = jnp.prod(densities, axis=0)
        weighted_sum = self.weights_ @ densities
        weighted_sum = weighted_sum + jnp.ones_like(weighted_sum) * 1e-18
        ll = jnp.sum(jnp.log(weighted_sum)).item()
        return ll
