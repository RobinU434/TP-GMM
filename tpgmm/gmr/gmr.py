import logging
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from numpy import ndarray

from tpgmm.utils.learning_modules import RegressionModel
from tpgmm.utils.stochastic import multivariate_gauss_cdf
from tpgmm.utils.arrays import get_subarray, identity_like
from tpgmm.tpgmm.tpgmm import TPGMM


class GaussianMixtureRegression(RegressionModel):
    """This class implements a gaussian mixture regression model.
    This model was described in 'A Tutorial on Task-Parameterized Movement Learning and Retrival' from S.Calinon at: https://calinon.ch/papers/Calinon-JIST2015.pdf

    The class fits a gaussian mixture regression on a given Gaussian Mixture model or a Task-Parameterized gaussian mixture model. Used equations are:
    \f[
        \mathcal{P}(\phi_t^\mathcal{O}|\phi_t^\mathcal{I}) \sim \sum_{i=1}^K h_i(\phi_t^\mathcal{I}) \mathcal{N}\left(\hat{\mu}_t^\mathcal{O}(\phi_t^\mathcal{I}), \hat{\Sigma}_t^\mathcal{O}\right)
    \f]
    \f[
        \hat{\mu}_i^\mathcal{O}(\phi_t^\mathcal{I}) = \mu_i^\mathcal{O} + \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}(\phi_t^\mathcal{I} - \mu_i^\mathcal{I})
    \f]
    \f[
        \hat{\Sigma}_t^\mathcal{O} = \Sigma_i^\mathcal{O} - \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}\Sigma_i^\mathcal{OI}
    \f]
    \f[
        h_i(\phi_t^\mathcal{I}) = \frac{\pi_i \mathcal{N}(\phi_t^\mathcal{I} \mid \mu_i^\mathcal{I}, \Sigma_i^\mathcal{I})}{\sum_k^K \pi_k \mathcal{N}\mathcal(\phi_t^\mathcal{I} \mid \mu_k^\mathcal{I}, \Sigma_k^\mathcal{I})}
    \f]

    Example:
    >>> # trajectory data in local reference frames with shape: (num_reference_frames, num_points, 4)
    >>> # last dimension could be for example: x, y, z, time
    >>> # in case you got multiple trajectories: concatenate all of them in axis 1
    >>> trajectories = load_trajectories()
    >>> tpgmm = TPGMM(n_components=5)
    >>> tpgmm.fit(trajectories)
    >>> # gmr for the first reference frame
    >>> gmr = GaussianMixtureRegression(weights=tpgmm.weights_, means=tpgmm.means_[0], covariances=tpgmm.covariances_[0], input_idx=[3])
    >>> gmr.fit(trajectory[0])
    """

    def __init__(
        self,
        weights: ndarray,
        means: ndarray,
        covariances: ndarray,
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

        # parameters from equation 5 and 6
        self.xi_: ndarray  # shape: (num_components, num_features)
        self.sigma_: ndarray  # shape: (num_components, num_features, num_features)

    @classmethod
    def from_tpgmm(
        cls: "GaussianMixtureRegression", tpgmm: TPGMM, input_idx: Iterable[int]
    ) -> "GaussianMixtureRegression":
        result = cls(
            weights=tpgmm.weights_.numpy(),
            means=tpgmm.means_.numpy(),
            covariances=tpgmm.covariances_.numpy(),
            input_idx=input_idx,
        )
        return result

    def _equation_5(self, translation: ndarray, rotation_matrix: ndarray):
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
        The mechanics are described by:
        \f[
            \mathcal{P}(\phi_t^\mathcal{O}|\phi_t^\mathcal{I}) \sim \sum_{i=1}^K h_i(\phi_t^\mathcal{I}) \mathcal{N}\left(\hat{\mu}_t^\mathcal{O}(\phi_t^\mathcal{I}), \hat{\Sigma}_t^\mathcal{O}\right)
        \f]
        in:
        self.h()
        \f[
            h_i(\phi_t^\mathcal{I}) = \frac{\pi_i \mathcal{N}(\phi_t^\mathcal{I} \mid \mu_i^\mathcal{I}, \Sigma_i^\mathcal{I})}{\sum_k^K \pi_k \mathcal{N}\mathcal(\phi_t^\mathcal{I} \mid \mu_k^\mathcal{I}, \Sigma_k^\mathcal{I})}
        \f]
        in:
        self.mu_hat()
        \f[
            \hat{\mu}_i^\mathcal{O}(\phi_t^\mathcal{I}) = \mu_i^\mathcal{O} + \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}(\phi_t^\mathcal{I} - \mu_i^\mathcal{I})
        \f]
        in:
        self.sigma_hat()
        \f[
            \hat{\Sigma}_t^\mathcal{O} = \Sigma_i^\mathcal{O} - \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}\Sigma_i^\mathcal{OI}
        \f]

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
        """this function is inspired by formula 16 in Calinon paper
        returns an array with probabilities. Each probability at
        index i is the probability that a data point corresponds to a
        gaussian distribution in gaussian_mixture with index i
        \f[
            h_i(\phi_t^\mathcal{I}) = \frac{\pi_i \mathcal{N}(\phi_t^\mathcal{I} \mid \mu_i^\mathcal{I}, \Sigma_i^\mathcal{I})}{\sum_k^K \pi_k \mathcal{N}\mathcal(\phi_t^\mathcal{I} \mid \mu_k^\mathcal{I}, \Sigma_k^\mathcal{I})}
        \f]


        Args:
            data (ndarray): datapoints. Shape (num_points, num_input_features) Note: num features is in our case there is only one input feature (time) at index -1: (x, y, z, time)

        Returns:
            probability for each data point if it belongs to a cluster certain cluster. Shape (num_datapoints, num_components)
        """
        probabilities = []
        # iterate trough all components
        for component_input_mean, component_input_covariance in zip(
            self._tile_mean(self.xi_)[0], self._tile_covariance(self.sigma_)[0]
        ):
            # if multivariate_gauss_cdf is numerically unstable ... use multivariate gaussian from scipy
            # here the custom implementation is used because it is ~4 times faster
            probabilities.append(
                multivariate_gauss_cdf(
                    data, component_input_mean, component_input_covariance
                )
            )
        probabilities = np.stack(probabilities).T  # shape: (num_points, num_components)

        weighted_probs = probabilities * self.gmm_weights

        cluster_probs = (weighted_probs.T / np.sum(weighted_probs, axis=1)).T

        return cluster_probs

    def _mu_hat_out(self, input_data: ndarray) -> ndarray:
        """this function is inspired by formula 14 in Calinon paper
        \f[
            \hat{\mu}_i^\mathcal{O}(\phi_t^\mathcal{I}) = \mu_i^\mathcal{O} + \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}(\phi_t^\mathcal{I} - \mu_i^\mathcal{I})
        \f]

        Args:
            data (ndarray): datapoints. Shape (num_points, num_input_features). Note: num features is in our case = 1 -> (time)
        Returns:
            ndarray: mu_hat with shape: (num_points, num_clusters, num_output_features)
        """

        # expand datapoints to shape: [num_points, num_clusters, num_input_features]
        input_data = np.expand_dims(input_data, axis=1)
        input_data = np.tile(input_data, [1, self.num_components, 1])

        cov_i, _, cov_oi, _ = self._tile_covariance(self.sigma_)
        mean_input, mean_output = self._tile_mean(self.xi_)

        # shape: (num_points, num_components, num_input_features)
        centered_points = input_data - mean_input
        # shape: (num_components, num_output_features, num_input_features)
        cluster_mats = cov_oi @ np.linalg.inv(cov_i)

        # perform matrix multiplication
        # i: num_components, j: num_points, k:num_output_features, h: num_input_features
        # shape: (num_points, num_components, num_output_features)
        mu_hat = np.einsum("ikh,jih->jik", cluster_mats, centered_points)

        mu_hat = mu_hat + mean_output
        return mu_hat

    def _sigma_hat_out(self) -> ndarray:
        """this function is inspired by formula 15 in Calinon paper
        \f[
            \hat{\Sigma}_t^\mathcal{O} = \Sigma_i^\mathcal{O} - \Sigma_i^\mathcal{OI}\Sigma_i^{\mathcal{I}, -1}\Sigma_i^\mathcal{OI}
        \f]

        Returns:
            ndarray: shape (num_components, num_output_features, num_output_features)
        """
        cov_i, cov_io, cov_oi, cov_o = self._tile_covariance(self.sigma_)

        return cov_o - cov_oi @ np.linalg.inv(cov_i) @ cov_io

    def _pad(
        self, rotation_matrix: ndarray, translation: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """pads the given arguments into A = [[I_input, 0], [0, rotation_matrix]], b = [0_input, translation].T
        with I_input as the identity matrix with size of self.input_index and b_input as zeros with length of self.input_index

        Args:
            rotation_matrix (ndarray): rotation matrix for rotating into desired frames. Shape (num_frames, num_output_features, num_output_features)
            translation (ndarray): translation matrix for translating into desired frames. Shape (num_frames, num_output_features)
        Returns:
            Tuple[ndarray, ndarray]: padded rotation matrix and translation
                - rotation matrix shape: (num_frames, num_features, num_features)
                - translation matrix shape: (num_frames, num_features)
        """

        # pad rotation matrix
        num_frames, num_output_features, _ = rotation_matrix.shape
        identity = np.eye(self.num_input_features)
        identity = np.repeat(identity[None], num_frames, axis=0)

        zeros_io = np.zeros((num_frames, self.num_input_features, num_output_features))
        zeros_oi = zeros_io.swapaxes(-1, -2)
        padded_rot_mat = np.block([[identity, zeros_io], [zeros_oi, rotation_matrix]])

        # pad translation
        zeros_o = np.zeros((num_frames, self.num_input_features))
        padded_translation = np.concatenate([zeros_o, translation], axis=-1)

        return padded_rot_mat, padded_translation

    def _sort_by_input(self, data: ndarray, axes: Iterable[int] = (0,)) -> ndarray:
        """sorts model parameters in feature space

        Inside the feature dimension put all input features at the front and all output_features at the back But keep the internal order of all elements

        Args:
            data (ndarray): shape: (..., num_features, ....) | (..., num_features, num_features, ...)
            axes (Iterable[int]): at which dimensions are the features located. Defaults to 0.

        Returns:
            ndarray: restructured data array
        """
        sort_index = [*self.input_idx, *self.output_idx]
        sort_index = [sort_index for _ in axes]

        return get_subarray(data, axes, sort_index)

    def _revoke_sort_by_input(
        self, data: ndarray, axes: Iterable[int] = (0,)
    ) -> ndarray:
        """function to revoke the self.sort_by_input function

        Args:
            data (ndarray): shape: (..., num_features, ....) | (..., num_features, num_features, ...)
            axes (Iterable[int]): at which dimensions are the features located. Defaults to 0.

        Returns:
            ndarray: restructured data array
        """
        sort_index = np.empty(self.num_features, dtype=int)
        sort_index[self.input_idx] = range(self.num_input_features)
        sort_index[self.output_idx] = range(self.num_input_features, self.num_features)
        sort_index = [sort_index.tolist() for _ in axes]
        return get_subarray(data, axes, sort_index)

    def _tile_mean(self, mean: ndarray) -> Tuple[ndarray, ndarray]:
        """tiles mean into: \f$\mu^\mathcal{I}\f$ and \f$\mu^\mathcal{O}\f$

        Args;
            mean (ndarray): mean data with expected shape: (..., num_features)
        Returns:
            Tuple[ndarray, ndarray]:
                - input mean \f$\mu^\mathcal{I}\f$: means of all components input features. Shape: (..., len(self.input_idx))
                - output mean \f$\mu^\mathcal{O}\f$: means of all components output features. Shape: (..., all_features - len(self.input_idx))

        """
        return get_subarray(mean, axes=[-1], indices=[self.input_idx]), np.delete(
            mean, self.input_idx, -1
        )

    def _tile_covariance(
        self, cov_mat: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Tiles the covariance matrix into 4 parts.
        \f[
            \Sigma = [[\Sigma^\mathcal{I}, \Sigma^\mathcal{IO}], [\Sigma^\mathcal{OI}, \Sigma^\mathcal{O}]]
        \f]

        Args:
            cov_mat (ndarray): covariance matrix. shape (...., num_features, num_features)

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]:
            - input: \f$\Sigma^\mathcal{I}\f$. Shape (n_components, len(self.input_idx), len(self.input_idx))
            - input_output: \f$\Sigma^\mathcal{IO}\f$ Shape (n_components, 1, n_features - len(self.input_idx))
            - output_input: \f$\Sigma^\mathcal{OI}\f$ Shape (n_components, n_features - len(self.input_idx), len(self.input_idx))
            - output: \f$\Sigma^\mathcal{O}\f$ Shape (n_components, n_features - len(self.input_idx), n_features - len(self.input_idx))
        """
        feature_index = [-2, -1]
        cov_i = get_subarray(cov_mat, feature_index, [self.input_idx, self.input_idx])
        cov_io = get_subarray(cov_mat, feature_index, [self.input_idx, self.output_idx])
        cov_oi = get_subarray(cov_mat, feature_index, [self.output_idx, self.input_idx])
        cov_o = get_subarray(cov_mat, feature_index, [self.output_idx, self.output_idx])

        return (
            cov_i,
            cov_io,
            cov_oi,
            cov_o,
        )

    @property
    def num_input_features(self) -> int:
        """get number of input features.
        Example:
        if you have 4 features: x, y, z, time. You can define with input_idx=[3] time as the input feature.
        This function will return then 1

        Returns:
            int: number of input features
        """
        return len(self.input_idx)

    @property
    def num_output_features(self) -> int:
        """get number of output features.
        Example:
        if you have 4 features: x, y, z, time. You can define with input_idx=[3] time as the input feature.
        This function will return then 3 for x,y and z

        Returns:
            int: number of output features
        """
        return self.num_features - self.num_input_features

    @property
    def output_idx(self) -> List[int]:
        return np.setdiff1d(np.array(range(self.num_features)), self.input_idx).tolist()

    @property
    def config(self) -> Dict[str, Any]:
        return {}
