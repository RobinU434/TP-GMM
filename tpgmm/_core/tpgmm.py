from abc import abstractmethod
from typing import Any, Dict, Tuple

from tpgmm._core.learning_modules import ClassificationModule


class BaseTPGMM(ClassificationModule):
    """Abstract base class for Task Parameterized Gaussian Mixture Model.

    This class defines the common interface and documentation for all TPGMM backend
    implementations (NumPy, PyTorch, JAX).

    It implements an Expectation Maximization Algorithm with:

    E-Step:
    \\f[
        h_{t, i} = \\frac{\\pi_i \\prod_{j=1}^P \\mathcal{N}\\left(X_t^{(j)} \\mid \\mu_i^{(j)}, \\Sigma_i^{(j)}\\right)}{\\sum_{k=1}^K \\pi_k \\prod_{j=1}^P \\mathcal{N}\\left(X_t^{(j)} \\mid \\mu_k^{(j)}, \\Sigma_k^{(j)}\\right)}
    \\f]

    M-Step:
    \\f[
        \\pi_i \\leftarrow \\frac{\\sum_{t=1}^N h_{t, i}}{N}
    \\f]
    \\f[
        \\mu_i^{(j)} \\leftarrow \\frac{\\sum_{t=1}^N h_{t, i} X_t^{(j)}}{\\sum_{t=1}^N h_{t, i}}
    \\f]
    \\f[
        \\Sigma_i^{(j)} \\leftarrow \\frac{\\sum_{t=1}^N h_{t, i} \\left(X_t^{(j)} - \\mu_i^{(j)}\\right)\\left(X_t^{(j)} - \\mu_i^{(j)}\\right)^T}{\\sum_{t=1}^N h_{t, i}}
    \\f]

    The optimization criterion is the log-likelihood:
    \\f[
        LL = \\frac{\\sum_{t=1}^N \\log\\left(\\sum_{k=1}^K \\pi_k \\prod_{j=1}^J\\mathcal{N}\\left(X_t^{(j)} \\mid \\mu_k^{(j)}, \\Sigma_k^{(j)}\\right)\\right)}{N}
    \\f]

    Variable explanation:
    \\f$N\\f$ ... number of components
    \\f$\\pi\\f$ ... weights between components
    \\f$i\\f$ ... component index
    \\f$j\\f$ ... frame index (like pick or place frame)
    \\f$\\mu\\f$ ... mean
    \\f$\\Sigma\\f$ ... variance / covariance matrix
    \\f$LL\\f$ ... log likelihood

    Examples:
        >>> trajectories = load_trajectories()  # shape: (num_reference_frames, num_points, num_features)
        >>> tpgmm = TPGMM(n_components=5)
        >>> tpgmm.fit(trajectories)

    Args:
        n_components (int): Number of Gaussian components.
        threshold (float): Threshold to break from EM algorithm. Defaults to 1e-7.
        max_iter (int): Maximum EM iterations. Defaults to 100.
        min_iter (int): Minimum EM iterations. Defaults to 5.
        weights_init: Initial weights. If set, replaces K-Means initialization. Defaults to None.
        means_init: Initial means. If set, replaces K-Means initialization. Defaults to None.
        reg_factor (float): Regularization factor for covariance matrix. Defaults to 1e-5.
        verbose (bool): Triggers print of learning stats. Defaults to False.

    Attributes:
        weights_: Weights between gaussian components. Shape (n_components,).
        means_: Mean matrix for each frame and component. Shape (num_frames, n_components, num_features).
        covariances_: Covariance matrix for each frame and component. Shape (num_frames, n_components, num_features, num_features).
    """

    def __init__(
        self,
        n_components: int,
        threshold: float = 1e-7,
        max_iter: int = 100,
        min_iter: int = 5,
        weights_init=None,
        means_init=None,
        reg_factor: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        super().__init__(n_components)
        self._max_iter = max_iter
        self._min_iter = min_iter
        self._threshold = threshold
        self._reg_factor = reg_factor
        self._verbose = verbose

        self.weights_ = weights_init
        self.means_ = means_init
        self.covariances_ = None

    @abstractmethod
    def fit(self, X) -> None:
        """Fit the TPGMM model using K-Means initialization and EM algorithm.

        Args:
            X: Data tensor. Expected shape: (num_frames, num_points, num_features).
        """
        ...

    @abstractmethod
    def predict(self, X):
        """Predict cluster labels for each data point in X.

        Args:
            X: Data in local reference frames. Shape (num_frames, num_points, num_features).

        Returns:
            The label for each data-point. Shape (num_points,).
        """
        ...

    @abstractmethod
    def predict_proba(self, X):
        """Predict cluster probabilities for each data point.

        Args:
            X: Data in local reference frames. Shape (num_frames, num_points, num_features).

        Returns:
            Cluster probabilities for each data_point. Shape (num_points, num_components).
        """
        ...

    @abstractmethod
    def score(self, X) -> float:
        """Calculate log likelihood score given data.

        Args:
            X: Data tensor. Expected shape (num_frames, num_points, num_features).

        Returns:
            float: Log likelihood of given data.
        """
        ...

    @abstractmethod
    def gauss_pdf(self, X):
        """Calculate gaussian probability density for a given data set.

        Args:
            X: Data. Shape (num_frames, num_points, num_features).

        Returns:
            Probability tensor. Shape (num_frames, n_components, num_points).
        """
        ...

    def bic(self, X) -> float:
        """Calculate the Bayesian Information Criterion.

        Args:
            X: Data tensor. Expected shape (num_frames, num_points, num_features).

        Returns:
            float: BIC score.
        """
        ...

    def aic(self, X) -> float:
        """Calculate the Akaike Information Criterion.

        Args:
            X: Data tensor. Expected shape (num_frames, num_points, num_features).

        Returns:
            float: AIC score.
        """
        ...

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
