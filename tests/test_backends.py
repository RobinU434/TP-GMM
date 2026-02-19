"""Tests for the multi-backend package structure and inheritance."""

from tpgmm._core.tpgmm import BaseTPGMM
from tpgmm._core.gmr import BaseGMR
from tpgmm._core.learning_modules import (
    LearningModule,
    ClassificationModule,
    RegressionModel,
)


class TestCoreInheritance:
    """Verify that base classes have the correct ABC hierarchy."""

    def test_base_tpgmm_inherits_classification_module(self):
        assert issubclass(BaseTPGMM, ClassificationModule)

    def test_base_gmr_inherits_regression_model(self):
        assert issubclass(BaseGMR, RegressionModel)

    def test_classification_module_inherits_learning_module(self):
        assert issubclass(ClassificationModule, LearningModule)

    def test_regression_model_inherits_learning_module(self):
        assert issubclass(RegressionModel, LearningModule)


class TestNumpyBackend:
    """Verify NumPy backend structure."""

    def test_tpgmm_inherits_base(self):
        from tpgmm.numpy.tpgmm import TPGMM
        assert issubclass(TPGMM, BaseTPGMM)

    def test_gmr_inherits_base(self):
        from tpgmm.numpy.gmr import GaussianMixtureRegression
        assert issubclass(GaussianMixtureRegression, BaseGMR)

    def test_package_imports(self):
        from tpgmm.numpy import TPGMM, GaussianMixtureRegression
        assert TPGMM is not None
        assert GaussianMixtureRegression is not None


class TestTorchBackend:
    """Verify PyTorch backend structure."""

    def test_tpgmm_inherits_base(self):
        from tpgmm.torch.tpgmm import TPGMM
        assert issubclass(TPGMM, BaseTPGMM)

    def test_gmr_inherits_base(self):
        from tpgmm.torch.gmr import GaussianMixtureRegression
        assert issubclass(GaussianMixtureRegression, BaseGMR)

    def test_package_imports(self):
        from tpgmm.torch import TPGMM, GaussianMixtureRegression
        assert TPGMM is not None
        assert GaussianMixtureRegression is not None


class TestJaxBackend:
    """Verify JAX backend structure."""

    def test_tpgmm_inherits_base(self):
        from tpgmm.jax.tpgmm import TPGMM
        assert issubclass(TPGMM, BaseTPGMM)

    def test_gmr_inherits_base(self):
        from tpgmm.jax.gmr import GaussianMixtureRegression
        assert issubclass(GaussianMixtureRegression, BaseGMR)

    def test_package_imports(self):
        from tpgmm.jax import TPGMM, GaussianMixtureRegression
        assert TPGMM is not None
        assert GaussianMixtureRegression is not None


class TestTopLevelImports:
    """Verify backward-compatible top-level imports."""

    def test_tpgmm_import(self):
        from tpgmm import TPGMM
        from tpgmm.torch.tpgmm import TPGMM as TorchTPGMM
        assert TPGMM is TorchTPGMM

    def test_gmr_import(self):
        from tpgmm import GaussianMixtureRegression
        from tpgmm.numpy.gmr import GaussianMixtureRegression as NumpyGMR
        assert GaussianMixtureRegression is NumpyGMR

    def test_legacy_tpgmm_path(self):
        from tpgmm.tpgmm.tpgmm import TPGMM
        from tpgmm.torch.tpgmm import TPGMM as TorchTPGMM
        assert TPGMM is TorchTPGMM

    def test_legacy_gmr_path(self):
        from tpgmm.gmr.gmr import GaussianMixtureRegression
        from tpgmm.numpy.gmr import GaussianMixtureRegression as NumpyGMR
        assert GaussianMixtureRegression is NumpyGMR


class TestNumpyTPGMMIntegration:
    """Verify NumPy TPGMM fit/predict/score work end-to-end."""

    def test_fit_and_predict(self):
        import numpy as np
        from tpgmm.numpy.tpgmm import TPGMM
        np.random.seed(0)
        X = np.random.randn(2, 40, 3)
        model = TPGMM(n_components=2, max_iter=5)
        model.fit(X)
        assert model.weights_.shape == (2,)
        assert model.means_.shape == (2, 2, 3)
        labels = model.predict(X)
        assert labels.shape == (40,)

    def test_score_bic_aic(self):
        import numpy as np
        from tpgmm.numpy.tpgmm import TPGMM
        np.random.seed(0)
        X = np.random.randn(2, 40, 3)
        model = TPGMM(n_components=2, max_iter=5)
        model.fit(X)
        assert isinstance(model.score(X), float)
        assert isinstance(model.bic(X), float)
        assert isinstance(model.aic(X), float)


class TestTorchGMRIntegration:
    """Verify Torch GMR fit/predict work end-to-end."""

    def test_fit_and_predict(self):
        import torch
        import numpy as np
        from tpgmm.torch.tpgmm import TPGMM
        from tpgmm.torch.gmr import GaussianMixtureRegression
        torch.manual_seed(0)
        np.random.seed(0)
        X = torch.randn(2, 40, 3)
        tpgmm = TPGMM(n_components=2, max_iter=5)
        tpgmm.fit(X)
        gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0])
        translation = np.array([[0.0, 0.0], [1.0, 1.0]])
        rotation = np.array([np.eye(2), np.eye(2)])
        gmr.fit(torch.tensor(translation, dtype=torch.float32),
                torch.tensor(rotation, dtype=torch.float32))
        input_data = torch.linspace(0, 1, 10).unsqueeze(1)
        mu, cov = gmr.predict(input_data)
        assert mu.shape == (10, 2)
        assert cov.shape == (10, 2, 2)


class TestJaxTPGMMIntegration:
    """Verify JAX TPGMM fit/predict/score work end-to-end."""

    def test_fit_and_predict(self):
        import numpy as np
        import jax.numpy as jnp
        from tpgmm.jax.tpgmm import TPGMM
        np.random.seed(0)
        X = jnp.array(np.random.randn(2, 40, 3))
        model = TPGMM(n_components=2, max_iter=5)
        model.fit(X)
        assert model.weights_.shape == (2,)
        assert model.means_.shape == (2, 2, 3)
        labels = model.predict(X)
        assert labels.shape == (40,)

    def test_score_bic_aic(self):
        import numpy as np
        import jax.numpy as jnp
        from tpgmm.jax.tpgmm import TPGMM
        np.random.seed(0)
        X = jnp.array(np.random.randn(2, 40, 3))
        model = TPGMM(n_components=2, max_iter=5)
        model.fit(X)
        assert isinstance(model.score(X), float)
        assert isinstance(float(model.bic(X)), float)
        assert isinstance(float(model.aic(X)), float)


class TestJaxGMRIntegration:
    """Verify JAX GMR fit/predict work end-to-end."""

    def test_fit_and_predict(self):
        import numpy as np
        import jax.numpy as jnp
        from tpgmm.jax.tpgmm import TPGMM
        from tpgmm.jax.gmr import GaussianMixtureRegression
        np.random.seed(0)
        X = jnp.array(np.random.randn(2, 40, 3))
        tpgmm = TPGMM(n_components=2, max_iter=5)
        tpgmm.fit(X)
        gmr = GaussianMixtureRegression.from_tpgmm(tpgmm, input_idx=[0])
        translation = np.array([[0.0, 0.0], [1.0, 1.0]])
        rotation = np.array([np.eye(2), np.eye(2)])
        gmr.fit(jnp.array(translation), jnp.array(rotation))
        input_data = jnp.linspace(0, 1, 10).reshape(-1, 1)
        mu, cov = gmr.predict(input_data)
        assert mu.shape == (10, 2)
        assert cov.shape == (10, 2, 2)
