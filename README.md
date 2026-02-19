# TPGMM — Task Parameterized Gaussian Mixture Models

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A Python library for learning and reproducing robotic trajectories using **Task Parameterized Gaussian Mixture Models** (TP-GMM) and **Gaussian Mixture Regression** (GMR). Supports three computational backends — **NumPy**, **PyTorch**, and **JAX** — with a unified API.

<p align="center">
  <img src="figures/TPGMM-wrokflow.png" alt="TPGMM Workflow" width="700">
</p>

---

## Features

- **Multi-backend support** — identical API across NumPy, PyTorch, and JAX
- **JIT-accelerated JAX backend** — full EM loop compiled via `@jax.jit` for high-throughput workloads
- **Vectorized Gaussian PDF** — einsum-based implementation across all backends, no Python loops in the hot path
- **Model selection utilities** — BIC, AIC, silhouette score, and Davies–Bouldin index
- **GMR regression** — condition on input dimensions to generate smooth output trajectories
- **Modular architecture** — shared abstract base classes ensure consistent behaviour and documentation

## Quick Start

```python
from tpgmm.torch import TPGMM, GaussianMixtureRegression

# Fit a TP-GMM on demonstration trajectories
# X shape: (num_frames, num_points, num_features)
model = TPGMM(n_components=5)
model.fit(X)

# Reproduce a trajectory via Gaussian Mixture Regression
gmr = GaussianMixtureRegression.from_tpgmm(model, input_idx=[0])
gmr.fit(translation, rotation_matrix)
mu, sigma = gmr.predict(query_points)
```

All three backends expose the same interface:

```python
from tpgmm.numpy import TPGMM, GaussianMixtureRegression
from tpgmm.torch import TPGMM, GaussianMixtureRegression
from tpgmm.jax   import TPGMM, GaussianMixtureRegression
```

---

## Installation

### From PyPI

```bash
# Base installation (NumPy backend)
pip install tpgmm

# With PyTorch backend
pip install tpgmm[torch]

# With JAX backend
pip install tpgmm[jax]

# All backends + examples + dev tools
pip install tpgmm[all]
```

### For Development (UV)

This project uses [UV](https://github.com/astral-sh/uv) as its package manager.

```bash
git clone https://github.com/yourusername/TaskParameterizedGaussianMixtureModels.git
cd TaskParameterizedGaussianMixtureModels

# Install with all dependencies
uv sync --all-extras

# Or install selectively
uv sync --extra torch          # PyTorch backend
uv sync --extra jax            # JAX backend
uv sync --extra examples       # Jupyter + matplotlib
uv sync --extra dev            # pytest, ruff, black
```

---

## Package Structure

```
tpgmm/
├── _core/                  # Abstract bases & shared utilities
│   ├── tpgmm.py            #   BaseTPGMM
│   ├── gmr.py              #   BaseGMR
│   ├── learning_modules.py  #   LearningModule, ClassificationModule, RegressionModel
│   ├── arrays.py            #   Array helpers (subscript, identity_like, get_subarray)
│   └── stochastic.py        #   Multivariate Gaussian CDF
├── numpy/                  # NumPy backend
│   ├── tpgmm.py
│   └── gmr.py
├── torch/                  # PyTorch backend
│   ├── tpgmm.py
│   └── gmr.py
├── jax/                    # JAX backend (JIT-compiled EM)
│   ├── tpgmm.py
│   └── gmr.py
└── utils/                  # I/O, geometry, casting, plotting
```

All backend implementations inherit from `tpgmm._core`, ensuring a consistent interface and shared documentation.

---

## API Reference

### `TPGMM`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | `int` | — | Number of Gaussian components |
| `threshold` | `float` | `1e-7` | EM convergence threshold |
| `max_iter` | `int` | `100` | Maximum EM iterations |
| `min_iter` | `int` | `5` | Minimum EM iterations |
| `reg_factor` | `float` | `1e-5` | Covariance regularization factor |
| `verbose` | `bool` | `False` | Print learning statistics |

**Methods:**

| Method | Description |
|--------|-------------|
| `fit(X)` | Fit the model via K-Means initialization and EM. `X` shape: `(num_frames, num_points, num_features)` |
| `predict(X)` | Return cluster labels for each data point |
| `predict_proba(X)` | Return cluster probabilities per data point |
| `score(X)` | Log-likelihood of the data |
| `bic(X)` / `aic(X)` | Bayesian / Akaike Information Criterion |
| `gauss_pdf(X)` | Gaussian PDF across all frames and components |

**Fitted Attributes:** `weights_`, `means_`, `covariances_`

### `GaussianMixtureRegression`

| Parameter | Type | Description |
|-----------|------|-------------|
| `weights` | array | Component weights from a fitted TPGMM |
| `means` | array | Component means |
| `covariances` | array | Component covariances |
| `input_idx` | `list[int]` | Indices of input (conditioning) features |

**Methods:**

| Method | Description |
|--------|-------------|
| `from_tpgmm(tpgmm, input_idx)` | Class method — create a GMR from a fitted TPGMM |
| `fit(translation, rotation_matrix)` | Transform TP-GMM into a single GMM for the given task frame |
| `predict(input_data)` | Return `(mu, sigma)` — conditional mean and covariance |

---

## Examples

Example notebooks are provided in [`examples/`](examples/):

| Notebook | Backend | Description |
|----------|---------|-------------|
| [`example_numpy.ipynb`](examples/example_numpy.ipynb) | NumPy | Full TPGMM + GMR pipeline |
| [`example_torch.ipynb`](examples/example_torch.ipynb) | PyTorch | Full TPGMM + GMR pipeline |
| [`example_jax.ipynb`](examples/example_jax.ipynb) | JAX | Full TPGMM + GMR pipeline with JIT |

Each notebook walks through data loading, model fitting, model selection (BIC), GMR regression, and trajectory visualization.

```bash
# Run with pip
pip install tpgmm[examples,torch]
jupyter notebook examples/example_torch.ipynb

# Run with UV
uv sync --extra examples --extra torch
uv run jupyter notebook examples/example_torch.ipynb
```

---

## Testing

Tests use [pytest](https://docs.pytest.org/) with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) for runtime profiling.

```bash
# With pip
pip install tpgmm[dev,torch,jax]
pytest tests/

# With UV
uv sync --all-extras
uv run pytest tests/
```

The test suite covers:

- **Unit tests** — array utilities, casting, file I/O, geometry helpers
- **Backend tests** — inheritance structure, import paths, integration tests for all three backends
- **Runtime benchmarks** — comparative timing of NumPy, PyTorch, and JAX backends

---

## Performance

All backends use a vectorized einsum-based Gaussian PDF with no Python loops in the computation. The JAX backend additionally JIT-compiles the full EM iteration for maximum throughput.

Representative benchmark (single-threaded, CPU):

| Backend | Mean | Relative |
|---------|------|----------|
| JAX (JIT) | ~10 ms | **1.0×** |
| PyTorch | ~15 ms | 1.5× |
| NumPy | ~90 ms | 9× |

> Benchmarks are data- and hardware-dependent. Run `uv run pytest tests/test_runtime.py` to profile on your machine.

---

## Citation

This implementation is based on the TP-GMM framework described by Sylvain Calinon:

> S. Calinon, "A Tutorial on Task-Parameterized Movement Learning and Retrieval,"
> *Intelligent Service Robotics*, 2016.
> [Paper](https://calinon.ch/papers/Calinon-JIST2015.pdf)

---

## Acknowledgements

This project originated during an internship at [Fraunhofer Italia](https://www.fraunhofer.it/de.html) under the supervision of [Marco Todescato](https://www.linkedin.com/in/marco-todescato/?originalSubdomain=de) (summer 2023). Thank you for the great advice and discussions.

## Contributing

Contributions are welcome. Please:

1. Fork the repository and create a feature branch.
2. Ensure your changes include appropriate tests.
3. Open a pull request with a clear description of the change.

For larger changes, please open an issue first to discuss the approach.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
