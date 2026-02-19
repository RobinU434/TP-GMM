import numpy as np
import torch
import jax.numpy as jnp
from tpgmm.torch.tpgmm import TPGMM as Torch_TPGMM
from tpgmm.jax.tpgmm import TPGMM as JAX_TPGMM
from tpgmm.numpy.tpgmm import TPGMM as Numpy_TPGMM


ITERATIONS = 10
ROUNDS = 100

def test_single_run_numpy():
    data = np.load("./tests/benchmark_data.npy")
    tpgmm = Numpy_TPGMM(6)
    print(np.isinf(data).sum())
    tpgmm.fit(data)
    
def test_single_run_torch():
    data = np.load("./tests/benchmark_data.npy")
    tpgmm = Torch_TPGMM(6)
    print(np.isinf(data).sum())
    tpgmm.fit(torch.from_numpy(data))

def test_single_run_jax():
    data = np.load("./tests/benchmark_data.npy")
    tpgmm = JAX_TPGMM(6)
    print(np.isinf(data).sum())
    tpgmm.fit(data)



def test_runtime_numpy(benchmark):
    # load benchmark_data
    data = np.load("./tests/benchmark_data.npy")
    print(np.isinf(data).sum())
    
    tpgmm = Numpy_TPGMM(6)
    benchmark(tpgmm.fit, data)
    
def test_runtime_torch(benchmark):
    # load benchmark_data
    data = np.load("./tests/benchmark_data.npy")
    print(np.isinf(data).sum())
    
    tpgmm = Torch_TPGMM(6)
    benchmark(tpgmm.fit, torch.from_numpy(data))

def test_runtime_jax(benchmark):
    # load benchmark_data
    data = np.load("./tests/benchmark_data.npy")
    print(np.isinf(data).sum())
    data = jnp.array(data)
    tpgmm = JAX_TPGMM(6)
    benchmark(tpgmm.fit, data)
