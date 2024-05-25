import logging
from typing import Iterable, Tuple

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def subscript(*args) -> Tuple[slice]:
    """Builds a slicing tuple to slice a multidimensional ArrayLike structure.

    Returns:
        Tuple[slice]: Tuple of slicing information.
    """
    # TODO: write tests
    selection = []
    for arg in args:
        if isinstance(arg, Iterable):
            selection.append(arg)
        elif isinstance(arg, int):
            selection.append(slice(arg, arg + 1))
        else:
            selection.append(slice(None))
    return tuple(selection)


def identity_like(array: ndarray | Tensor) -> ndarray:
    """Create an identity matrix with the same shape as the input array if the last two axes are the same.

    Args:
        array (ndarray | Tensor): Input array with shape (..., a, a), where a is the dimension of the covariance matrix.

    Returns:
        ndarray | Tensor: Identity matrix with the same shape as the input array, i.e., (..., a, a).
    """    
    array_shape = array.shape
    if array_shape[-1] != array_shape[-2]:
        logging.error(
            "Not possible to calculate a identity matrix because the last two axes are different"
        )
        return
    
    if isinstance(array, Tensor):
        identity = torch.zeros_like(array)
    else:
        identity = np.zeros_like(array)
    
    selection = [None] * len(array_shape)
    diagonal_index = list(range(array_shape[-1]))
    selection[-2] = diagonal_index
    selection[-1] = diagonal_index
    selection = subscript(*selection)
    identity[selection] = 1
    return identity


def get_subarray(data: Tensor, axes: Iterable[int], indices: Iterable[Iterable[int]]):
    """Extract a subarray from the input data based on the provided axes and indices.

    Args:
        data (Tensor): Input data array.
        axes (Iterable[int]): Axes along which the subarray is to be extracted.
        indices (Iterable[Iterable[int]]): Indices for the subarray extraction.

    Returns:
        Tensor: Subarray extracted from the input data based on the provided axes and indices.
    """
    # test every element of indices is list[int]
    for ele in indices:
        assert isinstance(ele, Iterable)

    num_data_axes = len(data.shape)
    for axis, index in zip(axes, indices):
        selection = [None] * num_data_axes
        selection[axis] = index
        selection = subscript(*selection)
        data = data[selection]
    return data
