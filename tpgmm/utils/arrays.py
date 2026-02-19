import logging
from typing import Iterable, Tuple

import numpy as np
from numpy import ndarray

# Re-export from new location for backward compatibility
from tpgmm._core.arrays import subscript, identity_like, get_subarray
