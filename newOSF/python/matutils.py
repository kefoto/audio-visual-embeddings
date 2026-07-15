"""Loaders that unwrap scipy.io.loadmat's MATLAB cell-array encoding into plain Python/numpy."""
import numpy as np
import scipy.io as sio


def load_mat(path):
    return sio.loadmat(path, squeeze_me=False)


def to_1d(arr):
    """Flatten a MATLAB row/column vector to a 1D numpy array."""
    return np.asarray(arr).reshape(-1)


def cellstr(arr):
    """MATLAB cell array of strings -> list[str], preserving element order."""
    return [str(x[0]) for x in to_1d(arr)]


def cellnum(arr):
    """MATLAB cell array of numeric vectors -> list[np.ndarray], each flattened to 1D."""
    return [to_1d(x).astype(float) for x in to_1d(arr)]
