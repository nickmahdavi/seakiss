import numpy as np

def pad(arr: np.ndarray, length: int) -> np.ndarray:
    if arr.size < length:
        return np.pad(arr, (0, length - arr.size))
    return arr[:length]

def sample_hamming_ternary(dim: int, hweight: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=int)
    positions = np.random.choice(dim, size=hweight, replace=False)
    values = np.random.choice([-1, 1], size=hweight)
    vector[positions] = values
    return vector

def sample_discrete_gaussian(dim: int, sigma: float) -> np.ndarray:
    assert sigma > 0
    return np.random.normal(0, sigma, size=dim).round().astype(int)

def sample_zero_one(dim: int, p_nonzero: float) -> np.ndarray:
    assert 0 <= p_nonzero <= 1
    return np.random.choice([-1, 0, 1], size=dim, p=[p_nonzero/2, 1-p_nonzero, p_nonzero/2])