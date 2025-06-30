from ring import CR, CRE

import numpy as np

def encode(input: np.ndarray, D: int, k: int, q: np.ndarray):
    N = 1 << k
    L = N >> 1
    Q = np.prod(q)

    input = np.pad(input, (0, max(0, L - input.size)))[:L]
    z = np.concatenate((input, np.conjugate(input[::-1])))

    z_hat = np.fft.ifft(z).real * 2 ** D
    coeffs = np.rint(z_hat).astype(object)
    coeffs_pos = (coeffs + Q) % Q
    return coeffs_pos % q[:, None]