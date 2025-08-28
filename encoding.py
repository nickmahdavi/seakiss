import numpy as np
from ring import CR, CRE
from util import (
    sample_hamming_ternary,
    sample_discrete_gaussian,
    sample_uniform_cring,
    sample_zero_one,
    pad
)

class CKKS:
    """
    Cheon-Kim-Kim-Song encryption implementation

    Parameters
    ----------

    level: security parameter (attacks should require Ω(2^λ) ops)
    q: 
    L: ~= Multiplicative depth
    D: Sal
    """
    def __init__(self, level: int, q: int, p: int, L: int, Delta_exp: int):
        self.level = level
        self.Q = [q * p ** i for i in range(L+1)]
        self.qL = self.Q[-1]
        self.M, self.h, self.P, self.sig = self._params_from_security_level(level)
        self.ring = CR(self.M, self.P * self.qL)

        s = sample_hamming_ternary(self.M, self.h)
        a = sample_uniform_cring(self.ring)
        e = sample_discrete_gaussian(self.M, self.sig ** 2)
        self.sk = (CRE(1, self.ring), CRE(s, self.ring))
        self.pk = (CRE(a, self.ring), CRE(-a * s + e, self.ring))

        self.Delta = 1 << Delta_exp


    def _params_from_security_level(self, level: int) -> tuple[int, int, int, float]:
        pass

    def encode(self, plaintext: np.ndarray) -> CRE:
        plaintext = pad(plaintext, self.ring.degree)
        scaled_proj = self.Delta * np.concatenate((plaintext, np.conjugate(plaintext[::-1])))
        coords = np.round(scaled_proj).astype(int) % self.qL


        return CRE(coords, self.ring)
