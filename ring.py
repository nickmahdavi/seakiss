from __future__ import annotations
from functools import cached_property
from typing import Generator
import numpy as np
from util import pad

class CR:
    """
    Cyclotomic ring of degree 2^k, mod an integer.
    """
    def __init__(self, k: int, mod: int | None = None):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")

        if mod is not None and (not isinstance(mod, int) or mod < 1):
            raise ValueError("mod must be a positive integer")

        self.k = k
        self.degree = 2 ** k
        self.n = self.degree * 2 # Roots of unity
        self.mod = mod
        self.zero = CRE([0], self)
        self.one = CRE([1], self)

    def __repr__(self) -> str:
        mod_R = f"/{self.mod}ℤ" if self.mod else ""
        return f"ℤ[x]{mod_R}/(x^(2^{self.k})+1)"

    def __eq__(self, other: CR) -> bool:
        if not isinstance(other, CR):
            return NotImplemented
        return (self.k, self.degree, self.mod) == (other.k, other.degree, other.mod)

    def __hash__(self) -> int:
        return hash((self.k, self.degree, self.mod))

    @cached_property
    def proots(self) -> np.ndarray:
        # Exponents of primitive roots of unity
        return np.array([k for k in range(self.n) if np.gcd(k, self.n) == 1])

    def _reduce(self, coeffs: np.ndarray) -> np.ndarray:
        if coeffs.size <= self.degree:
            coeffs = pad(coeffs, self.degree)
            return coeffs

        # Split into initial polynomial of at most deg n-1, and higher terms
        coeffs, upper = np.split(coeffs, [self.degree])

        # Since x^n \equiv -1, the signs alternate for segments of length n
        # Track the number of such blocks and the tail
        blocks, tail = np.divmod(upper.size, self.degree)

        # Handle the tail first
        if tail:
            sign = (1 if blocks % 2 else -1) # Odds subtract, evens add
            coeffs[:tail] += sign * upper[-tail:]
            if blocks:
                # Off with his head
                upper = upper[:-tail]
            else:
                return coeffs
        
        # Arrange into rows
        fold = upper.reshape(-1, self.degree)

        # Alternating signs in rows
        signs = ((np.arange(fold.shape[0], dtype=int) & 1) * -2 + 1)[:, None]

        coeffs = coeffs + (signs * fold).sum(axis=0)
        return coeffs
    
    def reduce(self, coeffs: np.ndarray) -> np.ndarray:
        res = self._reduce(coeffs)
        return res % self.mod if self.mod is not None else res
    
    def sample_uniform(self) -> CRE:
        assert self.mod is not None
        coeffs = np.random.randint(0, self.mod, size=self.degree)
        return CRE(coeffs, self)

    def iter_roots(self) -> Generator[complex, None, None]:
        for j in range(1, self.n):
            if np.gcd(j, self.n) == 1:
                yield np.exp(2j * np.pi * j / self.n)   

    def embed(self, coeffs: np.typing.ArrayLike, inv=False) -> np.ndarray:
        """Canonical embedding R -> H."""
        if inv:
            # Expects z in half-complex frequency space (C^n with a symmetry constraint)
            z_in = np.zeros(self.n, dtype=complex)
            z_in[self.proots] = coeffs
            return np.fft.fft(z_in)[:self.degree] / self.degree
        else:
            # Canonical embedding
            coeffs = self._reduce(coeffs)
            coeffs = pad(coeffs, self.n)
            return np.conj(np.fft.fft(coeffs)[self.proots])

    def encode(self, z: np.ndarray, Delta_exp: int) -> np.ndarray:
        assert z.shape == (self.degree // 2,)
        z_proj = 10 ** Delta_exp * np.concatenate((z, np.conj(z[::-1])))
        coeffs = np.round(self.embed(z_proj, inv=True))
        return CRE(coeffs.real.astype(int), self)


class CRE:
    """
    Element of cyclotomic ring.
    """
    def __init__(self, coeffs: np.typing.ArrayLike, ring: CR, asc_deg: bool = True):
        # TODO enforce integer coeffs
        self.ring = ring
        self.coeffs = np.array(coeffs if asc_deg else coeffs[::-1])
        self.coeffs = self.ring.reduce(self.coeffs)
    
    def __hash__(self) -> int:
        return hash((tuple(self.coeffs), self.ring))

    def __add__(self, other: CRE) -> CRE:
        if self.ring != other.ring:
            raise TypeError("Cannot add elements from different rings")

        result = self.coeffs + other.coeffs
        return self.__class__(result, self.ring)
    
    def __sub__(self, other: CRE) -> CRE:
        if self.ring != other.ring:
            raise TypeError("Cannot subtract elements from different rings")

        return self + -1 * other
    
    def __neg__(self) -> CRE:
        return self.__class__(self.coeffs * -1, self.ring)

    def __mul__(self, other: CRE | int) -> CRE:
        if isinstance(other, int):
            return self.__class__(other * self.coeffs, self.ring)
        if self.ring != other.ring:
            raise TypeError("Cannot multiply elements from different rings")
        result = np.convolve(self.coeffs, other.coeffs)
        return self.__class__(result, self.ring)
    
    def __rmul__(self, other: int) -> CRE:
        return self.__mul__(other)

    def __pow__(self, exp: int) -> CRE:
        if exp < 0:
            raise ValueError("Negative exponents not supported")

        if exp == 0:
            return self.ring.one
        if exp == 1:
            return self

        res = self.ring.one
        base = self

        while exp > 0:
            if exp & 1:
                res = res * base
            base = base * base
            exp >>= 1

        return res

    def __call__(self, x):
        if isinstance(x, CRE):
            res = self.ring.zero
        else:
            res = 0
        for pow, coeff in enumerate(self.coeffs):
            res += coeff * x ** pow
        return res

    def __eq__(self, other: CRE) -> bool:
        if not isinstance(other, CRE):
            return NotImplemented
        return np.array_equal(self.coeffs, other.coeffs) and self.ring == other.ring

    def __repr__(self) -> str:
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff == 0:
                continue
            if i == 0:
                terms.append(str(coeff))
            else:
                prefix = ("- " if coeff < 0 else "+ ") if terms else ""
                coeff_s = prefix + (str(abs(coeff)) if abs(coeff) != 1 else "")
                pow_s = f"^{i}" if i > 1 else ""
                terms.append(f"{coeff_s}x{pow_s}")
        return f"{' '.join(terms)} ∈ {self.ring}"

    def decode(self, Delta_exp: int) -> np.ndarray:
        scaled_coeffs = self.coeffs / 10 ** Delta_exp
        left, right = np.split(self.ring.embed(scaled_coeffs), [self.ring.degree // 2])
        assert np.allclose(left, np.conjugate(right[::-1]))
        return left

    def degree(self) -> int:
        terms = self.coeffs.nonzero()[0]
        return terms[-1] if terms.size > 0 else 0