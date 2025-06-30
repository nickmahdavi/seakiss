from __future__ import annotations
import numpy as np

class CR:
    def __init__(self, k: int, mod: int | None = None):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")
        
        if mod is not None and (not isinstance(mod, int) or mod < 1):
            raise ValueError("mod must be a positive integer")

        self.k = k
        self.n = 2 ** k
        self.mod = mod
        self.one = CRE([1], self)
        self.zero = CRE([0], self)
    
    def __repr__(self) -> str:
        mod_R = f"/{self.mod}Z" if self.mod else ""
        return f"Z[x]{mod_R}/(x^(2^{self.k})+1)"
    
    def __eq__(self, other: CR) -> bool:
        return (self.k, self.n, self.mod) == (other.k, other.n, other.mod)

    def _reduce(self, coeffs: np.ndarray) -> np.ndarray:
        # Split into initial polynomial of at most deg n-1, and higher terms
        coeffs, upper = np.split(coeffs, [self.n])

        if upper.size == 0:
            # Skip reduction and pad lower half if necessary
            if (size := coeffs.size) < self.n:
                coeffs = np.pad(coeffs, (0, self.n - size))
            return coeffs

        # Since x^n \equiv -1, the signs alternate for segments of length n
        # Track the number of such blocks and the tail
        blocks, tail = np.divmod(upper.size, self.n)

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
        fold = upper.reshape(-1, self.n)

        # Alternating signs in rows
        signs = ((np.arange(fold.shape[0], int) & 1) * -2 + 1)[:, None]

        coeffs = coeffs + (signs * fold).sum(axis=0)
        return coeffs
    
    def reduce(self, coeffs: np.ndarray) -> np.ndarray:
        res = self._reduce(coeffs)
        return res % self.mod if self.mod is not None else res

class CRE:
    def __init__(self, coeffs: np.typing.ArrayLike, ring: CR, asc_deg: bool = True):
        # TODO enforce integer coeffs

        self.ring = ring
        self.coeffs = np.array(coeffs if asc_deg else reversed(coeffs))
        self.coeffs = self.ring.reduce(self.coeffs)

    @classmethod
    def from_roots(cls, roots: np.ndarray, ring: CR) -> CRE:
        res = ring.one
        for root in roots:
            res *= CRE([-root, 1], ring)
        return res

    def __add__(self, other: CRE) -> CRE:
        assert self.ring == other.ring
        result = self.coeffs + other.coeffs
        return self.__class__(result, self.ring)
    
    def __sub__(self, other: CRE) -> CRE:
        return self + -1 * other
    
    def __neg__(self) -> CRE:
        return self.__class__(self.coeffs * -1, self.ring)

    def __mul__(self, other: CRE | int) -> CRE:
        if isinstance(other, int):
            return self.__class__(other * self.coeffs, self.ring)
        assert self.ring == other.ring
        result = np.convolve(self.coeffs, other.coeffs)
        return self.__class__(result, self.ring)
    
    def __rmul__(self, other: CRE | int) -> CRE:
        return self.__mul__(other)
    
    def __pow__(self, pow: int) -> CRE:
        res = self.ring.one
        base = self
        if pow == 0:
            return res
        while pow > 0:
            if pow & 1:
                res = res * base
            base = base * base
            pow >>= 1
        return res

    def __call__(self):
        raise NotImplementedError("Use as ring element, not functions Z -> Z.")

    def __eq__(self, other: CRE) -> bool:
        return np.array_equal(self.coeffs, other.coeffs) and self.ring == other.ring

    def __repr__(self) -> str:
        terms = []
        for i, coeff in enumerate(self.coeffs):
            if coeff == 0:
                continue
            if i == 0:
                terms.append(str(coeff))
            else:
                coeff_s = ("- " if coeff < 0 else "+ ") + (str(abs(coeff)) if abs(coeff) != 1 else "")
                pow_s = f"^{i}" if i > 1 else ""
                terms.append(f"{coeff_s}x{pow_s}")
        return f"{' '.join(terms)} \\in {self.ring}"