"""Utility functions and classes."""

from __future__ import annotations

from math import exp, expm1, inf, log, log1p, nan
from typing import TYPE_CHECKING

import numpy as np

try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

if TYPE_CHECKING:
    from typing_extensions import Self

    from mici.types import ScalarLike


def hash_array(array: np.ndarray) -> int:
    """Compute hash of a NumPy array by hashing data as a byte sequence.

    Args:
        array: NumPy array to compute hash of.

    Returns:
        Computed hash as an integer.
    """
    if XXHASH_AVAILABLE:
        # If fast Python wrapper of fast xxhash implementation is available use
        # in preference to built in hash function
        h = xxhash.xxh64()
        # Update hash by viewing array as byte sequence - no copy required
        h.update(array.view(np.byte).data)
        # Also update hash by array dtype, shape and strides to avoid clashes
        # between different views of same array
        h.update(bytes(f"{array.dtype}{array.shape}{array.strides}", "utf-8"))
        return h.intdigest()
    # Evaluate built-in hash function on *copy* of data as a byte sequence
    return hash(array.tobytes())


LOG_2: float = log(2.0)


def log1p_exp(val: float) -> float:
    """Numerically stable implementation of `log(1 + exp(val))`."""
    if val > 0.0:
        return val + log1p(exp(-val))
    return log1p(exp(val))


def log1m_exp(val: float) -> float:
    """Numerically stable implementation of `log(1 - exp(val))`."""
    if val >= 0.0:
        return nan
    if val > LOG_2:
        return log(-expm1(val))
    return log1p(-exp(val))


def log_sum_exp(val1: float, val2: float) -> float:
    """Numerically stable implementation of `log(exp(val1) + exp(val2))`."""
    if val1 == -inf and val2 == -inf:
        return -inf
    if val1 > val2:
        return val1 + log1p_exp(val2 - val1)
    return val2 + log1p_exp(val1 - val2)


def log_diff_exp(val1: float, val2: float) -> float:
    """Numerically stable implementation of `log(exp(val1) - exp(val2))`."""
    if val1 == -inf and val2 == -inf:
        return -inf
    if val1 < val2:
        return nan
    if val1 == val2:
        return -inf
    return val1 + log1m_exp(val2 - val1)


class LogRepFloat:
    """Numerically stable logarithmic representation of positive float values.

    Stores logarithm of value and overloads arithmetic operators to use more
    numerically stable implementations where possible.
    """

    def __init__(self, val: float | None = None, log_val: float | None = None) -> None:
        if log_val is None:
            if val is None:
                msg = "One of val or log_val must be specified."
                raise ValueError(msg)
            if val > 0:
                self.log_val = log(val)
            elif val == 0.0:
                self.log_val = -inf
            else:
                msg = "val must be non-negative."
                raise ValueError(msg)
        else:
            if val is not None:
                msg = "Specify only one of val and log_val."
                raise ValueError(msg)
            self.log_val = log_val

    @property
    def val(self) -> float:
        try:
            return exp(self.log_val)
        except OverflowError:
            return inf

    def __add__(self, other: ScalarLike) -> ScalarLike:
        if isinstance(other, LogRepFloat):
            return LogRepFloat(log_val=log_sum_exp(self.log_val, other.log_val))
        return self.val + other

    def __radd__(self, other: ScalarLike) -> ScalarLike:
        return self.__add__(other)

    def __iadd__(self, other: ScalarLike) -> Self:
        if other == 0:
            return self
        if isinstance(other, LogRepFloat):
            self.log_val = log_sum_exp(self.log_val, other.log_val)
        else:
            self.log_val = log_sum_exp(self.log_val, log(other))
        return self

    def __sub__(self, other: ScalarLike) -> ScalarLike:
        if isinstance(other, LogRepFloat):
            if self.log_val >= other.log_val:
                return LogRepFloat(log_val=log_diff_exp(self.log_val, other.log_val))
            return self.val - other.val
        return self.val - other

    def __rsub__(self, other: ScalarLike) -> ScalarLike:
        return (-self).__radd__(other)

    def __mul__(self, other: ScalarLike) -> ScalarLike:
        if isinstance(other, LogRepFloat):
            return LogRepFloat(log_val=self.log_val + other.log_val)
        return self.val * other

    def __rmul__(self, other: ScalarLike) -> ScalarLike:
        return self.__mul__(other)

    def __truediv__(self, other: ScalarLike) -> ScalarLike:
        if isinstance(other, LogRepFloat):
            return LogRepFloat(log_val=self.log_val - other.log_val)
        return self.val / other

    def __rtruediv__(self, other: ScalarLike) -> ScalarLike:
        return other / self.val

    def __neg__(self) -> float:
        return -self.val

    def __eq__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val == other.log_val
        return self.val == other

    def __ne__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val != other.log_val
        return self.val != other

    def __lt__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val < other.log_val
        return self.val < other

    def __gt__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val > other.log_val
        return self.val > other

    def __le__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val <= other.log_val
        return self.val <= other

    def __ge__(self, other: ScalarLike) -> bool:
        if isinstance(other, LogRepFloat):
            return self.log_val >= other.log_val
        return self.val >= other

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return f"LogRepFloat(val={self.val})"

    def __array__(self) -> np.ndarray:
        return np.array(self.val)
