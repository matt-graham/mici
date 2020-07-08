"""Utility functions and classes."""

import numpy as np
from math import log, exp, log1p, expm1, inf, nan
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False


def hash_array(array):
    """Compute hash of a NumPy array by hashing data as a byte sequence.

    Args:
        array (array): NumPy array to compute hash of.

    Returns:
        hash (int): Computed hash as an integer.
    """
    if XXHASH_AVAILABLE:
        # If fast Python wrapper of fast xxhash implementation is available use
        # in preference to built in hash function
        h = xxhash.xxh64()
        # Update hash by viewing array as byte sequence - no copy required
        h.update(array.view(np.byte).data)
        # Also update hash by array dtype, shape and strides to avoid clashes
        # between different views of same array
        h.update(bytes(f'{array.dtype}{array.shape}{array.strides}', 'utf-8'))
        return h.intdigest()
    else:
        # Evaluate built-in hash function on *copy* of data as a byte sequence
        return hash(array.tobytes())


LOG_2 = log(2.)


def log1p_exp(val):
    """Numerically stable implementation of `log(1 + exp(val))`."""
    if val > 0.:
        return val + log1p(exp(-val))
    else:
        return log1p(exp(val))


def log1m_exp(val):
    """Numerically stable implementation of `log(1 - exp(val))`."""
    if val >= 0.:
        return nan
    elif val > LOG_2:
        return log(-expm1(val))
    else:
        return log1p(-exp(val))


def log_sum_exp(val1, val2):
    """Numerically stable implementation of `log(exp(val1) + exp(val2))`."""
    if val1 == -inf and val2 == -inf:
        return -inf
    elif val1 > val2:
        return val1 + log1p_exp(val2 - val1)
    else:
        return val2 + log1p_exp(val1 - val2)


def log_diff_exp(val1, val2):
    """Numerically stable implementation of `log(exp(val1) - exp(val2))`."""
    if val1 == -inf and val2 == -inf:
        return -inf
    elif val1 < val2:
        return nan
    elif val1 == val2:
        return -inf
    else:
        return val1 + log1m_exp(val2 - val1)


class LogRepFloat(object):
    """Numerically stable logarithmic representation of positive float values.

    Stores logarithm of value and overloads arithmetic operators to use more
    numerically stable implementations where possible.
    """

    def __init__(self, val=None, log_val=None):
        if log_val is None:
            if val is None:
                raise ValueError('One of val or log_val must be specified.')
            elif val > 0:
                self.log_val = log(val)
            elif val == 0.:
                self.log_val = -inf
            else:
                raise ValueError('val must be non-negative.')
        else:
            if val is not None:
                raise ValueError('Specify only one of val and log_val.')
            else:
                self.log_val = log_val

    @property
    def val(self):
        try:
            return exp(self.log_val)
        except OverflowError:
            return inf

    def __add__(self, other):
        if isinstance(other, LogRepFloat):
            return LogRepFloat(
                log_val=log_sum_exp(self.log_val, other.log_val))
        else:
            return self.val + other

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, LogRepFloat):
            self.log_val = log_sum_exp(self.log_val, other.log_val)
        else:
            self.log_val = log_sum_exp(self.log_val, log(other))
        return self

    def __sub__(self, other):
        if isinstance(other, LogRepFloat):
            if self.log_val >= other.log_val:
                return LogRepFloat(
                    log_val=log_diff_exp(self.log_val, other.log_val))
            else:
                return self.val - other.val
        else:
            return self.val - other

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def __mul__(self, other):
        if isinstance(other, LogRepFloat):
            return LogRepFloat(log_val=self.log_val + other.log_val)
        else:
            return self.val * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, LogRepFloat):
            return LogRepFloat(log_val=self.log_val - other.log_val)
        else:
            return self.val / other

    def __rtruediv__(self, other):
        return other / self.val

    def __neg__(self):
        return -self.val

    def __eq__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val == other.log_val
        else:
            return self.val == other

    def __ne__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val != other.log_val
        else:
            return self.val != other

    def __lt__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val < other.log_val
        else:
            return self.val < other

    def __gt__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val > other.log_val
        else:
            return self.val > other

    def __le__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val <= other.log_val
        else:
            return self.val <= other

    def __ge__(self, other):
        if isinstance(other, LogRepFloat):
            return self.log_val >= other.log_val
        else:
            return self.val >= other

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return 'LogRepFloat(val={0})'.format(self.val)
