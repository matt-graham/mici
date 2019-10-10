"""Utility functions and classes."""

import numpy as np
from math import log, exp, log1p, expm1, inf, nan
import types
import sys
import inspect
import hmc


try:

    import tqdm
    import logging

    class TqdmHandler(logging.StreamHandler):
        """Simple log handler which uses tqdm write method."""

        def __init__(self):
            super().__init__()

        def emit(self, record):
            msg = self.format(record)
            tqdm.tqdm.write(msg)

    def setup_tqdm_logger():
        """Returns a logger which redirects log output to tqdm.write."""
        logger = logging.getLogger()
        handler = TqdmHandler()
        logger.addHandler(handler)
        return logger

except ImportError:
    pass


try:

    import arviz

    def convert_to_arviz_inference_data(
            traces, chain_stats, sample_stats_key=None):
        """Wrap chain outputs in an `arviz.InferenceData` container object.

        The `traces` and `chain_stats` arguments should correspond to a
        multiple-chain sampler output i.e. the returned values from a
        `sample_chains` call.

        Args:
            traces (dict[str, list[array]]): Trace arrays, with one entry per
                function in `trace_funcs` passed to sampler method. Each entry
                consists of a list of arrays, one per chain, with the first
                axes of the arrays corresponding to the sampling (draw) index.
            chain_stats (dict[str, list[array]]): Chain integration transition
                statistics as a dictionary with string keys describing the
                statistics recorded and values corresponding to a list of
                arrays with one array per chain and the first axis of the
                arrays corresponding to the sampling index.
            sample_stats_key (str): Optional. Key of transition in
                `chain_stats` to use the recorded statistics of to populate the
                `sampling_stats` group in the returned `InferenceData` object.

        Returns:
            arvix.InferenceData:
                An arviz data container with groups `posterior` and
                'sample_stats', both of instances of `xarray.Dataset`. The
                `posterior` group corresponds to the chain variable traces
                provides in the `traces` argument and the `sample_stats`
                group corresponds to the chain transitions statistics passed
                in the `chain_stats` argument (if multiple transition
                statistics dictionaries are present the `sample_stats_key`
                argument should be specified to indicate which to use).
        """
        if (sample_stats_key is not None and
                sample_stats_key not in chain_stats):
            raise ValueError(
                f'Specified `sample_stats_key` ({sample_stats_key}) does '
                f'not match any transition in `chain_stats`.')
        if sample_stats_key is not None:
            return arviz.InferenceData(
                posterior=arviz.dict_to_dataset(traces, library=hmc),
                sample_stats=arviz.dict_to_dataset(
                    chain_stats[sample_stats_key], library=hmc))
        elif not isinstance(next(iter(chain_stats.values())), dict):
            # chain_stats dictionary value not another dictionary therefore
            # assume corresponds to statistics for a single transition
            return arviz.InferenceData(
                posterior=arviz.dict_to_dataset(traces, library=hmc),
                sample_stats=arviz.dict_to_dataset(chain_stats, library=hmc))
        elif len(chain_stats) == 1:
            # single transtition statistics dictionary in chain_stats therefore
            # unambiguous to set sample_stats
            return arviz.InferenceData(
                posterior=arviz.dict_to_dataset(traces, library=hmc),
                sample_stats=arviz.dict_to_dataset(
                    chain_stats.popitem()[1], library=hmc))
        else:
            raise ValueError(
                '`sample_stats_key` must be specified as `chain_stats` '
                'contains multiple transtitiion statistics dictionaries.')

except ImportError:
    pass


def get_valid_filename(string):
    """Generate a valid filename from a string.

    Strips all characters which are not alphanumeric or a period (.), dash (-)
    or underscore (_).

    Based on https://stackoverflow.com/a/295146/4798943
    """
    return ''.join(c for c in string if (c.isalnum() or c in '._- '))


def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes.

    Original source: https://github.com/bosswissam/pysize

    MIT License

    Copyright (c) [2018] [Wissam Jarjoui]

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if (inspect.isgetsetdescriptor(d) or
                        inspect.ismemberdescriptor(d)):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif (hasattr(obj, '__iter__') and not
            isinstance(obj, (str, bytes, bytearray))):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen)
                    for s in obj.__slots__ if hasattr(obj, s))

    return size


def inherit_docstrings(cls):
    """Class decorator to inherit parent class docstrings if not specified.

    Taken from https://stackoverflow.com/a/38601305/4798943.
    """
    for name, func in vars(cls).items():
        if isinstance(func, types.FunctionType) and not func.__doc__:
            for parent in cls.__bases__:
                parfunc = getattr(parent, name, None)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
        elif isinstance(func, property) and not func.fget.__doc__:
            for parent in cls.__bases__:
                parprop = getattr(parent, name, None)
                if parprop and getattr(parprop.fget, '__doc__', None):
                    newprop = property(fget=func.fget,
                                       fset=func.fset,
                                       fdel=func.fdel,
                                       doc=parprop.fget.__doc__)
                    setattr(cls, name, newprop)
                    break

    return cls


LOG_2 = log(2.)


def log1p_exp(val):
    if val > 0.:
        return val + log1p(exp(-val))
    else:
        return log1p(exp(val))


def log1m_exp(val):
    if val >= 0.:
        return nan
    elif val > LOG_2:
        return log(-expm1(val))
    else:
        return log1p(-exp(a))


def log_sum_exp(val1, val2):
    if val1 > val2:
        return val1 + log1p_exp(val2 - val1)
    else:
        return val2 + log1p_exp(val1 - val2)


def log_diff_exp(val1, val2):
    if val1 < val2:
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
        if isinstance(other, LogRepFloat):
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
