"""Objects for recording state of a Markov chain and caching computations."""

import copy
from functools import wraps
from collections import Counter
from mici.errors import ReadOnlyStateError


def _cache_key_func(system, method):
    """Construct cache key for a given system and method pair."""
    if not isinstance(method, str):
        method = method.__name__
    return (f'{type(system).__name__}.{method}', id(system))


def cache_in_state(*depends_on):
    """Memoizing decorator for system methods.

    Used to decorate `mici.systems.System` methods which compute a function of
    one or more chain state variable(s), with the decorated method caching the
    value returned by the method being wrapped in the `ChainState` object to
    prevent the need for recomputation on future calls if the state variables
    the returned value depends on have not been changed in between the calls.

    Additionally for `ChainState` instances initialized with a `_call_counts`
    argument, the memoized method will update a counter for the method in the
    `_call_counts` attribute every time the method being decorated is called
    (i.e. when there isn't a valid cached value available).

    Args:
       *depends_on: One or more strings corresponding to the names of any state
           variables the value returned by the method depends on, e.g. 'pos' or
           'mom', such that the cache in the state object is correctly cleared
           when the value of any of these variables (attributes) of the state
           object changes.
    """
    def cache_in_state_decorator(method):
        @wraps(method)
        def wrapper(self, state):
            key = _cache_key_func(self, method)
            if key not in state._cache:
                for dep in depends_on:
                    state._dependencies[dep].add(key)
            if key not in state._cache or state._cache[key] is None:
                state._cache[key] = method(self, state)
                if state._call_counts is not None:
                    state._call_counts[key] += 1
            return state._cache[key]
        return wrapper
    return cache_in_state_decorator


def cache_in_state_with_aux(depends_on, auxiliary_outputs):
    """Memoizing decorator for system methods with possible auxiliary outputs.

    Used to decorate `mici.systems.System` methods which compute a function of
    one or more chain state variable(s), with the decorated method caching the
    value or values returned by the method being wrapped in the `ChainState`
    object to prevent the need for recomputation on future calls if the state
    variables the returned value(s) depends on have not been changed in between
    the calls.

    Compared to the `cache_in_state` decorator, this variant allows for methods
    which may optionally also return additional auxiliary outputs, such as
    intermediate result computed while computing the primary output, which
    correspond to the output of another system method decorated with the
    `cache_in_state` or `cache_in_state_with_aux` decorators. If such auxiliary
    outputs are returned they are also used to update cache entry for the
    corresponding decorated method, potentially saving recomputation in
    subsequent calls to that method. A common instance of this pattern is in
    derivative values computed using automatic differentiation (AD), with the
    primal value being differentiated usually either calculated alongside the
    derivative (in forward-mode AD) or calculated first in a forward-pass before
    the derivatives are calculated in a reverse-pass (in reverse-mode AD). By
    caching the value of the primal computed as part of the derivative
    calculation, a subsequent call to a method corresponding to calculation of
    the primal itself will retrieve the cached value and not recompute the
    primal, providing the relevant state variables the primal (and derivative)
    depend on have not been changed in between.

    Additionally for `ChainState` instances initialized with a `_call_counts`
    argument, the memoized method will update a counter for the method in the
    `_call_counts` attribute every time the method being decorated is called
    (i.e. when there isn't a valid cached value available).

    Args:
        depends_on (str or Tuple[str]): A string or tuple of strings, with each
            string corresponding to the name of a state variables the value(s)
            returned by the method depends on, e.g. 'pos' or 'mom', such that
            the cache in the state object is correctly cleared when the value of
            any of these variables (attributes) of the state object changes.
        auxiliary_outputs (str or Tuple[str]): A string or tuple of strings,
            with each string defining an auxiliary output the wrapped method may
            additionally return in addition to the primary output. If auxiliary
            outputs are returned, the returned value should be a tuple with
            first entry the 'primary' output corresponding to the value
            associated with the name of the method and the subsequent entries in
            the tuple corresponding to the auxiliary outputs in the order
            specified by the entries in the `auxiliary_outputs` argument. If the
            primary output is itself a tuple, it must be wrapped in another
            tuple even when no auxiliary outputs are being returned.
    """
    if isinstance(depends_on, str):
        depends_on = (depends_on,)
    if isinstance(auxiliary_outputs, str):
        auxiliary_outputs = (auxiliary_outputs,)

    def cache_in_state_with_aux_decorator(method):
        @wraps(method)
        def wrapper(self, state):
            prim_key = _cache_key_func(self, method)
            keys = [prim_key] + [
                _cache_key_func(self, a) for a in auxiliary_outputs]
            for i, key in enumerate(keys):
                if key not in state._cache:
                    for dep in depends_on:
                        state._dependencies[dep].add(key)
            if prim_key not in state._cache or state._cache[prim_key] is None:
                vals = method(self, state)
                if isinstance(vals, tuple):
                    for k, v in zip(keys, vals):
                        state._cache[k] = v
                else:
                    state._cache[prim_key] = vals
                if state._call_counts is not None:
                    state._call_counts[prim_key] += 1
            return state._cache[prim_key]
        return wrapper

    return cache_in_state_with_aux_decorator


class ChainState(object):
    """Markov chain state.

    As well as recording the chain state variable values, the state object is
    also used to cache derived quantities to avoid recalculation if these
    values are subsequently reused.

    Additionally for `ChainState` instances initialized with a `_call_counts`
    dictionary, any memoized system methods (i.e. those decorated with
    `cache_in_state` or `cache_in_state_with_aux`) will update a counter for the
    method in the state `_call_counts` dictionary attribute every time the
    decorated method is called (i.e. when there isn't a valid cached value
    available).
    """

    def __init__(self, *, _call_counts=None, _read_only=False,
                 _dependencies=None, _cache=None, **variables):
        """Create a new `ChainState` instance.

        Any keyword arguments passed to the constructor (with names not starting
        with an underscore) will be used to set state variable attributes of
        state object for example

            state = ChainState(pos=pos_val, mom=mom_val, dir=dir_val)

        will return a `ChainState` instance `state` with variable attributes
        `state.pos`, `state.mom` and `state.dir` with initial values set to
        `pos_val`, `mom_val` and `dir_val` respectively.

        Keyword arguments with a leading underscore in the name are reserved
        for additional arguments to the constructor not corresponding to
        state variables. Additionally the name `copy` should not be used as
        attribute access to this name will be blocked by the `copy` method.

        Kwargs:
            **variables: Keyword arguments corresponding to state variables. All
                names must not begin with an underscore and no name can be
                `copy`. See description above for details.
            _call_counts (None or Dict): If a dictionary is passed this will be
                used to store counts of the number of calls of system methods
                decorated with `cache_in_state` or `cache_in_state_with_aux`
                when called on this state object and when no cached value for
                the method is available so that the wrapped method is called.
                The `_call_counts` dictionary persists between all copies of a
                state so will count any decorated method calls on copies of the
                state as well - e.g. all copies of a state in a sampled Markov
                chain, allowing the `_call_counts` dictionary to be used to
                monitor the number of method call while sampling a chain.
            _read_only (bool): If `True` a `mici.errors.ReadOnlyStateError`
                exception will be raised when attempting to set any attributes
                of the state object after construction. Defaults to `False`.
            _dependencies (None or Dict): Intended for internal use only. If not
                `None` this should be a dictionary with string keys
                corresponding to the state variable names and values which are
                sets of strings indicating any dependencies of the relevant
                state variable in the cache.
            _cache (None or Dict): Intended for internal use only. If not `None`
                this should be a dictionary with keys corresponding to unique
                identifiers for methods decorated with the `cache_in_state` or
                `cache_in_state_with_aux` decorators and values corresponding
                to cached computed outputs of these methods or `None` for when
                a cached output is not available.
        """
        # Set attributes by directly writing to __dict__ to ensure set before
        # any call to __setattr__
        self.__dict__['_variables'] = variables
        if _dependencies is None:
            _dependencies = {name: set() for name in variables}
        self.__dict__['_dependencies'] = _dependencies
        if _cache is None:
            _cache = {}
        self.__dict__['_cache'] = _cache
        self.__dict__['_call_counts'] = (
            Counter(_call_counts) if not isinstance(_call_counts, Counter)
            else _call_counts)
        self.__dict__['_read_only'] = _read_only

    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if self._read_only:
            raise ReadOnlyStateError('ChainState instance is read-only.')
        if name in self._variables:
            self._variables[name] = value
            # clear any dependent cached values
            for dep in self._dependencies[name]:
                self._cache[dep] = None
        else:
            return super().__setattr__(name, value)

    def __contains__(self, name):
        return name in self._variables

    def copy(self, read_only=False):
        """Create a deep copy of the state object.

        Args:
            read_only (bool): Whether the state copy should be read-only.

        Returns:
            state_copy (ChainState): A copy of the state object with variable
                attributes that are independent copies of the original state
                object's variables.
        """
        return type(self)(
            _dependencies=self._dependencies, _cache=self._cache.copy(),
            _call_counts=self._call_counts, _read_only=read_only,
            **{name: copy.copy(val) for name, val in self._variables.items()})

    def __str__(self):
        return (
            '(\n ' +
            ',\n '.join([f'{k}={v}' for k, v in self._variables.items()]) +
            ')'
        )

    def __repr__(self):
        return type(self).__name__ + str(self)

    def __getstate__(self):
        return {
            'variables': self._variables,
            'dependencies': self._dependencies,
            # Don't pickle callable cached 'variables' such as derivative
            # functions
            'cache': {k: v for k, v in self._cache.items() if not callable(v)},
            'call_counts': self._call_counts,
            'read_only': self._read_only}

    def __setstate__(self, state):
        self.__dict__['_variables'] = state['variables']
        self.__dict__['_dependencies'] = state['dependencies']
        self.__dict__['_cache'] = state['cache']
        self.__dict__['_call_counts'] = state['call_counts']
        self.__dict__['_read_only'] = state['read_only']
