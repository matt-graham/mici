"""Objects for recording state of a Markov chain."""

import copy
from functools import wraps


def cache_in_state(*depends_on):
    """Decorator to memoize / cache output of a function of state variable(s).

    Used to wrap functions of a chain state vaiable(s) to allow caching of
    the values computed to prevent recomputation when possible.

    Args:
       *depends_on: One or more strings defining which state variables the
           computed values depend on e.g. 'pos', 'mom', such that the cache is
           correctly cleared when one of these parent dependency's value
           changes.
    """
    def cache_in_state_decorator(method):
        @wraps(method)
        def wrapper(self, state):
            key = type(self).__name__ + '.' + method.__name__
            if key not in state._cache:
                for dep in depends_on:
                    state._dependencies[dep].add(key)
            if key not in state._cache or state._cache[key] is None:
                state._cache[key] = method(self, state)
                if state._call_counts is not None:
                    if key not in state._call_counts:
                        state._call_counts[key] = 1
                    else:
                        state._call_counts[key] += 1
            return state._cache[key]
        return wrapper
    return cache_in_state_decorator


def multi_cache_in_state(depends_on, variables, primary_index=0):
    """Decorator to cache multiple outputs of a function of state variable(s).

    Used to wrap functions of a chain state vaiable(s) to allow caching of
    the values computed to prevent recomputation when possible.

    This variant allows for functions which also cache intermediate computed
    results which may be used separately elsewhere for example the value of a
    function calculate in the forward pass of a reverse-mode automatic-
    differentation implementation of its gradient.

    Args:
        depends_on (List[str]): A list of strings defining which state
            variables the computed values depend on e.g. `['pos', 'mom']`, such
            that the cache is correctly cleared when one of these parent
            dependency's value changes.
        variables (List[str]): A list of strings defining the variables in the
            state cache dict corresponding to the outputs of the wrapped
            function (method) in the corresponding returned order.
        primary_index (int): Index of primary output of function (i.e. value to
            be returned) in `variables` list / position in output of function.
    """
    def multi_cache_in_state_decorator(method):
        @wraps(method)
        def wrapper(self, state):
            type_prefix = type(self).__name__ + '.'
            prim_key = type_prefix + variables[primary_index]
            keys = [type_prefix + v for v in variables]
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
                    if prim_key not in state._call_counts:
                        state._call_counts[prim_key] = 1
                    else:
                        state._call_counts[prim_key] += 1
            return state._cache[prim_key]
        return wrapper
    return multi_cache_in_state_decorator


class ChainState(object):
    """Markov chain state.

    As well as recording the chain state variable values, the state object is
    also used to cache derived quantities to avoid recalculation if these
    values are subsequently reused.
    """

    def __init__(self, _dependencies=None, _cache=None, _call_counts=None,
                 **variables):
        """Create a new `ChainState` instance.

        Any keyword arguments passed to the constructor will be used to set
        state variable attributes of state object for example

            state = ChainState(pos=pos_val, mom=mom_val, dir=dir_val)

        will return a `ChainState` instance `state` with variable attributes
        `state.pos`, `state.mom` and `state.dir` with initial values set to
        `pos_val`, `mom_val` and `dir_val` respectively. The keyword arguments
        `_dependencies`, `_cache` and `_call_counts` are reserved respectively
        for the dependency set, cache dictionary and call count dictionary, and
        cannot be used as state variable names.
        """
        # Set _variables attribute by directly writing to __dict__ to ensure
        # set before any cally to __setattr__
        self.__dict__['_variables'] = variables
        if _dependencies is None:
            _dependencies = {name: set() for name in variables}
        self._dependencies = _dependencies
        if _cache is None:
            _cache = {}
        self._cache = _cache
        self._call_counts = _call_counts

    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self._variables:
            self._variables[name] = value
            # clear any dependent cached values
            for dep in self._dependencies[name]:
                self._cache[dep] = None
        else:
            return super().__setattr__(name, value)

    def __contains__(self, name):
        return name in self._variables

    def copy(self):
        """Create a deep copy of the state object.

        Returns:
            state_copy (ChainState): A copy of the state object which can be
                updated without affecting the original object's variables.
        """
        return type(self)(
            _dependencies=self._dependencies, _cache=self._cache.copy(),
            _call_counts=self._call_counts,
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
            'call_counts': self._call_counts}

    def __setstate__(self, state):
        self.__dict__['_variables'] = state['variables']
        self._dependencies = state['dependencies']
        self._cache = state['cache']
        self._call_counts = state['call_counts']
