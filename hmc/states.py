"""Objects for recording state of a Markov chain."""

import copy


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
        def wrapper(self, state):
            key = (type(self).__name__ + '.' + method.__name__, id(self))
            if key not in state.cache:
                for dep in depends_on:
                    state.dependencies[dep].add(key)
            if key not in state.cache or state.cache[key] is None:
                state.cache[key] = method(self, state)
            return state.cache[key]
        return wrapper
    return cache_in_state_decorator


def multi_cache_in_state(depends_on, vars, primary_index=0):
    """Decorator to cache multiple outputs of a function of state variable(s).

    Used to wrap functions of a chain state vaiable(s) to allow caching of
    the values computed to prevent recomputation when possible.

    This variant allows for functions which also cache intermediate computed
    results which may be used separately elsewhere for example the value of a
    function calculate in the forward pass of a reverse-mode automatic-
    differentation implementation of its gradient.

    Args:
        depends_on: a list of strings defining which state variables the
            computed values depend on e.g. ['pos', 'mom'], such that the cache
            is correctly cleared when one of these parent dependency's value
            changes.
        vars: a list of strings defining the variables in the state cache dict
            corresponding to the outputs of the wrapped function (method) in
            the corresponding returned order.
        primary_index: index of primary output of function (i.e. value to be
            returned) in vars list / position in output of function.
    """
    def multi_cache_in_state_decorator(method):
        def wrapper(self, state):
            id_ = id(self)
            type_prefix = type(self).__name__ + '.'
            prim_key = (type_prefix + vars[primary_index], id_)
            keys = [(type_prefix + v, id_) for v in vars]
            for i, key in enumerate(keys):
                if key not in state.cache:
                    for dep in depends_on:
                        state.dependencies[dep].add(key)
            if prim_key not in state.cache or state.cache[prim_key] is None:
                vals = method(self, state)
                if isinstance(vals, tuple):
                    for k, v in zip(keys, vals):
                        state.cache[k] = v
                else:
                    state.cache[prim_key] = vals
            return state.cache[prim_key]
        return wrapper
    return multi_cache_in_state_decorator


class ChainState(object):
    """Markov chain state.

    As well as recording the chain state variable values, the state object is
    also used to cache derived quantities to avoid recalculation if these
    values are subsequently reused.
    """

    def __init__(self, dependencies=None, cache=None, **vars):
        # set vars attribute by directly writing to __dict__ to ensure set
        # before any cally to __setattr__
        self.__dict__['vars'] = vars
        if dependencies is None:
            dependencies = {name: set() for name in vars}
        self.dependencies = dependencies
        if cache is None:
            cache = {}
        self.cache = cache

    def __getattr__(self, name):
        if name in self.vars:
            return self.vars[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self.vars:
            self.vars[name] = value
            # clear any dependent cached values
            for dep in self.dependencies[name]:
                self.cache[dep] = None
        else:
            return super().__setattr__(name, value)

    def copy(self):
        """Create a deep copy of the state object.

        Returns:
            A copy of the state object which can be updated without
            affecting the original object's attributes.
        """
        return type(self)(
            cache=self.cache.copy(), dependencies=self.dependencies,
            **{name: copy.copy(val) for name, val in self.vars.items()})

    def __str__(self):
        return (
            '(\n ' +
            ',\n '.join([f'{name}={val}' for name, val in self.vars.items()]) +
            ')'
        )

    def __repr__(self):
        return type(self).__name__ + str(self)

    def __getstate__(self):
        return self.vars

    def __setstate__(self, state):
        self.__dict__['vars'] = state
        self.dependencies = {name: set() for name in self.vars}
        self.cache = {}


class HamiltonianState(ChainState):
    """Chain state with position, momentum and integration direction variables.

    As well as recording the position and momentum state values and integration
    direction binary indicator, the state object is also used to cache derived
    quantities such as (components of) the Hamiltonian function and gradients
    for the current position and momentum values to avoid recalculation when
    these values are reused.
    """

    def __init__(self, pos, mom=None, dir=1, dependencies=None, cache=None):
        super().__init__(
            pos=pos, mom=mom, dir=dir, dependencies=dependencies, cache=cache)
