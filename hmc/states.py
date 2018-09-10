"""Objects for recording state of a simulated Hamiltonian system."""


def cache_in_state(*depends_on):
    """Decorator to memoize / cache output of a function of state variable(s).

    Used to wrap methods of Hamiltonian system objects to allow caching of
    values computed from state position and momentum variables to prevent
    recomputation when possible. The decorator takes as arguments a set of
    strings defining which state variables the computed values depend on
    e.g. 'pos', 'mom', such that the cache is correctly cleared when one of
    these parent dependency's value changes.
    """
    def cache_in_state_decorator(method):
        key = method.__name__
        def wrapper(self, state):
            if key not in state.cache:
                for dep in depends_on:
                    state.dependencies[dep].add(key)
            if key not in state.cache or state.cache[key] is None:
                state.cache[key] = method(self, state)
            return state.cache[key]
        return wrapper
    return cache_in_state_decorator


def multi_cache_in_state(depends_on, keys, primary_index=0):
    """Decorator to cache multiple outputs of a function of state variable(s).

    Used to wrap methods of Hamiltonian system objects to allow caching of
    values computed from state position and momentum variables to prevent
    recomputation when possible. This variant allows for functions which also
    cache intermediate computed results which may be used separately elsewhere
    for example the value of a function calculate in the forward pass of a
    reverse-mode automatic differentation implementation of its gradient.

    Args:
        depends_on: a list of strings defining which state variables the
            computed values depend on e.g. ['pos', 'mom'], such that the cache
            is correctly cleared when one of these parent dependency's value
            changes.
        keys: a list of strings defining the keys in the state cache dictionary
            corresponding to the outputs of the wrapped function (method) in
            the corresponding returned order.
        primary_index: index of primary output of function (i.e. value to be
            returned) in keys list / position in output of function.
    """
    prim_key = keys[primary_index]
    def multi_cache_in_state_decorator(method):
        def wrapper(self, state):
            for key in keys:
                if key not in state.cache:
                    for dep in depends_on:
                        state.dependencies[dep].add(key)
            if prim_key not in state.cache or state.cache[prim_key] is None:
                vals = method(self, state)
                for k, v in zip(keys, vals):
                    state.cache[k] = v
            return state.cache[prim_key]
        return wrapper
    return multi_cache_in_state_decorator


class HamiltonianState(object):
    """State of a Hamiltonian system.

    As well as recording the position and momentum state values and integration
    direction binary indicator, the state object is also used to cache derived
    quantities such as (components of) the Hamiltonian function and gradients
    for the current position and momentum values to avoid recalculation when
    these values are reused.
    """

    def __init__(self, pos, mom, dir=1, dependencies=None, cache=None):
        self._pos = pos
        self._mom = mom
        self.dir = dir
        if dependencies is None:
            dependencies = {'pos': set(), 'mom': set()}
        self.dependencies = dependencies
        if cache is None:
            cache = {}
        self.cache = cache

    @property
    def n_dim(self):
        return self.pos.shape[0]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        # clear any dependent cached values
        for dep in self.dependencies['pos']:
            self.cache[dep] = None

    @property
    def mom(self):
        return self._mom

    @mom.setter
    def mom(self, value):
        self._mom = value
        # clear any dependent cached values
        for dep in self.dependencies['mom']:
            self.cache[dep] = None

    def copy(self):
        return HamiltonianState(
            pos=self.pos.copy(), mom=self.mom.copy(), dir=self.dir,
            cache=self.cache.copy(), dependencies=self.dependencies)

    def __str__(self):
        return f'(\n  pos={self.pos},\n  mom={self.mom},\n  dir={self.dir})'

    def __repr__(self):
        return type(self).__name__ + str(self)
