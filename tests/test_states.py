import pytest
from unittest.mock import Mock
from itertools import combinations
import pickle
import numpy as np
import mici


def _bind_with_decorator(instance, name, func, decorator):
    """Bind func decorated with decorator as a method of instance and return.

    Based on https://stackoverflow.com/a/1015405
    """
    func.__name__ = name
    decorated_func = decorator(func)
    bound_method = decorated_func.__get__(instance, instance.__class__)
    setattr(instance, name, bound_method)
    return bound_method


@pytest.fixture
def state_vars():
    return {'spam': np.array([0.5, -1.]), 'ham': np.array(1), 'eggs': -2.}


def test_state_construction(state_vars):
    state = mici.states.ChainState(**state_vars)
    for key, val in state_vars.items():
        assert hasattr(state, key)
        assert getattr(state, key) is val


def test_state_copy(state_vars):
    state = mici.states.ChainState(**state_vars)
    state_copy = state.copy()
    for key, val in state_vars.items():
        assert hasattr(state_copy, key)
        assert np.all(getattr(state_copy, key) == val)


def test_state_copy_independent(state_vars):
    state = mici.states.ChainState(**state_vars)
    state_copy = state.copy()
    for key, val in state_vars.items():
        attr = getattr(state_copy, key)
        attr *= 2
        assert np.all(getattr(state, key) == val)


def test_state_contains(state_vars):
    state = mici.states.ChainState(**state_vars)
    for key in state_vars.keys():
        assert key in state


def test_state_read_only(state_vars):
    state = mici.states.ChainState(**state_vars, _read_only=True)
    for key in state_vars.keys():
        with pytest.raises(mici.errors.ReadOnlyStateError):
            setattr(state, key, None)


def test_state_copy_read_only(state_vars):
    state = mici.states.ChainState(**state_vars)
    state_copy = state.copy(read_only=True)
    for key in state_vars.keys():
        with pytest.raises(mici.errors.ReadOnlyStateError):
            setattr(state_copy, key, None)


def test_state_pickling(state_vars):
    state = mici.states.ChainState(**state_vars)
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    assert isinstance(unpickled_state, mici.states.ChainState)
    for key, val in state_vars.items():
        assert hasattr(unpickled_state, key)
        assert np.all(getattr(unpickled_state, key) == val)


def test_state_to_string(state_vars):
    state = mici.states.ChainState(**state_vars)
    assert isinstance(str(state), str)


def test_state_representation(state_vars):
    state = mici.states.ChainState(**state_vars)
    assert isinstance(repr(state), str)


def _mock_memoized_one_output_system_methods(system, state_vars):
    for var_name in state_vars.keys():
        name = f'{var_name}_method'
        func = lambda state: 2 * getattr(state, var_name)
        mocked_method = Mock(
            wraps=lambda self, state: func(state), name=name)
        bound_memoized_method = _bind_with_decorator(
            system, name, mocked_method,
            mici.states.cache_in_state(var_name))
        yield (var_name,), func, mocked_method, bound_memoized_method
    for var_names in combinations(state_vars.keys(), 2):
        name = f'{var_names[0]}_{var_names[1]}_method'
        func = lambda state: (
            getattr(state, var_names[0]) + getattr(state, var_names[1]))
        mocked_method = Mock(
            wraps=lambda self, state: func(state), name=name)
        bound_memoized_method = _bind_with_decorator(
            system, name, mocked_method,
            mici.states.cache_in_state(*var_names))
        yield var_names, func, mocked_method, bound_memoized_method


def test_cache_in_state(state_vars):
    system = Mock(name='MockSystem')
    state = mici.states.ChainState(**state_vars, _call_counts={})
    call_counts = state._call_counts
    for var_names, func, mocked_method, bound_memoized_method in (
            _mock_memoized_one_output_system_methods(system, state_vars)):
        assert mocked_method.call_count == 0, (
            'method to be memoized should not be called by decorator')
        ret_val_1 = bound_memoized_method(state)
        assert np.all(ret_val_1 == func(state)), (
            f'return value of memoized method should be same as func')
        assert mocked_method.call_count == 1, (
            'memoized method should be executed once on initial call')
        cache_key = mici.states._cache_key_func(system, mocked_method)
        assert call_counts[cache_key] == mocked_method.call_count, (
            f'state call counts value for {cache_key} incorrect')
        assert cache_key in state._cache, (
            f'state cache dict should contain {cache_key}')
        assert state._cache[cache_key] is ret_val_1, (
            f'cached value for key {cache_key} is incorrect')
        ret_val_2 = bound_memoized_method(state)
        assert mocked_method.call_count == 1, (
            'memoized method should not be executed again on second call')
        assert call_counts[cache_key] == mocked_method.call_count, (
            f'state call counts value for {cache_key} incorrect')
        assert ret_val_1 is ret_val_2, (
            'state cache should return same value when state unchanged')
        for var_name in var_names:
            setattr(state, var_name, state_vars[var_name])
            assert state._cache[cache_key] is None, (
                f'cached value for key {cache_key} should be None')
        ret_val_3 = bound_memoized_method(state)
        assert mocked_method.call_count == 2, (
            f'memoized method should be recalled after {var_names} update')
        assert call_counts[cache_key] == mocked_method.call_count, (
            f'state call counts value for {cache_key} incorrect')
        assert np.all(ret_val_3 == func(state)), (
            f'return value of memoized method should be same as unmemoized')


def _mock_memoized_aux_output_system_methods(system, state_vars):
    for var_name in state_vars.keys():
        name = f'{var_name}_method'
        aux_func = lambda state: -1 * getattr(state, var_name)
        prim_func = lambda state: 2 * getattr(state, var_name)
        mocked_method = Mock(
            wraps=lambda self, state: (prim_func(state), aux_func(state)),
            name=name)
        mocked_aux_method = Mock(
            wraps=lambda self, state: aux_func(state), name='aux_' + name)
        bound_memoized_method = _bind_with_decorator(
            system, name, mocked_method,
            mici.states.cache_in_state_with_aux(var_name, 'aux_' + name))
        bound_memoized_aux_method = _bind_with_decorator(
            system, 'aux_' + name, mocked_aux_method,
            mici.states.cache_in_state(var_name))
        yield ((var_name,), prim_func, aux_func, mocked_method,
               mocked_aux_method, bound_memoized_method,
               bound_memoized_aux_method)
    for var_names in combinations(state_vars.keys(), 2):
        name = f'{var_names[0]}_{var_names[1]}_method'
        prim_func = lambda state: (
            getattr(state, var_names[0]) + getattr(state, var_names[1]))
        aux_func = lambda state: -1 * getattr(state, var_names[0])
        mocked_method = Mock(
            wraps=lambda self, state: (prim_func(state), aux_func(state)),
            name=name)
        mocked_aux_method = Mock(
            wraps=lambda self, state: aux_func(state), name='aux_' + name)
        bound_memoized_method = _bind_with_decorator(
            system, name, mocked_method,
            mici.states.cache_in_state_with_aux(var_names, 'aux_' + name))
        bound_memoized_aux_method = _bind_with_decorator(
            system, 'aux_' + name, mocked_aux_method,
            mici.states.cache_in_state(*var_name))
        yield (var_names, prim_func, aux_func, mocked_method,
               mocked_aux_method, bound_memoized_method,
               bound_memoized_aux_method)


def test_cache_in_state_with_aux(state_vars):
    system = Mock(name='MockSystem')
    state = mici.states.ChainState(**state_vars, _call_counts={})
    call_counts = state._call_counts
    for (var_names, prim_func, aux_func, mocked_method, mocked_aux_method,
            bound_memoized_method, bound_memoized_aux_method) in (
            _mock_memoized_aux_output_system_methods(system, state_vars)):
        assert mocked_method.call_count == 0, (
            'method to be memoized should not be called by decorator')
        ret_val_1 = bound_memoized_method(state)
        assert np.all(ret_val_1 == prim_func(state)), (
            f'return value of memoized method should be same as prim_func')
        assert mocked_method.call_count == 1, (
            'memoized method should be executed once on first call')
        assert mocked_aux_method.call_count == 0, (
            'memoized aux method should not have been executed')
        prim_cache_key = mici.states._cache_key_func(system, mocked_method)
        aux_cache_key = mici.states._cache_key_func(
            system, mocked_aux_method)
        assert call_counts[prim_cache_key] == mocked_method.call_count, (
            f'state call counts value for {prim_cache_key} incorrect')
        assert call_counts[aux_cache_key] == mocked_aux_method.call_count, (
            f'state call counts value for {aux_cache_key} incorrect')
        assert prim_cache_key in state._cache, (
            f'state cache dict should contain {prim_cache_key}')
        assert aux_cache_key in state._cache, (
            f'state cache dict should contain {aux_cache_key}')
        assert state._cache[prim_cache_key] is ret_val_1, (
            f'cached value for key {prim_cache_key} is incorrect')
        assert np.all(state._cache[aux_cache_key] == aux_func(state)), (
            f'cached value for key {aux_cache_key} is incorrect')
        aux_ret_val = bound_memoized_aux_method(state)
        assert mocked_aux_method.call_count == 0, (
            'memoized aux method should not have been executed')
        assert call_counts[aux_cache_key] == mocked_aux_method.call_count, (
            f'state call counts value for {aux_cache_key} incorrect')
        assert np.all(aux_ret_val == aux_func(state)), (
            f'return value of aux memoized method is incorrect')
        ret_val_2 = bound_memoized_method(state)
        assert mocked_method.call_count == 1, (
            'memoized method should not be executed again on second call')
        assert call_counts[prim_cache_key] == mocked_method.call_count, (
            f'state call counts value for {prim_cache_key} incorrect')
        assert call_counts[aux_cache_key] == mocked_aux_method.call_count, (
            f'state call counts value for {aux_cache_key} incorrect')
        assert ret_val_1 is ret_val_2, (
            'state cache should return same value when state unchanged')
        for var_name in var_names:
            setattr(state, var_name, state_vars[var_name])
            assert state._cache[prim_cache_key] is None, (
                f'cached value for key {prim_cache_key} should be None')
            assert state._cache[aux_cache_key] is None, (
                f'cached value for key {aux_cache_key} should be None')
        ret_val_3 = bound_memoized_method(state)
        assert mocked_method.call_count == 2, (
            f'memoized method should be recalled after {var_names} update')
        assert call_counts[prim_cache_key] == mocked_method.call_count, (
            f'state call counts value for {prim_cache_key} incorrect')
        assert np.all(state._cache[aux_cache_key] == aux_func(state)), (
            f'cached value for key {aux_cache_key} is incorrect')
        assert np.all(ret_val_3 == prim_func(state)), (
            'return value of memoized method should be same as unmemoized')
