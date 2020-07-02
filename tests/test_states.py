import unittest
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


class ChainStateTestCase(unittest.TestCase):

    def setUp(self):
        self.vars = {'spam': np.array([0.5, -1.]),
                     'ham': np.array(1),
                     'eggs': -2.}

    def test_state_construction(self):
        state = mici.states.ChainState(**self.vars)
        for key, val in self.vars.items():
            self.assertTrue(hasattr(state, key))
            self.assertTrue(getattr(state, key) is val)

    def test_state_copy(self):
        state = mici.states.ChainState(**self.vars)
        state_copy = state.copy()
        for key, val in self.vars.items():
            self.assertTrue(hasattr(state_copy, key))
            self.assertTrue(np.all(getattr(state_copy, key) == val))

    def test_state_copy_independent(self):
        state = mici.states.ChainState(**self.vars)
        state_copy = state.copy()
        for key, val in self.vars.items():
            attr = getattr(state_copy, key)
            attr *= 2
            self.assertTrue(np.all(getattr(state, key) == val))

    def test_state_contains(self):
        state = mici.states.ChainState(**self.vars)
        for key in self.vars.keys():
            self.assertTrue(key in state)

    def test_state_read_only(self):
        state = mici.states.ChainState(**self.vars, _read_only=True)
        for key in self.vars.keys():
            with self.assertRaises(mici.errors.ReadOnlyStateError):
                setattr(state, key, None)

    def test_state_copy_read_only(self):
        state = mici.states.ChainState(**self.vars)
        state_copy = state.copy(read_only=True)
        for key in self.vars.keys():
            with self.assertRaises(mici.errors.ReadOnlyStateError):
                setattr(state_copy, key, None)

    def test_state_pickling(self):
        state = mici.states.ChainState(**self.vars)
        pickled_state = pickle.dumps(state)
        unpickled_state = pickle.loads(pickled_state)
        self.assertIsInstance(unpickled_state, mici.states.ChainState)
        for key, val in self.vars.items():
            self.assertTrue(hasattr(unpickled_state, key))
            self.assertTrue(np.all(getattr(unpickled_state, key) == val))

    def test_state_to_string(self):
        state = mici.states.ChainState(**self.vars)
        self.assertIsInstance(str(state), str)

    def test_state_representation(self):
        state = mici.states.ChainState(**self.vars)
        self.assertIsInstance(repr(state), str)

    def _mock_memoized_one_output_system_methods(self, system):
        for var_name in self.vars.keys():
            name = f'{var_name}_method'
            func = lambda state: 2 * getattr(state, var_name)
            mocked_method = Mock(
                wraps=lambda self, state: func(state), name=name)
            bound_memoized_method = _bind_with_decorator(
                system, name, mocked_method,
                mici.states.cache_in_state(var_name))
            yield (var_name,), func, mocked_method, bound_memoized_method
        for var_names in combinations(self.vars.keys(), 2):
            name = f'{var_names[0]}_{var_names[1]}_method'
            func = lambda state: (
                getattr(state, var_names[0]) + getattr(state, var_names[1]))
            mocked_method = Mock(
                wraps=lambda self, state: func(state), name=name)
            bound_memoized_method = _bind_with_decorator(
                system, name, mocked_method,
                mici.states.cache_in_state(*var_names))
            yield var_names, func, mocked_method, bound_memoized_method

    def test_cache_in_state(self):
        system = Mock(name='MockSystem')
        call_counts = {}
        state = mici.states.ChainState(**self.vars, _call_counts=call_counts)
        for var_names, func, mocked_method, bound_memoized_method in (
                self._mock_memoized_one_output_system_methods(system)):
            self.assertEqual(
                mocked_method.call_count, 0,
                'method to be memoized should not be called by decorator')
            ret_val_1 = bound_memoized_method(state)
            self.assertTrue(
                np.all(ret_val_1 == func(state)),
                f'return value of memoized method should be same as func')
            self.assertEqual(
                mocked_method.call_count, 1,
                'memoized method should be executed once on initial call')
            cache_key = mici.states._cache_key_func(system, mocked_method)
            self.assertEqual(
                call_counts[cache_key], mocked_method.call_count,
                f'state call counts value for {cache_key} incorrect')
            self.assertIn(
                cache_key, state._cache,
                f'state cache dict should contain {cache_key}')
            self.assertIs(
                state._cache[cache_key], ret_val_1,
                f'cached value for key {cache_key} is incorrect')
            ret_val_2 = bound_memoized_method(state)
            self.assertEqual(
                mocked_method.call_count, 1,
                'memoized method should not be executed again on second call')
            self.assertEqual(
                call_counts[cache_key], mocked_method.call_count,
                f'state call counts value for {cache_key} incorrect')
            self.assertIs(
                ret_val_1, ret_val_2,
                'state cache should return same value when state unchanged')
            for var_name in var_names:
                setattr(state, var_name, self.vars[var_name])
                self.assertIs(
                    state._cache[cache_key], None,
                    f'cached value for key {cache_key} should be None')
            ret_val_3 = bound_memoized_method(state)
            self.assertEqual(
                mocked_method.call_count, 2,
                f'memoized method should be recalled after {var_names} update')
            self.assertEqual(
                call_counts[cache_key], mocked_method.call_count,
                f'state call counts value for {cache_key} incorrect')
            self.assertTrue(
                np.all(ret_val_3 == func(state)),
                f'return value of memoized method should be same as unmemoized')

    def _mock_memoized_aux_output_system_methods(self, system):
        for var_name in self.vars.keys():
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
        for var_names in combinations(self.vars.keys(), 2):
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

    def test_cache_in_state_with_aux(self):
        system = Mock(name='MockSystem')
        call_counts = {}
        state = mici.states.ChainState(**self.vars, _call_counts=call_counts)
        for (var_names, prim_func, aux_func, mocked_method, mocked_aux_method,
                bound_memoized_method, bound_memoized_aux_method) in (
                self._mock_memoized_aux_output_system_methods(system)):
            self.assertEqual(
                mocked_method.call_count, 0,
                'method to be memoized should not be called by decorator')
            ret_val_1 = bound_memoized_method(state)
            self.assertTrue(
                np.all(ret_val_1 == prim_func(state)),
                f'return value of memoized method should be same as prim_func')
            self.assertEqual(
                mocked_method.call_count, 1,
                'memoized method should be executed once on first call')
            self.assertEqual(
                mocked_aux_method.call_count, 0,
                'memoized aux method should not have been executed')
            prim_cache_key = mici.states._cache_key_func(system, mocked_method)
            aux_cache_key = mici.states._cache_key_func(
                system, mocked_aux_method)
            self.assertEqual(
                call_counts[prim_cache_key], mocked_method.call_count,
                f'state call counts value for {prim_cache_key} incorrect')
            self.assertEqual(
                call_counts[aux_cache_key], mocked_aux_method.call_count,
                f'state call counts value for {aux_cache_key} incorrect')
            self.assertIn(
                prim_cache_key, state._cache,
                f'state cache dict should contain {prim_cache_key}')
            self.assertIn(
                aux_cache_key, state._cache,
                f'state cache dict should contain {aux_cache_key}')
            self.assertIs(
                state._cache[prim_cache_key], ret_val_1,
                f'cached value for key {prim_cache_key} is incorrect')
            self.assertTrue(
                np.all(state._cache[aux_cache_key] == aux_func(state)),
                f'cached value for key {aux_cache_key} is incorrect')
            aux_ret_val = bound_memoized_aux_method(state)
            self.assertEqual(
                mocked_aux_method.call_count, 0,
                'memoized aux method should not have been executed')
            self.assertEqual(
                call_counts[aux_cache_key], mocked_aux_method.call_count,
                f'state call counts value for {aux_cache_key} incorrect')
            self.assertTrue(
                np.all(aux_ret_val == aux_func(state)),
                f'return value of aux memoized method is incorrect')
            ret_val_2 = bound_memoized_method(state)
            self.assertEqual(
                mocked_method.call_count, 1,
                'memoized method should not be executed again on second call')
            self.assertEqual(
                call_counts[prim_cache_key], mocked_method.call_count,
                f'state call counts value for {prim_cache_key} incorrect')
            self.assertEqual(
                call_counts[aux_cache_key], mocked_aux_method.call_count,
                f'state call counts value for {aux_cache_key} incorrect')
            self.assertIs(
                ret_val_1, ret_val_2,
                'state cache should return same value when state unchanged')
            for var_name in var_names:
                setattr(state, var_name, self.vars[var_name])
                self.assertIs(
                    state._cache[prim_cache_key], None,
                    f'cached value for key {prim_cache_key} should be None')
                self.assertIs(
                    state._cache[aux_cache_key], None,
                    f'cached value for key {aux_cache_key} should be None')
            ret_val_3 = bound_memoized_method(state)
            self.assertEqual(
                mocked_method.call_count, 2,
                f'memoized method should be recalled after {var_names} update')
            self.assertEqual(
                call_counts[prim_cache_key], mocked_method.call_count,
                f'state call counts value for {prim_cache_key} incorrect')
            self.assertTrue(
                np.all(state._cache[aux_cache_key] == aux_func(state)),
                f'cached value for key {aux_cache_key} is incorrect')
            self.assertTrue(
                np.all(ret_val_3 == prim_func(state)),
                'return value of memoized method should be same as unmemoized')
