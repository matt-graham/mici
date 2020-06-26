import unittest
import copy
from collections.abc import Mapping
from types import MappingProxyType
from math import exp
import numpy as np
import mici


SEED = 3046987125
STATE_DIM = 10


class GenericAdapterTests:

    def test_initialize_generic(self):
        """Generic checks of initialize method.

        Checks method:
          1. Does not mutate chain_state argument.
          2. Returns a mapping-like object.
        """
        chain_state = self.chain_state.copy(read_only=True)
        adapter_state = self.adapter.initialize(chain_state, self.transition)
        self.assertIsInstance(adapter_state, Mapping)

    def test_update_generic(self):
        """Generic checks of update method.

        Checks method:
          1. Does not mutate chain_state argument.
          2. Does not mutate trans_stats argument.
        """
        chain_state = self.chain_state.copy()
        adapter_state = self.adapter.initialize(chain_state, self.transition)
        chain_state, trans_stats = self.transition.sample(chain_state, self.rng)
        chain_state_read_only = chain_state.copy(read_only=True)
        trans_stats_read_only = MappingProxyType(trans_stats)
        self.adapter.update(adapter_state, chain_state_read_only,
                            trans_stats_read_only, self.transition)

    def test_finalize_generic(self):
        """Generic checks of finalize method.

        Checks method:
          1. Accepts both a single and sequence of adapt_state arguments.
        """
        chain_state = self.chain_state.copy()
        adapter_state = self.adapter.initialize(chain_state, self.transition)
        for _ in range(5):
            chain_state, trans_stats = self.transition.sample(
                chain_state, self.rng)
            self.adapter.update(
                adapter_state, chain_state, trans_stats, self.transition)
        self.adapter.finalize(adapter_state.copy(), self.transition)
        self.adapter.finalize(
            [copy.deepcopy(adapter_state) for i in range(4)], self.transition)

    def _run_adaptive_chain(self, chain_state, n_transition):
        adapter_state = self.adapter.initialize(chain_state, self.transition)
        for _ in range(n_transition):
            chain_state.mom = self.transition.system.sample_momentum(
                chain_state, self.rng)
            chain_state, trans_stats = self.transition.sample(
                chain_state, self.rng)
            self.adapter.update(
                adapter_state, chain_state, trans_stats, self.transition)
        self.adapter.finalize(adapter_state, self.transition)
        return chain_state, adapter_state


class DualAveragingStepSizeAdapterTests(GenericAdapterTests):

    def test_initialize(self):
        adapter_state = self.adapter.initialize(
            self.chain_state, self.transition)
        self.assertIsNotNone(self.transition.integrator.step_size)
        self.assertGreater(self.transition.integrator.step_size, 0)
        self.assertEqual(adapter_state['iter'], 0)
        self.assertEqual(adapter_state['smoothed_log_step_size'], 0)
        self.assertEqual(adapter_state['adapt_stat_error'], 0)

    def test_finalize(self):
        adapter_state = self.adapter.initialize(
            self.chain_state, self.transition)
        self.adapter.finalize(adapter_state, self.transition)
        self.assertAlmostEqual(
            self.transition.integrator.step_size,
            exp(adapter_state['smoothed_log_step_size']))

    def test_successful_adaptation(self):
        chain_state = self.chain_state.copy()
        n_transition = 500
        chain_state, adapter_state = self._run_adaptive_chain(
            chain_state, n_transition)
        self.assertLess(abs(adapter_state['adapt_stat_error']), 0.02)
        sum_accept_stat = 0
        for _ in range(n_transition):
            chain_state.mom = self.transition.system.sample_momentum(
                chain_state, self.rng)
            chain_state, trans_stats = self.transition.sample(
                chain_state, self.rng)
            sum_accept_stat += trans_stats['accept_stat']
        av_accept_stat = sum_accept_stat / n_transition
        self.assertLess(
            abs(self.adapter.adapt_stat_target - av_accept_stat), 0.05)


class DualAveragingStepSizeAdapterWithEuclideanMetricSystem(
        DualAveragingStepSizeAdapterTests, unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        pos, mom = self.rng.standard_normal((2, STATE_DIM))
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2) / 2,
            grad_neg_log_dens=lambda pos: pos)
        integrator = mici.integrators.LeapfrogIntegrator(system, step_size=None)
        transition = mici.transitions.MultinomialDynamicIntegrationTransition(
            system, integrator)
        self.adapter = mici.adapters.DualAveragingStepSizeAdapter()
        self.chain_state = mici.states.ChainState(
            pos=pos, mom=mom, dir=1, _read_only=True)
        self.transition = transition


class DualAveragingStepSizeAdapterWithConstrainedEuclideanMetricSystem(
        DualAveragingStepSizeAdapterTests, unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        pos, mom = self.rng.standard_normal((2, STATE_DIM))
        pos /= np.sum(pos**2)**0.5
        mom -= (mom @ pos) * pos
        system = mici.systems.DenseConstrainedEuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2) / 2,
            grad_neg_log_dens=lambda pos: pos,
            constr=lambda pos: np.sum(pos**2)[None] - 1,
            jacob_constr=lambda pos: 2 * pos[None])
        integrator = mici.integrators.ConstrainedLeapfrogIntegrator(
            system, step_size=None)
        transition = mici.transitions.MultinomialDynamicIntegrationTransition(
            system, integrator)
        self.adapter = mici.adapters.DualAveragingStepSizeAdapter(
            log_step_size_reg_coefficient=0.1)
        self.chain_state = mici.states.ChainState(
            pos=pos, mom=mom, dir=1, _read_only=True)
        self.transition = transition


class TestOnlineVarianceMetricAdapter(GenericAdapterTests, unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        pos, mom = self.rng.standard_normal((2, STATE_DIM))
        self.var = np.exp(self.rng.standard_normal(STATE_DIM))
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2 / self.var) / 2,
            grad_neg_log_dens=lambda pos: pos / self.var)
        integrator = mici.integrators.LeapfrogIntegrator(system, 0.5)
        transition = mici.transitions.MultinomialDynamicIntegrationTransition(
            system, integrator)
        self.adapter = mici.adapters.OnlineVarianceMetricAdapter()
        self.chain_state = mici.states.ChainState(pos=pos, mom=mom, dir=1)
        self.transition = transition


class TestOnlineCovarianceMetricAdapter(GenericAdapterTests, unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        pos, mom = self.rng.standard_normal((2, STATE_DIM))
        system = mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2) / 2,
            grad_neg_log_dens=lambda pos: pos)
        integrator = mici.integrators.LeapfrogIntegrator(system, 0.5)
        transition = mici.transitions.MultinomialDynamicIntegrationTransition(
            system, integrator)
        self.adapter = mici.adapters.OnlineCovarianceMetricAdapter()
        self.chain_state = mici.states.ChainState(pos=pos, mom=mom, dir=1)
        self.transition = transition
