import copy
from collections.abc import Mapping
from math import exp
from types import MappingProxyType

import numpy as np
import pytest

import mici

SEED = 3046987125
STATE_DIM = 10


@pytest.fixture()
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture()
def transition(system, integrator):
    return mici.transitions.MultinomialDynamicIntegrationTransition(system, integrator)


@pytest.fixture()
def momentum_transition(system):
    return mici.transitions.IndependentMomentumTransition(system)


class GenericAdapterTests:
    def test_initialize_generic(self, adapter, chain_state, transition):
        """Generic checks of initialize method.

        Checks method:
          1. Does not mutate chain_state argument.
          2. Returns a mapping-like object.
        """
        chain_state = chain_state.copy(read_only=True)
        adapter_state = adapter.initialize(chain_state, transition)
        assert isinstance(adapter_state, Mapping)

    def test_update_generic(self, adapter, chain_state, transition, rng):
        """Generic checks of update method.

        Checks method:
          1. Does not mutate chain_state argument.
          2. Does not mutate trans_stats argument.
        """
        chain_state = chain_state.copy()
        adapter_state = adapter.initialize(chain_state, transition)
        chain_state, trans_stats = transition.sample(chain_state, rng)
        chain_state_read_only = chain_state.copy(read_only=True)
        trans_stats_read_only = MappingProxyType(trans_stats)
        adapter.update(
            adapter_state,
            chain_state_read_only,
            trans_stats_read_only,
            transition,
        )

    def test_finalize_generic(self, adapter, chain_state, transition, rng):
        """Generic checks of finalize method.

        Checks method:
          1. Accepts both a single and sequence of adapt_state arguments.
        """
        adapter_state = adapter.initialize(chain_state, transition)
        for _ in range(5):
            chain_state, trans_stats = transition.sample(chain_state, rng)
            adapter.update(adapter_state, chain_state, trans_stats, transition)
        adapter.finalize(adapter_state.copy(), chain_state, transition, rng)
        adapter.finalize(
            [copy.deepcopy(adapter_state) for i in range(4)],
            [chain_state.copy() for i in range(4)],
            transition,
            [copy.deepcopy(rng) for i in range(4)],
        )


class DualAveragingStepSizeAdapterTests(GenericAdapterTests):
    def test_initialize(self, adapter, chain_state, transition):
        adapter_state = adapter.initialize(chain_state, transition)
        assert transition.integrator.step_size is not None
        assert transition.integrator.step_size > 0
        assert adapter_state["iter"] == 0
        assert adapter_state["smoothed_log_step_size"] == 0
        assert adapter_state["adapt_stat_error"] == 0

    def test_initialize_nan_raises_error(self, adapter, chain_state, transition):
        chain_state.pos += np.nan
        with pytest.raises(mici.errors.AdaptationError):
            adapter.initialize(chain_state, transition)

    def test_finalize(self, adapter, chain_state, transition, rng):
        adapter_state = adapter.initialize(chain_state, transition)
        adapter.finalize(adapter_state, chain_state, transition, rng)
        assert np.isclose(
            transition.integrator.step_size,
            exp(adapter_state["smoothed_log_step_size"]),
        )

    def test_adaptation(
        self,
        adapter,
        chain_state,
        transition,
        momentum_transition,
        rng,
    ):
        n_transition = 500
        adapter_state = adapter.initialize(chain_state, transition)
        for _ in range(n_transition):
            chain_state, _ = momentum_transition.sample(chain_state, rng)
            chain_state, trans_stats = transition.sample(chain_state, rng)
            adapter.update(adapter_state, chain_state, trans_stats, transition)
        adapter.finalize(adapter_state, chain_state, transition, rng)
        assert abs(adapter_state["adapt_stat_error"]) < 0.02
        sum_accept_stat = 0
        for _ in range(n_transition):
            chain_state, _ = momentum_transition.sample(chain_state, rng)
            chain_state, trans_stats = transition.sample(chain_state, rng)
            sum_accept_stat += trans_stats["accept_stat"]
        av_accept_stat = sum_accept_stat / n_transition
        assert abs(adapter.adapt_stat_target - av_accept_stat) < 0.05


class TestDualAveragingStepSizeAdapterWithEuclideanMetricSystem(
    DualAveragingStepSizeAdapterTests,
):
    @pytest.fixture()
    def chain_state(self, rng):
        pos, mom = rng.standard_normal((2, STATE_DIM))
        return mici.states.ChainState(pos=pos, mom=mom, dir=1)

    @pytest.fixture()
    def system(self):
        return mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2) / 2,
            grad_neg_log_dens=lambda pos: pos,
        )

    @pytest.fixture()
    def integrator(self, system):
        return mici.integrators.LeapfrogIntegrator(system, step_size=None)

    @pytest.fixture()
    def adapter(self):
        return mici.adapters.DualAveragingStepSizeAdapter()


class TestDualAveragingStepSizeAdapterWithConstrainedEuclideanMetricSystem(
    DualAveragingStepSizeAdapterTests,
):
    @pytest.fixture()
    def adapter(self):
        return mici.adapters.DualAveragingStepSizeAdapter(
            log_step_size_reg_coefficient=0.1,
        )

    @pytest.fixture()
    def chain_state(self, rng):
        pos, mom = rng.standard_normal((2, STATE_DIM))
        pos /= np.sum(pos**2) ** 0.5
        mom -= (mom @ pos) * pos
        return mici.states.ChainState(pos=pos, mom=mom, dir=1)

    @pytest.fixture()
    def system(self):
        return mici.systems.DenseConstrainedEuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2) / 2,
            grad_neg_log_dens=lambda pos: pos,
            constr=lambda pos: np.sum(pos**2)[None] - 1,
            jacob_constr=lambda pos: 2 * pos[None],
        )

    @pytest.fixture()
    def integrator(self, system):
        return mici.integrators.ConstrainedLeapfrogIntegrator(
            system,
            step_size=None,
            projection_solver=mici.solvers.solve_projection_onto_manifold_quasi_newton,
        )


class TestOnlineVarianceMetricAdapter(GenericAdapterTests):
    @pytest.fixture()
    def adapter(self):
        return mici.adapters.OnlineVarianceMetricAdapter()

    @pytest.fixture()
    def system(self, rng):
        var = np.exp(rng.standard_normal(STATE_DIM))
        return mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: np.sum(pos**2 / var) / 2,
            grad_neg_log_dens=lambda pos: pos / var,
        )

    @pytest.fixture()
    def integrator(self, system):
        return mici.integrators.LeapfrogIntegrator(system, 0.5)

    @pytest.fixture()
    def chain_state(self, rng):
        pos, mom = rng.standard_normal((2, STATE_DIM))
        return mici.states.ChainState(pos=pos, mom=mom, dir=1)

    def test_adaptation(
        self,
        adapter,
        chain_state,
        transition,
        momentum_transition,
        rng,
    ):
        adapter_state = adapter.initialize(chain_state, transition)
        n_transition = 10
        pos_samples = np.full((n_transition, STATE_DIM), np.nan)
        for i in range(n_transition):
            chain_state, _ = momentum_transition.sample(chain_state, rng)
            chain_state, trans_stats = transition.sample(chain_state, rng)
            pos_samples[i] = chain_state.pos
            adapter.update(adapter_state, chain_state, trans_stats, transition)
        assert adapter_state["iter"] == n_transition
        assert np.allclose(adapter_state["mean"], pos_samples.mean(0))
        assert np.allclose(
            adapter_state["sum_diff_sq"],
            np.sum((pos_samples - pos_samples.mean(0)) ** 2, 0),
        )
        adapter.finalize(adapter_state, chain_state, transition, rng)
        metric = transition.system.metric
        assert isinstance(metric, mici.matrices.PositiveDiagonalMatrix)
        var_est = pos_samples.var(axis=0, ddof=1)
        weight = n_transition / (adapter.reg_iter_offset + n_transition)
        regularized_var_est = weight * var_est + (1 - weight) * adapter.reg_scale
        assert np.allclose(metric.diagonal, 1 / regularized_var_est)


class TestOnlineCovarianceMetricAdapter(GenericAdapterTests):
    @pytest.fixture()
    def adapter(self):
        return mici.adapters.OnlineCovarianceMetricAdapter()

    @pytest.fixture()
    def system(self, rng):
        prec = rng.standard_normal((STATE_DIM, STATE_DIM))
        prec = prec @ prec.T
        return mici.systems.EuclideanMetricSystem(
            neg_log_dens=lambda pos: (pos @ prec @ pos) / 2,
            grad_neg_log_dens=lambda pos: prec @ pos,
        )

    @pytest.fixture()
    def integrator(self, system):
        return mici.integrators.LeapfrogIntegrator(system, 0.5)

    @pytest.fixture()
    def chain_state(self, rng):
        pos, mom = rng.standard_normal((2, STATE_DIM))
        return mici.states.ChainState(pos=pos, mom=mom, dir=1)

    def test_adaptation(
        self,
        adapter,
        chain_state,
        transition,
        momentum_transition,
        rng,
    ):
        adapter_state = adapter.initialize(chain_state, transition)
        n_transition = 10
        pos_samples = np.full((n_transition, STATE_DIM), np.nan)
        for i in range(n_transition):
            chain_state, _ = momentum_transition.sample(chain_state, rng)
            chain_state, trans_stats = transition.sample(chain_state, rng)
            pos_samples[i] = chain_state.pos
            adapter.update(adapter_state, chain_state, trans_stats, transition)
        assert adapter_state["iter"] == n_transition
        assert np.allclose(adapter_state["mean"], pos_samples.mean(0))
        pos_minus_mean_samples = pos_samples - pos_samples.mean(0)
        assert np.allclose(
            adapter_state["sum_diff_outer"],
            np.einsum("ij,ik->jk", pos_minus_mean_samples, pos_minus_mean_samples),
        )
        adapter.finalize(adapter_state, chain_state, transition, rng)
        metric = transition.system.metric
        assert isinstance(metric, mici.matrices.PositiveDefiniteMatrix)
        covar_est = np.cov(pos_samples, rowvar=False, ddof=1)
        weight = n_transition / (adapter.reg_iter_offset + n_transition)
        regularized_covar_est = weight * covar_est + (
            1 - weight
        ) * adapter.reg_scale * np.identity(STATE_DIM)
        assert np.allclose(metric.inv.array, regularized_covar_est)
