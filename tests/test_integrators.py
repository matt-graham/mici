import numpy as np
import mici.integrators as integrators
import mici.systems as systems
from mici.states import ChainState
from mici.errors import IntegratorError

SEED = 3046987125
SIZES = {1, 2, 5, 10}
N_STEPS = {1, 5, 20, 100}
N_STATE = 10


class IntegratorTestCase(object):

    def __init__(self, integrator, states, h_diff_tol=5e-3, rng=None):
        self.integrator = integrator
        self.states = states
        self.h_diff_tol = h_diff_tol
        self.rng = np.random.RandomState(SEED) if rng is None else rng

    def integrate_with_reversal(self, init_state, n_step):
        state = init_state
        try:
            for s in range(n_step):
                state = self.integrator.step(state)
            state.dir *= -1
        except IntegratorError:
            state = init_state
        return state

    def check_reversibility(self, init_state, n_step):
        state = self.integrate_with_reversal(init_state, n_step)
        state = self.integrate_with_reversal(state, n_step)
        assert np.allclose(state.pos, init_state.pos), (
            f'integrator not returning on reversal to initial position.\n'
            f'initial position = {init_state.pos},\n'
            f'final position   = {state.pos}.')
        assert np.allclose(state.mom, init_state.mom), (
            f'integrator not returning on reversal to initial momentum.\n'
            f'initial momentum = {init_state.mom},\n'
            f'final momentum   = {state.mom}.')
        assert state.dir == init_state.dir, (
            'integrator not returning on reversal to initial direction.')

    def test_reversibility(self):
        for size in SIZES:
            for state in self.states[size]:
                for n_step in N_STEPS:
                    yield self.check_reversibility, state, n_step

    def check_hamiltonian_conservation(self, init_state, n_step):
        h_vals = [self.integrator.system.h(init_state)]
        state = init_state
        try:
            for s in range(n_step):
                state = self.integrator.step(state)
                h_vals.append(self.integrator.system.h(state))
        except IntegratorError:
            return
        diff_h = (
            np.mean(h_vals[:n_step//2]) - np.mean(h_vals[n_step//2:]))
        assert abs(diff_h) < self.h_diff_tol, (
            f'difference in average Hamiltonian over first and second halves '
            f'of trajectory ({diff_h}) is greater in magnitude than tolerance '
            f'({self.h_diff_tol}).')

    def test_hamiltonian_conservation(self):
        for size in SIZES:
            for state in self.states[size]:
                for n_step in [100, 200]:
                    yield self.check_hamiltonian_conservation, state, n_step

    def check_state_mutation(self, init_state):
        init_pos = init_state.pos.copy()
        init_mom = init_state.mom.copy()
        init_dir = init_state.dir
        state = self.integrator.step(init_state)
        assert init_state is not state, (
            'integrator not returning new state object')
        assert np.all(init_state.pos == init_pos), (
            'integrator modifiying passed state.pos attribute')
        assert np.all(init_state.mom == init_mom), (
            'integrator modifiying passed state.mom attribute')
        assert init_state.dir == init_dir, (
            'integrator modifiying passed state.dir attribute')

    def test_state_mutation(self):
        for size in SIZES:
            for state in self.states[size]:
                yield self.check_state_mutation, state


class LinearSystemIntegratorTestCase(IntegratorTestCase):

    def check_volume_preservation(self, init_states, n_step):
        init_zs, final_zs = [], []
        for state in init_states:
            init_zs.append(np.concatenate((state.pos, state.mom)))
            for s in range(n_step):
                state = self.integrator.step(state)
            final_zs.append(np.concatenate((state.pos, state.mom)))
        init_zs = np.column_stack(init_zs)
        final_zs = np.column_stack(final_zs)
        assert np.allclose(
            np.linalg.det(init_zs @ init_zs.T),
            np.linalg.det(final_zs @ final_zs.T)), (
            'state space volume spanned by initial and final state differs')

    def test_volume_conservation(self):
        for size in SIZES:
            for n_step in N_STEPS:
                yield self.check_volume_preservation, self.states[size], n_step


class TestLeapfrogIntegratorWithEuclideanMetricSystemLinear(
        LinearSystemIntegratorTestCase):

    def __init__(self):
        system = systems.EuclideanMetricSystem(
            lambda q: 0.5 * np.sum(q**2), grad_neg_log_dens=lambda q: q)
        integrator = integrators.LeapfrogIntegrator(system, 0.5)
        rng = np.random.RandomState(SEED)
        states = {size: [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            for size in SIZES}
        h_diff_tol = 5e-3
        super().__init__(integrator, states, h_diff_tol, rng)


class TestLeapfrogIntegratorWithEuclideanMetricSystemNonLinear(
        IntegratorTestCase):

    def __init__(self):
        system = systems.EuclideanMetricSystem(
            lambda q: 0.25 * np.sum(q**4), grad_neg_log_dens=lambda q: q**3)
        integrator = integrators.LeapfrogIntegrator(system, 0.1)
        rng = np.random.RandomState(SEED)
        states = {size: [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            for size in SIZES}
        h_diff_tol = 2e-2
        super().__init__(integrator, states, h_diff_tol, rng)


class TestLeapfrogIntegratorWithGaussianEuclideanMetricLinearSystem(
        LinearSystemIntegratorTestCase):

    def __init__(self):
        system = systems.GaussianEuclideanMetricSystem(
            lambda q: 0, grad_neg_log_dens=lambda q: 0 * q)
        integrator = integrators.LeapfrogIntegrator(system, 0.5)
        rng = np.random.RandomState(SEED)
        states = {size: [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            for size in SIZES}
        h_diff_tol = 1e-10
        super().__init__(integrator, states, h_diff_tol, rng)


class TestLeapfrogIntegratorWithGaussianEuclideanMetricLinearSystem(
        IntegratorTestCase):

    def __init__(self):
        system = systems.GaussianEuclideanMetricSystem(
            lambda q: 0.125 * np.sum(q**4),
            grad_neg_log_dens=lambda q: 0.5 * q**3)
        integrator = integrators.LeapfrogIntegrator(system, 0.1)
        rng = np.random.RandomState(SEED)
        states = {size: [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            for size in SIZES}
        h_diff_tol = 1e-2
        super().__init__(integrator, states, h_diff_tol, rng)


class TestImplicitLeapfrogIntegratorWithRiemannianMetricSystemLinear(
        LinearSystemIntegratorTestCase):

    def __init__(self):
        system = systems.DenseRiemannianMetricSystem(
            lambda q: 0.5 * np.sum(q**2), grad_neg_log_dens=lambda q: q,
            metric_func=lambda q: np.identity(q.shape[0]),
            vjp_metric_func=lambda q: lambda m: np.zeros_like(q))
        integrator = integrators.ImplicitLeapfrogIntegrator(system, 0.5)
        rng = np.random.RandomState(SEED)
        states = {size: [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            for size in SIZES}
        h_diff_tol = 5e-3
        super().__init__(integrator, states, h_diff_tol, rng)
