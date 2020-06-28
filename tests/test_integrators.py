import numpy as np
import mici.integrators as integrators
import mici.systems as systems
import mici.matrices as matrices
import mici.solvers as solvers
from mici.states import ChainState
from mici.errors import IntegratorError
from functools import wraps

SEED = 3046987125
SIZES = {1, 2, 5}
N_STEPS = {1, 5, 20}
N_STEPS_HAMILTONIAN = {200}
N_STATE = 5


def iterate_over_integrators_states(test):

    @wraps(test)
    def iterated_test(self):
        for (integrator, state_list) in self.integrators_and_state_lists:
            for state in state_list:
                yield (test, integrator, state)

    return iterated_test


def iterate_over_integrators_states_n_steps(test):

    @wraps(test)
    def iterated_test(self):
        for (integrator, state_list) in self.integrators_and_state_lists:
            for state in state_list:
                for n_step in N_STEPS:
                    yield (test, integrator, state, n_step)

    return iterated_test


def iterate_over_integrators_state_lists_n_steps(test):

    @wraps(test)
    def iterated_test(self):
        for (integrator, state_list) in self.integrators_and_state_lists:
            for n_step in N_STEPS:
                yield (test, integrator, state_list, n_step)

    return iterated_test


def _integrate_with_reversal(integrator, init_state, n_step):
    state = init_state
    try:
        for s in range(n_step):
            state = integrator.step(state)
        state.dir *= -1
    except IntegratorError:
        state = init_state
    return state


def _generate_rand_eigval_eigvec(rng, size, eigval_scale=0.1):
    eigval = np.exp(eigval_scale * rng.standard_normal(size))
    eigvec = np.linalg.qr(rng.standard_normal((size, size)))[0]
    return eigval, eigvec


def _generate_metrics(rng, size, eigval_scale=0.1):
    eigval, eigvec = _generate_rand_eigval_eigvec(rng, size, eigval_scale)
    return [
        matrices.IdentityMatrix(),
        matrices.IdentityMatrix(size),
        matrices.PositiveDiagonalMatrix(eigval),
        matrices.DensePositiveDefiniteMatrix((eigvec * eigval) @ eigvec.T),
        matrices.EigendecomposedPositiveDefiniteMatrix(eigvec, eigval)
    ]


class IntegratorTestCase(object):

    def __init__(self, integrators_and_state_lists, h_diff_tol=5e-3):
        self.integrators_and_state_lists = integrators_and_state_lists
        self.h_diff_tol = h_diff_tol

    @iterate_over_integrators_states_n_steps
    def test_reversibility(integrator, init_state, n_step):
        state = _integrate_with_reversal(integrator, init_state, n_step)
        state = _integrate_with_reversal(integrator, state, n_step)
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

    @staticmethod
    def check_approx_hamiltonian_conservation(
            integrator, init_state, n_step, h_diff_tol):
        h_vals = [integrator.system.h(init_state)]
        state = init_state
        try:
            for s in range(n_step):
                state = integrator.step(state)
                h_vals.append(integrator.system.h(state))
        except IntegratorError:
            return
        diff_h = (
            np.mean(h_vals[:n_step//2]) - np.mean(h_vals[n_step//2:]))
        assert abs(diff_h) < h_diff_tol, (
            f'difference in average Hamiltonian over first and second halves '
            f'of trajectory ({diff_h}) is greater in magnitude than tolerance '
            f'({h_diff_tol}).')

    def test_hamiltonian_conservation(self):
        for (integrator, state_list) in self.integrators_and_state_lists:
            for state in state_list:
                for n_step in N_STEPS_HAMILTONIAN:
                    yield (self.check_approx_hamiltonian_conservation,
                           integrator, state, n_step, self.h_diff_tol)

    @iterate_over_integrators_states
    def test_state_mutation(integrator, init_state):
        init_pos = init_state.pos.copy()
        init_mom = init_state.mom.copy()
        init_dir = init_state.dir
        state = integrator.step(init_state)
        assert init_state is not state, (
            'integrator not returning new state object')
        assert np.all(init_state.pos == init_pos), (
            'integrator modifiying passed state.pos attribute')
        assert np.all(init_state.mom == init_mom), (
            'integrator modifiying passed state.mom attribute')
        assert init_state.dir == init_dir, (
            'integrator modifiying passed state.dir attribute')


class LinearSystemIntegratorTestCase(IntegratorTestCase):

    @iterate_over_integrators_state_lists_n_steps
    def test_volume_preservation(integrator, init_states, n_step):
        init_zs, final_zs = [], []
        for state in init_states:
            init_zs.append(np.concatenate((state.pos, state.mom)))
            for s in range(n_step):
                state = integrator.step(state)
            final_zs.append(np.concatenate((state.pos, state.mom)))
        init_zs = np.column_stack(init_zs)
        final_zs = np.column_stack(final_zs)
        assert np.allclose(
            np.linalg.det(init_zs @ init_zs.T),
            np.linalg.det(final_zs @ final_zs.T)), (
            'state space volume spanned by initial and final state differs')


class ConstrainedSystemIntegratorTestCase(IntegratorTestCase):

    @iterate_over_integrators_states_n_steps
    def test_position_constraint(integrator, init_state, n_step):
        init_error = np.max(np.abs(integrator.system.constr(init_state)))
        assert init_error < 1e-8, (
            'Position constraint not satisfied at initial state '
            f'(|c| = {init_error:.1e})')
        final_state = _integrate_with_reversal(integrator, init_state, n_step)
        final_error = np.max(np.abs(integrator.system.constr(final_state)))
        assert final_error < 1e-8, (
            'Position constraint not satisfied at final state '
            f'(|c| = {final_error:.1e})')

    @iterate_over_integrators_states_n_steps
    def test_momentum_constraint(integrator, init_state, n_step):
        init_error = np.max(np.abs(
            integrator.system.jacob_constr(init_state) @
            integrator.system.dh_dmom(init_state)))
        assert init_error < 1e-8, (
            'Momentum constraint not satisfied at initial state '
            f'(|dc/dq @ dq/dt| = {init_error:.1e})')
        final_state = _integrate_with_reversal(integrator, init_state, n_step)
        final_error = np.max(np.abs(
            integrator.system.jacob_constr(final_state) @
            integrator.system.dh_dmom(final_state)))
        assert final_error < 1e-8, (
            'Momentum constraint not satisfied at final state '
            f'(|dc/dq @ dq/dt| = {final_error:.1e})')


class TestLeapfrogIntegratorLinearSystem(LinearSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in SIZES:
            for metric in _generate_metrics(rng, size):
                system = systems.EuclideanMetricSystem(
                    neg_log_dens=lambda q: 0.5 * np.sum(q**2),
                    metric=metric,
                    grad_neg_log_dens=lambda q: q)
                integrator = integrators.LeapfrogIntegrator(system, 0.25)
                state_list = [
                    ChainState(pos=q, mom=p, dir=1)
                    for q, p in rng.standard_normal((N_STATE, 2, size))]
                integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-2)


class TestLeapfrogIntegratorNonLinearSystem(IntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in SIZES:
            for metric in _generate_metrics(rng, size):
                system = systems.EuclideanMetricSystem(
                    neg_log_dens=lambda q: 0.25 * np.sum(q**4),
                    metric=metric,
                    grad_neg_log_dens=lambda q: q**3)
                integrator = integrators.LeapfrogIntegrator(system, 0.05)
                state_list = [
                    ChainState(pos=q, mom=p, dir=1)
                    for q, p in rng.standard_normal((N_STATE, 2, size))]
                integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-2)


class TestLeapfrogIntegratorGaussianLinearSystem(
        LinearSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in SIZES:
            for metric in _generate_metrics(rng, size):
                system = systems.GaussianEuclideanMetricSystem(
                    neg_log_dens=lambda q: 0,
                    metric=metric,
                    grad_neg_log_dens=lambda q: 0 * q)
                integrator = integrators.LeapfrogIntegrator(system, 0.5)
                state_list = [
                    ChainState(pos=q, mom=p, dir=1)
                    for q, p in rng.standard_normal((N_STATE, 2, size))]
                integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-10)


class TestLeapfrogIntegratorGaussianNonLinearSystem(IntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in SIZES:
            for metric in _generate_metrics(rng, size):
                system = systems.GaussianEuclideanMetricSystem(
                    neg_log_dens=lambda q: 0.125 * np.sum(q**4),
                    metric=metric,
                    grad_neg_log_dens=lambda q: 0.5 * q**3)
                integrator = integrators.LeapfrogIntegrator(system, 0.1)
                state_list = [
                    ChainState(pos=q, mom=p, dir=1)
                    for q, p in rng.standard_normal((N_STATE, 2, size))]
                integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-2)


class TestImplicitLeapfrogIntegratorLinearSystem(
        LinearSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in SIZES:
            system = systems.DenseRiemannianMetricSystem(
                lambda q: 0.5 * np.sum(q**2), grad_neg_log_dens=lambda q: q,
                metric_func=lambda q: np.identity(q.shape[0]),
                vjp_metric_func=lambda q: lambda m: np.zeros_like(q))
            integrator = integrators.ImplicitLeapfrogIntegrator(system, 0.5)
            state_list = [
                ChainState(pos=q, mom=p, dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size))]
            integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=5e-3)


class TestConstrainedLeapfrogIntegratorLinearSystem(
        ConstrainedSystemIntegratorTestCase, LinearSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in [s for s in SIZES if s > 1]:
            for metric in _generate_metrics(rng, size):
                system = systems.DenseConstrainedEuclideanMetricSystem(
                    neg_log_dens=lambda q: 0.5 * np.sum(q**2),
                    metric=metric,
                    grad_neg_log_dens=lambda q: q,
                    constr=lambda q: q[:1],
                    jacob_constr=lambda q: np.eye(1, q.shape[0], 0))
                integrator = integrators.ConstrainedLeapfrogIntegrator(
                    system, 0.1)
                state_list = [
                    ChainState(
                        pos=np.concatenate([np.array([0.]), q]),
                        mom=metric @ np.concatenate([np.array([0.]), p]),
                        dir=1)
                    for q, p in rng.standard_normal((N_STATE, 2, size - 1))
                ]
                integrators_and_state_lists.append(
                    (integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-2)


class TestConstrainedLeapfrogIntegratorNonLinearSystem(
        ConstrainedSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in [s for s in SIZES if s > 1]:
            for metric in _generate_metrics(rng, size):
                for projection_solver in [
                        solvers.solve_projection_onto_manifold_quasi_newton,
                        solvers.solve_projection_onto_manifold_newton]:
                    system = systems.DenseConstrainedEuclideanMetricSystem(
                        neg_log_dens=lambda q: 0.125 * np.sum(q**4),
                        metric=metric,
                        grad_neg_log_dens=lambda q: 0.5 * q**3,
                        constr=lambda q: q[0:1]**2 + q[1:2]**2 - 1.,
                        jacob_constr=lambda q: np.concatenate(
                            [2 * q[0:1], 2 * q[1:2],
                             np.zeros(q.shape[0] - 2)])[None])
                    integrator = integrators.ConstrainedLeapfrogIntegrator(
                        system, 0.1, projection_solver=projection_solver)
                    state_list = [
                        ChainState(
                            pos=np.concatenate([
                                np.array([np.cos(theta), np.sin(theta)]), q]),
                            mom=None, dir=1)
                        for theta, q in zip(
                            rng.uniform(size=N_STATE) * 2 * np.pi,
                            rng.standard_normal((N_STATE, size - 2)))
                    ]
                    for state in state_list:
                        state.mom = system.sample_momentum(state, rng)
                    integrators_and_state_lists.append(
                        (integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-2)


class TestConstrainedLeapfrogIntegratorGaussianLinearSystem(
        ConstrainedSystemIntegratorTestCase, LinearSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in [s for s in SIZES if s > 1]:
            system = systems.GaussianDenseConstrainedEuclideanMetricSystem(
                lambda q: 0., grad_neg_log_dens=lambda q: 0. * q,
                constr=lambda q: q[:1],
                jacob_constr=lambda q: np.identity(q.shape[0])[:1],
                mhp_constr=lambda q: lambda m: np.zeros_like(q))
            integrator = integrators.ConstrainedLeapfrogIntegrator(system, 0.5)
            state_list = [
                ChainState(pos=np.concatenate([np.array([0.]), q]),
                           mom=np.concatenate([np.array([0.]), p]), dir=1)
                for q, p in rng.standard_normal((N_STATE, 2, size - 1))]
            integrators_and_state_lists.append((integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=1e-10)


class TestConstrainedLeapfrogIntegratorGaussianNonLinearSystem(
        ConstrainedSystemIntegratorTestCase):

    def __init__(self):
        rng = np.random.RandomState(SEED)
        integrators_and_state_lists = []
        for size in [s for s in SIZES if s > 1]:
            for metric in _generate_metrics(rng, size):
                for projection_solver in [
                        solvers.solve_projection_onto_manifold_quasi_newton,
                        solvers.solve_projection_onto_manifold_newton]:
                    system = (
                        systems.GaussianDenseConstrainedEuclideanMetricSystem(
                            neg_log_dens=lambda q: 0.125 * np.sum(q**4),
                            metric=metric,
                            grad_neg_log_dens=lambda q: 0.5 * q**3,
                            constr=lambda q: q[0:1]**2 + q[1:2]**2 - 1.,
                            jacob_constr=lambda q: np.concatenate(
                                [2 * q[0:1], 2 * q[1:2],
                                 np.zeros(q.shape[0] - 2)])[None],
                            mhp_constr=lambda q: lambda m: 0 * m[0]))
                    integrator = integrators.ConstrainedLeapfrogIntegrator(
                        system, 0.05, projection_solver=projection_solver)
                    state_list = [
                        ChainState(
                            pos=np.concatenate([
                                np.array([np.cos(theta), np.sin(theta)]), q]),
                            mom=None, dir=1)
                        for theta, q in zip(
                            rng.uniform(size=N_STATE) * 2 * np.pi,
                            rng.standard_normal((N_STATE, size - 2)))
                    ]
                    for state in state_list:
                        state.mom = system.sample_momentum(state, rng)
                        state._read_only = True
                    integrators_and_state_lists.append(
                        (integrator, state_list))
        super().__init__(integrators_and_state_lists, h_diff_tol=5e-2)
