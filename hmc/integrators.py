"""Symplectic integrators for simulation of Hamiltonian dynamics."""

from hmc.errors import NonReversibleStepError
from hmc.solvers import (maximum_norm, solve_fixed_point_direct,
                         retract_onto_manifold_quasi_newton)


class ExplicitLeapfrogIntegrator(object):
    """
    Leapfrog integrator for Hamiltonian systems with tractable component flows.

    The Hamiltonian function is assumed to be expressible as the sum of two
    analytically tractable components for which the corresponding Hamiltonian
    flows can be exactly simulated. Specifically it is assumed that the
    Hamiltonian function `h` takes the form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where `pos` and `mom` are the position and momentum variables respectively,
    and `h1(pos)` and `h2(pos, mom)` Hamiltonian components for which the exact
    flows can be computed.
    """

    def __init__(self, system, step_size):
        self.system = system
        self.step_size = step_size

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.copy()
        self.system.h1_flow(state, 0.5 * dt)
        self.system.h2_flow(state, dt)
        self.system.h1_flow(state, 0.5 * dt)
        return state


LeapfrogIntegrator = ExplicitLeapfrogIntegrator


class ImplicitLeapfrogIntegrator(object):
    """
    Implicit leapfrog integrator for Hamiltonian with non-separable component.

    The Hamiltonian function is assumed to take the form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where `pos` and `mom` are the position and momentum variables respectively,
    and `h2` is a non-separable function of the position and momentum variables
    and for which exact simulation of the correspond Hamiltonian flow is not
    possible. A pair of implicit component updates are used to approximate the
    flow due to the `h2` Hamiltonian component, with a fixed-point iteration
    used to solve the non-linear system of equations.
    """

    def __init__(self, system, step_size, reverse_check_tol=1e-8,
                 reverse_check_norm=maximum_norm,
                 fixed_point_solver=solve_fixed_point_direct,
                 **fixed_point_solver_kwargs):
        self.system = system
        self.step_size = step_size
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = maximum_norm
        self.fixed_point_solver = fixed_point_solver
        self.fixed_point_solver_kwargs = fixed_point_solver_kwargs

    def solve_fixed_point(self, fixed_point_func, x_init):
        return self.fixed_point_solver(
            fixed_point_func, x_init, **self.fixed_point_solver_kwargs)

    def step_a(self, state, dt):
        self.system.h1_flow(state, dt)

    def step_b_fwd(self, state, dt):
        def fixed_point_func(mom):
            state.mom = mom
            return mom_init - dt * self.system.dh2_dpos(state)
        mom_init = state.mom
        state.mom = self.solve_fixed_point(fixed_point_func, mom_init)

    def step_b_adj(self, state, dt):
        mom_init = state.mom.copy()
        state.mom -= dt * self.system.dh2_dpos(state)
        state_back = state.copy()
        self.step_b_fwd(state_back, -dt)
        rev_diff = self.reverse_check_norm(state_back.mom - mom_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated momentums = {rev_diff:.1e}.')

    def step_c_fwd(self, state, dt):
        pos_init = state.pos.copy()
        state.pos += dt * self.system.dh2_dmom(state)
        state_back = state.copy()
        self.step_c_adj(state_back, -dt)
        rev_diff = self.reverse_check_norm(state_back.pos - pos_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated positions = {rev_diff:.1e}.')

    def step_c_adj(self, state, dt):
        def fixed_point_func(pos):
            state.pos = pos
            return pos_init + dt * self.system.dh2_dmom(state)
        pos_init = state.pos
        state.pos = self.solve_fixed_point(fixed_point_func, pos_init)

    def step(self, state):
        dt = 0.5 * state.dir * self.step_size
        state = state.copy()
        self.step_a(state, dt)
        self.step_b_fwd(state, dt)
        self.step_c_fwd(state, dt)
        self.step_c_adj(state, dt)
        self.step_b_adj(state, dt)
        self.step_a(state, dt)
        return state


class ConstrainedLeapfrogIntegrator(object):
    """
    Leapfrog integrator for constrained Hamiltonian systems.

    The Hamiltonian function is assumed to be expressible as the sum of two
    components for which the corresponding (unconstrained) Hamiltonian flows
    can be exactly simulated. Specifically it is assumed that the Hamiltonian
    function `h` takes the form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where `pos` and `mom` are the position and momentum variables respectively,
    and `h1(pos)` and `h2(pos, mom)` Hamiltonian components for which the exact
    flows can be computed.

    The system is assumed to be additionally subject to a set of holonomic
    constraints on the position component of the state i.e. that all valid
    states must satisfy

        all(constr(pos) == 0)

    for some vector constraint function `constr`, with the set of positions
    satisfying the constraints implicitly defining a manifold. There is also
    a corresponding constraint implied on the momentum variables which can
    be derived by differentiating the above with respect to time and using
    that under the Hamiltonian dynamics the time derivative of the position
    is equal to the negative derivative of the Hamiltonian function with
    respect to the momentum

        all(jacob_constr(pos) @ dh2_dmom(mom) == 0)

    The set of momentum variables satisfying the above for given position
    variables is termed the cotangent space of the manifold (at a position),
    and the set of (position, momentum) pairs for which the position is on the
    constraint manifold and the momentum in the corresponding cotangent space
    is termed the cotangent bundle.

    The integrator exactly preserves these constraints at all steps, such that
    if an initial position momentum pair `(pos, mom)` are in the cotangent
    bundle, the corresponding pair after calling the `step` method of the
    integrator will also be in the cotangent bundle.
    """

    def __init__(self, system, step_size, n_inner_step=1,
                 reverse_check_tol=1e-8, reverse_check_norm=maximum_norm,
                 retraction_solver=retract_onto_manifold_quasi_newton,
                 retraction_solver_kwargs=None):
        self.system = system
        self.step_size = step_size
        self.n_inner_step = n_inner_step
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm
        self.retraction_solver = retraction_solver
        if retraction_solver_kwargs is None:
            retraction_solver_kwargs = {
                'tol': 1e-8, 'max_iters': 100, 'norm': maximum_norm}
        self.retraction_solver_kwargs = retraction_solver_kwargs

    def retract_onto_manifold(self, state, state_prev, dh2_flow_pos_dmom):
        self.retraction_solver(
            state, state_prev, dh2_flow_pos_dmom, self.system,
            **self.retraction_solver_kwargs)

    def project_onto_cotangent_space(self, state):
        self.system.project_onto_cotangent_space(state.mom, state)

    def step_a(self, state, dt):
        self.system.h1_flow(state, dt)
        self.project_onto_cotangent_space(state)

    def step_b(self, state, dt):
        dt_i = dt / self.n_inner_step
        for i in range(self.n_inner_step):
            state_prev = state.copy()
            self.system.h2_flow(state, dt_i)
            pos_uncons = state.pos.copy()
            dh2_flow_dmom = self.system.dh2_flow_dmom(state, dt_i)
            self.retract_onto_manifold(state, state_prev, dh2_flow_dmom[0])
            state.mom += dh2_flow_dmom[1] @ (
                dh2_flow_dmom[0].inv @ (state.pos - pos_uncons))
            self.project_onto_cotangent_space(state)
            state_back = state.copy()
            self.system.h2_flow(state_back, -dt_i)
            self.retract_onto_manifold(state_back, state, -dh2_flow_dmom[0])
            rev_diff = self.reverse_check_norm(state_back.pos - state_prev.pos)
            if rev_diff > self.reverse_check_tol:
                raise NonReversibleStepError(
                    f'Non-reversible step. Distance between initial and '
                    f'forward-backward integrated positions = {rev_diff:.1e}.')

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.copy()
        self.step_a(state, 0.5 * dt)
        self.step_b(state, dt)
        self.step_a(state, 0.5 * dt)
        return state
