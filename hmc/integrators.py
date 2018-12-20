"""Symplectic integrators for simulation of Hamiltonian dynamics."""

import numpy as np
from hmc.errors import IntegratorError, NonReversibleStepError
from hmc.solvers import solve_fixed_point_direct
from hmc.utils import maximum_norm


class LeapfrogIntegrator(object):
    """
    Explicit leapfrog integrator for separable Hamiltonian systems.

    Here separable means that the Hamiltonian can be expressed as the sum of
    a term depending only on the position (target) variables, typically denoted
    the potential energy, and a second term depending only on the momentum
    variables, typically denoted the kinetic energy, i.e.

        h(pos, mom) = pot_energy(pos) + kin_energy(mom)

    with pos and mom the position and momentum variables respectively and
    pot_energy and kin_energy the potential and kinetic energy functions
    respectively.
    """

    def __init__(self, system, step_size):
        self.system = system
        self.step_size = step_size

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.copy()
        state.mom -= 0.5 * dt * self.system.dh_dpos(state)
        state.pos += dt * self.system.dh_dmom(state)
        state.mom -= 0.5 * dt * self.system.dh_dpos(state)
        return state


class SplitLeapfrogIntegrator(object):
    """
    Split leapfrog integrator for Hamiltonian systems with tractable component.

    The Hamiltonian function is assumed to have an analytically tractable
    component (e.g a quadratic form in the position and momentum variables) for
    which the corresponding Hamiltonian flow can be exactly simulated.
    Specifically it is assumed that the Hamiltonian function h takes the form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where pos and mom are the position and momentum variables respectively,
    with h2(pos, mom) a Hamiltonian for which the exact flow can be computed.
    """

    def __init__(self, system, step_size):
        self.system = system
        self.step_size = step_size

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.copy()
        state.mom -= 0.5 * dt * self.system.dh1_dpos(state)
        state = self.system.h2_exact_flow(state, dt)
        state.mom -= 0.5 * dt * self.system.dh1_dpos(state)
        return state


class GeneralisedLeapfrogIntegrator(object):
    """
    Implicit leapfrog integrator for non-separable Hamiltonian systems.

    The Hamiltonian function is assumed to take the form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where pos and mom are the position and momentum variables respectively, and
    h2 is a non-separable function of the position and momentum variables and
    for which exact simulation of the correspond Hamiltonian flow is not
    possible.
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
        state.mom -= dt * self.system.dh1_dpos(state)

    def step_b1(self, state, dt):
        def fixed_point_func(mom):
            state.mom = mom
            return mom_init - dt * self.system.dh2_dpos(state)
        mom_init = state.mom
        state.mom = self.solve_fixed_point(fixed_point_func, mom_init)

    def step_b2(self, state, dt):
        mom_init = state.mom.copy()
        state.mom -= dt * self.system.dh2_dpos(state)
        mom_fwd = state.mom.copy()
        self.step_b1(state, -dt)
        rev_diff = self.reverse_check_norm(state.mom - mom_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated momentums = {rev_diff:.1e}.')
        state.mom = mom_fwd

    def step_c1(self, state, dt):
        pos_init = state.pos.copy()
        state.pos += dt * self.system.dh_dmom(state)
        pos_fwd = state.pos.copy()
        self.step_c2(state, -dt)
        rev_diff = self.reverse_check_norm(state.pos - pos_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated positions = {rev_diff:.1e}.')
        state.pos = pos_fwd

    def step_c2(self, state, dt):
        def fixed_point_func(pos):
            state.pos = pos
            return pos_init + dt * self.system.dh_dmom(state)
        pos_init = state.pos
        state.pos = self.solve_fixed_point(fixed_point_func, pos_init)

    def step(self, state):
        dt = 0.5 * state.dir * self.step_size
        state = state.copy()
        self.step_a(state, dt)
        self.step_b1(state, dt)
        self.step_c1(state, dt)
        self.step_c2(state, dt)
        self.step_b2(state, dt)
        self.step_a(state, dt)
        return state


class BaseConstrainedLeapfrogIntegrator(object):
    """Leapfrog iterator for constrained separable Hamiltonian systems."""

    def __init__(self, system, step_size, n_inner_step=1,
                 reverse_check_tol=1e-8, reverse_check_norm=maximum_norm):
        self.system = system
        self.step_size = step_size
        self.n_inner_step = n_inner_step
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm

    def step_a(self, state, dt):
        raise NotImplementedError()

    def step_b_inner(self, state, dt):
        raise NotImplementedError()

    def solve_for_mom_post_projection(self, state, state_prev, dt):
        raise NotImplementedError()

    def step_b(self, state, dt):
        dt_i = dt / self.n_inner_step
        for i in range(self.n_inner_step):
            state_prev = state.copy()
            self.step_b_inner(state, dt_i)
            self.system.project_onto_manifold(state, state_prev)
            self.solve_for_mom_post_projection(state, state_prev, dt_i)
            self.system.project_onto_tangent_space(state)
            state_back = state.copy()
            self.step_b_inner(state_back, -dt_i)
            self.system.project_onto_manifold(state_back, state)
            rev_diff = self.reverse_check_norm(state_back.pos - state_prev.pos)
            if rev_diff > self.reverse_check_tol:
                raise NonReversibleStepError(
                    f'Non-reversible step. Distance between initial and '
                    f'forward-backward integrated positions = {rev_diff:.1e}.')

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.copy()
        self.step_a(state, 0.5 * dt)
        self.system.project_onto_tangent_space(state)
        self.step_b(state, dt)
        self.step_a(state, 0.5 * dt)
        self.system.project_onto_tangent_space(state)
        return state


class GeodesicConstrainedLeapfrogIntegrator(BaseConstrainedLeapfrogIntegrator):

    def step_a(self, state, dt):
        state.mom -= dt * self.system.dh_dpos(state)

    def step_b_inner(self, state, dt):
        state.pos += dt * self.system.dh_dmom(state)

    def solve_for_mom_post_projection(self, state, state_prev, dt):
        state.mom = self.system.solve_dh_dmom_for_mom(
            (state.pos - state_prev.pos) / dt)


class SplitConstrainedLeapfrogIntegrator(BaseConstrainedLeapfrogIntegrator):

    def step_a(self, state, dt):
        state.mom -= dt * self.system.dh1_dpos(state)

    def step_b_inner(self, state, dt):
        self.system.h2_exact_flow(state, dt)

    def solve_for_mom_post_projection(self, state, state_prev, dt):
        state.mom = self.system.solve_h2_flow_for_mom_gvn_pos(
            state.pos, state_prev.pos, dt)
