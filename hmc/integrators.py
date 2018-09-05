"""Symplectic integrators for simulation of Hamiltonian dynamics."""

import numpy as np
import scipy.linalg as sla
from hmc.solvers import solve_fixed_point_direct
from hmc.utils import max_abs


class IntegratorError(RuntimeError):
    """Error raised when integrator step fails."""


class LeapfrogIntegrator(object):
    """Explicit leapfrog integrator for separable Hamiltonian systems."""

    def __init__(self, system, step_size):
        self.system = system
        self.step_size = step_size

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.deep_copy()
        state.mom -= 0.5 * dt * self.system.dh_dpos(state)
        state.pos += dt * self.system.dh_dmom(state)
        state.mom -= 0.5 * dt * self.system.dh_dpos(state)
        return state


class GeneralisedLeapfrogIntegrator(object):
    """Implicit leapfrog integrator for non-separable Hamiltonian systems."""

    def __init__(self, system, step_size, reverse_check_tol=1e-8,
                 fixed_point_solver=solve_fixed_point_direct,
                 **fixed_point_solver_kwargs):
        self.system = system
        self.step_size = step_size
        self.reverse_check_tol = reverse_check_tol
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
        rev_diff = max_abs(state.mom - mom_init)
        if rev_diff > self.reverse_check_tol:
            raise IntegratorError(
                f'Non-reversible step. Maximum difference between initial and '
                f'forward-backward integrated momentum = {rev_diff:.1e}.')
        state.mom = mom_fwd

    def step_c1(self, state, dt):
        pos_init = state.pos.copy()
        state.pos += dt * self.system.dh_dmom(state)
        pos_fwd = state.pos.copy()
        self.step_c2(state, -dt)
        rev_diff = max_abs(state.pos - pos_init)
        if rev_diff > self.reverse_check_tol:
            raise IntegratorError(
                f'Non-reversible step. Difference between initial and '
                f'forward-backward integrated position = {rev_diff:.1e}.')
        state.pos = pos_fwd

    def step_c2(self, state, dt):
        def fixed_point_func(pos):
            state.pos = pos
            return pos_init + dt * self.system.dh_dmom(state)
        pos_init = state.pos
        state.pos = self.solve_fixed_point(fixed_point_func, pos_init)

    def step(self, state):
        dt = 0.5 * state.dir * self.step_size
        state = state.deep_copy()
        self.step_a(state, dt)
        self.step_b1(state, dt)
        self.step_c1(state, dt)
        self.step_c2(state, dt)
        self.step_b2(state, dt)
        self.step_a(state, dt)
        return state


class GeodesicLeapfrogIntegrator(object):
    """Leapfrog iterator for constrained separable Hamiltonian systems."""

    def __init__(self, system, step_size, n_inner_step=1, tol=1e-8,
                 max_iters=100):
        self.system = system
        self.step_size = step_size
        self.n_inner_step = n_inner_step
        self.tol = tol
        self.max_iters = max_iters

    def project_mom(self, mom, dc_dpos, chol_gram):
        mom -= dc_dpos.T @ sla.cho_solve(chol_gram, dc_dpos @ mom)

    def project_pos(self, pos, dc_dpos, chol_gram):
        for i in range(self.max_iters):
            delta = self.system.constr_func(pos)
            if max_abs(delta) < self.tol:
                return pos
            pos -= dc_dpos.T @ sla.cho_solve(chol_gram, delta)
        err = max_abs(self.system.constr_func(pos))
        raise IntegratorError(
            f'Quasi-Newton iteration did not converge. Last error {err:.1e}.')

    def step_a(self, state, dt):
        state.mom -= dt * self.system.dh_dpos(state)
        self.project_mom(state.mom, state.dc_dpos, state.chol_gram)

    def step_b(self, state, dt):
        dt_i = dt / self.n_inner_step
        for i in range(self.n_inner_step):
            pos_i = state.pos
            state.pos = state.pos + dt_i * state.mom
            self.project_pos(state.pos, state.dc_dpos, state.chol_gram)
            self.system.update_constr_jacob_and_chol_gram(state)
            state.mom = (state.pos - pos_i) / dt_i
            self.project_mom(state.mom, state.dc_dpos, state.chol_gram)
            pos_r = state.pos - dt_i * state.mom
            self.project_pos(pos_r, state.dc_dpos, state.chol_gram)
            rev_diff = max_abs(pos_i - pos_r)
            if rev_diff > 2 * self.tol:
                raise IntegratorError(
                    f'Non-reversible step. Difference between initial and '
                    f'forward-backward positions = {rev_diff:.1e}')

    def step(self, state):
        dt = state.dir * self.step_size
        state = state.deep_copy()
        self.step_a(state, 0.5 * dt)
        self.step_b(state, dt)
        self.step_a(state, 0.5 * dt)
        return state
