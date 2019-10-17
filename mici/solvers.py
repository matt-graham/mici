"""Solvers for non-linear systems of equations for implicit integrators."""

from mici.errors import ConvergenceError
import numpy as np


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return np.sum(vct**2)**0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return np.max(abs(vct))


def solve_fixed_point_direct(func, x0, tol=1e-8, max_iters=100,
                             norm=maximum_norm):
    """Solve fixed point equation `func(x) = x` using direct iteration.

    Args:
        func: Single argument function to find fixed point of.
        x0: Initial state (function argument).
        tol: Convergence tolerance - terminates when `norm(func(x) - x) < tol`.
        max_iters: Maximum number of iterations before raising exception.
        norm: Vector norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with `norm(func(x) - x) < tol`.

    Raises:
        ConvergenceError if not converged within `max_iters` iterations.
    """
    for i in range(max_iters):
        x = func(x0)
        if any(np.isnan(x)) or any(np.isinf(x)):
            raise ConvergenceError(
                f'Fixed point iteration diverged on iteration {i}.')
        error = norm(x - x0)
        if error < tol:
            return x
        x0 = x
    raise ConvergenceError(
        f'Fixed point iteration did not converge. Last error {error:.1e}.')


def solve_fixed_point_steffensen(func, x0, tol=1e-8, max_iters=100,
                                 norm=maximum_norm):
    """Solve fixed point equation `func(x) = x` using Steffensen's method.

    Steffennsen's method [1] achieves quadratic convergence but at the cost of
    two function evaluations per iteration so for functions where convergence
    is achieved in a small number of iterations, direct iteration may be
    cheaper.

    [1] : https://en.wikipedia.org/wiki/Steffensen%27s_method

    Args:
        func: Single argument function to find fixed point of.
        x0: Initial state (function argument).
        tol: Convergence tolerance - terminates when `norm(func(x) - x) < tol`.
        max_iters: Maximum number of iterations before raising exception.
        norm: Vector norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with `norm(func(x) - x) < tol`.

    Raises:
        ConvergenceError if not converged within `max_iters` iterations.
    """
    for i in range(max_iters):
        x1 = func(x0)
        x2 = func(x1)
        x = x0 - (x1 - x0)**2 / (x2 - 2 * x1 + x0)
        if any(np.isnan(x)) or any(np.isinf(x)):
            raise ConvergenceError(
                f'Fixed point iteration diverged on iteration {i}.')
        error = norm(x - x0)
        if error < tol:
            return x
        x0 = x
    raise ConvergenceError(
        f'Fixed point iteration did not converge. Last error {error:.1e}.')


def retract_onto_manifold_quasi_newton(state, state_prev, dt, system, tol=1e-8,
                                       max_iters=100, norm=maximum_norm):
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    inv_jacob_constr_inner_product = system.jacob_constr_inner_product(
        jacob_constr_prev, dh2_flow_pos_dmom).inv
    for i in range(max_iters):
        constr = system.constr(state)
        if np.any(np.isinf(constr)) or np.any(np.isnan(constr)):
            raise ConvergenceError(f'Quasi-Newton iteration diverged.')
        error = norm(constr)
        if error < tol:
            state.mom -= dh2_flow_mom_dmom @ mu
            return state
        delta_mu = jacob_constr_prev.T @ (
            inv_jacob_constr_inner_product @ constr)
        mu += delta_mu
        state.pos -= dh2_flow_pos_dmom @ delta_mu
    raise ConvergenceError(
        f'Quasi-Newton iteration did not converge. Last error {error:.1e}.')


def retract_onto_manifold_newton(state, state_prev, dt, system, tol=1e-8,
                                 max_iters=100, norm=maximum_norm):
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(dt)
    for i in range(max_iters):
        jacob_constr = system.jacob_constr(state)
        constr = system.constr(state)
        if np.any(np.isinf(constr)) or np.any(np.isnan(constr)):
            raise ConvergenceError(f'Newton iteration diverged.')
        error = norm(constr)
        if error < tol:
            state.mom -= dh2_flow_mom_dmom @ mu
            return state
        delta_mu = jacob_constr_prev.T @ (
            system.jacob_constr_inner_product(
                jacob_constr, dh2_flow_pos_dmom, jacob_constr_prev).inv @
            constr)
        mu += delta_mu
        state.pos -= dh2_flow_pos_dmom @ delta_mu
    raise ConvergenceError(
        f'Newton iteration did not converge. Last error {error:.1e}.')
