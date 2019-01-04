"""Solvers for non-linear systems of equations for implicit integrators."""

from hmc.errors import ConvergenceError
import numpy as np
import scipy.linalg as sla


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return np.sum(vct**2)**0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return np.max(abs(vct))


def solve_fixed_point_direct(func, x0, tol=1e-8, max_iters=100,
                             norm=maximum_norm):
    """Solve fixed point equation f(x) = x using direct iteration.

    Args:
        func: Single argument function to find fixed point of.
        x0: Initial state (function argument).
        tol: Convergence tolerance - terminates when max_abs(f(x) - x) < tol.
        max_iters: Maximum number of iterations before raising exception.

    Returns:
        Solution to fixed point equation with max_abs(f(x) - x) < tol.

    Raises:
        ConvergenceError if not converged within max_iters iterations.
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
    """Solve fixed point equation f(x) = x using Steffensen's method.

    Steffennsen's method [1] achieves quadratic convergence but at the cost of
    two function evaluations per iteration so for functions where convergence
    is achieved in a small number of iterations, direct iteration may be
    cheaper.

    [1] : https://en.wikipedia.org/wiki/Steffensen%27s_method

    Args:
        func: Single argument function to find fixed point of.
        x0: Initial state (function argument).
        tol: Convergence tolerance - terminates when max_abs(f(x) - x) < tol.
        max_iters: Maximum number of iterations before raising exception.

    Returns:
        Solution to fixed point equation with max_abs(f(x) - x) < tol.

    Raises:
        ConvergenceError if not converged within max_iters iterations.
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


def project_onto_manifold_quasi_newton(state, state_prev, system, tol=1e-8,
                                       max_iters=100, norm=maximum_norm):
    jacob_constr_prev = system.jacob_constr(state_prev)
    chol_gram_prev = (system.chol_gram(state_prev), True)
    for i in range(max_iters):
        constr = system.constr(state)
        if np.any(np.isinf(constr)) or np.any(np.isnan(constr)):
            raise ConvergenceError(f'Quasi-Newton iteration diverged.')
        error = norm(constr)
        if error < tol:
            return state
        state.pos -= system.metric.lmult_inv(
            jacob_constr_prev.T @ sla.cho_solve(chol_gram_prev, constr))
    raise ConvergenceError(
        f'Quasi-Newton iteration did not converge. Last error {error:.1e}.')


def project_onto_manifold_newton(state, state_prev, system, tol=1e-8,
                                 max_iters=100, norm=maximum_norm):
    inv_metr_jacob_constr_prev_t = system.inv_metric_jacob_constr_t(state_prev)
    for i in range(max_iters):
        jacob_constr = system.jacob_constr(state)
        constr = system.constr(state)
        if np.any(np.isinf(constr)) or np.any(np.isnan(constr)):
            raise ConvergenceError(f'Newton iteration diverged.')
        error = norm(constr)
        if error < tol:
            return state
        state.pos -= inv_metr_jacob_constr_prev_t @ sla.solve(
            jacob_constr @ inv_metr_jacob_constr_prev_t, constr)
    raise ConvergenceError(
        f'Newton iteration did not converge. Last error {error:.1e}.')
