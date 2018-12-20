"""Solvers for non-linear systems of equations for implicit integrators."""

from hmc.utils import maximum_norm
from hmc.errors import ConvergenceError
import numpy as np


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
                f'Fixed point iteration diverged, last error {error:.1e}.')
        error = norm(x - x0)
        if error < tol:
            return x
        x0 = x
    raise ConvergenceError(
        f'Fixed point iteration did not converge, last error {error:.1e}.')


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
                f'Fixed point iteration diverged, last error {error:.1e}.')
        error = norm(x - x0)
        if error < tol:
            return x
        x0 = x
    raise ConvergenceError(
        f'Fixed point iteration did not converge, last error {error:.1e}.')
