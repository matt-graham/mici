"""Solvers for non-linear systems of equations for implicit integrators."""

from mici.errors import ConvergenceError, LinAlgError
import numpy as np


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return np.sum(vct**2)**0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return np.max(abs(vct))


def solve_fixed_point_direct(
        func, x0, convergence_tol=1e-9, divergence_tol=1e10, max_iters=100,
        norm=maximum_norm):
    """Solve fixed point equation `func(x) = x` using direct iteration.

    Args:
        func (Callable[[array], array]): Function to find fixed point of.
        x0 (array): Initial state (function argument).
        convergence_tol (float): Convergence tolerance - solver successfully
            terminates when `norm(func(x) - x) < convergence_tol`.
        divergence_tol (float): Divergence tolerance - solver aborts if
            `norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters (int): Maximum number of iterations before raising exception.
        norm (Callable[[array], float]): Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with
        `norm(func(x) - x) < convergence_tol`.

    Raises:
        `mici.errors.ConvergenceError` if solver does not converge within
        `max_iters` iterations, diverges or encounters a `ValueError` during
        the iteration.
    """
    for i in range(max_iters):
        try:
            x = func(x0)
            error = norm(x - x0)
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f'Fixed point iteration diverged on iteration {i}.'
                    f'Last error={error:.1e}.')
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f'{type(e)} at iteration {i} of fixed point solver ({e}).')
    raise ConvergenceError(
        f'Fixed point iteration did not converge. Last error={error:.1e}.')


def solve_fixed_point_steffensen(
        func, x0, convergence_tol=1e-9, divergence_tol=1e10, max_iters=100,
        norm=maximum_norm):
    """Solve fixed point equation `func(x) = x` using Steffensen's method.

    Steffennsen's method [1] achieves quadratic convergence but at the cost of
    two function evaluations per iteration so for functions where convergence
    is achieved in a small number of iterations, direct iteration may be
    cheaper.

    [1] : https://en.wikipedia.org/wiki/Steffensen%27s_method

    Args:
        func (Callable[[array], array]): Function to find fixed point of.
        x0 (array): Initial state (function argument).
        convergence_tol (float): Convergence tolerance - solver successfully
            terminates when `norm(func(x) - x) < convergence_tol`.
        divergence_tol (float): Divergence tolerance - solver aborts if
            `norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters (int): Maximum number of iterations before raising exception.
        norm (Callable[[array], float]): Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with
        `norm(func(x) - x) < convergence_tol`.

    Raises:
        `mici.errors.ConvergenceError` if solver does not converge within
        `max_iters` iterations, diverges or encounters a `ValueError` during
        the iteration.
    """
    for i in range(max_iters):
        try:
            x1 = func(x0)
            x2 = func(x1)
            denom = x2 - 2 * x1 + x0
            # Set any zero values in denominator of update term to smalllest
            # floating point value to prevent divide-by-zero errors
            denom[abs(denom) == 0.] = np.finfo(x0.dtype).eps
            x = x0 - (x1 - x0)**2 / denom
            error = norm(x - x0)
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f'Fixed point iteration diverged on iteration {i}.'
                    f'Last error={error:.1e}.')
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f'{type(e)} at iteration {i} of fixed point solver ({e}).')
    raise ConvergenceError(
        f'Fixed point iteration did not converge. Last error={error:.1e}.')


def solve_projection_onto_manifold_quasi_newton(
        state, state_prev, dt, system, constraint_tol=1e-9, position_tol=1e-8,
        divergence_tol=1e10, max_iters=50, norm=maximum_norm):
    """Solve constraint equation using quasi-Newton method.

    Uses a quasi-Newton iteration to solve the non-linear system of equations
    in `λ`

        system.constr(
            state.pos + dh2_flow_pos_dmom @
                system.jacob_constr(state_prev).T @ λ) == 0

    where `dh2_flow_pos_dmom = system.dh2_flow_dmom(dt)[0]` is the derivative
    of the action of the (linear) `system.h2_flow` map on the state momentum
    component with respect to the position component, `state` is a post
    (unconstrained) `system.h2_flow` update state with position component
    outside of the manifold and `state_prev` is the corresponding pre-update
    state in the co-tangent bundle.

    Only requires re-evaluating the constraint function `system.constr` within
    the solver loop and no recomputation of matrix decompositions on each
    iteration.

    Args:
        state (mici.states.ChainState): Post `h2_flow `update state to project.
        state_prev (mici.states.ChainState): Previous state in co-tangent
            bundle manifold before `h2_flow` update which defines the
            co-tangent space to perform projection in.
        dt (float): Integrator time step used in `h2_flow` update.
        system (mici.systems.ConstrainedEuclideanMetricSystem): Hamiltonian
           system defining `h2_flow` and `constr` functions used to define
           constraint equation to solve.
        constraint_tol (float): Convergence tolerance in constraint space.
           Iteration will continue until `norm(constr(pos)) < constraint_tol`
           where `pos` is the position at the current iteration.
        position_tol (float): Convergence tolerance in position space.
           Iteration will continue until `norm(delt_pos) < position_tol`
           where `delta_pos` is the change in the position in the current
           iteration.
        divergence_tol (float): Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos`
            is the position at the current iteration and raises
            `mici.errors.ConvergenceError`.
        max_iters (int): Maximum number of iterations to perform before
            aborting and raising `mici.errors.ConvergenceError`.
        norm (Callable[[array], float]): Norm to use to test for convergence.

    Returns:
        Updated `state` object with position component satisfying constraint
        equation to within `constraint_tol`, i.e.
        `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        `mici.errors.ConvergenceError` if solver does not converge within
        `max_iters` iterations, diverges or encounters a `ValueError` during
        the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(dt))
    inv_jacob_constr_inner_product = system.jacob_constr_inner_product(
        jacob_constr_prev, dh2_flow_pos_dmom).inv
    for i in range(max_iters):
        try:
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (
                inv_jacob_constr_inner_product @ constr)
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f'Quasi-Newton solver diverged on iteration {i}. '
                    f'Last |constr|={error:.1e}, '
                    f'|delta_pos|={norm(delta_pos):.1e}.')
            elif error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(dt) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f'{type(e)} at iteration {i} of quasi-Newton solver ({e}).')
    raise ConvergenceError(
        f'Quasi-Newton solver did not converge with {max_iters} iterations. '
        f'Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}.')


def solve_projection_onto_manifold_newton(
        state, state_prev, dt, system, constraint_tol=1e-9, position_tol=1e-8,
        divergence_tol=1e10, max_iters=50, norm=maximum_norm):
    """Solve constraint equation using Newton method.

    Uses a Newton iteration to solve the non-linear system of equations in `λ`

        system.constr(
            state.pos + dh2_flow_pos_dmom @
                system.jacob_constr(state_prev).T @ λ) == 0

    where `dh2_flow_pos_dmom = system.dh2_flow_dmom(dt)[0]` is the derivative
    of the action of the (linear) `system.h2_flow` map on the state momentum
    component with respect to the position component, `state` is a post
    (unconstrained) `system.h2_flow` update state with position component
    outside of the manifold and `state_prev` is the corresponding pre-update
    state in the co-tangent bundle.

    Requires re-evaluating both the constraint function `system.constr` and
    constraint Jacobian `system.jacob_constr` within the solver loop and
    computation of matrix decompositions of a preconditioned matrix on each
    iteration.

    Args:
        state (mici.states.ChainState): Post `h2_flow `update state to project.
        state_prev (mici.states.ChainState): Previous state in co-tangent
            bundle manifold before `h2_flow` update which defines the
            co-tangent space to perform projection in.
        dt (float): Integrator time step used in `h2_flow` update.
        system (mici.systems.ConstrainedEuclideanMetricSystem): Hamiltonian
           system defining `h2_flow` and `constr` functions used to define
           constraint equation to solve.
        constraint_tol (float): Convergence tolerance in constraint space.
           Iteration will continue until `norm(constr(pos)) < constraint_tol`
           where `pos` is the position at the current iteration.
        position_tol (float): Convergence tolerance in position space.
           Iteration will continue until `norm(delt_pos) < position_tol`
           where `delta_pos` is the change in the position in the current
           iteration.
        divergence_tol (float): Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos`
            is the position at the current iteration and raises
            `mici.errors.ConvergenceError`.
        max_iters (int): Maximum number of iterations to perform before
            aborting and raising `mici.errors.ConvergenceError`.
        norm (Callable[[array], float]): Norm to use to test for convergence.

    Returns:
        Updated `state` object with position component satisfying constraint
        equation to within `constraint_tol`, i.e.
        `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        `mici.errors.ConvergenceError` if solver does not converge within
        `max_iters` iterations, diverges or encounters a `ValueError` during
        the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(dt))
    for i in range(max_iters):
        try:
            jacob_constr = system.jacob_constr(state)
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (
                system.jacob_constr_inner_product(
                    jacob_constr, dh2_flow_pos_dmom, jacob_constr_prev).inv @
                constr)
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f'Newton solver diverged at iteration {i}. '
                    f'Last |constr|={error:.1e}, '
                    f'|delta_pos|={norm(delta_pos):.1e}.')
            if error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(dt) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f'{type(e)} at iteration {i} of Newton solver ({e}).')
    raise ConvergenceError(
        f'Newton solver did not converge in {max_iters} iterations. '
        f'Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}.')
