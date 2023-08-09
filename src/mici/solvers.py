"""Solvers for non-linear systems of equations for implicit integrators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from mici.errors import ConvergenceError, LinAlgError

if TYPE_CHECKING:
    from mici.states import ChainState
    from mici.systems import (
        ConstrainedEuclideanMetricSystem,
        ConstrainedTractableFlowSystem,
    )
    from mici.types import ArrayFunction, ArrayLike, ScalarFunction


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return (vct**2).sum() ** 0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return (abs(vct)).max()


class FixedPointSolver(Protocol):
    """Solver for fixed point equation :code:`func(x) = x`."""

    def __call__(self, func: ArrayFunction, x0: ArrayLike, **kwargs) -> ArrayLike:
        """Solve fixed point equation.

        Args:
            func: Function to solve for fixed point of.
            x0: Point to initialize solver at.

        Returns:
            Fixed point solved for.
        """


def solve_fixed_point_direct(
    func: ArrayFunction,
    x0: ArrayLike,
    convergence_tol: float = 1e-9,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm: ScalarFunction = maximum_norm,
) -> ArrayLike:
    """Solve fixed point equation :code:`func(x) = x` using direct iteration.

    Args:
        func: Function to find fixed point of.
        x0: Initial state (function argument).
        convergence_tol: Convergence tolerance - solver successfully terminates when
            :code:`norm(func(x) - x) < convergence_tol`.
        divergence_tol: Divergence tolerance - solver aborts if
            :code:`norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters: Maximum number of iterations before raising exception.
        norm: Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with
        :code:`norm(func(x) - x) < convergence_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within
            :code:`max_iters` iterations, diverges or encounters a :py:exc:`ValueError`
            during the iteration.
    """
    for i in range(max_iters):
        try:
            x = func(x0)
            error = norm(x - x0)
            if error > divergence_tol or np.isnan(error):
                msg = (
                    f"Fixed point iteration diverged on iteration {i}. "
                    f"Last error={error:.1e}."
                )
                raise ConvergenceError(msg)
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            msg = f"{type(e)} at iteration {i} of fixed point solver ({e})."
            raise ConvergenceError(msg) from e
    msg = f"Fixed point iteration did not converge. Last error={error:.1e}."
    raise ConvergenceError(msg)


def solve_fixed_point_steffensen(
    func: ArrayFunction,
    x0: ArrayLike,
    convergence_tol: float = 1e-9,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm: ScalarFunction = maximum_norm,
) -> ArrayLike:
    """Solve fixed point equation :code:`func(x) = x` using Steffensen's method.

    Steffennsen's method achieves quadratic convergence but at the cost of two function
    evaluations per iteration so for functions where convergence is achieved in a small
    number of iterations, direct iteration may be cheaper.

    Args:
        func: Function to find fixed point of.
        x0: Initial state (function argument).
        convergence_tol: Convergence tolerance - solver successfully terminates
            when :code:`norm(func(x) - x) < convergence_tol`.
        divergence_tol: Divergence tolerance - solver aborts if
            :code:`norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters: Maximum number of iterations before raising exception.
        norm: Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with
        :code:`norm(func(x) - x) < convergence_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within
            :code:`max_iters` iterations, diverges or encounters a :py:exc:`ValueError`
            during the iteration.
    """
    for i in range(max_iters):
        try:
            x1 = func(x0)
            x2 = func(x1)
            denom = x2 - 2 * x1 + x0
            # Set any zero values in denominator of update term to smalllest
            # floating point value to prevent divide-by-zero errors
            denom[abs(denom) == 0.0] = np.finfo(x0.dtype).eps
            x = x0 - (x1 - x0) ** 2 / denom
            error = norm(x - x0)
            if error > divergence_tol or np.isnan(error):
                msg = (
                    f"Fixed point iteration diverged on iteration {i}. "
                    f"Last error={error:.1e}."
                )
                raise ConvergenceError(msg)
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            msg = f"{type(e)} at iteration {i} of fixed point solver ({e})."
            raise ConvergenceError(msg) from e
    msg = f"Fixed point iteration did not converge. Last error={error:.1e}."
    raise ConvergenceError(msg)


class ProjectionSolver(Protocol):
    r"""Solver for projection on to manifold step in constrained integrator.

    Solves an equation of the form

    .. math::

        r(\lambda) = c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda)) = 0,

    for the vector of Lagrange multipliers :math:`\lambda` to project a point on to the
    manifold defined by the zero level set of a constraint function :math:`c`, with
    :math:`\Phi_{2,1}` the flow map for the :math:`h_2` Hamiltonian component for the
    system restricted to the position component output. The map :math:`\Phi_{2,1}` is
    assumed to be linear in its second (momentum) argument.
    """

    def __call__(
        self,
        state: ChainState,
        state_prev: ChainState,
        time_step: float,
        system: ConstrainedTractableFlowSystem,
        **kwargs,
    ) -> ChainState:
        """Solve for projection on to manifold step.

        Args:
            state: Current chain state after unconstrained step.
            state_prev: Previous chain state on manifold.
            time_step: Integrator time step for unconstrained step.
            system: Hamiltonian system constrained dynamics are being simulated for.

        Returns:
            Chain state after projection on to manifold.
        """
        ...


def solve_projection_onto_manifold_quasi_newton(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: ConstrainedEuclideanMetricSystem,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    r"""Solve constraint equation using symmetric quasi-Newton method.

    Only requires re-evaluating the constraint function :code:`system.constr` within the
    solver loop and no recomputation of matrix decompositions on each iteration.

    Solves an equation of the form

    .. math::

        r(\lambda) = c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda)) = 0,

    for the vector of Lagrange multipliers :math:`\lambda` to project a point on to the
    manifold defined by the zero level set of a constraint function :math:`c`, with
    :math:`\Phi_{2,1}` the flow map for the :math:`h_2` Hamiltonian component for the
    system restricted to the position component output.

    The Jacobian of the residual function :math:`r` is

    .. math::

        \partial r(\lambda) =
        \partial c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda))
        \partial_2 (\Phi_{2,1}(t))(q, p + \partial c(q)^T \lambda)
        \partial c(q)^T

    where :math:`\partial_2 (\Phi_{2,1}(t))` is the Jacobian of the (position
    restricted) flow-map :math:`\Phi_{2,1}` with respect to its second (momentum)
    argument. We assume here that :math:`\Phi_{2,1}(t)` is linear in its second
    (momentum) argument, such that :math:`\partial_2 (\Phi_{2,1}(t))(q, p + \partial
    c(q)^T \lambda)` is constant with respect to :math:`\lambda`, and henceforth use the
    shorthand :math:`\partial_2 (\Phi_{2,1}(t))` to refer to this matrix.

    The full Newton update is

    .. math::

        \lambda' = \lambda - \partial r(\lambda)^{-1} r(\lambda)

    which requires evaluating :math:`\partial c` on each iteration and solving a linear
    system in the residual Jacobian :math:`\partial r(\lambda)`.

    The symmetric quasi-Newton iteration instead uses the approximation

    .. math::

        \partial c(\Phi_{2,1}(t)(q, p + \partial c(q)^T\lambda))
        \partial_2 (\Phi_{2,1}(t))
        \partial c(q)^T
        \approx
        \partial c(q)
        \partial_2 (\Phi_{2,1}(t))
        \partial c(q)^T

    with the corresponding update

    .. math::

        \lambda' =
        \lambda
        - (\partial c(q) \partial_2 (\Phi_{2,1}(t)) \partial c(q)^T)^{-1} r(\lambda)

    allowing a previously computed decomposition of the matrix

    .. math::

        \partial c(q) \partial_2 (\Phi_{2,1}(t)) \partial c(q)^T,

    to be used to solve the linear system in each iteration with no requirement to
    evaluate :math:`\partial c` (:code:`system.jacob_constr`) on each iteration.

    Args:
        state: Post :code:`h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before :code:`h2_flow` update
            which defines the co-tangent space to perform  projection in.
        time_step: Integrator time step used in :code:`h2_flow` update.
        system: Hamiltonian system defining :code:`h2_flow` and :code:`constr` functions
            used to define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until :code:`norm(constr(pos)) < constraint_tol` where :code:`pos`
            is the position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
            until :code:`norm(delta_pos) < position_tol` where :code:`delta_pos` is the
            change in the position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            :code:`norm(constr(pos)) > divergence_tol` on any iteration where
            :code:`pos` is the position at the current iteration and raises
            :py:exc:`mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            :py:exc:`mici.errors.ConvergenceError`.
        norm: Norm to use to test for convergence.

    Returns:
        Updated state object with position component satisfying constraint equation to
        within :code:`constraint_tol`, i.e.
        :code:`norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within
            :code:`max_iters` iterations, diverges or encounters a :py:exc:`ValueError`
            during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(
        state_prev,
        abs(time_step),
    )
    inv_jacob_constr_inner_product = system.jacob_constr_inner_product(
        jacob_constr_prev,
        dh2_flow_pos_dmom,
    ).inv
    for i in range(max_iters):
        try:
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (inv_jacob_constr_inner_product @ constr)
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                msg = (
                    f"Quasi-Newton solver diverged on iteration {i}. "
                    f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos):.1e}."
                )
                raise ConvergenceError(msg)
            if error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            msg = f"{type(e)} at iteration {i} of quasi-Newton solver ({e})."
            raise ConvergenceError(msg) from e
    msg = (
        f"Quasi-Newton solver did not converge with {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}."
    )
    raise ConvergenceError(msg)


def solve_projection_onto_manifold_newton(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: ConstrainedEuclideanMetricSystem,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    r"""Solve constraint equation using Newton's method.

    Requires re-evaluating both the constraint function :code:`system.constr` and
    constraint Jacobian :code:`system.jacob_constr` within the solver loop and
    computation of matrix decompositions on each iteration.

    Solves an equation of the form

    .. math::

        r(\lambda) = c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda)) = 0,

    for the vector of Lagrange multipliers :math:`\lambda` to project a point on to the
    manifold defined by the zero level set of a constraint function :math:`c`, with
    :math:`\Phi_{2,1}` the flow map for the :math:`h_2` Hamiltonian component for the
    system restricted to the position component output.

    The Jacobian of the residual function :math:`r` is

    .. math::

        \partial r(\lambda) =
        \partial c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda))
        \partial_2 (\Phi_{2,1}(t))(q, p + \partial c(q)^T \lambda)
        \partial c(q)^T

    where :math:`\partial_2 (\Phi_{2,1}(t))` is the Jacobian of the (position
    restricted) flow-map :math:`\Phi_{2,1}` with respect to its second (momentum)
    argument. We assume here that :math:`\Phi_{2,1}(t)` is linear in its second
    (momentum) argument, such that :math:`\partial_2 (\Phi_{2,1}(t))(q, p + \partial
    c(q)^T \lambda)` is constant with respect to :math:`\lambda`.

    The full Newton update is

    .. math::

        \lambda' = \lambda - \partial r(\lambda)^{-1} r(\lambda)

    which requires evaluating :math:`\partial c` on each iteration and solving a linear
    system in the residual Jacobian :math:`\partial r(\lambda)`.

    Args:
        state: Post :code:`h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before :code:`h2_flow` update
            which defines the co-tangent space to perform  projection in.
        time_step: Integrator time step used in :code:`h2_flow` update.
        system: Hamiltonian system defining :code:`h2_flow` and :code:`constr` functions
            used to define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until :code:`norm(constr(pos)) < constraint_tol` where :code:`pos`
            is the position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
            until :code:`norm(delta_pos) < position_tol` where :code:`delta_pos` is the
            change in the position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            :code:`norm(constr(pos)) > divergence_tol` on any iteration where
            :code:`pos` is the position at the current iteration and raises
            :py:exc:`mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            :py:exc:`mici.errors.ConvergenceError`.
        norm: Norm to use to test for convergence.

    Returns:
        Updated state object with position component satisfying constraint equation to
        within :code:`constraint_tol`, i.e.
        :code:`norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within
            :code:`max_iters` iterations, diverges or encounters a :py:exc:`ValueError`
            during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(
        state_prev,
        abs(time_step),
    )
    for i in range(max_iters):
        try:
            jacob_constr = system.jacob_constr(state)
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (
                system.jacob_constr_inner_product(
                    jacob_constr,
                    dh2_flow_pos_dmom,
                    jacob_constr_prev,
                ).inv
                @ constr
            )
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                msg = (
                    f"Newton solver diverged at iteration {i}. "
                    f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos):.1e}."
                )
                raise ConvergenceError(msg)
            if error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            msg = f"{type(e)} at iteration {i} of Newton solver ({e})."
            raise ConvergenceError(msg) from e
    msg = (
        f"Newton solver did not converge in {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}."
    )
    raise ConvergenceError(msg)


def solve_projection_onto_manifold_newton_with_line_search(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: ConstrainedEuclideanMetricSystem,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    max_line_search_iters: int = 10,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    r"""Solve constraint equation using Newton's method with backtracking line-search.

    Requires re-evaluating both the constraint function :code:`system.constr` and
    constraint Jacobian :code:`system.jacob_constr` within the solver loop and
    computation of matrix decompositions on each iteration.

    Solves an equation of the form

    .. math::

        r(\lambda) = c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda)) = 0,

    for the vector of Lagrange multipliers :math:`\lambda` to project a point on to the
    manifold defined by the zero level set of a constraint function :math:`c`, with
    :math:`\Phi_{2,1}` the flow map for the :math:`h_2` Hamiltonian component for the
    system restricted to the position component output.

    The Jacobian of the residual function :math:`r` is

    .. math::

        \partial r(\lambda) =
        \partial c(\Phi_{2,1}(t)(q, p + \partial c(q)^T \lambda))
        \partial_2 (\Phi_{2,1}(t))(q, p + \partial c(q)^T \lambda)
        \partial c(q)^T

    where :math:`\partial_2 (\Phi_{2,1}(t))` is the Jacobian of the (position
    restricted) flow-map :math:`\Phi_{2,1}` with respect to its second (momentum)
    argument. We assume here that :math:`\Phi_{2,1}(t)` is linear in its second
    (momentum) argument, such that :math:`\partial_2 (\Phi_{2,1}(t))(q, p + \partial
    c(q)^T \lambda)` is constant with respect to :math:`\lambda`.

    The scaled Newton update is

    .. math::

        \lambda'(\alpha) = \lambda - \alpha \partial r(\lambda)^{-1} r(\lambda)

    which requires evaluating :math:`\partial c` on each iteration and solving a linear
    system in the residual Jacobian :math:`\partial r(\lambda)`.

    The step size :math:`\alpha \in [0, 1]` is initialised at :math:`\alpha = 1` with a
    backtracking line search performed by multiplying :math:`\alpha` by 0.5 until
    :math:`|r(\lambda'(\alpha))| < |r(\lambda)`.

    Args:
        state: Post :code:`h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before :code:`h2_flow` update
            which defines the co-tangent space to perform  projection in.
        time_step: Integrator time step used in :code:`h2_flow` update.
        system: Hamiltonian system defining :code:`h2_flow` and :code:`constr` functions
            used to define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until :code:`norm(constr(pos)) < constraint_tol` where :code:`pos`
            is the position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
            until :code:`norm(delta_pos) < position_tol` where :code:`delta_pos` is the
            change in the position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            :code:`norm(constr(pos)) > divergence_tol` on any iteration where
            :code:`pos` is the position at the current iteration and raises
            :py:exc:`mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            :py:exc:`mici.errors.ConvergenceError`.
        max_line_search_iters: Maximum number of 'inner' line search iterations to
            perform to try to find step size along search direction which decreases
            residual norm.
        norm: Norm to use to test for convergence.

    Returns:
        Updated state object with position component satisfying constraint equation to
        within :code:`constraint_tol`, i.e.
        :code:`norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within
            :code:`max_iters` iterations, diverges or encounters a :py:exc:`ValueError`
            during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(
        state_prev,
        abs(time_step),
    )
    # Initialize with dummy values to avoid undefined name linter errors
    delta_pos, step_size = None, None
    for i in range(max_iters):
        try:
            jacob_constr = system.jacob_constr(state)
            constr = system.constr(state)
            error = norm(constr)
            if i > 0 and (error > divergence_tol or np.isnan(error)):
                msg = (
                    f"Newton solver diverged at iteration {i}. "
                    f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos):.1e}."
                )
                raise ConvergenceError(msg)
            if error < constraint_tol and (
                i == 0 or norm(step_size * delta_pos) < position_tol
            ):
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            delta_mu = jacob_constr_prev.T @ (
                system.jacob_constr_inner_product(
                    jacob_constr,
                    dh2_flow_pos_dmom,
                    jacob_constr_prev,
                ).inv
                @ constr
            )
            delta_pos = -dh2_flow_pos_dmom @ delta_mu
            pos_curr = state.pos.copy()
            step_size = 1.0
            for _ in range(max_line_search_iters):
                state.pos = pos_curr + step_size * delta_pos
                new_error = norm(system.constr(state))
                if new_error < error:
                    break
                step_size *= 0.5
            mu += step_size * delta_mu
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            msg = f"{type(e)} at iteration {i} of Newton solver ({e})."
            raise ConvergenceError(msg) from e
    msg = (
        f"Newton solver did not converge in {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(step_size * delta_pos)}."
    )
    raise ConvergenceError(msg)
