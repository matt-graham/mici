"""Solvers for non-linear systems of equations for implicit integrators."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from mici.errors import ConvergenceError, LinAlgError
import numpy as np

if TYPE_CHECKING:
    from mici.states import ChainState
    from mici.systems import System
    from mici.types import ScalarFunction, ArrayFunction, ArrayLike


def euclidean_norm(vct):
    """Calculate the Euclidean (L-2) norm of a vector."""
    return (vct ** 2).sum() ** 0.5


def maximum_norm(vct):
    """Calculate the maximum (L-infinity) norm of a vector."""
    return (abs(vct)).max()


class FixedPointSolver(Protocol):
    """Solver for fixed point equation, returning `x` such that `func(x) = x`."""

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
    """Solve fixed point equation `func(x) = x` using direct iteration.

    Args:
        func: Function to find fixed point of.
        x0: Initial state (function argument).
        convergence_tol: Convergence tolerance - solver successfully terminates when
            `norm(func(x) - x) < convergence_tol`.
        divergence_tol: Divergence tolerance - solver aborts if
            `norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters: Maximum number of iterations before raising exception.
        norm: Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with `norm(func(x) - x) < convergence_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within `max_iters`
            iterations, diverges or encounters a `ValueError` during the iteration.
    """
    for i in range(max_iters):
        try:
            x = func(x0)
            error = norm(x - x0)
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f"Fixed point iteration diverged on iteration {i}. "
                    f"Last error={error:.1e}."
                )
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of fixed point solver ({e})."
            )
    raise ConvergenceError(
        f"Fixed point iteration did not converge. Last error={error:.1e}."
    )


def solve_fixed_point_steffensen(
    func: ArrayFunction,
    x0: ArrayLike,
    convergence_tol: float = 1e-9,
    divergence_tol: float = 1e10,
    max_iters: int = 100,
    norm: ScalarFunction = maximum_norm,
) -> ArrayLike:
    """Solve fixed point equation `func(x) = x` using Steffensen's method.

    Steffennsen's method [1] achieves quadratic convergence but at the cost of two
    function evaluations per iteration so for functions where convergence is achieved in
    a small number of iterations, direct iteration may be cheaper.

    [1] : https://en.wikipedia.org/wiki/Steffensen%27s_method

    Args:
        func: Function to find fixed point of.
        x0: Initial state (function argument).
        convergence_tol: Convergence tolerance - solver successfully terminates
            when `norm(func(x) - x) < convergence_tol`.
        divergence_tol: Divergence tolerance - solver aborts if
            `norm(func(x) - x) > divergence_tol` on any iteration.
        max_iters: Maximum number of iterations before raising exception.
        norm: Norm to use to assess convergence.

    Returns:
        Solution to fixed point equation with `norm(func(x) - x) < convergence_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within `max_iters`
            iterations, diverges or encounters a `ValueError` during the iteration.
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
                raise ConvergenceError(
                    f"Fixed point iteration diverged on iteration {i}. "
                    f"Last error={error:.1e}."
                )
            if error < convergence_tol:
                return x
            x0 = x
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of fixed point solver ({e})."
            )
    raise ConvergenceError(
        f"Fixed point iteration did not converge. Last error={error:.1e}."
    )


class ProjectionSolver(Protocol):
    """Solver for projection on to manifold step in constrained integrator.

    Solves an equation of the form

    .. code::

        r(λ) = c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) = 0

    for the vector of Lagrange multipliers `λ` to project a point on to the manifold
    defined by the zero level set of a constraint function `c` (`system.constr`), with
    `Φ₂,₁` the position component of a flow map for the `h₂` Hamiltonian component for
    the system (i.e. `system.h2_flow`).
    """

    def __call__(
        self,
        state: ChainState,
        state_prev: ChainState,
        time_step: float,
        system: System,
        **kwargs,
    ) -> ChainState:
        """Solve for projection on to manifold step.

        Args:
            state: Current chain state after unconstrained step.
            state_prev: Previous chain state on manifold.
            time_step: Integrator time step for unconstrained step.
            system: Hamiltonian system dynamics are being simulated for.

        Returns:
            Chain state after projection on to manifold.
        """
        ...


def solve_projection_onto_manifold_quasi_newton(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: System,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    """Solve constraint equation using symmetric quasi-Newton method.

    Only requires re-evaluating the constraint function `system.constr` within the
    solver loop and no recomputation of matrix decompositions on each iteration.

    Solves an equation of the form

    .. code::

        r(λ) = c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) = 0

    for the vector of Lagrange multipliers `λ` to project a point on to the manifold
    defined by the zero level set of a constraint function `c` (`system.constr`), with
    `Φ₂,₁` the position component of a flow map for the `h₂` Hamiltonian component for
    the system (i.e. `system.h2_flow`).

    The Jacobian of the residual function `r` is

    .. code::

        ∂r(λ) = ∂c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) ∂₂Φ₂,₁ ∂c(q)ᵀ

    where `∂₂Φ₂,₁` is the Jacobian of the (linear) flow-map `Φ₂,₁` with respect to the
    second (momentum argument), such that the full Newton update is

    .. code::

        λ_(α) = λ - ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` (`system.jacob_constr`) on each iteration and solving
    a linear system in the residual Jacobian `∂r(λ)`.

    The symmetric quasi-Newton iteration instead uses the approximation

    .. code::

        ∂c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) ∂₂Φ₂,₁ ∂c(q)ᵀ ≈ ∂c(q) ∂₂Φ₂,₁ ∂c(q)ᵀ

    with the corresponding update

    .. code::

        λ = λ - (∂c(q) ∂₂Φ₂,₁ ∂c(q)ᵀ)⁻¹ r(λ)

    allowing a previously computed decomposition of the matrix `∂c(q) ∂₂Φ₂,₁ ∂c(q)ᵀ` to
    be used to solve the linear system in each iteration with no requirement to evaluate
    `∂c` (`system.jacob_constr`) on each iteration.

    Args:
        state: Post `h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before `h2_flow` update which
            defines the co-tangent space to perform  projection in.
        time_step: Integrator time step used in `h2_flow` update.
        system: Hamiltonian system defining `h2_flow` and `constr` functions used to
            define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until `norm(constr(pos)) < constraint_tol` where `pos` is the
            position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
            until `norm(delt_pos) < position_tol` where `delta_pos` is the change in the
            position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos` is the
            position at the current iteration and raises `mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            `mici.errors.ConvergenceError`.
        norm: Norm to use to test for convergence.

    Returns:
        Updated state object with position component satisfying constraint equation to
        within `constraint_tol`, i.e. `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within `max_iters`
            iterations, diverges or encounters a `ValueError` during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(time_step))
    inv_jacob_constr_inner_product = system.jacob_constr_inner_product(
        jacob_constr_prev, dh2_flow_pos_dmom
    ).inv
    for i in range(max_iters):
        try:
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (inv_jacob_constr_inner_product @ constr)
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f"Quasi-Newton solver diverged on iteration {i}. "
                    f"Last |constr|={error:.1e}, "
                    f"|delta_pos|={norm(delta_pos):.1e}."
                )
            elif error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of quasi-Newton solver ({e})."
            )
    raise ConvergenceError(
        f"Quasi-Newton solver did not converge with {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}."
    )


def solve_projection_onto_manifold_newton(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: System,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    """Solve constraint equation using Newton's method.

    Requires re-evaluating both the constraint function `system.constr` and constraint
    Jacobian `system.jacob_constr` within the solver loop and computation of matrix
    decompositions on each iteration.

    Solves an equation of the form

    .. code::

        r(λ) = c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) = 0

    for the vector of Lagrange multipliers `λ` to project a point on to the manifold
    defined by the zero level set of a constraint function `c` (`system.constr`), with
    `Φ₂,₁` the position component of a flow map for the `h₂` Hamiltonian component for
    the system (i.e. `system.h2_flow`).

    The Jacobian of the residual function `r` is

    .. code::

        ∂r(λ) = ∂c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) ∂₂Φ₂,₁ ∂c(q)ᵀ

    where `∂₂Φ₂,₁` is the Jacobian of the (linear) flow-map `Φ₂,₁` with respect to the
    second (momentum argument), such that the Newton update is

    .. code::

        λ = λ - ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` (`system.jacob_constr`) on each iteration and solving
    a linear system in the residual Jacobian `∂r(λ)`.

    Args:
        state: Post `h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before `h2_flow` update which
            defines the co-tangent space to perform projection in.
        time_step: Integrator time step used in `h2_flow` update.
        system: Hamiltonian system defining `h2_flow` and `constr` functions used to
            define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until `norm(constr(pos)) < constraint_tol` where `pos` is the
            position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
            until `norm(delt_pos) < position_tol` where `delta_pos` is the change in the
            position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos` is the
            position at the current iteration and raises `mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            `mici.errors.ConvergenceError`.
        norm: Norm to use to test for convergence.

    Returns:
        Updated `state` object with position component satisfying constraint equation to
        within `constraint_tol`, i.e. `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within `max_iters`
            iterations, diverges or encounters a `ValueError` during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(time_step))
    for i in range(max_iters):
        try:
            jacob_constr = system.jacob_constr(state)
            constr = system.constr(state)
            error = norm(constr)
            delta_mu = jacob_constr_prev.T @ (
                system.jacob_constr_inner_product(
                    jacob_constr, dh2_flow_pos_dmom, jacob_constr_prev
                ).inv
                @ constr
            )
            delta_pos = dh2_flow_pos_dmom @ delta_mu
            if error > divergence_tol or np.isnan(error):
                raise ConvergenceError(
                    f"Newton solver diverged at iteration {i}. "
                    f"Last |constr|={error:.1e}, "
                    f"|delta_pos|={norm(delta_pos):.1e}."
                )
            if error < constraint_tol and norm(delta_pos) < position_tol:
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            mu += delta_mu
            state.pos -= delta_pos
        except (ValueError, LinAlgError) as e:
            # Make robust to errors in intermediate linear algebra ops
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of Newton solver ({e})."
            )
    raise ConvergenceError(
        f"Newton solver did not converge in {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(delta_pos)}."
    )


def solve_projection_onto_manifold_newton_with_line_search(
    state: ChainState,
    state_prev: ChainState,
    time_step: float,
    system: System,
    constraint_tol: float = 1e-9,
    position_tol: float = 1e-8,
    divergence_tol: float = 1e10,
    max_iters: int = 50,
    max_line_search_iters: int = 10,
    norm: ScalarFunction = maximum_norm,
) -> ChainState:
    """Solve constraint equation using Newton's method with backtracking line-search.

    Requires re-evaluating both the constraint function `system.constr` and constraint
    Jacobian `system.jacob_constr` within the solver loop and computation of matrix
    decompositions on each iteration.

    Solves an equation of the form

    .. code::

        r(λ) = c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) = 0

    for the vector of Lagrange multipliers `λ` to project a point on to the manifold
    defined by the zero level set of a constraint function `c` (`system.constr`), with
    `Φ₂,₁` the position component of a flow map for the `h₂` Hamiltonian component for
    the system (i.e. `system.h2_flow`).

    The Jacobian of the residual function `r` is

    .. code::

        ∂r(λ) = ∂c(Φ₂,₁(q, p + ∂c(q)ᵀλ)) ∂₂Φ₂,₁ ∂c(q)ᵀ

    where `∂₂Φ₂,₁` is the Jacobian of the (linear) flow-map `Φ₂,₁` with respect to the
    second (momentum argument), such that the scaled Newton update is

    .. code::

        λ_(α) = λ - α * ∂r(λ)⁻¹ r(λ)

    which requires evaluating `∂c` (`system.jacob_constr`) on each iteration and solving
    a linear system in the residual Jacobian `∂r(λ)`.

    The step size `α ∈ [0, 1]` is initialised at `α = 1` with a backtracking line search
    performed by multiplying `α` by 0.5 until `|r(λ_(α))| < |r(λ)`.

    Args:
        state: Post `h2_flow` update state to project.
        state_prev: Previous state in co-tangent bundle before `h2_flow` update which
            defines the co-tangent space to perform projection in.
        time_step: Integrator time step used in `h2_flow` update.
        system: Hamiltonian system defining `h2_flow` and `constr` functions used to
            define constraint equation to solve.
        constraint_tol: Convergence tolerance in constraint space. Iteration will
            continue until `norm(constr(pos)) < constraint_tol` where `pos` is the
            position at the current iteration.
        position_tol: Convergence tolerance in position space. Iteration will continue
           until `norm(delt_pos) < position_tol` where `delta_pos` is the change in the
           position in the current iteration.
        divergence_tol: Divergence tolerance - solver aborts if
            `norm(constr(pos)) > divergence_tol` on any iteration where `pos` is the
            position at the current iteration and raises `mici.errors.ConvergenceError`.
        max_iters: Maximum number of iterations to perform before aborting and raising
            `mici.errors.ConvergenceError`.
        max_line_search_iters: Maximum number of 'inner' line search iterations to
            perform to try to find step size along search direction which decreases
            residual norm.
        norm: Norm to use to test for convergence.

    Returns:
        Updated state object with position component satisfying constraint equation to
        within `constraint_tol`, i.e. `norm(system.constr(state.pos)) < constraint_tol`.

    Raises:
        mici.errors.ConvergenceError: If solver does not converge within `max_iters`
            iterations, diverges or encounters a `ValueError` during the iteration.
    """
    mu = np.zeros_like(state.pos)
    jacob_constr_prev = system.jacob_constr(state_prev)
    # Use absolute value of dt and adjust for sign of dt in mom update below
    dh2_flow_pos_dmom, dh2_flow_mom_dmom = system.dh2_flow_dmom(abs(time_step))
    for i in range(max_iters):
        try:
            jacob_constr = system.jacob_constr(state)
            constr = system.constr(state)
            error = norm(constr)
            if i > 0 and (error > divergence_tol or np.isnan(error)):
                raise ConvergenceError(
                    f"Newton solver diverged at iteration {i}. "
                    f"Last |constr|={error:.1e}, "
                    f"|delta_pos|={norm(delta_pos):.1e}."
                )
            if error < constraint_tol and (
                i == 0 or norm(step_size * delta_pos) < position_tol
            ):
                state.mom -= np.sign(time_step) * dh2_flow_mom_dmom @ mu
                return state
            delta_mu = jacob_constr_prev.T @ (
                system.jacob_constr_inner_product(
                    jacob_constr, dh2_flow_pos_dmom, jacob_constr_prev
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
            raise ConvergenceError(
                f"{type(e)} at iteration {i} of Newton solver ({e})."
            )
    raise ConvergenceError(
        f"Newton solver did not converge in {max_iters} iterations. "
        f"Last |constr|={error:.1e}, |delta_pos|={norm(step_size * delta_pos)}."
    )
