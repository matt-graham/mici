"""Symplectic integrators for simulation of Hamiltonian dynamics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from mici.errors import NonReversibleStepError, AdaptationError
from mici.solvers import (
    maximum_norm,
    solve_fixed_point_direct,
    solve_projection_onto_manifold_quasi_newton,
    FixedPointSolver,
    ProjectionSolver,
)

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence
    from mici.states import ChainState
    from mici.systems import System, TractableFlowSystem
    from mici.types import NormFunction


class Integrator(ABC):
    r"""Base class for integrators for simulating Hamiltonian dynamics.

    For a Hamiltonian function :math:`h` with position variables :math:`q` and momentum
    variables :math:`p`, the canonical Hamiltonian dynamic is defined by the ordinary
    differential equation system

    .. math::

        \dot{q} = \nabla_2 h(q, p),  \qquad \dot{p} = -\nabla_1 h(q, p)

    with the flow map corresponding to the solution of the corresponding initial value
    problem a time-reversible and symplectic (and by consequence volume-preserving) map.

    Derived classes implement a :py:meth:`step` method which approximates the flow-map
    over some small time interval, while conserving the properties of being
    time-reversible and symplectic, with composition of this integrator step method
    allowing simulation of time-discretised trajectories of the Hamiltonian dynamics.
    """

    def __init__(self, system: System, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of.
            step_size: Integrator time step. If set to `None` it is assumed that a step
                size adapter will be used to set the step size before calling the `step`
                method.
        """
        self.system = system
        self.step_size = step_size

    def step(self, state: ChainState) -> ChainState:
        """Perform a single integrator step from a supplied state.

        Args:
            state: System state to perform integrator step from.

        Returns:
            New object corresponding to stepped state.
        """
        if self.step_size is None:
            raise AdaptationError(
                "Integrator `step_size` is `None`. This value should only be used if a "
                "step size adapter is being used to set the step size."
            )
        state = state.copy()
        self._step(state, state.dir * self.step_size)
        return state

    @abstractmethod
    def _step(self, state: ChainState, time_step: float):
        """Implementation of single integrator step.

        Args:
            state: System state to perform integrator step from. Updated in place.
            time_step: Integrator time step. May be positive or negative.
        """


class TractableFlowIntegrator(Integrator):
    """Base class for integrators for Hamiltonian systems with tractable component flows

    The Hamiltonian function is assumed to be expressible as the sum of two analytically
    tractable components for which the corresponding Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function :math:`h` takes
    the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p)

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    and :math:`h_1` and :math:`h_2` are Hamiltonian component functions for which the
    exact flows can be computed.
    """

    def __init__(self, system: TractableFlowSystem, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of with tractable
                Hamiltonian component flows.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
        """
        if not hasattr(system, "h1_flow") or not hasattr(system, "h2_flow"):
            raise ValueError(
                f"{type(self)} can only be used for systems with explicit `h1_flow` and"
                f" `h2_flow` Hamiltonian component flow maps. For systems in which only"
                f" `h1_flow` is available the `ImplicitLeapfrogIntegrator` class may be"
                f" used instead and for systems in which neither `h1_flow` or `h2_flow`"
                f" is available the `ImplicitMidpointIntegrator` class may be used."
            )
        super().__init__(system, step_size)


class LeapfrogIntegrator(TractableFlowIntegrator):
    """Leapfrog integrator for Hamiltonian systems with tractable component flows.

    For separable Hamiltonians of the

    .. math::

        h(q, p) = h_1(q) + h_2(p)

    where :math:`h_1` is the potential energy and :math:`h_2` is the kinetic energy,
    this integrator corresponds to the classic (position) Störmer-Verlet method.

    The integrator can also be applied to the more general Hamiltonian splitting

    .. math::

        h(q, p) = h_1(q) + h_2(q, p)

    providing the flows for :math:`h_1` and :math:`h_2` are both tractable.

    For more details see Sections 2.6 and 4.2.2 in Leimkuhler and Reich (2004).

    References:

      1. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
    """

    def _step(self, state: ChainState, time_step: float):
        self.system.h1_flow(state, 0.5 * time_step)
        self.system.h2_flow(state, time_step)
        self.system.h1_flow(state, 0.5 * time_step)


class SymmetricCompositionIntegrator(TractableFlowIntegrator):
    r"""Symmetric composition integrator for Hamiltonians with tractable component flows

    The Hamiltonian function is assumed to be expressible as the sum of two analytically
    tractable components for which the corresponding Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function :math:`h` takes
    the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p)

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    and :math:`h_1` and :math:`h_2` are Hamiltonian component functions for which the
    exact flows, respectively :math:`\Phi_1` and :math:`\Phi_2`, can be computed. An
    alternating composition can then be formed as

    .. math::

        \Psi(t) =
        A(a_S t) \circ B(b_S t) \circ \dots \circ A(a_1 t) \circ B(b_1 t) \circ A(a_0 t)

    where :math:`A = \Phi_1` and :math:`B = \Phi_2` or :math:`A = \Phi_2` and :math:`B =
    \Phi_1`, and :math:`(a_0,\dots, a_S)` and :math:`(b_1, \dots, b_S)` are a set of
    coefficients to be determined and :math:`S` is the number of stages in the
    composition.

    To ensure a consistency (i.e. the integrator is at least order one) we require that

    .. math::

        \sum_{s=0}^S a_s = \sum_{s=1}^S b_s = 1.

    For symmetric compositions we restrict that

    .. math::

        a_{S-m} = a_m, \quad b_{S+1-m} = b_s

    with symmetric consistent methods of at least order two.

    The combination of the symmetry and consistency requirements mean that a
    :math:`S`-stage symmetric composition method can be described by :math:`S - 1`
    'free' coefficients

    .. math::

        (a_0, b_1, a_1, \dots, a_K, b_K)

    with :math:`K = (S - 1) / 2` if :math:`S` is odd or

    .. math::

        (a_0, b_1, a_1, \dots, a_K)

    with :math:`K = (S - 2) / 2` if :math:`S` is even.

    The Störmer-Verlet 'leapfrog' integrator is a special case corresponding to the
    unique (symmetric and consistent) 1-stage integrator.

    For more details see Section 6.2 in Leimkuhler and Reich (2004).

    References:

      1. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
    """

    def __init__(
        self,
        system: TractableFlowSystem,
        free_coefficients: Sequence[float],
        step_size: Optional[float] = None,
        initial_h1_flow_step: bool = True,
    ):
        r"""
        Args:
            system: Hamiltonian system to integrate the dynamics of with tractable
                Hamiltonian component flows.
            free_coefficients: Sequence of :math:`S - 1` scalar values, where :math:`S`
                is the number of stages in the symmetric composition, specifying the
                free coefficients :math:`(a_0, b_1, a_1, \dots, a_K, b_K)` with :math:`K
                = (S - 1) / 2` if :math:`S` is odd or :math:`(a_0, b_1, a_1, \dots,
                a_K)` with :math:`k = (S - 2) / 2` if `S` is even.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
            initial_h1_flow_step: Whether the initial :math:`A` flow in the composition
                should correspond to the flow of the `h_1` Hamiltonian component
                (:code:`True`) or to the flow of the :math:`h_2` component
                (:code:`False`).
        """
        super().__init__(system, step_size)
        self.initial_h1_flow_step = initial_h1_flow_step
        n_free_coefficients = len(free_coefficients)
        coefficients = list(free_coefficients)
        coefficients.append(
            0.5 - sum(free_coefficients[(n_free_coefficients) % 2 :: 2])
        )
        coefficients.append(
            1 - 2 * sum(free_coefficients[(n_free_coefficients + 1) % 2 :: 2])
        )
        self.coefficients = coefficients + coefficients[-2::-1]
        flow_a = system.h1_flow if initial_h1_flow_step else system.h2_flow
        flow_b = system.h2_flow if initial_h1_flow_step else system.h1_flow
        self.flows = [flow_a, flow_b] * (n_free_coefficients + 1) + [flow_a]

    def _step(self, state, time_step):
        for coefficient, flow in zip(self.coefficients, self.flows):
            flow(state, coefficient * time_step)


class BCSSTwoStageIntegrator(SymmetricCompositionIntegrator):
    """Two-stage symmetric composition integrator due to Blanes, Casas & Sanz-Serna.

    Described in equation (6.4) in Blanes, Casas, Sanz-Serna (2014).

    References:

      1. Blanes, S., Casas, F., & Sanz-Serna, J. M. (2014).
         Numerical integrators for the Hybrid Monte Carlo method.
         SIAM Journal on Scientific Computing, 36(4), A1556-A1580.
    """

    def __init__(self, system: TractableFlowSystem, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of with tractable
                Hamiltonian component flows.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
        """
        a_0 = (3 - 3 ** 0.5) / 6
        super().__init__(system, (a_0,), step_size, True)


class BCSSThreeStageIntegrator(SymmetricCompositionIntegrator):
    """Three-stage symmetric composition integrator due to Blanes, Casas & Sanz-Serna.

    Described in equation (6.7) in Blanes, Casas, Sanz-Serna (2014).

    References:

      1. Blanes, S., Casas, F., & Sanz-Serna, J. M. (2014).
         Numerical integrators for the Hybrid Monte Carlo method.
         SIAM Journal on Scientific Computing, 36(4), A1556-A1580.
    """

    def __init__(self, system: TractableFlowSystem, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of with tractable
                Hamiltonian component flows.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
        """
        a_0 = 0.11888010966548
        b_1 = 0.29619504261126
        super().__init__(system, (a_0, b_1), step_size, True)


class BCSSFourStageIntegrator(SymmetricCompositionIntegrator):
    """Four-stage symmetric composition integrator due to Blanes, Casas & Sanz-Serna.

    Described in equation (6.8) in Blanes, Casas, Sanz-Serna (2014).

    References:

      1. Blanes, S., Casas, F., & Sanz-Serna, J. M. (2014).
         Numerical integrators for the Hybrid Monte Carlo method.
         SIAM Journal on Scientific Computing, 36(4), A1556-A1580.
    """

    def __init__(self, system: TractableFlowSystem, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of with tractable
                Hamiltonian component flows.
            step_size: Integrator time step. If set to :code:`None` it is assumed that
                a step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
        """
        a_0 = 0.071353913450279725904
        b_1 = 0.191667800000000000000
        a_1 = 0.268548791161230105820
        super().__init__(system, (a_0, b_1, a_1), step_size, True)


class ImplicitLeapfrogIntegrator(Integrator):
    """Implicit leapfrog integrator for Hamiltonians with a non-separable component.

    Also known as the generalised leapfrog method.

    The Hamiltonian function :math:`h` is assumed to take the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p)

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    :math:`h_1` is a Hamiltonian component function for which the exact flow can be
    computed and :math:`h_2` is a Hamiltonian component function of the position and
    momentum variables, which may be non-separable and for which exact simulation of the
    correspond Hamiltonian flow may not be possible.

    A pair of implicit component updates are used to approximate the flow due to the
    :math:`h_2` Hamiltonian component, with a fixed-point iteration used to solve the
    non-linear system of equations.

    The resulting implicit integrator is a symmetric second-order method corresponding
    to a symplectic partitioned Runge-Kutta method. See Section 6.3.2 in Leimkuhler and
    Reich (2004) for more details.

    References:

      1. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
    """

    def __init__(
        self,
        system: System,
        step_size: Optional[float] = None,
        reverse_check_tol: float = 2e-8,
        reverse_check_norm: NormFunction = maximum_norm,
        fixed_point_solver: FixedPointSolver = solve_fixed_point_direct,
        fixed_point_solver_kwargs: dict[str, Any] = None,
    ):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of.
            step_size: Integrator time step. If set to `None` it is assumed that a step
                size adapter will be used to set the step size before calling the `step`
                method.
            reverse_check_tol: Tolerance for check of reversibility of implicit
                sub-steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                `reverse_check_norm` argument) of `reverse_check_tol` of the original
                state position component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            fixed_point_solver:  Function which given a function `func` and initial
                guess `x0` iteratively solves the fixed point equation `func(x) = x`
                initialising the iteration with `x0` and returning an array
                corresponding to the solution if the iteration converges or raising a
                `mici.errors.ConvergenceError` otherwise. Defaults to
                `mici.solvers.solve_fixed_point_direct`.
            fixed_point_solver_kwargs: Dictionary of any keyword arguments to
                `fixed_point_solver`.
        """
        super().__init__(system, step_size)
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm
        self.fixed_point_solver = fixed_point_solver
        if fixed_point_solver_kwargs is None:
            fixed_point_solver_kwargs = {}
        self.fixed_point_solver_kwargs = fixed_point_solver_kwargs

    def _solve_fixed_point(
        self, fixed_point_func: Callable[[ArrayLike], ArrayLike], x_init: ArrayLike
    ) -> ArrayLike:
        return self.fixed_point_solver(
            fixed_point_func, x_init, **self.fixed_point_solver_kwargs
        )

    def _step_a(self, state: ChainState, time_step: float):
        self.system.h1_flow(state, time_step)

    def _step_b_fwd(self, state: ChainState, time_step: float):
        def fixed_point_func(mom):
            state.mom = mom
            return mom_init - time_step * self.system.dh2_dpos(state)

        mom_init = state.mom
        state.mom = self._solve_fixed_point(fixed_point_func, mom_init)

    def _step_b_adj(self, state: ChainState, time_step: float):
        mom_init = state.mom.copy()
        state.mom -= time_step * self.system.dh2_dpos(state)
        state_back = state.copy()
        self._step_b_fwd(state_back, -time_step)
        rev_diff = self.reverse_check_norm(state_back.mom - mom_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f"Non-reversible step. Distance between initial and "
                f"forward-backward integrated momentums = {rev_diff:.1e}."
            )

    def _step_c_fwd(self, state: ChainState, time_step: float):
        pos_init = state.pos.copy()
        state.pos += time_step * self.system.dh2_dmom(state)
        state_back = state.copy()
        self._step_c_adj(state_back, -time_step)
        rev_diff = self.reverse_check_norm(state_back.pos - pos_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f"Non-reversible step. Distance between initial and "
                f"forward-backward integrated positions = {rev_diff:.1e}."
            )

    def _step_c_adj(self, state: ChainState, time_step: float):
        def fixed_point_func(pos):
            state.pos = pos
            return pos_init + time_step * self.system.dh2_dmom(state)

        pos_init = state.pos
        state.pos = self._solve_fixed_point(fixed_point_func, pos_init)

    def _step(self, state: ChainState, time_step: float):
        self._step_a(state, time_step)
        self._step_b_fwd(state, time_step)
        self._step_c_fwd(state, time_step)
        self._step_c_adj(state, time_step)
        self._step_b_adj(state, time_step)
        self._step_a(state, time_step)


class ImplicitMidpointIntegrator(Integrator):
    """Implicit midpoint integrator for general Hamiltonians.

    The Hamiltonian function may be a general (non-separable) function of both the
    position variables `q` and momentum variables `p`.

    The flow is approximated with the composition of an implicit Euler half-step with
    ane explicit Euler half-step.

    The resulting implicit integrator is a second-order method corresponding to a
    symplectic one-stage  Runge-Kutta method. See Sections 4.1 and 6.3.1 in Leimkuhler
    and Reich (2004) for more details.

    References:

    Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
    Cambridge University Press.
    """

    def __init__(
        self,
        system: System,
        step_size: Optional[float] = None,
        reverse_check_tol: float = 2e-8,
        reverse_check_norm: NormFunction = maximum_norm,
        fixed_point_solver: FixedPointSolver = solve_fixed_point_direct,
        fixed_point_solver_kwargs: dict[str, Any] = None,
    ):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of.
            step_size: Integrator time step. If set to `None` it is assumed that a step
                size adapter will be used to set the step size before calling the `step`
                method.
            reverse_check_tol: Tolerance for check of reversibility of implicit
                sub-steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                `reverse_check_norm` argument) of `reverse_check_tol` of the original
                state position component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            fixed_point_solver:  Function which given a function `func` and initial
                guess `x0` iteratively solves the fixed point equation `func(x) = x`
                initialising the iteration with `x0` and returning an array
                corresponding to the solution if the iteration converges or raising a
                `mici.errors.ConvergenceError` otherwise. Defaults to
                `mici.solvers.solve_fixed_point_direct`.
            fixed_point_solver_kwargs: Dictionary of any keyword arguments to
                `fixed_point_solver`.
        """
        super().__init__(system, step_size)
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = maximum_norm
        self.fixed_point_solver = fixed_point_solver
        if fixed_point_solver_kwargs is None:
            fixed_point_solver_kwargs = {}
        self.fixed_point_solver_kwargs = fixed_point_solver_kwargs

    def _solve_fixed_point(
        self, fixed_point_func: Callable[[ArrayLike], ArrayLike], x_init: ArrayLike
    ) -> ArrayLike:
        return self.fixed_point_solver(
            fixed_point_func, x_init, **self.fixed_point_solver_kwargs
        )

    def _step_a_fwd(self, state: ChainState, time_step: float):
        pos_mom_init = np.concatenate([state.pos, state.mom])

        def fixed_point_func(pos_mom):
            state.pos, state.mom = np.split(pos_mom, 2)
            return pos_mom_init + np.concatenate(
                [
                    time_step * self.system.dh_dmom(state),
                    -time_step * self.system.dh_dpos(state),
                ]
            )

        state.pos, state.mom = np.split(
            self._solve_fixed_point(fixed_point_func, pos_mom_init), 2
        )

    def _step_a_adj(self, state: ChainState, time_step: float):
        state_prev = state.copy()
        state.pos += time_step * self.system.dh_dmom(state_prev)
        state.mom -= time_step * self.system.dh_dpos(state_prev)
        state_back = state.copy()
        self._step_a_fwd(state_back, -time_step)
        rev_diff = self.reverse_check_norm(
            np.concatenate(
                [state_back.pos - state_prev.pos, state_back.mom - state_prev.mom]
            )
        )
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f"Non-reversible step. Distance between initial and "
                f"forward-backward integrated (pos, mom) pairs = {rev_diff:.1e}."
            )

    def _step(self, state: ChainState, time_step: float):
        self._step_a_fwd(state, time_step / 2)
        self._step_a_adj(state, time_step / 2)


class ConstrainedLeapfrogIntegrator(TractableFlowIntegrator):
    """Leapfrog integrator for constrained Hamiltonian systems.

    The Hamiltonian function is assumed to be expressible as the sum of two components
    for which the corresponding (unconstrained) Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function `h` takes the
    form

        h(q, p) = h₁(q) + h₂(q, p)

    where `q` and `p` are the position and momentum variables respectively, and `h₁` and
    `h₂` Hamiltonian component functions for which the exact flows, respectively `Φ₁`
    and `Φ₂`, can be computed.

    The system is assumed to be additionally subject to a set of holonomic constraints
    on the position component of the state i.e. that all valid states must satisfy

        c(q) = 0

    for some differentiable and surjective vector constraint function `c` and the set of
    positions satisfying the constraints implicitly defining a manifold the dynamics
    remain confined to.

    The constraints are enforced by introducing a set of Lagrange multipliers `λ` of
    dimension equal to number of constraints, and defining a 'constrained' Hamiltonian

        h̅(q, p) = h₁(q) + h₂(q, p) + c(q)ᵀλ  subject to c(q) = 0

    with corresponding dynamics described by the system of differential algebraic
    equations

        q̇ = ∇₂h(q, p),
        ṗ = -∇₁h(q, p) - ∂c(q)ᵀλ,
        c(q) = 0.

    The dynamics implicitly define a set of constraints on the momentum variables,
    differentiating the constraint equation with respect to time giving that

        ∂c(q) ∇₂h(q, p) = 0

    The set of momentum variables satisfying the above for given position variables is
    termed the cotangent space of the manifold (at a position), and the set of
    position-momentum pairs for which the position is on the constraint manifold and the
    momentum in the corresponding cotangent space is termed the cotangent bundle.

    To define a second-order symmetric integrator which exactly (up to floating point
    error) preserves these constraints, forming a symplectic map on the cotangent
    bundle, we follow the approach of Reich (1996).

    We first define a map `Π` parametrised by a vector of Lagrange multipliers `λ`

        Π(λ)(q, p) = (q, p + ∂c(q)ᵀλ)

    with `λ` allowed to be an implicitly defined function of `q` and `p`.

    We then define a map `Ψ₁` in terms of the `h₁` flow map `Φ₁` as

        Ψ₁(t) = Π(λ) ∘ Φ₁(t)

    with `λ` implicitly defined such that `(q', p') = Ψ₁(t)(q, p) ⟹ ∂c(q') p' = 0` for
    any initial state `(q, p)` in the co-tangent bundle, with `c(q') = 0` trivially
    satisfied as `Φ₁` is an identity map in the position:

        Φ₁(t)(q, p) = (q, p - t ∇h₁(q))

    The map `Ψ₁(t)` therefore corresponds to taking an unconstrained step `Φ₁(t)` and
    then projecting the resulting momentum back in to the co-tangent space. For the
    usual case in which `h` includes only quadratic terms in the momentum `p` such that
    `∇₂h(q, p)` is a linear function of `p`, then `λ` can be analytically solved for
    to give a closed-form expression for the projection into the co-tangent space.

    We also define a map `Ψ₂` in terms of the `h₂` flow map `Φ₂` as

        Ψ₂(t) = Π(λ') ∘ Φ₂(t) ∘ Π(λ)

    such that for `(q', p') = Ψ₂(t)(q, p)`, `λ` is implicitly defined such that
    `c(q') = 0` and `λ'` is implicitly defined such that `∂c(q') ∇₂h(q', p') = 0`.

    This can be decomposed as first solving for `λ` such that

        c((Φ₂(t) ∘ Π(λ)(q, p))₁) = 0

    i.e. solving for the values of the Lagrange multipliers such that the position
    component of the output of `Φ₂(t) ∘ Π(λ)` is on the manifold, with this typically
    a non-linear system of equations that will need to be solved iteratively e.g. using
    Newton's method. The momentum output of `Φ₂(t) ∘ Π(λ)` is then projected in to the
    cotangent space to compute the final state pair, with this projection step as noted
    above typically having an analytic solution.

    The overall second-order integrator is then defined as the symmetric composition

        Ψ(t) = Ψ₁(t / 2) ∘ Ψ₂(t / N)ᴺ ∘ Ψ₁(t / 2)

    where `N` is a positive integer corresponding to the number of 'inner' `h₂` flow
    steps. This integrator exactly preserves the constraints at all steps, such that if
    an initial position momentum pair `(q, p)` are in the cotangent bundle, the
    corresponding pair after calling the `step` method of the integrator will also be in
    the cotangent bundle.

    For more details see Reich (1996) and section 7.5.1 in Leimkuhler and Reich (2004).

    References:

    Reich, S. (1996). Symplectic integration of constrained Hamiltonian systems by
    composition methods. SIAM journal on numerical analysis, 33(2), 475-491.

    Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
    Cambridge University Press.
    """

    def __init__(
        self,
        system: System,
        step_size: Optional[float] = None,
        n_inner_step: int = 1,
        reverse_check_tol: float = 2e-8,
        reverse_check_norm: NormFunction = maximum_norm,
        projection_solver: ProjectionSolver = solve_projection_onto_manifold_quasi_newton,
        projection_solver_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of.
            step_size: Integrator time step. If set to `None` it is assumed that a step
                size adapter will be used to set the step size before calling the `step`
                method.
            n_inner_step: Positive integer specifying number of 'inner' constrained
                `system.h2_flow` steps to take within each overall step. As the
                derivative `system.dh1_dpos` is not evaluated during the
                `system.h2_flow` steps, if this derivative is relatively expensive to
                compute compared to evaluating `system.h2_flow` then compared to using
                `n_inner_step = 1` (the default) for a given `step_size` it can be more
                computationally efficient to use `n_inner_step > 1` in combination
                within a larger `step_size`, thus reducing the number of
                `system.dh1_dpos` evaluations to simulate forward a given time while
                still controlling the effective time step used for the constrained
                `system.h2_flow` steps which involve solving a non-linear system of
                equations to retract the position component of the updated state back on
                to the manifold, with the iterative solver typically diverging if the
                time step used is too large.
            reverse_check_tol: Tolerance for check of reversibility of implicit
                sub-steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                `reverse_check_norm` argument) of `reverse_check_tol` of the original
                state position component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            projection_solver: Function which given two states `state` and `state_prev`,
                floating point time step `time_step` and a Hamiltonian system object
                `system` solves the non-linear system of equations in `λ`

                    system.constr(
                        state.pos + dh2_flow_pos_dmom @
                            system.jacob_constr(state_prev).T @ λ) == 0

                where `dh2_flow_pos_dmom = system.dh2_flow_dmom(time_step)[0]` is the
                derivative of the action of the (linear) `system.h2_flow` map on the
                state momentum component with respect to the position component. This is
                used to project the state position component back on to the manifold
                after an unconstrained `system.h2_flow` update.
            projection_solver_kwargs: Dictionary of any keyword arguments to
                `projection_solver`.
        """
        super().__init__(system, step_size)
        self.n_inner_step = n_inner_step
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm
        self.projection_solver = projection_solver
        if projection_solver_kwargs is None:
            projection_solver_kwargs = {}
        self.projection_solver_kwargs = projection_solver_kwargs

    def _h2_flow_retraction_onto_manifold(
        self, state: ChainState, state_prev: ChainState, time_step: float
    ):
        self.system.h2_flow(state, time_step)
        self.projection_solver(
            state, state_prev, time_step, self.system, **self.projection_solver_kwargs
        )

    def _project_onto_cotangent_space(self, state: ChainState):
        state.mom = self.system.project_onto_cotangent_space(state.mom, state)

    def _step_a(self, state: ChainState, time_step: float):
        self.system.h1_flow(state, time_step)
        self._project_onto_cotangent_space(state)

    def _step_b(self, state: ChainState, time_step: float):
        time_step_inner = time_step / self.n_inner_step
        for i in range(self.n_inner_step):
            state_prev = state.copy()
            self._h2_flow_retraction_onto_manifold(state, state_prev, time_step_inner)
            if i == self.n_inner_step - 1:
                # If at last inner step pre-evaluate dh1_dpos before projecting
                # state on to cotangent space, with computed value being
                # cached. During projection the constraint Jacobian at new
                # position will be calculated however if we are going to make a
                # h1_flow step immediately after we will evaluate dh1_dpos
                # which may involve evaluating the gradient of the log
                # determinant of the Gram matrix, during which we will evaluate
                # the constraint Jacobian in the forward pass anyway.
                # Pre-evaluating here therefore saves one extra Jacobian
                # evaluation when the target density includes a Gram matrix log
                # determinant term (and will not add any cost if this is not
                # the case as dh1_dpos will still be cached and reused).
                self.system.dh1_dpos(state)
            self._project_onto_cotangent_space(state)
            state_back = state.copy()
            self._h2_flow_retraction_onto_manifold(state_back, state, -time_step_inner)
            rev_diff = self.reverse_check_norm(state_back.pos - state_prev.pos)
            if rev_diff > self.reverse_check_tol:
                raise NonReversibleStepError(
                    f"Non-reversible step. Distance between initial and "
                    f"forward-backward integrated positions = {rev_diff:.1e}."
                )

    def _step(self, state: ChainState, time_step: float):
        self._step_a(state, 0.5 * time_step)
        self._step_b(state, time_step)
        self._step_a(state, 0.5 * time_step)
