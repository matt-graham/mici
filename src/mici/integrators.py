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
    solve_projection_onto_manifold_newton,
    FixedPointSolver,
    ProjectionSolver,
)

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence
    from mici.states import ChainState
    from mici.systems import ConstrainedTractableFlowSystem, System, TractableFlowSystem
    from mici.types import NormFunction


class Integrator(ABC):
    r"""Base class for integrators for simulating Hamiltonian dynamics.

    For a Hamiltonian function :math:`h` with position variables :math:`q` and momentum
    variables :math:`p`, the canonical Hamiltonian dynamic is defined by the ordinary
    differential equation system

    .. math::

        \dot{q} = \nabla_2 h(q, p),  \qquad \dot{p} = -\nabla_1 h(q, p),

    with the flow map :math:`\Phi` corresponding to the solution of the corresponding
    initial value problem a time-reversible and symplectic (and by consequence
    volume-preserving) map.

    Derived classes implement a :py:meth:`step` method which approximates the flow-map
    with :math:`\Psi(t) \approx \Phi(t)` over some small time interval :math:`t`, while
    conserving the properties of being time-reversible and symplectic, with composition
    of this integrator step method allowing simulation of time-discretised trajectories
    of the Hamiltonian dynamics.
    """

    def __init__(self, system: System, step_size: Optional[float] = None):
        """
        Args:
            system: Hamiltonian system to integrate the dynamics of.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                `step` method.
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
    r"""Base class for integrators for Hamiltonian systems with tractable flows.

    The Hamiltonian function is assumed to be expressible as the sum of two analytically
    tractable components for which the corresponding Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function :math:`h` takes
    the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p),

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    and :math:`h_1` and :math:`h_2` are Hamiltonian component functions for which the
    exact flow maps, :math:`\Phi_1` and :math:`\Phi_2` respectively, can be computed
    exactly.
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
    r"""Leapfrog integrator for Hamiltonian systems with tractable component flows.

    The overall integrator step :math:`\Psi` is defined by the symmetric composition

    .. math::

        \Psi(t) = \Phi_1(t/2) \circ \Phi_2(t) \circ \Phi_1(t/2)

    where :math:`\Phi_1` and :math:`\Phi_2` are the exact flow maps associated with the
    Hamiltonian components :math:`h_1` and :math:`h_2` respectively.

    For separable Hamiltonians of the

    .. math::

        h(q, p) = h_1(q) + h_2(p),

    where :math:`h_1` is the potential energy and :math:`h_2` is the kinetic energy,
    this integrator corresponds to the classic (position) Störmer-Verlet method.

    The integrator can also be applied to the more general Hamiltonian splitting

    .. math::

        h(q, p) = h_1(q) + h_2(q, p),

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
    r"""Symmetric composition integrator for Hamiltonians with tractable flows.

    The Hamiltonian function is assumed to be expressible as the sum of two analytically
    tractable components for which the corresponding Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function :math:`h` takes
    the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p),

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    and :math:`h_1` and :math:`h_2` are Hamiltonian component functions for which the
    exact flows, respectively :math:`\Phi_1` and :math:`\Phi_2`, can be computed. An
    alternating composition can then be formed as

    .. math::

        \Psi(t) = A(a_S t) \circ B(b_S t) \circ \dots \circ
                  A(a_1 t) \circ B(b_1 t) \circ A(a_0 t),

    where :math:`A = \Phi_1` and :math:`B = \Phi_2` or :math:`A = \Phi_2` and :math:`B =
    \Phi_1`, and :math:`(a_0,\dots, a_S)` and :math:`(b_1, \dots, b_S)` are a set of
    coefficients to be determined with :math:`S \geq 1`.

    To ensure a consistency (i.e. the integrator is at least order one) we require that

    .. math::

        \sum_{s=0}^S a_s = \sum_{s=1}^S b_s = 1.

    For symmetric compositions we restrict that

    .. math::

        a_{S-m} = a_m, \quad b_{S+1-m} = b_m,

    with symmetric consistent methods of at least order two.

    The combination of the symmetry and consistency requirements mean that for each
    :math:`S \geq 1` a symmetric composition method can be described by :math:`S - 1`
    'free' coefficients :math:`(a_0, b_1, \dots, a_{K-1}, b_K)`  with
    :math:`K = (S - 1) / 2` if :math:`S > 1` is odd (with no free coefficients for
    :math:`S = 1` case) or :math:`(a_0, b_1, \dots, a_K)` with :math:`K = (S - 2) / 2`
    if :math:`S > 2` is even (with a single free coefficient :math:`a_0` for :math:`S=2`
    case).

    The Störmer-Verlet 'leapfrog' integrator is the special case corresponding to the
    unique (symmetric and consistent) '1-stage' (:math:`S = 1`) integrator.

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
    r"""Two-stage symmetric composition integrator due to Blanes, Casas & Sanz-Serna.

    Described in equation (6.4) in Blanes, Casas, Sanz-Serna (2014).

    Corresponds to specific instance of :py:class:`SymmetricCompositionIntegrator` with
    :math:`S = 2` and free coefficient :math:`a_0 = (3 - \sqrt{3}) / 6`.

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
        a_0 = (3 - 3**0.5) / 6
        super().__init__(system, (a_0,), step_size, True)


class BCSSThreeStageIntegrator(SymmetricCompositionIntegrator):
    """Three-stage symmetric composition integrator due to Blanes, Casas & Sanz-Serna.

    Described in equation (6.7) in Blanes, Casas, Sanz-Serna (2014).

    Corresponds to specific instance of :py:class:`SymmetricCompositionIntegrator` with
    :math:`S = 3` and free coefficients :math:`a_0 = 0.11888010966548` and
    :math:`b_1 = 0.29619504261126`.

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

    Corresponds to specific instance of :py:class:`SymmetricCompositionIntegrator` with
    :math:`S = 4` and free coefficients :math:`a_0 = 0.071353913450279725904`,
    :math:`b_1 = 0.191667800000000000000` and :math:`a_1 = 268548791161230105820`.

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
    r"""Implicit leapfrog integrator for Hamiltonians with a non-separable component.

    Also known as the generalised leapfrog method.

    The Hamiltonian function :math:`h` is assumed to take the form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p),

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    :math:`h_1` is a Hamiltonian component function for which the exact flow can be
    computed and :math:`h_2` is a Hamiltonian component function of the position and
    momentum variables, which may be non-separable and for which exact simulation of the
    correspond Hamiltonian flow may not be possible.

    The overall integrator step :math:`\Psi` is defined by the symmetric composition

    .. math::

        \Psi(t) =
        A(t/2) \circ B(t/2) \circ C(t/2) \circ C^*(t/2) \circ B^*(t/2) \circ A^*(t/2),

    where the *adjoint* of a flow map :math:`X` is defined such that :math:`X^*(t) =
    X(-t)^{-1}` and the component maps are defined by

    .. math::

        A(t)(q, p) = A^*(t)(q, p) = (q, p - t\nabla h_1(q)), \\
        B(t)(q, p) = \lbrace (q, p') : p' = p - t \nabla_1 h_2(q, p') \rbrace, \\
        B^*(t)(q, p) = (q, p - t \nabla_1 h_2(q, p)), \\
        C(t)(q, p) = (q + t \nabla_2 h_2(q, p), p), \\
        C^*(t)(q, p) = \lbrace (q', p) : q' = q + t \nabla_2 h_2(q', p) \rbrace.

    The resulting implicit integrator is a symmetric second-order method corresponding
    to a symplectic partitioned Runge-Kutta method. See Section 6.3.2 in Leimkuhler and
    Reich (2004) for more details.

    Fixed-point iterations are used to solve the non-linear systems of equations in the
    implicit component updates (:math:`B` and :math:`C^*`). As the iterative solves may
    fail to converge, or may converge to one of multiple solutions, following the
    approach proposed by Zappa, Holmes-Cerfon and Goodman (2018), an explicit
    *reversibility check* is performed to ensure the overall integrator step is
    time-reversible. If the reversibility check fails or the iterative solver fails to
    converge an appropriate error is raised
    (:py:exc:`mici.errors.NonReversibleStepError` and
    :py:exc:`mici.errors.ConvergenceError` respectively).

    References:

      1. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
      2. Zappa, E., Holmes‐Cerfon, M., & Goodman, J. (2018). Monte Carlo on manifolds:
         sampling densities and integrating functions. Communications on Pure and
         Applied Mathematics, 71(12), 2609-2647.
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
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
            reverse_check_tol: Tolerance for check of reversibility of implicit
                sub-steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                :code:`reverse_check_norm` argument) of :code:`reverse_check_tol` of the
                original state position component. If this condition is not met a
                :py:exc:`mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                :py:func:`mici.solvers.maximum_norm`.
            fixed_point_solver:  Function which given a function :code:`func` and
                initial guess :code:`x0` iteratively solves the fixed point equation
                :code:`func(x) = x` initialising the iteration with :code:`x0` and
                returning an array corresponding to the solution if the iteration
                converges or raising a :py:class:`mici.errors.ConvergenceError`
                otherwise. Defaults to :py:exc:`mici.solvers.solve_fixed_point_direct`.
            fixed_point_solver_kwargs: Dictionary of any keyword arguments to
                :code:`fixed_point_solver`.
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
    r"""Implicit midpoint integrator for general Hamiltonians.

    The Hamiltonian function :math:`h` may be a general (non-separable) function of both
    the position variables :math:`q` and momentum variables :math:`p`.

    The Hamiltonian flow :math:`\Phi` is approximated with the symmetric composition
    :math:`\Psi(t) = A(t/2) \circ A^*(t/2)` of an implicit Euler half-step
    :math:`A(t/2)` with an explicit Euler half-step :math:`A^*(t/2)` (which is adjoint
    to the implicit Euler step, that is :math:`A^*(t) = A(-t)^{-1}`), with the
    components maps defined by

    .. math::

        A(t)(q, p) = \lbrace
        (q', p') :
        q' = q +  t \nabla_2 h(q', p'),
        p' = p - t \nabla_1 h(q', p'))
        \rbrace, \\
        A^*(t)(q, p) = (q +  t \nabla_2 h(q, p), p - t \nabla_1 h(q, p)).

    The resulting implicit integrator is a second-order method corresponding to a
    symplectic one-stage Runge-Kutta method. See Sections 4.1 and 6.3.1 in Leimkuhler
    and Reich (2004) for more details.

    A fixed-point iteration is used to solve the non-linear system of equations in the
    implicit Euler step :math:`A`. As the iterative solve may fail to converge, or may
    converge to one of multiple solutions, following the approach proposed by
    Zappa, Holmes-Cerfon and Goodman (2018), an explicit *reversibility check* is
    performed to ensure the overall integrator step is time-reversible. If the
    reversibility check fails or the iterative solver fails to converge an appropriate
    error is raised (:py:exc:`mici.errors.NonReversibleStepError` and
    :py:exc:`mici.errors.ConvergenceError` respectively).

    References:

      1. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
      2. Zappa, E., Holmes‐Cerfon, M., & Goodman, J. (2018). Monte Carlo on manifolds:
         sampling densities and integrating functions. Communications on Pure and
         Applied Mathematics, 71(12), 2609-2647.
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
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                `step` method.
            reverse_check_tol: Tolerance for check of reversibility of implicit Euler
                steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                :code:`reverse_check_norm` argument) of :code:`reverse_check_tol` of the
                original state position component. If this condition is not met a
                :py:exc:`mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                :py:func:`mici.solvers.maximum_norm`.
            fixed_point_solver: Function which given a function :code:`func` and initial
                guess :code:`x0` iteratively solves the fixed point equation
                :code:`func(x) = x` initialising the iteration with :code:`x0` and
                returning an array corresponding to the solution if the iteration
                converges or raising a :py:exc:`mici.errors.ConvergenceError`
                otherwise. Defaults to :py:func:`mici.solvers.solve_fixed_point_direct`.
            fixed_point_solver_kwargs: Dictionary of any keyword arguments to
                :code:`fixed_point_solver`.
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
    r"""Leapfrog integrator for constrained Hamiltonian systems.

    The Hamiltonian function is assumed to be expressible as the sum of two components
    for which the corresponding (unconstrained) Hamiltonian flows can be exactly
    simulated. Specifically it is assumed that the Hamiltonian function `h` takes the
    form

    .. math::

        h(q, p) = h_1(q) + h_2(q, p),

    where :math:`q` and :math:`p` are the position and momentum variables respectively,
    and :math:`h_1` and :math:`h_2` Hamiltonian component functions for which the exact
    flows, respectively :math:`\Phi_1` and :math:`\Phi_2`, can be computed.

    The system is assumed to be additionally subject to a set of holonomic constraints
    on the position component of the state i.e. that all valid states must satisfy

    .. math::

        c(q) = 0,

    for some differentiable and surjective vector constraint function :math:`c` and the
    set of positions satisfying the constraints implicitly defining a manifold the
    dynamics remain confined to.

    The constraints are enforced by introducing a set of Lagrange multipliers
    :math:`\lambda` of dimension equal to number of constraints, and defining a
    'constrained' Hamiltonian

    .. math::

        \bar{h}(q, p) = h_1(q) + h_2(q, p) + c(q)^T\lambda  ~\text{s.t.}~ c(q) = 0,

    with corresponding dynamics described by the system of differential algebraic
    equations

    .. math::

        \dot{q} = \nabla_2 h(q, p), \quad
        \dot{p} = -\nabla_1 h(q, p) - \partial c(q)^T\lambda, \quad
        c(q) = 0.

    The dynamics implicitly define a set of constraints on the momentum variables,
    differentiating the constraint equation with respect to time giving that

    .. math::

        \partial c(q) \nabla_2 h(q, p) = \partial c(q) \nabla_2 h_2(q, p) = 0.

    The set of momentum variables satisfying the above for given position variables is
    termed the cotangent space of the manifold (at a position), and the set of
    position-momentum pairs for which the position is on the constraint manifold and the
    momentum in the corresponding cotangent space is termed the cotangent bundle.

    To define a second-order symmetric integrator which exactly (up to floating point
    error) preserves these constraints, forming a symplectic map on the cotangent
    bundle, we follow the approach of Reich (1996).

    We first define a map :math:`\Pi` parametrised by a vector of Lagrange multipliers
    :math:`\lambda`

    .. math::

        \Pi(\lambda)(q, p) = (q, p + \partial c(q)^T \lambda),

    with :math:`\lambda` allowed to be an implicitly defined function of :math:`q` and
    :math:`p`.

    We then define a map :math:`A` in terms of the :math:`h_1` flow map :math:`\Phi_1`
    as

    .. math::

        A(t) = \Pi(\lambda) \circ \Phi_1(t),

    with :math:`\lambda` implicitly defined such that for :math:`(q', p') = A(t)(q, p)`
    we have that :math:`\partial c(q') \nabla_2 h_2(q', p') = 0` for any initial state
    :math:`(q, p)` in the co-tangent bundle, with :math:`c(q') = 0` trivially satisfied
    as :math:`\Phi_1` is an identity map in the position:

    .. math::

        \Phi_1(t)(q, p) = (q, p - t \nabla h_1(q)).

    The map :math:`A(t)` therefore corresponds to taking an unconstrained step according
    to the :math:`h_1` component flow map :math:`\Phi_1(t)` and then projecting the
    resulting updated momentum back in to the co-tangent space. For the usual case in
    which :math:`h` includes only quadratic terms in the momentum :math:`p` such that
    :math:`\nabla_2 h(q, p)` is a linear function of :math:`p`, then :math:`\lambda`
    can be analytically solved for to give a closed-form expression for the projection
    into the co-tangent space.

    We also define a map :math:`B` in terms of the :math:`h_2` flow map :math:`\Phi_2`
    as

    .. math::

        B(t) = \Pi(\lambda') \circ \Phi_2(t) \circ \Pi(\lambda),

    such that for :math:`(q', p') = B(t)(q, p)`, :math:`\lambda` is implicitly defined
    such that :math:`c(q') = 0` and :math:`\lambda'` is implicitly defined such that
    :math:`\partial c(q') \nabla_2 h(q', p') = 0`.

    This can be decomposed as first solving for :math:`\lambda` such that

    .. math::

        c((\Phi_2(t) \circ \Pi(\lambda)(q, p))_1)
        = c((\Phi_2(t)(q, p + \partial c(q)^T \lambda))_1) = 0,

    i.e. solving for the values of the Lagrange multipliers such that the position
    component of the output of :math:`\Phi_2(t) \circ \Pi(\lambda)` is on the manifold,
    with this typically a non-linear system of equations that will need to be solved
    iteratively e.g. using Newton's method. The momentum output of :math:`\Phi_2(t)
    \circ \Pi(\lambda)` is then projected in to the cotangent space to compute the final
    state pair, with this projection step as noted above typically having an analytic
    solution.

    For more details see Reich (1996) and section 7.5.1 in Leimkuhler and Reich (2004).

    The overall second-order integrator is then defined as the symmetric composition

    .. math::

        \Psi(t) = A(t / 2) \circ B(t / N)^N \circ A(t / 2),

    where :math:`N` is a positive integer corresponding to the number of 'inner'
    :math:`h_2` flow steps, following the 'geodesic integrator' formulation proposed by
    Leimkuhler and Matthews (2016). The additional flexibility introduced by having
    the possibility of :math:`N > 1` is particularly of use when evaluation of
    :math:`\Phi_1` is significantly more expensive than evaluation of :math:`\Phi_2`; in
    this case using :math:`N > 1` can allow a larger time step :math:`t` to be used than
    may be otherwise possible due to the need to ensure the iterative solver used in
    :math:`B` does not diverge, with a smaller step size :math:`t / N` used for the
    steps involving the iterative solves with the (cheaper) :math:`\Phi_2` flow map
    and a larger step size :math:`t` used for the steps involving the (more expensive)
    :math:`\Phi_1` flow map.

    This integrator exactly preserves the constraints at all steps, such that if an
    initial position momentum pair :math:`(q, p)` are in the cotangent bundle, the
    corresponding pair after calling the :py:meth:`step` method of the integrator will
    also be in the cotangent bundle, *providing the iterative solver converges*.

    As the iterative solves may fail to converge, or may converge to one of multiple
    solutions, following the approach proposed by Zappa, Holmes-Cerfon and Goodman
    (2018), an explicit *reversibility check* is performed to ensure the overall
    integrator step is time-reversible; see also Lelievre, Rousset and Stoltz (2019) for
    an analysis of this approach specifically in the context of Hamiltonian Monte Carlo.
    If the reversibility check fails or the iterative solver fails to converge an
    appropriate error is raised (:py:exc:`mici.errors.NonReversibleStepError` and
    :py:exc:`mici.errors.ConvergenceError` respectively).

    References:

      1. Reich, S. (1996). Symplectic integration of constrained Hamiltonian systems by
         composition methods. SIAM journal on numerical analysis, 33(2), 475-491.
      2. Leimkuhler, B., & Reich, S. (2004). Simulating Hamiltonian Dynamics (No. 14).
         Cambridge University Press.
      3. Leimkuhler, B., & Matthews, C. (2016). Efficient molecular dynamics using
         geodesic integration and solvent–solute splitting. Proceedings of the Royal
         Society A: Mathematical, Physical and Engineering Sciences, 472(2189),
         20160138.
      4. Zappa, E., Holmes‐Cerfon, M., & Goodman, J. (2018). Monte Carlo on manifolds:
         sampling densities and integrating functions. Communications on Pure and
         Applied Mathematics, 71(12), 2609-2647.
      5. Lelievre, T., Rousset, M., & Stoltz, G. (2019). Hybrid Monte Carlo methods for
         sampling probability measures on submanifolds. Numerische Mathematik, 143,
         379-421.
    """

    def __init__(
        self,
        system: ConstrainedTractableFlowSystem,
        step_size: Optional[float] = None,
        n_inner_step: int = 1,
        reverse_check_tol: float = 2e-8,
        reverse_check_norm: NormFunction = maximum_norm,
        projection_solver: ProjectionSolver = solve_projection_onto_manifold_newton,
        projection_solver_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            system: Hamiltonian system to integrate the constrained dynamics of.
            step_size: Integrator time step. If set to :code:`None` it is assumed that a
                step size adapter will be used to set the step size before calling the
                :py:meth:`step` method.
            n_inner_step: Positive integer specifying number of 'inner' constrained
                :code:`system.h2_flow` steps to take within each overall step. As the
                derivative :code:`system.dh1_dpos` is not evaluated during the
                :code:`system.h2_flow` steps, if this derivative is relatively expensive
                to compute compared to evaluating :code:`system.h2_flow` then compared
                to using :code:`n_inner_step = 1` (the default) for a given
                :code:`step_size` it can be more computationally efficient to use
                :code:`n_inner_step > 1` in combination within a larger
                :code:`step_size`, thus reducing the number of :code:`system.dh1_dpos`
                evaluations to simulate forward a given time while still controlling the
                effective time step used for the constrained :code:`system.h2_flow`
                steps which involve solving a non-linear system of equations to retract
                the position component of the updated state back on to the manifold,
                with the iterative solver typically diverging if the
                time step used is too large.
            reverse_check_tol: Tolerance for check of reversibility of implicit
                sub-steps which involve iterative solving of a non-linear system of
                equations. The step is assumed to be reversible if sequentially applying
                the forward and adjoint updates to a state returns to a state with a
                position component within a distance (defined by the
                `reverse_check_norm` argument) of :code:`reverse_check_tol` of the
                original state position component. If this condition is not met a
                :py:exc:`mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm: Norm function accepting a single one-dimensional array
                input and returning a non-negative floating point value defining the
                distance to use in the reversibility check. Defaults to
                :py:func:`mici.solvers.maximum_norm`.
            projection_solver: Function which given two states :code:`state` and
                :code:`state_prev`, floating point time step :code:`time_step` and a
                Hamiltonian system object :code:`system` solves the non-linear system of
                equations in :code:`λ`

                .. code::

                    system.constr(
                        state.pos
                        + dh2_flow_pos_dmom @ system.jacob_constr(state_prev).T @ λ
                    ) == 0

                where :code:`dh2_flow_pos_dmom = system.dh2_flow_dmom(time_step)[0]` is
                the derivative of the action of the (linear) :code:`system.h2_flow` map
                on the state momentum component with respect to the position component.
                This is used to project the state position component back on to the
                manifold after an unconstrained :code:`system.h2_flow` update. If the
                solver fails to convege a :py:exc:`mici.errors.ConvergenceError`
                exception is raised.
            projection_solver_kwargs: Dictionary of any keyword arguments to
                :code:`projection_solver`.
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
