"""Symplectic integrators for simulation of Hamiltonian dynamics."""

from abc import ABC, abstractmethod
from mici.errors import NonReversibleStepError, AdaptationError
from mici.solvers import (maximum_norm, solve_fixed_point_direct,
                          solve_projection_onto_manifold_quasi_newton)


class Integrator(ABC):
    """Base class for integrators."""

    def __init__(self, system, step_size=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to integrate the
                dynamics of.
            step_size (float or None): Integrator time step. If set to `None`
                (the default) it is assumed that a step size adapter will be
                used to set the step size before calling the `step` method.
        """
        self.system = system
        self.step_size = step_size

    def step(self, state):
        """Perform a single integrator step from a supplied state.

        Args:
            state (mici.states.ChainState): System state to perform integrator
                step from.

        Returns:
            new_state (mici.states.ChainState): New object corresponding to
                stepped state.
        """
        if self.step_size is None:
            raise AdaptationError(
                'Integrator `step_size` is `None`. This value should only be '
                'used if a step size adapter is being used to set the step '
                'size.')
        state = state.copy()
        self._step(state, state.dir * self.step_size)
        return state

    @abstractmethod
    def _step(self, state, dt):
        """Implementation of single integrator step.

        Args:
            state (mici.states.ChainState): System state to perform integrator
                step from. Updated in place.
            dt (float): Integrator time step. May be positive or negative.
        """


class LeapfrogIntegrator(Integrator):
    r"""
    Leapfrog integrator for Hamiltonian systems with tractable component flows.

    The Hamiltonian function is assumed to be expressible as the sum of two
    analytically tractable components for which the corresponding Hamiltonian
    flows can be exactly simulated. Specifically it is assumed that the
    Hamiltonian function \(h\) takes the form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    and \(h_1\) and \(h_2\) are Hamiltonian component functions for which the
    exact flows can be computed.
    """

    def __init__(self, system, step_size=None):
        if not hasattr(system, 'h1_flow') or not hasattr(system, 'h2_flow'):
            raise ValueError(
                'Explicit leapfrog integrator can only be used for systems '
                'with explicit `h1_flow` and `h2_flow` Hamiltonian component '
                'flow maps. For systems in which only `h1_flow` is available '
                'the `ImplicitLeapfrogIntegrator` class may be used instead.')
        super().__init__(system, step_size)

    def _step(self, state, dt):
        self.system.h1_flow(state, 0.5 * dt)
        self.system.h2_flow(state, dt)
        self.system.h1_flow(state, 0.5 * dt)


class ImplicitLeapfrogIntegrator(Integrator):
    r"""
    Implicit leapfrog integrator for Hamiltonian with non-separable component.

    The Hamiltonian function \(h\) is assumed to take the form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    \(h_1\) is a Hamiltonian component function for which the exact flow can be
    computed and \(h_2\) is a non-separable Hamiltonian component function of
    the position and momentum variables and for which exact simulation of the
    correspond Hamiltonian flow is not possible. A pair of implicit component
    updates are used to approximate the flow due to the \(h_2\) Hamiltonian
    component, with a fixed-point iteration used to solve the non-linear system
    of equations.
    """

    def __init__(self, system, step_size=None, reverse_check_tol=1e-8,
                 reverse_check_norm=maximum_norm,
                 fixed_point_solver=solve_fixed_point_direct,
                 fixed_point_solver_kwargs=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to integrate the
                dynamics of.
            step_size (float or None): Integrator time step. If set to `None`
                (the default) it is assumed that a step size adapter will be
                used to set the step size before calling the `step` method.
            reverse_check_tol (float): Tolerance for check of reversibility of
                implicit sub-steps which involve iterative solving of a
                non-linear system of equations. The step is assumed to be
                reversible if sequentially applying the forward and adjoint
                updates to a state returns to a state with a position component
                within a distance (defined by the `reverse_check_norm`
                argument) of `reverse_check_tol` of the original state position
                component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm (Callable[[array], float]): Norm function
                accepting a single one-dimensional array input and returning a
                non-negative floating point value defining the distance to use
                in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            fixed_point_solver (Callable[[Callable[array], array], array]):
                Function which given a function `func` and initial guess `x0`
                iteratively solves the fixed point equation `func(x) = x`
                initialising the iteration with `x0` and returning an array
                corresponding to the solution if the iteration converges or
                raising a `mici.errors.ConvergenceError` otherwise. Defaults to
                `mici.solvers.solve_fixed_point_direct`.
            fixed_point_solver_kwargs (None or Dict[str, object]): Dictionary
                of any keyword arguments to `fixed_point_solver`.
        """
        super().__init__(system, step_size)
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = maximum_norm
        self.fixed_point_solver = fixed_point_solver
        if fixed_point_solver_kwargs is None:
            fixed_point_solver_kwargs = {}
        self.fixed_point_solver_kwargs = fixed_point_solver_kwargs

    def _solve_fixed_point(self, fixed_point_func, x_init):
        return self.fixed_point_solver(
            fixed_point_func, x_init, **self.fixed_point_solver_kwargs)

    def _step_a(self, state, dt):
        self.system.h1_flow(state, dt)

    def _step_b_fwd(self, state, dt):
        def fixed_point_func(mom):
            state.mom = mom
            return mom_init - dt * self.system.dh2_dpos(state)
        mom_init = state.mom
        state.mom = self._solve_fixed_point(fixed_point_func, mom_init)

    def _step_b_adj(self, state, dt):
        mom_init = state.mom.copy()
        state.mom -= dt * self.system.dh2_dpos(state)
        state_back = state.copy()
        self._step_b_fwd(state_back, -dt)
        rev_diff = self.reverse_check_norm(state_back.mom - mom_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated momentums = {rev_diff:.1e}.')

    def _step_c_fwd(self, state, dt):
        pos_init = state.pos.copy()
        state.pos += dt * self.system.dh2_dmom(state)
        state_back = state.copy()
        self._step_c_adj(state_back, -dt)
        rev_diff = self.reverse_check_norm(state_back.pos - pos_init)
        if rev_diff > self.reverse_check_tol:
            raise NonReversibleStepError(
                f'Non-reversible step. Distance between initial and '
                f'forward-backward integrated positions = {rev_diff:.1e}.')

    def _step_c_adj(self, state, dt):
        def fixed_point_func(pos):
            state.pos = pos
            return pos_init + dt * self.system.dh2_dmom(state)
        pos_init = state.pos
        state.pos = self._solve_fixed_point(fixed_point_func, pos_init)

    def _step(self, state, dt):
        self._step_a(state, dt)
        self._step_b_fwd(state, dt)
        self._step_c_fwd(state, dt)
        self._step_c_adj(state, dt)
        self._step_b_adj(state, dt)
        self._step_a(state, dt)


class ConstrainedLeapfrogIntegrator(Integrator):
    r"""
    Leapfrog integrator for constrained Hamiltonian systems.

    The Hamiltonian function is assumed to be expressible as the sum of two
    components for which the corresponding (unconstrained) Hamiltonian flows
    can be exactly simulated. Specifically it is assumed that the Hamiltonian
    function \(h\) takes the form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    and \(h_1\) and \(h_2\) Hamiltonian component functions for which the exact
    flows can be computed.

    The system is assumed to be additionally subject to a set of holonomic
    constraints on the position component of the state i.e. that all valid
    states must satisfy

    \[ c(q) = 0. \]

    for some differentiable and surjective vector constraint function \(c\) and
    the set of positions satisfying the constraints implicitly defining a
    manifold. There is also a corresponding constraint implied on the momentum
    variables which can be derived by differentiating the above with respect to
    time and using that under the Hamiltonian dynamics the time derivative of
    the position is equal to the negative derivative of the Hamiltonian
    function with respect to the momentum

    \[ \partial c(q) \nabla_2 h(q, p) = 0. \]

    The set of momentum variables satisfying the above for given position
    variables is termed the cotangent space of the manifold (at a position),
    and the set of (position, momentum) pairs for which the position is on the
    constraint manifold and the momentum in the corresponding cotangent space
    is termed the cotangent bundle.

    The integrator exactly preserves these constraints at all steps, such that
    if an initial position momentum pair \((q, p)\) are in the cotangent
    bundle, the corresponding pair after calling the `step` method of the
    integrator will also be in the cotangent bundle.
    """

    def __init__(self, system, step_size=None, n_inner_step=1,
                 reverse_check_tol=2e-8, reverse_check_norm=maximum_norm,
                 projection_solver=solve_projection_onto_manifold_quasi_newton,
                 projection_solver_kwargs=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to integrate the
                dynamics of.
            step_size (float or None): Integrator time step. If set to `None`
                (the default) it is assumed that a step size adapter will be
                used to set the step size before calling the `step` method.
            n_inner_step (int): Positive integer specifying number of 'inner'
                constrained `system.h2_flow` steps to take within each overall
                step. As the derivative `system.dh1_dpos` is not evaluated
                during the `system.h2_flow` steps, if this derivative is
                relatively expensive to compute compared to evaluating
                `system.h2_flow` then compared to using `n_inner_step = 1` (the
                default) for a given `step_size` it can be more computationally
                efficient to use `n_inner_step > 1` in combination within a
                larger `step_size`, thus reducing the number of
                `system.dh1_dpos` evaluations to simulate forward a given time
                while still controlling the effective time step used for the
                constrained `system.h2_flow` steps which involve solving a
                non-linear system of equations to retract the position
                component of the updated state back on to the manifold, with
                the iterative solver typically diverging if the time step used
                is too large.
            reverse_check_tol (float): Tolerance for check of reversibility of
                implicit sub-steps which involve iterative solving of a
                non-linear system of equations. The step is assumed to be
                reversible if sequentially applying the forward and adjoint
                updates to a state returns to a state with a position component
                within a distance (defined by the `reverse_check_norm`
                argument) of `reverse_check_tol` of the original state position
                component. If this condition is not met a
                `mici.errors.NonReversibleStepError` exception is raised.
            reverse_check_norm (Callable[[array], float]): Norm function
                accepting a single one-dimensional array input and returning a
                non-negative floating point value defining the distance to use
                in the reversibility check. Defaults to
                `mici.solvers.maximum_norm`.
            projection_solver (Callable[
                    [ChainState, ChainState, float, System], ChainState]):
                Function which given two states `state` and `state_prev`,
                floating point time step `dt` and a Hamiltonian system object
                `system` solves the non-linear system of equations in `λ`

                    system.constr(
                        state.pos + dh2_flow_pos_dmom @
                            system.jacob_constr(state_prev).T @ λ) == 0

                where `dh2_flow_pos_dmom = system.dh2_flow_dmom(dt)[0]` is the
                derivative of the action of the (linear) `system.h2_flow` map
                on the state momentum component with respect to the position
                component. This is used to project the state position
                component back on to the manifold after an unconstrained
                `system.h2_flow` update. Defaults to
                `mici.solvers.solve_projection_onto_manifold_quasi_newton`.
            projection_solver_kwargs (None or Dict[str, object]): Dictionary of
                any keyword arguments to `projection_solver`.
        """
        super().__init__(system, step_size)
        self.n_inner_step = n_inner_step
        self.reverse_check_tol = reverse_check_tol
        self.reverse_check_norm = reverse_check_norm
        self.projection_solver = projection_solver
        if projection_solver_kwargs is None:
            projection_solver_kwargs = {}
        self.projection_solver_kwargs = projection_solver_kwargs

    def _h2_flow_retraction_onto_manifold(self, state, state_prev, dt):
        self.system.h2_flow(state, dt)
        self.projection_solver(state, state_prev, dt, self.system,
                               **self.projection_solver_kwargs)

    def _project_onto_cotangent_space(self, state):
        state.mom = self.system.project_onto_cotangent_space(state.mom, state)

    def _step_a(self, state, dt):
        self.system.h1_flow(state, dt)
        self._project_onto_cotangent_space(state)

    def _step_b(self, state, dt):
        dt_i = dt / self.n_inner_step
        for i in range(self.n_inner_step):
            state_prev = state.copy()
            self._h2_flow_retraction_onto_manifold(state, state_prev, dt_i)
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
            self._h2_flow_retraction_onto_manifold(state_back, state, -dt_i)
            rev_diff = self.reverse_check_norm(state_back.pos - state_prev.pos)
            if rev_diff > self.reverse_check_tol:
                raise NonReversibleStepError(
                    f'Non-reversible step. Distance between initial and '
                    f'forward-backward integrated positions = {rev_diff:.1e}.')

    def _step(self, state, dt):
        self._step_a(state, 0.5 * dt)
        self._step_b(state, dt)
        self._step_a(state, 0.5 * dt)
