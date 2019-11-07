"""Markov transition kernels."""

from abc import ABC, abstractmethod, abstractproperty
from functools import partial
import logging
import numpy as np
from mici.utils import LogRepFloat
from mici.errors import (
    IntegratorError, NonReversibleStepError, ConvergenceError)

logger = logging.getLogger(__name__)


class Transition(ABC):
    """Base class for Markov transition kernels.

    Defines expected interface for transitions by sampler classes.
    """

    @abstractproperty
    def state_variables(self):
        """A set of names of state variables accessed by this transition."""

    @abstractproperty
    def statistic_types(self):
        """A dictionary describing the statistics computed during transition.

        Either `None` if no statistics are returned by `sample` method or
        a dictionary with string keys and tuple values, with the keys defining
        the keys of the statistics returned in the `trans_stats` return value
        of the `sample` method and the first entry of the value tuples an
        appropriate NumPy `dtype` for the array used to store the corresponding
        statistic values and second entry the default value to initialise this
        array with.
        """

    @abstractmethod
    def sample(self, state, rng):
        """Sample a new chain state from the Markov transition kernel.

        Args:
            state (mici.states.ChainState): Current chain state to condition
                transition kernel on.
            rng (RandomState): NumPy RandomState random number generator.

        Returns:
            state (mici.states.ChainState): Updated state object.
            trans_stats (Dict[str, numeric] or None): Any statistics computed
                during the transition or `None` if no statistics.
        """


class MomentumTransition(Transition):
    """Base class for momentum transitions.

    Markov transition  kernel which leaves the conditional distribution on the
    momentum under the canonical distribution invariant, updating only the
    momentum component of the chain state.
    """

    state_variables = {'mom'}
    statistic_types = None

    def __init__(self, system):
        """
        Args:
            system (mici.systems.System): Hamiltonian system defining
                conditional distribution on momentum to leave invariant.
        """
        self.system = system

    @abstractmethod
    def sample(self, state, rng):
        """Sample a new momentum component to state.

        Assigns a new momentum component to state by sampling from a Markov
        transition kernel which leaves the conditional distribution on the
        momentum under the canonical distribution defined by the Hamiltonian
        system invariant.

        Args:
            state (mici.states.ChainState): Current chain state to sample new
                momentum (momentum updated in place).
            rng (RandomState): NumPy RandomState random number generator.

        Returns:
            state (mici.states.ChainState): Updated state object.
            trans_stats (Dict[str, numeric] or None): Any statistics computed
                during the transition or `None` if no statistics.
        """


class IndependentMomentumTransition(MomentumTransition):
    """Independent momentum transition.

    Independently resamples the momentum component of the state from its
    conditional distribution given the remaining state.
    """

    def sample(self, state, rng):
        state.mom = self.system.sample_momentum(state, rng)
        return state, None


class CorrelatedMomentumTransition(MomentumTransition):
    """Correlated (partial) momentum transition.

    Rather than independently sampling a new momentum, instead a pertubative
    Crank-Nicolson type update which produces a new momentum value with a
    specified correlation with the previous value is used. It is assumed that
    the conditional distribution of the momenta is zero-mean Gaussian such that
    the Crank-Nicolson update leaves the momenta conditional distribution
    exactly invariant. This approach is sometimes known as partial momentum
    refreshing or updating, and was originally proposed in [1].

    If the resampling coefficient is equal to zero then the momentum is not
    randomised at all and succesive applications of the coupled integration
    transitions will continue along the same simulated Hamiltonian trajectory.
    When an integration transition is accepted this means the subsequent
    simulated trajectory will continue evolving in the same direction and so
    not randomising the momentum will reduce random-walk behaviour. However on
    a rejection the integration direction is reversed and so without
    randomisation the trajectory will exactly backtrack along the previous
    tractory states. A resampling coefficient of one corresponds to the
    standard case of independent resampling of the momenta while intermediate
    values between zero and one correspond to varying levels of correlation
    between the pre and post update momentums.

    References:

      1. Horowitz, A.M., 1991. A generalized guided Monte Carlo algorithm.
         Phys. Lett. B, 268(CERN-TH-6172-91), pp.247-252.
    """

    def __init__(self, system, mom_resample_coeff=1.):
        """
        Args:
            system (mici.systems.System): Hamiltonian system defining
                conditional distribution on momentum to leave invariant.
            mom_resample_coeff (float): Scalar value in [0, 1] defining the
                momentum resampling coefficient.
        """
        super().__init__(system)
        assert mom_resample_coeff >= 0 and mom_resample_coeff <= 1, (
            'mom_resample_coeff should have a value in the interval [0, 1].')
        self.mom_resample_coeff = mom_resample_coeff

    def sample(self, state, rng):
        if self.mom_resample_coeff == 1:
            state.mom = self.system.sample_momentum(state, rng)
        elif self.mom_resample_coeff != 0:
            mom_ind = self.system.sample_momentum(state, rng)
            state.mom *= (1. - self.mom_resample_coeff**2)**0.5
            state.mom += self.mom_resample_coeff * mom_ind
        return state, None


class IntegrationTransition(Transition):
    """Base class for integration transtions.

    Markov transition kernel which leaves canonical distribution invariant and
    jointly updates the position and momentum components of the chain state by
    integrating the Hamiltonian dynamics of the system to propose new values
    for the state.
    """

    state_variables = {'pos', 'mom', 'dir'}
    statistic_types = {
        'hamiltonian': (np.float64, np.nan),
        'n_step': (np.int64, -1),
        'accept_prob': (np.float64, np.nan),
        'non_reversible_step': (np.bool, False),
        'convergence_error': (np.bool, False)
    }

    def __init__(self, system, integrator):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            integrator (mici.integrators.Integrator): Symplectic integrator
                appropriate to the specified Hamiltonian system.
        """
        self.system = system
        self.integrator = integrator

    @abstractmethod
    def sample(self, state, rng):
        """Sample a position-momentum pair using integration based proposal(s).

        Samples new position and momentum values from a Markov transition
        kernel which leaves the canonical distribution on the state space
        corresponding to the Hamiltonian system invariant.

        Args:
            state (mici.states.ChainState): Current chain state.
            rng (RandomState): NumPy RandomState random number generator.

        Returns:
            state (mici.states.ChainState): Updated state object.
            trans_stats (Dict[str, numeric]): A dictionary of statistics
                computed during the transition to be recorded.
        """


class MetropolisIntegrationTransition(IntegrationTransition):
    """Base for HMC methods using a Metropolis accept step to sample new state.

    In each transition a trajectory is generated by integrating the Hamiltonian
    dynamics from the current state in the current integration time direction
    for a number of integrator steps.

    The state at the end of the trajectory with the integration direction
    negated (this ensuring the proposed move is an involution) is used as the
    proposal in a Metropolis acceptance step. The integration direction is then
    deterministically negated again irrespective of the accept decision, with
    the effect being that on acceptance the integration direction will be equal
    to its initial value and on rejection the integration direction will be
    the negation of its initial value.
    """

    def _sample_n_step(self, state, n_step, rng):
        h_init = self.system.h(state)
        state_p = state
        try:
            for s in range(n_step):
                state_p = self.integrator.step(state_p)
        except IntegratorError as e:
            logger.info(
                f'Terminating trajectory due to integrator error:\n{e!s}')
            return state, {
                'hamiltonian': h_init, 'accept_prob': 0, 'n_step': s,
                'non_reversible_step': isinstance(e, NonReversibleStepError),
                'convergence_error': isinstance(e, ConvergenceError)}
        state_p.dir *= -1
        h_final = self.system.h(state_p)
        metrop_ratio = np.exp(h_init - h_final)
        accept_prob = 0 if np.isnan(metrop_ratio) else min(1, metrop_ratio)
        if rng.uniform() < accept_prob:
            state = state_p
        state.dir *= -1
        stats = {'hamiltonian': self.system.h(state),
                 'accept_prob': accept_prob, 'n_step': n_step,
                 'non_reversible_step': False, 'convergence_error': False}
        return state, stats


class MetropolisStaticIntegrationTransition(MetropolisIntegrationTransition):
    """Static integration transition with Metropolis sampling of new state.

    In this variant the trajectory is generated by integrating the state
    through time a fixed number of integrator steps. This is original proposed
    Hybrid Monte Carlo (often now instead termed Hamiltonian Monte Carlo)
    algorithm [1,2].

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, integrator, n_step):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            integrator (mici.integrators.Integrator): Symplectic integrator
                appropriate to the specified Hamiltonian system.
            n_step (int): Number of integrator steps to simulate in each
                transition.
        """
        super().__init__(system, integrator)
        assert n_step > 0, 'Number of integrator steps must be positive'
        self.n_step = n_step

    def sample(self, state, rng):
        return self._sample_n_step(state, self.n_step, rng)


class MetropolisRandomIntegrationTransition(MetropolisIntegrationTransition):
    """Random integration transition with Metropolis sampling of new state.

    In each transition a trajectory is generated by integrating the state in
    the current integration direction in time a random integer number of
    integrator steps sampled from the uniform distribution on an integer
    interval. The randomisation of the number of integration steps avoids the
    potential of the chain mixing poorly due to using an integration time close
    to the period of (near) periodic systems [1,2].

    References:

      1. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
      2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
         Physics Letters B, 226(3-4), pp.369-371.
    """

    def __init__(self, system, integrator, n_step_range):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            integrator (mici.integrators.Integrator): Symplectic integrator
                appropriate to the specified Hamiltonian system.
            n_step_range (Tuple[int, int]): Tuple `(lower, upper)` with two
                positive integer entries `lower` and `upper` (with
                `upper > lower`) specifying respectively the lower and upper
                bounds (inclusive) of integer interval to uniformly draw random
                number integrator steps to simulate in each transition.
        """
        super().__init__(system, integrator)
        n_step_lower, n_step_upper = n_step_range
        assert n_step_lower > 0 and n_step_lower < n_step_upper, (
            'Range bounds must be non-negative and first entry less than last')
        self.n_step_range = n_step_range

    def sample(self, state, rng):
        n_step = rng.random_integers(*self.n_step_range)
        return self._sample_n_step(state, n_step, rng)


def euclidean_no_u_turn_criterion(system, state_1, state_2, sum_mom):
    """No-U-turn termination criterion for Euclidean manifolds [1].

    Terminates trajectories when the velocities at the terminal states of
    the trajectory both have negative dot products with the vector from
    the position of the first terminal state to the position of the second
    terminal state, corresponding to further evolution of the trajectory
    reducing the distance between the terminal state positions.

    Args:
        system (mici.systems.System): Hamiltonian system being integrated.
        state_1 (mici.states.ChainState): First terminal state of trajectory.
        state_2 (mici.states.ChainState): Second terminal state of trajectory.
        sum_mom (array): Sum of momentums of trajectory states.

    Returns:
        terminate (bool): True if termination criterion is satisfied.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
    """
    return (
        np.sum(system.dh_dmom(state_1) * (state_2.pos - state_1.pos)) < 0 or
        np.sum(system.dh_dmom(state_2) * (state_2.pos - state_1.pos)) < 0)


def riemannian_no_u_turn_criterion(system, state_1, state_2, sum_mom):
    """Generalised no-U-turn termination criterion on Riemannian manifolds [2].

    Terminates trajectories when the velocities at the terminal states of
    the trajectory both have negative dot products with the sum of the
    the momentums across the trajectory from the first to second terminal state
    of the first terminal state to the position of the second terminal state.
    This generalises the no-U-turn criterion of [1] to Riemannian manifolds
    where due to the intrinsic curvature of the space the geodesic between
    two points is general no longer a straight line.

    Args:
        system (mici.systems.System): Hamiltonian system being integrated.
        state_1 (mici.states.ChainState): First terminal state of trajectory.
        state_2 (mici.states.ChainState): Second terminal state of trajectory.
        sum_mom (array): Sum of momentums of trajectory states.

    Returns:
        terminate (bool): True if termination criterion is satisfied.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Betancourt, M., 2013. Generalizing the no-U-turn sampler to Riemannian
         manifolds. arXiv preprint arXiv:1304.1920.
    """
    return (
        np.sum(system.dh_dmom(state_1) * sum_mom) < 0 or
        np.sum(system.dh_dmom(state_2) * sum_mom) < 0)


class MultinomialDynamicIntegrationTransition(IntegrationTransition):
    """Dynamic integration transition with multinomial sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps
    equal to the previous tree size [1,2] until a termination criteria on the
    tree leaves is met. The next chain state is chosen from the candidate
    states using a progressive multinomial sampling scheme [2] based on the
    relative probability densities of the different candidate states, with the
    resampling biased towards states further from the current state.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte
         Carlo. arXiv preprint arXiv:1701.02434.
    """

    def __init__(self, system, integrator,
                 max_tree_depth=10, max_delta_h=1000,
                 termination_criterion=riemannian_no_u_turn_criterion):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            integrator (mici.integrators.Integrator): Symplectic integrator
                appropriate to the specified Hamiltonian system.
            max_tree_depth (int): Maximum depth to expand trajectory binary
                tree to. The maximum number of integrator steps corresponds to
                `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian
                function over a trajectory before signalling a divergence.
            termination_criterion (
                    Callable[[System, ChainState, ChainState, array], bool]):
                Function computing criterion to use to determine when to
                terminate trajectory tree expansion. The function should take a
                Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory
                (sub-)tree being checked and an array containing the sum of the
                momentums over the trajectory (sub)-tree. Defaults to
                `riemannian_no_u_turn_criterion`.
        """
        super().__init__(system, integrator)
        assert max_tree_depth > 0, 'max_tree_depth must be non-negative'
        self.max_tree_depth = max_tree_depth
        self.max_delta_h = max_delta_h
        self._termination_criterion = partial(termination_criterion, system)
        self.statistic_types['tree_depth'] = (np.int64, -1)
        self.statistic_types['diverging'] = (np.bool, False)

    # Key to subscripts used in build_tree and sample_dynamics_transition
    # _p : proposal
    # _n : next
    # _l : left (negative direction)
    # _r : right (positive direction)
    # _s : subtree
    # _i : inner subsubtree
    # _o : outer subsubtree

    def _build_tree(
            self, depth, state, sum_mom, sum_weight, stats, h_init, rng):
        if depth == 0:
            # recursion base case
            try:
                state = self.integrator.step(state)
                h = self.system.h(state)
                h = np.inf if np.isnan(h) else h
                sum_mom += np.asarray(state.mom)
                sum_weight += LogRepFloat(log_val=-h)
                stats['sum_acc_prob'] += min(1, np.exp(h_init - h))
                stats['n_step'] += 1
                terminate = h - h_init > self.max_delta_h
                if terminate:
                    stats['diverging'] = True
                    logger.info(
                        f'Terminating build_tree due to integrator divergence '
                        f'(delta_h = {h - h_init:.1e}).')
            except IntegratorError as e:
                logger.info(
                    f'Terminating build_tree due to integrator error:\n{e!s}')
                stats['non_reversible_step'] = isinstance(
                    e, NonReversibleStepError)
                stats['convergence_error'] = isinstance(e, ConvergenceError)
                state = None
                terminate = True
            return terminate, state, state, state
        sum_mom_i, sum_mom_o = np.zeros((2,) + state.mom.shape)
        sum_weight_i, sum_weight_o = LogRepFloat(0.), LogRepFloat(0.)
        # build inner subsubtree
        terminate_i, state_i, state, state_pi = self._build_tree(
            depth - 1, state, sum_mom_i, sum_weight_i, stats, h_init, rng)
        if terminate_i:
            return True, None, None, None
        # build outer subsubtree
        terminate_o, _, state_o, state_po = self._build_tree(
            depth - 1, state, sum_mom_o, sum_weight_o, stats, h_init, rng)
        if terminate_o:
            return True, None, None, None
        # independently sample proposal from 2 subsubtrees by relative weights
        sum_weight_s = sum_weight_i + sum_weight_o
        accept_o_prob = sum_weight_o / sum_weight_s
        state_p = state_po if rng.uniform() < accept_o_prob else state_pi
        # update overall tree weight
        sum_weight += sum_weight_s
        # calculate termination criteria for subtree
        sum_mom_s = sum_mom_i + sum_mom_o
        terminate_s = self._termination_criterion(state_i, state_o, sum_mom_s)
        # update overall tree summed momentum
        sum_mom += sum_mom_s
        return terminate_s, state_i, state_o, state_p

    def sample(self, state, rng):
        h_init = self.system.h(state)
        sum_mom = np.asarray(state.mom).copy()
        sum_weight = LogRepFloat(log_val=-h_init)
        stats = {'n_step': 0, 'sum_acc_prob': 0.}
        state_n, state_l, state_r = state, state.copy(), state.copy()
        # set integration directions of initial left and right tree leaves
        state_l.dir = -1
        state_r.dir = +1
        for depth in range(self.max_tree_depth):
            # uniformly sample direction to expand tree in
            direction = 2 * (rng.uniform() < 0.5) - 1
            sum_mom_s = np.zeros(state.mom.shape)
            sum_weight_s = LogRepFloat(0.)
            if direction == 1:
                # expand tree by adding subtree to right edge
                terminate_s, _, state_r, state_p = self._build_tree(
                    depth, state_r, sum_mom_s, sum_weight_s, stats, h_init,
                    rng)
            else:
                # expand tree by adding subtree to left edge
                terminate_s, _, state_l, state_p = self._build_tree(
                    depth, state_l, sum_mom_s, sum_weight_s, stats, h_init,
                    rng)
            if terminate_s:
                break
            # progressively sample new state by choosing between
            # current new state and proposal from new subtree, biasing
            # towards the new subtree proposal
            if rng.uniform() < sum_weight_s / sum_weight:
                state_n = state_p
            sum_weight += sum_weight_s
            sum_mom += sum_mom_s
            if self._termination_criterion(state_l, state_r, sum_mom):
                break
        sum_acc_prob = stats.pop('sum_acc_prob')
        if stats['n_step'] > 0:
            stats['accept_prob'] = sum_acc_prob / stats['n_step']
        else:
            stats['accept_prob'] = 0.
        stats['hamiltonian'] = self.system.h(state_n)
        stats['tree_depth'] = depth
        return state_n, stats
