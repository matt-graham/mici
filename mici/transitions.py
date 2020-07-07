"""Markov transition kernels."""

from abc import ABC, abstractmethod, abstractproperty
from collections import namedtuple
import logging
import numpy as np
from mici.utils import LogRepFloat
from mici.errors import (IntegratorError, NonReversibleStepError,
                         ConvergenceError, HamiltonianDivergenceError)

logger = logging.getLogger(__name__)


def _process_integrator_error(exception, stats):
    logger.info(f'Terminating trajectory due to error:\n{exception!s}')
    # Only set stats fields to True if exception is of matching type.
    # Corresponding fields should be set to False by default for transitions
    # which potentially raise these errors.
    if isinstance(exception, HamiltonianDivergenceError):
        stats['diverging'] = True
    elif isinstance(exception, NonReversibleStepError):
        stats['non_reversible_step'] = True
    elif isinstance(exception, ConvergenceError):
        stats['convergence_error'] = True


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
        statistic values and second entry the default value to initialize this
        array with.
        """

    @abstractmethod
    def sample(self, state, rng):
        """Sample a new chain state from the Markov transition kernel.

        Args:
            state (mici.states.ChainState): Current chain state to condition
                transition kernel on.
            rng (numpy.random.Generator): Numpy random number generator.

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
            rng (numpy.random.Generator): Numpy random number generator.

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
    randomized at all and succesive applications of the coupled integration
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
        'n_step': (np.int64, -1),
        'accept_stat': (np.float64, np.nan),
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
            rng (numpy.random.Generator): Numpy random number generator.

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

    def __init__(self, system, integrator):
        super().__init__(system, integrator)
        self.statistic_types['metrop_accept_prob'] = (np.float64, np.nan)

    def _sample_n_step(self, state, n_step, rng):
        h_init = self.system.h(state)
        state_p = state
        integration_error = False
        try:
            for s in range(n_step):
                state_p = self.integrator.step(state_p)
        except IntegratorError as e:
            integration_error = True
            stats = {'n_step': s}
            _process_integrator_error(e, stats)
        else:
            stats = {'n_step': n_step}
            # Reverse integration direction of proposal to form an involution
            state_p.dir *= -1
        if state_p is not state:
            h_final = self.system.h(state_p)
            metrop_ratio = np.exp(h_init - h_final)
            accept_prob = 0 if np.isnan(metrop_ratio) else min(1, metrop_ratio)
        else:
            accept_prob = 0.
        stats['metrop_accept_prob'] = accept_prob
        stats['accept_stat'] = accept_prob if not integration_error else 0
        if not integration_error and rng.uniform() < accept_prob:
            state = state_p
        # Reverse integration direction of new state
        # As extended target distribution is symmetric in direction indicator
        # this always leaves the distribution invariant
        state.dir *= -1
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
        n_step = rng.integers(*self.n_step_range)
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
    """Generalized no-U-turn termination criterion on Riemannian manifolds [2].

    Terminates trajectories when the velocities at the terminal states of
    the trajectory both have negative dot products with the sum of the
    the momentums across the trajectory from the first to second terminal state
    of the first terminal state to the position of the second terminal state.
    This generalizes the no-U-turn criterion of [1] to Riemannian manifolds
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


_SubTree = namedtuple('_SubTree', [
    'negative', 'positive', 'sum_mom', 'weight', 'depth'])


class DynamicIntegrationTransition(IntegrationTransition):
    """Base class for dynamic integration transitions.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps equal
    to the previous tree size until a termination criteria on the tree's
    subtrees is met. The next chain state is chosen from the candidate states
    using a progressive sampling scheme based on relative weights of the
    different candidate states, with the sampling biased towards states further
    from the current state [1, 2].

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte
         Carlo. arXiv preprint arXiv:1701.02434.
    """
    def __init__(self, system, integrator,
                 max_tree_depth=10, max_delta_h=1000,
                 termination_criterion=riemannian_no_u_turn_criterion,
                 do_extra_subtree_checks=True):
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
            do_extra_subtree_checks (bool): Whether to perform additional
                termination criterion checks on overlapping subtrees of the
                current tree to improve robustness in systems with dynamics
                which are well approximated by independent system of simple
                harmonic oscillators. In such systems (corresponding to e.g.
                a standard normal target distribution and identity metric
                matrix representation) at certain step sizes a 'resonant'
                behaviour is seen by which the termination criterion fails to
                detect that the trajectory has expanded past a half-period i.e.
                has 'U-turned' resulting in trajectories continuing to expand,
                potentially up until the `max_tree_depth` limit is hit. For
                more details see the Stan Discourse discussion at kutt.it/yAkIES
                If `do_extra_subtree_checks` is set to `True` additional
                termination criterion checks are performed on overlapping
                subtrees which help to reduce this resonant behaviour at the
                cost of more conservative trajectory termination in some
                correlated models and some overhead from additional checks.
        """
        super().__init__(system, integrator)
        assert max_tree_depth > 0, 'max_tree_depth must be non-negative'
        self.max_tree_depth = max_tree_depth
        self.max_delta_h = max_delta_h
        self.termination_criterion = termination_criterion
        self.do_extra_subtree_checks = do_extra_subtree_checks
        self.statistic_types['av_metrop_accept_prob'] = (np.float64, np.nan)
        self.statistic_types['reject_prob'] = (np.float64, np.nan)
        self.statistic_types['tree_depth'] = (np.int64, -1)
        self.statistic_types['diverging'] = (np.bool, False)

    def _termination_criterion(self, tree, neg_subtree, pos_subtree):
        # If performing extra subtree checks evaluate lazily i.e. only evaluate
        # if initial whole tree check fails. Extra subtree checks also only
        # performed for trees of depth 2 and above (i.e. containing at least
        # 4 states) as for trees of depth 1 they are redundant.
        if self.termination_criterion(
                self.system, tree.negative, tree.positive, tree.sum_mom):
            return True
        elif tree.depth > 1 and self.do_extra_subtree_checks:
            if self.termination_criterion(
                    self.system, neg_subtree.negative, pos_subtree.negative,
                    neg_subtree.sum_mom + pos_subtree.negative.mom):
                return True
            elif self.termination_criterion(
                    self.system, neg_subtree.positive, pos_subtree.positive,
                    pos_subtree.sum_mom + neg_subtree.positive.mom):
                return True
        return False

    def _new_leave(self, state, h, aux_info):
        return _SubTree(
            negative=state, positive=state, sum_mom=np.asarray(state.mom),
            weight=self._weight_function(h, aux_info), depth=0)

    def _merge_subtrees(self, neg_subtree, pos_subtree):
        assert neg_subtree.depth == pos_subtree.depth, (
            'Cannot merge subtrees of different depths')
        return _SubTree(
            negative=neg_subtree.negative, positive=pos_subtree.positive,
            weight=neg_subtree.weight + pos_subtree.weight,
            sum_mom=neg_subtree.sum_mom + pos_subtree.sum_mom,
            depth=neg_subtree.depth + 1)

    def _init_aux_vars(self, state, rng):
        return {'h_init': self.system.h(state)}

    @abstractmethod
    def _weight_function(self, h, aux_vars):
        pass

    @abstractmethod
    def _weight_ratio(self, numerator, denominator):
        pass

    @abstractmethod
    def _check_divergence(self, h, aux_vars):
        pass

    def _build_tree(self, depth, state, stats, rng, aux_vars):
        if depth == 0:
            # recursion base case
            try:
                # integrate forward/backward one step depending on state.dir
                state = self.integrator.step(state)
                h = self.system.h(state)
                h = np.inf if np.isnan(h) else h
                # create new tree leave
                tree = self._new_leave(state, h, aux_vars)
                proposal = state
                # accumulate stats
                stats['sum_metrop_accept_prob'] += min(
                    1, np.exp(aux_vars['h_init'] - h))
                stats['n_step'] += 1
                # default to assuming valid and then check for divergence
                terminate = False
                self._check_divergence(h, aux_vars)
            except IntegratorError as e:
                _process_integrator_error(e, stats)
                terminate, tree, proposal = True, None, None
            return terminate, tree, proposal
        # build 'inner' subtree, i.e. starting from current state
        terminate, inner_tree, inner_proposal = self._build_tree(
            depth - 1, state, stats, rng, aux_vars)
        if terminate:
            return terminate, None, None
        # build 'outer' subtree, i.e. starting from terminus of inner subtree
        state = inner_tree.positive if state.dir == 1 else inner_tree.negative
        terminate, outer_tree, outer_proposal = self._build_tree(
            depth - 1, state, stats, rng, aux_vars)
        if terminate:
            return terminate, None, None
        # merge two subtrees accounting for integration direction
        neg_subtree = inner_tree if state.dir == 1 else outer_tree
        pos_subtree = outer_tree if state.dir == 1 else inner_tree
        tree = self._merge_subtrees(neg_subtree, pos_subtree)
        # sample new proposal from two subtree proposals according to weights
        accept_outer_prob = self._weight_ratio(outer_tree.weight, tree.weight)
        proposal = (
            outer_proposal if rng.uniform() < accept_outer_prob else
            inner_proposal)
        # check termination criterion on tree and subtrees
        terminate = self._termination_criterion(tree, neg_subtree, pos_subtree)
        return terminate, tree, proposal

    def sample(self, state, rng):
        stats = {'n_step': 0, 'sum_metrop_accept_prob': 0., 'reject_prob': 1.}
        aux_vars = self._init_aux_vars(state, rng)
        tree = self._new_leave(state, aux_vars['h_init'], aux_vars)
        next_state = state
        for depth in range(self.max_tree_depth):
            # uniformly sample direction to expand tree in
            direction = 2 * (rng.uniform() < 0.5) - 1
            state = tree.positive if direction == 1 else tree.negative
            state.dir = direction
            # expand tree by building new subtree of current depth
            terminate, new_tree, new_proposal = self._build_tree(
                depth, state, stats, rng, aux_vars)
            if terminate:
                break
            # progressively sample new state by choosing between
            # current new state and proposal from new subtree, biasing
            # towards the new subtree proposal
            accept_proposal_prob = self._weight_ratio(
                new_tree.weight, tree.weight)
            if rng.uniform() < accept_proposal_prob:
                next_state = new_proposal
            # each proposal acceptance independent therefore overall probability
            # of 'rejecting' - i.e. not accepting all proposals is product of
            # probabilties of not accepting each proposal
            stats['reject_prob'] *= (1. - accept_proposal_prob)
            # merge new subtree into current tree accounting for direction
            neg_subtree = tree if direction == 1 else new_tree
            pos_subtree = new_tree if direction == 1 else tree
            tree = self._merge_subtrees(neg_subtree, pos_subtree)
            # check termination criterion on new tree and subtrees
            if self._termination_criterion(tree, neg_subtree, pos_subtree):
                break
        sum_accept_prob = stats.pop('sum_metrop_accept_prob')
        if stats['n_step'] > 0:
            stats['av_metrop_accept_prob'] = sum_accept_prob / stats['n_step']
        else:
            stats['av_metrop_accept_prob'] = 0.
        if any(stats.get(key, False) for key in
               ['diverging', 'convergence_error', 'non_reversible_step']):
            stats['accept_stat'] = 0.
        else:
            stats['accept_stat'] = stats['av_metrop_accept_prob']
        stats['tree_depth'] = depth
        return next_state, stats


class MultinomialDynamicIntegrationTransition(DynamicIntegrationTransition):
    """Dynamic integration transition with multinomial sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps
    equal to the previous tree size [1,2] until a termination criteria on the
    tree leaves is met. The next chain state is chosen from the candidate
    states using a progressive multinomial sampling scheme [2] based on the
    relative probability densities of the different candidate states, with the
    sampling biased towards states further from the current state.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte
         Carlo. arXiv preprint arXiv:1701.02434.
    """

    def _weight_function(self, h, aux_vars):
        return LogRepFloat(log_val=-h)

    def _weight_ratio(self, numerator, denominator):
        return min(numerator / denominator, 1)

    def _check_divergence(self, h, aux_vars):
        if h - aux_vars['h_init'] > self.max_delta_h:
            raise HamiltonianDivergenceError(
                f'delta_h = {h - aux_vars["h_init"]}')


class SliceDynamicIntegrationTransition(DynamicIntegrationTransition):
    """Dynamic integration transition with slice sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps equal
    to the previous tree size until a termination criteria on the tree leaves is
    met. The next chain state is chosen from the candidate states using a
    progressive slice sampling scheme based on the relative probability
    densities of the different candidate states, with the slice sampler biased
    towards states further from the current state.

    When used with the `euclidean_no_u_turn_criterion` this transition is
    equivalent to the transitions in 'Algorithm 3: Efficient No-U-Turn Sampler'
    in [1].

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
    """

    def _init_aux_vars(self, state, rng):
        aux_vars = super()._init_aux_vars(state, rng)
        aux_vars['log_u'] = np.log(rng.uniform()) - aux_vars['h_init']
        return aux_vars

    def _weight_function(self, h, aux_vars):
        return (aux_vars['log_u'] <= -h) * 1

    def _weight_ratio(self, numerator, denominator):
        return (
            min(numerator / denominator, 1) if denominator > 0 else
            min(numerator, 1))

    def _check_divergence(self, h, aux_vars):
        if h + aux_vars['log_u'] > self.max_delta_h:
            raise HamiltonianDivergenceError(
                f'delta_h = {h + aux_vars["log_u"]}')
