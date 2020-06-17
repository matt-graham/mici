"""Methods for adaptively setting algorithmic parameters of transitions."""

from abc import ABC, abstractmethod, abstractproperty
from math import exp, log
from mici.errors import IntegratorError, AdaptationError


class Adapter(ABC):
    """Abstract adapter for implementing schemes to adapt transition parameters.

    Adaptation schemes are assumed to be based on updating a collection of
    adaptation variables (collectively termed the adapter state here) after
    each chain transition based on the sampled chain state and/or statistics of
    the transition such as an acceptance probability statistic. After completing
    a chain of one or more adaptive transitions, the final adapter state may be
    used to perform a final update to the transition parameters.
    """

    @abstractmethod
    def initialise(self, chain_state, transition):
        """Initialise adapter state prior to starting adaptive transitions.

        Args:
            chain_state (mici.states.ChainState): Initial chain state adaptive
                transition will be started from. May be used to calculate
                initial adapter state but should not be mutated by method.
            transition (mici.transitions.Transition): Markov transition being
                adapted. Attributes of the transition or child objects may be
                updated in-place by the method.

        Returns:
            adapt_state (Dict[str, Any]): Initial adapter state.
        """

    @abstractmethod
    def update(self, adapt_state, chain_state, trans_stats, transition):
        """Update adapter state after sampling transition being adapted.

        Args:
            adapt_state (Dict[str, Any]): Current adapter state. Entries will
                be updated in-place by the method.
            chain_state (mici.states.ChainState): Current chain state following
                sampling from transition being adapted. May be used to calculate
                adapter state updates but should not be mutated by method.
            trans_stats (Dict[str, numeric]): Dictionary of statistics
                associated with transition being adapted. May be used to
                calculate adapter state updates but should not be mutated by
                method.
            transition (mici.transitions.Transition): Markov transition being
                adapted. Attributes of the transition or child objects may be
                updated in-place by the method.
        """

    @abstractmethod
    def finalise(self, adapt_state, transition):
        """Update transition parameters based on final adapter state or states.

        Optionally, if multiple adapter states are available, e.g. from a set of
        independent adaptive chains, then these adaptation information from all
        the chains may be combined to set the transition parameter(s).

        Args:
            adapt_state (Dict[str, Any] or List[Dict[str, Any]]): Final adapter
                state or a list of adapter states.
            transition (mici.transitions.Transition): Markov transition being
                adapted. Attributes of the transition or child objects will be
                updated in-place by the method.
        """


class DualAveragingStepSizeAdapter(Adapter):
    """Dual averaging integrator step size adapter.

    Implementation of the dual algorithm step size adaptation algorithm
    described in [1], a modified version of the stochastic optimisation scheme
    of [2]. By default the adaptation is performed to control the `accept_prob`
    statistic of an integration transition to be close to a target value but
    the statistic adapted on can be altered by changing the `adapt_stat_func`.


    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
      2. Nesterov, Y., 2009. Primal-dual subgradient methods for convex
         problems. Mathematical programming 120(1), pp.221-259.
    """

    def __init__(self, adapt_stat_target=0.8, adapt_stat_func=None,
                 log_step_size_reg_target=None,
                 log_step_size_reg_coefficient=0.05, iter_decay_coeff=0.75,
                 iter_offset=10, max_init_step_size_iters=100):
        """
        Args:
            adapt_stat_target (float): Target value for the transition statistic
                being controlled during adaptation.
            adapt_stat_func (Callable[[Dict[str, numeric]], numeric]): Function
                which given a dictionary of transition statistics outputs the
                value of the statistic to control during adaptation. By default
                this is set to a function which simply selects the 'accept_stat'
                value in the statistics dictionary.
            log_step_size_reg_target (float or None): Value to regularise the
                controlled output (logarithm of the integrator step size)
                towards. If `None` set to `log(10 * init_step_size)` where
                `init_step_size` is the initial 'reasonable' step size found by
                a coarse search as recommended in Hoffman and Gelman (2014).
                This has the effect of giving the dual averaging algorithm a
                tendency towards testing step sizes larger than the initial
                value, with typically integrating with a larger step size having
                a lower computational cost.
            log_step_size_reg_coefficient (float): Coefficient controlling
                amount of regularisation of controlled output (logarithm of the
                integrator step size) towards `log_step_size_reg_target`.
                Defaults to 0.05 as recommended in Hoffman and Gelman (2014).
            iter_decay_coeff (float): Coefficient controlling exponent of
                decay in schedule weighting stochastic updates to smoothed log
                step size estimate. Should be in the interval (0.5, 1] to ensure
                asymptotic convergence of adaptation. A value of 1 gives equal
                weight to the whole history of updates while setting to a
                smaller value increasingly highly weights recent updates, giving
                a tendency to 'forget' early updates. Defaults to 0.75 as
                recommended in Hoffman and Gelman (2014).
            iter_offset (int): Offset used for the iteration based weighting of
                the adaptation statistic error estimate. Should be set to a
                non-negative value. A value > 0 has the effect of stabilising
                early iterations. Defaults to the value of 10 as recommended in
                Hoffman and Gelman (2014).
            max_init_step_size_iters (int): Maximum number of iterations to use
                in initial search for a reasonable step size with an
                `AdaptationError` exception raised if a suitable step size is
                not found within this many iterations.
        """
        self.adapt_stat_target = adapt_stat_target
        if adapt_stat_func is None:
            def adapt_stat_func(stats): return stats['accept_stat']
        self.adapt_stat_func = adapt_stat_func
        self.log_step_size_reg_target = log_step_size_reg_target
        self.log_step_size_reg_coefficient = log_step_size_reg_coefficient
        self.iter_decay_coeff = iter_decay_coeff
        self.iter_offset = iter_offset
        self.max_init_step_size_iters = max_init_step_size_iters

    def initialise(self, chain_state, transition):
        integrator = transition.integrator
        system = transition.system
        adapt_state = {
            'iter': 0,
            'smoothed_log_step_size': 0.,
            'adapt_stat_error': 0.,
        }
        init_step_size = (
            self._find_and_set_init_step_size(chain_state, system, integrator)
            if integrator.step_size is None else integrator.step_size)
        if self.log_step_size_reg_target is None:
            adapt_state['log_step_size_reg_target'] = log(10 * init_step_size)
        else:
            adapt_state['log_step_size_reg_target'] = (
                self.log_step_size_reg_target)
        return adapt_state

    def _find_and_set_init_step_size(self, state, system, integrator):
        """Find initial step size by coarse search using single step statistics.

        Implementation of Algorithm 4 in Hoffman and Gelman (2014).
        """
        init_state = state.copy()
        h_init = system.h(init_state)
        integrator.step_size = 1.
        try:
            state = integrator.step(init_state)
            delta_h = h_init - system.h(state)
            sign = 2 * (delta_h  > -log(2)) - 1
        except IntegratorError:
            sign = 1
        for s in range(self.max_init_step_size_iters):
            try:
                state = integrator.step(init_state)
                delta_h = h_init - system.h(state)
                if sign * delta_h < -sign * log(2):
                    return  integrator.step_size
                else:
                    integrator.step_size *= 2.**sign
            except IntegratorError:
                integrator.step_size /= 2.
        raise AdaptationError(
            f'Could not find reasonable initial step size in {s + 1} iterations'
            f' (final step size {integrator.step_size}).')

    def update(self, chain_state, adapt_state, trans_stats, transition):
        adapt_state['iter'] += 1
        error_weight = 1 / (self.iter_offset + adapt_state['iter'])
        adapt_state['adapt_stat_error'] *= (1 - error_weight)
        adapt_state['adapt_stat_error'] += error_weight * (
            self.adapt_stat_target - self.adapt_stat_func(trans_stats))
        smoothing_weight = (1 / adapt_state['iter'])**self.iter_decay_coeff
        log_step_size = adapt_state['log_step_size_reg_target'] - (
            adapt_state['adapt_stat_error'] * adapt_state['iter']**0.5 /
            self.log_step_size_reg_coefficient)
        adapt_state['smoothed_log_step_size'] *= (1 - smoothing_weight)
        adapt_state['smoothed_log_step_size'] += (
            smoothing_weight * log_step_size)
        transition.integrator.step_size = exp(log_step_size)

    def finalise(self, adapt_state, transition):
        if isinstance(adapt_state, dict):
            transition.integrator.step_size = exp(
                adapt_state['smoothed_log_step_size'])
        else:
            transition.integrator.step_size = sum(
                exp(a['smoothed_log_step_size'])
                for a in adapt_state) / len(adapt_state)
