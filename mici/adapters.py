"""Methods for adaptively setting algorithmic parameters of transitions."""

from abc import ABC, abstractmethod, abstractproperty
from math import exp, log
import numpy as np
from mici.errors import IntegratorError, AdaptationError
from mici.matrices import (
    PositiveDiagonalMatrix, DensePositiveDefiniteMatrix)


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
                state or a list of adapter states. Arrays / buffers associated
                with the adapter state entries may be recycled to reduce memory
                usage - if so the corresponding entries will be removed from
                the adapter state dictionary / dictionaries.
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
            sign = 2 * (delta_h > -log(2)) - 1
        except IntegratorError:
            sign = 1
        for s in range(self.max_init_step_size_iters):
            try:
                state = integrator.step(init_state)
                delta_h = h_init - system.h(state)
                if sign * delta_h < -sign * log(2):
                    return integrator.step_size
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


class WelfordVarianceMetricAdapter(Adapter):
    """Metric adapter using variance estimates and optional regularisation.

    Uses Welford's algorithm [1] to stably compute an online estimate of the
    sample variances of the chain state position components during sampling. The
    variance estimates are optionally regularised towards a common scalar value,
    with increasing weight for small number of samples, to decrease the effect
    of noisy estimates for small sample sizes, following the approach in Stan
    [2]. The metric matrix representation is set to a diagonal matrix with
    diagonal elements corresponding to the reciprocal of the (regularised)
    variance estimates.

    References:

      1. Welford, B. P., 1962. Note on a method for calculating corrected sums
         of squares and products. Technometrics, 4(3), pp. 419–420.
      2. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B.,
         Betancourt, M., Brubaker, M., Guo, J., Li, P. and Riddell, A., 2017.
         Stan: A probabilistic programming language. Journal of Statistical
         Software, 76(1).
    """

    def __init__(self, reg_iter_offset=5, reg_scale=1e-3):
        """
        Args:
            reg_iter_offset (int): Iteration offset used for calculating
                iteration dependent weighting between regularisation target and
                current covariance estimate. Higher values cause stronger
                regularisation during initial iterations.
            reg_scale (float): Positive scalar defining value variance estimates
                are regularised towards.
        """
        self.reg_iter_offset = reg_iter_offset
        self.reg_scale = reg_scale

    def initialise(self, chain_state, transition):
        return {
            'iter': 0,
            'mean': np.zeros_like(chain_state.pos),
            'sum_diff_sq': np.zeros_like(chain_state.pos)
        }

    def update(self, chain_state, adapt_state, trans_stats, transition):
        adapt_state['iter'] += 1
        pos_minus_mean = chain_state.pos - adapt_state['mean']
        adapt_state['mean'] += pos_minus_mean / adapt_state['iter']
        adapt_state['sum_diff_sq'] += pos_minus_mean * (
            chain_state.pos - adapt_state['mean'])

    def _regularise_var_est(self, var_est, n_iter):
        """Update variance estimates by regularising towards common scalar.

        Performed in place to prevent further array allocations.
        """
        if self.reg_iter_offset is not None and self.reg_iter_offset != 0:
            var_est *= n_iter / (self.reg_iter_offset + n_iter)
            var_est += self.reg_scale * (
                self.reg_iter_offset / (self.reg_iter_offset + n_iter))

    def finalise(self, adapt_state, transition):
        if isinstance(adapt_state, dict):
            n_iter = adapt_state['iter']
            var_est = adapt_state.pop('sum_diff_sq')
            var_est /= (n_iter - 1)
        else:
            # Use pooled variance estimator
            # https://en.wikipedia.org/wiki/Pooled_variance
            for i, a in enumerate(adapt_state):
                if i == 0:
                    n_iter = a['iter']
                    n_state = 1
                    var_est = a.pop('sum_diff_sq')
                else:
                    n_iter += a['iter']
                    n_state += 1
                    var_est += a.pop('sum_diff_sq')
            var_est /= (n_iter - n_state)
        self._regularise_var_est(var_est, n_iter)
        transition.system.metric = PositiveDiagonalMatrix(var_est).inv


class WelfordCovarianceMetricAdapter(Adapter):
    """Metric adapter using variance estimates and optional regularisation.

    Uses Welford's algorithm [1] to stably compute an online estimate of the
    sample covariances of the chain state position components during sampling.
    The covariance matrix estimate is optionally regularised towards a scaled
    identity matrix, with increasing weight for small number of samples, to
    decrease the effect of noisy estimates for small sample sizes, following the
    approach in Stan [2]. The metric matrix representation is set to a dense
    positive definite matrix corresponding to the inverse of the (regularised)
    covariance estimate.

    References:

      1. Welford, B. P., 1962. Note on a method for calculating corrected sums
         of squares and products. Technometrics, 4(3), pp. 419–420.
      2. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B.,
         Betancourt, M., Brubaker, M., Guo, J., Li, P. and Riddell, A., 2017.
         Stan: A probabilistic programming language. Journal of Statistical
         Software, 76(1).
    """

    def __init__(self, reg_iter_offset=5, reg_coefficient=1e-3):
        """
        Args:
            reg_iter_offset (int): Iteration offset used for calculating
                iteration dependent weighting between regularisation target and
                current covariance estimate. Higher values cause stronger
                regularisation during initial iterations.
            reg_scale (float): Positive scalar defining value variance estimates
                are regularised towards.
        """
        self.reg_iter_offset = reg_iter_offset
        self.reg_coefficient = reg_coefficient

    def initialise(self, chain_state, transition):
        dim_pos = chain_state.pos.shape[0]
        dtype = chain_state.pos.dtype
        return {
            'iter': 0,
            'mean': np.zeros(shape=(dim_pos,), dtype=dtype),
            'sum_diff_outer': np.zeros(shape=(dim_pos, dim_pos), dtype=dtype)
        }

    def update(self, chain_state, adapt_state, trans_stats, transition):
        adapt_state['iter'] += 1
        pos_minus_mean = chain_state.pos - adapt_state['mean']
        adapt_state['mean'] += pos_minus_mean / adapt_state['iter']
        adapt_state['sum_diff_outer'] += pos_minus_mean[None, :] * (
            chain_state.pos - adapt_state['mean'])[:, None]

    def _regularise_covar_est(self, covar_est, n_iter):
        """Update covariance estimate by regularising towards identity.

        Performed in place to prevent further array allocations.
        """
        covar_est *= (n_iter / (self.reg_iter_offset + n_iter))
        covar_est_diagonal = np.einsum('ii->i', covar_est)
        covar_est_diagonal += self.reg_coefficient * (
            self.reg_iter_offset / (self.reg_iter_offset + n_iter))

    def finalise(self, adapt_state, transition):
        if isinstance(adapt_state, dict):
            n_iter = adapt_state['iter']
            covar_est = adapt_state.pop('sum_diff_outer')
            covar_est /= (n_iter - 1)
        else:
            # Use pooled covariance estimator
            # https://en.wikipedia.org/wiki/Pooled_covariance_matrix
            for i, a in enumerate(adapt_state):
                if i == 0:
                    n_iter = a['iter']
                    n_state = 1
                    covar_est = a.pop('sum_diff_outer')
                else:
                    n_iter += a['iter']
                    n_state += 1
                    covar_est += a.pop('sum_diff_outer')
            covar_est /= (n_iter - n_state)
        self._regularise_covar_est(covar_est, n_iter)
        transition.system.metric = DensePositiveDefiniteMatrix(covar_est).inv

