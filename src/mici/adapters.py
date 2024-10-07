"""Methods for adaptively setting algorithmic parameters of transitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import exp, log
from typing import TYPE_CHECKING

import numpy as np

from mici.errors import AdaptationError, IntegratorError
from mici.matrices import DensePositiveDefiniteMatrix, PositiveDiagonalMatrix

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from numpy.random import Generator
    from numpy.typing import ArrayLike

    from mici.integrators import Integrator
    from mici.states import ChainState
    from mici.systems import System
    from mici.transitions import Transition
    from mici.types import (
        AdaptationStatisticFunction,
        AdapterState,
        ReducerFunction,
        TransitionStatistics,
    )


class Adapter(ABC):
    """Abstract adapter for implementing schemes to adapt transition parameters.

    Adaptation schemes are assumed to be based on updating a collection of adaptation
    variables (collectively termed the adapter state here) after each chain transition
    based on the sampled chain state and/or statistics of the transition such as an
    acceptance probability statistic. After completing a chain of one or more adaptive
    transitions, the final adapter state may be used to perform a final update to the
    transition parameters.
    """

    @abstractmethod
    def initialize(
        self,
        chain_state: ChainState,
        transition: Transition,
    ) -> AdapterState:
        """Initialize adapter state prior to starting adaptive transitions.

        Args:
            chain_state: Initial chain state adaptive transition will be started from.
                May be used to calculate initial adapter state but should not be mutated
                by method.
            transition: Markov transition being adapted. Attributes of the transition or
                child objects may be updated in-place by the method.

        Returns:
            Initial adapter state.
        """

    @abstractmethod
    def update(
        self,
        adapt_state: AdapterState,
        chain_state: ChainState,
        trans_stats: TransitionStatistics,
        transition: Transition,
    ):
        """Update adapter state after sampling from transition being adapted.

        Args:
            adapt_state: Current adapter state. Entries will be updated in-place by the
                method.
            chain_state: Current chain state following sampling from transition being
                adapted. May be used to calculate adapter state updates but should not
                be mutated by method.
            trans_stats: Dictionary of statistics associated with transition being
                adapted. May be used to calculate adapter state updates but should not
                be mutated by method.
            transition: Markov transition being adapted. Attributes of the transition or
                child objects may be updated in-place by the method.
        """

    @abstractmethod
    def finalize(
        self,
        adapt_states: AdapterState | Iterable[AdapterState],
        chain_states: ChainState | Iterable[ChainState],
        transition: Transition,
        rngs: Generator | Iterable[Generator],
    ):
        """Update transition parameters based on final adapter state or states.

        Optionally, if multiple adapter states are available, e.g. from a set of
        independent adaptive chains, then these adaptation information from all the
        chains may be combined to set the transition parameter(s).

        Args:
            adapt_states: Final adapter state or a list of per chain adapter states.
                Arrays / buffers associated with the adapter state entries may be
                recycled to reduce memory usage - if so  the corresponding entries will
                be removed from the adapter state dictionary / dictionaries.
            chain_states: Final state of chain (or states of chains) in current
                sampling stage. May be updated in-place if transition parameters altered
                by adapter require updating any state components.
            transition: Markov transition being dapted. Attributes of the transition or
                child objects will be updated in-place by the method.
            rngs: Random number generator for the chain or a list of per-chain random
                number generators. Used to resample any components of states needing to
                be updated due to adaptation if required.
        """

    @property
    @abstractmethod
    def is_fast(self) -> bool:
        """Whether the adapter is 'fast' or 'slow'.

        An adapter which requires only local information to adapt the transition
        parameters should be classified as fast while one which requires more global
        information and so more chain iterations should be classified as slow i.e.
        :code:`is_fast == False`.
        """


def arithmetic_mean_log_step_size_reducer(log_step_sizes: Collection[float]) -> float:
    """Compute arithmetic mean of step sizes from their logs.

    Args:
        log_step_sizes: Logarithms of per-chain estimated step sizes.

    Returns:
        Arithmetic mean of estimated step sizes.
    """
    return sum(exp(x) for x in log_step_sizes) / len(log_step_sizes)


def geometric_mean_log_step_size_reducer(log_step_sizes: Collection[float]) -> float:
    """Compute geometric mean of step sizes from their logs.

    Args:
        log_step_sizes: Logarithms of per-chain estimated step sizes.

    Returns:
        Geometric mean of estimated step sizes.
    """
    return exp(sum(x for x in log_step_sizes) / len(log_step_sizes))


def min_log_step_size_reducer(log_step_sizes: Collection[float]) -> float:
    """Compute minimum of step sizes from their logs.

    Args:
        log_step_sizes: Logarithms of per-chain estimated step sizes.

    Returns:
        Minimum of estimated step sizes.
    """
    return exp(min(log_step_sizes))


def default_adapt_stat_func(stats: TransitionStatistics) -> float:
    """Function to extract default statistic used for step-size adaptation.

    Args:
        stats: Dictionary of transition statistics.

    Returns:
        Acceptance statistic.
    """
    return stats["accept_stat"]


class DualAveragingStepSizeAdapter(Adapter):
    """Dual averaging integrator step size adapter.

    Implementation of the dual algorithm step size adaptation algorithm described in
    Hoffman and Gelman (2014), a modified version of the stochastic optimisation scheme
    of Nesterov (2009). By default the adaptation is performed to control the
    :code:`accept_stat` statistic of an integration transition to be close to a target
    value but the statistic adapted on can be altered by changing the
    :code:`adapt_stat_func`.

    References:
      1. Hoffman, M.D. and Gelman, A. (2014). The No-U-turn sampler: adaptively setting
         path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research,
         15(1), pp.1593-1623.
      2. Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems.
         Mathematical programming 120(1), pp.221-259.
    """

    is_fast = True

    def __init__(
        self,
        adapt_stat_target: float = 0.8,
        adapt_stat_func: AdaptationStatisticFunction | None = None,
        log_step_size_reg_target: float | None = None,
        log_step_size_reg_coefficient: float = 0.05,
        iter_decay_coeff: float = 0.75,
        iter_offset: int = 10,
        max_init_step_size_iters: int = 100,
        log_step_size_reducer: ReducerFunction | None = None,
    ):
        """
        Args:
            adapt_stat_target: Target value for the transition statistic
                being controlled during adaptation.
            adapt_stat_func: Function which given a dictionary of transition statistics
                outputs the value of the statistic to control during adaptation. By
                default this is set to a function which simply selects the
                :code:'accept_stat' value in the statistics dictionary.
            log_step_size_reg_target: Value to regularize the controlled output
                (logarithm of the integrator step size) towards. If :code:`None` set to
                :code:`log(10 * init_step_size)` where :code:`init_step_size` is the
                initial 'reasonable' step size found by a coarse search as recommended
                in Hoffman and Gelman (2014). This has the effect of giving the dual
                averaging algorithm a tendency towards testing step sizes larger than
                the initial value, with typically integrating with a larger step size
                having a lower computational cost.
            log_step_size_reg_coefficient: Coefficient controlling amount of
                regularisation of controlled output (logarithm of the integrator step
                size) towards :code:`log_step_size_reg_target`. Defaults to 0.05 as
                recommended in Hoffman and Gelman (2014).
            iter_decay_coeff: Coefficient controlling exponent of decay in schedule
                weighting stochastic updates to smoothed log step size estimate. Should
                be in the interval (0.5, 1] to ensure asymptotic convergence of
                adaptation. A value of 1 gives equal weight to the whole history of
                updates while setting to a smaller value increasingly highly weights
                recent updates, giving a tendency to 'forget' early updates.
                Defaults to 0.75 as recommended in Hoffman and Gelman (2014).
            iter_offset: Offset used for the iteration based weighting of the adaptation
                statistic error estimate. Should be set to a non-negative value. A value
                > 0 has the effect of stabilising early iterations. Defaults to the
                value of 10 as recommended in Hoffman and Gelman (2014).
            max_init_step_size_iters: Maximum number of iterations to use in initial
                search for a reasonable step size with an
                :py:exc:`mici.errors.AdaptationError` exception raised if a suitable
                step size is not found within this many iterations.
            log_step_size_reducer: Reduction to apply to final per-chain step sizes
                estimates to produce overall integrator step size for main chain stages.
                The specified function should accept a sequence of logarithms of
                estimated step sizes and output a non-negative step size to use. If
                :code:`None`, the default, a function which computes the arithmetic mean
                of the per-chain step sizes is used.
        """
        self.adapt_stat_target = adapt_stat_target
        self.adapt_stat_func = (
            default_adapt_stat_func if adapt_stat_func is None else adapt_stat_func
        )
        self.log_step_size_reg_target = log_step_size_reg_target
        self.log_step_size_reg_coefficient = log_step_size_reg_coefficient
        self.iter_decay_coeff = iter_decay_coeff
        self.iter_offset = iter_offset
        self.max_init_step_size_iters = max_init_step_size_iters
        self.log_step_size_reducer = (
            arithmetic_mean_log_step_size_reducer
            if log_step_size_reducer is None
            else log_step_size_reducer
        )

    def initialize(
        self,
        chain_state: ChainState,
        transition: Transition,
    ) -> AdapterState:
        integrator = transition.integrator
        system = transition.system
        adapt_state = {
            "iter": 0,
            "smoothed_log_step_size": 0.0,
            "adapt_stat_error": 0.0,
        }
        init_step_size = self._find_and_set_init_step_size(
            chain_state,
            system,
            integrator,
        )
        if self.log_step_size_reg_target is None:
            adapt_state["log_step_size_reg_target"] = log(10 * init_step_size)
        else:
            adapt_state["log_step_size_reg_target"] = self.log_step_size_reg_target
        return adapt_state

    def _find_and_set_init_step_size(
        self,
        state: ChainState,
        system: System,
        integrator: Integrator,
    ) -> float:
        """Find initial step size by coarse search using single step statistics.

        Adaptation of Algorithm 4 in Hoffman and Gelman (2014).

        Compared to the Hoffman and Gelman algorithm, this version makes two changes:

          1. The absolute value of the change in Hamiltonian over a step being larger or
             smaller than log(2) is used to determine whether the step size is too big
             or small as opposed to the value of the equivalent Metropolis accept
             probability being larger or smaller than 0.5. Although a negative change in
             the Hamiltonian over a step of magnitude more than log(2) will lead to an
             accept probability of 1 for the forward move, the corresponding reversed
             move will have an accept probability less than 0.5, and so a change in the
             Hamiltonian over a step of magnitude more than log(2) irrespective of the
             sign of the change is indicative of the minimum acceptance probability over
             both forward and reversed steps being less than 0.5.
          2. To allow for integrators for which an integrator step may fail due to e.g.
             a convergence error in an iterative solver, the step size is also
             considered to be too big if any of the step sizes tried in the search
             result in a failed integrator step, with in this case the step size always
             being decreased on subsequent steps irrespective of the initial Hamiltonian
             error, until a integrator step successfully completes and the absolute
             value of the change in Hamiltonian is below the threshold of log(2)
             (corresponding to a minimum acceptance probability over forward and
             reversed steps of 0.5).
        """
        init_state = state.copy()
        h_init = system.h(init_state)
        if np.isnan(h_init):
            msg = "Hamiltonian evaluating to NaN at initial state."
            raise AdaptationError(msg)
        integrator.step_size = 1
        delta_h_threshold = log(2)
        for s in range(self.max_init_step_size_iters):
            try:
                state = integrator.step(init_state)
                delta_h = abs(h_init - system.h(state))
                if s == 0 or np.isnan(delta_h):
                    step_size_too_big = np.isnan(delta_h) or delta_h > delta_h_threshold
                if (step_size_too_big and delta_h <= delta_h_threshold) or (
                    not step_size_too_big and delta_h > delta_h_threshold
                ):
                    return integrator.step_size
                if step_size_too_big:
                    integrator.step_size /= 2
                else:
                    integrator.step_size *= 2
            except IntegratorError:  # noqa: PERF203
                step_size_too_big = True
                integrator.step_size /= 2
        msg = (
            f"Could not find reasonable initial step size in "
            f"{self.max_init_step_size_iters} iterations (final step size "
            f"{integrator.step_size}). A very large final step size may indicate that "
            f"the target distribution is improper such that the negative log density "
            f"is flat in one or more directions while a very small final step size may "
            f"indicate that the density function is insufficiently smooth at the point "
            f"initialized at."
        )
        raise AdaptationError(msg)

    def update(
        self,
        adapt_state: AdapterState,
        chain_state: ChainState,  # noqa: ARG002
        trans_stats: TransitionStatistics,
        transition: Transition,
    ):
        adapt_state["iter"] += 1
        error_weight = 1 / (self.iter_offset + adapt_state["iter"])
        adapt_state["adapt_stat_error"] *= 1 - error_weight
        adapt_state["adapt_stat_error"] += error_weight * (
            self.adapt_stat_target - self.adapt_stat_func(trans_stats)
        )
        smoothing_weight = (1 / adapt_state["iter"]) ** self.iter_decay_coeff
        log_step_size = adapt_state["log_step_size_reg_target"] - (
            adapt_state["adapt_stat_error"]
            * adapt_state["iter"] ** 0.5
            / self.log_step_size_reg_coefficient
        )
        adapt_state["smoothed_log_step_size"] *= 1 - smoothing_weight
        adapt_state["smoothed_log_step_size"] += smoothing_weight * log_step_size
        transition.integrator.step_size = exp(log_step_size)

    def finalize(
        self,
        adapt_states: AdapterState | Iterable[AdapterState],
        chain_states: ChainState | Iterable[ChainState],  # noqa: ARG002
        transition: Transition,
        rngs: Generator | Iterable[Generator],  # noqa: ARG002
    ):
        if isinstance(adapt_states, dict):
            transition.integrator.step_size = exp(
                adapt_states["smoothed_log_step_size"],
            )
        else:
            transition.integrator.step_size = self.log_step_size_reducer(
                [adapt_state["smoothed_log_step_size"] for adapt_state in adapt_states],
            )


class OnlineVarianceMetricAdapter(Adapter):
    """Diagonal metric adapter using online variance estimates.

    Uses Welford's algorithm (Welford, 1962) to stably compute an online estimate of the
    sample variances of the chain state position components during sampling. If online
    estimates are available from multiple independent chains, the final variance
    estimate is calculated from the per-chain statistics using the parallel / batched
    incremental variance algorithm described by Chan et al. (1979). The variance
    estimates are optionally regularized towards a common scalar value, with increasing
    weight for small number of samples, to decrease the effect of noisy estimates for
    small sample sizes, following the approach in Stan (Carpenter et al., 2017). The
    metric matrix representation is set to a diagonal matrix with diagonal elements
    corresponding to the reciprocal of the (regularized) variance estimates.

    References:
      1. Welford, B. P. (1962). Note on a method for calculating corrected sums of
         squares and products. Technometrics, 4(3), pp. 419-420.
      2. Chan, T. F., Golub, G. H. and LeVeque, R. J. (1979). Updating formulae and a
         pairwise algorithm for computing sample variances. Technical Report
         STAN-CS-79-773, Department of Computer Science, Stanford University.
      3. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B., Betancourt,
         M., Brubaker, M., Guo, J., Li, P. and Riddell, A.  (2017). Stan: A
         probabilistic programming language. Journal of Statistical Software, 76(1).
    """

    is_fast = False

    def __init__(self, reg_iter_offset: int = 5, reg_scale: float = 1e-3):
        """
        Args:
            reg_iter_offset: Iteration offset used for calculating iteration dependent
                weighting between regularisation target and current covariance estimate.
                Higher values cause stronger regularisation during initial iterations. A
                value of zero corresponds to no regularisation; this should only be used
                if the sample covariance is guaranteed to be positive definite.
            reg_scale: Positive scalar defining value variance estimates are regularized
                towards.
        """
        self.reg_iter_offset = reg_iter_offset
        self.reg_scale = reg_scale

    def initialize(
        self,
        chain_state: ChainState,
        transition: Transition,  # noqa: ARG002
    ) -> AdapterState:
        return {
            "iter": 0,
            "mean": np.zeros_like(chain_state.pos),
            "sum_diff_sq": np.zeros_like(chain_state.pos),
        }

    def update(
        self,
        adapt_state: AdapterState,
        chain_state: ChainState,
        trans_stats: TransitionStatistics,  # noqa: ARG002
        transition: Transition,  # noqa: ARG002
    ):
        # Use Welford (1962) incremental algorithm to update statistics to
        # calculate online variance estimate
        # https://en.wikipedia.org/wiki/
        #   Algorithms_for_calculating_variance#Welford's_online_algorithm
        adapt_state["iter"] += 1
        pos_minus_mean = chain_state.pos - adapt_state["mean"]
        adapt_state["mean"] += pos_minus_mean / adapt_state["iter"]
        adapt_state["sum_diff_sq"] += pos_minus_mean * (
            chain_state.pos - adapt_state["mean"]
        )

    def _regularize_var_est(self, var_est: ArrayLike, n_iter: int):
        """Update variance estimates by regularizing towards common scalar.

        Performed in place to prevent further array allocations.
        """
        if self.reg_iter_offset is not None and self.reg_iter_offset != 0:
            var_est *= n_iter / (self.reg_iter_offset + n_iter)
            var_est += self.reg_scale * (
                self.reg_iter_offset / (self.reg_iter_offset + n_iter)
            )

    def finalize(
        self,
        adapt_states: AdapterState | Iterable[AdapterState],
        chain_states: ChainState | Iterable[ChainState],
        transition: Transition,
        rngs: Generator | Iterable[Generator],
    ):
        if isinstance(adapt_states, dict):
            n_iter = adapt_states["iter"]
            var_est = adapt_states.pop("sum_diff_sq")
            chain_states = [chain_states]
            rngs = [rngs]
        else:
            # Use Chan et al. (1979) parallel variance estimation algorithm
            # to combine per-chain statistics
            # https://en.wikipedia.org/wiki/
            #    Algorithms_for_calculating_variance#Parallel_algorithm
            for i, adapt_state in enumerate(adapt_states):
                if i == 0:
                    n_iter = adapt_state["iter"]
                    mean_est = adapt_state.pop("mean")
                    var_est = adapt_state.pop("sum_diff_sq")
                else:
                    n_iter_prev = n_iter
                    n_iter += adapt_state["iter"]
                    mean_diff = mean_est - adapt_state["mean"]
                    mean_est *= n_iter_prev
                    mean_est += adapt_state["iter"] * adapt_state["mean"]
                    mean_est /= n_iter
                    var_est += adapt_state["sum_diff_sq"]
                    var_est += (
                        mean_diff**2 * (adapt_state["iter"] * n_iter_prev) / n_iter
                    )
        if n_iter < 2:  # noqa: PLR2004
            msg = "At least two chain samples required to compute a variance estimates."
            raise AdaptationError(msg)
        var_est /= n_iter - 1
        self._regularize_var_est(var_est, n_iter)
        transition.system.metric = PositiveDiagonalMatrix(var_est).inv
        # Resample momentum to account for altered distribution due to new metric
        for chain_state, rng in zip(chain_states, rngs, strict=True):
            chain_state.mom = transition.system.sample_momentum(chain_state, rng)


class OnlineCovarianceMetricAdapter(Adapter):
    """Dense metric adapter using online covariance estimates.

    Uses Welford's algorithm (Welford, 1962) to stably compute an online estimate of the
    sample covariance matrix of the chain state position components during sampling. If
    online estimates are available from multiple independent chains, the final
    covariance matrix estimate is calculated from the per-chain statistics using a
    covariance variant due to Schubert and Gertz (2018) of the parallel / batched
    incremental variance algorithm described by Chan et al. (1979). The covariance
    matrix estimates are optionally regularized towards a scaled identity matrix, with
    increasing weight for small number of samples, to decrease the effect of noisy
    estimates for small sample sizes, following the approach in Stan (Carpenter et al.,
    2017). The metric matrix representation is set to a dense positive definite matrix
    corresponding to the inverse of the (regularized) covariance matrix estimate.


    References:
      1. Welford, B. P. (1962). Note on a method for calculating corrected sums of
         squares and products. Technometrics, 4(3), pp. 419-420.
      2. Schubert, E. and Gertz, M. (2018). Numerically stable parallel computation of
         (co-)variance. ACM. p. 10. doi:10.1145/3221269.3223036.
      3. Chan, T. F., Golub, G. H. and LeVeque, R. J. (1979). Updating formulae and a
         pairwise algorithm for computing sample variances. Technical Report
         STAN-CS-79-773, Department of Computer Science, Stanford University.
      4. Carpenter, B., Gelman, A., Hoffman, M.D., Lee, D., Goodrich, B., Betancourt,
         M., Brubaker, M., Guo, J., Li, P. and Riddell, A.  (2017). Stan: A
         probabilistic programming language. Journal of Statistical Software, 76(1).
    """

    is_fast = False

    def __init__(self, reg_iter_offset: int = 5, reg_scale: float = 1e-3):
        """
        Args:
            reg_iter_offset: Iteration offset used for calculating iteration
                dependent weighting between regularisation target and current covariance
                estimate. Higher values cause stronger regularisation during initial
                iterations.
            reg_scale: Positive scalar defining value variance estimates are
                regularized towards.
        """
        self.reg_iter_offset = reg_iter_offset
        self.reg_scale = reg_scale

    def initialize(
        self,
        chain_state: ChainState,
        transition: Transition,  # noqa: ARG002
    ) -> AdapterState:
        dim_pos = chain_state.pos.shape[0]
        dtype = chain_state.pos.dtype
        return {
            "iter": 0,
            "mean": np.zeros(shape=(dim_pos,), dtype=dtype),
            "sum_diff_outer": np.zeros(shape=(dim_pos, dim_pos), dtype=dtype),
        }

    def update(
        self,
        adapt_state: AdapterState,
        chain_state: ChainState,
        trans_stats: TransitionStatistics,  # noqa: ARG002
        transition: Transition,  # noqa: ARG002
    ):
        # Use Welford (1962) incremental algorithm to update statistics to
        # calculate online covariance estimate
        # https://en.wikipedia.org/wiki/
        #  Algorithms_for_calculating_variance#Online
        adapt_state["iter"] += 1
        pos_minus_mean = chain_state.pos - adapt_state["mean"]
        adapt_state["mean"] += pos_minus_mean / adapt_state["iter"]
        adapt_state["sum_diff_outer"] += (
            pos_minus_mean[None, :] * (chain_state.pos - adapt_state["mean"])[:, None]
        )

    def _regularize_covar_est(self, covar_est: ArrayLike, n_iter: int):
        """Update covariance estimate by regularising towards identity.

        Performed in place to prevent further array allocations.
        """
        covar_est *= n_iter / (self.reg_iter_offset + n_iter)
        covar_est_diagonal = np.einsum("ii->i", covar_est)
        covar_est_diagonal += self.reg_scale * (
            self.reg_iter_offset / (self.reg_iter_offset + n_iter)
        )

    def finalize(
        self,
        adapt_states: AdapterState | Iterable[AdapterState],
        chain_states: ChainState | Iterable[ChainState],
        transition: Transition,
        rngs: Generator | Iterable[Generator],
    ):
        if isinstance(adapt_states, dict):
            n_iter = adapt_states["iter"]
            covar_est = adapt_states.pop("sum_diff_outer")
            chain_states = [chain_states]
            rngs = [rngs]
        else:
            # Use Schubert and Gertz (2018) parallel covariance estimation
            # algorithm to combine per-chain statistics
            for i, adapt_state in enumerate(adapt_states):
                if i == 0:
                    n_iter = adapt_state["iter"]
                    mean_est = adapt_state.pop("mean")
                    covar_est = adapt_state.pop("sum_diff_outer")
                else:
                    n_iter_prev = n_iter
                    n_iter += adapt_state["iter"]
                    mean_diff = mean_est - adapt_state["mean"]
                    mean_est *= n_iter_prev
                    mean_est += adapt_state["iter"] * adapt_state["mean"]
                    mean_est /= n_iter
                    covar_est += adapt_state["sum_diff_outer"]
                    covar_est += (
                        np.outer(mean_diff, mean_diff)
                        * (adapt_state["iter"] * n_iter_prev)
                        / n_iter
                    )
        if n_iter < 2:  # noqa: PLR2004
            msg = "At least two chain samples required to compute a variance estimates."
            raise AdaptationError(msg)
        covar_est /= n_iter - 1
        self._regularize_covar_est(covar_est, n_iter)
        transition.system.metric = DensePositiveDefiniteMatrix(covar_est).inv
        # Resample momentum to account for altered distribution due to new metric
        for chain_state, rng in zip(chain_states, rngs, strict=True):
            chain_state.mom = transition.system.sample_momentum(chain_state, rng)
