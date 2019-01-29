"""Monte Carlo sampler classes for peforming inference."""

import logging
import signal
from collections import OrderedDict
import numpy as np
import hmc
import hmc.transitions as trans
from hmc.states import ChainState, HamiltonianState

try:
    import tqdm.auto as tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
try:
    import arviz
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
try:
    import randomgen
    RANDOMGEN_AVAILABLE = True
except ImportError:
    RANDOMGEN_AVAILABLE = False
# Preferentially import Pool from multiprocess library if available as able
# to serialise much wider range of types including autograd functions
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool


logger = logging.getLogger(__name__)


def extract_pos(state):
    """Helper function to extract position from chain state."""
    return state.pos


def extract_mom(state):
    """Helper function to extract momentum from chain state."""
    return state.mom


def _ignore_sigint_initialiser():
    """Initialiser for multi-process workers to force ignoring SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MarkovChainMonteCarloMethod(object):
    """Generic Markov chain Monte Carlo (MCMC) sampler.

    Generates a Markov chain from some initial state by iteratively applying
    a sequence of Markov transition operators.
    """

    def __init__(self, rng, transitions):
        """
        Args:
            rng: Numpy RandomState random number generator instance.
            transitions: Ordered dictionary of Markov chain transitions to
                sequentially sample from on each chain iteration.
        """
        self.rng = rng
        self.transitions = transitions

    def _init_chain_stats(self, n_sample, n_chain=None):
        shape = (n_sample,) if n_chain is None else (n_chain, n_sample)
        return {
            trans_key:
            {key: np.full(shape, val, dtype) for key, (dtype, val)
             in transition.statistic_types.items()}
            for trans_key, transition in self.transitions.items()
            if hasattr(transition, 'statistic_types')
        }

    def _sample_chain(self, rng, n_sample, init_state, chain_var_funcs=None,
                      tqdm_desc='Sampling', tqdm_position=None):
        if not isinstance(init_state, ChainState):
            state = ChainState(init_state)
        else:
            state = init_state
        chain_stats = self._init_chain_stats(n_sample)
        chains = {}
        if TQDM_AVAILABLE:
            sample_range = tqdm.trange(
                n_sample, desc=tqdm_desc, unit='it', dynamic_ncols=True,
                position=tqdm_position)
        else:
            sample_range = range(n_sample)
        try:
            for sample_index in sample_range:
                for trans_key, transition in self.transitions.items():
                    state, trans_stats = transition.sample(state, rng)
                    if trans_stats is not None:
                        if trans_key not in chain_stats:
                            raise RuntimeError(
                                f'Transition {trans_key} returned statistics '
                                f'but does not have a `statistic_types` '
                                f'attribute')
                        for key, value in trans_stats.items():
                            if key in chain_stats[trans_key]:
                                chain_stats[trans_key][key][sample_index] = (
                                    trans_stats[key])
                for key, chain_func in chain_var_funcs.items():
                    var = chain_func(state)
                    if sample_index == 0:
                        chains[key] = np.full((n_sample,) + var.shape, np.nan)
                    chains[key][sample_index] = var
        except KeyboardInterrupt:
            return chains, chain_stats, sample_index
        return chains, chain_stats, sample_index + 1

    def sample_chain(self, n_sample, init_state, chain_var_funcs):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_sample (int): Number of samples (iterations) to draw per chain.
            init_state (ChainState or array):
               Initial chain state. Can be either an array specifying the state
               or a `ChainState` instance.
            chain_var_funcs (dict[str, callable]): Dictionary of functions
               which compute the chain variables to be recorded at each
               iteration, with each function being passed the current state
               and returning an array corresponding to the variable(s) to be
               stored. The keys to the functions are used to index the chain
               variable arrays in the returned data.

        Returns:
            chains (dict[str, array]):
                Chain variable arrays, with one entry per function in
                `chain_var_funcs` with the same key. Each entry consists of an
                arrays with the leading dimension of the arrays corresponding
                to the sampling (draw) index.
            chain_stats (dict[str, dict[str, array]]):
                Dictionary of chain transition statistics. Outer dictionary
                contains entries for each chain transition which returns
                statistics (e.g. acceptance probabilities) on each iteration.
                For each such transition, a dictionary is returned with string
                keys describing the statistics recorded and array values with
                the leading dimension corresponding to the sampling index.
        """
        chains, chain_stats, sample_index = self._sample_chain(
            self.rng, n_sample, init_state, chain_var_funcs)
        if sample_index != n_sample:
            logger.exception(
                f'Sampling manually interrupted at iteration {sample_index}. '
                f'Arrays containing chain variables and statistics computed '
                f'before interruption will be returned, all entries for '
                f'iteration {sample_index} and above should be ignored.')
        return chains, chain_stats

    def _collate_chain_outputs(
            self, n_sample, chains_stats_and_sample_indices):
        chains_stack = {}
        n_chain = len(chains_stats_and_sample_indices)
        chain_stats_stack = self._init_chain_stats(n_sample, n_chain)
        for chain_index, (chains, chain_stats, sample_index) in enumerate(
                chains_stats_and_sample_indices):
            for key, val in chains.items():
                if chain_index == 0:
                    chains_stack[key] = [val]
                else:
                    chains_stack[key].append(val)
            for trans_key, trans_stats in chain_stats.items():
                for key, val in chain_stats[trans_key].items():
                    chain_stats_stack[trans_key][key][chain_index] = val
        for key, val in chains_stack.items():
            chains_stack[key] = np.stack(val, axis=0)
        return chains_stack, chain_stats_stack

    def sample_chains(self, n_sample, init_states, chain_var_funcs,
                      n_process=1):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_sample (int): Number of samples (iterations) to draw per chain.
            init_states (Iterable[ChainState] or Iterable[array]):
                Initial chain states. Each entry can be either an array
                specifying the state or a `ChainState` instance. One chain will
                be run for each state in the iterable sequence.
            chain_var_funcs (dict[str, callable]): Dictionary of functions
                which compute the chain variables to be recorded at each
                iteration, with each function being passed the current state
                and returning an array corresponding to the variable(s) to be
                stored. The keys to the functions are used to index the chain
                variable arrays in the returned data.
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If
                set to `None` then the number of processes will default to the
                output of `os.cpu_count()`.

        Returns:
            chains (dict[str, list[array]]):
                Chain variable array lists, with one entry per function in
                `chain_var_funcs` with the same key. Each entry consists of a
                list of arrays, one per chain, with the leading dimension of
                the arrays corresponding to the sampling (draw) index.
            chain_stats (dict[str, dict[str, list[array]]]):
                Dictionary of chain transition statistics. Outer dictionary
                contains entries for each chain transition which returns
                statistics (e.g. acceptance probabilities) on each iteration.
                For each such transition, a dictionary is returned with string
                keys describing the statistics recorded and list of array
                values with one array per chain and the leading dimension of
                the arrays corresponding to the sampling index.
        """
        n_chain = len(init_states)
        if RANDOMGEN_AVAILABLE:
            seed = self.rng.randint(2**64, dtype='uint64')
            rngs = [randomgen.Xorshift1024(seed).jump(i).generator
                    for i in range(n_chain)]
        else:
            seeds = (self.rng.choice(2**16, n_chain, False) * 2**16 +
                     self.rng.choice(2**16, n_chain, False))
            rngs = [np.random.RandomState(seed) for seed in seeds]
        if n_process == 1:
            chain_outputs = []
            for c, (rng, init_state) in enumerate(zip(rngs, init_states)):
                chains, chain_stats, n_sample_chain = self._sample_chain(
                    rng, n_sample, init_state, chain_var_funcs, f'Chain {c}')
                chain_outputs.append((chains, chain_stats, n_sample_chain))
                if n_sample_chain != n_sample:
                    logger.error(
                        f'Sampling manually interrupted at chain {c} iteration'
                        f' {n_sample_chain}. Arrays containing chain variables'
                        f' and statistics computed before interruption will'
                        f' be returned, all entries for iteration '
                        f' {n_sample_chain} and above of chain {c} should be'
                        f' ignored.')
                    break
        else:
            with Pool(n_process, _ignore_sigint_initialiser) as pool:
                try:
                    chain_outputs = pool.starmap(
                        self._sample_chain,
                        zip(rngs,
                            [n_sample] * n_chain,
                            init_states,
                            [chain_var_funcs] * n_chain,
                            [f'Chain {i}' for i in range(n_chain)],
                            range(n_chain)))
                except KeyboardInterrupt:
                    pool.terminate()
                    pool.join()
                    raise
        return self._collate_chain_outputs(n_sample, chain_outputs)

    if ARVIZ_AVAILABLE:

        def sample_chains_arviz(
                self, n_sample, init_states, chain_var_funcs=None,
                n_process=1, sample_stats_key=None):
            """Sample one or more Markov chains from given initial states.

            Performs a specified number of chain iterations (each of which may
            be composed of multiple individual Markov transitions), recording
            the outputs of functions of the sampled chain state after each
            iteration. The chains may be run in parallel across multiple
            independent processes or sequentially. Chain data is returned in an
            `arviz.InferenceData` container object.

            Args:
                n_sample (int): Number of samples to draw per chain.
                init_states (Iterable[ChainState] or Iterable[array]):
                    Initial chain states. Each entry can be either an array
                    specifying the state or a `ChainState` instance. One chain
                    will be run for each state in the iterable sequence.
                chain_var_funcs (dict[str, callable]): Dictionary of functions
                    which compute the chain variables to be recorded at each
                    iteration, with each function being passed the current
                    state and returning an array corresponding to the
                    variable(s) to be stored. The keys to the functions are
                    used to index the chain variable arrays in the returned
                    data.
                n_process (int or None): Number of parallel processes to run
                    chains over. If set to one then chains will be run
                    sequentially in otherwise a `multiprocessing.Pool` object
                    will be used to dynamically assign the chains across
                    multiple processes. If set to `None` then the number of
                    processes will default to the output of `os.cpu_count()`.
                sample_stats_key (str): Key of transition to use the
                    recorded statistics of to populate the `sampling_stats`
                    group in the returned `InferenceData` object.

            Returns:
                arvix.InferenceData:
                    An arviz data container with groups `posterior` and
                    'sample_stats', both of instances of `xarray.Dataset`.
                    The `posterior` group corresponds to the chain variable
                    samples computed using the `chain_var_funcs` entries (with
                    the data variable keys corresponding to the keys there).
                    The `sample_stats` group corresponds to the statistics of
                    the transition indicated by the `sample_stats_key`
                    argument.
            """
            if (sample_stats_key is not None and
                    sample_stats_key not in self.transitions):
                raise ValueError(
                    f'Specified `sample_stats_key` ({sample_stats_key}) does '
                    f'not match any transition.')
            chains, chain_stats = self.sample_chains(
                n_sample, init_states, chain_var_funcs, n_process)
            if sample_stats_key is None:
                return arviz.InferenceData(
                    posterior=arviz.dict_to_dataset(chains, library=hmc))
            else:
                return arviz.InferenceData(
                    posterior=arviz.dict_to_dataset(chains, library=hmc),
                    sample_stats=arviz.dict_to_dataset(
                        chain_stats[sample_stats_key], library=hmc))


class HamiltonianMonteCarlo(MarkovChainMonteCarloMethod):
    """Wrapper class for Hamiltonian Monte Carlo (HMC) methods.

    Here HMC is defined as a MCMC method which augments the original target
    variable (henceforth position variable) with a momentum variable with a
    user specified conditional distribution given the position variable. In
    each chain iteration two Markov transitions leaving the resulting joint
    distribution on position and momentum variables invariant are applied -
    the momentum variables are updated in a transition which leaves their
    conditional distribution invariant (momentum transition) and then a
    trajectory in the joint space is generated by numerically integrating a
    Hamiltonian dynamic with an appropriate symplectic integrator which is
    exactly reversible, volume preserving and approximately conserves the joint
    probability density of the target-momentum state pair. One state from the
    resulting trajectory is then selected as the next joint chain state using
    an appropriate sampling scheme such that the joint distribution is left
    exactly invariant (integration transition).

    There are various options available for both the momentum transition and
    integration transition, with by default the momentum transition set to be
    independent resampling of the momentum variables from their conditional
    distribution.

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, rng, integration_transition,
                 momentum_transition=None):
        """
        Args:
            system: Hamiltonian system to be simulated.
            rng: Numpy RandomState random number generator instance.
            integration_transition: Markov transition operator which jointly
                updates the position and momentum components of the chain
                state by integrating the Hamiltonian dynamics of the system
                to propose new values for the state.
            momentum_transition: Markov transition operator which updates only
                the momentum component of the chain state. If set to `None` a
                transitiion operator which independently samples the momentum
                from its conditional distribution will be used.
        """
        self.system = system
        self.rng = rng
        if momentum_transition is None:
            momentum_transition = trans.IndependentMomentumTransition(system)
        super().__init__(rng, OrderedDict(
            momentum_transition=momentum_transition,
            integration_transition=integration_transition))

    def _sample_chain(self, rng, n_sample, init_state, chain_var_funcs=None,
                      tqdm_desc='Sampling', tqdm_position=None):
        if not isinstance(init_state, HamiltonianState):
            state = HamiltonianState(init_state)
        else:
            state = init_state
        if state.mom is None:
            state.mom = self.system.sample_momentum(state, self.rng)
        if chain_var_funcs is None:
            chain_var_funcs = {'pos': extract_pos}
        return super()._sample_chain(rng, n_sample, init_state,
                                     chain_var_funcs, tqdm_desc, tqdm_position)

    def sample_chain(self, n_sample, init_state, chain_var_funcs=None):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_sample (int): Number of samples (iterations) to draw per chain.
            init_states (HamiltonianState or array):
               Initial chain state. The state can be either an array specifying
               the state position component or a `HamiltonianState` instance.
               If an array is passed or the `mom` attribute of the state is not
               set, a momentum component will be independently sampled from its
               conditional distribution.
            chain_var_funcs (dict[str, callable]): Dictionary of functions
               which compute the chain variables to be recorded at each
               iteration, with each function being passed the current state
               and returning an array corresponding to the variable(s) to be
               stored. By default (or if set to `None`) a single function which
               returns the position component of the state is used. The keys
               to the functions are used to index the chain variable arrays in
               the returned data.

        Returns:
            chains (dict[str, array]):
                Chain variable arrays, with one entry per function in
                `chain_var_funcs` with the same key. Each entry consists of an
                arrays with the leading dimension of the arrays corresponding
                to the sampling (draw) index.
            chain_stats (dict[str, dict[str, array]]):
                Dictionary of chain transition statistics. Outer dictionary
                contains entries for each chain transition which returns
                statistics (e.g. acceptance probabilities) on each iteration.
                For each such transition, a dictionary is returned with string
                keys describing the statistics recorded and list of array
                values with one array per chain and the leading dimension of
                the arrays corresponding to the sampling index.
        """
        chains, chain_stats = super().sample_chain(
            n_sample, init_state, chain_var_funcs)
        return chains, chain_stats.get('integration_transition', {})

    def sample_chains(self, n_sample, init_states, chain_var_funcs=None,
                      n_process=1):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_sample (int): Number of samples (iterations) to draw per chain.
            init_states (Iterable[HamiltonianState] or Iterable[array]):
               Initial chain states. Each state can be either an array
               specifying the state position component or a `HamiltonianState`
               instance. If an array is passed or the `mom` attribute of the
               state is not set, a momentum component will be independently
               sampled from its conditional distribution. One chain will be run
               for each state in the iterable sequence.
            chain_var_funcs (dict[str, callable]): Dictionary of functions
               which compute the chain variables to be recorded at each
               iteration, with each function being passed the current state
               and returning an array corresponding to the variable(s) to be
               stored. By default (or if set to `None`) a single function which
               returns the position component of the state is used. The keys
               to the functions are used to index the chain variable arrays in
               the returned data.
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will default to the
                output of `os.cpu_count()`.

        Returns:
            chains (dict[str, list[array]]):
                Chain variable array lists, with one entry per function in
                `chain_var_funcs` with the same key. Each entry consists of a
                list of arrays (one per chain) with the leading dimension of
                the arrays corresponding to the sampling (draw) index.
            chain_stats (dict[str, dict[str, list[array]]]):
                Dictionary of chain transition statistics. Outer dictionary
                contains entries for each chain transition which returns
                statistics (e.g. acceptance probabilities) on each iteration.
                For each such transition, a dictionary is returned with string
                keys describing the statistics recorded and list of array
                values with one array per chain and the leading dimension of
                the arrays corresponding to the sampling index.
        """
        chains, chain_stats = super().sample_chains(
            n_sample, init_states, chain_var_funcs, n_process)
        return chains, chain_stats.get('integration_transition', {})

    if ARVIZ_AVAILABLE:

        def sample_chains_arviz(self, n_sample, init_states,
                                chain_var_funcs=None, n_process=1):
            """Sample one or more Markov chains from given initial states.

            Performs a specified number of chain iterations (each of which may
            be composed of multiple individual Markov transitions), recording
            the outputs of functions of the sampled chain state after each
            iteration. The chains may be run in parallel across multiple
            independent processes or sequentially. Chain data is returned in an
            `arviz.InferenceData` container object.

            Args:
                n_sample (int): Number of samples to draw per chain.
                init_states (Iterable[HamiltonianState] or Iterable[array]):
                   Initial chain states. Each state can be either an array
                   specifying the state position component or a
                   `HamiltonianState` instance. If an array is passed or the
                   `mom` attribute of the state is not set, a momentum
                   component will be independently sampled from its conditional
                   distribution. One chain will be run for each state in the
                   iterable sequence.
                chain_var_funcs (dict[str, callable]): Dictionary of functions
                   which compute the chain variables to be recorded at each
                   iteration, with each function being passed the current state
                   and returning an array corresponding to the variable(s) to
                   be stored. By default (or if set to `None`) a single
                   function which returns the position component of the state
                   is used. The keys to the functions are used to index the
                   chain variable arrays in the returned data.
                n_process (int or None): Number of parallel processes to run
                    chains over. If set to one then chains will be run
                    sequentially in otherwise a `multiprocessing.Pool` object
                    will be used to dynamically assign the chains across
                    multiple processes. If set to `None` then the number of
                    processes will default to the output of `os.cpu_count()`.

            Returns:
                arvix.InferenceData:
                    An arviz data container with groups `posterior` and
                    'sample_stats', both of instances of `xarray.Dataset`.
                    The `posterior` group corresponds to the chain variable
                    samples computed using the `chain_var_funcs` entries (with
                    the data variable keys corresponding to the keys there).
                    The `sample_stats` group corresponds to the statistics of
                    the integration transition such as the acceptance
                    probabilities and number of integrator steps.
            """
            return super().sample_chains_arviz(
                n_sample, init_states, chain_var_funcs,
                n_process=n_process, sample_stats_key='integration_transition')


class StaticMetropolisHMC(HamiltonianMonteCarlo):
    """Static integration time HMC implementation with Metropolis sampling.

    In each transition a trajectory is generated by integrating the Hamiltonian
    dynamics from the current state in the current integration time direction
    for a fixed integer number of integrator steps.

    The state at the end of the trajectory with the integration direction
    negated (this ensuring the proposed move is an involution) is used as the
    proposal in a Metropolis acceptance step. The integration direction is then
    deterministically negated again irrespective of the accept decision, with
    the effect being that on acceptance the integration direction will be equal
    to its initial value and on rejection the integration direction will be
    the negation of its initial value.

    This is original proposed Hybrid Monte Carlo (often now instead termed
    Hamiltonian Monte Carlo) algorithm [1,2].

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, integrator, rng, n_step,
                 momentum_transition=None):
        integration_transition = trans.MetropolisStaticIntegrationTransition(
            system, integrator, n_step)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def n_step(self):
        return self.transitions['integration_transition'].n_step

    @n_step.setter
    def n_step(self, value):
        self.transitions['integration_transition'].n_step = value


class RandomMetropolisHMC(HamiltonianMonteCarlo):
    """Random integration time HMC with Metropolis sampling of new state.

    In each transition a trajectory is generated by integrating the Hamiltonian
    dynamics from the current state in the current integration time direction
    for a random integer number of integrator steps sampled from the uniform
    distribution on an integer interval.

    The state at the end of the trajectory with the integration direction
    negated (this ensuring the proposed move is an involution) is used as the
    proposal in a Metropolis acceptance step. The integration direction is then
    deterministically negated again irrespective of the accept decision, with
    the effect being that on acceptance the integration direction will be equal
    to its initial value and on rejection the integration direction will be
    the negation of its initial value.

    The randomisation of the number of integration steps avoids the potential
    of the chain mixing poorly due to using an integration time close to the
    period of (near) periodic systems [1,2].

    References:

      1. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
      2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
         Physics Letters B, 226(3-4), pp.369-371.
    """

    def __init__(self, system, integrator, rng, n_step_range,
                 momentum_transition=None):
        integration_transition = trans.MetropolisRandomIntegrationTransition(
            system, integrator, n_step_range)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def n_step_range(self):
        return self.transitions['integration_transition'].n_step_range

    @n_step_range.setter
    def n_step_range(self, value):
        self.transitions['integration_transition'].n_step_range = value


class DynamicMultinomialHMC(HamiltonianMonteCarlo):
    """Dynamic integration time HMC with multinomial sampling of new state.

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

    def __init__(self, system, integrator, rng,
                 max_tree_depth=10, max_delta_h=1000,
                 termination_criterion=trans.riemannian_no_u_turn_criterion,
                 momentum_transition=None):
        integration_transition = trans.MultinomialDynamicIntegrationTransition(
            system, integrator, max_tree_depth, max_delta_h,
            termination_criterion)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def max_tree_depth(self):
        return self.transitions['integration_transition'].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        self.transitions['integration_transition'].max_tree_depth = value

    @property
    def max_delta_h(self):
        return self.transitions['integration_transition'].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions['integration_transition'].max_delta_h = value
