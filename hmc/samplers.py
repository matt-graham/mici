"""Monte Carlo sampler classes for peforming inference."""

import os
import logging
import tempfile
import signal
from collections import OrderedDict
import numpy as np
import hmc
import hmc.transitions as trans
from hmc.states import ChainState, HamiltonianState
from hmc.utils import get_size, get_valid_filename

try:
    import tqdm.auto as tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
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

    def _generate_memmap_filename(self, dir_path, prefix, key, index):
        key_str = get_valid_filename(str(key))
        if index is None:
            index = 0
        return os.path.join(dir_path, f'{prefix}_{index}_{key_str}.npy')

    def _open_new_memmap(self, filename, shape, dtype, default_val):
        memmap = np.lib.format.open_memmap(
            filename, dtype=dtype, mode='w+', shape=shape)
        memmap[:] = default_val
        return memmap

    def _memmaps_to_filenames(self, obj):
        if isinstance(obj, np.memmap):
            return obj.filename
        elif isinstance(obj, dict):
            return {k: self._memmaps_to_filenames(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._memmaps_to_filenames(v) for v in obj]

    def _init_chain_stats(self, n_sample, memmap_enabled, memmap_path,
                          chain_index):
        chain_stats = {}
        for trans_key, trans in self.transitions.items():
            chain_stats[trans_key] = {}
            if hasattr(trans, 'statistic_types'):
                for key, (dtype, val) in trans.statistic_types.items():
                    if memmap_enabled:
                        filename = self._generate_memmap_filename(
                            memmap_path, 'stats', f'{trans_key}_{key}',
                            chain_index)
                        chain_stats[trans_key][key] = self._open_new_memmap(
                            filename, (n_sample,), dtype, val)
                    else:
                        chain_stats[trans_key][key] = np.full(
                            n_sample, val, dtype)
        return chain_stats

    def _sample_chain(self, rng, n_sample, init_state, trace_funcs,
                      chain_index, parallel_chains, memmap_enabled,
                      memmap_path):
        if not isinstance(init_state, ChainState):
            state = ChainState(init_state)
        else:
            state = init_state
        chain_stats = self._init_chain_stats(
            n_sample, memmap_enabled, memmap_path, chain_index)
        # Initialise chain trace arrays
        traces = {}
        for key, trace_func in trace_funcs.items():
            var = trace_func(state)
            if memmap_enabled:
                filename = self._generate_memmap_filename(
                    memmap_path, 'trace', key, chain_index)
                traces[key] = self._open_new_memmap(
                    filename, (n_sample,) + var.shape, np.float64, np.nan)
            else:
                traces[key] = np.full((n_sample,) + var.shape, np.nan)
        total_return_nbytes = get_size(chain_stats) + get_size(traces)
        # Check if running in parallel and if total number of bytes to be
        # returned exceeds pickle limit
        if parallel_chains and total_return_nbytes > 2**31 - 1:
            raise RuntimeError(
                f'Total number of bytes allocated for arrays to be returned '
                f'({total_return_nbytes / 2**30:.2f} GiB) exceeds size limit '
                f'for returning results of a process (2 GiB). Try rerunning '
                f'with chain memory-mapping enabled (`memmap_enabled=True`).')
        if TQDM_AVAILABLE:
            desc = ('Sampling' if chain_index is None
                    else f'Chain {chain_index}')
            position = chain_index if parallel_chains else None
            sample_range = tqdm.trange(
                n_sample, desc=desc, unit='it', dynamic_ncols=True,
                position=position)
        else:
            sample_range = range(n_sample)
        try:
            for sample_index in sample_range:
                for trans_key, transition in self.transitions.items():
                    state, trans_stats = transition.sample(state, rng)
                    if trans_stats is not None:
                        if trans_key not in chain_stats:
                            logger.warning(
                                f'Transition {trans_key} returned statistics '
                                f'but has no `statistic_types` attribute.')
                        for key, val in trans_stats.items():
                            if key in chain_stats[trans_key]:
                                chain_stats[trans_key][key][sample_index] = val
                for key, trace_func in trace_funcs.items():
                    traces[key][sample_index] = trace_func(state)
        except KeyboardInterrupt:
            if memmap_enabled:
                for trace in traces.values:
                    trace.flush()
                for trans_stats in chain_stats.values():
                    for stat in trans_stats.values():
                        stat.flush()
        else:
            # If not interrupted increment sample_index so that it equals
            # n_sample to flag chain completed sampling
            sample_index += 1
        if parallel_chains and memmap_enabled:
                trace_filenames = self._memmaps_to_filenames(traces)
                stats_filenames = self._memmaps_to_filenames(chain_stats)
                return trace_filenames, stats_filenames, sample_index
        return traces, chain_stats, sample_index

    def sample_chain(self, n_sample, init_state, trace_funcs,
                     memmap_enabled=False, memmap_path=None):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_sample (int): Number of samples (iterations) to draw per chain.
            init_state (ChainState or array):
               Initial chain state. Can be either an array specifying the state
               or a `ChainState` instance.
            trace_funcs (dict[str, callable]): Dictionary of functions which
               compute the variables to be recorded at each chain iteration,
               with each trace function being passed the current state
               and returning an array corresponding to the variable(s) to be
               stored. The keys to the functions are used to index the trace
               arrays in the returned data.
            memmap_enabled (bool): Whether to memory-map arrays used to store
               chain data to files on disk to avoid excessive system memory
               usage for long chains and/or large chain states. The chain
               data is written to `.npy` files in the directory specified by
               `memmap_path` (or a temporary directory if not provided). These
               files persist after the termination of the function so should be
               manually deleted when no longer required.
            memmap_path (str): Path to directory to write memory-mapped chain
               data to. If not provided, a temporary directory will be created
               and the chain data written to files there.

        Returns:
            traces (dict[str, array]):
                Chain trace arrays, with one entry per function in
                `trace_funcs` with the same key. Each entry consists of an
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
        # Create temporary directory if memory mapping and no path provided
        if memmap_enabled and memmap_path is None:
            memmap_path = tempfile.mkdtemp()
        traces, chain_stats, sample_index = self._sample_chain(
            rng=self.rng, n_sample=n_sample, init_state=init_state,
            trace_funcs=trace_funcs, chain_index=None, parallel_chains=False,
            memmap_enabled=memmap_enabled, memmap_path=memmap_path)
        if sample_index != n_sample:
            logger.exception(
                f'Sampling manually interrupted at iteration {sample_index}. '
                f'Arrays containing chain traces and statistics computed '
                f'before interruption will be returned, all entries for '
                f'iteration {sample_index} and above should be ignored.')
        return traces, chain_stats

    def _collate_chain_outputs(
            self, n_sample, traces_stats_and_sample_indices, load_memmaps,
            stack_chain_arrays=False):
        traces_stack = {}
        n_chain = len(traces_stats_and_sample_indices)
        chain_stats_stack = {}
        for chain_index, (traces, chain_stats, sample_index) in enumerate(
                traces_stats_and_sample_indices):
            for key, val in traces.items():
                if load_memmaps:
                    val = np.lib.format.open_memmap(val)
                if chain_index == 0:
                    traces_stack[key] = [val]
                else:
                    traces_stack[key].append(val)
            for trans_key, trans_stats in chain_stats.items():
                if chain_index == 0:
                    chain_stats_stack[trans_key] = {}
                for key, val in trans_stats.items():
                    if load_memmaps:
                        val = np.lib.format.open_memmap(val)
                    if chain_index == 0:
                        chain_stats_stack[trans_key][key] = [val]
                    else:
                        chain_stats_stack[trans_key][key].append(val)
        if stack_chain_arrays:
            for key, val in traces_stack.items():
                traces_stack[key] = np.stack(val)
            for trans_key, trans_stats in chain_stats_stack.items():
                for key, val in trans_stats.items():
                    trans_stats[key] = np.stack(val)
        return traces_stack, chain_stats_stack

    def sample_chains(self, n_sample, init_states, trace_funcs, n_process=1,
                      memmap_enabled=False, memmap_path=None,
                      stack_chain_arrays=False):
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
            trace_funcs (dict[str, callable]): Dictionary of functions which
                compute the variables to be recorded at each iteration, with
                each function being passed the current state and returning an
                array corresponding to the variable(s) to be stored. The keys
                to the functions are used to index the trace arrays in the
                returned data.
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If
                set to `None` then the number of processes will default to the
                output of `os.cpu_count()`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
               chain data to files on disk to avoid excessive system memory
               usage for long chains and/or large chain states. The chain data
               is written to `.npy` files in the directory specified by
               `memmap_path` (or a temporary directory if not provided). These
               files persist after the termination of the function so should be
               manually deleted when no longer required.
            memmap_path (str): Path to directory to write memory-mapped chain
               data to. If not provided, a temporary directory will be created
               and the chain data written to files there.
            stack_chain_arrays (bool): Whether to stack the lists of per-chain
               arrays in the returned dictionaries into new arrays with the
               chain index as the first axis. Note if set to `True` when
               memory-mapping is enabled (`memmap_enabled=True`) all
               memory-mapped arrays will be loaded from disk in to memory.

        Returns:
            traces (dict[str, list[array]] or dict[str, array]):
                Trace arrays, with one entry per function in `trace_funcs` with
                the same key. Each entry consists of a list of arrays, one per
                chain, with the first axes of the arrays corresponding to the
                sampling (draw) index, or if `stack_chain_arrays=True` each
                entry is an array with the first axes corresponding to the
                chain index, the second axis the sampling (draw) index; in both
                cases the size and number of remaining array axes is determined
                by the shape of the arrays returned by the corresponding
                `trace_funcs` entry.
            chain_stats (dict[str, dict[str, list[array]]]):
                Dictionary of chain transition statistics. Outer dictionary
                contains entries for each chain transition which returns
                statistics (e.g. acceptance probabilities) on each iteration.
                For each such transition, a dictionary is returned with string
                keys describing the statistics recorded and values
                corresponding to either a list of arrays with one array per
                chain and the first axis of the arrays corresponding to the
                sampling index, or if `stack_chain_arrays=True` each entry is
                an array with the first axes corresponding to the chain index,
                the second axis the sampling (draw) index.
        """
        n_chain = len(init_states)
        # Create temp directory if memory-mapping enabled and no path provided
        if memmap_enabled and memmap_path is None:
            memmap_path = tempfile.mkdtemp()
        if RANDOMGEN_AVAILABLE:
            seed = self.rng.randint(2**64, dtype='uint64')
            rngs = [randomgen.Xorshift1024(seed).jump(i).generator
                    for i in range(n_chain)]
        else:
            seeds = (self.rng.choice(2**16, n_chain, False) * 2**16 +
                     self.rng.choice(2**16, n_chain, False))
            rngs = [np.random.RandomState(seed) for seed in seeds]
        if n_process == 1:
            # Using single process therefore run chains sequentially
            chain_outputs = []
            for c, (rng, init_state) in enumerate(zip(rngs, init_states)):
                traces, chain_stats, n_sample_chain = self._sample_chain(
                    rng=rng, n_sample=n_sample, init_state=init_state,
                    trace_funcs=trace_funcs, chain_index=c,
                    parallel_chains=False, memmap_enabled=memmap_enabled,
                    memmap_path=memmap_path)
                chain_outputs.append((traces, chain_stats, n_sample_chain))
                if n_sample_chain != n_sample:
                    logger.error(
                        f'Sampling manually interrupted at chain {c} iteration'
                        f' {n_sample_chain}. Arrays containing chain traces'
                        f' and statistics computed before interruption will'
                        f' be returned, all entries for iteration '
                        f' {n_sample_chain} and above of chain {c} should be'
                        f' ignored.')
                    break
        else:
            # Run chains in parallel using a multiprocess(ing).Pool
            # Child processes made to ignore SIGINT signals to allow handling
            # of KeyboardInterrupts in parent process
            with Pool(n_process, _ignore_sigint_initialiser) as pool:
                try:
                    chain_outputs = pool.starmap(
                        self._sample_chain,
                        zip(rngs,
                            [n_sample] * n_chain,
                            init_states,
                            [trace_funcs] * n_chain,
                            range(n_chain),  # chain_index
                            [True] * n_chain,  # parallel_chains flags
                            [memmap_enabled] * n_chain,
                            [memmap_path] * n_chain,))
                except KeyboardInterrupt:
                    # Close any still running processes
                    pool.terminate()
                    pool.join()
                    err_message = 'Sampling manually interrupted.'
                    if memmap_enabled:
                        err_message += (
                            f' Chain data recorded so far is available in '
                            f'directory {memmap_path}.')
                    logger.error(err_message)
                    raise
        # When running parallel jobs with memory-mapping enabled, data arrays
        # returned by processes as file paths to array memory-maps therfore
        # load memory-maps objects from file before returing results
        load_memmaps = memmap_enabled and n_process > 1
        return self._collate_chain_outputs(
            n_sample, chain_outputs, load_memmaps, stack_chain_arrays)


class HamiltonianMCMC(MarkovChainMonteCarloMethod):
    """Wrapper class for Hamiltonian Markov chain Monte Carlo (H-MCMC) methods.

    Here H-MCMC is defined as a MCMC method which augments the original target
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

    def _preprocess_init_state(self, init_state):
        """Make sure initial state is a HamiltonianState and has momentum."""
        if isinstance(init_state, np.ndarray):
            # If array use to set position component of new HamiltonianState
            init_state = HamiltonianState(pos=init_state)
        elif not isinstance(init_state, HamiltonianState):
            raise TypeError(
                'init_state should be a NumPy array or `HamiltonianState`.')
        if init_state.mom is None:
            init_state.mom = self.system.sample_momentum(init_state, self.rng)
        return init_state

    def sample_chain(self, n_sample, init_state, trace_funcs=None, **kwargs):
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
            trace_funcs (dict[str, callable]): Dictionary of functions which
                compute the variables to be recorded at each iteration, with
                each function being passed the current state and returning an
                array corresponding to the variable(s) to be stored. The keys
                to the functions are used to index the trace arrays in the
                returned data.

        Kwargs:
            memmap_enabled (bool): Whether to memory-map arrays used to store
               chain data to files on disk to avoid excessive system memory
               usage for long chains and/or large chain states. The chain data
               is written to `.npy` files in the directory specified by
               `memmap_path` (or a temporary directory if not provided). These
               files persist after the termination of the function so should be
               manually deleted when no longer required.
            memmap_path (str): Path to directory to write memory-mapped chain
               data to. If not provided, a temporary directory will be created
               and the chain data written to files there.

        Returns:
            traces (dict[str, array]):
                Chain trace arrays, with one entry per function in
                `trace_funcs` with the same key. Each entry consists of an
                arrays with the leading dimension of the arrays corresponding
                to the sampling (draw) index.
            chain_stats (dict[str, dict[str, array]]):
                Dictionary of chain integration transition statistics, with
                string keys describing the statistics recorded and array values
                with the leading dimension corresponding to the sampling index.
        """
        init_state = self._preprocess_init_state(init_state)
        if trace_funcs is None:
            trace_funcs = {'pos': extract_pos}
        traces, chain_stats = super().sample_chain(
            n_sample, init_state, trace_funcs, **kwargs)
        return traces, chain_stats.get('integration_transition', {})

    def sample_chains(self, n_sample, init_states, trace_funcs=None, **kwargs):
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
                sampled from its conditional distribution. One chain will be
                run for each state in the iterable sequence.
            trace_funcs (dict[str, callable]): Dictionary of functions which
                compute the variables to be recorded at each iteration, with
                each function being passed the current state and returning an
                array corresponding to the variable(s) to be stored. By default
                (or if set to `None`) a single function which returns the
                position component of the state is used. The keys to the
                functions are used to index the chain variable arrays in the
                returned data.

        Kwargs:
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will default to the
                output of `os.cpu_count()`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
               chain data to files on disk to avoid excessive system memory
               usage for long chains and/or large chain states. The chain data
               is written to `.npy` files in the directory specified by
               `memmap_path` (or a temporary directory if not provided). These
               files persist after the termination of the function so should be
               manually deleted when no longer required.
            memmap_path (str): Path to directory to write memory-mapped chain
               data to. If not provided, a temporary directory will be created
               and the chain data written to files there.
            stack_chain_arrays (bool): Whether to stack the lists of per-chain
               arrays in the returned dictionaries into new arrays with the
               chain index as the first axis. Note if set to `True` when
               memory-mapping is enabled (`memmap_enabled=True`) all
               memory-mapped arrays will be loaded from disk in to memory.

        Returns:
            traces (dict[str, list[array]] or dict[str, array]):
                Trace arrays, with one entry per function in `trace_funcs` with
                the same key. Each entry consists of a list of arrays, one per
                chain, with the first axes of the arrays corresponding to the
                sampling (draw) index, or if `stack_chain_arrays=True` each
                entry is an array with the first axes corresponding to the
                chain index, the second axis the sampling (draw) index; in both
                cases the size and number of remaining array axes is determined
                by the shape of the arrays returned by the corresponding
                `trace_funcs` entry.
            chain_stats (dict[str, list[array]] or dict[str, array]):
                Chain integration transition statistics as a dictionary with
                string keys describing the statistics recorded and values
                corresponding to either a list of arrays with one array per
                chain and the first axis of the arrays corresponding to the
                sampling index, or if `stack_chain_arrays=True` each entry is
                an array with the first axes corresponding to the chain index,
                the second axis the sampling (draw) index.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        if trace_funcs is None:
            trace_funcs = {'pos': extract_pos}
        traces, chain_stats = super().sample_chains(
            n_sample, init_states, trace_funcs, **kwargs)
        return traces, chain_stats.get('integration_transition', {})


class StaticMetropolisHMC(HamiltonianMCMC):
    """Static integration time H-MCMC implementation with Metropolis sampling.

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


class RandomMetropolisHMC(HamiltonianMCMC):
    """Random integration time H-MCMC with Metropolis sampling of new state.

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


class DynamicMultinomialHMC(HamiltonianMCMC):
    """Dynamic integration time HMCMC with multinomial sampling of new state.

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
