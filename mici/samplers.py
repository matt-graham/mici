"""Monte Carlo sampler classes for peforming inference."""

import os
import sys
import inspect
import queue
from contextlib import ExitStack, contextmanager
from pathlib import Path
from pickle import PicklingError
import logging
import tempfile
import signal
from warnings import warn
from collections import OrderedDict
import numpy as np
from numpy.random import default_rng
import mici.transitions as trans
from mici.states import ChainState
from mici.progressbars import (
    SequenceProgressBar,
    LabelledSequenceProgressBar,
    DummyProgressBar,
    _ProxySequenceProgressBar,
)
from mici.errors import AdaptationError
from mici.adapters import DualAveragingStepSizeAdapter
from mici.stagers import WarmUpStager, WindowedWarmUpStager

# Preferentially import from multiprocess library if available as able to
# serialize much wider range of types including autograd functions
try:
    from multiprocess import Pool
    from multiprocess.managers import SyncManager

    MULTIPROCESS_AVAILABLE = True
except ImportError:
    from multiprocessing import Pool
    from multiprocessing.managers import SyncManager

    MULTIPROCESS_AVAILABLE = False

try:
    from threadpoolctl import threadpool_limits

    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False

try:
    from contextlib import nullcontext
except ImportError:
    # Fallback for nullcontext context manager for Python 3.6
    # https://stackoverflow.com/a/55902915
    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


logger = logging.getLogger(__name__)


def _ignore_sigint_initializer():
    """Initializer for processes to force ignoring SIGINT interrupt signals."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@contextmanager
def _ignore_sigint_manager():
    """Context-managed SyncManager which ignores SIGINT interrupt signals."""
    manager = SyncManager()
    try:
        manager.start(_ignore_sigint_initializer)
        yield manager
    finally:
        manager.shutdown()


@contextmanager
def _pool_context_manager(n_process):
    """Context-manager for process pool that ensures clean exiting.

    Compared to built-in context-manager protocol implementation on Pool object which
    calls the `terminate` method on exit which immediately stops the worker processes,
    this manager instead ensures a clean exit by calling `close` to prevent any
    additional jobs being submitted to pool, and then `join` to wait for processes to
    exit.
    """
    pool = Pool(n_process)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def _get_valid_filename(string):
    """Generate a valid filename from a string.

    Strips all characters which are not alphanumeric or a period (.), dash (-)
    or underscore (_).

    Based on https://stackoverflow.com/a/295146/4798943

    Args:
        string (str): String file name to process.

    Returns:
        str: Generated file name.
    """
    return "".join(c for c in string if (c.isalnum() or c in "._- "))


def _generate_memmap_filenames(dir_path, prefix, key, indices):
    """Generate new memory-map filenames."""
    key_str = _get_valid_filename(str(key))
    return [
        os.path.join(dir_path, f"{prefix}_{index}_{key_str}.npy") for index in indices
    ]


def _open_new_memmap(file_path, shape, default_val, dtype):
    """Open a new memory-mapped array object and fill with a default-value.

    Args:
        file_path (str): Path to write memory-mapped array to.
        shape (Tuple[int, ...]): Shape of new array.
        default_val: Value to fill array with. Should be compatible with
            specified `dtype`.
        dtype (str or numpy.dtype): NumPy data-type for array.

    Returns
        memmap (numpy.memmap): Memory-mapped array object.
    """
    if isinstance(shape, int):
        shape = (shape,)
    memmap = np.lib.format.open_memmap(file_path, dtype=dtype, mode="w+", shape=shape)
    memmap[:] = default_val
    return memmap


def _memmaps_to_file_paths(pytree):
    """Convert any memmaps in pytree to corresponding file paths.

    Acts recursively on arbitrary 'pytrees' of nested dict/tuple/lists with any object
    which is none of a memory map, dictionary, tuple or list returned as is.

    Arg:
        pytree: Pytree of objects to convert.

    Returns:
        Pytree with same structure as `pytree` and any memory maps converted to paths.
    """
    if isinstance(pytree, np.memmap):
        return Path(pytree.filename)
    elif isinstance(pytree, dict):
        return {k: _memmaps_to_file_paths(v) for k, v in pytree.items()}
    elif isinstance(pytree, list):
        return [_memmaps_to_file_paths(v) for v in pytree]
    elif isinstance(pytree, tuple):
        return tuple(_memmaps_to_file_paths(v) for v in pytree)
    else:
        return pytree


def _file_paths_to_memmaps(pytree):
    """Convert any paths in pytree to corresponding memory-mapped arrays.

    Acts recursively on arbitrary 'pytrees' of nested dict/tuple/lists with any object
    which is none of a path, dictionary, tuple or list returned as is.

    Arg:
        pytree: Pytree of objects to convert.

    Returns:
        Pytree with same structure as `pytree` and any paths converted to memory maps.
    """
    if isinstance(pytree, Path):
        return np.lib.format.open_memmap(pytree)
    elif isinstance(pytree, dict):
        return {k: _file_paths_to_memmaps(v) for k, v in pytree.items()}
    elif isinstance(pytree, list):
        return [_file_paths_to_memmaps(v) for v in pytree]
    elif isinstance(pytree, tuple):
        return tuple(_file_paths_to_memmaps(v) for v in pytree)
    else:
        return pytree


def _zip_dict(**kwargs):
    """Zip iterable keyword arguments in to an iterable of dictionaries.

    Acts analogously to built-in `zip` in taking a variable number of iterables as
    arguments and returning an iterable which iterates over collections of values
    from each of the iterables, however unlike `zip` here the input iterables must be
    specified as keyword arguments and the returned collections are dictionaries rather
    than tuples, with keys specified by the argument names.

    For example

        list(_zip_dict(a=[1, 2, 3], b=[4, 5, 6]))

    Produces

        [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
    """
    return (dict(zip(kwargs.keys(), val_set)) for val_set in zip(*kwargs.values()))


def _check_and_process_init_state(state, transitions):
    """Check initial chain state is valid and convert dict to ChainState."""
    for trans_key, transition in transitions.items():
        for var_key in transition.state_variables:
            if var_key not in state:
                raise ValueError(
                    f"init_state does contain have {var_key} value required by"
                    f" {trans_key} transition."
                )
    if not isinstance(state, (ChainState, dict)):
        raise TypeError("init_state should be a dictionary or ChainState.")
    return ChainState(**state) if isinstance(state, dict) else state


def _init_stats(transitions, n_chain, n_iter, use_memmap, memmap_path):
    """Initialize dictionary of per-transition chain statistics array dicts."""
    stats = {}
    for trans_key, transition in transitions.items():
        if transition.statistic_types is not None:
            stats[trans_key] = {}
            for key, (dtype, val) in transition.statistic_types.items():
                if use_memmap:
                    stats[trans_key][key] = [
                        _open_new_memmap(filename, n_iter, val, dtype)
                        for filename in _generate_memmap_filenames(
                            memmap_path, "stats", f"{trans_key}_{key}", range(n_chain)
                        )
                    ]
                else:
                    stats[trans_key][key] = list(np.full((n_chain, n_iter), val, dtype))
    return stats


def _init_traces(trace_funcs, init_states, n_iter, use_memmap, memmap_path):
    """Initialize dictionary of chain trace arrays."""
    traces = {}
    n_chain = len(init_states)
    for trace_func in trace_funcs:
        for key, val in trace_func(init_states[0]).items():
            val = np.array(val) if np.isscalar(val) else val
            init = np.nan if np.issubdtype(val.dtype, np.inexact) else 0
            if use_memmap:
                traces[key] = [
                    _open_new_memmap(filename, (n_iter,) + val.shape, init, val.dtype)
                    for filename in _generate_memmap_filenames(
                        memmap_path, "trace", key, range(n_chain)
                    )
                ]
            else:
                traces[key] = list(
                    np.full((n_chain, n_iter,) + val.shape, init, val.dtype)
                )
    return traces


def _construct_chain_iterators(
    n_iter, chain_iterator_class, n_chain=None, position_offset=0
):
    """Set up chain iterator progress bar object(s)."""
    return [
        chain_iterator_class(
            range(n_iter),
            description=f"Chain {c+1}/{n_chain}",
            position=(c + position_offset, n_chain + position_offset),
        )
        for c in range(n_chain)
    ]


def _update_chain_stats(sample_index, chain_stats, trans_key, trans_stats):
    """Update chain statistics arrays for current chain iteration."""
    if trans_stats is not None:
        for key, val in trans_stats.items():
            chain_stats[trans_key][key][sample_index] = val


def _update_monitor_stats(monitor_stats, monitor_dict, trans_key, trans_stats):
    """Update dictionary of per-iteration monitored statistics."""
    if trans_key in monitor_stats:
        for stats_key in monitor_stats[trans_key]:
            try:
                val = trans_stats[stats_key]
            except KeyError:
                raise KeyError(
                    f"Monitored statistics key {stats_key} is not present in "
                    f"statistics returned by transition {trans_key}."
                )
            if stats_key not in monitor_dict:
                monitor_dict[stats_key] = val
            else:
                monitor_dict[f"{trans_key}.{stats_key}"] = val


def _flush_memmap_chain_data(chain_traces, chain_stats):
    """Flush all pending writes to memory-mapped chain data arrays to disk."""
    if chain_traces is not None:
        for trace in chain_traces.values():
            if hasattr(trace, "flush"):
                trace.flush()
    if chain_stats is not None:
        for trans_stats in chain_stats.values():
            for stat in trans_stats.values():
                if hasattr(stat, "flush"):
                    stat.flush()


def _sample_chain(
    init_state,
    chain_iterator,
    rng,
    transitions,
    trace_funcs=None,
    chain_traces=None,
    chain_stats=None,
    load_memmaps=False,
    chain_index=0,
    sampling_index_offset=0,
    monitor_stats=None,
    adapters=None,
):
    """Sample a chain by iteratively appyling a sequence of transition kernels.

    Args:
        init_state (mici.states.ChainState or Dict[str, object]): Initial chain state.
            Either a `mici.states.ChainState` object or a dictionary with entries
            specifying initial values for all state variables used by chain transition
            `sample` methods.
        chain_iterator (Iterable[Tuple[int, Dict]]): Iterable object which is iterated
            over to produce sample indices and (empty) iteration statistic dictionaries
            to output monitored chain statistics to during sampling.
        rng (numpy.random.Generator): Numpy random number generator.
        transitions (OrderedDict[str, Transition]): Ordered dictionary of Markov
            transitions kernels to sequentially sample from on each chain iteration.
        trace_funcs (Sequence[Callable[[ChainState], Dict[str, array]]]): Sequence of
            functions which compute the variables to be recorded at each chain
            iteration, with each trace function being passed the current state and
            returning a dictionary of scalar or array values corresponding to the
            variable(s) to be stored. The keys in the returned dictionaries are used to
            index the trace arrays in the returned traces dictionary. If a key appears
            in multiple dictionaries only the the value corresponding to the last trace
            function to return that key will be stored.
        chain_traces (Dict[str, Union[array, str]]): Dictionary of chain trace arrays
            (or string file paths to memory-mapped stores of those arrays if
            `load_memmaps=True`) , to record traced output in. Values in dictionary are
            arrays (or paths to memmaped arrays) which variables outputted by trace
            functions in `trace_funcs` are recorded in. The leading dimension of each
            array corresponds to the sampling (draw) index and must be of size greater
            than `sampling_index_offset + n_iter` where `n_iter` is the number of chain
            iterations. The key for each value is the corresponding key in the
            dictionary returned by the trace function which computes the traced value.
        chain_stats (Dict[str, Dict[str, Union[array, str]]]): Dictionary of chain
            transition statistic dictionaries to record chain statistics in. Values in
            outer dictionary are dictionaries of statistics for each chain transition,
            keyed by the string key for the transition. The values in each inner
            transition dictionary are arrays (or paths to memmaped arrays if
            `load_memmaps=True`) which chain statistic values will be recorded in. The
            leading dimension of each array corresponds to the sampling (draw) index and
            must be of size greater than `sampling_index_offset + n_iter` where `n_iter`
            is the number of chain iterations. The key for each value is a string
            description of the corresponding transition statistic.
        load_memmaps (bool): Flag indicating whether `chain_traces` and `chain_stats`
            dictionary contain file paths to saved memory-mapped arrays which must be
            loaded before writing chain data.
        chain_index (int): Identifier for chain when sampling multiple chains.
        sampling_index_offset (int): Non-negative integer specifying sampling (draw)
            index in trace and statistic arrays to begin recording values at.
        monitor_stats (Dict[str, Sequence[str]]): String-keyed dictionary of sequences
            of strings, with dictionary key the key of a Markov transition in the
            `transitions` dict passed to the the `__init__` method and the corresponding
            sequence, the keys of statistics returned by the transition (as defined by
            the `statistics_type` attribute of transition). The mean over samples
            computed so far of the statistics associated with any valid key-pairs will
            be monitored during sampling by printing as postfix to progress bar.
        adapters (Dict[str, Sequence[Adapter]): Dictionary of sequences of
            `mici.adapters.Adapter` instances keyed by strings corresponding to the key
            of the transition in the `transitions` dictionary to apply the adapters to.
            Each adapter is able to adaptatively set the parameters of a transition
            while sampling a chain. Note that the adapter updates for each transition
            are applied in the order the adapters appear in the iterable and so if
            multiple adapters change the same parameter(s) the order will matter.
            Adaptation based on the chain state history breaks the Markov property and
            so any chain samples while adaptation is active should not be used in
            estimates of expectations.

    Returns:
        final_state (mici.states.ChainState): State of chain after final iteration. May
            be used to resume sampling a chain by passing as the initial state to a new
            `sample_chain` call.
        adapter_states (Dict[str, List[Dict[str, Any]]]): Dictionary of per-transition
            adapter states. Dictionary is keyed by transition key (i.e. the key of a
            Markov transition in the `transitions` dict) with values lists of
            dictionaries corresponding to the states of any adapters applied to that
            transition.
        exception (None or Exception): Any handled exception which may affect how the
            returned outputs are processed by the caller.
    """
    state = _check_and_process_init_state(init_state, transitions)
    if load_memmaps:
        chain_traces = _file_paths_to_memmaps(chain_traces)
        chain_stats = _file_paths_to_memmaps(chain_stats)
    adapter_states = {}
    try:
        if adapters is not None:
            for trans_key, adapter_list in adapters.items():
                adapter_states[trans_key] = []
                for adapter in adapter_list:
                    adapter_states[trans_key].append(
                        adapter.initialize(state, transitions[trans_key])
                    )
    except AdaptationError as exception:
        logger.error(
            f"Initialisation of {type(adapter).__name__} for chain "
            f"{chain_index + 1} failed: {exception}"
        )
        return state, adapter_states, exception
    try:
        sample_index = 0
        with chain_iterator:
            for sample_index, monitor_dict in chain_iterator:
                for trans_key, transition in transitions.items():
                    state, trans_stats = transition.sample(state, rng)
                    if adapters is not None and trans_key in adapters:
                        for adapter, adapter_state in zip(
                            adapters[trans_key], adapter_states[trans_key]
                        ):
                            adapter.update(
                                adapter_state, state, trans_stats, transition
                            )
                    if chain_stats is not None:
                        _update_chain_stats(
                            sample_index + sampling_index_offset,
                            chain_stats,
                            trans_key,
                            trans_stats,
                        )
                    if monitor_stats is not None:
                        _update_monitor_stats(
                            monitor_stats, monitor_dict, trans_key, trans_stats
                        )
                if chain_traces is not None and trace_funcs is not None:
                    for trace_func in trace_funcs:
                        for key, val in trace_func(state).items():
                            chain_traces[key][
                                sample_index + sampling_index_offset
                            ] = val
    except KeyboardInterrupt as e:
        exception = e
        logger.error(
            f"Sampling manually interrupted for chain {chain_index + 1} at"
            f" iteration {sample_index}. Arrays containing chain traces and"
            f" statistics computed before interruption will be returned."
        )
    else:
        exception = None
    finally:
        # flush any updates to memory-mapped chain data to disk before exiting
        _flush_memmap_chain_data(chain_traces, chain_stats)
    return state, adapter_states, exception


def _collate_chain_outputs(chain_outputs):
    """Unzip list of tuples of chain outputs in to tuple of stacked outputs."""
    final_states_stack = []
    adapt_states_stack = {}
    for final_state, adapt_states in chain_outputs:
        final_states_stack.append(final_state)
        for trans_key, adapt_state_list in adapt_states.items():
            if trans_key not in adapt_states_stack:
                adapt_states_stack[trans_key] = [[a] for a in adapt_state_list]
            else:
                for i, adapt_state in enumerate(adapt_state_list):
                    adapt_states_stack[trans_key][i].append(adapt_state)
    return final_states_stack, adapt_states_stack


def _get_per_chain_rngs(base_rng, n_chain):
    """Construct random number generators (RNGs) for each of a set of chains.

    If the base RNG bit generator has a `jumped` method this is used to produce
    a sequence of independent random substreams. Otherwise if the base RNG bit
    generator has a `_seed_seq` attribute this is used to spawn a sequence off
    generators.
    """
    if hasattr(base_rng, "bit_generator"):
        bit_generator = base_rng.bit_generator
    elif hasattr(base_rng, "_bit_generator"):
        bit_generator = base_rng._bit_generator
    else:
        bit_generator = None
    if bit_generator is not None and hasattr(bit_generator, "jumped"):
        return [default_rng(bit_generator.jumped(i)) for i in range(n_chain)]
    elif bit_generator is not None and hasattr(bit_generator, "_seed_seq"):
        seed_sequence = bit_generator._seed_seq
        return [default_rng(seed) for seed in seed_sequence.spawn(n_chain)]
    else:
        raise ValueError(f"Unsupported random number generator type {type(base_rng)}.")


def _sample_chains_sequential(chain_iterators, per_chain_kwargs, **common_kwargs):
    """Sample multiple chains sequentially in a single process."""
    chain_outputs = []
    exception = None
    for chain_index, (chain_iterator, chain_kwargs) in enumerate(
        zip(chain_iterators, per_chain_kwargs)
    ):
        *outputs, exception = _sample_chain(
            chain_iterator=chain_iterator,
            chain_index=chain_index,
            **chain_kwargs,
            **common_kwargs,
        )
        # Returned exception being AdaptationError indicates chain terminated
        # due to adapter initialisation failing therefore do not store returned
        # chain outputs
        if not isinstance(exception, AdaptationError):
            chain_outputs.append(outputs)
        # If returned handled exception was a manual interrupt break and return
        if isinstance(exception, KeyboardInterrupt):
            break
    return (*_collate_chain_outputs(chain_outputs), exception)


def _sample_chains_worker(chain_queue, iter_queue, common_kwargs):
    """Worker process function for parallel sampling of chains.

    Consumes chain arguments from a shared queue and outputs chain progress
    updates to a second shared queue.
    """
    chain_outputs = []
    while not chain_queue.empty():
        try:
            chain_index, n_iter, chain_kwargs = chain_queue.get(block=False)
            max_threads = common_kwargs.pop("max_threads_per_process", None)
            context = (
                threadpool_limits(limits=max_threads)
                if THREADPOOLCTL_AVAILABLE
                else nullcontext()
            )
            with context:
                *outputs, exception = _sample_chain(
                    chain_index=chain_index,
                    chain_iterator=_ProxySequenceProgressBar(
                        range(n_iter), chain_index, iter_queue
                    ),
                    load_memmaps=True,
                    **chain_kwargs,
                    **common_kwargs,
                )
            # Returned exception being AdaptationError indicates chain
            # terminated due to adapter initialisation failing therefore do not
            # store returned chain outputs and put None value on iteration queue
            # to indicate to parent process chain terminated
            if isinstance(exception, AdaptationError):
                iter_queue.put(None)
            else:
                chain_outputs.append((chain_index, outputs))
            # If returned handled exception was a manual interrupt put exception
            # on iteration queue to communicate to parent process and break
            if isinstance(exception, KeyboardInterrupt):
                iter_queue.put(exception)
                break
        except queue.Empty:
            pass
        except Exception as exception:
            # Log exception here so that correct traceback is logged
            logger.error(
                "Exception encountered in chain worker process:", exc_info=exception
            )
            # Put exception on iteration queue to be reraised in parent process
            iter_queue.put(exception)
    return chain_outputs


def _finalize_adapters(adapter_states_dict, chain_states, adapters, transitions, rngs):
    """Finalize adapter updates to transitions based on final adapter states."""
    for trans_key, adapter_states_list in adapter_states_dict.items():
        for adapter_states, adapter in zip(adapter_states_list, adapters[trans_key]):
            adapter.finalize(adapter_states, chain_states, transitions[trans_key], rngs)


def _sample_chains_parallel(
    chain_iterators, per_chain_kwargs, n_process, **common_kwargs
):
    """Sample multiple chains in parallel over multiple processes."""
    n_iters = [len(it) for it in chain_iterators]
    n_chain = len(chain_iterators)
    with _ignore_sigint_manager() as manager, _pool_context_manager(n_process) as pool:
        results = None
        exception = None
        try:
            # Shared queue for workers to output chain progress updates to
            iter_queue = manager.Queue()
            # Shared queue for workers to get arguments for _sample_chain calls
            # from on initialising each chain
            chain_queue = manager.Queue()
            for c, (chain_kwargs, n_iter) in enumerate(zip(per_chain_kwargs, n_iters)):
                # Map memmaps to their filepaths prior to putting on argument queue to
                # avoid serializing potentially large memory mapped arrays
                chain_kwargs["chain_stats"] = _memmaps_to_file_paths(
                    chain_kwargs["chain_stats"]
                )
                chain_kwargs["chain_traces"] = _memmaps_to_file_paths(
                    chain_kwargs["chain_traces"]
                )
                chain_queue.put((c, n_iter, chain_kwargs))
            # Start n_process worker processes which each have access to the
            # shared queues, returning results asynchronously
            results = pool.starmap_async(
                _sample_chains_worker,
                [(chain_queue, iter_queue, common_kwargs) for p in range(n_process)],
            )
            # Start loop to use chain progress updates outputted to iter_queue
            # by worker processes to update progress bars, using an ExitStack
            # to ensure all progress bars are context managed so that they
            # are closed properly on for example manual interrupts
            with ExitStack() as stack:
                pbars = [stack.enter_context(it) for it in chain_iterators]
                # Deadlock seems to occur when directly using results.ready()
                # method to check if all chains completed sampling in
                # while loop condition therefore manually keep track of
                # number of completed chains
                chains_completed = 0
                while not (iter_queue.empty() and chains_completed == n_chain):
                    iter_queue_item = iter_queue.get()
                    # Queue item being None indicates a chain terminated early
                    # due to a non-fatal error e.g. an error in initialising an
                    # adapter. In this case we continue sampling any other
                    # remaining chains but increment the completed chains
                    # counter to ensure correct termination of chain progress
                    # update loop
                    if iter_queue_item is None:
                        chains_completed += 1
                    # If queue item is KeyboardInterrupt exception break out of
                    # chain progress update loop but do not reraise exception
                    # so that partial chain outputs are returned
                    elif isinstance(iter_queue_item, KeyboardInterrupt):
                        exception = iter_queue_item
                        break
                    # Re raise any other exception passed from worker processes
                    elif isinstance(iter_queue_item, Exception):
                        raise RuntimeError(
                            "Unhandled exception in chain worker process."
                        ) from iter_queue_item
                    else:
                        # Otherwise unpack and update progress bar
                        chain_index, sample_index, data_dict = iter_queue_item
                        pbars[chain_index].update(sample_index, data_dict)
                        if sample_index == n_iters[chain_index]:
                            chains_completed += 1
        except (PicklingError, AttributeError) as e:
            if not MULTIPROCESS_AVAILABLE and (
                isinstance(e, PicklingError) or "pickle" in str(e)
            ):
                raise RuntimeError(
                    "Error encountered while trying to run chains on multiple"
                    "processes in parallel. The inbuilt multiprocessing module"
                    " uses pickle to communicate between processes and pickle "
                    "does support pickling anonymous or nested functions. If "
                    "you use anonymous or nested functions in your model "
                    "functions or are using autograd to automatically compute "
                    "derivatives (autograd uses anonymous and nested "
                    "functions) then installing the Python package "
                    "multiprocess, which is able to serialise anonymous and "
                    "nested functions and will be used in preference to "
                    "multiprocessing by this package when available, may "
                    "resolve this error."
                ) from e
            else:
                raise e
        except KeyboardInterrupt as e:
            # Interrupts handled in child processes therefore ignore here
            exception = e
        if results is not None:
            # Join all output lists from per-process workers in to single list
            indexed_chain_outputs = sum((res for res in results.get()), [])
            # Sort list by chain index (first element of tuple entries) and
            # then create new list with chain index removed
            chain_outputs = [outp for i, outp in sorted(indexed_chain_outputs)]
        else:
            chain_outputs = []
    return (*_collate_chain_outputs(chain_outputs), exception)


class MarkovChainMonteCarloMethod(object):
    """Generic Markov chain Monte Carlo (MCMC) sampler.

    Generates a Markov chain from some initial state by iteratively applying
    a sequence of Markov transition operators.
    """

    def __init__(self, rng, transitions):
        """
        Args:
            rng (numpy.random.Generator): Numpy random number generator.
            transitions (OrderedDict[str, Transition]): Ordered dictionary of
                Markov transitions kernels to sequentially sample from on each
                chain iteration.
        """
        if isinstance(rng, np.random.RandomState):
            warn(
                "Use of numpy.random.RandomState random number generators is "
                "deprecated. Please use a numpy.random.Generator instance "
                "instead for example from a call to numpy.random.default_rng.",
                DeprecationWarning,
            )
            rng = np.random.Generator(rng._bit_generator)
        self.rng = rng
        self.transitions = transitions

    def sample_chains(
        self,
        n_warm_up_iter,
        n_main_iter,
        init_states,
        trace_funcs=None,
        adapters=None,
        stager=None,
        n_process=1,
        trace_warm_up=False,
        max_threads_per_process=None,
        force_memmap=False,
        memmap_path=None,
        monitor_stats=None,
        display_progress=True,
        progress_bar_class=None,
    ):
        """Sample Markov chains from given initial states with optional adaptive warm up

        One or more Markov chains are sampled, with each chain iteration consisting of
        one or more Markov transitions. The chains are split into multiple *stages* with
        zero or more adaptive warm up stages followed by the main non-adaptive sampling
        stage. During the adaptive stage(s) parameters of the transition(s) are
        adaptively tuned based on the chain state and/or transition statistics.

        The chains (including both adaptive and non-adaptive stages) may be run in
        parallel across multiple independent processes or sequentially. In all cases all
        chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per chain.
                Depending on the `mici.stagers.Stager` instance specified by the
                `stager` argument the warm up iterations may be split between one or
                more adaptive stages. If zero, only a single non-adaptive stage is used.
            n_main_iter (int): Number of iterations (samples to draw) per chain during
                main (non-adaptive) sampling stage.
            init_states (Iterable[Union[ChainState, Dict[str, Any]]]): Initial chain
                states. Each entry can be either a `ChainState` object or a dictionary
                with entries specifying initial values for all state variables used by
                chain transition `sample` methods.

        Kwargs:
            trace_funcs (Optional[Sequence[Callable[[ChainState], Dict[str, array]]]]):
                Sequence of functions which compute the variables to be recorded at each
                chain iteration (during only the main non-adaptive sampling stage if
                `trace_warm_up` is False), with each trace function passed the current
                state and returning a dictionary of scalar or array values corresponding
                to the variable(s) to be stored. The keys in the returned dictionaries
                are used to index the trace arrays in the returned traces dictionary. If
                a key appears in multiple dictionaries only the the value corresponding
                to the last trace function to return that key will be stored. If `None`
                or an empty sequence no variables are traced.
            adapters (Optional[Dict[str, Sequence[Adapter]]): Dictionary of sequences
                of `mici.adapters.Adapter` instances keyed by strings corresponding to
                the key of the transition in the `transitions` dictionary to apply the
                adapters to, to use to adaptatively set parameters of the transitions
                during the adaptive stages of the chains. Note that the adapter updates
                are applied in the order the adapters appear in the sueqences and so if
                multiple adapters change the same parameter(s) the order will matter. If
                `None` or an empty sequence no adapters are used.
            stager (Optional[Stager]): Chain iteration stager object which controls the
                split of the chain iterations into the adaptive warm up and non-adaptive
                main stages. If set to `None` (the default) and all adapters specified
                by the `adapters` argument are of the fast type (i.e. their `is_fast`
                attribute is `True`) then a `mici.stagers.WarmUpStager` instance will be
                used corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by the
                adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments) will be
                used, corresponding to using multiple adaptive warm up stages with only
                the fast-type adapters active in some - see docstring of
                `mici.stagers.WarmUpStager` for details.
            n_process (Optional[int]): Number of parallel processes to run chains  over.
                If `n_process=1` then chains will be run sequentially otherwise a
                `multiprocessing.Pool` object will be used to dynamically assign the
                chains across multiple processes. If set to `None` then the number of
                processes will be set to the output of `os.cpu_count()`. Default is
                `n_process=1`.
            trace_warm_up (bool): Whether to record chain traces and statistics during
                warm-up stage iterations (True) or only record traces and statistics in
                the iterations of the final non-adaptive stage (True, the default).
            max_threads_per_process (Optional[int]): If `threadpoolctl` is available
                this argument may be used to limit the maximum number of threads that
                can be used in thread pools used in libraries supported by
                `threadpoolctl`, which include BLAS and OpenMP implementations. This
                argument will only have an effect if `n_process > 1` such that chains
                are being run on multiple processes and only if `threadpoolctl` is
                installed in the current Python environment. If set to `None` (the
                default) no limits are set.
            force_memmap (bool): Whether to force arrays used to store chain data to be
                memory-mapped to files on disk to avoid excessive system memory usage
                for long chains and/or large chain states. The chain data is written to
                `.npy` files in the directory specified by `memmap_path` (or a temporary
                directory if `memmap_path` is `None`). Chain data is always memory
                mapped when sampling chains in parallel on multiple processes.
            memmap_path (Optional[str]): Path to directory to write memory-mapped chain
                data to. If `None` (the default) and memory-mapping is enabled then a
                temporary directory will be created and the chain data written to files
                there, with the created files being deleted in this case once the last
                reference to them is closed.
            monitor_stats (Optional[Dict[str, List[str]]]): String-keyed dictionary of
                lists of strings, with dictionary key the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and the
                corresponding list, the keys of statistics returned by the transition
                (as defined by the `statistics_type` attribute of transition). The mean
                over samples computed so far of the statistics associated with any valid
                key-pairs will be monitored during sampling by printing as postfix to
                progress bar. If `None` no statistics are monitored.
            display_progress (bool): Whether to display a progress bar to track the
                completed chain sampling iterations. Default value is `True`, i.e. to
                display progress bar.
            progress_bar_class (Optional[mici.progressbars.ProgressBar]): Class or
                factory function for progress bar to use to show chain progress if
                enabled (`display_progress=True`). Defaults to
                `mici.progressbars.SequenceProgressBar` if `None` and
                `display_progress=True`.

        Returns:
            final_states (List[ChainState]): States of chains after final iteration. May
                be used to resume sampling a chain by passing as the initial states to a
                new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays. Values in
                dictionary are list of arrays of variables outputted by trace functions
                in `trace_funcs` with each array in the list corresponding to a single
                chain and the leading dimension of each array corresponding to the
                iteration (draw) index in the main non-adaptive sampling stage. The key
                for each value is the corresponding key in the dictionary returned by
                the trace function which computed the traced value.
            chain_stats (Dict[str, Dict[str, List[array]]]): Dictionary of chain
                transition statistic dictionaries. Values in outer dictionary are
                dictionaries of statistics for each chain transition, keyed by the
                string key for the transition. The values in each inner transition
                dictionary are lists of arrays of chain statistic values with each array
                in the list corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index in the main
                non-adaptive sampling stage. The key for each value is a string
                description of the corresponding integration transition statistic.
        """
        if not display_progress:
            progress_bar_class = DummyProgressBar
        elif progress_bar_class is None:
            progress_bar_class = SequenceProgressBar
        n_chain = len(init_states)
        n_trace_iter = n_warm_up_iter + n_main_iter if trace_warm_up else n_main_iter
        init_states = [
            _check_and_process_init_state(state, self.transitions)
            for state in init_states
        ]
        # memory-mapping of chain data used if force_memmap flag set or sampling chains
        # in parallel
        use_memmap = force_memmap or n_process > 1
        # use context-manager to ensure any temporary directory created for memory-maps
        # deleted before exiting
        if use_memmap and memmap_path is None:
            memmap_path_context = tempfile.TemporaryDirectory()
        else:
            memmap_path_context = nullcontext(memmap_path)
        with memmap_path_context as memmap_path:
            traces = (
                None
                if trace_funcs is None or len(trace_funcs) == 0
                else _init_traces(
                    trace_funcs, init_states, n_trace_iter, use_memmap, memmap_path
                )
            )
            stats = _init_stats(
                self.transitions, n_chain, n_trace_iter, use_memmap, memmap_path,
            )
            per_chain_rngs = _get_per_chain_rngs(self.rng, n_chain)
            per_chain_traces = (
                [None] * n_chain if traces is None else list(_zip_dict(**traces))
            )
            per_chain_stats = list(
                _zip_dict(**{k: _zip_dict(**v) for k, v in stats.items()})
            )
            common_kwargs = {
                "transitions": self.transitions,
                "monitor_stats": monitor_stats,
            }
            if n_process == 1:
                # Using single process therefore run chains sequentially
                sample_chains_func = _sample_chains_sequential
            else:
                # Run chains in parallel using a multiprocess(ing).Pool
                common_kwargs["n_process"] = n_process
                common_kwargs["max_threads_per_process"] = max_threads_per_process
                sample_chains_func = _sample_chains_parallel
            if stager is None:
                # use single warm-up stage stager if no adapters or all fast type
                if adapters is None or all(
                    a.is_fast for a_list in adapters.values() for a in a_list
                ):
                    stager = WarmUpStager()
                else:
                    stager = WindowedWarmUpStager()
            sampling_stages = stager.stages(
                n_warm_up_iter, n_main_iter, adapters, trace_funcs, trace_warm_up
            )
            chain_states = init_states
            sampling_index_offset = 0
            with LabelledSequenceProgressBar(
                sampling_stages, "Sampling stage", position=(0, n_chain + 1)
            ) as sampling_stages_pb:
                chain_iterators = _construct_chain_iterators(
                    1, progress_bar_class, n_chain, 1
                )
                for stage, _ in sampling_stages_pb:
                    for chain_it in chain_iterators:
                        chain_it.sequence = range(stage.n_iter)
                    chain_states, adapter_states, exception = sample_chains_func(
                        chain_iterators=chain_iterators,
                        per_chain_kwargs=_zip_dict(
                            init_state=chain_states,
                            rng=per_chain_rngs,
                            chain_traces=per_chain_traces
                            if stage.trace_funcs is not None
                            else [None] * n_chain,
                            chain_stats=per_chain_stats
                            if stage.record_stats
                            else [None] * n_chain,
                        ),
                        sampling_index_offset=sampling_index_offset,
                        trace_funcs=stage.trace_funcs,
                        adapters=stage.adapters,
                        **common_kwargs,
                    )
                    if len(adapter_states) > 0:
                        _finalize_adapters(
                            adapter_states,
                            chain_states,
                            stage.adapters,
                            self.transitions,
                            per_chain_rngs,
                        )
                    if stage.trace_funcs is not None:
                        sampling_index_offset += stage.n_iter
                    if isinstance(exception, KeyboardInterrupt):
                        return chain_states, traces, stats
        return chain_states, traces, stats


class HamiltonianMCMC(MarkovChainMonteCarloMethod):
    """Wrapper class for Hamiltonian Markov chain Monte Carlo (H-MCMC) methods.

    Here H-MCMC is defined as a MCMC method which augments the original target variable
    (henceforth position variable) with a momentum variable with a user specified
    conditional distribution given the position variable. In each chain iteration two
    Markov transitions leaving the resulting joint distribution on position and momentum
    variables invariant are applied - the momentum variables are updated in a transition
    which leaves their conditional distribution invariant (momentum transition) and then
    a trajectory in the joint space is generated by numerically integrating a
    Hamiltonian dynamic with an appropriate symplectic integrator which is exactly
    reversible, volume preserving and approximately conserves the joint probability
    density of the target-momentum state pair; one state from the resulting trajectory
    is then selected as the next joint chain state using an appropriate sampling scheme
    such that the joint distribution is left exactly invariant (integration transition).

    There are various options available for both the momentum transition and integration
    transition, with by default the momentum transition set to be independent resampling
    of the momentum variables from their conditional distribution.

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, rng, integration_transition, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integration_transition (mici.transitions.IntegrationTransition): Markov
                transition kernel which leaves canonical distribution invariant and
                jointly updates the position and momentum components of the chain state
                by integrating the Hamiltonian dynamics of the system to propose new
                values for the state.
            momentum_transition (None or mici.transitions.MomentumTransition): Markov
                transition kernel which leaves the conditional distribution on the
                momentum under the canonical distribution invariant, updating only the
                momentum component of the chain state. If set to `None` the momentum
                transition operator `mici.transitions.IndependentMomentumTransition`
                will be used, which independently samples the momentum from its
                conditional distribution.
        """
        self.system = system
        self.rng = rng
        if momentum_transition is None:
            momentum_transition = trans.IndependentMomentumTransition(system)
        super().__init__(
            rng,
            OrderedDict(
                momentum_transition=momentum_transition,
                integration_transition=integration_transition,
            ),
        )

    def _preprocess_init_state(self, init_state):
        """Make sure initial state is a ChainState and has momentum."""
        if isinstance(init_state, np.ndarray):
            # If array use to set position component of new ChainState
            init_state = ChainState(pos=init_state, mom=None, dir=1)
        elif not isinstance(init_state, ChainState) or "mom" not in init_state:
            raise TypeError(
                "init_state should be an array or `ChainState` with " "`mom` attribute."
            )
        if init_state.mom is None:
            init_state.mom = self.system.sample_momentum(init_state, self.rng)
        return init_state

    def _default_trace_func(self, state):
        """Default function of the chain state traced while sampling."""
        # This needs to be a method rather than for example a local nested
        # function in the __set_sample_chains_kwargs_defaults method to ensure
        # that it remains pickleable and so can be piped to a separate process
        # when running multiple chains using multiprocessing
        return {"pos": state.pos, "hamiltonian": self.system.h(state)}

    def sample_chains(self, n_warm_up_iter, n_main_iter, init_states, **kwargs):
        """Sample Markov chains from given initial states with adaptive warm up.

        One or more Markov chains are sampled, with each chain iteration consisting of a
        momentum transition followed by an integration transition. The chains are split
        into multiple *stages* with zero or more adaptive warm up stages followed by the
        main non-adaptive sampling stage. During the adaptive stage(s) parameters of the
        integration transition such as the integrator step size are adaptively tuned
        based on the chain state and/or transition statistics.

        The default settings use a single (fast) `DualAveragingStepSizeAdapter` adapter
        instance which adapts the integrator step-size using a dual-averaging algorithm
        in a single adaptive stage.

        The chains (including both adaptive and non-adaptive stages) may be run in
        parallel across multiple independent processes or sequentially. In all cases all
        chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per chain.
                Depending on the `mici.stagers.Stager` instance specified by the
                `stager` argument the warm up iterations may be split between one or
                more adaptive stages. If zero, only a single non-adaptive stage is used.
            n_main_iter (int): Number of iterations (samples to draw) per chain during
                main (non-adaptive) sampling stage.
            init_states (Iterable[Union[ChainState, array]]): Initial chain states. Each
                state can be either an array specifying the state position component or
                a `mici.states.ChainState` instance. If an array is passed or the `mom`
                attribute of the state is not set, a momentum component will be
                independently sampled from its conditional distribution. One chain will
                be run for each state in the iterable.

        Kwargs:
            trace_funcs (Optional[Sequence[Callable[[ChainState], Dict[str, array]]]]):
                Sequence of functions which compute the variables to be recorded at each
                chain iteration (during only the main non-adaptive sampling stage if
                `trace_warm_up` is False), with each trace function passed the current
                state and returning a dictionary of scalar or array values corresponding
                to the variable(s) to be stored. The keys in the returned dictionaries
                are used to index the trace arrays in the returned traces dictionary. If
                a key appears in multiple dictionaries only the the value corresponding
                to the last trace function to return that key will be stored. Default is
                to use a single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state under the key
                `hamiltonian`.
            adapters (Optional[Sequence[Adapter]]): Sequence of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the integration
                transition such as the step size during the adaptive stages of the
                chains. Note that the adapter updates are applied in the order the
                adapters appear in the sequence and so if multiple adapters change the
                same parameter(s) the order will matter. If `None` or an empty sequence
                no adapters are used. Default is to use a single instance of
                `mici.adapters.DualAveragingStepSizeAdapter` with its default
                parameters.
            stager (Optional[Stager]): Chain iteration stager object which controls the
                split of the chain iterations into the adaptive warm up and non-adaptive
                main stages. If set to `None` (the default) and all adapters specified
                by the `adapters` argument are of the fast type (i.e. their `is_fast`
                attribute is `True`) then a `mici.stagers.WarmUpStager` instance will be
                used corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by the
                adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments) will be
                used, corresponding to using multiple adaptive warm up stages with only
                the fast-type adapters active in some - see docstring of
                `mici.stagers.WarmUpStager` for details.
            n_process (Optional[int]): Number of parallel processes to run chains  over.
                If `n_process=1` then chains will be run sequentially otherwise a
                `multiprocessing.Pool` object will be used to dynamically assign the
                chains across multiple processes. If set to `None` then the number of
                processes will be set to the output of `os.cpu_count()`. Default is
                `n_process=1`.
            max_threads_per_process (Optional[int]): If `threadpoolctl` is available
                this argument may be used to limit the maximum number of threads that
                can be used in thread pools used in libraries supported by
                `threadpoolctl`, which include BLAS and OpenMP implementations. This
                argument will only have an effect if `n_process > 1` such that chains
                are being run on multiple processes and only if `threadpoolctl` is
                installed in the current Python environment. If set to `None` (the
                default) no limits are set.
            force_memmap (bool): Whether to force arrays used to store chain data to be
                memory-mapped to files on disk to avoid excessive system memory usage
                for long chains and/or large chain states. The chain data is written to
                `.npy` files in the directory specified by `memmap_path` (or a temporary
                directory if `memmap_path` is `None`). Chain data is always memory
                mapped when sampling chains in parallel on multiple processes.
            memmap_path (Optional[str]): Path to directory to write memory-mapped chain
                data to. If `None` (the default) and memory-mapping is enabled then a
                temporary directory will be created and the chain data written to files
                there, with the created files being deleted in this case once the last
                reference to them is closed.
            monitor_stats (Sequence[str]): Sequence of string keys of (integration)
                transition statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default is to
                print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to track the
                completed chain sampling iterations. Default value is `True`, i.e. to
                display progress bar.
            progress_bar_class (mici.progressbars.ProgressBar): Class or factory
                function for progress bar to use to show chain progress if enabled
                (`display_progress=True`). Defaults to
                `mici.progressbars.SequenceProgressBar`.

        Returns:
            final_states (List[ChainState]): States of chains after final iteration. May
                be used to resume sampling a chain by passing as the initial states to a
                new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays. Values in
                dictionary are list of arrays of variables outputted by trace functions
                in `trace_funcs` with each array in the list corresponding to a single
                chain and the leading dimension of each array corresponding to the
                iteration (draw) index (within the main non-adaptive sampling stage if
                `trace_warm_up` is False). The key for each value is the corresponding
                key in the dictionary returned by the trace function which computed the
                traced value.
            stats (Dict[str, List[array]]): Dictionary of chain statistics. Values in
                dictionary are lists of arrays of chain statistic values with each array
                in the list corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index (within the main
                non-adaptive sampling stage if `trace_warm_up` is False). The key for
                each value is a string description of the corresponding integration
                transition statistic.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        # default to single dual-averaging step size adapter
        if "adapters" not in kwargs:
            kwargs["adapters"] = [DualAveragingStepSizeAdapter()]
        # default to tracing position component of state and Hamiltonian
        if "trace_funcs" not in kwargs:
            kwargs["trace_funcs"] = [self._default_trace_func]
        # if `monitor_stats` specified, expand all statistics keys to key pairs
        # with transition key set to `integration_transition`
        if "monitor_stats" in kwargs:
            if kwargs["monitor_stats"] is not None:
                kwargs["monitor_stats"] = {
                    "integration_transition": kwargs["monitor_stats"]
                }
        else:
            kwargs["monitor_stats"] = {"integration_transition": ["accept_stat"]}
        # if adapters kwarg specified, wrap adapter list in dictionary with
        # adapters applied to integration transition
        if "adapters" in kwargs and kwargs["adapters"] is not None:
            kwargs["adapters"] = {"integration_transition": kwargs["adapters"]}
        final_states, traces, stats = super().sample_chains(
            n_warm_up_iter, n_main_iter, init_states, **kwargs
        )
        stats = stats.get("integration_transition", {})
        return final_states, traces, stats


class StaticMetropolisHMC(HamiltonianMCMC):
    """Static integration time H-MCMC implementation with Metropolis sampling.

    In each transition a trajectory is generated by integrating the Hamiltonian dynamics
    from the current state in the current integration time direction for a fixed integer
    number of integrator steps.

    The state at the end of the trajectory with the integration direction negated (this
    ensuring the proposed move is an involution) is used as the proposal in a Metropolis
    acceptance step. The integration direction is then deterministically negated again
    irrespective of the accept decision, with the effect being that on acceptance the
    integration direction will be equal to its initial value and on rejection the
    integration direction will be the negation of its initial value.

    This is original proposed Hybrid Monte Carlo (often now instead termed Hamiltonian
    Monte Carlo) algorithm [1, 2].

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, integrator, rng, n_step, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to use to
                simulate dynamics in integration transition.
            n_step (int): Number of integrator steps to simulate in each integration
                transition.
            momentum_transition (Optional[mici.transitions.MomentumTransition]): Markov
                transition kernel which leaves the conditional distribution on the
                momentum under the canonical distribution invariant, updating only the
                momentum component of the chain state. If set to `None` the momentum
                transition operator `mici.transitions.IndependentMomentumTransition`
                will be used, which independently samples the momentum from its
                conditional distribution.
        """
        integration_transition = trans.MetropolisStaticIntegrationTransition(
            system, integrator, n_step
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def n_step(self):
        """Number of integrator steps per integrator transition."""
        return self.transitions["integration_transition"].n_step

    @n_step.setter
    def n_step(self, value):
        assert value > 0, "n_step must be non-negative"
        self.transitions["integration_transition"].n_step = value


class RandomMetropolisHMC(HamiltonianMCMC):
    """Random integration time H-MCMC with Metropolis sampling of new state.

    In each transition a trajectory is generated by integrating the Hamiltonian dynamics
    from the current state in the current integration time direction for a random
    integer number of integrator steps sampled from the uniform distribution on an
    integer interval.

    The state at the end of the trajectory with the integration direction negated (this
    ensuring the proposed move is an involution) is used as the proposal in a Metropolis
    acceptance step. The integration direction is then deterministically negated again
    irrespective of the accept decision, with the effect being that on acceptance the
    integration direction will be equal to its initial value and on rejection the
    integration direction will be the negation of its initial value.

    The randomisation of the number of integration steps avoids the potential of the
    chain mixing poorly due to using an integration time close to the period of (near)
    periodic systems [1, 2].

    References:

      1. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
      2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
         Physics Letters B, 226(3-4), pp.369-371.
    """

    def __init__(self, system, integrator, rng, n_step_range, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to use to
                simulate dynamics in integration transition.
            n_step_range (Tuple[int, int]): Tuple `(lower, upper)` with two positive
                integer entries `lower` and `upper` (with `upper > lower`) specifying
                respectively the lower and upper bounds (inclusive) of integer interval
                to uniformly draw random number integrator steps to simulate in each
                integration transition.
            momentum_transition (Optional[mici.transitions.MomentumTransition]): Markov
                transition kernel which leaves the conditional distribution on the
                momentum under the canonical distribution invariant, updating only the
                momentum component of the chain state. If set to `None` the momentum
                transition operator `mici.transitions.IndependentMomentumTransition`
                will be used, which independently samples the momentum from its
                conditional distribution.
        """
        integration_transition = trans.MetropolisRandomIntegrationTransition(
            system, integrator, n_step_range
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def n_step_range(self):
        """Interval to uniformly draw number of integrator steps from."""
        return self.transitions["integration_transition"].n_step_range

    @n_step_range.setter
    def n_step_range(self, value):
        n_step_lower, n_step_upper = value
        assert (
            n_step_lower > 0 and n_step_lower < n_step_upper
        ), "Range bounds must be non-negative and first entry less than last"
        self.transitions["integration_transition"].n_step_range = value


class DynamicMultinomialHMC(HamiltonianMCMC):
    """Dynamic integration time H-MCMC with multinomial sampling of new state.

    In each transition a binary tree of states is recursively computed by integrating
    randomly forward and backward in time by a number of steps equal to the previous
    tree size [1, 2] until a termination criteria on the tree leaves is met. The next
    chain state is chosen from the candidate states using a progressive multinomial
    sampling scheme [2] based on the relative probability densities of the different
    candidate states, with the resampling biased towards states further from the current
    state.

    When used with the default settings of `riemannian_no_u_turn_criterion` termination
    criterion and extra subtree checks enabled, this sampler is equivalent to the
    default 'NUTS' MCMC algorithm (minus adaptation) used in
    [Stan](https://mc-stan.org/) as of version v2.23.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler: adaptively setting
         path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research,
         15(1), pp.1593-1623.
      2. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
         arXiv preprint arXiv:1701.02434.
    """

    def __init__(
        self,
        system,
        integrator,
        rng,
        max_tree_depth=10,
        max_delta_h=1000,
        termination_criterion=trans.riemannian_no_u_turn_criterion,
        do_extra_subtree_checks=True,
        momentum_transition=None,
    ):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to use to
                simulate dynamics in integration transition.
            max_tree_depth (int): Maximum depth to expand trajectory binary tree to in
                integrator transition. The maximum number of integrator steps
                corresponds to `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian function
                over a trajectory in integrator transition before signalling a
                divergence.
            termination_criterion (Callable): Function computing criterion to use to
                determine when to terminate trajectory tree expansion. The function
                should take a Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory (sub-)tree being
                checked and an array containing the sum of the momentums over the
                trajectory (sub)-tree. Defaults to
                `mici.transitions.riemannian_no_u_turn_criterion`.
            do_extra_subtree_checks (bool): Whether to perform additional termination
                criterion checks on overlapping subtrees of the current tree to improve
                robustness in systems with dynamics which are well approximated by
                independent system of simple harmonic oscillators. In such systems
                (corresponding to e.g. a standard normal target distribution and
                identity metric matrix representation) at certain step sizes a
                'resonant' behaviour is seen by which the termination criterion fails to
                detect that the trajectory has expanded past a half-period i.e. has
                'U-turned' resulting in trajectories continuing to expand, potentially
                up until the `max_tree_depth` limit is hit. For more details see [this
                Stan Discourse discussion](kutt.it/yAkIES). If `do_extra_subtree_checks`
                is set to `True` additional termination criterion checks are performed
                on overlapping subtrees which help to reduce this resonant behaviour at
                the cost of more conservative trajectory termination in some correlated
                models and some overhead from additional checks.
            momentum_transition (Optional[mici.transitions.MomentumTransition]): Markov
                transition kernel which leaves the conditional distribution on the
                momentum under the canonical distribution invariant, updating only the
                momentum component of the chain state. If set to `None` the momentum
                transition operator `mici.transitions.IndependentMomentumTransition`
                will be used, which independently samples the momentum from its
                conditional distribution.
        """
        integration_transition = trans.MultinomialDynamicIntegrationTransition(
            system,
            integrator,
            max_tree_depth,
            max_delta_h,
            termination_criterion,
            do_extra_subtree_checks,
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def max_tree_depth(self):
        """Maximum depth to expand trajectory binary tree to."""
        return self.transitions["integration_transition"].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        assert value > 0, "max_tree_depth must be non-negative"
        self.transitions["integration_transition"].max_tree_depth = value

    @property
    def max_delta_h(self):
        """Change in Hamiltonian over trajectory to trigger divergence."""
        return self.transitions["integration_transition"].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions["integration_transition"].max_delta_h = value


class DynamicSliceHMC(HamiltonianMCMC):
    """Dynamic integration time H-MCMC with slice sampling of new state.

    In each transition a binary tree of states is recursively computed by integrating
    randomly forward and backward in time by a number of steps equal to the previous
    tree size [1] until a termination criteria on the tree leaves is met. The next chain
    state is chosen from the candidate states using a progressive slice sampling scheme
    [1] based on the relative probability densities of the different candidate states,
    with the sampling biased towards states further from the current state.

    When used with the default setting of `euclidean_no_u_turn_criterion` termination
    criterion and extra subtree checks disabled, this sampler is equivalent to
    'Algorithm 3: Efficient No-U-Turn Sampler' in [1], i.e. the 'classic NUTS'
    algorithm.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
    """

    def __init__(
        self,
        system,
        integrator,
        rng,
        max_tree_depth=10,
        max_delta_h=1000,
        termination_criterion=trans.euclidean_no_u_turn_criterion,
        do_extra_subtree_checks=False,
        momentum_transition=None,
    ):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to use to
                simulate dynamics in integration transition.
            max_tree_depth (int): Maximum depth to expand trajectory binary tree to in
                integrator transition. The maximum number of integrator steps
                corresponds to `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian function
                over a trajectory in integrator transition before signalling a
                divergence.
            termination_criterion (Callable): Function computing criterion to use to
                determine when to terminate trajectory tree expansion. The function
                should take a Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory (sub-)tree being
                checked and an array containing the sum of the momentums over the
                trajectory (sub)-tree. Defaults to
                `mici.transitions.riemannian_no_u_turn_criterion`.
            do_extra_subtree_checks (bool): Whether to perform additional termination
                criterion checks on overlapping subtrees of the current tree to improve
                robustness in systems with dynamics which are well approximated by
                independent system of simple harmonic oscillators. In such systems
                (corresponding to e.g. a standard normal target distribution and
                identity metric matrix representation) at certain step sizes a
                'resonant' behaviour is seen by which the termination criterion fails to
                detect that the trajectory has expanded past a half-period i.e. has
                'U-turned' resulting in trajectories continuing to expand, potentially
                up until the `max_tree_depth` limit is hit. For more details see [this
                Stan Discourse discussion](kutt.it/yAkIES). If `do_extra_subtree_checks`
                is set to `True` additional termination criterion checks are performed
                on overlapping subtrees which help to reduce this resonant behaviour at
                the cost of more conservative trajectory termination in some correlated
                models and some overhead from additional checks.
            momentum_transition (Optional[mici.transitions.MomentumTransition]): Markov
                transition kernel which leaves the conditional distribution on the
                momentum under the canonical distribution invariant, updating only the
                momentum component of the chain state. If set to `None` the momentum
                transition operator `mici.transitions.IndependentMomentumTransition`
                will be used, which independently samples the momentum from its
                conditional distribution.
        """
        integration_transition = trans.SliceDynamicIntegrationTransition(
            system,
            integrator,
            max_tree_depth,
            max_delta_h,
            termination_criterion,
            do_extra_subtree_checks,
        )
        super().__init__(system, rng, integration_transition, momentum_transition)

    @property
    def max_tree_depth(self):
        """Maximum depth to expand trajectory binary tree to."""
        return self.transitions["integration_transition"].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        assert value > 0, "max_tree_depth must be non-negative"
        self.transitions["integration_transition"].max_tree_depth = value

    @property
    def max_delta_h(self):
        """Change in Hamiltonian over trajectory to trigger divergence."""
        return self.transitions["integration_transition"].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions["integration_transition"].max_delta_h = value
