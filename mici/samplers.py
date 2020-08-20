"""Monte Carlo sampler classes for peforming inference."""

import os
import sys
import inspect
import queue
from contextlib import ExitStack, contextmanager
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
from mici.progressbars import (ProgressBar, LabelledSequenceProgressBar,
                               DummyProgressBar, _ProxyProgressBar)
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
    def nullcontext():
        yield None


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
    return ''.join(c for c in string if (c.isalnum() or c in '._- '))


def _generate_memmap_filename(dir_path, prefix, key, index):
    """Generate a new memory-map filename."""
    key_str = _get_valid_filename(str(key))
    return os.path.join(dir_path, f'{prefix}_{index}_{key_str}.npy')


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
    memmap = np.lib.format.open_memmap(
        file_path, dtype=dtype, mode='w+', shape=shape)
    memmap[:] = default_val
    return memmap


def _memmaps_to_file_paths(obj):
    """Convert memmap objects to corresponding file paths.

    Acts recursively on arbitrary 'pytrees' of nested dict/tuple/lists with
    memmap leaves.

    Arg:
        obj: NumPy memmap object or pytree of memmap objects to convert.

    Returns:
        File path string or pytree of file path strings.
    """
    if isinstance(obj, np.memmap):
        return obj.filename
    elif isinstance(obj, dict):
        return {k: _memmaps_to_file_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_memmaps_to_file_paths(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_memmaps_to_file_paths(v) for v in obj)


def _check_and_process_init_state(state, transitions):
    """Check initial chain state is valid and convert dict to ChainState."""
    for trans_key, transition in transitions.items():
        for var_key in transition.state_variables:
            if var_key not in state:
                raise ValueError(
                    f'init_state does contain have {var_key} value required by'
                    f' {trans_key} transition.')
    if not isinstance(state, (ChainState, dict)):
        raise TypeError(
            'init_state should be a dictionary or ChainState.')
    return ChainState(**state) if isinstance(state, dict) else state


def _init_chain_stats(transitions, n_iter, memmap_enabled, memmap_path,
                      chain_index):
    """Initialize dictionary of per-transition chain statistics array dicts."""
    chain_stats = {}
    for trans_key, transition in transitions.items():
        chain_stats[trans_key] = {}
        if transition.statistic_types is not None:
            for key, (dtype, val) in transition.statistic_types.items():
                if memmap_enabled:
                    filename = _generate_memmap_filename(
                        memmap_path, 'stats', f'{trans_key}_{key}',
                        chain_index)
                    chain_stats[trans_key][key] = _open_new_memmap(
                        filename, n_iter, val, dtype)
                else:
                    chain_stats[trans_key][key] = np.full(n_iter, val, dtype)
    return chain_stats


def _init_traces(trace_funcs, init_state, n_iter, memmap_enabled,
                 memmap_path, chain_index):
    """Initialize dictionary of chain trace arrays."""
    traces = {}
    for trace_func in trace_funcs:
        for key, val in trace_func(init_state).items():
            val = np.array(val) if np.isscalar(val) else val
            init = np.nan if np.issubdtype(val.dtype, np.inexact) else 0
            if memmap_enabled:
                filename = _generate_memmap_filename(
                    memmap_path, 'trace', key, chain_index)
                traces[key] = _open_new_memmap(
                    filename, (n_iter,) + val.shape, init, val.dtype)
            else:
                traces[key] = np.full((n_iter,) + val.shape, init, val.dtype)
    return traces


def _get_obj_byte_size(obj, seen=None):
    """Recursively finds size of objects in bytes.

    Original source: https://github.com/bosswissam/pysize

    MIT License

    Copyright (c) [2018] [Wissam Jarjoui]

    Args:
        obj (object): Object to calculate size of.
        seen (None or Set): Set of objects seen so far.

    Returns:
        int: Byte size of `obj`.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if (inspect.isgetsetdescriptor(d) or
                        inspect.ismemberdescriptor(d)):
                    size += _get_obj_byte_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((_get_obj_byte_size(v, seen) for v in obj.values()))
        size += sum((_get_obj_byte_size(k, seen) for k in obj.keys()))
    elif (hasattr(obj, '__iter__') and not
            isinstance(obj, (str, bytes, bytearray))):
        size += sum((_get_obj_byte_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(_get_obj_byte_size(getattr(obj, s), seen)
                    for s in obj.__slots__ if hasattr(obj, s))

    return size


def _check_chain_data_size(traces, chain_stats):
    """Check that byte size of sample chain calls below pickle limit."""
    total_return_nbytes = (
        _get_obj_byte_size(traces) + _get_obj_byte_size(chain_stats))
    # Check if total number of bytes to be returned exceeds pickle limit
    if total_return_nbytes > 2**31 - 1:
        raise RuntimeError(
            f'Total number of bytes allocated for chain data to be returned '
            f'({total_return_nbytes / 2**30:.2f} GiB) exceeds size limit for '
            f'returning results of a process (2 GiB). Try rerunning with '
            f' memory-mapping enabled (`memmap_enabled=True`).')


def _construct_chain_iterators(
        n_iter, chain_iterator_class, n_chain=None, position_offset=0):
    """Set up chain iterator progress bar object(s)."""
    if n_chain is None:
        return chain_iterator_class(range(n_iter), description='Chain 1/1')
    else:
        return [
            chain_iterator_class(
                range(n_iter), description=f'Chain {c+1}/{n_chain}',
                position=(c + position_offset, n_chain + position_offset))
            for c in range(n_chain)]


def _update_chain_stats(sample_index, chain_stats, trans_key, trans_stats):
    """Update chain statistics arrays for current chain iteration."""
    if trans_stats is not None:
        if sample_index == 0 and trans_key not in chain_stats:
            raise KeyError(
                f'Transition {trans_key} returned statistics but has no '
                f'statistic_types attribute.')
        for key, val in trans_stats.items():
            if sample_index == 0 and key not in chain_stats[trans_key]:
                raise KeyError(
                    f'Transition {trans_key} returned {key} statistic but it '
                    f'is not included in its statistic_types attribute.')
            chain_stats[trans_key][key][sample_index] = val


def _update_monitor_stats(
        sample_index, chain_stats, monitor_stats, monitor_dict):
    """Update dictionary of per-iteration monitored statistics."""
    for (trans_key, stats_key) in monitor_stats:
        if sample_index == 0 and not (trans_key in chain_stats and
                                      stats_key in chain_stats[trans_key]):
            raise KeyError(
                f'Statistics key pair {(trans_key, stats_key)} to be monitored'
                'is not present in chain statistics returned by transitions.')
        val = chain_stats[trans_key][stats_key][sample_index]
        if stats_key not in monitor_dict:
            monitor_dict[stats_key] = val
        else:
            monitor_dict[f'{trans_key}.{stats_key}'] = val


def _flush_memmap_chain_data(traces, chain_stats):
    """Flush all pending writes to memory-mapped chain data arrays to disk."""
    for trace in traces.values():
        trace.flush()
    for trans_stats in chain_stats.values():
        for stat in trans_stats.values():
            stat.flush()


def _try_resize_dim_0_inplace(array, new_shape_0):
    """Try to resize 0-axis of array in-place or return a view otherwise."""
    if new_shape_0 >= array.shape[0]:
        return array
    try:
        # Try to truncate arrays by resizing in place
        array.resize((new_shape_0,) + array.shape[1:])
        return array
    except ValueError:
        # In place resize not possible therefore return truncated view
        return array[:new_shape_0]


def _truncate_chain_data(sample_index, traces, chain_stats):
    """Truncate first dimension of chain arrays to sample_index < n_iter."""
    for key in traces:
        traces[key] = _try_resize_dim_0_inplace(traces[key], sample_index)
    for trans_stats in chain_stats.values():
        for key in trans_stats:
            trans_stats[key] = _try_resize_dim_0_inplace(
                trans_stats[key], sample_index)


def _sample_chain(init_state, chain_iterator, rng, transitions, trace_funcs,
                  chain_index=0, parallel_chains=False, memmap_enabled=False,
                  memmap_path=None, monitor_stats=None, adapters=None):
    """Sample a chain by iteratively appyling a sequence of transition kernels.

    Args:
        init_state (mici.states.ChainState or Dict[str, object]): Initial chain
            state. Either a `mici.states.ChainState` object or a dictionary
            with entries specifying initial values for all state variables used
            by chain transition `sample` methods.
        chain_iterator (Iterable[Tuple[int, Dict]]): Iterable object which
            is iterated over to produce sample indices and (empty) iteration
            statistic dictionaries to output monitored chain statistics to
            during sampling.
        rng (numpy.random.Generator): Numpy random number generator.
        transitions (OrderedDict[str, Transition]): Ordered dictionary of
            Markov transitions kernels to sequentially sample from on each
            chain iteration.
        trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
            List of functions which compute the variables to be recorded at
            each chain iteration, with each trace function being passed the
            current state and returning a dictionary of scalar or array values
            corresponding to the variable(s) to be stored. The keys in the
            returned dictionaries are used to index the trace arrays in the
            returned traces dictionary. If a key appears in multiple
            dictionaries only the the value corresponding to the last trace
            function to return that key will be stored.
        chain_index (int): Identifier for chain when sampling multiple chains.
        parallel_chains (bool): Whether multiple chains are being sampled in
            parallel.
        memmap_enabled (bool): Whether to memory-map arrays used to store chain
            data to files on disk to avoid excessive system memory usage for
            long chains and/or large chain states. The chain data is written to
            `.npy` files in the directory specified by `memmap_path` (or a
            temporary directory if not provided). These files persist after the
            termination of the function so should be manually deleted when no
            longer required. Default is to for memory mapping to be disabled.
        memmap_path (str): Path to directory to write memory-mapped chain data
            to. If not provided, a temporary directory will be created and the
            chain data written to files there.
        monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string key
            pairs, with first entry the key of a Markov transition in the
            `transitions` dict passed to the the `__init__` method and the
            second entry the key of a chain statistic that will be returned in
            the `chain_stats` dictionary. The mean over samples computed so far
            of the chain statistics associated with any valid key-pairs will be
            monitored during sampling by printing as postfix to progress bar.
        adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
            `mici.adapters.Adapter` instances keyed by strings corresponding to
            the key of the transition in the `transitions` dictionary to apply
            the adapters to. Each adapter is able to adaptatively set the
            parameters of a transition while sampling a chain. Note that the
            adapter updates for each transition are applied in the order the
            adapters appear in the iterable and so if multiple adapters change
            the same parameter(s) the order will matter. Adaptation based on the
            chain state history breaks the Markov property and so any chain
            samples while adaptation is active should not be used in estimates
            of expectations.

    Returns:
        final_state (mici.states.ChainState): State of chain after final
            iteration. May be used to resume sampling a chain by passing as the
            initial state to a new `sample_chain` call.
        traces (Dict[str, array]): Dictionary of chain trace arrays. Values in
            dictionary are arrays of variables outputted by trace functions in
            `trace_funcs` with leading dimension of array corresponding to the
            sampling (draw) index. The key for each value is the corresponding
            key in the dictionary returned by the trace function which computed
            the traced value.
        chain_stats (Dict[str, Dict[str, array]]): Dictionary of chain
            transition statistic dictionaries. Values in outer dictionary are
            dictionaries of statistics for each chain transition, keyed by the
            string key for the transition. The values in each inner transition
            dictionary are arrays of chain statistic values with the leading
            dimension of each array corresponding to the sampling (draw) index.
            The key for each value is a string description of the corresponding
            integration transition statistic.
        exception (None or Exception): Any handled exception which may affect
            how the returned outputs are processed by the caller.
    """
    state = _check_and_process_init_state(init_state, transitions)
    n_iter = len(chain_iterator)
    # Create temporary directory if memory mapping and no path provided
    if memmap_enabled and memmap_path is None:
        memmap_path = tempfile.mkdtemp()
    chain_stats = _init_chain_stats(
        transitions, n_iter, memmap_enabled, memmap_path, chain_index)
    traces = _init_traces(
        trace_funcs, state, n_iter, memmap_enabled, memmap_path, chain_index)
    adapter_states = {}
    try:
        if adapters is not None:
            for trans_key, adapter_list in adapters.items():
                adapter_states[trans_key] = []
                for adapter in adapter_list:
                    adapter_states[trans_key].append(
                        adapter.initialize(state, transitions[trans_key]))
    except AdaptationError as exception:
        logger.error(
            f'Initialisation of {type(adapter).__name__} for chain '
            f'{chain_index + 1} failed: {exception}')
        return state, traces, chain_stats, adapter_states, exception
    try:
        sample_index = 0
        if parallel_chains and not memmap_enabled:
            _check_chain_data_size(traces, chain_stats)
        with chain_iterator:
            for sample_index, monitor_dict in chain_iterator:
                for trans_key, transition in transitions.items():
                    state, trans_stats = transition.sample(state, rng)
                    if adapters is not None and trans_key in adapters:
                        for adapter, adapter_state in zip(
                                adapters[trans_key], adapter_states[trans_key]):
                            adapter.update(
                                adapter_state, state, trans_stats, transition)
                    _update_chain_stats(
                        sample_index, chain_stats, trans_key, trans_stats)
                for trace_func in trace_funcs:
                    for key, val in trace_func(state).items():
                        traces[key][sample_index] = val
                if monitor_stats is not None:
                    _update_monitor_stats(
                        sample_index, chain_stats, monitor_stats, monitor_dict)
    except KeyboardInterrupt as e:
        exception = e
        logger.error(
            f'Sampling manually interrupted for chain {chain_index + 1} at'
            f' iteration {sample_index}. Arrays containing chain traces and'
            f' statistics computed before interruption will be returned.')
        # Sampling interrupted therefore truncate returned arrays unless using
        # memory mapping with parallel chains as will only return file paths
        if not (parallel_chains and memmap_enabled):
            _truncate_chain_data(sample_index, traces, chain_stats)
    else:
        exception = None
    if memmap_enabled:
        _flush_memmap_chain_data(traces, chain_stats)
    if parallel_chains and memmap_enabled:
        traces = _memmaps_to_file_paths(traces)
        chain_stats = _memmaps_to_file_paths(chain_stats)
    return state, traces, chain_stats, adapter_states, exception


def _collate_chain_outputs(chain_outputs):
    """Unzip list of tuples of chain outputs in to tuple of lists of outputs.

    As well as collating chain outputs, any string file path values
    corresponding to memmap file paths are swapped for the corresponding
    memmap objects.
    """
    final_states_stack = []
    traces_stack = {}
    stats_stack = {}
    adapt_states_stack = {}
    for chain_index, (final_state, traces,
                      stats, adapt_states) in enumerate(chain_outputs):
        final_states_stack.append(final_state)
        for key, val in traces.items():
            # if value is string => file path to memory mapped array
            if isinstance(val, str):
                val = np.lib.format.open_memmap(val)
            if chain_index == 0:
                traces_stack[key] = [val]
            else:
                traces_stack[key].append(val)
        for trans_key, trans_stats in stats.items():
            if chain_index == 0:
                stats_stack[trans_key] = {}
            for key, val in trans_stats.items():
                # if value is string => file path to memory mapped array
                if isinstance(val, str):
                    val = np.lib.format.open_memmap(val)
                if chain_index == 0:
                    stats_stack[trans_key][key] = [val]
                else:
                    stats_stack[trans_key][key].append(val)
        for trans_key, adapt_state_list in adapt_states.items():
            if trans_key not in adapt_states_stack:
                adapt_states_stack[trans_key] = [[a] for a in adapt_state_list]
            else:
                for i, adapt_state in enumerate(adapt_state_list):
                    adapt_states_stack[trans_key][i].append(adapt_state)
    return final_states_stack, traces_stack, stats_stack, adapt_states_stack


def _get_per_chain_rngs(base_rng, n_chain):
    """Construct random number generators (RNGs) for each of a set of chains.

    If the base RNG bit generator has a `jumped` method this is used to produce
    a sequence of independent random substreams. Otherwise if the base RNG bit
    generator has a `_seed_seq` attribute this is used to spawn a sequence off
    generators.
    """
    if hasattr(base_rng, 'bit_generator'):
        bit_generator = base_rng.bit_generator
    elif hasattr(base_rng, '_bit_generator'):
        bit_generator = base_rng._bit_generator
    else:
        bit_generator = None
    if bit_generator is not None and hasattr(bit_generator, 'jumped'):
        return [default_rng(bit_generator.jumped(i)) for i in range(n_chain)]
    elif bit_generator is not None and hasattr(bit_generator, '_seed_seq'):
        seed_sequence = bit_generator._seed_seq
        return [default_rng(seed) for seed in seed_sequence.spawn(n_chain)]
    else:
        raise ValueError(
            f'Unsupported random number generator type {type(base_rng)}.')


def _sample_chains_sequential(init_states, rngs, chain_iterators, **kwargs):
    """Sample multiple chains sequentially in a single process."""
    chain_outputs = []
    exception = None
    for chain_index, (init_state, rng, chain_iterator) in enumerate(
            zip(init_states, rngs, chain_iterators)):
        *outputs, exception = _sample_chain(
            chain_iterator=chain_iterator, init_state=init_state, rng=rng,
            chain_index=chain_index, parallel_chains=False, **kwargs)
        # Returned exception being AdaptationError indicates chain terminated
        # due to adapter initialisation failing therefore do not store returned
        # chain outputs
        if not isinstance(exception, AdaptationError):
            chain_outputs.append(outputs)
        # If returned handled exception was a manual interrupt break and return
        if isinstance(exception, KeyboardInterrupt):
            break
    return (*_collate_chain_outputs(chain_outputs), exception)


def _sample_chains_worker(chain_queue, iter_queue):
    """Worker process function for parallel sampling of chains.

    Consumes chain arguments from a shared queue and outputs chain progress
    updates to a second shared queue.
    """
    chain_outputs = []
    while not chain_queue.empty():
        try:
            chain_index, init_state, rng, n_iter, kwargs = chain_queue.get(
                block=False)
            max_threads = kwargs.pop('max_threads_per_process', None)
            context = (threadpool_limits(limits=max_threads)
                       if THREADPOOLCTL_AVAILABLE else nullcontext())
            with context:
                *outputs, exception = _sample_chain(
                    init_state=init_state, rng=rng, chain_index=chain_index,
                    chain_iterator=_ProxyProgressBar(
                        range(n_iter), chain_index, iter_queue),
                    parallel_chains=True, **kwargs)
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
            logger.error('Exception encountered in chain worker process:',
                         exc_info=exception)
            # Put exception on iteration queue to be reraised in parent process
            iter_queue.put(exception)
    return chain_outputs


def _finalize_adapters(adapter_states, adapters, transitions):
    """Finalize adapter updates to transitions based on final adapter states."""
    for trans_key, adapter_state_list in adapter_states.items():
        for adapter_state, adapter in zip(
                adapter_state_list, adapters[trans_key]):
            adapter.finalize(adapter_state, transitions[trans_key])


def _sample_chains_parallel(init_states, rngs, chain_iterators, n_process,
                            **kwargs):
    """Sample multiple chains in parallel over multiple processes."""
    n_iters = [len(it) for it in chain_iterators]
    n_chain = len(chain_iterators)
    with _ignore_sigint_manager() as manager, Pool(n_process) as pool:
        results = None
        exception = None
        try:
            # Shared queue for workers to output chain progress updates to
            iter_queue = manager.Queue()
            # Shared queue for workers to get arguments for _sample_chain calls
            # from on initialising each chain
            chain_queue = manager.Queue()
            for c, (init_state, rng, n_iter) in enumerate(
                    zip(init_states, rngs, n_iters)):
                chain_queue.put((c, init_state, rng, n_iter, kwargs))
            # Start n_process worker processes which each have access to the
            # shared queues, returning results asynchronously
            results = pool.starmap_async(
                _sample_chains_worker,
                [(chain_queue, iter_queue) for p in range(n_process)])
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
                            'Unhandled exception in chain worker process.'
                        ) from iter_queue_item
                    else:
                        # Otherwise unpack and update progress bar
                        chain_index, sample_index, data_dict = iter_queue_item
                        pbars[chain_index].update(sample_index, data_dict)
                        if sample_index == n_iters[chain_index]:
                            chains_completed += 1
        except (PicklingError, AttributeError) as e:
            if not MULTIPROCESS_AVAILABLE and (
                    isinstance(e, PicklingError) or 'pickle' in str(e)):
                raise RuntimeError(
                    'Error encountered while trying to run chains on multiple'
                    'processes in parallel. The inbuilt multiprocessing module'
                    ' uses pickle to communicate between processes and pickle '
                    'does support pickling anonymous or nested functions. If '
                    'you use anonymous or nested functions in your model '
                    'functions or are using autograd to automatically compute '
                    'derivatives (autograd uses anonymous and nested '
                    'functions) then installing the Python package '
                    'multiprocess, which is able to serialise anonymous and '
                    'nested functions and will be used in preference to '
                    'multiprocessing by this package when available, may '
                    'resolve this error.'
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
            warn('Use of numpy.random.RandomState random number generators is '
                 'deprecated. Please use a numpy.random.Generator instance '
                 'instead for example from a call to numpy.random.default_rng.',
                 DeprecationWarning)
            rng = np.random.Generator(rng._bit_generator)
        self.rng = rng
        self.transitions = transitions

    def __set_sample_chain_kwargs_defaults(self, kwargs):
        if 'memmap_enabled' not in kwargs:
            kwargs['memmap_enabled'] = False
        if kwargs['memmap_enabled'] and kwargs.get('memmap_path') is None:
            kwargs['memmap_path'] = tempfile.mkdtemp()
        display_progress = kwargs.pop('display_progress', True)
        if not display_progress:
            kwargs['progress_bar_class'] = DummyProgressBar
        elif 'progress_bar_class' not in kwargs:
            kwargs['progress_bar_class'] = ProgressBar

    def sample_chain(self, n_iter, init_state, trace_funcs, **kwargs):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_state (mici.states.ChainState or Dict[str, object]): Initial
                chain state. Either a `mici.states.ChainState` object or a
                dictionary with entries specifying initial values for all state
                variables used by chain transition `sample` methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.

        Kwargs:
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
                `mici.adapters.Adapter` instances keyed by strings corresponding
                to the key of the transition in the `transitions` dictionary to
                apply the adapters to. Each adapter is able to adaptatively set
                the parameters of a transition while sampling a chain. Note that
                the adapter updates for each transition are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations.

        Returns:
            final_state (mici.states.ChainState): State of chain after final
                iteration. May be used to resume sampling a chain by passing as
                the initial state to a new `sample_chain` call.
            traces (Dict[str, array]): Dictionary of chain trace arrays. Values
                in dictionary are arrays of variables outputted by trace
                functions in `trace_funcs` with leading dimension of array
                corresponding to the sampling (draw) index. The key for each
                value is the corresponding key in the dictionary returned by
                the trace function which computed the traced value.
            chain_stats (Dict[str, Dict[str, array]]): Dictionary of chain
                transition statistic dictionaries. Values in outer dictionary
                are dictionaries of statistics for each chain transition, keyed
                by the string key for the transition. The values in each inner
                transition dictionary are arrays of chain statistic values with
                the leading dimension of each array corresponding to the
                sampling (draw) index. The key for each value is a string
                description of the corresponding integration transition
                statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        chain_iterator = _construct_chain_iterators(
            n_iter, kwargs.pop('progress_bar_class'))
        final_state, traces, chain_stats, adapter_states, _ = _sample_chain(
            init_state=init_state, chain_iterator=chain_iterator,
            transitions=self.transitions, rng=self.rng,
            trace_funcs=trace_funcs, parallel_chains=False, **kwargs)
        if len(adapter_states) > 0:
            _finalize_adapters(
                adapter_states, kwargs['adapters'], self.transitions)

        return final_state, traces, chain_stats

    def sample_chains(self, n_iter, init_states, trace_funcs, n_process=1,
                      **kwargs):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_states (Iterable[ChainState] or Iterable[Dict[str, object]]):
                Initial chain states. Each entry can be either a `ChainState`
                object or a dictionary with entries specifying initial values
                for all state variables used by chain transition `sample`
                methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.
            n_process (int or None): Number of parallel processes to run chains
                over. If set to one then chains will be run sequentially in
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If
                set to `None` then the number of processes will default to the
                output of `os.cpu_count()`.

        Kwargs:
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling  by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables of
                `mici.adapters.Adapter` instances keyed by strings corresponding
                to the key of the transition in the `transitions` dictionary to
                apply the adapters to. Each adapter is able to adaptatively set
                the parameters of a transition while sampling a chain. Note that
                the adapter updates for each transition are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the sampling (draw) index. The key
                for each value is the corresponding key in the dictionary
                returned by the trace function which computed the traced value.
            chain_stats (Dict[str, Dict[str, List[array]]]): Dictionary of
                chain transition statistic dictionaries. Values in outer
                dictionary are dictionaries of statistics for each chain
                transition, keyed by the string key for the transition. The
                values in each inner transition dictionary are lists of arrays
                of chain statistic values with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the sampling (draw) index. The key
                for each value is a string description of the corresponding
                integration transition statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        n_chain = len(init_states)
        rngs = _get_per_chain_rngs(self.rng, n_chain)
        chain_iterators = _construct_chain_iterators(
            n_iter, kwargs.pop('progress_bar_class'), n_chain)
        if n_process == 1:
            # Using single process therefore run chains sequentially
            kwargs.pop('max_threads_per_process', None)
            *states_traces_stats, adapter_states, _ = _sample_chains_sequential(
                init_states=init_states, rngs=rngs,
                chain_iterators=chain_iterators, transitions=self.transitions,
                trace_funcs=trace_funcs, **kwargs)
        else:
            # Run chains in parallel using a multiprocess(ing).Pool
            *states_traces_stats, adapter_states, _ = _sample_chains_parallel(
                init_states=init_states, rngs=rngs,
                chain_iterators=chain_iterators, transitions=self.transitions,
                trace_funcs=trace_funcs, n_process=n_process, **kwargs)
        if len(adapter_states) > 0:
            _finalize_adapters(
                adapter_states, kwargs['adapters'], self.transitions)
        return states_traces_stats

    def sample_chains_with_adaptive_warm_up(
            self, n_warm_up_iter, n_main_iter, init_states, trace_funcs,
            adapters, stager=None, n_process=1, **kwargs):
        """Sample Markov chains from given initial states with adaptive warm up.

        One or more Markov chains are sampled, with each chain iteration
        consisting of one or more Markov transitions. The chains are split into
        multiple *stages* with one or more adaptive warm up stages followed by
        the main non-adaptive sampling stage. During the adaptive stage(s)
        parameters of the transition(s) are adaptively tuned based on the chain
        state and/or transition statistics.

        The chains (including both adaptive and non-adaptive stages) may be run
        in parallel across multiple independent processes or sequentially. In
        all cases all chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per
                chain. Depending on the `mici.stagers.Stager` instance specified
                by the `stage arguments the warm up iterations may be split
                between one or more adaptive stages.
            n_main_iter (int): Number of iterations (samples to draw) per chain
                during main (non-adaptive) sampling stage.
            init_states (Iterable[ChainState] or Iterable[Dict[str, object]]):
                Initial chain states. Each entry can be either a `ChainState`
                object or a dictionary with entries specifying initial values
                for all state variables used by chain transition `sample`
                methods.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration during the main non-adaptive sampling
                stage, with each trace function being passed the current state
                and returning a dictionary of scalar or array values
                corresponding to the variable(s) to be stored. The keys in the
                returned dictionaries are used to index the trace arrays in the
                returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.
            adapters (Dict[str, Iterable[Adapter]): Dictionary of iterables
                of `mici.adapters.Adapter` instances keyed by strings
                corresponding to the key of the transition in the `transitions`
                dictionary to apply the adapters to, to use to adaptatively set
                parameters of the transitions during the adaptive stages of  the
                chains. Note that the adapter updates are applied in the order
                the adapters appear in the iterables and so if multiple adapters
                change the same parameter(s) the order will matter.

        Kwargs:
            stager (mici.stagers.Stager or None): Chain iteration stager object
                which controls the split of the chain iterations into the
                adaptive warm up and non-adaptive main stages. If set to `None`
                (the default) and all adapters specified by the `adapters`
                argument are of the fast type (i.e. their `is_fast` attribute is
                `True`) then a `mici.stagers.WarmUpStager` instance will be used
                corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by
                the adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments)
                will be used, corresponding to using multiple adaptive warm up
                stages with only the fast-type adapters active in some - see
                docstring of `mici.stagers.WarmUpStager` for details.
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is to
                for memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[Tuple[str, str]]): List of tuples of string
                key pairs, with first entry the key of a Markov transition in
                the `transitions` dict passed to the the `__init__` method and
                the second entry the key of a chain statistic that will be
                returned in the `chain_stats` dictionary. The mean over samples
                computed so far of the chain statistics associated with any
                valid key-pairs will be monitored during sampling  by printing
                as postfix to progress bar.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index in the
                main non-adaptive sampling stage. The key for each value is the
                corresponding key in the dictionary returned by the trace
                function which computed the traced value.
            chain_stats (Dict[str, Dict[str, List[array]]]): Dictionary of
                chain transition statistic dictionaries. Values in outer
                dictionary are dictionaries of statistics for each chain
                transition, keyed by the string key for the transition. The
                values in each inner transition dictionary are lists of arrays
                of chain statistic values with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index in the
                main non-adaptive sampling stage. The key for each value is a
                string description of the corresponding integration transition
                statistic.
        """
        self.__set_sample_chain_kwargs_defaults(kwargs)
        n_chain = len(init_states)
        rngs = _get_per_chain_rngs(self.rng, n_chain)
        progress_bar_class = kwargs.pop('progress_bar_class')
        common_sample_chains_kwargs = {
            'rngs': rngs, 'transitions': self.transitions, **kwargs}
        if n_process > 1:
            sample_chains_func = _sample_chains_parallel
            common_sample_chains_kwargs['n_process'] = n_process
        else:
            sample_chains_func = _sample_chains_sequential
            common_sample_chains_kwargs.pop('max_threads_per_process', None)
        if stager is None:
            if all(a.is_fast for a_list in adapters.values() for a in a_list):
                stager = WarmUpStager()
            else:
                stager = WindowedWarmUpStager()
        sampling_stages = stager.stages(
            n_warm_up_iter, n_main_iter, adapters, trace_funcs)
        chain_states = init_states
        with LabelledSequenceProgressBar(
                    sampling_stages, 'Sampling stage', position=(0, n_chain + 1)
                ) as sampling_stages_pb:
            chain_iterators = _construct_chain_iterators(
                n_warm_up_iter, progress_bar_class, n_chain, 1)
            for stage, _ in sampling_stages_pb:
                for chain_it in chain_iterators:
                    chain_it.sequence = range(stage.n_iter)
                chain_states, traces, stats, adapter_states, exception = (
                    sample_chains_func(
                        init_states=chain_states, trace_funcs=stage.trace_funcs,
                        adapters=stage.adapters,
                        chain_iterators=chain_iterators,
                        **common_sample_chains_kwargs))
                if len(adapter_states) > 0:
                    _finalize_adapters(
                        adapter_states, stage.adapters, self.transitions)
                if isinstance(exception, KeyboardInterrupt):
                    return chain_states, traces, stats
        return chain_states, traces, stats


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
    probability density of the target-momentum state pair; one state from the
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
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integration_transition (mici.transitions.IntegrationTransition):
                Markov transition kernel which leaves canonical distribution
                invariant and jointly updates the position and momentum
                components of the chain state by integrating the Hamiltonian
                dynamics of the system to propose new values for the state.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        self.system = system
        self.rng = rng
        if momentum_transition is None:
            momentum_transition = trans.IndependentMomentumTransition(system)
        super().__init__(rng, OrderedDict(
            momentum_transition=momentum_transition,
            integration_transition=integration_transition))

    def _preprocess_init_state(self, init_state):
        """Make sure initial state is a ChainState and has momentum."""
        if isinstance(init_state, np.ndarray):
            # If array use to set position component of new ChainState
            init_state = ChainState(pos=init_state, mom=None, dir=1)
        elif not isinstance(init_state, ChainState) or 'mom' not in init_state:
            raise TypeError(
                'init_state should be an array or `ChainState` with '
                '`mom` attribute.')
        if init_state.mom is None:
            init_state.mom = self.system.sample_momentum(init_state, self.rng)
        return init_state

    def _default_trace_func(self, state):
        """Default function of the chain state traced while sampling."""
        # This needs to be a method rather than for example a local nested
        # function in the __set_sample_chain_kwargs_defaults method to ensure
        # that it remains pickleable and so can be piped to a separate process
        # when running multiple chains using multiprocessing
        return {'pos': state.pos, 'hamiltonian': self.system.h(state)}

    def __set_sample_chain_kwargs_defaults(self, kwargs):
        # default to tracing position component of state and Hamiltonian
        if 'trace_funcs' not in kwargs:
            kwargs['trace_funcs'] = [self._default_trace_func]
        # if `monitor_stats` specified, expand all statistics keys to key pairs
        # with transition key set to `integration_transition`
        if 'monitor_stats' in kwargs:
            kwargs['monitor_stats'] = [
                ('integration_transition', stats_key)
                for stats_key in kwargs['monitor_stats']]
        else:
            kwargs['monitor_stats'] = [
                ('integration_transition', 'accept_stat')]
        # if adapters kwarg specified, wrap adapter list in dictionary with
        # adapters applied to integration transition
        if 'adapters' in kwargs and kwargs['adapters'] is not None:
            kwargs['adapters'] = {'integration_transition': kwargs['adapters']}

    def sample_chain(self, n_iter, init_state, **kwargs):
        """Sample a Markov chain from a given initial state.

        Performs a specified number of chain iterations (each of which may be
        composed of multiple individual Markov transitions), recording the
        outputs of functions of the sampled chain state after each iteration.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_state (mici.states.ChainState or array): Initial chain state.
                The state can be either an array specifying the state position
                component or a `mici.states.ChainState` instance. If an array
                is passed or the `mom` attribute of the state is not set, a
                momentum component will be independently sampled from its
                conditional distribution.

        Kwargs:
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array ]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored. Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state
                under the key `hamiltonian`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided, a temporary directory will be created
                and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Iterable[Adapter]): Sequence of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size while sampling a
                chain. Note that the adapter updates are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations. Default is to use no
                adapters.

        Returns:
            final_state (mici.states.ChainState): State of chain after final
                iteration. May be used to resume sampling a chain by passing as
                the initial state to a new `sample_chain` call.
            traces (Dict[str, array]): Dictionary of chain trace arrays. Values
                in dictionary are arrays of variables outputted by trace
                functions in `trace_funcs` with leading dimension of array
                corresponding to the sampling (draw) index. The key for each
                value is the corresponding key in the dictionary returned by
                the trace function which computed the traced value.
            chain_stats (Dict[str, array]): Dictionary of chain integration
                transition statistics. Values in dictionary are arrays of chain
                statistic values with the leading dimension of each array
                corresponding to the sampling (draw) index. The key for each
                value is a string description of the corresponding integration
                transition statistic.
        """
        init_state = self._preprocess_init_state(init_state)
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_state, traces, chain_stats = super().sample_chain(
            n_iter, init_state, **kwargs)
        chain_stats = chain_stats.get('integration_transition', {})
        return final_state, traces, chain_stats

    def sample_chains(self, n_iter, init_states, **kwargs):
        """Sample one or more Markov chains from given initial states.

        Performs a specified number of chain iterations, each of consisting of a
        momentum transition followed by an integration transition, recording the
        outputs of functions of the sampled chain state after each iteration.
        The chains may be run in parallel across multiple independent processes
        or sequentially. In all cases all chains use independent random draws.

        Args:
            n_iter (int): Number of iterations (samples to draw) per chain.
            init_states (Iterable[ChainState] or Iterable[array]): Initial
                chain states. Each state can be either an array specifying the
                state position component or a `mici.states.ChainState`
                instance. If an array is passed or the `mom` attribute of the
                state is not set, a momentum component will be independently
                sampled from its conditional distribution. One chain will be
                run for each state in the iterable sequence.

        Kwargs:
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration, with each trace function being passed the
                current state and returning a dictionary of scalar or array
                values corresponding to the variable(s) to be stored. The keys
                in the returned dictionaries are used to index the trace arrays
                in the returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.  Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state
                under the key `hamiltonian`.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided (the default), a temporary directory
                will be created and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.
            adapters (Iterable[Adapter]): Sequence of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size while sampling a
                chain. Note that the adapter updates are applied in the order
                the adapters appear in the iterable and so if multiple adapters
                change the same parameter(s) the order will matter. Adaptation
                based on the chain state history breaks the Markov property and
                so any chain samples while adaptation is active should not be
                used in estimates of expectations. Default is to use no
                adapters.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index. The key
                for each value is the corresponding key in the dictionary
                returned by the trace function which computed the traced value.
            chain_stats (Dict[str, List[array]]): Dictionary of chain
                integration transition statistics. Values in dictionary are
                lists of arrays of chain statistic values with each array in
                the list corresponding to a single chain and the leading
                dimension of each array corresponding to the iteration (draw)
                index. The key for each value is a string description of the
                corresponding integration transition statistic.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_states, traces, chain_stats = super().sample_chains(
            n_iter, init_states, **kwargs)
        chain_stats = chain_stats.get('integration_transition', {})
        return final_states, traces, chain_stats

    def sample_chains_with_adaptive_warm_up(
            self, n_warm_up_iter, n_main_iter, init_states, **kwargs):
        """Sample Markov chains from given initial states with adaptive warm up.

        One or more Markov chains are sampled, with each chain iteration
        consisting of a momentum transition followed by an integration
        transition. The chains are split into multiple *stages* with one or more
        adaptive warm up stages followed by the main non-adaptive sampling
        stage. During the adaptive stage(s) parameters of the integration
        transition such as the integrator step size are adaptively tuned based
        on the chain state and/or transition statistics.

        The default settings use a single (fast) `DualAveragingStepSizeAdapter`
        adapter instance which adapts the integrator step-size using a
        dual-averaging algorithm in a single adaptive stage.

        The chains (including both adaptive and non-adaptive stages) may be run
        in parallel across multiple independent processes or sequentially. In
        all cases all chains use independent random draws.

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per
                chain. Depending on the `mici.stagers.Stager` instance specified
                by the `stage arguments the warm up iterations may be split
                between one or more adaptive stages.
            n_main_iter (int): Number of iterations (samples to draw) per chain
                during main (non-adaptive) sampling stage.
            init_states (Iterable[ChainState] or Iterable[array]): Initial
                chain states. Each state can be either an array specifying the
                state position component or a `mici.states.ChainState`
                instance. If an array is passed or the `mom` attribute of the
                state is not set, a momentum component will be independently
                sampled from its conditional distribution. One chain will be
                run for each state in the iterable sequence.

        Kwargs:
            n_process (int or None): Number of parallel processes to run chains
                over. If `n_process=1` then chains will be run sequentially
                otherwise a `multiprocessing.Pool` object will be used to
                dynamically assign the chains across multiple processes. If set
                to `None` then the number of processes will be set to the
                output of `os.cpu_count()`. Default is `n_process=1`.
            max_threads_per_process (int or None): If `threadpoolctl` is
                available this argument may be used to limit the maximum number
                of threads that can be used in thread pools used in libraries
                supported by `threadpoolctl`, which include BLAS and OpenMP
                implementations. This argument will only have an effect if
                `n_process > 1` such that chains are being run on multiple
                processes and only if `threadpoolctl` is installed in the
                current Python environment. If set to `None` (the default) no
                limits are set.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration during the main non-adaptive sampling
                stage, with each trace function being passed the current state
                and returning a dictionary of scalar or array values
                corresponding to the variable(s) to be stored. The keys in the
                returned dictionaries are used to index the trace arrays in the
                returned traces dictionary. If a key appears in multiple
                dictionaries only the the value corresponding to the last trace
                function to return that key will be stored.  Default is to use a
                single function which recordes the position component of the
                state under the key `pos` and the Hamiltonian at the state under
                the key `hamiltonian`.
            adapters (Iterable[Adapter]): List of `mici.adapters.Adapter`
                instances to use to adaptatively set parameters of the
                integration transition such as the step size during the adaptive
                stages of the chains. Note that the adapter updates are applied
                in the order the adapters appear in the iterable and so if
                multiple adapters change the same parameter(s) the order will
                matter. Default is to use a single instance of
                `mici.adapters.DualAveragingStepSizeAdapter` with its default
                parameters.
            stager (mici.stagers.Stager or None): Chain iteration stager object
                which controls the split of the chain iterations into the
                adaptive warm up and non-adaptive main stages. If set to `None`
                (the default) and all adapters specified by the `adapters`
                argument are of the fast type (i.e. their `is_fast` attribute is
                `True`) then a `mici.stagers.WarmUpStager` instance will be used
                corresponding to using a single adaptive warm up stage will all
                adapters active. If set to `None` and the adapters specified by
                the adapters argument are not all of the fast type, then a
                `mici.stagers.WindowedWarmUpStager` (with its default arguments)
                will be used, corresponding to using multiple adaptive warm up
                stages with only the fast-type adapters active in some - see
                docstring of `mici.stagers.WarmUpStager` for details.
            memmap_enabled (bool): Whether to memory-map arrays used to store
                chain data to files on disk to avoid excessive system memory
                usage for long chains and/or large chain states. The chain data
                is written to `.npy` files in the directory specified by
                `memmap_path` (or a temporary directory if not provided). These
                files persist after the termination of the function so should
                be manually deleted when no longer required. Default is for
                memory mapping to be disabled.
            memmap_path (str): Path to directory to write memory-mapped chain
                data to. If not provided (the default), a temporary directory
                will be created and the chain data written to files there.
            monitor_stats (Iterable[str]): List of string keys of chain
                statistics to monitor mean of over samples computed so far
                during sampling by printing as postfix to progress bar. Default
                is to print only the mean `accept_stat` statistic.
            display_progress (bool): Whether to display a progress bar to
                track the completed chain sampling iterations. Default value
                is `True`, i.e. to display progress bar.
            progress_bar_class (mici.progressbars.BaseProgressBar): Class or
                factory function for progress bar to use to show chain
                progress if enabled (`display_progress=True`). Defaults to
                `mici.progressbars.ProgressBar`.

        Returns:
            final_states (List[ChainState]): States of chains after final
                iteration. May be used to resume sampling a chain by passing as
                the initial states to a new `sample_chains` call.
            traces (Dict[str, List[array]]): Dictionary of chain trace arrays.
                Values in dictionary are list of arrays of variables outputted
                by trace functions in `trace_funcs` with each array in the list
                corresponding to a single chain and the leading dimension of
                each array corresponding to the iteration (draw) index within
                the main non-adaptive sampling stage. The key for each value is
                the corresponding key in the dictionary returned by the trace
                function which computed the traced value.
            chain_stats (Dict[str, List[array]]): Dictionary of chain
                integration transition statistics. Values in dictionary are
                lists of arrays of chain statistic values with each array in the
                list corresponding to a single chain and the leading dimension
                of each array corresponding to the iteration (draw) index within
                the main non-adaptive sampling stage. The key for each value is
                a string description of the corresponding integration transition
                statistic.
        """
        init_states = [self._preprocess_init_state(i) for i in init_states]
        if 'adapters' not in kwargs:
            kwargs['adapters'] = [DualAveragingStepSizeAdapter()]
        self.__set_sample_chain_kwargs_defaults(kwargs)
        final_states, traces, chain_stats = (
            super().sample_chains_with_adaptive_warm_up(
                n_warm_up_iter, n_main_iter, init_states, **kwargs))
        chain_stats = chain_stats.get('integration_transition', {})
        return final_states, traces, chain_stats


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
    Hamiltonian Monte Carlo) algorithm [1, 2].

    References:

      1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
         Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
      2. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
    """

    def __init__(self, system, integrator, rng, n_step,
                 momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            n_step (int): Number of integrator steps to simulate in each
                integration transition.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.MetropolisStaticIntegrationTransition(
            system, integrator, n_step)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def n_step(self):
        """Number of integrator steps per integrator transition."""
        return self.transitions['integration_transition'].n_step

    @n_step.setter
    def n_step(self, value):
        assert value > 0, 'n_step must be non-negative'
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
    period of (near) periodic systems [1, 2].

    References:

      1. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
         Handbook of Markov Chain Monte Carlo, 2(11), p.2.
      2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
         Physics Letters B, 226(3-4), pp.369-371.
    """

    def __init__(self, system, integrator, rng, n_step_range,
                 momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            n_step_range (Tuple[int, int]): Tuple `(lower, upper)` with two
                positive integer entries `lower` and `upper` (with
                `upper > lower`) specifying respectively the lower and upper
                bounds (inclusive) of integer interval to uniformly draw random
                number integrator steps to simulate in each integration
                transition.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.MetropolisRandomIntegrationTransition(
            system, integrator, n_step_range)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def n_step_range(self):
        """Interval to uniformly draw number of integrator steps from."""
        return self.transitions['integration_transition'].n_step_range

    @n_step_range.setter
    def n_step_range(self, value):
        n_step_lower, n_step_upper = value
        assert n_step_lower > 0 and n_step_lower < n_step_upper, (
            'Range bounds must be non-negative and first entry less than last')
        self.transitions['integration_transition'].n_step_range = value


class DynamicMultinomialHMC(HamiltonianMCMC):
    """Dynamic integration time H-MCMC with multinomial sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps
    equal to the previous tree size [1, 2] until a termination criteria on the
    tree leaves is met. The next chain state is chosen from the candidate
    states using a progressive multinomial sampling scheme [2] based on the
    relative probability densities of the different candidate states, with the
    resampling biased towards states further from the current state.

    When used with the default settings of `riemannian_no_u_turn_criterion`
    termination criterion and extra subtree checks enabled, this sampler is
    equivalent to the default 'NUTS' MCMC algorithm (minus adaptation) used in
    [Stan](https://mc-stan.org/) as of version v2.23.

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
                 do_extra_subtree_checks=True, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            max_tree_depth (int): Maximum depth to expand trajectory binary
                tree to in integrator transition. The maximum number of
                integrator steps corresponds to `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian
                function over a trajectory in integrator transition before
                signalling a divergence.
            termination_criterion (
                    Callable[[System, ChainState, ChainState, array], bool]):
                Function computing criterion to use to determine when to
                terminate trajectory tree expansion. The function should take a
                Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory
                (sub-)tree being checked and an array containing the sum of the
                momentums over the trajectory (sub)-tree. Defaults to
                `mici.transitions.riemannian_no_u_turn_criterion`.
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
                potentially up until the `max_tree_depth` limit is hit. For more
                details see [this Stan Discourse discussion](kutt.it/yAkIES).
                If `do_extra_subtree_checks` is set to `True` additional
                termination criterion checks are performed on overlapping
                subtrees which help to reduce this resonant behaviour at the
                cost of more conservative trajectory termination in some
                correlated models and some overhead from additional checks.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.MultinomialDynamicIntegrationTransition(
            system, integrator, max_tree_depth, max_delta_h,
            termination_criterion, do_extra_subtree_checks)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def max_tree_depth(self):
        """Maximum depth to expand trajectory binary tree to."""
        return self.transitions['integration_transition'].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        assert value > 0, 'max_tree_depth must be non-negative'
        self.transitions['integration_transition'].max_tree_depth = value

    @property
    def max_delta_h(self):
        """Change in Hamiltonian over trajectory to trigger divergence."""
        return self.transitions['integration_transition'].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions['integration_transition'].max_delta_h = value


class DynamicSliceHMC(HamiltonianMCMC):
    """Dynamic integration time H-MCMC with slice sampling of new state.

    In each transition a binary tree of states is recursively computed by
    integrating randomly forward and backward in time by a number of steps equal
    to the previous tree size [1] until a termination criteria on the tree
    leaves is met. The next chain state is chosen from the candidate states
    using a progressive slice sampling scheme [1] based on the relative
    probability densities of the different candidate states, with the sampling
    biased towards states further from the current state.

    When used with the default setting of `euclidean_no_u_turn_criterion`
    termination criterion and extra subtree checks disabled, this sampler is
    equivalent to 'Algorithm 3: Efficient No-U-Turn Sampler' in [1], i.e. the
    'classic NUTS' algorithm.

    References:

      1. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler:
         adaptively setting path lengths in Hamiltonian Monte Carlo.
         Journal of Machine Learning Research, 15(1), pp.1593-1623.
    """

    def __init__(self, system, integrator, rng,
                 max_tree_depth=10, max_delta_h=1000,
                 termination_criterion=trans.euclidean_no_u_turn_criterion,
                 do_extra_subtree_checks=False, momentum_transition=None):
        """
        Args:
            system (mici.systems.System): Hamiltonian system to be simulated.
            rng (numpy.random.Generator): Numpy random number generator.
            integrator (mici.integrators.Integrator): Symplectic integrator to
                use to simulate dynamics in integration transition.
            max_tree_depth (int): Maximum depth to expand trajectory binary
                tree to in integrator transition. The maximum number of
                integrator steps corresponds to `2**max_tree_depth`.
            max_delta_h (float): Maximum change to tolerate in the Hamiltonian
                function over a trajectory in integrator transition before
                signalling a divergence.
            termination_criterion (
                    Callable[[System, ChainState, ChainState, array], bool]):
                Function computing criterion to use to determine when to
                terminate trajectory tree expansion. The function should take a
                Hamiltonian system as its first argument, a pair of states
                corresponding to the two edge nodes in the trajectory
                (sub-)tree being checked and an array containing the sum of the
                momentums over the trajectory (sub)-tree. Defaults to
                `mici.transitions.euclidean_no_u_turn_criterion`.
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
                potentially up until the `max_tree_depth` limit is hit. For more
                details see [this Stan Discourse discussion](kutt.it/yAkIES).
                If `do_extra_subtree_checks` is set to `True` additional
                termination criterion checks are performed on overlapping
                subtrees which help to reduce this resonant behaviour at the
                cost of more conservative trajectory termination in some
                correlated models and some overhead from additional checks.
            momentum_transition (None or mici.transitions.MomentumTransition):
                Markov transition kernel which leaves the conditional
                distribution on the momentum under the canonical distribution
                invariant, updating only the momentum component of the chain
                state. If set to `None` the momentum transition operator
                `mici.transitions.IndependentMomentumTransition` will be used,
                which independently samples the momentum from its conditional
                distribution.
        """
        integration_transition = trans.SliceDynamicIntegrationTransition(
            system, integrator, max_tree_depth, max_delta_h,
            termination_criterion, do_extra_subtree_checks)
        super().__init__(system, rng, integration_transition,
                         momentum_transition)

    @property
    def max_tree_depth(self):
        """Maximum depth to expand trajectory binary tree to."""
        return self.transitions['integration_transition'].max_tree_depth

    @max_tree_depth.setter
    def max_tree_depth(self, value):
        assert value > 0, 'max_tree_depth must be non-negative'
        self.transitions['integration_transition'].max_tree_depth = value

    @property
    def max_delta_h(self):
        """Change in Hamiltonian over trajectory to trigger divergence."""
        return self.transitions['integration_transition'].max_delta_h

    @max_delta_h.setter
    def max_delta_h(self, value):
        self.transitions['integration_transition'].max_delta_h = value
