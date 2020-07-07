"""Classes for controlling sampling of Markov chains split into stages."""

import abc
from collections import OrderedDict, namedtuple


ChainStage = namedtuple('ChainStage', ['n_iter', 'adapters', 'trace_funcs'])


class Stager(abc.ABC):
    """Abstract chain iteration stager."""

    @abc.abstractmethod
    def stages(self, n_warm_up_iter, n_main_iter, adapters, trace_funcs):
        """Create dictionary specifying labels and parameters of sampling stages.

        Constructs an ordered dictionary with entries corresponding to the
        sequence of sampling stages when running chains with one or more initial
        adaptation stages. The keys of each entry are string labels for the
        sampling stage and the values a 3-tuple `(n_iter, adapters,
        trace_funcs)` with `n_iter` the number of chain iterations in the stage,
        `adapters` a dictionary of transition adapters to pass to the
        `_sample_chain` call (or `None` if no adaptation to be used) and
        `trace_func` a list of trace functions to pass to the `_sample_chain`
        call (list may be empty if no values to be traced).

        Args:
            n_warm_up_iter (int): Number of adaptive warm up iterations per
                chain. Depending on the adapters specified by the `adapters`
                argument the warm up iterations may be split between one or more
                adaptive stages.
            n_main_iter (int): Number of iterations (samples to draw) per chain
                during main (non-adaptive) sampling stage.
            trace_funcs (Iterable[Callable[[ChainState], Dict[str, array]]]):
                List of functions which compute the variables to be recorded at
                each chain iteration during the final (non-adaptive) sampling
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
                parameters of the transitions during the adaptive stages of the
                chains. Note that the adapter updates are applied in the order
                the adapters appear in the iterables and so if multiple adapters
                change the same parameter(s) the order will matter.

        Returns:
            OrderedDict[str, ChainStage]: Ordered dictionary specifying sampling
                stage parameters.
        """


class WarmUpStager(Stager):
    """Chain iteration stager with a single adaptive warm up stage.

    Sampling is split in to two stages:

      1. An adaptive warm up stage will all adapters active.
      2. A main sampling stage with no adapters active.

    Only in the main sampling stage are traces of the chain state recorded by
    storing the outputs of functions of the sampled chain state after each
    iteration.
    """

    def stages(self, n_warm_up_iter, n_main_iter, adapters, trace_funcs):
        # adaptive warm up stage
        sampling_stages = OrderedDict(
            {'Adaptive warm up': ChainStage(n_warm_up_iter, adapters, [])})
        # main non-adaptive stage
        sampling_stages['Main non-adaptive'] = ChainStage(
            n_main_iter, None, trace_funcs)
        return sampling_stages


class WindowedWarmUpStager(Stager):
    """Chain iteration stager with a hierarchy of adaptive warm up stages.

    Following the approach of [Stan](https://mc-stan.org) the adaptive
    stages are split in to two types - 'fast' adaptation stages which adjust
    only transition parameters which can be adapted quickly using local
    information and 'slow' adaptation stages which *addtionally* adjust
    transition parameters which require more global information. The
    adapters to be used in both the fast and slow adaptation stages will be
    referred to as the *fast adapters* and the adapters to use in only the
    slow adaptation stages the *slow adapters*. Each adapter self identifies
    if it is a fast adapter by whether the `is_fast` attribute is set to `True`.

    The adaptive warm up iterations are split into three stages:

      1. An initial fast adaptive stage with only fast adapters active.
      2. A slow adaptive stage with both slow and fast adapters active.
      3. A final adaptive stage with only fast adapters active.

    The slow sampling stage (2) is further split in to a sequence of
    growing, memoryless windows with the adapter stages reset at the
    beginning of each window, and the number of iterations in each window
    increasing (by default doubling). The split of the iterations in each of
    these stages can be controlled using the keyword arguments
    `n_init_fast_stage_iter`, `n_init_slow_window_iter`,
    `n_final_fast_stage_iter` and `slow_window_multiplier` (see descriptions
    below).

    After the initial adaptive warm up stages a subsequent main sampling stage
    with no further adaptation is performed. Only in this main sampling stage
    are traces of the chain state recorded by storing the outputs of functions
    of the sampled chain state after each iteration.
    """

    def __init__(
            self, n_init_slow_window_iter=25, n_init_fast_stage_iter=75,
            n_final_fast_stage_iter=50, slow_window_multiplier=2):
        """
        Args:
            n_init_slow_window_iter (int): Number of iterations in the initial
                (smallest) window in the slow adaptation stage. Defaults to 25.
                If the sum of `n_init_slow_window_iter`,
                `n_init_fast_stage_iter` and `n_final_fast_stage_iter` is more
                than `n_warm_up_iter` then `n_init_slow_window_iter` is set to
                approximately 75% of `n_warm_up_iter` (with a single window
                being used in the slow adaptation stage in this case).
            n_init_fast_stage_iter (int): Number of iterations in the initial
                fast adaptation stage. Defaults to 75. If the sum of
                `n_init_slow_window_iter`, n_init_fast_stage_iter` and
                `n_final_fast_stage_iter` is more than `n_warm_up_iter` then
                `n_init_fast_stage_iter` is set to approximately 15% of
                `n_warm_up_iter`.
            n_final_fast_stage_iter (int): Number of iterations in the final
                fast adaptation stage. Defaults to 50. If the sum of
                `n_init_slow_window_iter`, `n_init_fast_stage_iter` and
                `n_final_fast_stage_iter` is more than `n_warm_up_iter` then
                `n_init_fast_stage_iter` is set to approximately 10% of
                `n_warm_up_iter`.
            slow_window_multiplier (float): Multiplier by which to increase the
                number of iterations of each subsequent slow adaptation window
                by. Defaults to 2 such that each window doubles in size.
        """
        self.n_init_slow_window_iter = n_init_slow_window_iter
        self.n_init_fast_stage_iter = n_init_fast_stage_iter
        self.n_final_fast_stage_iter = n_final_fast_stage_iter
        self.slow_window_multiplier = slow_window_multiplier

    def stages(self, n_warm_up_iter, n_main_iter, adapters, trace_funcs):
        fast_adapters = {
            trans_key: [adapter for adapter in adapter_list if adapter.is_fast]
            for trans_key, adapter_list in adapters.items()}
        if (self.n_init_fast_stage_iter + self.n_init_slow_window_iter +
                self.n_final_fast_stage_iter) > n_warm_up_iter:
            n_init_fast_stage_iter = int(0.15 * n_warm_up_iter)
            n_final_fast_stage_iter = int(0.1 * n_warm_up_iter)
            n_init_slow_window_iter = (
                n_warm_up_iter - n_init_fast_stage_iter -
                n_final_fast_stage_iter)
        else:
            n_init_slow_window_iter = self.n_init_slow_window_iter
            n_init_fast_stage_iter = self.n_init_fast_stage_iter
            n_final_fast_stage_iter = self.n_final_fast_stage_iter
        # initial fast adaptation stage
        sampling_stages = OrderedDict(
            {'Initial fast adaptive':
                ChainStage(n_init_fast_stage_iter, fast_adapters, [])})
        # growing size slow adaptation windows
        n_window_iter = n_init_slow_window_iter
        slow_windows = []
        counter = 0
        n_slow_stage_iter = (
            n_warm_up_iter - n_init_fast_stage_iter - n_final_fast_stage_iter)
        while counter < n_slow_stage_iter:
            # check if iteration counter at end of next loop iteration will be
            # greater than total number of warm up iterations and if so set
            # number of iterations in current window to be equal to all
            # remaining warm up iterations
            counter_next = (
                counter + int((1 + self.slow_window_multiplier) * n_window_iter)
            )
            if counter_next > n_slow_stage_iter:
                n_window_iter = n_slow_stage_iter - counter
            slow_windows.append(n_window_iter)
            counter += n_window_iter
            n_window_iter = int(self.slow_window_multiplier * n_window_iter)
        for i, n_iter in enumerate(slow_windows):
            sampling_stages[
                f'Slow adaptive ({i + 1}/{len(slow_windows)})'] = ChainStage(
                    n_iter, adapters, [])
        # final fast adaptation stage
        sampling_stages['Final fast adaptive'] = ChainStage(
            n_final_fast_stage_iter, fast_adapters, [])
        # main non-adaptive stage
        sampling_stages['Main non-adaptive'] = ChainStage(
            n_main_iter, None, trace_funcs)
        return sampling_stages
