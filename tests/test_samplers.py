import pytest
import numpy as np
import mici


SEED = 3046987125
STATE_DIM = 2
N_ITER = 2
N_WARM_UP_ITER = 2
N_CHAIN = 2


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def neg_log_dens(pos):
    return np.sum(pos**2) / 2


def grad_neg_log_dens(pos):
    return pos


@pytest.fixture
def system():
    return mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens)


@pytest.fixture
def integrator(system):
    return mici.integrators.LeapfrogIntegrator(system, step_size=0.5)


class MarkovChainMonteCarloMethodTests:

    def test_sampler_attributes(self, sampler, rng):
        assert sampler.rng is rng
        assert isinstance(sampler.transitions, dict)

    def test_sample_chain(
            self, sampler, init_state, trace_funcs, adapters, kwargs):
        final_state, traces, stats = sampler.sample_chain(
            init_state=init_state, n_iter=N_ITER, trace_funcs=trace_funcs,
            adapters=adapters, **kwargs)
        trace_vars = {}
        for trace_func in trace_funcs:
            trace_vars.update(trace_func(final_state))
        assert isinstance(final_state, mici.states.ChainState)
        assert isinstance(traces, dict)
        assert isinstance(stats, dict)
        assert traces.keys() == trace_vars.keys()
        self.check_traces_single_chain(traces, N_ITER, trace_vars)
        self.check_stats_dict_single_chain(stats, N_ITER, sampler.transitions)

    # def test_sample_chains(
    #         self, sampler, init_states, trace_funcs, adapters, n_process,
    #         kwargs):
    #     final_states, traces, stats = sampler.sample_chains(
    #         init_states=init_states, n_iter=N_ITER, trace_funcs=trace_funcs,
    #         n_process=n_process, adapters=adapters, **kwargs)
    #     trace_vars = {}
    #     for trace_func in trace_funcs:
    #         trace_vars.update(trace_func(final_states[0]))
    #     assert all(
    #         isinstance(state, mici.states.ChainState) for state in final_states)
    #     assert isinstance(traces, dict)
    #     assert isinstance(stats, dict)
    #     assert traces.keys() == trace_vars.keys()
    #     self.check_traces_multiple_chain(
    #         traces, N_ITER, len(init_states), trace_vars)
    #     self.check_stats_dict_multiple_chain(
    #         stats, N_ITER, len(init_states), sampler.transitions)

    def test_sample_chains(
            self, sampler, init_states, trace_funcs, adapters, n_process,
            kwargs):
        self.check_sample_chains_method(
            sampler.sample_chains, sampler.transitions, (N_ITER,),
            init_states=init_states, trace_funcs=trace_funcs,
            n_process=n_process, adapters=adapters, **kwargs)

    def test_sample_chains_with_adaptive_warm_up(
            self, sampler, init_states, trace_funcs, adapters, n_process,
            stager, kwargs):
        self.check_sample_chains_method(
            sampler.sample_chains_with_adaptive_warm_up, sampler.transitions,
            (N_WARM_UP_ITER, N_ITER,), init_states=init_states,
            trace_funcs=trace_funcs, n_process=n_process, adapters=adapters,
            stager=stager, **kwargs)

    def check_sample_chains_method(
            self, sample_chains_method, transitions, n_iter, init_states,
            trace_funcs, **kwargs):
        final_states, traces, stats = sample_chains_method(
            *n_iter, init_states=init_states, trace_funcs=trace_funcs, **kwargs)
        trace_vars = {}
        for trace_func in trace_funcs:
            trace_vars.update(trace_func(final_states[0]))
        assert all(
            isinstance(state, mici.states.ChainState) for state in final_states)
        assert isinstance(traces, dict)
        assert isinstance(stats, dict)
        assert traces.keys() == trace_vars.keys()
        self.check_traces_multiple_chain(
            traces, n_iter[-1], len(init_states), trace_vars)
        self.check_stats_dict_multiple_chain(
            stats,  n_iter[-1], len(init_states), transitions)

    def check_trace_array(self, trace_array, n_iter, var_shape):
        assert isinstance(trace_array, np.ndarray)
        assert trace_array.shape[0] == n_iter
        assert trace_array.shape[1:] == var_shape

    def check_traces_single_chain(self, traces, n_iter, trace_vars):
        for trace_key, trace_array in traces.items():
            self.check_trace_array(
                trace_array, n_iter, np.shape(trace_vars[trace_key]))

    def check_traces_multiple_chain(self, traces, n_iter, n_chain, trace_vars):
        for trace_key, trace_array_list in traces.items():
            assert len(trace_array_list) == n_chain
            for trace_array in trace_array_list:
                self.check_trace_array(
                    trace_array, n_iter, np.shape(trace_vars[trace_key]))

    def check_stat_array(self, stat_array, n_iter, dtype):
        assert stat_array.shape[0] == n_iter
        assert stat_array.dtype == dtype

    def check_trans_stats_dict_single_chain(
            self, trans_stats, n_iter, statistic_types):
        for stat_key, stat_array in trans_stats.items():
            self.check_stat_array(
                stat_array, n_iter, statistic_types[stat_key][0])

    def check_stats_dict_single_chain(self, stats, n_iter, transitions):
        for trans_key, trans_stats in stats.items():
            assert isinstance(trans_key, str)
            assert trans_key in transitions
            statistic_types = transitions[trans_key].statistic_types
            self.check_trans_stats_dict_single_chain(
                trans_stats, n_iter, statistic_types)

    def check_trans_stats_dict_multiple_chain(
            self, trans_stats, n_iter, statistic_types):
        for stat_key, stat_array_list in trans_stats.items():
            for stat_array in stat_array_list:
                self.check_stat_array(
                    stat_array, n_iter, statistic_types[stat_key][0])

    def check_stats_dict_multiple_chain(
            self, stats, n_iter, n_chain, transitions):
        for trans_key, trans_stats in stats.items():
            assert isinstance(trans_key, str)
            assert trans_key in transitions
            statistic_types = transitions[trans_key].statistic_types
            assert trans_stats.keys() == statistic_types.keys()
            self.check_trans_stats_dict_multiple_chain(
                trans_stats, n_iter, statistic_types)


class HamiltonianMCMCTests(MarkovChainMonteCarloMethodTests):

    @pytest.fixture(
        params=('ChainState_with_mom', 'ChainState_no_mom', 'array'))
    def init_state(self, rng, request):
        pos = rng.standard_normal(STATE_DIM)
        if request.param == 'array':
            return pos
        elif request.param == 'ChainState_with_mom':
            return mici.states.ChainState(pos=pos, mom=None, dir=1)
        elif request.param == 'ChainState_no_mom':
            mom = rng.standard_normal(STATE_DIM)
            return mici.states.ChainState(pos=pos, mom=mom, dir=1)
        else:
            raise ValueError(f'Invalid param {request.param}')

    @pytest.fixture(
        params=('ChainState_with_mom', 'ChainState_no_mom', 'array'))
    def init_states(self, rng, request):
        init_states = []
        for c in range(N_CHAIN):
            pos = rng.standard_normal(STATE_DIM)
            if request.param == 'array':
                state = pos
            elif request.param == 'ChainState_no_mom':
                state = mici.states.ChainState(pos=pos, mom=None, dir=1)
            elif request.param == 'ChainState_with_mom':
                mom = rng.standard_normal(STATE_DIM)
                state = mici.states.ChainState(pos=pos, mom=mom, dir=1)
            else:
                raise ValueError(f'Invalid param {request.param}')
            init_states.append(state)
        return init_states

    @pytest.fixture(
        params=((None, None), (True, None), (True, 'tmp'), (False, None)))
    def memmap_enabled_and_path(self, request, tmp_path_factory):
        memmap_enabled, memmap_path = request.param
        if memmap_path == 'tmp':
            memmap_path = tmp_path_factory.mktemp('traces-and-stats')
        return memmap_enabled, memmap_path

    @pytest.fixture(
        params=(None, [], ['accept_stat'], ['accept_stat', 'n_step']))
    def monitor_stats(self, request):
        return request.param

    @pytest.fixture(params=(None, True, False))
    def display_progress(self, request):
        return request.param

    @pytest.fixture(params=(None, 'DualAveragingStepSizeAdapter'))
    def adapters(self, request):
        if request.param is None:
            return []
        elif request.param == 'DualAveragingStepSizeAdapter':
            return [mici.adapters.DualAveragingStepSizeAdapter()]
        else:
            raise ValueError(f'Invalid param {request.param}')

    @pytest.fixture(params=(None, 'WarmUpStager'))
    def stager(self, request):
        if request.param is None:
            return None
        elif request.param == 'WarmUpStager':
            return mici.stagers.WarmUpStager()
        else:
            raise ValueError(f'Invalid param {request.param}')

    @pytest.fixture
    def kwargs(self, memmap_enabled_and_path, monitor_stats, display_progress):
        kwargs = {}
        memmap_enabled, memmap_path = memmap_enabled_and_path
        if memmap_enabled is not None:
            kwargs['memmap_enabled'] = memmap_enabled
        if memmap_enabled is not None:
            kwargs['memmap_path'] = memmap_path
        if monitor_stats is not None:
            kwargs['monitor_stats'] = monitor_stats
        if display_progress is not None:
            kwargs['display_progress'] = display_progress
        return kwargs

    @pytest.fixture(params=(1, N_CHAIN))
    def n_process(self, request):
        return request.param

    @pytest.fixture
    def trace_funcs(self, sampler):
        return [sampler._default_trace_func]

    def check_stats_dict_single_chain(self, stats, n_iter, transitions):
        statistic_types = transitions['integration_transition'].statistic_types
        self.check_trans_stats_dict_single_chain(stats, n_iter, statistic_types)

    def check_stats_dict_multiple_chain(
            self, stats, n_iter, n_chain, transitions):
        statistic_types = transitions['integration_transition'].statistic_types
        self.check_trans_stats_dict_multiple_chain(
            stats, n_iter, statistic_types)


class TestStaticMetropolisHMC(HamiltonianMCMCTests):

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.StaticMetropolisHMC(
            system=system, integrator=integrator, rng=rng, n_step=2)


class TestRandomMetropolisHMC(HamiltonianMCMCTests):

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.RandomMetropolisHMC(
            system=system, integrator=integrator, rng=rng, n_step_range=(1, 3))


class TestDynamicMultinomialHMC(HamiltonianMCMCTests):

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.DynamicMultinomialHMC(
            system=system, integrator=integrator, rng=rng, max_tree_depth=2)


class TestDynamicDynamicSliceHMC(HamiltonianMCMCTests):

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.DynamicSliceHMC(
            system=system, integrator=integrator, rng=rng, max_tree_depth=2)
