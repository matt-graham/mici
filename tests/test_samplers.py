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
    return np.sum(pos ** 2) / 2


def grad_neg_log_dens(pos):
    return pos


@pytest.fixture
def system():
    return mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
    )


@pytest.fixture
def integrator(system):
    return mici.integrators.LeapfrogIntegrator(system, step_size=0.5)


def test_memmaps_to_file_paths(tmp_path):
    filename = tmp_path / "mm.npy"
    key = "mm"
    mm = np.lib.format.open_memmap(filename, dtype=np.float64, mode="w+", shape=(1, 2))
    mm[:] = np.arange(mm.size)
    obj = {key: [(mm,)]}
    filename_obj = mici.samplers._memmaps_to_file_paths(obj)
    assert filename_obj == {key: [(filename,)]}


def test_get_obj_size():
    obj = {"a": np.empty(1000, dtype=np.complex128), "b": np.empty(100, dtype=np.int8)}
    byte_size = mici.samplers._get_obj_byte_size(obj)
    array_byte_size = obj["a"].nbytes + obj["b"].nbytes
    assert byte_size > array_byte_size
    assert byte_size > (
        mici.samplers._get_obj_byte_size(obj["a"])
        + mici.samplers._get_obj_byte_size(obj["b"])
    )
    assert byte_size < mici.samplers._get_obj_byte_size((obj,))


def test_check_chain_data_size_raises():
    class MockLarge:
        def __sizeof__(self):
            return 2 ** 32

    traces = {"pos": [MockLarge() for i in range(32)]}
    stats = {"integration": {"accept_prob": [MockLarge() for i in range(32)]}}
    with pytest.raises(RuntimeError, match="Total number of bytes allocated"):
        mici.samplers._check_chain_data_size(traces, stats)


@pytest.fixture(params=("Generator(PCG64)", "Generator(SFC64)", "RandomState"))
def base_rng(request):
    if request.param == "Generator(PCG64)":
        return np.random.Generator(np.random.PCG64(SEED))
    elif request.param == "Generator(SFC64)":
        return np.random.Generator(np.random.SFC64(SEED))
    else:
        return np.random.RandomState(SEED)


def test_get_per_chain_rngs(base_rng):
    rngs = mici.samplers._get_per_chain_rngs(base_rng, 4)
    for i, rng in enumerate(rngs):
        assert isinstance(rng, np.random.Generator)
        if i != 0:
            assert rng is not rngs[i - 1]


def test_get_per_chain_rngs_raises():
    with pytest.raises(ValueError, match="Unsupported random number generator"):
        rngs = mici.samplers._get_per_chain_rngs(None, 4)


@pytest.mark.parametrize("offset", (-1, 0, 1))
@pytest.mark.parametrize("read_only", (True, False))
def test_truncate_chain_data(offset, read_only):
    n_iter = 5
    pos_trace = np.empty((n_iter, 2))
    accept_prob_array = np.empty(n_iter)
    if read_only:
        pos_trace.flags.writeable = False
    traces = {"pos": pos_trace}
    stats = {"i": {"accept_prob": accept_prob_array}}
    mici.samplers._truncate_chain_data(n_iter + offset, traces, stats)
    assert traces["pos"].shape[0] == min(n_iter, n_iter + offset)
    assert stats["i"]["accept_prob"].shape[0] == min(n_iter, n_iter + offset)


class MarkovChainMonteCarloMethodTests:
    def test_sampler_attributes(self, sampler, rng):
        assert sampler.rng is rng
        assert isinstance(sampler.transitions, dict)

    def test_sample_chain(self, sampler, init_state, trace_funcs, adapters, kwargs):
        final_state, traces, stats = sampler.sample_chain(
            init_state=init_state,
            n_iter=N_ITER,
            trace_funcs=trace_funcs,
            adapters=adapters,
            **kwargs,
        )
        trace_vars = {}
        for trace_func in trace_funcs:
            trace_vars.update(trace_func(final_state))
        assert isinstance(final_state, mici.states.ChainState)
        assert isinstance(traces, dict)
        assert isinstance(stats, dict)
        assert traces.keys() == trace_vars.keys()
        self.check_traces_single_chain(traces, N_ITER, trace_vars)
        self.check_stats_dict(stats, N_ITER, None, sampler.transitions)

    def test_sample_chains(
        self, sampler, init_states, trace_funcs, adapters, n_process, kwargs
    ):
        self.check_sample_chains_method(
            sampler.sample_chains,
            sampler.transitions,
            (N_ITER,),
            init_states=init_states,
            trace_funcs=trace_funcs,
            n_process=n_process,
            adapters=adapters,
            **kwargs,
        )

    def test_sample_chains_with_adaptive_warm_up(
        self, sampler, init_states, trace_funcs, adapters, n_process, stager, kwargs
    ):
        self.check_sample_chains_method(
            sampler.sample_chains_with_adaptive_warm_up,
            sampler.transitions,
            (N_WARM_UP_ITER, N_ITER,),
            init_states=init_states,
            trace_funcs=trace_funcs,
            n_process=n_process,
            adapters=adapters,
            stager=stager,
            **kwargs,
        )

    def check_sample_chains_method(
        self,
        sample_chains_method,
        transitions,
        n_iter,
        init_states,
        trace_funcs,
        **kwargs,
    ):

        final_states, traces, stats = sample_chains_method(
            *n_iter, init_states=init_states, trace_funcs=trace_funcs, **kwargs
        )
        trace_vars = {}
        for trace_func in trace_funcs:
            trace_vars.update(trace_func(final_states[0]))
        assert all(isinstance(state, mici.states.ChainState) for state in final_states)
        assert isinstance(traces, dict)
        assert isinstance(stats, dict)
        assert traces.keys() == trace_vars.keys()
        self.check_traces_multiple_chain(
            traces, n_iter[-1], len(init_states), trace_vars
        )
        self.check_stats_dict(stats, n_iter[-1], len(init_states), transitions)

    def check_trace_array(self, trace_array, n_iter, var_shape):
        assert isinstance(trace_array, np.ndarray)
        assert trace_array.shape[0] == n_iter
        assert trace_array.shape[1:] == var_shape

    def check_traces_single_chain(self, traces, n_iter, trace_vars):
        for trace_key, trace_array in traces.items():
            self.check_trace_array(trace_array, n_iter, np.shape(trace_vars[trace_key]))

    def check_traces_multiple_chain(self, traces, n_iter, n_chain, trace_vars):
        for trace_key, trace_array_list in traces.items():
            assert len(trace_array_list) == n_chain
            for trace_array in trace_array_list:
                self.check_trace_array(
                    trace_array, n_iter, np.shape(trace_vars[trace_key])
                )

    def check_stat_array(self, stat_array, n_iter, dtype):
        assert stat_array.shape[0] == n_iter
        assert stat_array.dtype == dtype

    def check_trans_stats_dict_single_chain(self, trans_stats, n_iter, statistic_types):
        for stat_key, stat_array in trans_stats.items():
            self.check_stat_array(stat_array, n_iter, statistic_types[stat_key][0])

    def check_trans_stats_dict_multiple_chain(
        self, trans_stats, n_iter, statistic_types
    ):
        for stat_key, stat_array_list in trans_stats.items():
            for stat_array in stat_array_list:
                self.check_stat_array(stat_array, n_iter, statistic_types[stat_key][0])

    def check_stats_dict(self, stats, n_iter, n_chain, transitions):
        for trans_key, trans_stats in stats.items():
            assert isinstance(trans_key, str)
            assert trans_key in transitions
            statistic_types = transitions[trans_key].statistic_types
            if statistic_types is None:
                assert len(trans_stats) == 0
            else:
                assert trans_stats.keys() == statistic_types.keys()
                if n_chain is not None:
                    self.check_trans_stats_dict_multiple_chain(
                        trans_stats, n_iter, statistic_types
                    )
                else:
                    self.check_trans_stats_dict_single_chain(
                        trans_stats, n_iter, statistic_types
                    )


class TestMarkovChainMonteCarloMethod(MarkovChainMonteCarloMethodTests):
    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.MarkovChainMonteCarloMethod(
            rng=rng,
            transitions={
                "momentum": mici.transitions.CorrelatedMomentumTransition(system, 0.5),
                "integration": mici.transitions.MetropolisStaticIntegrationTransition(
                    system, integrator, 1
                ),
            },
        )

    @pytest.fixture(params=("ChainState", "dict"))
    def init_state(self, rng, request):
        pos, mom = rng.standard_normal((2, STATE_DIM))
        if request.param == "dict":
            return {"pos": pos, "mom": mom, "dir": 1}
        elif request.param == "ChainState":
            return mici.states.ChainState(pos=pos, mom=mom, dir=1)
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture(params=("ChainState", "dict"))
    def init_states(self, rng, request):
        init_states = []
        for c in range(N_CHAIN):
            pos, mom = rng.standard_normal((2, STATE_DIM))
            if request.param == "dict":
                state = {"pos": pos, "mom": mom, "dir": 1}
            elif request.param == "ChainState":
                state = mici.states.ChainState(pos=pos, mom=mom, dir=1)
            else:
                raise ValueError(f"Invalid param {request.param}")
            init_states.append(state)
        return init_states

    @pytest.fixture(params=((None, None), (True, None), (True, "tmp"), (False, None)))
    def memmap_enabled_and_path(self, request, tmp_path_factory):
        memmap_enabled, memmap_path = request.param
        if memmap_path == "tmp":
            memmap_path = tmp_path_factory.mktemp("traces-and-stats")
        return memmap_enabled, memmap_path

    @pytest.fixture(params=(None, (("integration", "accept_stat"),)))
    def monitor_stats(self, request):
        return request.param

    @pytest.fixture(params=(None, True, False))
    def display_progress(self, request):
        return request.param

    @pytest.fixture(params=(None, "DualAveragingStepSizeAdapter"))
    def adapters(self, request):
        if request.param is None:
            return {}
        elif request.param == "DualAveragingStepSizeAdapter":
            return {"integration": [mici.adapters.DualAveragingStepSizeAdapter()]}
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture(params=(None, "WarmUpStager"))
    def stager(self, request):
        if request.param is None:
            return None
        elif request.param == "WarmUpStager":
            return mici.stagers.WarmUpStager()
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture
    def kwargs(self, memmap_enabled_and_path, monitor_stats, display_progress):
        kwargs = {}
        memmap_enabled, memmap_path = memmap_enabled_and_path
        if memmap_enabled is not None:
            kwargs["memmap_enabled"] = memmap_enabled
        if memmap_enabled is not None:
            kwargs["memmap_path"] = memmap_path
        if monitor_stats is not None:
            kwargs["monitor_stats"] = monitor_stats
        if display_progress is not None:
            kwargs["display_progress"] = display_progress
        return kwargs

    @pytest.fixture(params=(1, N_CHAIN))
    def n_process(self, request):
        return request.param

    @staticmethod
    def pos_mom_trace_func(state):
        return {"pos": state.pos, "mom": state.mom}

    @pytest.fixture
    def trace_funcs(self):
        return [self.pos_mom_trace_func]


class HamiltonianMCMCTests(MarkovChainMonteCarloMethodTests):
    @pytest.fixture(params=("ChainState_with_mom", "ChainState_no_mom", "array"))
    def init_state(self, rng, request):
        pos = rng.standard_normal(STATE_DIM)
        if request.param == "array":
            return pos
        elif request.param == "ChainState_with_mom":
            return mici.states.ChainState(pos=pos, mom=None, dir=1)
        elif request.param == "ChainState_no_mom":
            mom = rng.standard_normal(STATE_DIM)
            return mici.states.ChainState(pos=pos, mom=mom, dir=1)
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture(params=("ChainState_with_mom", "ChainState_no_mom", "array"))
    def init_states(self, rng, request):
        init_states = []
        for c in range(N_CHAIN):
            pos = rng.standard_normal(STATE_DIM)
            if request.param == "array":
                state = pos
            elif request.param == "ChainState_no_mom":
                state = mici.states.ChainState(pos=pos, mom=None, dir=1)
            elif request.param == "ChainState_with_mom":
                mom = rng.standard_normal(STATE_DIM)
                state = mici.states.ChainState(pos=pos, mom=mom, dir=1)
            else:
                raise ValueError(f"Invalid param {request.param}")
            init_states.append(state)
        return init_states

    @pytest.fixture(params=(None, [], ["accept_stat", "n_step"]))
    def monitor_stats(self, request):
        return request.param

    @pytest.fixture(params=(None, "DualAveragingStepSizeAdapter"))
    def adapters(self, request):
        if request.param is None:
            return []
        elif request.param == "DualAveragingStepSizeAdapter":
            return [mici.adapters.DualAveragingStepSizeAdapter()]
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture(params=(None, "WarmUpStager"))
    def stager(self, request):
        if request.param is None:
            return None
        elif request.param == "WarmUpStager":
            return mici.stagers.WarmUpStager()
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture
    def kwargs(self, monitor_stats):
        kwargs = {}
        if monitor_stats is not None:
            kwargs["monitor_stats"] = monitor_stats
        return kwargs

    @pytest.fixture(params=(1, N_CHAIN))
    def n_process(self, request):
        return request.param

    @pytest.fixture
    def trace_funcs(self, sampler):
        return [sampler._default_trace_func]

    def check_stats_dict(self, stats, n_iter, n_chain, transitions):
        statistic_types = transitions["integration_transition"].statistic_types
        if n_chain is None:
            self.check_trans_stats_dict_single_chain(stats, n_iter, statistic_types)
        else:
            self.check_trans_stats_dict_multiple_chain(stats, n_iter, statistic_types)


class TestStaticMetropolisHMC(HamiltonianMCMCTests):

    n_step = 2

    def test_max_tree_depth(self, sampler):
        assert sampler.n_step == self.n_step
        new_n_step = self.n_step + 1
        sampler.n_step = new_n_step
        assert sampler.n_step == new_n_step
        assert sampler.transitions["integration_transition"].n_step == (new_n_step)

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.StaticMetropolisHMC(
            system=system, integrator=integrator, rng=rng, n_step=self.n_step
        )


class TestRandomMetropolisHMC(HamiltonianMCMCTests):

    n_step_range = (1, 3)

    def test_max_tree_depth(self, sampler):
        assert sampler.n_step_range == self.n_step_range
        new_n_step_range = (self.n_step_range[0] + 1, self.n_step_range[1] + 1)
        sampler.n_step_range = new_n_step_range
        assert sampler.n_step_range == new_n_step_range
        assert sampler.transitions["integration_transition"].n_step_range == (
            new_n_step_range
        )

    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.RandomMetropolisHMC(
            system=system,
            integrator=integrator,
            rng=rng,
            n_step_range=self.n_step_range,
        )


class DynamicHMCTests(HamiltonianMCMCTests):

    max_tree_depth = 2
    max_delta_h = 1000

    def test_max_tree_depth(self, sampler):
        assert sampler.max_tree_depth == self.max_tree_depth
        new_max_tree_depth = self.max_tree_depth + 1
        sampler.max_tree_depth = new_max_tree_depth
        assert sampler.max_tree_depth == new_max_tree_depth
        assert sampler.transitions["integration_transition"].max_tree_depth == (
            new_max_tree_depth
        )

    def test_max_delta_h(self, sampler):
        assert sampler.max_delta_h == self.max_delta_h
        new_max_delta_h = self.max_delta_h + 1
        sampler.max_delta_h = new_max_delta_h
        assert sampler.max_delta_h == new_max_delta_h
        assert sampler.transitions["integration_transition"].max_delta_h == (
            new_max_delta_h
        )


class TestDynamicMultinomialHMC(DynamicHMCTests):
    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.DynamicMultinomialHMC(
            system=system,
            integrator=integrator,
            rng=rng,
            max_tree_depth=self.max_tree_depth,
            max_delta_h=self.max_delta_h,
        )


class TestDynamicDynamicSliceHMC(DynamicHMCTests):
    @pytest.fixture
    def sampler(self, integrator, system, rng):
        return mici.samplers.DynamicSliceHMC(
            system=system,
            integrator=integrator,
            rng=rng,
            max_tree_depth=self.max_tree_depth,
            max_delta_h=self.max_delta_h,
        )

