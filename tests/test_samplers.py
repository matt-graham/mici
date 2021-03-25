import pytest
import numpy as np
import mici


SEED = 3046987125
STATE_DIM = 2
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


@pytest.fixture(params=("dict", "list", "tuple", "nested"))
def memmap_obj_and_filename_obj(request, tmp_path):
    filename = tmp_path / "mm.npy"
    mm = np.lib.format.open_memmap(filename, dtype=np.float64, mode="w+", shape=(1,))
    mm[:] = np.arange(mm.size)
    if request.param == "dict":

        def pytree_func(obj):
            return {"k": obj}

    elif request.param == "list":

        def pytree_func(obj):
            return [obj]

    elif request.param == "tuple":

        def pytree_func(obj):
            return (obj,)

    elif request.param == "nested":

        def pytree_func(obj):
            return {"k": [(obj,)]}

    else:
        raise ValueError(f"Invalid param {request.param}")
    return pytree_func(mm), pytree_func(filename)


def test_memmaps_to_file_paths(memmap_obj_and_filename_obj):
    memmap_obj, filename_obj = memmap_obj_and_filename_obj
    generated_filename_obj = mici.samplers._memmaps_to_file_paths(memmap_obj)
    assert filename_obj == generated_filename_obj


def test_file_paths_to_memmaps(memmap_obj_and_filename_obj):
    memmap_obj, filename_obj = memmap_obj_and_filename_obj
    generated_memmap_obj = mici.samplers._file_paths_to_memmaps(filename_obj)
    assert memmap_obj == generated_memmap_obj


def test_zip_dict():
    assert list(mici.samplers._zip_dict(a=[1, 2, 3], b=[4, 5, 6])) == [
        {"a": 1, "b": 4},
        {"a": 2, "b": 5},
        {"a": 3, "b": 6},
    ]


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
        mici.samplers._get_per_chain_rngs(None, 4)


def test_random_state_raises_deprecation_warning(integrator, system):
    rng = np.random.RandomState(SEED)
    with pytest.deprecated_call():
        mici.samplers.MarkovChainMonteCarloMethod(rng=rng, transitions={})


class MarkovChainMonteCarloMethodTests:
    def test_sampler_attributes(self, sampler, rng):
        assert sampler.rng is rng
        assert isinstance(sampler.transitions, dict)

    @pytest.mark.parametrize("n_warm_up_iter", (0, 2))
    @pytest.mark.parametrize("n_main_iter", (0, 2))
    @pytest.mark.parametrize("trace_warm_up", (True, False))
    @pytest.mark.parametrize("n_process", (1, N_CHAIN))
    def test_sample_chains(
        self,
        sampler,
        n_warm_up_iter,
        n_main_iter,
        init_states,
        trace_funcs,
        trace_warm_up,
        n_process,
        kwargs,
    ):
        final_states, traces, stats = sampler.sample_chains(
            n_warm_up_iter=n_warm_up_iter,
            n_main_iter=n_main_iter,
            init_states=init_states,
            trace_funcs=trace_funcs,
            trace_warm_up=trace_warm_up,
            n_process=n_process,
            **kwargs,
        )
        trace_vars = {}
        if trace_funcs is not None:
            for trace_func in trace_funcs:
                trace_vars.update(trace_func(final_states[0]))
        assert all(isinstance(state, mici.states.ChainState) for state in final_states)
        n_trace_iter = n_main_iter + n_warm_up_iter if trace_warm_up else n_main_iter
        if trace_funcs is None or len(trace_funcs) == 0:
            assert traces is None
        else:
            assert isinstance(traces, dict)
            assert traces.keys() == trace_vars.keys()
            self.check_traces(traces, n_trace_iter, len(init_states), trace_vars)
        assert isinstance(stats, dict)
        self.check_stats_dict(
            stats, n_trace_iter, len(init_states), sampler.transitions
        )

    def check_traces(self, traces, n_iter, n_chain, trace_vars):
        for trace_key, trace_array_list in traces.items():
            assert len(trace_array_list) == n_chain
            for trace_array in trace_array_list:
                assert isinstance(trace_array, np.ndarray)
                assert not np.any(np.isnan(trace_array))
                assert trace_array.shape[0] == n_iter
                assert trace_array.shape[1:] == np.shape(trace_vars[trace_key])

    def check_trans_stats_dict(self, trans_stats, n_iter, statistic_types):
        for stat_key, stat_array_list in trans_stats.items():
            for stat_array in stat_array_list:
                assert stat_array.shape[0] == n_iter
                assert not np.any(np.isnan(stat_array))
                assert stat_array.dtype == statistic_types[stat_key][0]

    def check_stats_dict(self, stats, n_iter, n_chain, transitions):
        for trans_key, trans_stats in stats.items():
            assert isinstance(trans_key, str)
            assert trans_key in transitions
            statistic_types = transitions[trans_key].statistic_types
            if statistic_types is None:
                assert len(trans_stats) == 0
            else:
                assert trans_stats.keys() == statistic_types.keys()
                self.check_trans_stats_dict(trans_stats, n_iter, statistic_types)


class TestMarkovChainMonteCarloMethod(MarkovChainMonteCarloMethodTests):
    @staticmethod
    def pos_mom_trace_func(state):
        return {"pos": state.pos, "mom": state.mom}

    @pytest.fixture(params=("pos_mom_trace_func", "empty", None))
    def trace_funcs(self, request):
        if request.param == "pos_mom_trace_func":
            return [self.pos_mom_trace_func]
        elif request.param == "empty":
            return []
        elif request.param is None:
            return None
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture(params=(None, (("integration", "accept_stat"),)))
    def monitor_stats(self, request):
        return request.param

    @pytest.fixture(params=(None, "empty", "DualAveragingStepSizeAdapter"))
    def adapters(self, request):
        if request.param is None:
            return None
        elif request.param == "empty":
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

    @pytest.fixture(params=((True, None), (True, "tmp"), (False, None)))
    def force_memmap_and_memmap_path(self, request, tmp_path_factory):
        force_memmap, memmap_path = request.param
        if memmap_path == "tmp":
            memmap_path = tmp_path_factory.mktemp("traces-and-stats")
        return force_memmap, memmap_path

    @pytest.fixture(params=(True, False))
    def display_progress(self, request):
        return request.param

    @pytest.fixture
    def kwargs(
        self,
        monitor_stats,
        adapters,
        stager,
        force_memmap_and_memmap_path,
        display_progress,
    ):
        force_memmap, memmap_path = force_memmap_and_memmap_path
        return {
            "monitor_stats": monitor_stats,
            "adapters": adapters,
            "stager": stager,
            "force_memmap": force_memmap,
            "memmap_path": memmap_path,
            "display_progress": display_progress,
        }

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


class HamiltonianMCMCTests(MarkovChainMonteCarloMethodTests):
    @pytest.fixture(params=(None, ("accept_stat", "n_step")))
    def monitor_stats(self, request):
        return request.param

    @pytest.fixture(params=(None, "DualAveragingStepSizeAdapter"))
    def adapters(self, request):
        if request.param is None:
            return None
        elif request.param == "DualAveragingStepSizeAdapter":
            return [mici.adapters.DualAveragingStepSizeAdapter()]
        else:
            raise ValueError(f"Invalid param {request.param}")

    @pytest.fixture
    def kwargs(self, monitor_stats, adapters):
        kwargs = {}
        if monitor_stats is not None:
            kwargs["monitor_stats"] = monitor_stats
        if adapters is not None:
            kwargs["adapters"] = adapters
        return kwargs

    @pytest.fixture
    def trace_funcs(self, sampler):
        return [sampler._default_trace_func]

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

    def check_stats_dict(self, stats, n_iter, n_chain, transitions):
        statistic_types = transitions["integration_transition"].statistic_types
        self.check_trans_stats_dict(stats, n_iter, statistic_types)


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
