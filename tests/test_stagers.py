import pytest
import mici

N_MAIN_ITER = 1


class StagerTests:
    def test_stages(self, stager, n_warm_up_iter, adapters):
        stages = stager.stages(
            n_warm_up_iter, N_MAIN_ITER, adapters, [lambda state: None]
        )
        assert isinstance(stages, dict)
        n_total_iter = 0
        for stage_key, stage in stages.items():
            assert isinstance(stage_key, str)
            assert isinstance(stage, mici.stagers.ChainStage)
            assert stage.n_iter >= 0
            assert stage.adapters is None or isinstance(stage.adapters, dict)
            assert isinstance(stage.trace_funcs, list)
            n_total_iter += stage.n_iter
        assert n_total_iter == n_warm_up_iter + N_MAIN_ITER


class TestWarmUpStager(StagerTests):
    @pytest.fixture
    def adapters(self):
        return {
            "integration_transition": [mici.adapters.DualAveragingStepSizeAdapter()]
        }

    @pytest.fixture
    def stager(self):
        return mici.stagers.WarmUpStager()

    @pytest.fixture
    def n_warm_up_iter(self):
        return 1


class TestWindowedWarmUpStager(StagerTests):
    @pytest.fixture
    def adapters(self):
        return {
            "integration_transition": [
                mici.adapters.DualAveragingStepSizeAdapter(),
                mici.adapters.OnlineVarianceMetricAdapter(),
            ]
        }

    @pytest.fixture(params=((), (125, 50, 25, 3)))
    def stager(self, request):
        return mici.stagers.WindowedWarmUpStager(*request.param)

    @pytest.fixture(params=(5, 10, 100, 107, 500, 1003))
    def n_warm_up_iter(self, request):
        return request.param
