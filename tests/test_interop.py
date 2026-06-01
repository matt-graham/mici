import importlib

import numpy as np
import pytest
from packaging.version import parse as parse_version

import mici

try:
    import arviz

    ARVIZ_AVAILABLE = importlib.util.find_spec("arviz") is not None
except ImportError:
    ARVIZ_AVAILABLE = False

try:
    import pymc

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


try:
    import stan  # noqa: F401

    STAN_AVAILABLE = True
except ImportError:
    STAN_AVAILABLE = False


SEED = 3046987125
NUM_CHAIN = 2
NUM_SAMPLE = 10


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def traces(rng):
    return {
        "a": list(rng.standard_normal((NUM_CHAIN, NUM_SAMPLE))),
        "b": list(rng.standard_normal((NUM_CHAIN, NUM_SAMPLE))),
        "energy": list(rng.standard_normal((NUM_CHAIN, NUM_SAMPLE))),
        "lp": list(rng.standard_normal((NUM_CHAIN, NUM_SAMPLE))),
    }


@pytest.fixture
def stats(rng):
    return {
        "accept_stat": list(rng.uniform(0, 1, (NUM_CHAIN, NUM_SAMPLE))),
        "n_step": list(rng.integers(0, 10, (NUM_CHAIN, NUM_SAMPLE))),
    }


if ARVIZ_AVAILABLE:

    def _check_arviz_data_object(data, traces):
        for group in ("posterior", "sample_stats"):
            assert group in data
            assert data[group].coords["chain"].size == NUM_CHAIN
            assert data[group].coords["draw"].size == NUM_SAMPLE
        assert set(data.posterior.keys()) == set(traces.keys())
        assert set(data.sample_stats.keys()) == {
            "acceptance_rate",
            "n_steps",
            "energy",
            "lp",
        }

    def test_convert_to_inference_data(traces, stats):
        if parse_version(arviz.__version__) >= parse_version("1.0.0"):
            pytest.skip("arviz.InferenceData only available in versions < 1.0")
        inference_data = mici.interop.convert_to_inference_data(traces, stats)
        assert isinstance(inference_data, arviz.InferenceData)
        _check_arviz_data_object(inference_data, traces)

    def test_convert_to_inference_data_raises(traces, stats):
        if parse_version(arviz.__version__) <= parse_version("1.0.0"):
            pytest.skip(
                "arviz.InferenceData available in versions < 1.0 so error not expected"
            )
        with pytest.raises(RuntimeError, match=r"InferenceData was removed"):
            mici.interop.convert_to_inference_data(traces, stats)

    def test_convert_to_datatree(traces, stats):
        if parse_version(arviz.__version__) < parse_version("1.0.0"):
            pytest.skip("DataTree only available in ArviZ versions >= 1.0")
        data_tree = mici.interop.convert_to_datatree(traces, stats)
        _check_arviz_data_object(data_tree, traces)

    def test_convert_to_datatree_raises(traces, stats):
        if parse_version(arviz.__version__) >= parse_version("1.0.0"):
            pytest.skip(
                "DataTree available in ArviZ versions >= 1.0 so error not expected"
            )
        with pytest.raises(RuntimeError, match=r"xarray\.DataTree support"):
            mici.interop.convert_to_datatree(traces, stats)


if PYMC_AVAILABLE:

    @pytest.fixture
    def pymc_model():
        with pymc.Model() as model:
            pymc.Normal("x", mu=0, sigma=1)
            pymc.Uniform("y", lower=0, upper=1)
        return model

    @pytest.mark.parametrize("init", ["adapt_diag", "jitter+adapt_diag", "adapt_full"])
    @pytest.mark.parametrize("progressbar", [True, False])
    def test_sample_pymc_model(pymc_model, init, progressbar):
        traces = mici.interop.sample_pymc_model(
            model=pymc_model,
            draws=NUM_SAMPLE,
            tune=NUM_SAMPLE,
            random_seed=SEED,
            init=init,
            progressbar=progressbar,
            cores=1,
        )
        assert isinstance(traces, dict)
        for var in pymc_model.unobserved_RVs:
            assert var.name in traces
            assert traces[var.name].shape[1] == NUM_SAMPLE
            assert not np.any(np.isnan(traces[var.name]))

    if ARVIZ_AVAILABLE:

        def test_sample_pymc_model_inference_data(pymc_model):
            if parse_version(arviz.__version__) >= parse_version("1.0.0"):
                pytest.skip("arviz.InferenceData only available in versions < 1.0")
            inference_data = mici.interop.sample_pymc_model(
                model=pymc_model,
                chains=NUM_CHAIN,
                draws=NUM_SAMPLE,
                random_seed=SEED,
                return_inferencedata=True,
                cores=1,
            )
            assert isinstance(inference_data, arviz.InferenceData)
            for var in pymc_model.unobserved_RVs:
                assert var.name in inference_data.posterior
                assert not np.any(np.isnan(inference_data.posterior[var.name]))
            assert inference_data.posterior.coords["chain"].size == NUM_CHAIN
            assert inference_data.posterior.coords["draw"].size == NUM_SAMPLE


if STAN_AVAILABLE:

    @pytest.fixture
    def stan_model_code_data_and_params():
        model_code = """
        parameters {
            real x;
            real <lower=0, upper=1> y;
        }
        model {
            x ~ normal(0, 1);
            y ~ uniform(0, 1);
        }
        """
        return model_code, {}, {"x", "y"}

    @pytest.mark.parametrize("metric", ["unit_e", "diag_e", "dense_e"])
    @pytest.mark.parametrize("adapt_engaged", [True, False])
    def test_sample_stan_model(stan_model_code_data_and_params, metric, adapt_engaged):
        model_code, data, params = stan_model_code_data_and_params
        traces = mici.interop.sample_stan_model(
            model_code=model_code,
            data=data,
            num_chains=NUM_CHAIN,
            num_samples=NUM_SAMPLE,
            num_warmup=NUM_SAMPLE,
            seed=SEED,
            metric=metric,
            adapt_engaged=adapt_engaged,
        )
        assert isinstance(traces, dict)
        for param in params:
            assert param in traces
            assert traces[param].shape[0] == NUM_SAMPLE * NUM_CHAIN
            assert not np.any(np.isnan(traces[param]))

    if ARVIZ_AVAILABLE:

        def test_sample_stan_model_inference_data(stan_model_code_data_and_params):
            if parse_version(arviz.__version__) >= parse_version("1.0.0"):
                pytest.skip("arviz.InferenceData only available in versions < 1.0")
            model_code, data, params = stan_model_code_data_and_params
            inference_data = mici.interop.sample_stan_model(
                model_code=model_code,
                data=data,
                num_chains=NUM_CHAIN,
                num_samples=NUM_SAMPLE,
                num_warmup=NUM_SAMPLE,
                seed=SEED,
                return_inferencedata=True,
            )
            assert isinstance(inference_data, arviz.InferenceData)
            for param in params:
                assert param in inference_data.posterior
                assert not np.any(np.isnan(inference_data.posterior[param]))
            assert inference_data.posterior.coords["chain"].size == NUM_CHAIN
            assert inference_data.posterior.coords["draw"].size == NUM_SAMPLE
