import numpy as np
import pytest

from mici.interface import sample_constrained_hmc_chains, sample_hmc_chains
from mici.samplers import HMCSampleChainsOutputs
from mici.states import ChainState

SEED = 3046987125


def neg_log_dens(q):
    return (q**2).sum() / 2


def grad_neg_log_dens(q):
    return q


def constr(q):
    return q[0:1]


def jacob_constr(q):
    jacob = np.zeros((1, q.shape[0]))
    jacob[:, 0] = 1
    return jacob


def _check_sample_chains_output(
    results: HMCSampleChainsOutputs,
    *,
    n_warm_up_iter: int,
    n_main_iter: int,
    n_chain: int,
    trace_warm_up: bool,
) -> None:
    n_trace_iter = n_warm_up_iter + n_main_iter if trace_warm_up else n_main_iter
    assert isinstance(results, HMCSampleChainsOutputs)
    assert len(results.final_states) == n_chain
    assert all(isinstance(state, ChainState) for state in results.final_states)
    assert all(len(trace) == n_chain for trace in results.traces.values())
    assert all(len(stat) == n_chain for stat in results.statistics.values())
    assert all(
        t.shape[0] == n_trace_iter for trace in results.traces.values() for t in trace
    )
    assert all(
        s.shape[0] == n_trace_iter for stat in results.statistics.values() for s in stat
    )


@pytest.mark.parametrize(("n_warm_up_iter", "n_main_iter"), [(5, 5), (0, 5), (5, 0)])
@pytest.mark.parametrize("n_chain", [1, 2])
@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("trace_warm_up", [True, False])
@pytest.mark.parametrize("backend", [None, "jax", "autograd"])
def test_sample_hmc_chains(
    n_warm_up_iter, n_main_iter, n_chain, dimension, trace_warm_up, backend
):
    rng = np.random.default_rng(SEED)
    init_states = rng.standard_normal((n_chain, dimension))
    if backend is None:
        kwargs = {"grad_neg_log_dens": grad_neg_log_dens, "n_worker": n_chain}
    else:
        kwargs = {}
    integrator_kwargs = {"step_size": 0.5} if n_warm_up_iter == 0 else {}
    results = sample_hmc_chains(
        n_warm_up_iter,
        n_main_iter,
        init_states,
        neg_log_dens,
        backend=backend,
        trace_warm_up=trace_warm_up,
        seed=rng,
        integrator_kwargs=integrator_kwargs,
        **kwargs,
    )
    _check_sample_chains_output(
        results,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        n_chain=n_chain,
        trace_warm_up=trace_warm_up,
    )


@pytest.mark.parametrize(("n_warm_up_iter", "n_main_iter"), [(5, 5), (0, 5), (5, 0)])
@pytest.mark.parametrize("n_chain", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("trace_warm_up", [True, False])
@pytest.mark.parametrize("backend", [None, "jax", "autograd"])
def test_sample_constrained_hmc_chains(
    n_warm_up_iter, n_main_iter, n_chain, dimension, trace_warm_up, backend
):
    rng = np.random.default_rng(SEED)
    init_states = rng.standard_normal((n_chain, dimension))
    init_states[:, 0] = 0
    assert all(np.allclose(constr(s), 0.0) for s in init_states)
    if backend is None:
        kwargs = {
            "grad_neg_log_dens": grad_neg_log_dens,
            "jacob_constr": jacob_constr,
            "n_worker": n_chain,
        }
    else:
        kwargs = {}
    integrator_kwargs = {"step_size": 0.25} if n_warm_up_iter == 0 else {}
    results = sample_constrained_hmc_chains(
        n_warm_up_iter,
        n_main_iter,
        init_states,
        neg_log_dens,
        constr,
        backend=backend,
        trace_warm_up=trace_warm_up,
        seed=rng,
        integrator_kwargs=integrator_kwargs,
        **kwargs,
    )
    _check_sample_chains_output(
        results,
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        n_chain=n_chain,
        trace_warm_up=trace_warm_up,
    )
