"""Utilities for interfacing with external probabilistic programming libraries."""

from __future__ import annotations

import importlib
import os
import mici
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, Optional, Union
    from numpy.typing import ArrayLike
    import arviz
    import pymc3


def convert_to_inference_data(
    traces: dict[str, list[ArrayLike]],
    stats: dict[str, list[ArrayLike]],
    energy_key: Optional[str] = "energy",
    lp_key: Optional[str] = "lp",
) -> arviz.InferenceData:
    """Convert Mici :code:`sample_chains` output to :py:class:`arviz.InferenceData` object.

    Args:
        traces: Traces output from Mici 
            :py:meth:`mici.samplers.MarkovChainMonteCarloMethod.sample_chains` call. A
            dictionary of variables traced over sampled chains with the dictionary keys
            the variable names and the values a list of arrays, one array per sampled
            chain, with the first array dimension corresponding to the draw index and
            any remaining dimensions, the variable dimensions.
        stats: Statistics output from Mici `sample_chains` call. A dictionary of chain
            statistics traced over sampled chains with the dictionary keys the
            statistics names and the values a list of arrays, one array per sampled
            chain, with the array dimension corresponding to the draw index.
        energy_key: The key of an entry in the `traces` dictionary corresponding the
            value of the Hamiltonian energy for the accepted proposal (up to an additive
            constant). If present the corresponding values will be added to the
            `sample_stats` group of the returned `InferenceData` object.
        lp_key: The key of an entry in the `traces` dictionary corresponding the value
            of the joint log posterior density for the model (up to an additive
            constant). If present the corresponding values will be added to the
            `sample_stats` group of the returned `InferenceData` object.

    Returns:
        ArviZ inference data object with traced chain data stored in the `posterior`
        group and additional chain statistics in the `sample_stats` group.
    """
    import arviz

    stats = stats.copy()
    stats["n_steps"] = stats.pop("n_step")
    stats["acceptance_rate"] = stats.pop("accept_stat")
    if energy_key is not None and energy_key in traces:
        stats["energy"] = traces[energy_key]
    if lp_key is not None and lp_key in traces:
        stats["lp"] = traces[lp_key]
    return arviz.InferenceData(
        posterior=arviz.dict_to_dataset(traces, library=mici),
        sample_stats=arviz.dict_to_dataset(stats, library=mici),
    )


def sample_pymc3_model(
    draws: int = 1000,
    *,
    tune: int = 1000,
    chains: Optional[int] = None,
    cores: Optional[int] = None,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    init: Literal["auto", "adapt_diag", "jitter+adapt_diag", "adapt_full"] = "auto",
    jitter_max_retries: int = 10,
    trace: Optional[list] = None,
    return_inferencedata: bool = False,
    model: Optional[pymc3.model.Model] = None,
    target_accept: float = 0.8,
    max_treedepth: int = 10,
) -> Union[arviz.InferenceData, dict[str, ArrayLike]]:
    """Generate approximate samples from posterior defined by a PyMC3 model.

    Uses dynamic multinomial HMC algorithm in Mici with adaptive warm-up phase.

    This function replicates the interface of the :py:func:`pymc3.sampling.sample`
    function to allow using as a (partial) drop-in replacement.

    Args:
        draws: The number of samples to draw.
        tune: Number of iterations to tune, defaults to 1000. Samplers adjust the step
            sizes, scalings or similar during tuning. Tuning samples will be drawn in
            addition to the number specified in the `draws` argument, and will be
            discarded.
        chains: The number of chains to sample. Running independent chains is important
            for some convergence statistics and can also reveal multiple modes in the
            posterior. If :code::`None`, then set to either :code:`cores` or 2,
            whichever is larger.
        cores: The number of chains to run in parallel. If :code:`None`, set to the
            number of CPU cores in the system, but at most 4.
        random_seed: Seed for NumPy random number generator used for generating random
            variables while sampling chains. If :code:`None` then generator will be
            seeded with entropy from operating system.
        progressbar: Whether or not to display a progress bar.
        init: Initialization method to use. One of:

            * :code:`"adapt_diag"`: Start with a identity mass matrix and then adapt a
              diagonal based on the variance of the tuning samples. All chains use the
              test value (usually the prior mean) as starting point.
            * :code:`jitter+adapt_diag`: Same as :code:`"adapt_diag"`, but add uniform
              jitter in [-1, 1] to the starting point in each chain. Also chosen if
              :code:`init="auto"`.
            * :code:`"adapt_full"`: Adapt a dense mass matrix using the sample
              covariances.

        jitter_max_retries: Maximum number of repeated attempts (per chain) at creating
            an initial matrix with uniform jitter that yields a finite probability. This
            applies to `"jitter+adapt_diag"` and :code:`"jitter+adapt_full"`
            :py:obj:`init` methods.
        trace: A list of variables to track. If :code`None`, then
            :code:`model.unobserved_RVs` is used.
        return_inferencedata: Whether to return the traces as an
            :py:class:`arviz.InferenceData` (:code:`True`) object or a
            :py:class:`dict` (:code:`False`).
        model: PyMC3 model defining posterior distribution to sample from. May be 
            :code:`None` if function is called from within model context manager.
        target_accept: Target value for the acceptance statistic being controlled during
            adaptive warm-up.
        max_treedepth: Maximum depth to expand trajectory binary tree to in integrator
            transition. The maximum number of integrator steps corresponds to
            :code:`2**max_treedepth`.

    Returns:
        A dictionary or :py:class:`arviz.InferenceData` object containing the sampled
        chain output. Dictionary output (when :code:`return_inferencedata=False`) has
        string keys corresponding to the name of each traced variable in the model, with
        the values being the corresponding values of the variables traced across the
        chains as NumPy arrays, with the first dimension the chain index  (of size equal
        to :code:`chains`), the second dimension the draw index (of size equal to
        :code:`draws`) and any remaining dimensions corresponding to the dimensions of
        the traced variable. If :code:`return_inferencedata=True` an
        :py:class:`arviz.InferenceData` object is instead returned with traced chain
        data stored in the :code:`posterior` group and additional chain statistics in
        the :code:`sample_stats` group.
    """

    import pymc3

    if return_inferencedata and importlib.util.find_spec("arviz") is None:
        raise ValueError("Cannot return InferenceData as ArviZ is not installed")

    model = pymc.modelcontext(model)

    if cores is None:
        cores = min(4, os.cpu_count() // 2)  # assume 2 threads per CPU core

    if chains is None:
        chains = max(2, cores)

    if trace is None:
        trace = model.unobserved_RVs

    if init == "auto" or "jitter+adapt_diag":
        use_dense_metric = False
        jitter_init = True
    elif init == "adapt_diag":
        use_dense_metric = False
        jitter_init = False
    elif init == "adapt_full":
        use_dense_metric = True
        jitter_init = False
    else:
        raise ValueError(
            'init must be "auto", "jitter+adapt_diag", "adapt_diag" or "adapt_full"'
        )

    val_and_grad_log_dens = model.logp_dlogp_function()
    val_and_grad_log_dens.set_extra_values({})

    def grad_neg_log_dens(pos):
        val, grad = val_and_grad_log_dens(pos)
        return -grad, -val

    def neg_log_dens(pos):
        val, _ = val_and_grad_log_dens(pos)
        return -val

    def trace_func(state):
        var_dict = val_and_grad_log_dens.array_to_dict(state.pos)
        trace_dict = {}
        for rv in trace:
            if rv.name in var_dict:
                trace_dict[rv.name] = var_dict[rv.name]
            else:
                trace_dict[rv.name] = rv.transformation.backward(
                    var_dict[rv.transformed.name]
                ).eval()
        trace_dict["lp"] = -system.neg_log_dens(state)
        trace_dict["energy"] = system.h(state)
        return trace_dict

    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
    )

    integrator = mici.integrators.LeapfrogIntegrator(system)
    rng = np.random.default_rng(random_seed)
    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_treedepth=max_treedepth
    )

    step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(target_accept)
    metric_adapter = (
        mici.adapters.OnlineCovarianceMetricAdapter()
        if use_dense_metric
        else mici.adapters.OnlineVarianceMetricAdapter()
    )
    adapters = [step_size_adapter, metric_adapter]

    if jitter_init:
        mean = val_and_grad_log_dens.dict_to_array(model.test_point)
        init_states = []
        for c in range(chains):
            for t in range(jitter_max_retries):
                pos = mean + rng.uniform(-1, 1, mean.shape)
                if np.isfinite(val_and_grad_log_dens(pos)[0]):
                    break
            init_states.append(pos)
    else:
        init_states = [
            val_and_grad_log_dens.dict_to_array(model.test_point) for c in range(chains)
        ]

    _, traces, stats = sampler.sample_chains(
        n_warm_up_iter=tune,
        n_main_iter=draws,
        init_states=init_states,
        adapters=adapters,
        trace_funcs=[trace_func],
        n_process=cores,
        display_progress=progressbar,
        monitor_stats=["accept_stat", "n_step", "diverging"],
    )

    if return_inferencedata:
        return convert_to_inference_data(traces, stats)
    else:
        return {k: np.stack(v) for k, v in traces.items()}


def _get_stan_model_unconstrained_param_dim(model):
    param_size_list = [np.prod(dim, dtype=np.int64) for dim in model.dims]
    n_dim = sum(param_size_list)
    while True:
        try:
            model.log_prob([0] * n_dim)
            return n_dim
        except RuntimeError:
            param_size_list.pop()
            n_dim = sum(param_size_list)


def sample_stan_model(
    model_code: str,
    data: dict,
    *,
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    save_warmup: bool = False,
    metric: Literal["unit_e", "diag_e", "dense_e"] = "diag_e",
    stepsize: float = 1.0,
    adapt_engaged: bool = True,
    delta: float = 0.8,
    gamma: float = 0.05,
    kappa: float = 0.75,
    t0: int = 10,
    init_buffer: int = 75,
    term_buffer: int = 50,
    window: int = 25,
    max_depth: int = 10,
    seed: Optional[int] = None,
    return_inferencedata: bool = False,
) -> Union[arviz.InferenceData, dict[str, ArrayLike]]:
    """Generate approximate samples from posterior defined by a Stan model.

    Uses dynamic multinomial HMC algorithm in Mici with adaptive warm-up phase.

    This function follows a similar argument naming scheme to the PyStan
    :py:meth:`stan.model.Model.sample` method (which itself follows CmdStan) to allow
    using as a (partial) drop-in replacement.

    Args:
        model_code: Stan program code describing a Stan model.
        data: A Python dictionary or mapping providing the data for the model. Variable
            names are the keys and the values are their associated values.
        num_samples: A non-negative integer specifying the number of non-warm-up
            iterations per chain.
        num_warmup: A non-negative integer specifying the number of warm-up iterations
            per chain.
        num_chains: A positive integer specifying the number of Markov chains.
        save_warmup: Whether to save warm-up chain data (`True`) or not (`False`).
        metric: String specifying metric type. One of "unit_e", "diag_e" or "dense_e",
            indicating respectively to used a fixed identity matrix metric
            representation, to use a diagonal metric matrix representation adapted based
            on estimates of the marginal posterior variances, to use a dense metric
            matrix representation based on estimates of the posterior covariance matrix.
        stepsize: Initial integrator step size.
        adapt_engaged: Whether adaptation is engaged (`True`) or not (`False`).
        delta: Adaptation target acceptance statistic.
        gamma: Adaptation regularization scale.
        kappa: Adaptation relaxation exponent.
        t0: Adaptation iteration offset.
        init_buffer: Width of initial fast adaptation interval.
        term_buffer: Width of final fast adaptation interval.
        window: Initial width of slow adaptation interval.
        max_depth: Maximum depth of binary trajectory tree.
        seed: Seed for Numpy random number generator used for generating random
            variables while sampling chains. If `None` then generator will be seeded
            with entropy from operating system.
        return_inferencedata: Whether to return the traces as an `arviz.InferenceData`
            (`True`) object or a dict (`False`).

    Returns:
        A dictionary or ArviZ `InferenceData` object containing the sampled chain
        output. Dictionary output (when `return_inferencedata=False`) has string keys
        corresponding to the name of each traced variable in the model, with the values
        being the corresponding values of the variables traced across the chains as
        NumPy arrays, with the first dimension the flattened draw index across all
        chains (of size equal to `num_chains * num_samples`) and any remaining
        dimensions corresponding to the dimensions of the traced variable. If
        `return_inferencedata=True` an ArviZ `InferenceData` object is instead returned
        with traced chain data stored in the `posterior` group and additional chain
        statistics in the `sample_stats` group.
    """

    import stan

    if return_inferencedata and importlib.util.find_spec("arviz") is None:
        raise ValueError("Cannot return InferenceData as ArviZ is not installed")

    model = stan.build(model_code, data=data)

    def neg_log_dens(u):
        return -model.log_prob(list(u))

    def grad_neg_log_dens(u):
        return -np.array(model.grad_log_prob(list(u)))

    param_size_list = [np.prod(dim, dtype=np.int64) for dim in model.dims]

    def trace_func(state):
        param_array = np.array(model.constrain_pars(list(state.pos)))
        trace_dict = {
            name: val.reshape(shape)
            for name, val, shape in zip(
                model.param_names,
                np.split(param_array, np.cumsum(param_size_list)[:-1]),
                model.dims,
            )
        }
        trace_dict["lp"] = -system.neg_log_dens(state)
        trace_dict["energy"] = system.h(state)
        return trace_dict

    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=neg_log_dens, grad_neg_log_dens=grad_neg_log_dens
    )

    integrator = mici.integrators.LeapfrogIntegrator(system, step_size=stepsize)
    rng = np.random.default_rng(seed)
    sampler = mici.samplers.DynamicMultinomialHMC(
        system, integrator, rng, max_tree_depth=max_depth
    )

    if adapt_engaged:
        step_size_adapter = mici.adapters.DualAveragingStepSizeAdapter(
            adapt_stat_target=delta,
            iter_offset=t0,
            iter_decay_coeff=kappa,
            log_step_size_reg_coefficient=gamma,
        )
        adapters = [step_size_adapter]

        if metric == "diag_e":
            adapters.append(mici.adapters.OnlineVarianceMetricAdapter())
        elif metric == "dense_e":
            adapters.append(mici.adapters.OnlineCovarianceMetricAdapter())

        if len(adapters) > 1:
            stager = mici.stagers.WindowedWarmUpStager(
                n_init_fast_stage_iter=init_buffer,
                n_final_fast_stage_iter=term_buffer,
                n_init_slow_window_iter=window,
            )
        else:
            stager = mici.stagers.WarmUpStager()
    else:
        adapters = None
        stager = None

    dim_u = _get_stan_model_unconstrained_param_dim(model)
    init_states = rng.uniform(-2, 2, size=(num_chains, dim_u))

    _, traces, stats = sampler.sample_chains(
        n_warm_up_iter=num_warmup,
        n_main_iter=num_samples,
        init_states=init_states,
        adapters=adapters,
        stager=stager,
        trace_funcs=[trace_func],
        monitor_stats=["accept_stat", "n_step", "diverging"],
        trace_warmup=save_warmup,
    )

    if return_inferencedata:
        return convert_to_inference_data(traces, stats)
    else:
        return {k: np.concatenate(v).swapaxes(0, -1) for k, v in traces.items()}
