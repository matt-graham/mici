"""Higher-level functional interface to Mici.

Functions for generating approximate samples from target distributions using Markov
chain Monte Carlo methods. These functions abstract away the details of constructing
the relevant Mici class objects, at the expense of less fine-grained control over the
algorithms used. For more complex use-cases, directly construct relevant classes from
the :py:mod:`.samplers`, :py:mod:`.systems` and :py:mod:`.integrators` modules and using
their object-oriented interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import mici

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mici.types import (
        ArrayFunction,
        GradientFunction,
        JacobianFunction,
        MatrixHessianProductFunction,
        ScalarFunction,
    )


def _preprocess_kwargs(*kwargs_dicts: dict) -> tuple[dict]:
    return tuple({} if d is None else d for d in kwargs_dicts)


def sample_hmc_chains(
    n_warm_up_iter: int,
    n_main_iter: int,
    init_states: Iterable[mici.states.ChainState | np.typing.NDArray],
    neg_log_dens: ScalarFunction,
    *,
    backend: str | None = None,
    seed: np.random.Generator | int | None = None,
    grad_neg_log_dens: GradientFunction | None = None,
    system_class: type[mici.systems.System] = mici.systems.EuclideanMetricSystem,
    integrator_class: type[
        mici.integrators.Integrator
    ] = mici.integrators.LeapfrogIntegrator,
    sampler_class: type[
        mici.samplers.HamiltonianMonteCarlo
    ] = mici.samplers.DynamicMultinomialHMC,
    system_kwargs: dict | None = None,
    integrator_kwargs: dict | None = None,
    sampler_kwargs: dict | None = None,
    **kwargs,
) -> mici.samplers.HMCSampleChainsOutputs:
    """
    Sample Hamiltonian Monte Carlo chains for a given target distribution.

    Samples one or more Markov chains with given initial states, with a stationary
    distribution specified by a function evaluating the negative log density of the
    target distribution on an unconstrained Euclidean space. Each chain has zero or more
    adaptive warm-up iterations, during which the parameters of the chain transitions
    can be automatically tuned.

    This function allows changing the types of the system, integrator and sampler
    classes used from their defaults and passing additional keyword arguments to their
    initialisers. For finer-grained control and more complex use cases it is recommended
    to initialise the objects directly and use the :code:`sample_chains` method of the
    resulting sampler object.

    Args:
        n_warm_up_iter: Number of adaptive warm up iterations per chain. Depending on
            the :py:class:`.stagers.Stager` instance specified by the :code:`stager`
            argument the warm up iterations may be split between one or more adaptive
            stages. If zero, only a single non-adaptive stage is used.
        n_main_iter: Number of iterations (samples to draw) per chain during main
            (non-adaptive) sampling stage.
        init_states: Initial chain states. Each state can be either an array  specifying
            the state position component or a :py:class:`.states.ChainState` instance.
            If an array is passed or the :code:`mom` attribute of the state is not set,
            a momentum component will be independently sampled from its conditional
            distribution. One chain will be run for each state in the iterable.
        neg_log_dens: Function which given a position array returns the negative
            logarithm of an unnormalized probability density on the position space with
            respect to a reference measure, with the corresponding distribution on the
            position space being the target distribution it is wished to draw
            approximate samples from.
        backend: Name of automatic differentiation backend to use. See
            :py:mod:`.autodiff` subpackage documentation for details of available
            backends. If `None` (the default) no automatic differentiation fallback will
            be used and so all derivative functions must be specified explicitly.
        seed: Integer value to seed NumPy random generator with, or :code:`None` to use
            default (non-fixed) seeding scheme, or an already seeded
            :py:class:`numpy.random.Generator` instance.
        grad_neg_log_dens: Function which given a position array returns the derivative
            of `neg_log_dens` with respect to the position array argument. Optionally
            the function may instead return a 2-tuple of values with the first being the
            array corresponding to the derivative and the second being the value of the
            `neg_log_dens` evaluated at the passed position array. If `None` is passed
            (the default) an automatic differentiation fallback will be used to attempt
            to construct the derivative of `neg_log_dens` automatically.
        system_class: The Hamiltonian system class to use.
        integrator_class: The symplectic integrator class to use.
        sampler_class: The Hamiltonian Monte Carlo sampler class to use.
        system_kwargs: Any additional keyword arguments to system class initialiser.
        integrator_kwargs: Any additional keyword arguments to integrator class
            intitialiser.
        sampler_kwargs: Any additional keyword arguments to sampler class
            initialiser.

    Keyword Args:
        **kwargs: Additional keyword arguments to pass to the
            :py:meth:`.samplers.HamiltonianMonteCarlo.sample_chains` method called to
            sample the chains.

    Returns:
        Named tuple :code:`(final_states, traces, statistics)` corresponding to states
        of chains after final iteration, dictionary of chain trace arrays and dictionary
        of chain statistics dictionaries.

    """
    rng = np.random.default_rng(seed)
    system_kwargs, integrator_kwargs, sampler_kwargs = _preprocess_kwargs(
        system_kwargs, integrator_kwargs, sampler_kwargs
    )
    system = system_class(
        neg_log_dens=neg_log_dens,
        grad_neg_log_dens=grad_neg_log_dens,
        backend=backend,
        **system_kwargs,
    )
    integrator = integrator_class(system=system, **integrator_kwargs)
    sampler = sampler_class(
        system=system, integrator=integrator, rng=rng, **sampler_kwargs
    )
    return sampler.sample_chains(
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        init_states=init_states,
        **kwargs,
    )


def sample_constrained_hmc_chains(
    n_warm_up_iter: int,
    n_main_iter: int,
    init_states: Iterable[mici.states.ChainState | np.typing.NDArray],
    neg_log_dens: ScalarFunction,
    constr: ArrayFunction,
    *,
    backend: str | None = None,
    seed: np.random.Generator | int | None = None,
    grad_neg_log_dens: GradientFunction | None = None,
    jacob_constr: JacobianFunction | None = None,
    mhp_constr: MatrixHessianProductFunction | None = None,
    dens_wrt_hausdorff: bool = True,
    system_class: type[
        mici.systems.System
    ] = mici.systems.DenseConstrainedEuclideanMetricSystem,
    integrator_class: type[
        mici.integrators.Integrator
    ] = mici.integrators.ConstrainedLeapfrogIntegrator,
    sampler_class: type[
        mici.samplers.HamiltonianMonteCarlo
    ] = mici.samplers.DynamicMultinomialHMC,
    system_kwargs: dict | None = None,
    integrator_kwargs: dict | None = None,
    sampler_kwargs: dict | None = None,
    **kwargs,
) -> mici.samplers.HMCSampleChainsOutputs:
    """
    Sample constrained Hamiltonian Monte Carlo chains for a given target distribution.

    Samples one or more Markov chains with given initial states, with a stationary
    distribution on an implicitly-defined manifold embedded in an ambient Euclidean
    space, specified by functions evaluating the negative log density of the target
    distribution and specified by a constraint function for which the zero level-set of
    specifies the manifold the distribution is supported on. Each chain has zero or more
    adaptive warm-up iterations, during which the parameters of the chain transitions
    can be automatically tuned.

    This function allows changing the types of the system, integrator and sampler
    classes used from their defaults and passing additional keyword arguments to their
    initialisers. For finer-grained control and more complex use cases it is recommended
    to initialise the objects directly and use the :code:`sample_chains` method of the
    resulting sampler object.

    Args:
        n_warm_up_iter: Number of adaptive warm up iterations per chain. Depending on
            the :py:class:`.stagers.Stager` instance specified by the :code:`stager`
            argument the warm up iterations may be split between one or more adaptive
            stages. If zero, only a single non-adaptive stage is used.
        n_main_iter: Number of iterations (samples to draw) per chain during main
            (non-adaptive) sampling stage.
        init_states: Initial chain states. Each state can be either an array  specifying
            the state position component or a :py:class:`.states.ChainState` instance.
            If an array is passed or the :code:`mom` attribute of the state is not set,
            a momentum component will be independently sampled from its conditional
            distribution. One chain will be run for each state in the iterable.
        neg_log_dens: Function which given a position array returns the negative
            logarithm of an unnormalized probability density on the position space with
            respect to a reference measure, with the corresponding distribution on the
            position space being the target distribution it is wished to draw
            approximate samples from.
        constr: Function which given a position array return as a 1D array the value
            of the (vector-valued) constraint function, the zero level-set of which
            implicitly defines the manifold the dynamic is simulated on.
        backend: Name of automatic differentiation backend to use. See
            :py:mod:`.autodiff` subpackage documentation for details of available
            backends. If `None` (the default) no automatic differentiation fallback will
            be used and so all derivative functions must be specified explicitly.
        seed: Integer value to seed NumPy random generator with, or :code:`None` to use
            default (non-fixed) seeding scheme, or an already seeded
            :py:class:`numpy.random.Generator` instance.
        grad_neg_log_dens: Function which given a position array returns the derivative
            of `neg_log_dens` with respect to the position array argument. Optionally
            the function may instead return a 2-tuple of values with the first being the
            array corresponding to the derivative and the second being the value of the
            `neg_log_dens` evaluated at the passed position array. If `None` is passed
            (the default) an automatic differentiation fallback will be used to attempt
            to construct the derivative of `neg_log_dens` automatically.
        jacob_constr: Function which given a position array computes the Jacobian
            (matrix / 2D array of partial derivatives) of the output of the constraint
            function :code:`c = constr(q)` with respect to the position array argument
            :code:`q`, returning the computed Jacobian as a 2D array :code:`jacob` with
            :code:`jacob[i, j] = ∂c[i] / ∂q[j]`. Optionally the function may instead
            return a 2-tuple of values with the first being the array corresponding to
            the Jacobian and the second being the value of :code:`constr` evaluated at
            the passed position array. If :code:`None` is passed (the default) an
            automatic differentiation fallback will be used to attempt to construct a
            function to compute the Jacobian (and value) of :code:`constr`
            automatically.
        mhp_constr: Function which given a position array returns another function
            which takes a 2D array as an argument and returns the
            *matrix-Hessian-product* (MHP) of the constraint function :code:`constr`
            with respect to the position array argument. The MHP is here defined as a
            function of a :code:`(dim_constr, dim_pos)` shaped 2D array :code:`m` as
            :code:`mhp(m) = sum(m[:, :, None] * hess[:, :, :], axis=(0, 1))` where
            :code:`hess` is the :code:`(dim_constr, dim_pos, dim_pos)` shaped
            vector-Hessian of :code:`c = constr(q)` with respect to :code:`q` i.e. the
            array of second-order partial derivatives of such that :code:`hess[i, j, k]
            = ∂²c[i] / (∂q[j] ∂q[k])`. Optionally the function may instead return a
            3-tuple of values with the first a function to compute a MHP of
            :code:`constr`, the second a 2D array corresponding to the Jacobian of
            :code:`constr`, and the third the value of :code:`constr`, all evaluated at
            the passed position array. If :code:`None` is passed (the default) an
            automatic differentiation fallback will be used to attempt to construct a
            function which calculates the MHP (and Jacobian and value) of :code:`constr`
            automatically. Only used if :code:`dens_wrt_hausdorff == False`.
        dens_wrt_hausdorff: If :code:`constr` is specified, whether the
            :code:`neg_log_dens` function specifies the (negative logarithm) of the
            density of the target distribution with respect to the Hausdorff measure on
            the manifold directly (:code:`True`) or alternatively the negative logarithm
            of a density of a prior distriubtion on the unconstrained (ambient) position
            space with respect to the Lebesgue measure, with the target distribution
            then corresponding to the posterior distribution when conditioning on the
            event :code:`constr(pos) == 0` (:code:`False`).
        system_class: The Hamiltonian system class to use.
        integrator_class: The symplectic integrator class to use.
        sampler_class: The Hamiltonian Monte Carlo sampler class to use.
        system_kwargs: Any additional keyword arguments to system class initialiser.
        integrator_kwargs: Any additional keyword arguments to integrator class
            intitialiser.
        sampler_kwargs: Any additional keyword arguments to sampler class
            initialiser.

    Keyword Args:
        **kwargs: Additional keyword arguments to pass to the
            :py:meth:`.samplers.HamiltonianMonteCarlo.sample_chains` method called
            to sample the chains.

    Returns:
        Named tuple :code:`(final_states, traces, statistics)` corresponding to
        states of chains after final iteration, dictionary of chain trace arrays
        and dictionary of chain statistics dictionaries.

    """
    rng = np.random.default_rng(seed)
    system_kwargs, integrator_kwargs, sampler_kwargs = _preprocess_kwargs(
        system_kwargs, integrator_kwargs, sampler_kwargs
    )
    system = system_class(
        neg_log_dens=neg_log_dens,
        grad_neg_log_dens=grad_neg_log_dens,
        constr=constr,
        jacob_constr=jacob_constr,
        mhp_constr=mhp_constr,
        dens_wrt_hausdorff=dens_wrt_hausdorff,
        backend=backend,
        **system_kwargs,
    )
    integrator = integrator_class(system=system, **integrator_kwargs)
    sampler = sampler_class(
        system=system, integrator=integrator, rng=rng, **sampler_kwargs
    )
    return sampler.sample_chains(
        n_warm_up_iter=n_warm_up_iter,
        n_main_iter=n_main_iter,
        init_states=init_states,
        **kwargs,
    )
