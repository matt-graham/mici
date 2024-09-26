"""Automatic differentation fallback for constructing derivative functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from mici import autograd_wrapper, jax_wrapper, symnum_wrapper

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Callable, Optional


"""Names of valid differential operators.

Any automatic differentiation framework wrapper module will need to provide all of these
operators as callables (with a single function as argument) to fully support all of the
required derivative functions.
"""
DIFF_OPS = (
    # vector Jacobian product and value
    "vjp_and_value",
    # gradient and value for scalar valued functions
    "grad_and_value",
    # Hessian matrix, gradient and value for scalar valued functions
    "hessian_grad_and_value",
    # matrix Tressian product, gradient and value for scalar valued functions
    "mtp_hessian_grad_and_value",
    # Jacobian matrix and value for vector valued functions
    "jacobian_and_value",
    # matrix Hessian product, Jacobian matrix and value for vector valued functions
    "mhp_jacobian_and_value",
)


class AutodiffBackend(NamedTuple):
    """Automatic differentiation backend framework.

    Consists of a module defining differential operators, a boolean flag indicating if
    backend is available in current environment and optionally a function wrapper which
    applies any post processing required to functions.
    """

    module: ModuleType
    available: bool
    function_wrapper: Optional[Callable] = None


"""Available autodifferentiation framework backends."""
_REGISTERED_BACKENDS = {
    "jax": AutodiffBackend(
        jax_wrapper,
        jax_wrapper.JAX_AVAILABLE,
        jax_wrapper.jit_and_return_numpy_arrays,
    ),
    "jax_nojit": AutodiffBackend(
        jax_wrapper,
        jax_wrapper.JAX_AVAILABLE,
        jax_wrapper.return_numpy_arrays,
    ),
    "autograd": AutodiffBackend(autograd_wrapper, autograd_wrapper.AUTOGRAD_AVAILABLE),
    "symnum": AutodiffBackend(symnum_wrapper, symnum_wrapper.SYMNUM_AVAILABLE),
}

"""Name of default automatic differentiation backend to use.

Defaults to first available backend from `jax`, `autograd` and `symnum` (in that order)
or to `None` if none are available.
"""
DEFAULT_BACKEND = next(
    (name for name, backend in _REGISTERED_BACKENDS.items() if backend.available), None,
)


def _get_backend(name: str):
    # Normalize name string to all lowercase to make invariant to capitalization
    name = name.lower()
    if name not in _REGISTERED_BACKENDS:
        msg = (
            f"Selected autodiff backend {name} not recognised: "
            f"available options are {tuple(_REGISTERED_BACKENDS)}."
        )
        raise ValueError(msg)
    return _REGISTERED_BACKENDS[name]


def wrap_function(function: Callable, backend: Optional[str]):
    """Apply function wrapper for automatic differentiation backend to a function.

    Backends may define a function wrapper which applies any post processing required to
    functions using framework - for example ensuring the function returns NumPy arrays
    or just-in-time compiling the function.

    Args:
        function: Function to wrap.
        backend: Name of automatic differentiation framework backend to use. If `None`
            function is returned unchanged.

    Returns:
        Wrapped function.
    """
    if backend is None:
        return function
    backend = _get_backend(backend)
    if backend.function_wrapper is not None:
        return backend.function_wrapper(function)
    else:
        return function


def autodiff_fallback(
    diff_func: Optional[Callable],
    func: Callable,
    diff_op_name: str,
    name: str,
    backend: Optional[str],
) -> Callable:
    """Generate derivative function automatically if not provided.

    Uses automatic differentiation to generate a function corresponding to a
    differential operator applied to a function if an alternative implementation of the
    derivative function has not been provided.

    Args:
        diff_func: Either a callable implementing the required derivative function or
            `None` if none was provided.
        func: Function to differentiate.
        diff_op_name: String specifying name of differential operator from automatic
            differentiation framework wrapper to use to generate required derivative
            function.
        name: Name of derivative function to use in error message.
        backend: Name of automatic differentiation framework backend to use. If `None`
            `diff_func` must be provided.

    Returns:
        `diff_func` value if not `None` otherwise generated derivative of `func` by
        applying named differential operator from automatic differentiation backend.
    """
    if diff_func is not None:
        return diff_func
    elif diff_func is None and backend is None:
        msg = (
            f"Automatic differentiation backend specified as `None` so {name} must"
            "be provided directly."
        )
        raise ValueError(msg)
    elif diff_op_name not in DIFF_OPS:
        msg = f"Differential operator {diff_op_name} is not defined."
        raise ValueError(msg)
    else:
        autodiff_backend = _get_backend(backend)
        if autodiff_backend.available:
            diff_func = getattr(autodiff_backend.module, diff_op_name)(func)
            return wrap_function(diff_func, backend)
        else:
            msg = (
                f"{backend} selected as autodiff backend but is not available in "
                f"current environment therefore {name} must be provided directly."
            )
            raise ValueError(msg)
