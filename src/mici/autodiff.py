"""Automatic differentation fallback for constructing derivative functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from mici import autograd_wrapper, jax_wrapper

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


class _AutodiffBackend(NamedTuple):

    wrapper_module: ModuleType
    postprocess_diff_function: Optional[Callable] = None


"""Available autodifferentiation framework backends."""
AVAILABLE_BACKENDS = {
    "autograd": _AutodiffBackend(autograd_wrapper),
    "jax": _AutodiffBackend(jax_wrapper, jax_wrapper.jit_and_return_numpy_arrays),
}


def autodiff_fallback(
    diff_func: Optional[Callable],
    func: Callable,
    diff_op_name: str,
    name: str,
    backend: str = "jax",
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
        backend: Name of automatic differentiation framework backend to use.

    Returns:
        `diff_func` value if not `None` otherwise generated derivative of `func` by
        applying named differential operator.
    """
    # Normalize backend string to all lowercase to make invariant to capitalization
    backend = backend.lower()
    if diff_func is not None:
        return diff_func
    elif diff_op_name not in DIFF_OPS:
        msg = f"Differential operator {diff_op_name} is not defined."
        raise ValueError(msg)
    elif backend not in AVAILABLE_BACKENDS:
        msg = (
            f"Selected autodiff backend {backend} not recognised: "
            f"available options are {AVAILABLE_BACKENDS}."
        )
        raise ValueError(msg)
    else:
        autodiff_backend = AVAILABLE_BACKENDS[backend]
        if getattr(autodiff_backend.wrapper_module, f"{backend.upper()}_AVAILABLE"):
            diff_func = getattr(autograd_wrapper, diff_op_name)(func)
            if autodiff_backend.postprocess_diff_function is not None:
                diff_func = autodiff_backend.postprocess_diff_function(diff_func)
            return diff_func
        else:
            msg = (
                f"{backend} selected as autodiff backend but is not available in "
                f"current environment therefore {name} must be provided directly."
            )
            raise ValueError(msg)
