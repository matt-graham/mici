"""Automatic differentation fallback for constructing derivative functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mici import autograd_wrapper

if TYPE_CHECKING:
    from typing import Callable, Optional


"""List of names of valid differential operators.

Any automatic differentiation framework wrapper module will need to provide all of these
operators as callables (with a single function as argument) to fully support all of the
required derivative functions.
"""
DIFF_OPS = [
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
]


def autodiff_fallback(
    diff_func: Optional[Callable],
    func: Callable,
    diff_op_name: str,
    name: str,
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

    Returns:
        `diff_func` value if not `None` otherwise generated derivative of `func` by
        applying named differential operator.
    """
    if diff_func is not None:
        return diff_func
    elif diff_op_name not in DIFF_OPS:
        msg = f"Differential operator {diff_op_name} is not defined."
        raise ValueError(msg)
    elif autograd_wrapper.AUTOGRAD_AVAILABLE:
        return getattr(autograd_wrapper, diff_op_name)(func)
    elif not autograd_wrapper.AUTOGRAD_AVAILABLE:
        msg = f"Autograd not available therefore {name} must be provided."
        raise ValueError(msg)
    return None
