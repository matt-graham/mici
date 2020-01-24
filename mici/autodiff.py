"""Automatic differentation fallback for constructing derivative functions."""

import mici.autograd_wrapper as autograd_wrapper


"""List of names of valid differential operators.

Any automatic differentiation framework wrapper module will need to provide
all of these operators as callables (with a single function as argument) to
fully support all of the required derivative functions.
"""
DIFF_OPS = [
    # vector Jacobian product and value
    'vjp_and_value',
    # gradient and value for scalar valued functions
    'grad_and_value',
    # Hessian matrix, gradient and value for scalar valued functions
    'hessian_grad_and_value',
    # matrix Tressian product, gradient and value for scalar valued
    # functions
    'mtp_hessian_grad_and_value',
    # Jacobian matrix and value for vector valued functions
    'jacobian_and_value',
    # matrix Hessian product, Jacobian matrix and value for vector valued
    # functions
    'mhp_jacobian_and_value',
]


def autodiff_fallback(diff_func, func, diff_op_name, name):
    """Generate derivative function automatically if not provided.

    Uses automatic differentiation to generate a function corresponding to a
    differential operator applied to a function if an alternative
    implementation of the derivative function has not been provided.

    Args:
        diff_func (None or Callable): Either a callable implementing the
            required derivative function or `None` if none was provided.
        func (Callable): Function to differentiate.
        diff_op_name (str): String specifying name of differential operator
            from automatic differentiation framework wrapper to use to generate
            required derivative function.
        name (str): Name of derivative function to use in error message.

    Returns:
        Callable: `diff_func` value if not `None` otherwise generated
            derivative of `func` by applying named differential operator.
    """
    if diff_func is not None:
        return diff_func
    elif diff_op_name not in DIFF_OPS:
        raise ValueError(
            f'Differential operator {diff_op_name} is not defined.')
    elif autograd_wrapper.AUTOGRAD_AVAILABLE:
        return getattr(autograd_wrapper, diff_op_name)(func)
    elif not autograd_wrapper.AUTOGRAD_AVAILABLE:
        raise ValueError(
            f'Autograd not available therefore {name} must be provided.')
