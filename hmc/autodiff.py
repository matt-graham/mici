"""Automatic differentation fallback for construting derivative functions."""

AUTOGRAD_AVAILABLE = True
try:
    import hmc.autograd_wrapper as autograd_wrapper
except ImportError:
    AUTOGRAD_AVAILABLE = False


DIFF_OPS = [
    'vjp_and_value',
    'grad_and_value',
    'hessian_grad_and_value',
    'mtp_hessian_grad_and_value',
    'jacobian_and_value',
    'mhp_jacobian_and_value',
]


def autodiff_fallback(diff_func, func, diff_op_name, name):
    if diff_func is not None:
        return diff_func
    elif diff_op_name not in DIFF_OPS:
        raise ValueError(
            f'Differential operator {diff_op_name} is not defined.')
    elif AUTOGRAD_AVAILABLE:
        return getattr(autograd_wrapper, diff_op_name)(func)
    elif not AUTOGRAD_AVAILABLE:
        raise ValueError(
            f'Autograd not available therefore {name} must be provided.')
