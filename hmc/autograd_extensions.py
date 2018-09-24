"""Additional autograd differential operators."""

from autograd.wrap_util import unary_to_nary
from autograd.core import make_vjp as _make_vjp
from autograd.extend import vspace
import autograd.numpy as np


@unary_to_nary
def grad_and_value(fun, x):
    """Returns a function that returns both gradient and value. """
    vjp, ans = _make_vjp(fun, x)
    if not vspace(ans).size == 1:
        raise TypeError("grad_and_value only applies to real scalar-output "
                        "functions. Try jacobian, elementwise_grad or "
                        "holomorphic_grad.")
    return vjp(vspace(ans).ones()), ans

