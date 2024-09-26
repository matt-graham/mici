"""Autograd differential operators."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

AUTOGRAD_AVAILABLE = True
try:
    import autograd.numpy as np
    from autograd.builtins import tuple as atuple
    from autograd.core import make_vjp
    from autograd.extend import vspace
    from autograd.wrap_util import unary_to_nary
except ImportError:
    AUTOGRAD_AVAILABLE = False

if TYPE_CHECKING:
    from typing import Callable

    from mici.types import (
        ArrayFunction,
        ArrayLike,
        MatrixHessianProduct,
        MatrixTressianProduct,
        ScalarFunction,
        ScalarLike,
        VectorJacobianProduct,
    )


def _wrapped_unary_to_nary(func: Callable) -> Callable:
    """Use functools.wraps with unary_to_nary decorator."""
    if AUTOGRAD_AVAILABLE:
        return wraps(func)(unary_to_nary(func))
    else:
        return func


@_wrapped_unary_to_nary
def grad_and_value(fun: ScalarFunction, x: ArrayLike) -> tuple[ArrayLike, ScalarLike]:
    """Makes a function that returns both gradient and value of a function."""
    vjp, val = make_vjp(fun, x)
    if vspace(val).size != 1:
        msg = "grad_and_value only applies to real scalar-output functions."
        raise TypeError(msg)
    return vjp(vspace(val).ones()), val


@_wrapped_unary_to_nary
def vjp_and_value(
    fun: ScalarFunction, x: ArrayLike,
) -> tuple[VectorJacobianProduct, ArrayLike]:
    """
    Makes a function that returns vector-Jacobian-product and value of a function.

    For a vector-valued function `fun` the vector-Jacobian-product (VJP) is here
    defined as a function of a vector `v` corresponding to

        vjp(v) = v @ j

    where `j` is the Jacobian of `f = fun(x)` wrt `x` i.e. the rank-2
    tensor of first-order partial derivatives of the vector-valued function,
    such that

        j[i, k] = ∂f[i] / ∂x[k]
    """
    return make_vjp(fun, x)


@_wrapped_unary_to_nary
def jacobian_and_value(fun: ArrayFunction, x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Makes a function that returns both the Jacobian and value of a function.

    Assumes that the function `fun` broadcasts along the first dimension of the
    input being differentiated with respect to such that a batch of outputs can
    be computed concurrently for a batch of inputs.
    """
    val = fun(x)
    v_vspace = vspace(val)
    x_vspace = vspace(x)
    x_rep = np.tile(x, (v_vspace.size,) + (1,) * x_vspace.ndim)
    vjp_rep, _ = make_vjp(fun, x_rep)
    jacobian_shape = v_vspace.shape + x_vspace.shape
    basis_vectors = np.array(list(v_vspace.standard_basis()))
    jacobian = vjp_rep(basis_vectors)
    return np.reshape(jacobian, jacobian_shape), val


@_wrapped_unary_to_nary
def mhp_jacobian_and_value(
    fun: ArrayFunction,
    x: ArrayLike,
) -> tuple[MatrixHessianProduct, ArrayLike, ArrayLike]:
    """
    Makes a function that returns MHP, Jacobian and value of a function.

    For a vector-valued function `fun` the matrix-Hessian-product (MHP) is here
    defined as a function of a matrix `m` corresponding to

        mhp(m) = sum(m[:, :, None] * h[:, :, :], axis=(0, 1))

    where `h` is the vector-Hessian of `f = fun(x)` wrt `x` i.e. the rank-3
    tensor of second-order partial derivatives of the vector-valued function,
    such that

        h[i, j, k] = ∂²f[i] / (∂x[j] ∂x[k])

    Assumes that the function `fun` broadcasts along the first dimension of the
    input being differentiated with respect to such that a batch of outputs can
    be computed concurrently for a batch of inputs.
    """
    mhp, (jacob, val) = make_vjp(lambda x: atuple(jacobian_and_value(fun)(x)), x)
    return lambda m: mhp((m, vspace(val).zeros())), jacob, val


@_wrapped_unary_to_nary
def hessian_grad_and_value(
    fun: ArrayFunction,
    x: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ScalarLike]:
    """
    Makes a function that returns the Hessian, gradient & value of a function.

    Assumes that the function `fun` broadcasts along the first dimension of the
    input being differentiated with respect to such that a batch of outputs can
    be computed concurrently for a batch of inputs.
    """

    def grad_fun(x):
        vjp, val = make_vjp(fun, x)
        return vjp(vspace(val).ones()), val

    x_vspace = vspace(x)
    x_rep = np.tile(x, (x_vspace.size,) + (1,) * x_vspace.ndim)
    vjp_grad, (grad, val) = make_vjp(lambda x: atuple(grad_fun(x)), x_rep)
    hessian_shape = x_vspace.shape + x_vspace.shape
    basis_vectors = np.array(list(x_vspace.standard_basis()))
    hessian = vjp_grad((basis_vectors, vspace(val).zeros()))
    return np.reshape(hessian, hessian_shape), grad[0], val[0]


@_wrapped_unary_to_nary
def mtp_hessian_grad_and_value(
    fun: ArrayFunction,
    x: ArrayLike,
) -> tuple[MatrixTressianProduct, ArrayLike, ArrayLike, ScalarLike]:
    """
    Makes a function that returns MTP, Jacobian and value of a function.

    For a scalar-valued function `fun` the matrix-Tressian-product (MTP) is
    here defined as a function of a matrix `m` corresponding to

        mtp(m) = sum(m[:, :] * t[:, :, :], axis=(-1, -2))

    where `t` is the 'Tressian' of `f = fun(x)` wrt `x` i.e. the 3D array of
    third-order partial derivatives of the scalar-valued function such that

        t[i, j, k] = ∂³f / (∂x[i] ∂x[j] ∂x[k])

    Assumes that the function `fun` broadcasts along the first dimension of the
    input being differentiated with respect to such that a batch of outputs can
    be computed concurrently for a batch of inputs.
    """
    mtp, (hessian, grad, val) = make_vjp(
        lambda x: atuple(hessian_grad_and_value(fun)(x)),
        x,
    )
    return (
        lambda m: mtp((m, vspace(grad).zeros(), vspace(val).zeros())),
        hessian,
        grad,
        val,
    )
