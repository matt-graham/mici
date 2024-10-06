"""Autograd differential operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

AUTOGRAD_AVAILABLE = True
try:
    import autograd.numpy as np
    from autograd.builtins import tuple as atuple
    from autograd.core import make_vjp
    from autograd.extend import vspace
except ImportError:
    AUTOGRAD_AVAILABLE = False

if TYPE_CHECKING:

    from mici.types import (
        ArrayFunction,
        GradientFunction,
        HessianFunction,
        JacobianFunction,
        MatrixHessianProductFunction,
        MatrixTressianProductFunction,
        ScalarFunction,
        VectorJacobianProductFunction,
    )


def grad_and_value(func: ScalarFunction) -> GradientFunction:
    """Makes a function that returns both gradient and value of a function."""

    def grad_and_value_func(x):
        vjp, val = make_vjp(func, x)
        if vspace(val).size != 1:
            msg = "grad_and_value only applies to real scalar-output functions."
            raise TypeError(msg)
        return vjp(vspace(val).ones()), val

    return grad_and_value_func


def vjp_and_value(func: ScalarFunction) -> VectorJacobianProductFunction:
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

    def vjp_and_value_func(x):
        return make_vjp(func, x)

    return vjp_and_value_func


def jacobian_and_value(func: ArrayFunction) -> JacobianFunction:
    """Makes a function that returns both the Jacobian and value of a function."""

    def jacobian_and_value_func(x):
        vjp, val = make_vjp(func, x)
        val_vspace = vspace(val)
        jacobian_shape = val_vspace.shape + vspace(x).shape
        jacobian_rows = map(vjp, val_vspace.standard_basis())
        return np.reshape(np.stack(jacobian_rows), jacobian_shape), val

    return jacobian_and_value_func


def mhp_jacobian_and_value(func: ArrayFunction) -> MatrixHessianProductFunction:
    """
    Makes a function that returns MHP, Jacobian and value of a function.

    For a vector-valued function `fun` the matrix-Hessian-product (MHP) is here
    defined as a function of a matrix `m` corresponding to

        mhp(m) = sum(m[:, :, None] * h[:, :, :], axis=(0, 1))

    where `h` is the vector-Hessian of `f = fun(x)` wrt `x` i.e. the rank-3
    tensor of second-order partial derivatives of the vector-valued function,
    such that

        h[i, j, k] = ∂²f[i] / (∂x[j] ∂x[k])
    """

    def mhp_jacobian_and_value_func(x):
        mhp, (jacob, val) = make_vjp(lambda x: atuple(jacobian_and_value(func)(x)), x)
        return lambda m: mhp((m, vspace(val).zeros())), jacob, val

    return mhp_jacobian_and_value_func


def hessian_grad_and_value(func: ArrayFunction) -> HessianFunction:
    """Makes a function that returns the Hessian, gradient & value of a function."""

    def grad_func(x):
        vjp, val = make_vjp(func, x)
        return vjp(vspace(val).ones()), val

    def hessian_grad_and_value_func(x):
        x_vspace = vspace(x)
        vjp_grad, (grad, val) = make_vjp(lambda x: atuple(grad_func(x)), x)
        hessian_shape = x_vspace.shape + x_vspace.shape
        zeros = vspace(val).zeros()
        hessian_rows = (vjp_grad((v, zeros)) for v in x_vspace.standard_basis())
        return np.reshape(np.stack(hessian_rows), hessian_shape), grad, val

    return hessian_grad_and_value_func


def mtp_hessian_grad_and_value(func: ArrayFunction) -> MatrixTressianProductFunction:
    """
    Makes a function that returns MTP, Jacobian and value of a function.

    For a scalar-valued function `fun` the matrix-Tressian-product (MTP) is
    here defined as a function of a matrix `m` corresponding to

        mtp(m) = sum(m[:, :] * t[:, :, :], axis=(-1, -2))

    where `t` is the 'Tressian' of `f = fun(x)` wrt `x` i.e. the 3D array of
    third-order partial derivatives of the scalar-valued function such that

        t[i, j, k] = ∂³f / (∂x[i] ∂x[j] ∂x[k])
    """

    def mtp_hessian_grad_and_value_func(x):
        mtp, (hessian, grad, val) = make_vjp(
            lambda x: atuple(hessian_grad_and_value(func)(x)),
            x,
        )
        return (
            lambda m: mtp((m, vspace(grad).zeros(), vspace(val).zeros())),
            hessian,
            grad,
            val,
        )

    return mtp_hessian_grad_and_value_func
