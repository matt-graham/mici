"""SymNum differential operators and helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

SYMNUM_AVAILABLE = True
try:
    import symnum
except ImportError:
    SYMNUM_AVAILABLE = False

if TYPE_CHECKING:
    from mici.types import (
        ArrayFunction,
        ArrayLike,
        GradientFunction,
        JacobianFunction,
        MatrixHessianProductFunction,
        MatrixTressianProduct,
        ScalarFunction,
        ScalarLike,
        VectorJacobianProductFunction,
    )


def grad_and_value(func: ScalarFunction) -> GradientFunction:
    """Makes a function that returns both the Jacobian and value of a function."""
    return symnum.grad(func, return_aux=True)


def vjp_and_value(func: ArrayFunction) -> VectorJacobianProductFunction:
    """Makes a function that returns vector-Jacobian-product and value of a function.

    For a vector-valued function `fun` the vector-Jacobian-product (VJP) is here
    defined as a function of a vector `v` corresponding to

        vjp(v) = v @ j

    where `j` is the Jacobian of `f = fun(x)` wrt `x` i.e. the rank-2
    tensor of first-order partial derivatives of the vector-valued function,
    such that

        j[i, k] = ∂f[i] / ∂x[k]
    """
    return symnum.vector_jacobian_product(func, return_aux=True)


def jacobian_and_value(func: ArrayFunction) -> JacobianFunction:
    """Makes a function that returns both the Jacobian and value of a function."""
    return symnum.jacobian(func, return_aux=True)


def mhp_jacobian_and_value(func: ArrayFunction) -> MatrixHessianProductFunction:
    """Makes a function that returns MHP, Jacobian and value of a function.

    For a vector-valued function `fun` the matrix-Hessian-product (MHP) is here
    defined as a function of a matrix `m` corresponding to

        mhp(m) = sum(m[:, :, None] * h[:, :, :], axis=(0, 1))

    where `h` is the vector-Hessian of `f = fun(x)` wrt `x` i.e. the rank-3
    tensor of second-order partial derivatives of the vector-valued function,
    such that

        h[i, j, k] = ∂²f[i] / (∂x[j] ∂x[k])
    """
    return symnum.matrix_hessian_product(func, return_aux=True)


def hessian_grad_and_value(
    func: ScalarFunction,
) -> tuple[ArrayLike, ArrayLike, ScalarLike]:
    """Makes a function that returns the Hessian, gradient and value of a function."""
    return symnum.hessian(func, return_aux=True)


def mtp_hessian_grad_and_value(
    func: ScalarFunction,
) -> tuple[MatrixTressianProduct, ArrayLike, ArrayLike, ScalarLike]:
    """
    Makes a function that returns MTP, Jacobian and value of a function.

    For a scalar-valued function `fun` the matrix-Tressian-product (MTP) is
    here defined as a function of a matrix `m` corresponding to

        mtp(m) = sum(m[:, :] * t[:, :, :], axis=(-1, -2))

    where `t` is the 'Tressian' of `f = fun(x)` wrt `x` i.e. the 3D array of
    third-order partial derivatives of the scalar-valued function such that

        t[i, j, k] = ∂³f / (∂x[i] ∂x[j] ∂x[k])
    """
    return symnum.matrix_tressian_product(func, return_aux=True)
