"""Additional JAX differential operators."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

JAX_AVAILABLE = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    JAX_AVAILABLE = False

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
    )


def jit_and_return_numpy_arrays(func, **jit_kwargs):

    jitted_function = jax.jit(func, **jit_kwargs)

    def jitted_func_returning_numpy_arrays(*args, **kwargs):
        return_value = jitted_function(*args, **kwargs)
        if isinstance(return_value, tuple):
            return tuple(
                np.asarray(val) if isinstance(val, jax.Array) else val
                for val in return_value
            )
        else:
            return (
                np.asarray(return_value)
                if isinstance(return_value, jax.Array)
                else return_value
            )

    return jitted_func_returning_numpy_arrays


def grad_and_value(func: ScalarFunction) -> GradientFunction:
    """Makes a function that returns both the Jacobian and value of a function."""

    def grad_and_value_func(x):
        value, grad = jax.value_and_grad(func)(x)
        return grad, value

    return grad_and_value_func


def jacobian_and_value(func: ArrayFunction) -> JacobianFunction:
    """Makes a function that returns both the Jacobian and value of a function."""

    def value_and_jacobian_func(x):
        value, pullback = jax.vjp(func, x)
        basis = jnp.eye(value.size, dtype=value.dtype)
        (jac,) = jax.vmap(pullback)(basis)
        return jac, value

    return value_and_jacobian_func


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

    def _mhp_jacobian_and_value_func(x):
        jac, mhp, value = jax.vjp(jacobian_and_value(func), x, has_aux=True)
        return mhp, jac, value

    @jax.jit
    def mhp_jacobian_and_value_func(x):
        mhp, jac, value = _mhp_jacobian_and_value_func(x)
        return lambda m: np.asarray(mhp(m)[0]), jac, value

    return mhp_jacobian_and_value_func


def hessian_grad_and_value(
    func: ScalarFunction,
) -> tuple[ArrayLike, ArrayLike, ScalarLike]:
    """Makes a function that returns the Hessian, gradient and value of a function."""

    def hessian_grad_and_value_func(x):
        basis = jnp.eye(x.size, dtype=x.dtype)
        grad_and_value_func = grad_and_value(func)
        pushforward = partial(jax.jvp, grad_and_value_func, (x,), has_aux=True)
        grad, hessian, value = jax.vmap(pushforward, out_axes=(None, -1, None))(
            (basis,)
        )
        return hessian, grad, value

    return hessian_grad_and_value_func


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

    def hessian_and_aux_func(x):
        hessian, grad, value = hessian_grad_and_value(func)(x)
        return hessian, (grad, value)

    @jax.jit
    def _mtp_hessian_grad_and_value_func(x):
        hessian, mtp, (grad, value) = jax.vjp(hessian_and_aux_func, x, has_aux=True)
        return mtp, hessian, grad, value

    def mtp_hessian_grad_and_value_func(x):
        mtp, hessian, grad, value = _mtp_hessian_grad_and_value_func(x)
        return lambda m: np.asarray(mtp(m)[0]), hessian, grad, value

    return mtp_hessian_grad_and_value_func
