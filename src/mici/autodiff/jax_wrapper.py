"""JAX differential operators and helper functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, ParamSpec, TypeAlias, TypeVar

import numpy as np

JAX_AVAILABLE = True
try:
    import jax
    import jax.numpy as jnp
    from jax import Array as JaxArray
except ImportError:
    JAX_AVAILABLE = False
    JaxArray = TypeVar("JaxArray")

if TYPE_CHECKING:
    from mici.types import (
        ArrayFunction,
        ArrayLike,
        GradientFunction,
        HessianFunction,
        JacobianFunction,
        MatrixHessianProduct,
        MatrixHessianProductFunction,
        MatrixTressianProduct,
        MatrixTressianProductFunction,
        ScalarFunction,
        ScalarLike,
        VectorJacobianProduct,
        VectorJacobianProductFunction,
    )


P = ParamSpec("P")
JaxArrayFunction: TypeAlias = Callable[
    P, "JaxArray | JaxArrayFunction | tuple[JaxArray | JaxArrayFunction, ...]"
]
NumPyArrayFunction: TypeAlias = Callable[
    P, "np.ndarray | NumPyArrayFunction | tuple[np.ndarray | NumPyArrayFunction, ...]"
]


def jit_and_return_numpy_arrays(function: JaxArrayFunction) -> NumPyArrayFunction:
    """Wrap a JIT compiled function returning JAX arrays to instead return NumPy arrays.

    Args:
        function: Function to wrap. Should return one of: a single JAX array, a callable
            returning one or more JAX array or a tuple of one or more JAX arrays or
            functions returning one or more JAX arrays.
        **jit_kwargs: Any keyword arguments to pass to `jax.jit` operator.

    Returns:
        Wrapped function. Any values returned by original function which are JAX arrays
        will instead be NumPy arrays, while any values which are callables returning
        JAX arrays will instead return NumPy arrays.
    """
    jitted_function = jax.jit(function)
    return return_numpy_arrays(jitted_function)


def return_numpy_arrays(function: JaxArrayFunction) -> NumPyArrayFunction:
    """Wrap a function returning JAX arrays to instead return NumPy arrays.

    Args:
        function: Function to wrap. Should return one of: a single JAX array, a callable
            returning one or more JAX array or a tuple of one or more JAX arrays or
            functions returning one or more JAX arrays.

    Returns:
        Wrapped function. Any values returned by original function which are JAX arrays
        will instead be NumPy arrays, while any values which are callables returning
        JAX arrays will instead return NumPy arrays.
    """

    def as_numpy_array(
        value: JaxArray | JaxArrayFunction,
    ) -> np.ndarray | NumPyArrayFunction:
        if callable(value):
            return return_numpy_arrays(value)
        if isinstance(value, jax.Array):
            return np.asarray(value)
        return value

    def function_returning_numpy_arrays(
        *args: P.args, **kwargs: P.kwargs
    ) -> np.ndarray | NumPyArrayFunction | tuple[np.ndarray | NumPyArrayFunction, ...]:
        return_value = function(*args, **kwargs)
        if isinstance(return_value, tuple):
            return tuple(as_numpy_array(value) for value in return_value)
        return as_numpy_array(return_value)

    return function_returning_numpy_arrays


def grad_and_value(func: ScalarFunction) -> GradientFunction:
    """Makes a function that returns both the Jacobian and value of a function."""

    def grad_and_value_func(x: ArrayLike) -> tuple[ArrayLike, ScalarLike]:
        value, grad = jax.value_and_grad(func)(x)
        return grad, value

    return grad_and_value_func


def _detuple_vjp(vjp_func: jax.tree_util.Partial) -> jax.tree_util.Partial:
    """Transform a VJP of function with one return value so it returns an array."""
    return jax.tree_util.Partial(
        lambda *args, **kwargs: vjp_func.func(*args, **kwargs)[0],
        *vjp_func.args,
        **vjp_func.keywords,
    )


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

    def vjp_and_value_func(x: ArrayLike) -> tuple[VectorJacobianProduct, ArrayLike]:
        value, vjp = jax.vjp(func, x)
        return _detuple_vjp(vjp), value

    return vjp_and_value_func


def jacobian_and_value(func: ArrayFunction) -> JacobianFunction:
    """Makes a function that returns both the Jacobian and value of a function."""

    def jacobian_and_value_func(x: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        value, pullback = jax.vjp(func, x)
        basis = jnp.eye(value.size, dtype=value.dtype)
        (jac,) = jax.vmap(pullback)(basis)
        return jac, value

    return jacobian_and_value_func


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

    def mhp_jacobian_and_value_func(
        x: ArrayLike,
    ) -> tuple[MatrixHessianProduct, ArrayLike, ArrayLike]:
        jac, mhp, value = jax.vjp(jacobian_and_value(func), x, has_aux=True)
        return _detuple_vjp(mhp), jac, value

    return mhp_jacobian_and_value_func


def hessian_grad_and_value(func: ScalarFunction) -> HessianFunction:
    """Makes a function that returns the Hessian, gradient and value of a function."""

    def hessian_grad_and_value_func(
        x: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike, ScalarLike]:
        basis = jnp.eye(x.size, dtype=x.dtype)
        grad_and_value_func = grad_and_value(func)
        pushforward = partial(jax.jvp, grad_and_value_func, (x,), has_aux=True)
        grad, hessian, value = jax.vmap(pushforward, out_axes=(None, -1, None))(
            (basis,),
        )
        return hessian, grad, value

    return hessian_grad_and_value_func


def mtp_hessian_grad_and_value(func: ScalarFunction) -> MatrixTressianProductFunction:
    """Makes a function that returns MTP, Jacobian and value of a function.

    For a scalar-valued function `fun` the matrix-Tressian-product (MTP) is
    here defined as a function of a matrix `m` corresponding to

        mtp(m) = sum(m[:, :] * t[:, :, :], axis=(-1, -2))

    where `t` is the 'Tressian' of `f = fun(x)` wrt `x` i.e. the 3D array of
    third-order partial derivatives of the scalar-valued function such that

        t[i, j, k] = ∂³f / (∂x[i] ∂x[j] ∂x[k])
    """

    def hessian_and_aux_func(
        x: ArrayLike,
    ) -> tuple[ArrayLike, tuple[ArrayLike, ScalarLike]]:
        hessian, grad, value = hessian_grad_and_value(func)(x)
        return hessian, (grad, value)

    def mtp_hessian_grad_and_value_func(
        x: ArrayLike,
    ) -> tuple[MatrixTressianProduct, ArrayLike, ArrayLike, ScalarLike]:
        hessian, mtp, (grad, value) = jax.vjp(hessian_and_aux_func, x, has_aux=True)
        return _detuple_vjp(mtp), hessian, grad, value

    return mtp_hessian_grad_and_value_func
