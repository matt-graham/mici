import numpy as np
import pytest

from mici.autodiff import (
    _REGISTERED_BACKENDS,
    DIFF_OPS,
    _get_backend,
    autodiff_fallback,
    wrap_function,
)

N_POINTS_TO_TEST = 5

BACKENDS_AVAIALBLE = [
    name for name, backend in _REGISTERED_BACKENDS.items() if backend.available
]

SCALAR_FUNCTION_DIFF_OPS = [
    "grad_and_value",
    "hessian_grad_and_value",
    "mtp_hessian_grad_and_value",
]
VECTOR_FUNCTION_DIFF_OPS = ["jacobian_and_value", "mhp_jacobian_and_value"]


def torus_function_and_derivatives(numpy_module):
    toroidal_rad = 1.0
    poloidal_rad = 0.5

    def constr(q):
        x, y, z = q
        return numpy_module.array(
            [((x**2 + y**2) ** 0.5 - toroidal_rad) ** 2 + z**2 - poloidal_rad**2],
        )

    def jacob_constr(q):
        x, y, z = q
        r = (x**2 + y**2) ** 0.5
        return np.array(
            [[2 * x * (r - toroidal_rad) / r, 2 * y * (r - toroidal_rad) / r, 2 * z]],
        )

    def mhp_constr(q):
        x, y, z = q
        r = (x**2 + y**2) ** 0.5
        r_cubed = r**3
        return lambda m: np.array(
            [
                2 * (toroidal_rad / r_cubed) * (m[0, 0] * x**2 + m[0, 1] * x * y)
                + 2 * m[0, 0] * (1 - toroidal_rad / r),
                2 * (toroidal_rad / r_cubed) * (m[0, 1] * y**2 + m[0, 0] * x * y)
                + 2 * m[0, 1] * (1 - toroidal_rad / r),
                2 * m[0, 2],
            ],
        )

    return {
        "function": constr,
        "jacobian_function": jacob_constr,
        "mhp_function": mhp_constr,
    }


def linear_function_and_derivatives(_):

    constr_matrix = np.array([[1.0, -1.0, 2.0, 3.0], [-3.0, 2.0, 0.0, 5.0]])

    def constr(q):
        return constr_matrix @ q

    def jacob_constr(_):
        return constr_matrix

    def mhp_constr(_):
        return lambda _: np.zeros(constr_matrix.shape[1])

    return {
        "function": constr,
        "jacobian_function": jacob_constr,
        "mhp_function": mhp_constr,
    }


def quadratic_form_function_and_derivatives(_):

    matrix = np.array([[1.3, -0.2], [-0.2, 2.5]])

    def quadratic_form(q):
        return q @ matrix @ q / 2

    def grad_quadratic_form(q):
        return matrix @ q

    def hessian_quadratic_form(_):
        return matrix

    def mtp_quadratic_form(_):
        return lambda _: np.zeros(matrix.shape[0])

    return {
        "function": quadratic_form,
        "grad_function": grad_quadratic_form,
        "hessian_function": hessian_quadratic_form,
        "mtp_function": mtp_quadratic_form,
    }


def cubic_function_and_derivatives(_):

    def cubic(q):
        return (q**3).sum() / 6

    def grad_cubic(q):
        return q**2 / 2

    def hessian_cubic(q):
        return np.diag(q)

    def mtp_cubic(_):
        return lambda m: m.diagonal()

    return {
        "function": cubic,
        "grad_function": grad_cubic,
        "hessian_function": hessian_cubic,
        "mtp_function": mtp_cubic,
    }


def quartic_function_and_derivatives(_):

    def quartic(q):
        return (q**4).sum() / 24

    def grad_quartic(q):
        return q**3 / 6

    def hessian_quartic(q):
        return np.diag(q**2 / 2)

    def mtp_quartic(q):
        return lambda m: m.diagonal() * q

    return {
        "function": quartic,
        "grad_function": grad_quartic,
        "hessian_function": hessian_quartic,
        "mtp_function": mtp_quartic,
    }


def numpify_function(function_and_derivatives, *arg_shapes):
    import symnum

    function_and_derivatives["function"] = symnum.numpify_func(
        function_and_derivatives["function"],
        *arg_shapes,
    )


@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
def test_module_defines_all_diffops(backend_name):
    backend = _get_backend(backend_name)
    for diff_op_name in DIFF_OPS:
        assert hasattr(backend.module, diff_op_name)
        assert callable(getattr(backend.module, diff_op_name))


def get_numpy_module(backend):
    if backend in ("jax", "jax_nojit"):
        import jax.numpy

        return jax.numpy
    elif backend == "autograd":
        import autograd.numpy

        return autograd.numpy
    elif backend == "symnum":
        import symnum.numpy

        return symnum.numpy
    else:
        msg = f"Unrecognised backend {backend}"
        raise ValueError(msg)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.mark.parametrize("diff_op_name", DIFF_OPS)
def test_autodiff_fallback_with_no_backend_raises(diff_op_name):
    with pytest.raises(ValueError, match="None"):
        autodiff_fallback(
            None, lambda q: q, diff_op_name, diff_op_name + "_function", None,
        )


@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
def test_autodiff_fallback_with_invalid_diff_op_raises(backend_name):
    with pytest.raises(ValueError, match="not defined"):
        autodiff_fallback(None, lambda q: q, "foo", "bar", backend_name)


@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
def test_wrap_function(backend_name):

    def function(q):
        return q**2

    assert wrap_function(function, None) is function
    wrapped_function = wrap_function(function, backend_name)
    assert callable(wrapped_function)
    test_input = np.arange(5)
    output = wrapped_function(test_input)
    assert isinstance(output, np.ndarray)
    assert np.allclose(wrapped_function(test_input), function(test_input))


@pytest.mark.parametrize("diff_op_name", VECTOR_FUNCTION_DIFF_OPS)
@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    [(torus_function_and_derivatives, 1, 3), (linear_function_and_derivatives, 2, 4)],
    ids=lambda p: p[0].__name__,
)
def test_vector_function_diff_ops(
    diff_op_name, backend_name, function_and_derivatives_and_dim, rng,
):
    construct_function_and_derivatives, dim_c, dim_q = function_and_derivatives_and_dim
    numpy_module = get_numpy_module(backend_name)
    function_and_derivatives = construct_function_and_derivatives(numpy_module)
    if backend_name == "symnum":
        numpify_function(function_and_derivatives, dim_q)
    diff_op_function = autodiff_fallback(
        None,
        function_and_derivatives["function"],
        diff_op_name,
        diff_op_name + "_function",
        backend_name,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        derivatives_and_values = diff_op_function(q)
        # diff_op_function returns derivatives in descending order (of derivative) while
        # derivatives are in increasing order in function_and_derivatives
        for (function_name, expected_value_function), test_value in zip(
            function_and_derivatives.items(),
            reversed(derivatives_and_values),
            False,
        ):
            if function_name.startswith("mhp"):
                m = rng.standard_normal((dim_c, dim_q))
                assert np.allclose(test_value(m), expected_value_function(q)(m))
            else:
                assert np.allclose(test_value, expected_value_function(q))


@pytest.mark.parametrize("diff_op_name", SCALAR_FUNCTION_DIFF_OPS)
@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
@pytest.mark.parametrize(
    "function_and_derivatives_and_dim_q",
    [
        (quadratic_form_function_and_derivatives, 2),
        (cubic_function_and_derivatives, 1),
        (cubic_function_and_derivatives, 3),
        (quartic_function_and_derivatives, 2),
    ],
    ids=lambda p: p[0].__name__,
)
def test_scalar_function_diff_ops(
    diff_op_name, backend_name, function_and_derivatives_and_dim_q, rng,
):
    construct_function_and_derivatives, dim_q = function_and_derivatives_and_dim_q
    numpy_module = get_numpy_module(backend_name)
    function_and_derivatives = construct_function_and_derivatives(numpy_module)
    if backend_name == "symnum":
        numpify_function(function_and_derivatives, dim_q)
    diff_op_function = autodiff_fallback(
        None,
        function_and_derivatives["function"],
        diff_op_name,
        diff_op_name + "_function",
        backend_name,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        derivatives_and_values = diff_op_function(q)
        # diff_op_function returns derivatives in descending order (of derivative) while
        # derivatives are in increasing order in function_and_derivatives
        for (function_name, expected_value_function), test_value in zip(
            function_and_derivatives.items(),
            reversed(derivatives_and_values),
            False,
        ):
            if function_name.startswith("mtp"):
                m = rng.standard_normal((dim_q, dim_q))
                assert np.allclose(test_value(m), expected_value_function(q)(m))
            else:
                assert np.allclose(test_value, expected_value_function(q))


@pytest.mark.parametrize("backend_name", BACKENDS_AVAIALBLE)
@pytest.mark.parametrize(
    "function_and_derivatives_and_dim",
    [(torus_function_and_derivatives, 1, 3), (linear_function_and_derivatives, 2, 4)],
    ids=lambda p: p[0].__name__,
)
def test_vjp_and_value(backend_name, function_and_derivatives_and_dim, rng):
    construct_function_and_derivatives, dim_c, dim_q = function_and_derivatives_and_dim
    numpy_module = get_numpy_module(backend_name)
    function_and_derivatives = construct_function_and_derivatives(numpy_module)
    if backend_name == "symnum":
        numpify_function(function_and_derivatives, dim_q)
    vjp_and_value_function = autodiff_fallback(
        None,
        function_and_derivatives["function"],
        "vjp_and_value",
        "vjp_and_value_function",
        backend_name,
    )
    for _ in range(N_POINTS_TO_TEST):
        q = rng.standard_normal(dim_q)
        vjp, value = vjp_and_value_function(q)
        assert np.allclose(function_and_derivatives["function"](q), value)
        v = rng.standard_normal(value.shape)
        assert np.allclose(v @ function_and_derivatives["jacobian_function"](q), vjp(v))
