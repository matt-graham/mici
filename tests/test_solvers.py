import pytest
import mici
import numpy as np
from collections import namedtuple


@pytest.fixture(params=("direct", "steffensen"))
def fixed_point_solver(request):
    if request.param == "direct":
        return mici.solvers.solve_fixed_point_direct
    elif request.param == "steffensen":
        return mici.solvers.solve_fixed_point_steffensen


FixedPointProblem = namedtuple(
    "FixedPointProblem", ("function", "fixed_point", "initial_point")
)


@pytest.fixture(params=("babylonian", "cosine", "ratio"))
def convergent_fixed_point_problem(request):
    if request.param == "babylonian":
        y = np.array([3.0, 5.0, 7.0])
        return FixedPointProblem(lambda x: (y / x + x) / 2, y ** 0.5, np.ones_like(y))
    elif request.param == "ratio":
        y = np.array([3.0, 5.0, 7.0])
        return FixedPointProblem(lambda x: (x + y) / (x + 1), y ** 0.5, np.ones_like(y))
    elif request.param == "cosine":
        return FixedPointProblem(
            lambda x: np.cos(x), np.array([0.7390851332151607]), np.array([1.0])
        )


@pytest.fixture(params=("doubling", "quadratic"))
def divergent_fixed_point_problem(request):
    if request.param == "doubling":
        return FixedPointProblem(lambda x: 2 * x, None, np.arange(3))
    elif request.param == "quadratic":
        return FixedPointProblem(
            lambda x: 1 + x**2, None, np.arange(3)
        )


@pytest.fixture(params=(1e-6, 1e-8, 1e-10))
def convergence_tol(request):
    return request.param


@pytest.fixture(params=("maximum", "euclidean"))
def norm(request):
    if request.param == "maximum":
        return mici.solvers.maximum_norm
    elif request.param == "euclidean":
        return mici.solvers.euclidean_norm


@pytest.mark.parametrize("convergence_tol", (1e-6, 1e-8, 1e-10))
def test_fixed_point_solver_convergence(
    fixed_point_solver, convergent_fixed_point_problem, convergence_tol, norm
):
    fixed_point = fixed_point_solver(
        func=convergent_fixed_point_problem.function,
        x0=convergent_fixed_point_problem.initial_point,
        convergence_tol=convergence_tol,
        norm=norm,
    )
    assert (
        norm(fixed_point - convergent_fixed_point_problem.fixed_point) < convergence_tol
    )


def test_fixed_point_solver_divergence(
    fixed_point_solver, divergent_fixed_point_problem
):
    with pytest.raises(mici.errors.ConvergenceError):
        fixed_point_solver(
            func=divergent_fixed_point_problem.function,
            x0=divergent_fixed_point_problem.initial_point,
            max_iters=10000,
        )


def test_fixed_point_solver_max_iters_exceeded_convergence_error(
    fixed_point_solver, convergent_fixed_point_problem
):
    with pytest.raises(mici.errors.ConvergenceError):
        fixed_point_solver(
            func=convergent_fixed_point_problem.function,
            x0=convergent_fixed_point_problem.initial_point,
            convergence_tol=1e-10,
            max_iters=1,
        )


def test_fixed_point_solver_handle_value_error(fixed_point_solver):
    def func(x):
        raise ValueError()

    with pytest.raises(mici.errors.ConvergenceError):
        fixed_point_solver(func=func, x0=np.array([1.0]))


def test_fixed_point_solver_handle_linalg_error(fixed_point_solver):
    def func(x):
        raise mici.errors.LinAlgError()

    with pytest.raises(mici.errors.ConvergenceError):
        fixed_point_solver(func=func, x0=np.array([1.0]))

