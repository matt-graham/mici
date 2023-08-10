"""Type aliases."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any, Callable, TypeVar, Union

from numpy import number
from numpy.typing import ArrayLike

from mici.matrices import Matrix, PositiveDefiniteMatrix
from mici.states import ChainState
from mici.systems import System
from mici.utils import LogRepFloat

ScalarLike = Union[bool, int, float, LogRepFloat, number]
"""Scalar like objects."""

MatrixLike = Union[ArrayLike, Matrix]
"""Matrix like objects."""

MetricLike = Union[ArrayLike, PositiveDefiniteMatrix]
"""Metric (positive definite matrix) like objects."""

TransitionStatistics = dict[str, ScalarLike]
"""Dictionary of statistics computed by :py:meth:`mici.transitions.Transition.sample`"""

AdaptationStatisticFunction = Callable[[TransitionStatistics], float]
"""Function returning adaptation statistic given dictionary of transition statistics."""

ReducerFunction = Callable[[Collection[float]], float]
"""Function combining per-chain control variables from adapters."""

AdapterState = dict[str, Any]
"""Dictionary defining current state of an :py:class:`mici.adapters.Adapter`."""

ScalarFunction = Callable[[ArrayLike], ScalarLike]
"""Function taking an array-like input and returning a scalar-like output."""

ArrayFunction = Callable[[ArrayLike], ArrayLike]
"""Function taking an array-like input and returning a array-like output."""

GradientFunction = Callable[[ArrayLike], Union[ArrayLike, tuple[ArrayLike, ScalarLike]]]
"""Function returning the gradient of a scalar-valued function.

May optionally also return scalar-like value of  function.
"""

HessianFunction = Callable[
    [ArrayLike],
    Union[ArrayLike, tuple[ArrayLike, ArrayLike, ScalarLike]],
]
"""Function returning the Hessian matrix of a scalar-valued function.

May optionally also return gradient and value of function.
"""

MatrixTressianProduct = Callable[[ArrayLike], ArrayLike]
"""Function returning the product between a matrix and 'Tressian' of a scalar function.

Tressian here is the 3-dimensional array of third-order derivatives of a scalar
function.
"""

MatrixTressianProductFunction = Callable[
    [ArrayLike],
    Union[
        MatrixTressianProduct,
        tuple[MatrixTressianProduct, ArrayLike, ArrayLike, ScalarLike],
    ],
]
"""Function returning a matrix-Tressian product function for a scalar function.

Tressian here is the 3-dimensional array of third-order derivatives of a scalar
function.

May optionally also return the Hessian matrix, gradient and value of function.
"""

JacobianFunction = Callable[[ArrayLike], Union[ArrayLike, tuple[ArrayLike, ArrayLike]]]
"""Function returning the Jacobian of a array (vector) valued function.

May optionally aslo return the array value of the function.
"""

MatrixHessianProduct = Callable[[ArrayLike], ArrayLike]
"""Function returning the product between a matrix and Hessian of an array function.

Hessian here is the 3-dimensional array of second-order derivatives of an array-valued
function.
"""

MatrixHessianProductFunction = Callable[
    [ArrayLike],
    Union[
        MatrixHessianProduct,
        tuple[MatrixHessianProduct, ArrayLike, ArrayLike],
    ],
]
"""Function returning a matrix-Hessian product function for an array function.

Hessian here is the 3-dimensional array of second-order derivatives of an array-valued
function.

May optionally also return the Jacobian matrix and array value of function.
"""

VectorJacobianProduct = Callable[[ArrayLike], ArrayLike]
"""Function returning the product between an array and Jacobian of a array function.

Jacobian here is the (d+1)-dimensional array of first-order derivatives of an
d-dimensional array-valued function taking a 1-dimensional array as input.
"""

VectorJacobianProductFunction = Callable[
    [ArrayLike],
    Union[VectorJacobianProduct, tuple[VectorJacobianProduct, ArrayLike]],
]
"""Function returning a vector-Jaccobian product function for an array function.

Jacobian here is the (d+1)-dimensional array of first-order derivatives of an
d-dimensional array-valued function taking a 1-dimensional array as input.

May optionally also return the array value of function.
"""

T = TypeVar("T")
PyTree = Union[dict[Any, "PyTree"], list["PyTree"], tuple["PyTree"], T]
"""Arbitrarily nested structure of Python dict, list and tuple types."""

TraceFunction = Callable[[ChainState], dict[str, ArrayLike]]
"""Function returning a dictionary of chain variables to trace given chain state."""

TerminationCriterion = Callable[[System, ChainState, ChainState, ArrayLike], bool]
"""Function indicating whether to terminate trajectory tree expansion."""

ChainIterator = Iterable[tuple[int, dict]]
