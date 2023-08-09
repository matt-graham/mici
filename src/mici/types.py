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
MatrixLike = Union[ArrayLike, Matrix]
MetricLike = Union[ArrayLike, PositiveDefiniteMatrix]

AdaptationStatisticFunction = Callable[[dict[str, float]], float]
ReducerFunction = Callable[[Collection[float]], float]

AdapterState = dict[str, Any]
TransitionStatistics = dict[str, ScalarLike]

ScalarFunction = Callable[[ArrayLike], ScalarLike]
ArrayFunction = Callable[[ArrayLike], ArrayLike]

GradientFunction = Callable[[ArrayLike], Union[ArrayLike, tuple[ArrayLike, ScalarLike]]]
HessianFunction = Callable[
    [ArrayLike],
    Union[ArrayLike, tuple[ArrayLike, ArrayLike, ScalarLike]],
]
MatrixTressianProduct = Callable[[ArrayLike], ArrayLike]
MatrixTressianProductFunction = Callable[
    [ArrayLike],
    Union[
        MatrixTressianProduct,
        tuple[MatrixTressianProduct, ArrayLike, ArrayLike, ScalarLike],
    ],
]
JacobianFunction = Callable[[ArrayLike], Union[ArrayLike, tuple[ArrayLike, ArrayLike]]]
MatrixHessianProduct = Callable[[ArrayLike], ArrayLike]
MatrixHessianProductFunction = Callable[
    [ArrayLike],
    Union[
        MatrixHessianProduct,
        tuple[MatrixHessianProduct, ArrayLike, ArrayLike],
    ],
]
VectorJacobianProduct = Callable[[ArrayLike], ArrayLike]
VectorJacobianProductFunction = Callable[
    [ArrayLike],
    Union[VectorJacobianProduct, tuple[VectorJacobianProduct, ArrayLike]],
]

T = TypeVar("T")
PyTree = Union[dict[Any, "PyTree"], list["PyTree"], tuple["PyTree"], T]

TraceFunction = Callable[[ChainState], dict[str, ArrayLike]]

TerminationCriterion = Callable[[System, ChainState, ChainState, ArrayLike], bool]

ChainIterator = Iterable[tuple[int, dict]]
