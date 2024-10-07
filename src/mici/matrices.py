"""Structured matrix classes implementing basic linear algebra operations."""

from __future__ import annotations

import abc
import numbers
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from mici.errors import LinAlgError
from mici.utils import hash_array

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal

    from numpy.typing import NDArray
    from typing_extensions import Self

    from mici.types import MatrixLike, ScalarLike


def _choose_matrix_product_class(matrix_l: Matrix, matrix_r: Matrix) -> MatrixProduct:
    if matrix_l.shape[0] == matrix_l.shape[1] and matrix_r.shape == matrix_l.shape:
        if isinstance(matrix_l, InvertibleMatrix) and isinstance(
            matrix_r,
            InvertibleMatrix,
        ):
            return InvertibleMatrixProduct
        return SquareMatrixProduct
    return MatrixProduct


def _is_scalar(val: Any) -> bool:  # noqa: ANN401
    return isinstance(val, numbers.Number) or (
        hasattr(val, "__array__") and np.ndim(val) == 0
    )


class Matrix(abc.ABC):
    """Base class for matrix-like objects.

    Implements overloads of the matrix multiplication operator `@`, as well as
    the standard multiplication and division operators `*` and `/` when the
    second argument is a scalar quantity.
    """

    __array_priority__ = 1

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        """
        Args:
            shape: Shape of matrix `(num_rows, num_columns)`.
        """
        self._shape = shape
        self._hash = None
        self._transpose = None
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
            self.__dict__[k] = v

    def __array__(self) -> NDArray:
        return self.array

    def __mul__(self, other: ScalarLike) -> Matrix:
        if _is_scalar(other):
            if other == 0:
                msg = "Scalar multiplication by zero not implemented."
                raise NotImplementedError(msg)
            return self._scalar_multiply(other)
        return NotImplemented

    def __rmul__(self, other: ScalarLike) -> Matrix:
        return self.__mul__(other)

    def __truediv__(self, other: ScalarLike) -> Matrix:
        if _is_scalar(other):
            if other == 0:
                msg = "Scalar division by zero not implemented."
                raise NotImplementedError(msg)
            return self._scalar_multiply(1 / other)
        return NotImplemented

    def __neg__(self) -> Matrix:
        return self._scalar_multiply(-1)

    def __matmul__(self, other: MatrixLike) -> Matrix | NDArray:
        if self.shape[1] is not None and other.shape[0] != self.shape[1]:
            msg = (
                f"Inconsistent dimensions for matrix multiplication: {self.shape} and "
                f"{other.shape}."
            )
            raise ValueError(msg)
        if isinstance(other, Matrix):
            matrix_product_class = _choose_matrix_product_class(self, other)
            return matrix_product_class((self, other), check_shapes=False)
        return self._left_matrix_multiply(other)

    def __rmatmul__(self, other: MatrixLike) -> Matrix | NDArray:
        if self.shape[0] is not None and other.shape[-1] != self.shape[0]:
            msg = (
                f"Inconsistent dimensions for matrix multiplication: {other.shape} and "
                f"{self.shape}."
            )
            raise ValueError(msg)
        if isinstance(other, Matrix):
            matrix_product_class = _choose_matrix_product_class(self, other)
            return matrix_product_class((other, self), check_shapes=False)
        return self._right_matrix_multiply(other)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of matrix as a tuple `(num_rows, num_columns)`."""
        return self._shape

    @property
    @abc.abstractmethod
    def array(self) -> NDArray:
        """Full dense representation of matrix as a 2D array."""

    @abc.abstractmethod
    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        """Left multiply argument by the represented matrix.

        Args:
            other (array): Argument to left-multiply.

        Returns:
            result (array): Result of left-multiplying `other` by the
                represented matrix.
        """

    @abc.abstractmethod
    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        """Right multiply argument by the represented matrix.

        Args:
            other: Argument to right-multiply.

        Returns:
            result: Result of right-multiplying `other` by the represented matrix.
        """

    @abc.abstractmethod
    def _scalar_multiply(self, scalar: ScalarLike) -> Matrix:
        """Calculate result of multiplying represented matrix by a scalar.

        Args:
            scalar: Scalar argument to multiply by.

        Returns:
            Result of multiplying represented matrix by `scalar` as another `Matrix`
            object.
        """

    @property
    def transpose(self) -> Matrix:
        """Transpose of matrix."""
        if self._transpose is None:
            self._transpose = self._construct_transpose()
        return self._transpose

    T = transpose

    @abc.abstractmethod
    def _construct_transpose(self) -> Matrix:
        """Construct transpose of matrix."""

    @property
    def diagonal(self) -> NDArray:
        """Diagonal of matrix as a 1D array."""
        return self.array.diagonal()

    def __str__(self) -> str:
        return f"(shape={self.shape})"

    def __repr__(self) -> str:
        return type(self).__name__ + str(self)

    @abc.abstractmethod
    def _compute_hash(self) -> int:
        """Compute hash value for matrix object."""

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = self._compute_hash()
        return self._hash

    @abc.abstractmethod
    def _check_equality(self, other: Matrix) -> bool:
        """Check for equality with another instance of the same class."""

    def __eq__(self, other: Matrix) -> bool:
        return other is self or (
            other.__class__ == self.__class__ and self._check_equality(other)
        )


class ExplicitArrayMatrix(Matrix):
    """Matrix with an explicit array representation."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        if "_array" not in kwargs:
            msg = "_array must be specified in kwargs"
            raise ValueError(msg)
        try:
            kwargs["_array"] = np.asarray_chkfinite(kwargs["_array"])
        except ValueError as e:
            msg = "Array is not finite."
            raise LinAlgError(msg) from e
        super().__init__(shape, **kwargs)

    @property
    def array(self) -> NDArray:
        return self._array

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self._array @ other

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return other @ self._array

    def _compute_hash(self) -> int:
        return hash_array(self._array)

    def _check_equality(self, other: Matrix) -> bool:
        return np.array_equal(self.array, other.array)


class ImplicitArrayMatrix(Matrix):
    """Matrix with an implicit array representation."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        """
        Args:
        shape: Shape of matrix `(num_rows, num_columns)`.
        """
        super().__init__(shape, **kwargs)
        self._array = None

    @property
    def array(self) -> NDArray:
        """Full dense representation of matrix as a 2D array.

        Generally accessing this property should be avoided wherever possible
        as the resulting array object may use a lot of memory and operations
        with it will not be able to exploit any structure in the matrix.
        """
        if self._array is None:
            self._array = self._construct_array()
        return self._array

    @abc.abstractmethod
    def _construct_array(self) -> NDArray:
        """Construct full dense representation of matrix as a 2D array.

        Generally calling this method should be avoided wherever possible as
        the returned array object may use a lot of memory and operations with
        it will not be able to exploit any structure in the matrix.
        """


class MatrixProduct(ImplicitArrayMatrix):
    """Matrix implicitly defined as a product of a sequence of matrices.

    Each adjacent pair of matrices in the sequence must have compatible shapes.
    """

    def __init__(
        self, matrices: Iterable[Matrix], *, check_shapes: bool = True
    ) -> None:
        """
        Args:
            matrices: Sequence of matrices forming product in left-to-right order.
            check_shapes: Whether to check if all successive pairs of he matrix sequence
                have compatible shapes, i.e. equal inner dimensions.
        """
        self._matrices = tuple(matrices)
        if check_shapes:
            for matrix_l, matrix_r in pairwise(matrices):
                if matrix_l.shape[1] != matrix_r.shape[0]:
                    msg = (
                        f"Matrices {matrix_l} and {matrix_r} have inconsistent inner "
                        f"dimensions for forming a matrix product."
                    )
                    raise ValueError(msg)
        super().__init__((self._matrices[0].shape[0], self._matrices[-1].shape[1]))

    @property
    def matrices(self) -> tuple[Matrix]:
        return self._matrices

    def _scalar_multiply(self, scalar: ScalarLike) -> MatrixProduct:
        return type(self)((ScaledIdentityMatrix(scalar, self.shape[0]), *self.matrices))

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        for matrix in reversed(self.matrices):
            other = matrix @ other
        return other

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        for matrix in self.matrices:
            other = other @ matrix
        return other

    def _construct_transpose(self) -> MatrixProduct:
        return type(self)(tuple(matrix.T for matrix in reversed(self.matrices)))

    def _construct_array(self) -> NDArray:
        return self.matrices[0].array @ MatrixProduct(self.matrices[1:])

    def _compute_hash(self) -> int:
        return hash(tuple(matrix for matrix in self.matrices))

    def _check_equality(self, other: MatrixProduct) -> bool:
        return len(other.matrices) == len(self.matrices) and all(
            matrix_s == matrix_o
            for matrix_s, matrix_o in zip(self.matrices, other.matrices, strict=True)
        )


class SquareMatrix(Matrix):
    """Base class for matrices with equal numbers of rows and columns."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        if shape[0] != shape[1]:
            msg = f"{shape} is not a valid shape for a square matrix."
            raise ValueError(msg)
        super().__init__(shape, **kwargs)

    @property
    @abc.abstractmethod
    def log_abs_det(self) -> float:
        """Logarithm of absolute value of determinant of matrix.

        For matrix representations of metrics it is proportional to the logarithm of the
        density of then Riemannian measure associated with metric with respect to the
        Lebesgue measure.
        """


class SquareMatrixProduct(MatrixProduct, SquareMatrix):
    """Matrix implicitly defined as a product of a sequence of square matrices.

    All the matrices must have the same shape.
    """

    def __init__(
        self, matrices: Iterable[Matrix], *, check_shapes: bool = True
    ) -> None:
        matrices = tuple(matrices)
        if check_shapes:
            if matrices[0].shape[0] != matrices[0].shape[1]:
                msg = f"{matrices[0]} is not square."
                raise ValueError(msg)
            for matrix in matrices[1:]:
                if matrix.shape != matrices[0].shape:
                    msg = f"{matrices[0]} and {matrix} have different shapes."
                    raise ValueError(msg)
        super().__init__(matrices, check_shapes=False)

    @property
    def log_abs_det(self) -> float:
        return sum(matrix.log_abs_det for matrix in self.matrices)


class InvertibleMatrix(SquareMatrix):
    """Base class for non-singular square matrices."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        super().__init__(shape, **kwargs)
        self._inv = None

    @property
    def inv(self) -> SquareMatrix:
        """Inverse of matrix as a `Matrix` object.

        This will not necessarily form an explicit representation of the inverse matrix
        but may instead return a `Matrix` object that implements the matrix
        multiplication operators by solving the linear system defined by the original
        matrix object.
        """
        if self._inv is None:
            self._inv = self._construct_inv()
        return self._inv

    @abc.abstractmethod
    def _construct_inv(self) -> SquareMatrix:
        """Construct inverse of matrix as a `Matrix` object.

        This will not necessarily form an explicit representation of the inverse matrix
        but may instead return a `Matrix` object that implements the matrix
        multiplication operators by solving the linear system defined by the original
        matrix object.
        """


class InvertibleMatrixProduct(SquareMatrixProduct, InvertibleMatrix):
    """Matrix defined as a product of a sequence of invertible matrices.

    All the matrices must have the same shape.
    """

    def __init__(
        self,
        matrices: Iterable[InvertibleMatrix],
        *,
        check_shapes: bool = True,
    ) -> None:
        matrices = tuple(matrices)
        for matrix in matrices:
            if not isinstance(matrix, InvertibleMatrix):
                msg = f"matrix {matrix} is not invertible."
                raise TypeError(msg)
        super().__init__(matrices, check_shapes=check_shapes)

    def _construct_inv(self) -> InvertibleMatrixProduct:
        return InvertibleMatrixProduct(
            tuple(matrix.inv for matrix in reversed(self.matrices)),
        )


class SymmetricMatrix(SquareMatrix):
    """Base class for square matrices which are equal to their transpose."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        self._eigval = None
        self._eigvec = None
        super().__init__(shape, **kwargs)

    def _compute_eigendecomposition(self) -> None:
        self._eigval, eigvec = nla.eigh(self.array)
        self._eigvec = OrthogonalMatrix(eigvec)

    @property
    def eigval(self) -> NDArray:
        """Eigenvalues of matrix as a 1D array."""
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigval

    @property
    def eigvec(self) -> OrthogonalMatrix:
        """Eigenvectors of matrix stacked as columns of a `Matrix` object."""
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigvec

    def _construct_transpose(self) -> SymmetricMatrix:
        return self

    @property
    def log_abs_det(self) -> float:
        return np.log(np.abs(self.eigval)).sum()


class PositiveDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):
    """Base class for positive definite matrices."""

    def __init__(self, shape: tuple[int, int], **kwargs) -> None:
        self._sqrt = None
        super().__init__(shape, **kwargs)

    @property
    def sqrt(self) -> Matrix:
        """Square-root of matrix satisfying `matrix == sqrt @ sqrt.T`.

        This will in general not correspond to the unique, if defined, symmetric square
        root of a symmetric matrix but instead may return any matrix satisfying the
        above property.
        """
        if self._sqrt is None:
            self._sqrt = self._construct_sqrt()
        return self._sqrt

    @abc.abstractmethod
    def _construct_sqrt(self) -> Matrix:
        """Construct qquare-root of matrix satisfying `matrix == sqrt @ sqrt.T`.

        This will in general not correspond to the unique, if defined,
        symmetric square root of a symmetric matrix but instead may return any
        matrix satisfying the above property.
        """


class IdentityMatrix(PositiveDefiniteMatrix, ImplicitArrayMatrix):
    """Matrix representing identity operator on a vector space.

    Array representation has ones on diagonal elements and zeros elsewhere.
    May be defined with an implicit shape represented by `(None, None)` which
    will allow use for subset of operations where shape is not required to be
    known.
    """

    def __init__(self, size: int | None = None) -> None:
        """
        Args:
            size: Number of rows / columns in matrix or `None` if matrix is to be
                implicitly shaped.
        """
        super().__init__((size, size))

    def _scalar_multiply(self, scalar: ScalarLike) -> Matrix:
        if scalar > 0:
            return PositiveScaledIdentityMatrix(scalar, self.shape[0])
        return ScaledIdentityMatrix(scalar, self.shape[0])

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return other

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return other

    @property
    def eigval(self) -> NDArray:
        return self.diagonal

    def _construct_sqrt(self) -> IdentityMatrix:
        return self

    @property
    def eigvec(self) -> IdentityMatrix:
        return self

    def _construct_inv(self) -> IdentityMatrix:
        return self

    @property
    def diagonal(self) -> NDArray:
        return np.ones(self.shape[0])

    def _construct_array(self) -> NDArray:
        if self.shape[0] is None:
            msg = (
                "Cannot get array representation for identity matrix with implicit "
                "size."
            )
            raise RuntimeError(msg)
        return np.identity(self.shape[0])

    @property
    def log_abs_det(self) -> float:
        return 0.0

    def _compute_hash(self) -> int:
        return hash(self.shape)

    def _check_equality(self, other: Matrix) -> bool:
        return self.shape == other.shape


class DifferentiableMatrix(InvertibleMatrix):
    """Parameterically defined matrix defining gradient of scalar operations.

    Parameterically-defined here means the matrix is constructed as a function
    of one or more parameters, with the convention that the parameters
    correspond to **the first parameter in the `__init__` method of the
    `DifferentiableMatrix` subclass**, with multiple parameters wrapped in to
    for example a tuple, dict or list.

    The gradient is defined for the scalar functions of the matrix parameters
    implemented by the method `log_abs_det`, corresponding to

        f(params) = log(abs(det(matrix(params))))

    and by the quadratic form `vector @ matrix(params).inv @ vector`.

    In both cases the gradients are with respect to the parameter(s). The
    returned gradients will have the same structure as the first parameter of
    the `__init__` method of the relevant `DifferentiableMatrix` subclass,
    for example if the first parameter is a tuple or dict of arrays then the
    returned gradients will be respectively a tuple or dict of arrays of the
    same shapes and with the same indices / keys.
    """

    @property
    @abc.abstractmethod
    def grad_log_abs_det(self) -> NDArray:
        """Gradient of logarithm of absolute value of determinant of matrix."""

    @abc.abstractmethod
    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        """Gradient of quadratic form `vector @ matrix.inv @ vector`.

        Args:
            vector: 1D array representing vector to evaluate quadratic form at.
        """


class ScaledIdentityMatrix(SymmetricMatrix, DifferentiableMatrix, ImplicitArrayMatrix):
    """Matrix representing scalar multiplication operation on a vector space.

    Array representation has common scalar on diagonal elements and zeros
    elsewhere. May be defined with an implicit shape reprsented by
    `(None, None)` which will allow use for subset of operations where shape
    is not required to be known.
    """

    def __init__(self, scalar: float, size: int | None = None) -> None:
        """
        Args:
            scalar: Scalar multiplier for identity matrix.
            size: Number of rows / columns in matrix. If `None` the matrix will be
                implicitly-shaped and only the subset of operations which do not rely on
                an explicit shape will be available.
        """
        if scalar == 0:
            msg = "scalar must be non-zero"
            raise ValueError(msg)
        self._scalar = scalar
        super().__init__((size, size))

    @property
    def scalar(self) -> float:
        """Scalar multiplier."""
        return self._scalar

    def _scalar_multiply(self, scalar: ScalarLike) -> ScaledIdentityMatrix:
        return ScaledIdentityMatrix(scalar * self._scalar, self.shape[0])

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self._scalar * other

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return self._scalar * other

    @property
    def eigval(self) -> NDArray:
        return self.diagonal

    @property
    def eigvec(self) -> IdentityMatrix:
        return IdentityMatrix(self.shape[0])

    def _construct_inv(self) -> ScaledIdentityMatrix:
        return ScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    @property
    def diagonal(self) -> NDArray:
        return self._scalar * np.ones(self.shape[0])

    def _construct_array(self) -> NDArray:
        if self.shape[0] is None:
            msg = (
                "Cannot get array representation for scaled identity matrix with "
                "implicit size."
            )
            raise RuntimeError(msg)
        return self._scalar * np.identity(self.shape[0])

    @property
    def log_abs_det(self) -> float:
        if self.shape[0] is None:
            msg = (
                "Cannot get log determinant for scaled identity matrix with implicit "
                "size."
            )
            raise RuntimeError(msg)
        return self.shape[0] * np.log(abs(self._scalar))

    @property
    def grad_log_abs_det(self) -> ScalarLike:
        return self.shape[0] / self._scalar

    def grad_quadratic_form_inv(self, vector: NDArray) -> float:
        return -np.sum(vector**2) / self._scalar**2

    def __str__(self) -> str:
        return f"(shape={self.shape}, scalar={self._scalar})"

    def _compute_hash(self) -> int:
        return hash((self.shape, self.scalar))

    def _check_equality(self, other: ScaledIdentityMatrix) -> bool:
        return self.shape == other.shape and self.scalar == other.scalar


class PositiveScaledIdentityMatrix(ScaledIdentityMatrix, PositiveDefiniteMatrix):
    """Specialisation of `ScaledIdentityMatrix` with positive scalar parameter.

    Restricts the `scalar` parameter to be strictly positive.
    """

    def __init__(self, scalar: ScalarLike, size: int | None = None) -> None:
        if scalar <= 0:
            msg = "Scalar multiplier must be positive."
            raise ValueError(msg)
        super().__init__(scalar, size)

    def _scalar_multiply(self, scalar: ScalarLike) -> ScaledIdentityMatrix:
        if scalar > 0:
            return PositiveScaledIdentityMatrix(scalar * self._scalar, self.shape[0])
        return super()._scalar_multiply(scalar)

    def _construct_inv(self) -> PositiveScaledIdentityMatrix:
        return PositiveScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    def _construct_sqrt(self) -> PositiveScaledIdentityMatrix:
        return PositiveScaledIdentityMatrix(self._scalar**0.5, self.shape[0])


class DiagonalMatrix(SymmetricMatrix, DifferentiableMatrix, ImplicitArrayMatrix):
    """Matrix with non-zero elements only along its diagonal."""

    def __init__(self, diagonal: NDArray) -> None:
        """
        Args:
            diagonal: 1D array specifying diagonal elements of matrix.
        """
        if diagonal.ndim != 1:
            msg = "Specified diagonal must be a 1D array."
            raise ValueError(msg)
        super().__init__((diagonal.size, diagonal.size), _diagonal=diagonal)

    @property
    def diagonal(self) -> NDArray:
        return self._diagonal

    def _scalar_multiply(self, scalar: ScalarLike) -> DiagonalMatrix:
        return DiagonalMatrix(self.diagonal * scalar)

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        if other.ndim == 2:  # noqa: PLR2004
            return self.diagonal[:, None] * other
        if other.ndim == 1:
            return self.diagonal * other
        msg = (
            "Left matrix multiplication only defined for one or two dimensional "
            "right hand sides."
        )
        raise ValueError(msg)

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return self.diagonal * other

    @property
    def eigvec(self) -> IdentityMatrix:
        return IdentityMatrix(self.shape[0])

    @property
    def eigval(self) -> NDArray:
        return self.diagonal

    def _construct_inv(self) -> DiagonalMatrix:
        return DiagonalMatrix(1.0 / self.diagonal)

    def _construct_array(self) -> NDArray:
        return np.diag(self.diagonal)

    @property
    def grad_log_abs_det(self) -> NDArray:
        return 1.0 / self.diagonal

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        return -((self.inv @ vector) ** 2)

    def _compute_hash(self) -> int:
        return hash_array(self.diagonal)

    def _check_equality(self, other: DiagonalMatrix) -> bool:
        return np.array_equal(self.diagonal, other.diagonal)


class PositiveDiagonalMatrix(DiagonalMatrix, PositiveDefiniteMatrix):
    """Specialisation of `DiagonalMatrix` with positive diagonal parameter.

    Restricts all values in `diagonal` array parameter to be strictly positive.
    """

    def __init__(self, diagonal: NDArray) -> None:
        if not np.all(diagonal > 0):
            msg = "Diagonal values must all be positive."
            raise ValueError(msg)
        super().__init__(diagonal)

    def _scalar_multiply(self, scalar: ScalarLike) -> DiagonalMatrix:
        if scalar > 0:
            return PositiveDiagonalMatrix(self.diagonal * scalar)
        return super()._scalar_multiply(scalar)

    def _construct_inv(self) -> PositiveDiagonalMatrix:
        return PositiveDiagonalMatrix(1.0 / self.diagonal)

    def _construct_sqrt(self) -> PositiveDiagonalMatrix:
        return PositiveDiagonalMatrix(self.diagonal**0.5)


def _make_array_triangular(array: NDArray, *, lower: bool) -> NDArray:
    """Make array lower/upper triangular by zeroing above/below diagonal."""
    return np.tril(array) if lower else np.triu(array)


class TriangularMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Matrix with non-zero values only in lower or upper triangle elements."""

    def __init__(
        self,
        array: NDArray,
        *,
        lower: bool = True,
        make_triangular: bool = True,
    ) -> None:
        """
        Args:
            array: 2D array containing lower / upper triangular element values of
                matrix. Any values above (below) diagonal are ignored for lower (upper)
                triangular matrices i.e. when `lower == True` (`lower == False`).
            lower: Whether the matrix is lower-triangular (`True`) or upper-triangular
                (`False`).
            make_triangular: Whether to ensure `array` is triangular by explicitly
                zeroing entries in upper triangle if `lower == True` and in lower
                triangle if `lower == False`.
        """
        array = _make_array_triangular(array, lower=lower) if make_triangular else array
        super().__init__(array.shape, _array=array)
        self._lower = lower

    def _scalar_multiply(self, scalar: ScalarLike) -> TriangularMatrix:
        return TriangularMatrix(
            self.array * scalar,
            lower=self.lower,
            make_triangular=False,
        )

    @property
    def lower(self) -> bool:
        return self._lower

    def _construct_inv(self) -> InverseTriangularMatrix:
        return InverseTriangularMatrix(
            self.array,
            lower=self.lower,
            make_triangular=False,
        )

    def _construct_transpose(self) -> TriangularMatrix:
        return TriangularMatrix(
            self.array.T,
            lower=not self.lower,
            make_triangular=False,
        )

    @property
    def log_abs_det(self) -> float:
        return np.log(np.abs(self.diagonal)).sum()

    def __str__(self) -> str:
        return f"(shape={self.shape}, lower={self.lower})"


class InverseTriangularMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Triangular matrix implicitly specified by its inverse."""

    def __init__(
        self,
        inverse_array: NDArray,
        *,
        lower: bool = True,
        make_triangular: bool = True,
    ) -> None:
        """
        Args:
            inverse_array: 2D containing values of *inverse* of this matrix, with the
                inverse of a lower (upper) triangular matrix being itself lower (upper)
                triangular. Any values above (below) diagonal are ignored for lower
                (upper) triangular matrices i.e. when `lower == True` (`lower ==
                False`).
            lower: Whether the matrix is lower-triangular (`True`) or upper-triangular
                (`False`).
            make_triangular: Whether to ensure `inverse_array` is triangular by
                explicitly zeroing entries in upper triangle if `lower == True` and in
                lower triangle if `lower == False`.
        """
        inverse_array = np.asarray_chkfinite(inverse_array)
        inverse_array = (
            _make_array_triangular(inverse_array, lower=lower)
            if make_triangular
            else inverse_array
        )
        super().__init__(inverse_array.shape, _inverse_array=inverse_array)
        self._lower = lower

    def _scalar_multiply(self, scalar: ScalarLike) -> InverseTriangularMatrix:
        return InverseTriangularMatrix(
            self._inverse_array / scalar,
            lower=self.lower,
            make_triangular=False,
        )

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return sla.solve_triangular(
            self._inverse_array,
            other,
            lower=self.lower,
            check_finite=False,
        )

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return sla.solve_triangular(
            self._inverse_array,
            other.T,
            lower=self.lower,
            trans=1,
            check_finite=False,
        ).T

    @property
    def lower(self) -> bool:
        return self._lower

    def _construct_inv(self) -> TriangularMatrix:
        return TriangularMatrix(
            self._inverse_array,
            lower=self.lower,
            make_triangular=False,
        )

    def _construct_transpose(self) -> InverseTriangularMatrix:
        return InverseTriangularMatrix(
            self._inverse_array.T,
            lower=not self.lower,
            make_triangular=False,
        )

    def _construct_array(self) -> NDArray:
        return sla.solve_triangular(
            self._inverse_array,
            np.identity(self.shape[0]),
            lower=self.lower,
            check_finite=False,
        )

    @property
    def diagonal(self) -> NDArray:
        return 1.0 / self._inverse_array.diagonal()

    @property
    def log_abs_det(self) -> float:
        return -self.inv.log_abs_det

    def __str__(self) -> str:
        return f"(shape={self.shape}, lower={self.lower})"

    def _compute_hash(self) -> int:
        return hash_array(self._inverse_array)

    def _check_equality(self, other: InverseTriangularMatrix) -> bool:
        return np.array_equal(self._inverse_array, other._inverse_array)  # noqa: SLF001


class _BaseTriangularFactoredDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):
    def __init__(self, size: int, sign: Literal[-1, 1] = 1, **kwargs) -> None:
        super().__init__((size, size), **kwargs)
        if sign not in (-1, 1):
            msg = "sign must be equal to +1 or -1"
            raise ValueError(msg)
        self._sign = sign

    @property
    def factor(self) -> TriangularMatrix | InverseTriangularMatrix:
        """Triangular matrix with `matrix = sign * factor @ factor.T`."""
        return self._factor

    @property
    def sign(self) -> int:
        """Signed binary value with `matrix = sign * factor @ factor.T`."""
        return self._sign

    def _construct_inv(self) -> TriangularFactoredDefiniteMatrix:
        return TriangularFactoredDefiniteMatrix(
            factor=self.factor.inv.T,
            sign=self._sign,
        )

    @property
    def log_abs_det(self) -> float:
        return 2 * self.factor.log_abs_det

    def __str__(self) -> str:
        return f"(shape={self.shape}, sign={self._sign})"


class TriangularFactoredDefiniteMatrix(
    _BaseTriangularFactoredDefiniteMatrix,
    DifferentiableMatrix,
    ImplicitArrayMatrix,
):
    """Matrix specified as a signed self-product of a triangular factor.

    The matrix is assumed to have the form

        matrix = sign * factor @ factor.T

    for and upper- or lower-trinagular matrix `factor` and signed binary value `sign`
    (i.e. `sign == +1 or sign == -1`), with the matrix being positive definite if `sign
    == +1` and negative definite if `sign == -1` under the assumption that `factor` is
    non-singular.
    """

    def __init__(
        self,
        factor: NDArray | TriangularMatrix | InverseTriangularMatrix,
        sign: Literal[-1, 1] = 1,
        factor_is_lower: bool | None = None,
    ) -> None:
        """
        Args:
            factor: The triangular factor parameterising the matrix. Defined either a
                2D array, in which case only the lower- or upper-triangular elements are
                used depending on the value of the `factor_is_lower` boolean keyword
                argument, or as a `TriangularMatrix` / `InverseTriangularMatrix`
                instance in which case `factor_is_lower` is ignored, with `factor.lower`
                instead determining if the factor is lower- or upper-triangular.
            sign: +/-1 multiplier of factor product, corresponding respectively to a
                strictly positive- or negative-definite matrix.
            factor_is_lower: Whether the array `factor` is lower- or upper-triangular.
        """
        if not isinstance(factor, TriangularMatrix | InverseTriangularMatrix):
            if factor_is_lower not in (True, False):
                msg = (
                    "For array `factor` parameter `factor_is_lower` must be specified "
                    "as a boolean value."
                )
                raise ValueError(msg)
            factor = TriangularMatrix(factor, lower=factor_is_lower)
        self._factor = factor
        super().__init__(factor.shape[0], sign=sign)

    def _scalar_multiply(self, scalar: ScalarLike) -> TriangularFactoredDefiniteMatrix:
        return TriangularFactoredDefiniteMatrix(
            factor=abs(scalar) ** 0.5 * self.factor,
            sign=self.sign * np.sign(scalar),
        )

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self.sign * (self.factor @ (self.factor.T @ other))

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return self.sign * ((other @ self.factor) @ self.factor.T)

    @property
    def grad_log_abs_det(self) -> NDArray:
        return np.diag(2 / self.factor.diagonal)

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        inv_factor_vector = self.factor.inv @ vector
        inv_vector = self.inv @ vector
        return _make_array_triangular(
            -2 * self.sign * np.outer(inv_vector, inv_factor_vector),
            lower=self.factor.lower,
        )

    def _construct_array(self) -> NDArray:
        return self.sign * (self.factor @ self.factor.array.T)

    def _compute_hash(self) -> int:
        return hash((self.factor, self.sign))

    def _check_equality(self, other: TriangularFactoredDefiniteMatrix) -> bool:
        return self.sign == other.sign and self.factor == other.factor


class TriangularFactoredPositiveDefiniteMatrix(
    TriangularFactoredDefiniteMatrix,
    PositiveDefiniteMatrix,
):
    """Positive definite matrix parametrized a triangular matrix product.

    The matrix is assumed to have the parameterisation

        matrix = factor @ factor.T

    where `factor` is an upper- or lower-triangular matrix. Note for the case `factor`
    is lower-triangular this corresponds to the standard Cholesky factorisation of a
    positive definite matrix.
    """

    def __init__(
        self,
        factor: NDArray | TriangularMatrix | InverseTriangularMatrix,
        *,
        factor_is_lower: bool = True,
    ) -> None:
        """
        Args:
            factor: The triangular factor parameterising the matrix. Defined either a
                2D array, in which case only the lower- or upper-triangular elements are
                used depending on the value of the `factor_is_lower` boolean keyword
                argument, or as a `TriangularMatrix` / `InverseTriangularMatrix`
                instance in which case `factor_is_lower` is ignored, with `factor.lower`
                instead determining if the factor is lower- or upper-triangular.
            factor_is_lower: Whether the array `factor` is lower- or upper-triangular.
        """
        super().__init__(factor, sign=1, factor_is_lower=factor_is_lower)

    def _scalar_multiply(self, scalar: ScalarLike) -> TriangularFactoredDefiniteMatrix:
        if scalar > 0:
            return TriangularFactoredPositiveDefiniteMatrix(
                factor=scalar**0.5 * self.factor,
            )
        return super()._scalar_multiply(scalar)

    def _construct_inv(self) -> TriangularFactoredPositiveDefiniteMatrix:
        return TriangularFactoredPositiveDefiniteMatrix(factor=self.factor.inv.T)

    def _construct_sqrt(self) -> TriangularMatrix | InverseTriangularMatrix:
        return self.factor


class DenseDefiniteMatrix(
    _BaseTriangularFactoredDefiniteMatrix,
    DifferentiableMatrix,
    ExplicitArrayMatrix,
):
    """Definite matrix specified by a dense 2D array."""

    def __init__(
        self,
        array: NDArray,
        factor: TriangularMatrix | InverseTriangularMatrix | None = None,
        *,
        is_posdef: bool = True,
    ) -> None:
        """
        Args:
            array : 2D array specifying matrix entries.
            factor: Optional argument giving the triangular factorisation of the matrix
                such that `matrix = factor @ factor.T` if `is_posdef=True` or `matrix =
                -factor @ factor.T` otherwise. If not pre-computed and specified at
                initialisation a factorisation will only be computed when first required
                by an operation which depends on the factor.
            is_posdef: Whether matrix (and so corresponding array representation) is
                positive definite, with the matrix assumed to be negative-definite if
                not. This is **not** checked on initialisation, and so if `array` is
                positive (negative) definite and `is_posdef` is `False` (`True`) then a
                `LinAlgError` exception will be if a later attempt is made to factorize
                the matrix.
        """
        super().__init__(array.shape[0], sign=1 if is_posdef else -1, _array=array)
        self._factor = factor

    def _scalar_multiply(self, scalar: ScalarLike) -> DenseDefiniteMatrix:
        if (scalar > 0) == (self._sign == 1):
            return DensePositiveDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else abs(scalar) ** 0.5 * self._factor,
            )
        return DenseDefiniteMatrix(
            scalar * self.array,
            None if self._factor is None else abs(scalar) ** 0.5 * self._factor,
            is_posdef=False,
        )

    @property
    def factor(self) -> TriangularMatrix | InverseTriangularMatrix:
        if self._factor is None:
            try:
                self._factor = TriangularMatrix(
                    nla.cholesky(self._sign * self._array),
                    lower=True,
                    make_triangular=False,
                )
            except nla.LinAlgError as e:
                msg = "Cholesky factorisation failed."
                raise LinAlgError(msg) from e
        return self._factor

    @property
    def grad_log_abs_det(self) -> NDArray:
        return self.inv.array

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        inv_matrix_vector = self.inv @ vector
        return -np.outer(inv_matrix_vector, inv_matrix_vector)

    def _construct_inv(self) -> DenseDefiniteMatrix:
        return DenseDefiniteMatrix(
            super()._construct_inv().array,
            factor=self.factor.inv.T,
            is_posdef=(self._sign == 1),
        )


class DensePositiveDefiniteMatrix(DenseDefiniteMatrix, PositiveDefiniteMatrix):
    """Positive-definite matrix specified by a dense 2D array."""

    def __init__(
        self,
        array: NDArray,
        factor: TriangularMatrix | InverseTriangularMatrix | None = None,
    ) -> None:
        """
        Args:
            array: 2D array specifying matrix entries.
            factor: Optional argument giving the triangular factorisation of the matrix
                such that `matrix = factor @ factor.T`. If not pre-computed and
                specified at initialisation a factorisation will only be computed when
                first required by an operation which depends on the factor.
        """
        super().__init__(array=array, factor=factor, is_posdef=True)

    def _construct_inv(self) -> DensePositiveDefiniteMatrix:
        return DensePositiveDefiniteMatrix(
            super()._construct_inv(),
            factor=self.factor.inv.T,
        )

    def _construct_sqrt(self) -> TriangularMatrix | InverseTriangularMatrix:
        return self.factor


class DensePositiveDefiniteProductMatrix(DensePositiveDefiniteMatrix):
    """Positive-definite matrix specified as a signed symmetric product.

    The matrix is assumed to have the form

        matrix = rect_matrix @ pos_def_matrix @ rect_matrix.T

    for a dense rectangular matrix `rect_matrix` with shape `(dim_0, dim_1)` (`dim_1 >
    dim_0`) positive definite matrix `pos_def_matrix` with shape `(dim_1, dim_1)`, with
    the resulting matrix being positive definite under the assumption that `rect_matrix`
    has full row rank.
    """

    def __init__(
        self,
        rect_matrix: NDArray | Matrix,
        pos_def_matrix: PositiveDefiniteMatrix | None = None,
    ) -> None:
        """
        Args:
            rect_matrix: Rectangular matrix of shape `(dim_0, dim_1)` with it and its
                transpose forming the leftmost and righmost term respectively in the
                symmetric matrix product defining the matrix.
            pos_def_matrix: Optional positive positive definite matrix with shape
                `(dim_inner, dim_inner)` specifying inner term in symmetric matrix
                product defining matrix. If `None` an identity matrix is used.
        """
        if not rect_matrix.shape[0] < rect_matrix.shape[1]:
            msg = "rect_matrix must have more columns than rows"
            raise ValueError(msg)
        if not isinstance(rect_matrix, Matrix):
            rect_matrix = DenseRectangularMatrix(rect_matrix)
        self._rect_matrix = rect_matrix
        if pos_def_matrix is None:
            pos_def_matrix = IdentityMatrix(rect_matrix.shape[1])
        self._pos_def_matrix = pos_def_matrix
        _array = rect_matrix @ (pos_def_matrix @ rect_matrix.T.array)
        super().__init__(_array)

    @property
    def grad_log_abs_det(self) -> NDArray:
        return 2 * (self.inv @ (self._rect_matrix.array @ self._pos_def_matrix))

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        inv_matrix_vector = self.inv @ vector
        return -2 * np.outer(
            inv_matrix_vector,
            self._pos_def_matrix @ (self._rect_matrix.T @ inv_matrix_vector),
        )


class DenseSquareMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Dense non-singular square matrix."""

    def __init__(
        self,
        array: NDArray,
        lu_and_piv: tuple[NDArray, NDArray] | None = None,
        lu_transposed: bool | None = None,
    ) -> None:
        """
        Args:
            array: 2D array specifying matrix entries.
            lu_and_piv: Pivoted LU factorisation represented as a tuple with first
                element a 2D array containing the lower and upper triangular factors
                (with the unit diagonal of the lower triangular factor not stored) and
                the second element a 1D array containing the pivot indices. Corresponds
                to the output of `scipy.linalg.lu_factor` and input to
                `scipy.linalg.lu_solve`.
            lu_transposed: Whether LU factorisation is of original array or its
                transpose.
        """
        super().__init__(array.shape, _array=array)
        self._lu_and_piv = lu_and_piv
        self._lu_transposed = lu_transposed

    def _scalar_multiply(self, scalar: ScalarLike) -> DenseSquareMatrix:
        if self._lu_and_piv is None or self._lu_transposed is None:
            return DenseSquareMatrix(scalar * self._array)
        old_lu, piv = self._lu_and_piv
        # Multiply upper-triangle by scalar
        new_lu = old_lu + (scalar - 1) * np.triu(old_lu)
        return DenseSquareMatrix(
            scalar * self._array,
            (new_lu, piv),
            self._lu_transposed,
        )

    @property
    def lu_and_piv(self) -> tuple[NDArray, NDArray]:
        """Pivoted LU factorisation of matrix."""
        if self._lu_and_piv is None:
            self._lu_and_piv = sla.lu_factor(self._array, check_finite=False)
            self._lu_transposed = False
        return self._lu_and_piv

    @property
    def log_abs_det(self) -> float:
        lu, piv = self.lu_and_piv
        return np.log(np.abs(lu.diagonal())).sum()

    def _construct_transpose(self) -> DenseSquareMatrix:
        lu_and_piv = self.lu_and_piv
        return DenseSquareMatrix(self._array.T, lu_and_piv, not self._lu_transposed)

    def _construct_inv(self) -> InverseLUFactoredSquareMatrix:
        lu_and_piv = self.lu_and_piv
        return InverseLUFactoredSquareMatrix(
            self._array,
            inv_lu_and_piv=lu_and_piv,
            inv_lu_transposed=self._lu_transposed,
        )


class InverseLUFactoredSquareMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Square matrix implicitly defined by LU factorisation of inverse."""

    def __init__(
        self,
        inv_array: NDArray,
        inv_lu_and_piv: tuple[NDArray, NDArray],
        *,
        inv_lu_transposed: bool,
    ) -> None:
        """
        Args:
            inv_array: 2D array specifying inverse matrix entries.
            inv_lu_and_piv: Pivoted LU factorisation represented as a tuple with first
                element a 2D array containing the lower and upper triangular factors
                (with the unit diagonal of the lower triangular factor not stored) and
                the second element a 1D array containing the pivot indices. Corresponds
                to the output of `scipy.linalg.lu_factor` and input to
                `scipy.linalg.lu_solve`.
            inv_lu_transposed: Whether LU factorisation is of inverse of array or
                transpose of inverse of array.
        """
        super().__init__(inv_array.shape)
        self._inv_array = inv_array
        self._inv_lu_and_piv = inv_lu_and_piv
        self._inv_lu_transposed = inv_lu_transposed

    def _scalar_multiply(self, scalar: ScalarLike) -> InverseLUFactoredSquareMatrix:
        old_inv_lu, piv = self._inv_lu_and_piv
        # Divide upper-triangle by scalar
        new_inv_lu = old_inv_lu - (scalar - 1) / scalar * np.triu(old_inv_lu)
        return InverseLUFactoredSquareMatrix(
            self._inv_array / scalar,
            (new_inv_lu, piv),
            inv_lu_transposed=self._inv_lu_transposed,
        )

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return sla.lu_solve(
            self._inv_lu_and_piv,
            other,
            self._inv_lu_transposed,
            check_finite=False,
        )

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return sla.lu_solve(
            self._inv_lu_and_piv,
            other.T,
            not self._inv_lu_transposed,
            check_finite=False,
        ).T

    @property
    def log_abs_det(self) -> float:
        return -np.log(np.abs(self._inv_lu_and_piv[0].diagonal())).sum()

    def _construct_array(self) -> NDArray:
        return self @ np.identity(self.shape[0])

    def _construct_inv(self) -> DenseSquareMatrix:
        return DenseSquareMatrix(
            self._inv_array,
            self._inv_lu_and_piv,
            self._inv_lu_transposed,
        )

    def _construct_transpose(self) -> InverseLUFactoredSquareMatrix:
        return InverseLUFactoredSquareMatrix(
            self._inv_array.T,
            self._inv_lu_and_piv,
            inv_lu_transposed=not self._inv_lu_transposed,
        )

    def _compute_hash(self) -> int:
        return hash_array(self._inv_array)

    def _check_equality(self, other: InverseLUFactoredSquareMatrix) -> bool:
        return np.array_equal(self._inv_array, other._inv_array)  # noqa: SLF001


class DenseSymmetricMatrix(SymmetricMatrix, InvertibleMatrix, ExplicitArrayMatrix):
    """Dense non-singular symmetric matrix."""

    def __init__(
        self,
        array: NDArray,
        eigvec: NDArray | OrthogonalMatrix | None = None,
        eigval: NDArray | None = None,
    ) -> None:
        """
        Args:
            array: Explicit 2D array representation of matrix.
            eigvec: Optional. If specified either a 2D array or an `OrthogonalMatrix`
                instance, in both cases the columns of the matrix corresponding to the
                orthonormal set of eigenvectors of the matrix being constructed.
            eigval: Optional. If specified a 1D array containing the eigenvalues of the
                matrix being constructed, with `eigval[i]` the eigenvalue associated
                with column `i` of `eigvec`.
        """
        super().__init__(array.shape, _array=array)
        if isinstance(eigvec, np.ndarray):
            eigvec = OrthogonalMatrix(eigvec)
        self._eigvec = eigvec
        self._eigval = eigval

    def _scalar_multiply(self, scalar: ScalarLike) -> DenseSymmetricMatrix:
        return DenseSymmetricMatrix(
            self.array * scalar,
            self._eigvec,
            None if self._eigval is None else self._eigval * scalar,
        )

    def _construct_inv(self) -> EigendecomposedSymmetricMatrix:
        return EigendecomposedSymmetricMatrix(self.eigvec, 1 / self.eigval)


class OrthogonalMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Square matrix with columns and rows that are orthogonal unit vectors."""

    def __init__(self, array: NDArray) -> None:
        """
        Args:
            array: Explicit 2D array representation of matrix.
        """
        super().__init__(array.shape, _array=array)

    def _scalar_multiply(self, scalar: ScalarLike) -> ScaledOrthogonalMatrix:
        return ScaledOrthogonalMatrix(scalar, self.array)

    @property
    def log_abs_det(self) -> float:
        return 0.0

    def _construct_transpose(self) -> OrthogonalMatrix:
        return OrthogonalMatrix(self.array.T)

    def _construct_inv(self) -> OrthogonalMatrix:
        return self.T


class ScaledOrthogonalMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Matrix corresponding to orthogonal matrix multiplied by a scalar.

    Matrix is assumed to have the paramterisation

        matrix = scalar * orth_array

    where `scalar` is a real-valued scalar and `orth_array` is an orthogonal matrix
    represented as a square 2D array.
    """

    def __init__(self, scalar: float, orth_array: NDArray) -> None:
        """
        Args:
            scalar: Scalar multiplier as a floating point value.
            orth_array: 2D array representation of orthogonal matrix.
        """
        super().__init__(orth_array.shape, _orth_array=orth_array)
        self._scalar = scalar

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self._scalar * (self._orth_array @ other)

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return self._scalar * (other @ self._orth_array)

    def _scalar_multiply(self, scalar: ScalarLike) -> ScaledOrthogonalMatrix:
        return ScaledOrthogonalMatrix(scalar * self._scalar, self._orth_array)

    def _construct_array(self) -> NDArray:
        return self._scalar * self._orth_array

    @property
    def diagonal(self) -> NDArray:
        return self._scalar * self._orth_array.diagonal()

    @property
    def log_abs_det(self) -> float:
        return self.shape[0] * np.log(abs(self._scalar))

    def _construct_transpose(self) -> ScaledOrthogonalMatrix:
        return ScaledOrthogonalMatrix(self._scalar, self._orth_array.T)

    def _construct_inv(self) -> ScaledOrthogonalMatrix:
        return ScaledOrthogonalMatrix(1 / self._scalar, self._orth_array.T)

    def _compute_hash(self) -> int:
        return hash((self._scalar, hash_array(self._orth_array)))

    def _check_equality(self, other: ScaledOrthogonalMatrix) -> bool:
        return self._scalar == other._scalar and (  # noqa: SLF001
            np.array_equal(self._orth_array, other._orth_array)  # noqa: SLF001
        )


class EigendecomposedSymmetricMatrix(
    SymmetricMatrix,
    InvertibleMatrix,
    ImplicitArrayMatrix,
):
    """Symmetric matrix parametrized by its eigendecomposition.

    The matrix is assumed to have the parameterisation

        matrix = eigvec @ diag(eigval) @ eigvec.T

    where `eigvec` is an orthogonal matrix, with columns corresponding to the
    eigenvectors of `matrix` and `eigval` is 1D array of the corresponding eigenvalues
    of `matrix`.
    """

    def __init__(self, eigvec: NDArray | OrthogonalMatrix, eigval: NDArray) -> None:
        """
        Args:
            eigvec: Either a 2D array or an `OrthogonalMatrix` instance, in both cases
                the columns of the matrix corresponding to the orthonormal set of
                eigenvectors of the matrix being constructed.
            eigval: A 1D array containing the eigenvalues of the matrix being
                constructed, with `eigval[i]` the eigenvalue associated with column `i`
                of `eigvec`.
        """
        if isinstance(eigvec, np.ndarray):
            eigvec = OrthogonalMatrix(eigvec)
        super().__init__(eigvec.shape)
        self._eigvec = eigvec
        self._eigval = eigval
        if not isinstance(eigval, np.ndarray) or eigval.size == 1:
            self.diag_eigval = ScaledIdentityMatrix(eigval)
        else:
            self.diag_eigval = DiagonalMatrix(eigval)

    def _scalar_multiply(self, scalar: ScalarLike) -> EigendecomposedSymmetricMatrix:
        return EigendecomposedSymmetricMatrix(self.eigvec, self.eigval * scalar)

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self.eigvec @ (self.diag_eigval @ (self.eigvec.T @ other))

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return ((other @ self.eigvec) @ self.diag_eigval) @ self.eigvec.T

    def _construct_inv(self) -> EigendecomposedSymmetricMatrix:
        return EigendecomposedSymmetricMatrix(self.eigvec, 1 / self.eigval)

    def _construct_array(self) -> NDArray:
        if self.shape[0] is None:
            msg = (
                "Cannot get array representation for symmetric eigendecomposed matrix "
                "with implicit size."
            )
            raise RuntimeError(msg)
        return self @ np.identity(self.shape[0])

    def _compute_hash(self) -> int:
        return hash((hash_array(self.eigval), self.eigvec))

    def _check_equality(self, other: EigendecomposedSymmetricMatrix) -> bool:
        return np.array_equal(self.eigval, other.eigval) and (
            self.eigvec == other.eigvec
        )


class EigendecomposedPositiveDefiniteMatrix(
    EigendecomposedSymmetricMatrix,
    PositiveDefiniteMatrix,
):
    """Positive definite matrix parametrized by its eigendecomposition.

    The matrix is assumed to have the parameterisation

        matrix = eigvec @ diag(eigval) @ eigvec.T

    where `eigvec` is an orthogonal matrix, with columns corresponding to the
    eigenvectors of `matrix` and `eigval` is 1D array of the corresponding strictly
    positive eigenvalues of `matrix`.
    """

    def __init__(self, eigvec: NDArray | OrthogonalMatrix, eigval: NDArray) -> None:
        if not np.all(eigval > 0):
            msg = "Eigenvalues must all be positive."
            raise ValueError(msg)
        super().__init__(eigvec, eigval)

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        if scalar > 0:
            return EigendecomposedPositiveDefiniteMatrix(
                self.eigvec,
                self.eigval * scalar,
            )
        return super()._scalar_multiply(scalar)

    def _construct_inv(self) -> EigendecomposedPositiveDefiniteMatrix:
        return EigendecomposedPositiveDefiniteMatrix(self.eigvec, 1 / self.eigval)

    def _construct_sqrt(self) -> EigendecomposedPositiveDefiniteMatrix:
        return EigendecomposedPositiveDefiniteMatrix(self.eigvec, self.eigval**0.5)


class SoftAbsRegularizedPositiveDefiniteMatrix(
    EigendecomposedPositiveDefiniteMatrix,
    DifferentiableMatrix,
):
    """Matrix transformed to be positive definite by regularising eigenvalues.

    Matrix is parametrized by a symmetric array `symmetric_array`, of which an
    eigendecomposition is formed `eigvec, eigval = eigh(symmetric_array)`, with the
    output matrix then `matrix = eigvec @ softabs(eigval) @ eigvec.T` where `softabs` is
    a smooth approximation to the absolute function.
    """

    def __init__(self, symmetric_array: NDArray, softabs_coeff: float) -> None:
        """
        Args:
            symmetric_array: 2D square array with symmetric values, i.e.
                `symmetric_array[i, j] == symmetric_array[j, i]` for all indices `i` and
                `j` which represents symmetric matrix to form eigenvalue-regularized
                transformation of.
            softabs_coeff: Positive regularisation coefficient for smooth approximation
                to absolute value. As the value tends to infinity the approximation
                becomes increasingly close to the absolute function.
        """
        if softabs_coeff <= 0:
            msg = "softabs_coeff must be positive."
            raise ValueError(msg)
        self._softabs_coeff = softabs_coeff
        self.unreg_eigval, eigvec = nla.eigh(symmetric_array)
        eigval = self.softabs(self.unreg_eigval)
        super().__init__(eigvec, eigval)

    def softabs(self, x: NDArray) -> NDArray:
        """Smooth approximation to absolute function."""
        return x / np.tanh(x * self._softabs_coeff)

    def grad_softabs(self, x: NDArray) -> NDArray:
        """Derivative of smooth approximation to absolute function."""
        return (
            1.0 / np.tanh(self._softabs_coeff * x)
            - self._softabs_coeff * x / np.sinh(self._softabs_coeff * x) ** 2
        )

    @property
    def grad_log_abs_det(self) -> NDArray:
        grad_eigval = self.grad_softabs(self.unreg_eigval) / self.eigval
        return EigendecomposedSymmetricMatrix(self.eigvec, grad_eigval).array

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        num_j_mtx = self.eigval[:, None] - self.eigval[None, :]
        num_j_mtx += np.diag(self.grad_softabs(self.unreg_eigval))
        den_j_mtx = self.unreg_eigval[:, None] - self.unreg_eigval[None, :]
        np.fill_diagonal(den_j_mtx, 1)
        j_mtx = num_j_mtx / den_j_mtx
        e_vct = (self.eigvec.T @ vector) / self.eigval
        return -((self.eigvec @ (np.outer(e_vct, e_vct) * j_mtx)) @ self.eigvec.T)


class BlockMatrix(ImplicitArrayMatrix):
    """Matrix with non-zero entries defined by a series of submatrix blocks."""

    @property
    @abc.abstractmethod
    def blocks(self) -> tuple[Matrix]:
        """Non-zero blocks of matrix as a tuple of Matrix instances."""

    def _compute_hash(self) -> int:
        return hash(tuple(block for block in self.blocks))

    def _check_equality(self, other: BlockMatrix) -> bool:
        return len(other.blocks) == len(self.blocks) and all(
            block_s == block_o
            for block_s, block_o in zip(self.blocks, other.blocks, strict=True)
        )


class SquareBlockDiagonalMatrix(InvertibleMatrix, BlockMatrix):
    """Square matrix with non-zero values only in blocks along diagonal."""

    def __init__(self, blocks: Iterable[SquareMatrix]) -> None:
        """
        Args:
            blocks: Sequence of square matrices defining non-zero blocks along diagonal
                of matrix in order left-to-right.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, SquareMatrix) for block in self._blocks):
            msg = "All blocks must be square"
            raise ValueError(msg)
        sizes = tuple(block.shape[0] for block in self._blocks)
        total_size = sum(sizes)
        super().__init__((total_size, total_size))
        self._sizes = sizes
        self._splits = np.cumsum(sizes[:-1])

    @property
    def blocks(self) -> tuple[SquareMatrix]:
        """Blocks containing non-zero values left-to-right along diagonal."""
        return self._blocks

    def _split(self, other: NDArray, axis: int = 0) -> list[NDArray]:
        if other.shape[axis] != self.shape[0]:
            msg = f"Cannot split other along axis with size {other.shape[axis]}."
            raise ValueError(msg)
        return np.split(other, self._splits, axis=axis)

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return np.concatenate(
            [
                block @ part
                for block, part in zip(
                    self._blocks,
                    self._split(other, axis=0),
                    strict=True,
                )
            ],
            axis=0,
        )

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return np.concatenate(
            [
                part @ block
                for block, part in zip(
                    self._blocks,
                    self._split(other, axis=-1),
                    strict=True,
                )
            ],
            axis=-1,
        )

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        return SquareBlockDiagonalMatrix(
            tuple(scalar * block for block in self._blocks),
        )

    def _construct_array(self) -> NDArray:
        return sla.block_diag(*(block.array for block in self._blocks))

    def _construct_transpose(self) -> SquareBlockDiagonalMatrix:
        return SquareBlockDiagonalMatrix(tuple(block.T for block in self._blocks))

    def _construct_sqrt(self) -> SquareBlockDiagonalMatrix:
        return SquareBlockDiagonalMatrix(tuple(block.sqrt for block in self._blocks))

    @property
    def diagonal(self) -> NDArray:
        return np.concatenate([block.diagonal for block in self._blocks])

    def _construct_inv(self) -> SquareBlockDiagonalMatrix:
        return type(self)(tuple(block.inv for block in self._blocks))

    @property
    def eigval(self) -> NDArray:
        return np.concatenate([block.eigval for block in self._blocks])

    @property
    def eigvec(self) -> SquareBlockDiagonalMatrix:
        return SquareBlockDiagonalMatrix(tuple(block.eigvec for block in self._blocks))

    @property
    def log_abs_det(self) -> float:
        return sum(block.log_abs_det for block in self._blocks)


class SymmetricBlockDiagonalMatrix(SquareBlockDiagonalMatrix, SymmetricMatrix):
    """Symmetric specialisation of `SquareBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be symmetric, i.e. `block.T ==
    block`.
    """

    def __init__(self, blocks: Iterable[SymmetricMatrix]) -> None:
        """
        Args:
            blocks: Sequence of symmetric matrices defining non-zero blocks along
                diagonal of matrix in order left-to-right.
        """
        blocks = tuple(blocks)
        if not all(isinstance(block, SymmetricMatrix) for block in blocks):
            msg = "All blocks must be symmetric"
            raise ValueError(msg)
        super().__init__(blocks)

    def _scalar_multiply(self, scalar: ScalarLike) -> SymmetricBlockDiagonalMatrix:
        return SymmetricBlockDiagonalMatrix(
            tuple(scalar * block for block in self._blocks),
        )

    def _construct_transpose(self) -> SymmetricBlockDiagonalMatrix:
        return self


class PositiveDefiniteBlockDiagonalMatrix(
    SquareBlockDiagonalMatrix,
    PositiveDefiniteMatrix,
    DifferentiableMatrix,
):
    """Positive definite specialisation of `SymmetricBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be positive definite.
    """

    def __init__(self, blocks: Iterable[PositiveDefiniteMatrix]) -> None:
        """
        Args:
            blocks: Sequence of positive-definite matrices defining non-zero blocks
                along diagonal of matrix in order left-to-right.
        """
        blocks = tuple(blocks)
        if not all(isinstance(block, PositiveDefiniteMatrix) for block in blocks):
            msg = "All blocks must be positive definite"
            raise ValueError(msg)
        self.is_differentiable = all(
            isinstance(block, DifferentiableMatrix) for block in blocks
        )
        super().__init__(blocks)

    def _scalar_multiply(self, scalar: ScalarLike) -> SquareBlockDiagonalMatrix:
        if scalar > 0:
            return PositiveDefiniteBlockDiagonalMatrix(
                tuple(scalar * block for block in self._blocks),
            )
        return super()._scalar_multiply(scalar)

    def _construct_transpose(self) -> Self:
        return self

    def _construct_sqrt(self) -> SquareBlockDiagonalMatrix:
        return SquareBlockDiagonalMatrix(tuple(block.sqrt for block in self._blocks))

    @property
    def grad_log_abs_det(self) -> float:
        if self.is_differentiable:
            return tuple(block.grad_log_abs_det for block in self._blocks)
        msg = "Not all blocks are differentiable"
        raise RuntimeError(msg)

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        if self.is_differentiable:
            return tuple(
                block.grad_quadratic_form_inv(vector_part)
                for block, vector_part in zip(
                    self._blocks,
                    self._split(vector, axis=0),
                    strict=True,
                )
            )
        msg = "Not all blocks are differentiable"
        raise RuntimeError(msg)


class DenseRectangularMatrix(ExplicitArrayMatrix):
    """Dense rectangular matrix."""

    def __init__(self, array: NDArray) -> None:
        """
        Args:
            array: 2D array specifying matrix entries.
        """
        super().__init__(array.shape, _array=array)

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        return DenseRectangularMatrix(scalar * self.array)

    def _construct_transpose(self) -> DenseRectangularMatrix:
        return DenseRectangularMatrix(self.array.T)


class BlockRowMatrix(BlockMatrix):
    """Matrix composed of horizontal concatenation of a series of blocks."""

    def __init__(self, blocks: Iterable[Matrix]) -> None:
        """
        Args:
            blocks: Sequence of matrices defining a row of blocks in order left-to-right
                which when horizontally concatenated give the overall matrix.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, Matrix) for block in self._blocks):
            msg = "All blocks must be matrices"
            raise ValueError(msg)
        if len({block.shape[0] for block in self._blocks}) > 1:
            msg = "All blocks must have same row-dimension."
            raise ValueError(msg)
        col_dims = tuple(block.shape[1] for block in self._blocks)
        super().__init__(shape=(self._blocks[0].shape[0], sum(col_dims)))
        self._splits = np.cumsum(col_dims[:-1])

    @property
    def blocks(self) -> tuple[Matrix]:
        """Blocks of matrix in left-to-right order."""
        return self._blocks

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        if other.shape[0] != self.shape[1]:
            msg = "Inconsistent inner dimension for matrix multiply."
            raise ValueError(msg)
        return sum(
            [
                block @ part
                for block, part in zip(
                    self._blocks,
                    np.split(other, self._splits, axis=0),
                    strict=True,
                )
            ],
        )

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return np.concatenate([other @ block for block in self._blocks], axis=-1)

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        return BlockRowMatrix(tuple(scalar * block for block in self._blocks))

    def _construct_array(self) -> NDArray:
        return np.concatenate([block.array for block in self._blocks], axis=1)

    def _construct_transpose(self) -> BlockColumnMatrix:
        return BlockColumnMatrix(tuple(block.T for block in self._blocks))


class BlockColumnMatrix(BlockMatrix):
    """Matrix composed of vertical concatenation of a series of blocks."""

    def __init__(self, blocks: Iterable[Matrix]) -> None:
        """
        Args:
            blocks : Sequence of matrices defining a column of blocks in order
                top-to-bottom which when vertically concatenated give the overall
                matrix.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, Matrix) for block in self._blocks):
            msg = "All blocks must be matrices"
            raise ValueError(msg)
        if len({block.shape[1] for block in self._blocks}) > 1:
            msg = "All blocks must have same column-dimension."
            raise ValueError(msg)
        row_dims = tuple(block.shape[0] for block in self._blocks)
        super().__init__(shape=(sum(row_dims), self._blocks[0].shape[1]))
        self._splits = np.cumsum(row_dims[:-1])

    @property
    def blocks(self) -> tuple[Matrix]:
        """Blocks of matrix in top-to-bottom order."""
        return self._blocks

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return np.concatenate([block @ other for block in self._blocks], axis=0)

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        if other.shape[-1] != self.shape[0]:
            msg = "Inconsistent inner dimension for matrix multiply"
            raise ValueError(msg)
        return sum(
            [
                part @ block
                for block, part in zip(
                    self._blocks,
                    np.split(other, self._splits, axis=-1),
                    strict=True,
                )
            ],
        )

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        return BlockColumnMatrix(tuple(scalar * block for block in self._blocks))

    def _construct_array(self: NDArray) -> NDArray:
        return np.concatenate([block.array for block in self._blocks], axis=0)

    def _construct_transpose(self) -> BlockRowMatrix:
        return BlockRowMatrix(tuple(block.T for block in self._blocks))


class SquareLowRankUpdateMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Square matrix equal to a low-rank update to a square matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            square_matrix + sign * left_factor_matrix @ inner_square_matrix @
            right_factor_matrix)

    where `left_factor_matrix` and `right_factor_matrix` are rectangular with shapes
    `(dim_outer, dim_inner)` and `(dim_inner, dim_outer)` resp., `square_matrix` is
    square with shape `(dim_outer, dim_outer)`, `inner_square_matrix` is square with
    shape `(dim_inner, dim_inner)` and `sign` is one of {-1, +1} and determines whether
    a low-rank update (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the inverse
    and determinant of the matrix can be computed at a cost of `O(dim_inner**3 +
    dim_inner**2 * dim_outer)` plus the cost of inverting / evaluating the determinant
    of `square_matrix`, which for `square_matrix` instances with special structure such
    as diagonality or with an existing factorisation, will typically be cheaper than the
    `O(dim_outer**3)` cost of evaluating the inverse or determinant directly.
    """

    def __init__(
        self,
        left_factor_matrix: Matrix,
        right_factor_matrix: Matrix,
        square_matrix: SquareMatrix,
        inner_square_matrix: SquareMatrix | None = None,
        capacitance_matrix: SquareMatrix | None = None,
        sign: Literal[-1, 1] = 1,
    ) -> None:
        """
        Args:
            left_factor_matrix: Rectangular matrix with shape `(dim_outer, dim_inner)`
                forming leftmost term in matrix product defining low-rank update.
            right_factor_matrix: Rectangular matrix with shape `(dim_inner, dim_outer)`
                forming rightmost term in matrix product defining low-rank update.
            square_matrix: Square matrix to perform low-rank update (or downdate) to.
            inner_square_matrix: Optional square matrix with shape
                `(dim_inner, dim_inner)` specifying inner term in matrix product
                defining low-rank update. If `None` an identity matrix is used.
            capacitance_matrix: Square matrix equal to.

                    inner_square_matrix.inv
                    + right_factor_matrix @ square_matrix.inv @ left_factor_matrix

                and with shape `(dim_inner, dim_inner)` which is used in constructing
                inverse and computation of determinant of the low-rank updated matrix,
                with this argument optional and typically only passed when this matrix
                has already been computed in a previous computation.
            sign: One of {-1, +1}, determining whether a low-rank update (`sign = 1`) or
                'downdate' (`sign = -1`) is peformed.
        """
        dim_outer, dim_inner = left_factor_matrix.shape
        if square_matrix.shape[0] != dim_outer:
            msg = (
                f"Inconsistent factor and square matrix shapes: outer dimensions "
                f"{dim_outer} and {square_matrix.shape[0]}."
            )
            raise ValueError(msg)
        if square_matrix.shape[0] != square_matrix.shape[1]:
            msg = "square_matrix argument must be square"
            raise ValueError(msg)
        if not isinstance(left_factor_matrix, Matrix):
            left_factor_matrix = DenseRectangularMatrix(left_factor_matrix)
        if not isinstance(right_factor_matrix, Matrix):
            right_factor_matrix = DenseRectangularMatrix(right_factor_matrix)
        if right_factor_matrix.shape != (dim_inner, dim_outer):
            msg = (
                f"Inconsistent factor matrix shapes: {left_factor_matrix.shape} and "
                f"{right_factor_matrix.shape}."
            )
            raise ValueError(msg)
        if inner_square_matrix is None:
            inner_square_matrix = IdentityMatrix(dim_inner)
        elif inner_square_matrix.shape != (dim_inner, dim_inner):
            msg = (
                f"inner_square matrix must be square and of shape "
                f"{dim_inner, dim_inner}."
            )
            raise ValueError(msg)
        self.left_factor_matrix = left_factor_matrix
        self.right_factor_matrix = right_factor_matrix
        self.square_matrix = square_matrix
        self.inner_square_matrix = inner_square_matrix
        self._capacitance_matrix = capacitance_matrix
        self._sign = sign
        super().__init__((dim_outer, dim_outer))

    def _left_matrix_multiply(self, other: NDArray) -> NDArray:
        return self.square_matrix @ other + (
            self._sign
            * self.left_factor_matrix
            @ (self.inner_square_matrix @ (self.right_factor_matrix @ other))
        )

    def _right_matrix_multiply(self, other: NDArray) -> NDArray:
        return (
            other @ self.square_matrix
            + (
                self._sign
                * (other @ self.left_factor_matrix)
                @ self.inner_square_matrix
            )
            @ self.right_factor_matrix
        )

    def _scalar_multiply(self, scalar: ScalarLike) -> NDArray:
        return type(self)(
            self.left_factor_matrix,
            self.right_factor_matrix,
            scalar * self.square_matrix,
            scalar * self.inner_square_matrix,
            (
                self._capacitance_matrix / scalar
                if self._capacitance_matrix is not None
                else None
            ),
            self._sign,
        )

    def _construct_array(self) -> NDArray:
        return self.square_matrix.array + (
            self._sign
            * self.left_factor_matrix
            @ (self.inner_square_matrix @ self.right_factor_matrix.array)
        )

    @property
    def capacitance_matrix(self) -> DenseSquareMatrix:
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DenseSquareMatrix(
                self.inner_square_matrix.inv.array
                + self.right_factor_matrix
                @ (self.square_matrix.inv @ self.left_factor_matrix.array),
            )
        return self._capacitance_matrix

    @property
    def diagonal(self) -> NDArray:
        return self.square_matrix.diagonal + self._sign * (
            (self.left_factor_matrix.array @ self.inner_square_matrix)
            * self.right_factor_matrix.T.array
        ).sum(1)

    def _construct_transpose(self) -> SquareLowRankUpdateMatrix:
        return type(self)(
            self.right_factor_matrix.T,
            self.left_factor_matrix.T,
            self.square_matrix.T,
            self.inner_square_matrix.T,
            (
                self._capacitance_matrix.T
                if self._capacitance_matrix is not None
                else None
            ),
            self._sign,
        )

    def _construct_inv(self) -> SquareLowRankUpdateMatrix:
        return type(self)(
            self.square_matrix.inv @ self.left_factor_matrix,
            self.right_factor_matrix @ self.square_matrix.inv,
            self.square_matrix.inv,
            self.capacitance_matrix.inv,
            self.inner_square_matrix.inv,
            -self._sign,
        )

    @property
    def log_abs_det(self) -> float:
        return (
            self.square_matrix.log_abs_det
            + self.inner_square_matrix.log_abs_det
            + self.capacitance_matrix.log_abs_det
        )

    def _compute_hash(self) -> int:
        return hash(
            (
                self.left_factor_matrix,
                self.right_factor_matrix,
                self.square_matrix,
                self.inner_square_matrix,
            ),
        )

    def _check_equality(self, other: SquareLowRankUpdateMatrix) -> bool:
        return (
            self.left_factor_matrix == other.left_factor_matrix
            and self.right_factor_matrix == other.right_factor_matrix
            and self.square_matrix == other.square_matrix
            and self.inner_square_matrix == other.inner_square_matrix
        )


class SymmetricLowRankUpdateMatrix(
    SquareLowRankUpdateMatrix,
    SymmetricMatrix,
    InvertibleMatrix,
):
    """Symmetric matrix equal to a low-rank update to a symmetric matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            symmetric_matrix + sign * factor_matrix @ inner_symmetric_matrix @
            factor_matrix.T)

    where `factor_matrix` is rectangular with shape `(dim_outer, dim_inner)`,
    `symmetric_matrix` is symmetric with shape `(dim_outer, dim_outer)`,
    `inner_symmetric_matrix` is symmetric with shape `(dim_inner, dim_inner)` and `sign`
    is one of {-1, +1} and determines whether a low-rank update (`sign = 1`) or
    'downdate' (`sign = -1`) is peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the inverse
    and determinant of the matrix can be computed at a cost of `O(dim_inner**3 +
    dim_inner**2 * dim_outer)` plus the cost of inverting / evaluating the determinant
    of `square_matrix`, which for `square_matrix` instances with special structure such
    as diagonality or with an existing factorisation, will typically be cheaper than the
    `O(dim_outer**3)` cost of evaluating the inverse or determinant directly.
    """

    def __init__(
        self,
        factor_matrix: Matrix,
        symmetric_matrix: SymmetricMatrix,
        inner_symmetric_matrix: SymmetricMatrix | None = None,
        capacitance_matrix: SymmetricMatrix | None = None,
        sign: Literal[-1, 1] = 1,
    ) -> None:
        """
        Args:
            factor_matrix (Matrix): Rectangular matrix with shape
                `(dim_outer, dim_inner)` with it and its transpose forming the
                leftmost and righmost term respectively in the matrix product
                defining the low-rank update.
            symmetric_matrix: Symmetric matrix to perform low-rank update (or downdate)
                to.
            inner_symmetric_matrix: Optional symmetric matrix with shape
                `(dim_inner, dim_inner)` specifying inner term in matrix product
                defining low-rank update. If `None` an identity matrix is used.
            capacitance_matrix: Symmetric matrix  equal to.

                    inner_symmetric_matrix.inv
                    + factor_matrix.T @  symmetric_matrix.inv @ factor_matrix

                and with shape `(dim_inner, dim_inner)` which is used in constructing
                inverse and computation of determinant of the low-rank updated matrix,
                with this argument optional and typically only passed when this matrix
                has already been computed in a previous computation.
            sign: One of {-1, +1}, determining whether a low-rank update (`sign = 1`) or
                'downdate' (`sign = -1`) is peformed.
        """
        dim_inner = factor_matrix.shape[1]
        if symmetric_matrix.T is not symmetric_matrix:
            msg = "symmetric_matrix must be symmetric"
            raise ValueError(msg)
        if inner_symmetric_matrix is None:
            inner_symmetric_matrix = IdentityMatrix(dim_inner)
        if inner_symmetric_matrix.T is not inner_symmetric_matrix:
            msg = "inner_symmetric_matrix must be symmetric"
            raise ValueError(msg)
        if not isinstance(factor_matrix, Matrix):
            factor_matrix = DenseRectangularMatrix(factor_matrix)
        self.factor_matrix = factor_matrix
        self.symmetric_matrix = symmetric_matrix
        self.inner_symmetric_matrix = inner_symmetric_matrix
        super().__init__(
            left_factor_matrix=factor_matrix,
            right_factor_matrix=factor_matrix.T,
            square_matrix=symmetric_matrix,
            inner_square_matrix=inner_symmetric_matrix,
            capacitance_matrix=capacitance_matrix,
            sign=sign,
        )

    def _scalar_multiply(self, scalar: ScalarLike) -> SymmetricLowRankUpdateMatrix:
        return type(self)(
            self.factor_matrix,
            scalar * self.symmetric_matrix,
            scalar * self.inner_symmetric_matrix,
            (
                self._capacitance_matrix / scalar
                if self._capacitance_matrix is not None
                else None
            ),
            self._sign,
        )

    @property
    def capacitance_matrix(self) -> DenseSymmetricMatrix:
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DenseSymmetricMatrix(
                self.inner_symmetric_matrix.inv.array
                + self.factor_matrix.T
                @ (self.symmetric_matrix.inv @ self.factor_matrix.array),
            )
        return self._capacitance_matrix

    def _construct_inv(self) -> SymmetricLowRankUpdateMatrix:
        return type(self)(
            self.symmetric_matrix.inv @ self.factor_matrix,
            self.symmetric_matrix.inv,
            self.capacitance_matrix.inv,
            self.inner_symmetric_matrix.inv,
            -self._sign,
        )

    def _construct_transpose(self) -> SymmetricLowRankUpdateMatrix:
        return self

    def _compute_hash(self) -> int:
        return hash((self.factor_matrix, self.square_matrix, self.inner_square_matrix))

    def _check_equality(self, other: SymmetricLowRankUpdateMatrix) -> bool:
        return (
            self.factor_matrix == other.factor_matrix
            and self.symmetric_matrix == other.symmetric_matrix
            and self.inner_symmetric_matrix == other.inner_symmetric_matrix
        )


class PositiveDefiniteLowRankUpdateMatrix(
    SymmetricLowRankUpdateMatrix,
    PositiveDefiniteMatrix,
    DifferentiableMatrix,
):
    """Positive-definite matrix equal to low-rank update to a square matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            pos_def_matrix + sign * factor_matrix @ inner_pos_def_matrix @
            factor_matrix.T)

    where `factor_matrix` is rectangular with shape `(dim_outer, dim_inner)`,
    `pos_def_matrix` is positive-definite with shape `(dim_outer, dim_outer)`,
    `inner_pos_def_matrix` is positive-definite with shape `(dim_inner, dim_inner)` and
    `sign` is one of {-1, +1} and determines whether a low-rank update (`sign = 1`) or
    'downdate' (`sign = -1`) is peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the inverse,
    determinant and square-root of the matrix can all be computed at a cost of
    `O(dim_inner**3 + dim_inner**2 * dim_outer)` plus the cost of inverting / evaluating
    the determinant / square_root of `pos_def_matrix`, which for `pos_def_matrix`
    instances with special structure such as diagonality or with an existing
    factorisation, will typically be cheaper than the `O(dim_outer**3)` cost of
    evaluating the inverse, determinant or square-root directly.
    """

    def __init__(
        self,
        factor_matrix: Matrix,
        pos_def_matrix: PositiveDefiniteMatrix,
        inner_pos_def_matrix: PositiveDefiniteMatrix | None = None,
        capacitance_matrix: PositiveDefiniteMatrix | None = None,
        sign: Literal[-1, 1] = 1,
    ) -> None:
        """
        Args:
            factor_matrix: Rectangular matrix with shape `(dim_outer, dim_inner)` with
                it and its transpose forming the leftmost and righmost term respectively
                in the matrix product defining the low-rank update.
            pos_def_matrix: Positive-definite matrix to perform low-rank update (or
                downdate) to.
            inner_pos_def_matrix: Optional positive definite matrix with shape
                `(dim_inner, dim_inner)` specifying inner term in matrix product
                defining low-rank update. If `None` an identity matrix is used.
            capacitance_matrix: Positive-definite matrix equal to.

                    inner_pos_def_matrix.inv
                    + factor_matrix.T @ pos_def_matrix.inv @ factor_matrix

                and with shape `(dim_inner, dim_inner)` which is used in constructing
                inverse and computation of determinant of the low-rank updated matrix,
                with this argument optional and typically only passed when this matrix
                has already been computed in a previous computation.
            sign: One of {-1, +1}, determining whether a low-rank update (`sign = 1`) or
                'downdate' (`sign = -1`) is peformed.
        """
        dim_inner = factor_matrix.shape[1]
        if not isinstance(factor_matrix, Matrix):
            factor_matrix = DenseRectangularMatrix(factor_matrix)
        self.factor_matrix = factor_matrix
        self.pos_def_matrix = pos_def_matrix
        if inner_pos_def_matrix is None:
            inner_pos_def_matrix = IdentityMatrix(dim_inner)
        self.inner_pos_def_matrix = inner_pos_def_matrix
        super().__init__(
            factor_matrix=factor_matrix,
            symmetric_matrix=pos_def_matrix,
            inner_symmetric_matrix=inner_pos_def_matrix,
            capacitance_matrix=capacitance_matrix,
            sign=sign,
        )

    def _scalar_multiply(self, scalar: ScalarLike) -> SymmetricLowRankUpdateMatrix:
        if scalar > 0:
            return PositiveDefiniteLowRankUpdateMatrix(
                self.factor_matrix,
                scalar * self.pos_def_matrix,
                scalar * self.inner_pos_def_matrix,
                (
                    self._capacitance_matrix / scalar
                    if self._capacitance_matrix is not None
                    else None
                ),
                self._sign,
            )
        return SymmetricLowRankUpdateMatrix(
            self.factor_matrix,
            scalar * self.pos_def_matrix,
            scalar * self.inner_pos_def_matrix,
            (
                self._capacitance_matrix / scalar
                if self._capacitance_matrix is not None
                else None
            ),
            self._sign,
        )

    @property
    def capacitance_matrix(self) -> DensePositiveDefiniteMatrix:
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DensePositiveDefiniteMatrix(
                self.inner_pos_def_matrix.inv.array
                + self.factor_matrix.T
                @ (self.pos_def_matrix.inv @ self.factor_matrix.array),
            )
        return self._capacitance_matrix

    def _construct_sqrt(self) -> MatrixProduct:
        # Uses O(dim_inner**3 + dim_inner**2 * dim_outer) cost implementation proposed
        # in
        #   Ambikasaran, O'Neill & Singh (2016). Fast symmetric factorization
        #   of hierarchical matrices with applications. arxiv:1405.0223.
        # Variable naming below follows notation in Algorithm 1 in paper
        w_matrix = self.pos_def_matrix.sqrt
        k_matrix = self.inner_pos_def_matrix
        u_matrix = w_matrix.inv @ self.factor_matrix
        l_matrix = TriangularMatrix(
            nla.cholesky(u_matrix.T @ u_matrix.array),
            lower=True,
            make_triangular=False,
        )
        i_outer, i_inner = (
            IdentityMatrix(u_matrix.shape[0]),
            np.identity(
                u_matrix.shape[1],
            ),
        )
        m_matrix = sla.sqrtm(i_inner + l_matrix.T @ (k_matrix @ l_matrix.array))
        x_matrix = DenseSymmetricMatrix(
            l_matrix.inv.T @ ((m_matrix - i_inner) @ l_matrix.inv),
        )
        return w_matrix @ SymmetricLowRankUpdateMatrix(u_matrix, i_outer, x_matrix)

    @property
    def grad_log_abs_det(self) -> NDArray:
        return 2 * (self.inv @ (self.factor_matrix.array @ self.inner_pos_def_matrix))

    def grad_quadratic_form_inv(self, vector: NDArray) -> NDArray:
        inv_matrix_vector = self.inv @ vector
        return -2 * np.outer(
            inv_matrix_vector,
            self.inner_pos_def_matrix @ (self.factor_matrix.T @ inv_matrix_vector),
        )
