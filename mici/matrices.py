"""Structured matrix classes implementing basic linear algebra operations."""

import abc
import numbers
import numpy as np
from mici.errors import LinAlgError
import numpy.linalg as nla
import scipy.linalg as sla
from mici.utils import hash_array


def _choose_matrix_product_class(matrix_l, matrix_r):
    if (matrix_l.shape[0] == matrix_l.shape[1] and
            matrix_r.shape == matrix_l.shape):
        if (isinstance(matrix_l, InvertibleMatrix) and
                isinstance(matrix_r, InvertibleMatrix)):
            return InvertibleMatrixProduct
        else:
            return SquareMatrixProduct
    else:
        return MatrixProduct


def _is_scalar(val):
    return (
        isinstance(val, numbers.Number) or
        (hasattr(val, '__array__') and np.ndim(val) == 0))


class Matrix(abc.ABC):
    """Base class for matrix-like objects.

    Implements overloads of the matrix multiplication operator `@`, as well as
    the standard multiplication and division operators `*` and `/` when the
    second argument is a scalar quantity.
    """

    __array_priority__ = 1

    def __init__(self, shape, **kwargs):
        """
        Args:
           shape (Tuple[int, int]): Shape of matrix `(num_rows, num_columns)`.
        """
        self._shape = shape
        self._hash = None
        self._transpose = None
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
            self.__dict__[k] = v

    def __array__(self):
        return self.array

    def __mul__(self, other):
        if _is_scalar(other):
            if other == 0:
                raise NotImplementedError(
                    'Scalar multiplication by zero not implemented.')
            return self._scalar_multiply(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if _is_scalar(other):
            if other == 0:
                raise NotImplementedError(
                    'Scalar division by zero not implemented.')
            return self._scalar_multiply(1 / other)
        else:
            return NotImplemented

    def __neg__(self):
        return self._scalar_multiply(-1)

    def __matmul__(self, other):
        if self.shape[1] is not None and other.shape[0] != self.shape[1]:
            raise ValueError(
                f'Inconsistent dimensions for matrix multiplication: '
                f'{self.shape} and {other.shape}.')
        if isinstance(other, Matrix):
            matrix_product_class = _choose_matrix_product_class(self, other)
            return matrix_product_class((self, other), check_shapes=False)
        else:
            return self._left_matrix_multiply(other)

    def __rmatmul__(self, other):
        if self.shape[0] is not None and other.shape[-1] != self.shape[0]:
            raise ValueError(
                f'Inconsistent dimensions for matrix multiplication: '
                f'{other.shape} and {self.shape}.')
        if isinstance(other, Matrix):
            matrix_product_class = _choose_matrix_product_class(self, other)
            return matrix_product_class((other, self), check_shapes=False)
        else:
            return self._right_matrix_multiply(other)

    @property
    def shape(self):
        """Shape of matrix as a tuple `(num_rows, num_columns)`."""
        return self._shape

    @property
    @abc.abstractmethod
    def array(self):
        """Full dense representation of matrix as a 2D array."""

    @abc.abstractmethod
    def _left_matrix_multiply(self, other):
        """Left multiply argument by the represented matrix.

        Args:
            other (array): Argument to left-multiply.

        Returns:
            result (array): Result of left-multiplying `other` by the
                represented matrix.
        """

    @abc.abstractmethod
    def _right_matrix_multiply(self, other):
        """Right multiply argument by the represented matrix.

        Args:
            other (array): Argument to right-multiply.

        Returns:
            result (array): Result of right-multiplying `other` by the
                represented matrix.
        """

    @abc.abstractmethod
    def _scalar_multiply(self, scalar):
        """Calculate result of multiplying represented matrix by a scalar.

        Args:
            scalar (float): Scalar argument to multiply by.

        Returns:
            result (Matrix): Result of multiplying represented matrix by
                `scalar` as another `Matrix` object.
        """

    @property
    def transpose(self):
        """Transpose of matrix."""
        if self._transpose is None:
            self._transpose = self._construct_transpose()
        return self._transpose

    T = transpose

    @abc.abstractmethod
    def _construct_transpose(self):
        """Construct transpose of matrix."""

    @property
    def diagonal(self):
        """Diagonal of matrix as a 1D array."""
        return self.array.diagonal()

    def __str__(self):
        return f'(shape={self.shape})'

    def __repr__(self):
        return type(self).__name__ + str(self)


    @abc.abstractmethod
    def _compute_hash(self):
        """Compute hash value for matrix object."""

    def __hash__(self):
        if self._hash is None:
            self._hash = self._compute_hash()
        return self._hash

    @abc.abstractmethod
    def _check_equality(self, other):
        """Check for equality with another instance of the same class."""

    def __eq__(self, other):
        return other is self or (
            other.__class__ == self.__class__ and self._check_equality(other))


class ExplicitArrayMatrix(Matrix):
    """Matrix with an explicit array representation."""

    def __init__(self, shape, **kwargs):
        if '_array' not in kwargs:
            raise ValueError('_array must be specified in kwargs')
        else:
            kwargs['_array'] = np.asarray_chkfinite(kwargs['_array'])
        super().__init__(shape, **kwargs)

    @property
    def array(self):
        return self._array

    def _left_matrix_multiply(self, other):
        return self._array @ other

    def _right_matrix_multiply(self, other):
        return other @ self._array

    def _compute_hash(self):
        return hash_array(self._array)

    def _check_equality(self, other):
        return np.array_equal(self.array, other.array)


class ImplicitArrayMatrix(Matrix):
    """Matrix with an implicit array representation."""

    def __init__(self, shape, **kwargs):
        """
        Args:
           shape (Tuple[int, int]): Shape of matrix `(num_rows, num_columns)`.
        """
        super().__init__(shape, **kwargs)
        self._array = None

    @property
    def array(self):
        """Full dense representation of matrix as a 2D array.

        Generally accessing this property should be avoided wherever possible
        as the resulting array object may use a lot of memory and operations
        with it will not be able to exploit any structure in the matrix.
        """
        if self._array is None:
            self._array = self._construct_array()
        return self._array

    @abc.abstractmethod
    def _construct_array(self):
        """Construct full dense representation of matrix as a 2D array.

        Generally calling this method should be avoided wherever possible as
        the returned array object may use a lot of memory and operations with
        it will not be able to exploit any structure in the matrix.
        """


class MatrixProduct(ImplicitArrayMatrix):
    """Matrix implicitly defined as a product of a sequence of matrices.

    Each adjacent pair of matrices in the sequence must have compatible shapes.
    """

    def __init__(self, matrices, check_shapes=True):
        """
        Args:
            matrices(Iterable[Matrix]): Sequence of matrices forming product in
                left-to-right order.
            check_shapes (boolean): Whether to check if all successive pairs of
                the matrix sequence have compatible shapes, i.e. equal inner
                dimensions.
        """
        self._matrices = tuple(matrices)
        if check_shapes:
            for matrix_l, matrix_r in zip(matrices[:-1], matrices[1:]):
                if matrix_l.shape[1] != matrix_r.shape[0]:
                    raise ValueError(
                        f'Matrices {matrix_l} and {matrix_r} have inconsistent'
                        f' inner dimensions for forming a matrix product.')
        super().__init__((self._matrices[0].shape[0],
                          self._matrices[-1].shape[1]))

    @property
    def matrices(self):
        return self._matrices

    def _scalar_multiply(self, scalar):
        return type(self)((
            ScaledIdentityMatrix(scalar, self.shape[0]), *self.matrices))

    def _left_matrix_multiply(self, other):
        for matrix in reversed(self.matrices):
            other = matrix @ other
        return other

    def _right_matrix_multiply(self, other):
        for matrix in self.matrices:
            other = other @ matrix
        return other

    def _construct_transpose(self):
        return type(self)(
            tuple(matrix.T for matrix in reversed(self.matrices)))

    def _construct_array(self):
        return self.matrices[0].array @ MatrixProduct(self.matrices[1:])

    def _compute_hash(self):
        return hash(tuple(matrix for matrix in self.matrices))

    def _check_equality(self, other):
        return (
            len(other.matrices) == len(self.matrices) and
            all(matrix_s == matrix_o for matrix_s, matrix_o
                in zip(self.matrices, other.matrices))
        )


class SquareMatrix(Matrix):
    """Base class for matrices with equal numbers of rows and columns."""

    def __init__(self, shape, **kwargs):
        if shape[0] != shape[1]:
            raise ValueError(
                f'{shape} is not a valid shape for a square matrix.')
        super().__init__(shape, **kwargs)

    @property
    @abc.abstractmethod
    def log_abs_det(self):
        """Logarithm of absolute value of determinant of matrix.

        For matrix representations of metrics it is proportional to the
        logarithm of the density of then Riemannian measure associated with
        metric with respect to the Lebesgue measure.
        """


class SquareMatrixProduct(MatrixProduct, SquareMatrix):
    """Matrix implicitly defined as a product of a sequence of square matrices.

    All the matrices must have the same shape.
    """

    def __init__(self, matrices, check_shapes=True):
        matrices = tuple(matrices)
        if check_shapes:
            if matrices[0].shape[0] != matrices[0].shape[1]:
                raise ValueError(f'{matrices[0]} is not square.')
            for matrix in matrices[1:]:
                if matrix.shape != matrices[0].shape:
                    raise ValueError(
                        f'{matrices[0]} and {matrix} have different shapes.')
        super().__init__(matrices, check_shapes=False)

    @property
    def log_abs_det(self):
        return sum(matrix.log_abs_det for matrix in self.matrices)


class InvertibleMatrix(SquareMatrix):
    """Base class for non-singular square matrices."""


    def __init__(self, shape, **kwargs):
        super().__init__(shape, **kwargs)
        self._inv = None

    @property
    def inv(self):
        """Inverse of matrix as a `Matrix` object.

        This will not necessarily form an explicit representation of the
        inverse matrix but may instead return a `Matrix` object that implements
        the matrix multiplication operators by solving the linear system
        defined by the original matrix object.
        """
        if self._inv is None:
            self._inv = self._construct_inv()
        return self._inv

    @abc.abstractmethod
    def _construct_inv(self):
        """Construct inverse of matrix as a `Matrix` object.

        This will not necessarily form an explicit representation of the
        inverse matrix but may instead return a `Matrix` object that implements
        the matrix multiplication operators by solving the linear system
        defined by the original matrix object.
        """


class InvertibleMatrixProduct(SquareMatrixProduct, InvertibleMatrix):
    """Matrix defined as a product of a sequence of invertible matrices.

    All the matrices must have the same shape.
    """

    def __init__(self, matrices, check_shapes=True):
        matrices = tuple(matrices)
        for matrix in matrices:
            if not isinstance(matrix, InvertibleMatrix):
                raise ValueError(f'matrix {matrix} is not invertible.')
        super().__init__(matrices, check_shapes)

    def _construct_inv(self):
        return InvertibleMatrixProduct(
            tuple(matrix.inv for matrix in reversed(self.matrices)))


class SymmetricMatrix(SquareMatrix):
    """Base class for square matrices which are equal to their transpose."""

    def __init__(self, shape, **kwargs):
        self._eigval = None
        self._eigvec = None
        super().__init__(shape, **kwargs)

    def _compute_eigendecomposition(self):
        self._eigval, eigvec = nla.eigh(self.array)
        self._eigvec = OrthogonalMatrix(eigvec)

    @property
    def eigval(self):
        """Eigenvalues of matrix as a 1D array."""
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigval

    @property
    def eigvec(self):
        """Eigenvectors of matrix stacked as columns of a `Matrix` object."""
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigvec

    def _construct_transpose(self):
        return self

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.eigval)).sum()


class PositiveDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):
    """Base class for positive definite matrices."""

    def __init__(self, shape, **kwargs):
        self._sqrt = None
        super().__init__(shape, **kwargs)

    @property
    def sqrt(self):
        """Square-root of matrix satisfying `matrix == sqrt @ sqrt.T`.

        This will in general not correspond to the unique, if defined,
        symmetric square root of a symmetric matrix but instead may return any
        matrix satisfying the above property.
        """
        if self._sqrt is None:
            self._sqrt = self._construct_sqrt()
        return self._sqrt

    @abc.abstractmethod
    def _construct_sqrt(self):
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

    def __init__(self, size=None):
        """
        Args:
            size (int or None): Number of rows / columns in matrix or `None` if
                matrix is to be implicitly shaped.
        """
        super().__init__((size, size))

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveScaledIdentityMatrix(
                scalar, self.shape[0])
        else:
            return ScaledIdentityMatrix(scalar, self.shape[0])

    def _left_matrix_multiply(self, other):
        return other

    def _right_matrix_multiply(self, other):
        return other

    @property
    def eigval(self):
        return self.diagonal

    def _construct_sqrt(self):
        return self

    @property
    def eigvec(self):
        return self

    def _construct_inv(self):
        return self

    @property
    def diagonal(self):
        return np.ones(self.shape[0])

    def _construct_array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for identity matrix with '
                'implicit size.')
        return np.identity(self.shape[0])

    @property
    def log_abs_det(self):
        return 0.

    def _compute_hash(self):
        return hash(self.shape)

    def _check_equality(self, other):
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
    def grad_log_abs_det(self):
        """Gradient of logarithm of absolute value of determinant of matrix."""

    @abc.abstractmethod
    def grad_quadratic_form_inv(self, vector):
        """Gradient of quadratic form `vector @ matrix.inv @ vector`.

        Args:
            vector (array): 1D array representing vector to evaluate quadratic
                form at.
        """


class ScaledIdentityMatrix(
        SymmetricMatrix, DifferentiableMatrix, ImplicitArrayMatrix):
    """Matrix representing scalar multiplication operation on a vector space.

    Array representation has common scalar on diagonal elements and zeros
    elsewhere. May be defined with an implicit shape reprsented by
    `(None, None)` which will allow use for subset of operations where shape
    is not required to be known.
    """

    def __init__(self, scalar, size=None):
        """
        Args:
            scalar (float): Scalar multiplier for identity matrix.
            size (int): Number of rows / columns in matrix. If `None` the
                matrix will be implicitly-shaped and only the subset of
                operations which do not rely on an explicit shape will be
                available.
        """
        if scalar == 0:
            raise ValueError('scalar must be non-zero')
        self._scalar = scalar
        super().__init__((size, size))

    @property
    def scalar(self):
        """Scalar multiplier."""
        return self._scalar

    def _scalar_multiply(self, scalar):
        return ScaledIdentityMatrix(scalar * self._scalar, self.shape[0])

    def _left_matrix_multiply(self, other):
        return self._scalar * other

    def _right_matrix_multiply(self, other):
        return self._scalar * other

    @property
    def eigval(self):
        return self.diagonal

    @property
    def eigvec(self):
        return IdentityMatrix(self.shape[0])

    def _construct_inv(self):
        return ScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    @property
    def diagonal(self):
        return self._scalar * np.ones(self.shape[0])

    def _construct_array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for scaled identity matrix '
                'with implicit size.')
        return self._scalar * np.identity(self.shape[0])

    @property
    def log_abs_det(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get log determinant for scaled identity matrix with '
                'implicit size.')
        return self.shape[0] * np.log(abs(self._scalar))

    @property
    def grad_log_abs_det(self):
        return self.shape[0] / self._scalar

    def grad_quadratic_form_inv(self, vector):
        return -np.sum(vector**2) / self._scalar**2

    def __str__(self):
        return f'(shape={self.shape}, scalar={self._scalar})'

    def _compute_hash(self):
        return hash((self.shape, self.scalar))

    def _check_equality(self, other):
        return self.shape == other.shape and self.scalar == other.scalar


class PositiveScaledIdentityMatrix(
        ScaledIdentityMatrix, PositiveDefiniteMatrix):
    """Specialisation of `ScaledIdentityMatrix` with positive scalar parameter.

    Restricts the `scalar` parameter to be strictly positive.
    """

    def __init__(self, scalar, size=None):
        if scalar <= 0:
            raise ValueError('Scalar multiplier must be positive.')
        super().__init__(scalar, size)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveScaledIdentityMatrix(
                scalar * self._scalar, self.shape[0])
        else:
            return super()._scalar_multiply(scalar)

    def _construct_inv(self):
        return PositiveScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    def _construct_sqrt(self):
        return PositiveScaledIdentityMatrix(self._scalar**0.5, self.shape[0])


class DiagonalMatrix(
        SymmetricMatrix, DifferentiableMatrix, ImplicitArrayMatrix):
    """Matrix with non-zero elements only along its diagonal."""

    def __init__(self, diagonal):
        """
        Args:
            diagonal (array): 1D array specifying diagonal elements of matrix.
        """
        if diagonal.ndim != 1:
            raise ValueError('Specified diagonal must be a 1D array.')
        super().__init__((diagonal.size, diagonal.size), _diagonal=diagonal)

    @property
    def diagonal(self):
        return self._diagonal

    def _scalar_multiply(self, scalar):
        return DiagonalMatrix(self.diagonal * scalar)

    def _left_matrix_multiply(self, other):
        if other.ndim == 2:
            return self.diagonal[:, None] * other
        elif other.ndim == 1:
            return self.diagonal * other
        else:
            raise ValueError(
                'Left matrix multiplication only defined for one or two '
                'dimensional right hand sides.')

    def _right_matrix_multiply(self, other):
        return self.diagonal * other

    @property
    def eigvec(self):
        return IdentityMatrix(self.shape[0])

    @property
    def eigval(self):
        return self.diagonal

    def _construct_inv(self):
        return DiagonalMatrix(1. / self.diagonal)

    def _construct_array(self):
        return np.diag(self.diagonal)

    @property
    def grad_log_abs_det(self):
        return 1. / self.diagonal

    def grad_quadratic_form_inv(self, vector):
        return -(self.inv @ vector)**2

    def _compute_hash(self):
        return hash_array(self.diagonal)

    def _check_equality(self, other):
        return np.array_equal(self.diagonal, other.diagonal)


class PositiveDiagonalMatrix(DiagonalMatrix, PositiveDefiniteMatrix):
    """Specialisation of `DiagonalMatrix` with positive diagonal parameter.

    Restricts all values in `diagonal` array parameter to be strictly positive.
    """

    def __init__(self, diagonal):
        if not np.all(diagonal > 0):
            raise ValueError('Diagonal values must all be positive.')
        super().__init__(diagonal)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveDiagonalMatrix(self.diagonal * scalar)
        else:
            return super()._scalar_multiply(scalar)

    def _construct_inv(self):
        return PositiveDiagonalMatrix(1. / self.diagonal)

    def _construct_sqrt(self):
        return PositiveDiagonalMatrix(self.diagonal**0.5)


def _make_array_triangular(array, lower):
    """Make array lower/upper triangular by zeroing above/below diagonal."""
    return np.tril(array) if lower else np.triu(array)


class TriangularMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Matrix with non-zero values only in lower or upper triangle elements."""

    def __init__(self, array, lower=True, make_triangular=True):
        """
        Args:
            array (array): 2D array containing lower / upper triangular element
                values of matrix. Any values above (below) diagonal are
                ignored for lower (upper) triangular matrices i.e. when
                `lower == True` (`lower == False`).
            lower (bool): Whether the matrix is lower-triangular (`True`) or
                upper-triangular (`False`).
            make_triangular (bool): Whether to ensure `array` is triangular
                by explicitly zeroing entries in upper triangle if
                `lower == True` and in lower triangle if `lower == False`.
        """
        array = (
            _make_array_triangular(array, lower) if make_triangular else array)
        super().__init__(array.shape, _array=array)
        self._lower = lower

    def _scalar_multiply(self, scalar):
        return TriangularMatrix(
            self.array * scalar, self.lower, make_triangular=False)

    @property
    def lower(self):
        return self._lower

    def _construct_inv(self):
        return InverseTriangularMatrix(
            self.array, lower=self.lower, make_triangular=False)

    def _construct_transpose(self):
        return TriangularMatrix(
            self.array.T, lower=not self.lower, make_triangular=False)

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.diagonal)).sum()

    def __str__(self):
        return f'(shape={self.shape}, lower={self.lower})'


class InverseTriangularMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Triangular matrix implicitly specified by its inverse."""

    def __init__(self, inverse_array, lower=True, make_triangular=True):
        """
        Args:
            inverse_array (array): 2D containing values of *inverse* of this
                matrix, with the inverse of a lower (upper) triangular matrix
                being itself lower (upper) triangular. Any values above (below)
                diagonal are ignored for lower (upper) triangular matrices i.e.
                when `lower == True` (`lower == False`).
            lower (bool): Whether the matrix is lower-triangular (`True`) or
                upper-triangular (`False`).
            make_triangular (bool): Whether to ensure `inverse_array` is
                triangular by explicitly zeroing entries in upper triangle if
                `lower == True` and in lower triangle if `lower == False`.
        """
        inverse_array = np.asarray_chkfinite(inverse_array)
        inverse_array = (
            _make_array_triangular(inverse_array, lower) if make_triangular
            else inverse_array)
        super().__init__(inverse_array.shape, _inverse_array=inverse_array)
        self._lower = lower

    def _scalar_multiply(self, scalar):
        return InverseTriangularMatrix(
            self._inverse_array / scalar, self.lower, make_triangular=False)

    def _left_matrix_multiply(self, other):
        return sla.solve_triangular(
            self._inverse_array, other, lower=self.lower, check_finite=False)

    def _right_matrix_multiply(self, other):
        return sla.solve_triangular(
            self._inverse_array, other.T, lower=self.lower, trans=1,
            check_finite=False).T

    @property
    def lower(self):
        return self._lower

    def _construct_inv(self):
        return TriangularMatrix(
            self._inverse_array, lower=self.lower, make_triangular=False)

    def _construct_transpose(self):
        return InverseTriangularMatrix(
            self._inverse_array.T, lower=not self.lower, make_triangular=False)

    def _construct_array(self):
        return sla.solve_triangular(
            self._inverse_array, np.identity(self.shape[0]), lower=self.lower,
            check_finite=False)

    @property
    def diagonal(self):
        return 1. / self._inverse_array.diagonal()

    @property
    def log_abs_det(self):
        return -self.inv.log_abs_det

    def __str__(self):
        return f'(shape={self.shape}, lower={self.lower})'

    def _compute_hash(self):
        return hash_array(self._inverse_array)

    def _check_equality(self, other):
        return np.array_equal(self._inverse_array, other._inverse_array)


class _BaseTriangularFactoredDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):

    def __init__(self, size, sign=1, **kwargs):
        super().__init__((size, size), **kwargs)
        if not (sign == 1 or sign == -1):
            raise ValueError('sign must be equal to +1 or -1')
        self._sign = sign

    @property
    def factor(self):
        """Triangular matrix with `matrix = sign * factor @ factor.T`"""
        return self._factor

    @property
    def sign(self):
        """Signed binary valueith `matrix = sign * factor @ factor.T`"""
        return self._sign

    def _construct_inv(self):
        return TriangularFactoredDefiniteMatrix(
            factor=self.factor.inv.T, sign=self._sign)

    @property
    def log_abs_det(self):
        return 2 * self.factor.log_abs_det

    def __str__(self):
        return f'(shape={self.shape}, sign={self._sign})'


class TriangularFactoredDefiniteMatrix(
        _BaseTriangularFactoredDefiniteMatrix, DifferentiableMatrix,
        ImplicitArrayMatrix):
    """Matrix specified as a signed self-product of a triangular factor.

    The matrix is assumed to have the form

        matrix = sign * factor @ factor.T

    for and upper- or lower-trinagular matrix `factor` and signed binary value
    `sign` (i.e. `sign == +1 or sign == -1`), with the matrix being positive
    definite if `sign == +1` and negative definite if `sign == -1` under the
    assumption that `factor` is non-singular.
    """

    def __init__(self, factor, sign=1, factor_is_lower=None):
        """
        Args:
            factor (array or TriangularMatrix or InverseTriangularMatrix): The
                triangular factor parameterising the matrix. Defined either a
                2D array, in which case only the lower- or upper-triangular
                elements are used depending on the value of the
                `factor_is_lower` boolean keyword argument, or as a
                `TriangularMatrix` / `InverseTriangularMatrix` instance in
                which case `factor_is_lower` is ignored, with `factor.lower`
                instead determining if the factor is lower- or
                upper-triangular.
            sign (int): +/-1 multiplier of factor product, corresponding
                respectively to a strictly positive- or negative-definite
                matrix.
            factor_is_lower (boolean): Whether the array `factor` is lower-
                or upper-triangular.
        """
        if not isinstance(factor, (TriangularMatrix, InverseTriangularMatrix)):
            if factor_is_lower not in (True, False):
                raise ValueError(
                    'For array `factor` parameter `factor_is_lower` must be '
                    'specified as a boolean value.')
            factor = TriangularMatrix(factor, factor_is_lower)
        self._factor = factor
        super().__init__(factor.shape[0], sign=sign)

    def _scalar_multiply(self, scalar):
        return TriangularFactoredDefiniteMatrix(
            factor=abs(scalar)**0.5 * self.factor,
            sign=self.sign * np.sign(scalar))

    def _left_matrix_multiply(self, other):
        return self.sign * (self.factor @ (self.factor.T @ other))

    def _right_matrix_multiply(self, other):
        return self.sign * ((other @ self.factor) @ self.factor.T)

    @property
    def grad_log_abs_det(self):
        return np.diag(2 / self.factor.diagonal)

    def grad_quadratic_form_inv(self, vector):
        inv_factor_vector = self.factor.inv @ vector
        inv_vector = self.inv @ vector
        return _make_array_triangular(
            -2 * self.sign * np.outer(inv_vector, inv_factor_vector),
            self.factor.lower)

    def _construct_array(self):
        return self.sign * (self.factor @ self.factor.array.T)

    def _compute_hash(self):
        return hash((self.factor, self.sign))

    def _check_equality(self, other):
        return self.sign == other.sign and self.factor == other.factor


class TriangularFactoredPositiveDefiniteMatrix(
        TriangularFactoredDefiniteMatrix, PositiveDefiniteMatrix):
    """Positive definite matrix parametrized a triangular matrix product.

    The matrix is assumed to have the parameterisation

        matrix = factor @ factor.T

    where `factor` is an upper- or lower-triangular matrix. Note for the case
    `factor` is lower-triangular this corresponds to the standard Cholesky
    factorisation of a positive definite matrix.
    """

    def __init__(self, factor, factor_is_lower=True):
        """
        Args:
            factor (array or TriangularMatrix or InverseTriangularMatrix): The
                triangular factor parameterising the matrix. Defined either a
                2D array, in which case only the lower- or upper-triangular
                elements are used depending on the value of the
                `factor_is_lower` boolean keyword argument, or as a
                `TriangularMatrix` / `InverseTriangularMatrix` instance in
                which case `factor_is_lower` is ignored, with `factor.lower`
                instead determining if the factor is lower- or
                upper-triangular.
            factor_is_lower (boolean): Whether the array `factor` is lower-
                or upper-triangular.
        """
        super().__init__(factor, sign=1, factor_is_lower=factor_is_lower)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return TriangularFactoredPositiveDefiniteMatrix(
                factor=scalar**0.5 * self.factor)
        else:
            return super()._scalar_multiply(scalar)

    def _construct_inv(self):
        return TriangularFactoredPositiveDefiniteMatrix(
            factor=self.factor.inv.T)

    def _construct_sqrt(self):
        return self.factor


class DenseDefiniteMatrix(_BaseTriangularFactoredDefiniteMatrix,
                          DifferentiableMatrix, ExplicitArrayMatrix):
    """Definite matrix specified by a dense 2D array."""

    def __init__(self, array, factor=None, is_posdef=True):
        """
        Args:
            array (array): 2D array specifying matrix entries.
            factor (None or TriangularMatrix or InverseTriangularMatrix):
                Optional argument giving the triangular factorisation of the
                matrix such that `matrix = factor @ factor.T` if
                `is_posdef=True` or `matrix = -factor @ factor.T` otherwise.
                If not pre-computed and specified at initialisation a
                factorisation will only be computed when first required by
                an operation which depends on the factor.
            is_posdef (boolean): Whether matrix (and so corresponding array
                representation) is positive definite, with the matrix assumed
                to be negative-definite if not. This is **not** checked on
                initialisation, and so if `array` is positive (negative)
                definite and `is_posdef` is `False` (`True`) then a
                `LinAlgError` exception will be if a later attempt is made to
                factorize the matrix.
        """
        super().__init__(
            array.shape[0], sign=1 if is_posdef else -1, _array=array)
        self._factor = factor

    def _scalar_multiply(self, scalar):
        if (scalar > 0) == (self._sign == 1):
            return DensePositiveDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)
        else:
            return DenseDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor, is_posdef=False)

    @property
    def factor(self):
        if self._factor is None:
            try:
                self._factor = TriangularMatrix(
                    nla.cholesky(self._sign * self._array), lower=True,
                    make_triangular=False)
            except nla.LinAlgError as e:
                raise LinAlgError('Cholesky factorisation failed.') from e
        return self._factor

    @property
    def grad_log_abs_det(self):
        return self.inv.array

    def grad_quadratic_form_inv(self, vector):
        inv_matrix_vector = self.inv @ vector
        return -np.outer(inv_matrix_vector, inv_matrix_vector)

    def _construct_inv(self):
        return DenseDefiniteMatrix(
            super()._construct_inv().array, factor=self.factor.inv.T,
            is_posdef=(self._sign == 1))


class DensePositiveDefiniteMatrix(DenseDefiniteMatrix, PositiveDefiniteMatrix):
    """Positive-definite matrix specified by a dense 2D array."""

    def __init__(self, array, factor=None):
        """
        Args:
            array (array): 2D array specifying matrix entries.
            factor (None or TriangularMatrix or InverseTriangularMatrix):
                Optional argument giving the triangular factorisation of the
                matrix such that `matrix = factor @ factor.T`. If not
                pre-computed and specified at initialisation a factorisation
                will only be computed when first required by an operation which
                depends on the factor.
        """
        super().__init__(array=array, factor=factor, is_posdef=True)

    def _construct_inv(self):
        return DensePositiveDefiniteMatrix(
            super()._construct_inv(), factor=self.factor.inv.T)

    def _construct_sqrt(self):
        return self.factor


class DensePositiveDefiniteProductMatrix(DensePositiveDefiniteMatrix):
    """Positive-definite matrix specified as a signed symmetric product.

    The matrix is assumed to have the form

        matrix = rect_matrix @ pos_def_matrix @ rect_matrix.T

    for a dense rectangular matrix `rect_matrix` with shape `(dim_0, dim_1)`
    (`dim_1 > dim_0`) positive definite matrix `pos_def_matrix` with shape
    `(dim_1, dim_1)`, with the resulting matrix being positive definite under
    the assumption that `rect_matrix` has full row rank.
    """

    def __init__(self, rect_matrix, pos_def_matrix=None):
        """
        Args:
            rect_matrix (array or Matrix): Rectangular matrix of shape
                `(dim_0, dim_1)` with it and its transpose forming the leftmost
                and righmost term respectively in the symmetric matrix product
                defining the matrix.
            pos_def_matrix (None or PositiveDefiniteMatrix): Optional positive
                positive definite matrix with shape `(dim_inner, dim_inner)`
                specifying inner term in symmetric matrix product defining
                matrix. If `None` an identity matrix is used.
        """
        if not rect_matrix.shape[0] < rect_matrix.shape[1]:
            raise ValueError('rect_matrix must have more columns than rows')
        if not isinstance(rect_matrix, Matrix):
            rect_matrix = DenseRectangularMatrix(rect_matrix)
        self._rect_matrix = rect_matrix
        if pos_def_matrix is None:
            pos_def_matrix = IdentityMatrix(rect_matrix.shape[1])
        self._pos_def_matrix = pos_def_matrix
        _array = rect_matrix @ (pos_def_matrix @ rect_matrix.T.array)
        super().__init__(_array)

    @property
    def grad_log_abs_det(self):
        return 2 * (self.inv @ (
            self._rect_matrix.array @ self._pos_def_matrix))

    def grad_quadratic_form_inv(self, vector):
        inv_matrix_vector = self.inv @ vector
        return -2 * np.outer(
            inv_matrix_vector,
            self._pos_def_matrix @ (self._rect_matrix.T @ inv_matrix_vector))


class DenseSquareMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Dense non-singular square matrix."""

    def __init__(self, array, lu_and_piv=None, lu_transposed=None):
        """
        Args:
            array (array): 2D array specifying matrix entries.
            lu_and_piv (Tuple[array, array]): Pivoted LU factorisation
                represented as a tuple with first element a 2D array containing
                the lower and upper triangular factors (with the unit diagonal
                of the lower triangular factor not stored) and the second
                element a 1D array containing the pivot indices. Corresponds
                to the output of `scipy.linalg.lu_factor` and input to
                `scipy.linalg.lu_solve`.
            lu_transposed (bool): Whether LU factorisation is of original array
                or its transpose.
        """
        super().__init__(array.shape, _array=array)
        self._lu_and_piv = lu_and_piv
        self._lu_transposed = lu_transposed

    def _scalar_multiply(self, scalar):
        if self._lu_and_piv is None or self._lu_transposed is None:
            return DenseSquareMatrix(scalar * self._array)
        else:
            old_lu, piv = self._lu_and_piv
            # Multiply upper-triangle by scalar
            new_lu = old_lu + (scalar - 1) * np.triu(old_lu)
            return DenseSquareMatrix(
                scalar * self._array, (new_lu, piv), self._lu_transposed)

    @property
    def lu_and_piv(self):
        """Pivoted LU factorisation of matrix."""
        if self._lu_and_piv is None:
            self._lu_and_piv = sla.lu_factor(self._array, check_finite=False)
            self._lu_transposed = False
        return self._lu_and_piv

    @property
    def log_abs_det(self):
        lu, piv = self.lu_and_piv
        return np.log(np.abs(lu.diagonal())).sum()

    def _construct_transpose(self):
        lu_and_piv = self.lu_and_piv
        return DenseSquareMatrix(
            self._array.T, lu_and_piv, not self._lu_transposed)

    def _construct_inv(self):
        lu_and_piv = self.lu_and_piv
        return InverseLUFactoredSquareMatrix(
            self._array, lu_and_piv, self._lu_transposed)


class InverseLUFactoredSquareMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Square matrix implicitly defined by LU factorisation of inverse."""

    def __init__(self, inv_array, inv_lu_and_piv, inv_lu_transposed):
        """
        Args:
            inv_array (array): 2D array specifying inverse matrix entries.
            inv_lu_and_piv (Tuple[array, array]): Pivoted LU factorisation
                represented as a tuple with first element a 2D array containing
                the lower and upper triangular factors (with the unit diagonal
                of the lower triangular factor not stored) and the second
                element a 1D array containing the pivot indices. Corresponds
                to the output of `scipy.linalg.lu_factor` and input to
                `scipy.linalg.lu_solve`.
            inv_lu_transposed (bool): Whether LU factorisation is of inverse of
                array or transpose of inverse of array.
        """
        super().__init__(inv_array.shape)
        self._inv_array = inv_array
        self._inv_lu_and_piv = inv_lu_and_piv
        self._inv_lu_transposed = inv_lu_transposed

    def _scalar_multiply(self, scalar):
        old_inv_lu, piv = self._inv_lu_and_piv
        # Divide upper-triangle by scalar
        new_inv_lu = old_inv_lu - (scalar - 1) / scalar * np.triu(old_inv_lu)
        return InverseLUFactoredSquareMatrix(
            self._inv_array / scalar, (new_inv_lu, piv),
            self._inv_lu_transposed)

    def _left_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inv_lu_and_piv, other, self._inv_lu_transposed,
            check_finite=False)

    def _right_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inv_lu_and_piv, other.T, not self._inv_lu_transposed,
            check_finite=False).T

    @property
    def log_abs_det(self):
        return -np.log(np.abs(self._inv_lu_and_piv[0].diagonal())).sum()

    def _construct_array(self):
        return self @ np.identity(self.shape[0])

    def _construct_inv(self):
        return DenseSquareMatrix(
            self._inv_array, self._inv_lu_and_piv, self._inv_lu_transposed)

    def _construct_transpose(self):
        return InverseLUFactoredSquareMatrix(
            self._inv_array.T, self._inv_lu_and_piv,
            not self._inv_lu_transposed)

    def _compute_hash(self):
        return hash_array(self._inv_array)

    def _check_equality(self, other):
        return np.array_equal(self._inv_array, other._inv_array)


class DenseSymmetricMatrix(
        SymmetricMatrix, InvertibleMatrix, ExplicitArrayMatrix):
    """Dense non-singular symmetric matrix."""

    def __init__(self, array, eigvec=None, eigval=None):
        """
        Args:
            array (array): Explicit 2D array representation of matrix.
            eigvec (None or array or OrthogonalMatrix): Optional. If specified
                either a 2D array or an `OrthogonalMatrix` instance, in both
                cases the columns of the matrix corresponding to the
                orthonormal set of eigenvectors of the matrix being
                constructed.
            eigval (None or array): Optional. If specified a 1D array
                containing the eigenvalues of the matrix being constructed,
                with `eigval[i]` the eigenvalue associated with column `i` of
                `eigvec`.
        """
        super().__init__(array.shape, _array=array)
        if isinstance(eigvec, np.ndarray):
            eigvec = OrthogonalMatrix(eigvec)
        self._eigvec = eigvec
        self._eigval = eigval

    def _scalar_multiply(self, scalar):
        return DenseSymmetricMatrix(
            self.array * scalar, self._eigvec,
            None if self._eigval is None else self._eigval * scalar)

    def _construct_inv(self):
        return EigendecomposedSymmetricMatrix(self.eigvec, 1 / self.eigval)


class OrthogonalMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Square matrix with columns and rows that are orthogonal unit vectors."""

    def __init__(self, array):
        """
        Args:
            array (array): Explicit 2D array representation of matrix.
        """
        super().__init__(array.shape, _array=array)

    def _scalar_multiply(self, scalar):
        return ScaledOrthogonalMatrix(scalar, self.array)

    @property
    def log_abs_det(self):
        return 0

    def _construct_transpose(self):
        return OrthogonalMatrix(self.array.T)

    def _construct_inv(self):
        return self.T


class ScaledOrthogonalMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Matrix corresponding to orthogonal matrix multiplied by a scalar.

    Matrix is assumed to have the paramterisation

        matrix = scalar * orth_array

    where `scalar` is a real-valued scalar and `orth_array` is an orthogonal
    matrix represented as a square 2D array.
    """

    def __init__(self, scalar, orth_array):
        """
        Args:
            scalar (float): Scalar multiplier as a floating point value.
            orth_array (array): 2D array representation of orthogonal matrix.
        """
        super().__init__(orth_array.shape, _orth_array=orth_array)
        self._scalar = scalar

    def _left_matrix_multiply(self, other):
        return self._scalar * (self._orth_array @ other)

    def _right_matrix_multiply(self, other):
        return self._scalar * (other @ self._orth_array)

    def _scalar_multiply(self, scalar):
        return ScaledOrthogonalMatrix(scalar * self._scalar, self._orth_array)

    def _construct_array(self):
        return self._scalar * self._orth_array

    @property
    def diagonal(self):
        return self._scalar * self._orth_array.diagonal()

    @property
    def log_abs_det(self):
        return self.shape[0] * np.log(abs(self._scalar))

    def _construct_transpose(self):
        return ScaledOrthogonalMatrix(self._scalar, self._orth_array.T)

    def _construct_inv(self):
        return ScaledOrthogonalMatrix(1 / self._scalar, self._orth_array.T)

    def _compute_hash(self):
        return hash((self._scalar, hash_array(self._orth_array)))

    def _check_equality(self, other):
        return self._scalar == other._scalar and (
            np.array_equal(self._orth_array, other._orth_array))


class EigendecomposedSymmetricMatrix(
        SymmetricMatrix, InvertibleMatrix, ImplicitArrayMatrix):
    """Symmetric matrix parametrized by its eigendecomposition.

    The matrix is assumed to have the parameterisation

        matrix = eigvec @ diag(eigval) @ eigvec.T

    where `eigvec` is an orthogonal matrix, with columns corresponding to
    the eigenvectors of `matrix` and `eigval` is 1D array of the corresponding
    eigenvalues of `matrix`.
    """

    def __init__(self, eigvec, eigval):
        """
        Args:
            eigvec (array or OrthogonalMatrix): Either a 2D array or an
                `OrthogonalMatrix` instance, in both cases the columns of the
                matrix corresponding to the orthonormal set of eigenvectors of
                the matrix being constructed.
            eigval (array): A 1D array containing the eigenvalues of the matrix
                being constructed, with `eigval[i]` the eigenvalue associated
                with column `i` of `eigvec`.
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

    def _scalar_multiply(self, scalar):
        return EigendecomposedSymmetricMatrix(
            self.eigvec, self.eigval * scalar)

    def _left_matrix_multiply(self, other):
        return self.eigvec @ (self.diag_eigval @ (self.eigvec.T @ other))

    def _right_matrix_multiply(self, other):
        return ((other @ self.eigvec) @ self.diag_eigval) @ self.eigvec.T

    def _construct_inv(self):
        return EigendecomposedSymmetricMatrix(self.eigvec, 1 / self.eigval)

    def _construct_array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for symmetric '
                'eigendecomposed matrix with implicit size.')
        return self @ np.identity(self.shape[0])

    def _compute_hash(self):
        return hash((hash_array(self.eigval), self.eigvec))

    def _check_equality(self, other):
        return np.array_equal(self.eigval, other.eigval) and (
            self.eigvec == other.eigvec)


class EigendecomposedPositiveDefiniteMatrix(
        EigendecomposedSymmetricMatrix, PositiveDefiniteMatrix):
    """Positive definite matrix parametrized by its eigendecomposition.

    The matrix is assumed to have the parameterisation

        matrix = eigvec @ diag(eigval) @ eigvec.T

    where `eigvec` is an orthogonal matrix, with columns corresponding to
    the eigenvectors of `matrix` and `eigval` is 1D array of the corresponding
    strictly positive eigenvalues of `matrix`.
    """

    def __init__(self, eigvec, eigval):
        if not np.all(eigval > 0):
            raise ValueError('Eigenvalues must all be positive.')
        super().__init__(eigvec, eigval)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return EigendecomposedPositiveDefiniteMatrix(
                self.eigvec, self.eigval * scalar)
        else:
            return super()._scalar_multiply(scalar)

    def _construct_inv(self):
        return EigendecomposedPositiveDefiniteMatrix(
            self.eigvec, 1 / self.eigval)

    def _construct_sqrt(self):
        return EigendecomposedSymmetricMatrix(self.eigvec, self.eigval**0.5)


class SoftAbsRegularizedPositiveDefiniteMatrix(
        EigendecomposedPositiveDefiniteMatrix, DifferentiableMatrix):
    """Matrix transformed to be positive definite by regularising eigenvalues.

    Matrix is parametrized by a symmetric array `symmetric_array`, of which an
    eigendecomposition is formed `eigvec, eigval = eigh(symmetric_array)`, with
    the output matrix then `matrix = eigvec @ softabs(eigval) @ eigvec.T`
    where `softabs` is a smooth approximation to the absolute function.
    """

    def __init__(self, symmetric_array, softabs_coeff):
        """
        Args:
            symmetric_array (array): 2D square array with symmetric values,
                i.e. `symmetric_array[i, j] == symmetric_array[j, i]` for all
                indices `i` and `j` which represents symmetric matrix to
                form eigenvalue-regularized transformation of.
            softabs_coeff (float): Positive regularisation coefficient for
                smooth approximation to absolute value. As the value tends to
                infinity the approximation becomes increasingly close to the
                absolute function.
        """
        if softabs_coeff <= 0:
            raise ValueError('softabs_coeff must be positive.')
        self._softabs_coeff = softabs_coeff
        self.unreg_eigval, eigvec = nla.eigh(symmetric_array)
        eigval = self.softabs(self.unreg_eigval)
        super().__init__(eigvec, eigval)

    def softabs(self, x):
        """Smooth approximation to absolute function."""
        return x / np.tanh(x * self._softabs_coeff)

    def grad_softabs(self, x):
        """Derivative of smooth approximation to absolute function."""
        return (
            1. / np.tanh(self._softabs_coeff * x) -
            self._softabs_coeff * x / np.sinh(self._softabs_coeff * x)**2)

    @property
    def grad_log_abs_det(self):
        grad_eigval = self.grad_softabs(self.unreg_eigval) / self.eigval
        return EigendecomposedSymmetricMatrix(self.eigvec, grad_eigval).array

    def grad_quadratic_form_inv(self, vector):
        num_j_mtx = self.eigval[:, None] - self.eigval[None, :]
        num_j_mtx += np.diag(self.grad_softabs(self.unreg_eigval))
        den_j_mtx = self.unreg_eigval[:, None] - self.unreg_eigval[None, :]
        np.fill_diagonal(den_j_mtx, 1)
        j_mtx = num_j_mtx / den_j_mtx
        e_vct = (self.eigvec.T @ vector) / self.eigval
        return -(
            (self.eigvec @ (np.outer(e_vct, e_vct) * j_mtx)) @ self.eigvec.T)


class BlockMatrix(ImplicitArrayMatrix):
    """Matrix with non-zero entries defined by a series of submatrix blocks."""

    @property
    @abc.abstractmethod
    def blocks(self):
        """Non-zero blocks of matrix as a tuple of Matrix instances"""

    def _compute_hash(self):
        return hash(tuple(block for block in self.blocks))

    def _check_equality(self, other):
        return (
            len(other.blocks) == len(self.blocks) and
            all(block_s == block_o for block_s, block_o
                in zip(self.blocks, other.blocks))
        )


class SquareBlockDiagonalMatrix(InvertibleMatrix, BlockMatrix):
    """Square matrix with non-zero values only in blocks along diagonal."""

    def __init__(self, blocks):
        """
        Args:
            blocks (Iterable[SquareMatrix]): Sequence of square matrices
                defining non-zero blocks along diagonal of matrix in order
                left-to-right.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, SquareMatrix) for block in self._blocks):
            raise ValueError('All blocks must be square')
        sizes = tuple(block.shape[0] for block in self._blocks)
        total_size = sum(sizes)
        super().__init__((total_size, total_size))
        self._sizes = sizes
        self._splits = np.cumsum(sizes[:-1])

    @property
    def blocks(self):
        """Blocks containing non-zero values left-to-right along diagonal."""
        return self._blocks

    def _split(self, other, axis=0):
        assert other.shape[axis] == self.shape[0]
        return np.split(other, self._splits, axis=axis)

    def _left_matrix_multiply(self, other):
        return np.concatenate(
            [block @ part for block, part in
             zip(self._blocks, self._split(other, axis=0))], axis=0)

    def _right_matrix_multiply(self, other):
        return np.concatenate(
            [part @ block for block, part in
             zip(self._blocks, self._split(other, axis=-1))], axis=-1)

    def _scalar_multiply(self, scalar):
        return SquareBlockDiagonalMatrix(
            tuple(scalar * block for block in self._blocks))

    def _construct_array(self):
        return sla.block_diag(*(block.array for block in self._blocks))

    def _construct_transpose(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.T for block in self._blocks))

    def _construct_sqrt(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.sqrt for block in self._blocks))

    @property
    def diagonal(self):
        return np.concatenate([block.diagonal for block in self._blocks])

    def _construct_inv(self):
        return type(self)(tuple(block.inv for block in self._blocks))

    @property
    def eigval(self):
        return np.concatenate([block.eigval for block in self._blocks])

    @property
    def eigvec(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.eigvec for block in self._blocks))

    @property
    def log_abs_det(self):
        return sum(block.log_abs_det for block in self._blocks)


class SymmetricBlockDiagonalMatrix(SquareBlockDiagonalMatrix, SymmetricMatrix):
    """Symmetric specialisation of `SquareBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be symmetric, i.e.
    `block.T == block`.
    """

    def __init__(self, blocks):
        """
        Args:
            blocks (Iterable[SymmetricMatrix]): Sequence of symmetric matrices
                defining non-zero blocks along diagonal of matrix in order
                left-to-right.
        """
        blocks = tuple(blocks)
        if not all(isinstance(block, SymmetricMatrix) for block in blocks):
            raise ValueError('All blocks must be symmetric')
        super().__init__(blocks)

    def _scalar_multiply(self, scalar):
        return SymmetricBlockDiagonalMatrix(
            tuple(scalar * block for block in self._blocks))

    def _construct_transpose(self):
        return self


class PositiveDefiniteBlockDiagonalMatrix(
        SquareBlockDiagonalMatrix, PositiveDefiniteMatrix, 
        DifferentiableMatrix):
    """Positive definite specialisation of `SymmetricBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be positive definite.
    """

    def __init__(self, blocks):
        """
        Args:
            blocks (Iterable[PositiveDefinite]): Sequence of positive-definite
                matrices defining non-zero blocks along diagonal of matrix in
                order left-to-right.
        """
        blocks = tuple(blocks)
        if not all(isinstance(block, PositiveDefiniteMatrix)
                   for block in blocks):
            raise ValueError('All blocks must be positive definite')
        self.is_differentiable = all(
            [isinstance(block, DifferentiableMatrix) for block in blocks])
        super().__init__(blocks)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveDefiniteBlockDiagonalMatrix(
                tuple(scalar * block for block in self._blocks))
        else:
            return super()._scalar_multiply(scalar)

    def _construct_transpose(self):
        return self

    def _construct_sqrt(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.sqrt for block in self._blocks))

    @property
    def grad_log_abs_det(self):
        if self.is_differentiable:
            return tuple(
                block.grad_log_abs_det for block in self._blocks)
        else:
            raise RuntimeError('Not all blocks are differentiable')

    def grad_quadratic_form_inv(self, vector):
        if self.is_differentiable:
            return tuple(
                block.grad_quadratic_form_inv(vector_part)
                for block, vector_part in
                zip(self._blocks, self._split(vector, axis=0)))
        else:
            raise RuntimeError('Not all blocks are differentiable')


class DenseRectangularMatrix(ExplicitArrayMatrix):
    """Dense rectangular matrix."""

    def __init__(self, array):
        """
        Args:
            array (array): 2D array specifying matrix entries.
        """
        super().__init__(array.shape, _array=array)

    def _scalar_multiply(self, scalar):
        return DenseRectangularMatrix(scalar * self.array)

    def _construct_transpose(self):
        return DenseRectangularMatrix(self.array.T)


class BlockRowMatrix(BlockMatrix):
    """Matrix composed of horizontal concatenation of a series of blocks."""

    def __init__(self, blocks):
        """
        Args:
            blocks (Iterable[Matrix]): Sequence of matrices defining a row of
                blocks in order left-to-right which when horizontally
                concatenated give the overall matrix.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, Matrix) for block in self._blocks):
            raise ValueError('All blocks must be matrices')
        if len(set([block.shape[0] for block in self._blocks])) > 1:
            raise ValueError('All blocks must have same row-dimension.')
        col_dims = tuple(block.shape[1] for block in self._blocks)
        super().__init__(shape=(self._blocks[0].shape[0], sum(col_dims)))
        self._splits = np.cumsum(col_dims[:-1])

    @property
    def blocks(self):
        """Blocks of matrix in left-to-right order."""
        return self._blocks

    def _left_matrix_multiply(self, other):
        assert other.shape[0] == self.shape[1]
        return sum(
            [block @ part for block, part in
             zip(self._blocks, np.split(other, self._splits, axis=0))])

    def _right_matrix_multiply(self, other):
        return np.concatenate(
            [other @ block for block in self._blocks], axis=-1)

    def _scalar_multiply(self, scalar):
        return BlockRowMatrix(
            tuple(scalar * block for block in self._blocks))

    def _construct_array(self):
        return np.concatenate([block.array for block in self._blocks], axis=1)

    def _construct_transpose(self):
        return BlockColumnMatrix(tuple(block.T for block in self._blocks))


class BlockColumnMatrix(BlockMatrix):
    """Matrix composed of vertical concatenation of a series of blocks."""

    def __init__(self, blocks):
        """
        Args:
            blocks (Iterable[Matrix]): Sequence of matrices defining a column
                of blocks in order top-to-bottom which when vertically
                concatenated give the overall matrix.
        """
        self._blocks = tuple(blocks)
        if not all(isinstance(block, Matrix) for block in self._blocks):
            raise ValueError('All blocks must be matrices')
        if len(set([block.shape[1] for block in self._blocks])) > 1:
            raise ValueError('All blocks must have same column-dimension.')
        row_dims = tuple(block.shape[0] for block in self._blocks)
        super().__init__(shape=(sum(row_dims), self._blocks[0].shape[1]))
        self._splits = np.cumsum(row_dims[:-1])

    @property
    def blocks(self):
        """Blocks of matrix in top-to-bottom order."""
        return self._blocks

    def _left_matrix_multiply(self, other):
        return np.concatenate(
            [block @ other for block in self._blocks], axis=0)

    def _right_matrix_multiply(self, other):
        assert other.shape[-1] == self.shape[0]
        return sum(
            [part @ block for block, part in
             zip(self._blocks, np.split(other, self._splits, axis=-1))])

    def _scalar_multiply(self, scalar):
        return BlockColumnMatrix(
            tuple(scalar * block for block in self._blocks))

    def _construct_array(self):
        return np.concatenate([block.array for block in self._blocks], axis=0)

    def _construct_transpose(self):
        return BlockRowMatrix(tuple(block.T for block in self._blocks))


class SquareLowRankUpdateMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Square matrix equal to a low-rank update to a square matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            square_matrix + sign *
            left_factor_matrix @ inner_square_matrix @ right_factor_matrix)

    where `left_factor_matrix` and `right_factor_matrix` are rectangular
    with shapes `(dim_outer, dim_inner)` and `(dim_inner, dim_outer)`
    resp., `square_matrix` is square with shape `(dim_outer, dim_outer)`,
    `inner_square_matrix` is square with shape `(dim_inner, dim_inner)` and
    `sign` is one of {-1, +1} and determines whether a low-rank update
    (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the
    inverse and determinant of the matrix can be computed at a cost of
    `O(dim_inner**3 + dim_inner**2 * dim_outer)` plus the cost of inverting /
    evaluating the determinant of `square_matrix`, which for `square_matrix`
    instances with special structure such as diagonality or with an existing
    factorisation, will typically be cheaper than the `O(dim_outer**3)` cost
    of evaluating the inverse or determinant directly.
    """

    def __init__(self, left_factor_matrix, right_factor_matrix, square_matrix,
                 inner_square_matrix=None, capacitance_matrix=None, sign=1):
        """
        Args:
            left_factor_matrix (Matrix): Rectangular matrix with shape
                `(dim_outer, dim_inner)` forming leftmost term in matrix
                product defining low-rank update.
            right_factor_matrix (Matrix): Rectangular matrix with shape
                `(dim_inner, dim_outer)` forming rightmost term in matrix
                product defining low-rank update.
            square_matrix (SquareMatrix): Square matrix to perform low-rank
                update (or downdate) to.
            inner_square_matrix (None or SquareMatrix): Optional square matrix
                with shape `(dim_inner, dim_inner)` specifying inner term in
                matrix product defining low-rank update. If `None` an identity
                matrix is used.
            capacitance_matrix (None or SquareMatrix): Square matrix equal to
                `inner_square_matrix.inv + right_factor_matrix @
                square_matrix.inv @ left_factor_matrix` and with shape
                `(dim_inner, dim_inner)` which is used in constructing inverse
                and computation of determinant of the low-rank updated matrix,
                with this argument optional and typically only passed when
                this matrix has already been computed in a previous
                computation.
            sign (int): One of {-1, +1}, determining whether a low-rank update
                (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.
        """
        dim_outer, dim_inner = left_factor_matrix.shape
        if square_matrix.shape[0] != dim_outer:
            raise ValueError(f'Inconsistent factor and square matrix shapes: '
                             f'outer dimensions {dim_outer} and '
                             f'{square_matrix.shape[0]}.')
        if square_matrix.shape[0] != square_matrix.shape[1]:
            raise ValueError('square_matrix argument must be square')
        if not isinstance(left_factor_matrix, Matrix):
            left_factor_matrix = DenseRectangularMatrix(left_factor_matrix)
        if not isinstance(right_factor_matrix, Matrix):
            right_factor_matrix = DenseRectangularMatrix(right_factor_matrix)
        if right_factor_matrix.shape != (dim_inner, dim_outer):
            raise ValueError(f'Inconsistent factor matrix shapes: '
                             f'{left_factor_matrix.shape} and '
                             f'{right_factor_matrix.shape}.')
        if inner_square_matrix is None:
            inner_square_matrix = IdentityMatrix(dim_inner)
        elif inner_square_matrix.shape != (dim_inner, dim_inner):
            raise ValueError(f'inner_square matrix must be square and of shape'
                             f' {(dim_inner, dim_inner)}.')
        self.left_factor_matrix = left_factor_matrix
        self.right_factor_matrix = right_factor_matrix
        self.square_matrix = square_matrix
        self.inner_square_matrix = inner_square_matrix
        self._capacitance_matrix = capacitance_matrix
        self._sign = sign
        super().__init__((dim_outer, dim_outer))

    def _left_matrix_multiply(self, other):
        return self.square_matrix @ other + (
            self._sign * self.left_factor_matrix @ (
                self.inner_square_matrix @ (self.right_factor_matrix @ other)))

    def _right_matrix_multiply(self, other):
        return other @ self.square_matrix + (
            self._sign * (
                other @ self.left_factor_matrix) @ self.inner_square_matrix
            ) @ self.right_factor_matrix

    def _scalar_multiply(self, scalar):
        return type(self)(
            self.left_factor_matrix, self.right_factor_matrix,
            scalar * self.square_matrix, scalar * self.inner_square_matrix,
            self._capacitance_matrix / scalar
            if self._capacitance_matrix is not None else None, self._sign)

    def _construct_array(self):
        return self.square_matrix.array + (
            self._sign * self.left_factor_matrix @ (
                self.inner_square_matrix @ self.right_factor_matrix.array))

    @property
    def capacitance_matrix(self):
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DenseSquareMatrix(
                self.inner_square_matrix.inv.array +
                self.right_factor_matrix @ (
                    self.square_matrix.inv @ self.left_factor_matrix.array))
        return self._capacitance_matrix

    @property
    def diagonal(self):
        return self.square_matrix.diagonal + self._sign * (
            (self.left_factor_matrix.array @ self.inner_square_matrix) *
            self.right_factor_matrix.T.array).sum(1)

    def _construct_transpose(self):
        return type(self)(
            self.right_factor_matrix.T, self.left_factor_matrix.T,
            self.square_matrix.T, self.inner_square_matrix.T,
            self._capacitance_matrix.T
            if self._capacitance_matrix is not None else None,
            self._sign)

    def _construct_inv(self):
        return type(self)(
            self.square_matrix.inv @ self.left_factor_matrix,
            self.right_factor_matrix @ self.square_matrix.inv,
            self.square_matrix.inv, self.capacitance_matrix.inv,
            self.inner_square_matrix.inv, -self._sign)

    @property
    def log_abs_det(self):
        return (
            self.square_matrix.log_abs_det +
            self.inner_square_matrix.log_abs_det +
            self.capacitance_matrix.log_abs_det)

    def _compute_hash(self):
        return hash((self.left_factor_matrix, self.right_factor_matrix,
                     self.square_matrix, self.inner_square_matrix))

    def _check_equality(self, other):
        return (
            self.left_factor_matrix == other.left_factor_matrix and
            self.right_factor_matrix == other.right_factor_matrix and
            self.square_matrix == other.square_matrix and
            self.inner_square_matrix == other.inner_square_matrix)


class SymmetricLowRankUpdateMatrix(
        SquareLowRankUpdateMatrix, SymmetricMatrix, InvertibleMatrix):
    """Symmetric matrix equal to a low-rank update to a symmetric matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            symmetric_matrix +
            sign * factor_matrix @ inner_symmetric_matrix @ factor_matrix.T)

    where `factor_matrix` is rectangular with shape `(dim_outer, dim_inner)`,
    `symmetric_matrix` is symmetric with shape `(dim_outer, dim_outer)`,
    `inner_symmetric_matrix` is symmetric with shape `(dim_inner, dim_inner)`
    and `sign` is one of {-1, +1} and determines whether a low-rank update
    (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the
    inverse and determinant of the matrix can be computed at a cost of
    `O(dim_inner**3 + dim_inner**2 * dim_outer)` plus the cost of inverting /
    evaluating the determinant of `square_matrix`, which for `square_matrix`
    instances with special structure such as diagonality or with an existing
    factorisation, will typically be cheaper than the `O(dim_outer**3)` cost
    of evaluating the inverse or determinant directly.
    """

    def __init__(self, factor_matrix, symmetric_matrix,
                 inner_symmetric_matrix=None, capacitance_matrix=None, sign=1):
        """
        Args:
            factor_matrix (Matrix): Rectangular matrix with shape
                `(dim_outer, dim_inner)` with it and its transpose forming the
                leftmost and righmost term respectively in the matrix product
                defining the low-rank update.
            symmetric_matrix (SymmetricMatrix): Symmetric matrix to perform
                low-rank update (or downdate) to.
            inner_symmetric_matrix (None or SymmetricMatrix): Optional
                symmetric matrix with shape `(dim_inner, dim_inner)` specifying
                inner term in matrix product defining low-rank update. If
                `None` an identity matrix is used.
            capacitance_matrix (None or SymmetricMatrix): Symmetric matrix
                equal to `inner_symmetric_matrix.inv + factor_matrix.T @
                symmetric_matrix.inv @ factor_matrix` and with shape
                `(dim_inner, dim_inner)` which is used in constructing inverse
                and computation of determinant of the low-rank updated matrix,
                with this argument optional and typically only passed when
                this matrix has already been computed in a previous
                computation.
            sign (int): One of {-1, +1}, determining whether a low-rank update
                (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.
        """
        dim_inner = factor_matrix.shape[1]
        if symmetric_matrix.T is not symmetric_matrix:
            raise ValueError('symmetric_matrix must be symmetric')
        if inner_symmetric_matrix is None:
            inner_symmetric_matrix = IdentityMatrix(dim_inner)
        if inner_symmetric_matrix.T is not inner_symmetric_matrix:
            raise ValueError('inner_symmetric_matrix must be symmetric')
        if not isinstance(factor_matrix, Matrix):
            factor_matrix = DenseRectangularMatrix(factor_matrix)
        self.factor_matrix = factor_matrix
        self.symmetric_matrix = symmetric_matrix
        self.inner_symmetric_matrix = inner_symmetric_matrix
        super().__init__(
            factor_matrix, factor_matrix.T, symmetric_matrix,
            inner_symmetric_matrix, capacitance_matrix, sign)

    def _scalar_multiply(self, scalar):
        return type(self)(
            self.factor_matrix, scalar * self.symmetric_matrix,
            scalar * self.inner_symmetric_matrix,
            self._capacitance_matrix / scalar
            if self._capacitance_matrix is not None else None, self._sign)

    @property
    def capacitance_matrix(self):
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DenseSymmetricMatrix(
                self.inner_symmetric_matrix.inv.array +
                self.factor_matrix.T @ (
                    self.symmetric_matrix.inv @ self.factor_matrix.array))
        return self._capacitance_matrix

    def _construct_inv(self):
        return type(self)(
            self.symmetric_matrix.inv @ self.factor_matrix,
            self.symmetric_matrix.inv, self.capacitance_matrix.inv,
            self.inner_symmetric_matrix.inv, -self._sign)

    def _construct_transpose(self):
        return self

    def _compute_hash(self):
        return hash((
            self.factor_matrix, self.square_matrix, self.inner_square_matrix))

    def _check_equality(self, other):
        return (
            self.factor_matrix == other.factor_matrix and
            self.square_matrix == other.square_matrix and
            self.inner_square_matrix == other.inner_square_matrix)


class PositiveDefiniteLowRankUpdateMatrix(
        SymmetricLowRankUpdateMatrix, PositiveDefiniteMatrix,
        DifferentiableMatrix):
    """Positive-definite matrix equal to low-rank update to a square matrix.

    The matrix is assumed to have the parametrisation

        matrix = (
            pos_def_matrix +
            sign * factor_matrix @ inner_pos_def_matrix @ factor_matrix.T)

    where `factor_matrix` is rectangular with shape `(dim_outer, dim_inner)`,
    `pos_def_matrix` is positive-definite with shape `(dim_outer, dim_outer)`,
    `inner_pos_def_matrix` is positive-definite with shape
    `(dim_inner, dim_inner)` and `sign` is one of {-1, +1} and determines
    whether a low-rank update (`sign = 1`) or 'downdate' (`sign = -1`) is
    peformed.

    By exploiting the Woodbury matrix identity and matrix determinant lemma the
    inverse, determinant and square-root of the matrix can all be computed at a
    cost of `O(dim_inner**3 + dim_inner**2 * dim_outer)` plus the cost of
    inverting / evaluating the determinant / square_root of `pos_def_matrix`,
    which for `pos_def_matrix` instances with special structure such as
    diagonality or with an existing factorisation, will typically be cheaper
    than the `O(dim_outer**3)` cost of evaluating the inverse, determinant or
    square-root directly.
    """

    def __init__(self, factor_matrix, pos_def_matrix,
                 inner_pos_def_matrix=None, capacitance_matrix=None, sign=1):
        """
        Args:
            factor_matrix (Matrix): Rectangular matrix with shape
                `(dim_outer, dim_inner)` with it and its transpose forming the
                leftmost and righmost term respectively in the matrix product
                defining the low-rank update.
            pos_def_matrix (PositiveDefiniteMatrix): Positive-definite matrix
                to perform low-rank update (or downdate) to.
            inner_pos_def_matrix (None or PositiveDefiniteMatrix): Optional
                positive definite matrix with shape `(dim_inner, dim_inner)`
                specifying inner term in matrix product defining low-rank
                update. If `None` an identity matrix is used.
            capacitance_matrix (None or PositiveDefiniteMatrix): Positive-
                definite matrix equal to `inner_pos_def_matrix.inv +
                factor_matrix.T @ pos_def_matrix.inv @ factor_matrix` and with
                shape `(dim_inner, dim_inner)` which is used in constructing
                inverse and computation of determinant of the low-rank updated
                matrix, with this argument optional and typically only passed
                when this matrix has already been computed in a previous
                computation.
            sign (int): One of {-1, +1}, determining whether a low-rank update
                (`sign = 1`) or 'downdate' (`sign = -1`) is peformed.
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
            factor_matrix, pos_def_matrix, inner_pos_def_matrix,
            capacitance_matrix, sign)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveDefiniteLowRankUpdateMatrix(
                self.factor_matrix, scalar * self.pos_def_matrix,
                scalar * self.inner_pos_def_matrix,
                self._capacitance_matrix / scalar
                if self._capacitance_matrix is not None else None, self._sign)
        else:
            return SymmetricLowRankUpdateMatrix(
                self.factor_matrix, scalar * self.pos_def_matrix,
                scalar * self.inner_pos_def_matrix,
                self._capacitance_matrix / scalar
                if self._capacitance_matrix is not None else None, self._sign)

    @property
    def capacitance_matrix(self):
        if self._capacitance_matrix is None:
            self._capacitance_matrix = DensePositiveDefiniteMatrix(
                self.inner_pos_def_matrix.inv.array +
                self.factor_matrix.T @ (
                    self.pos_def_matrix.inv @
                    self.factor_matrix.array))
        return self._capacitance_matrix

    def _construct_sqrt(self):
        # Uses O(dim_inner**3 + dim_inner**2 * dim_outer) cost implementation
        # proposed in
        #   Ambikasaran, O'Neill & Singh (2016). Fast symmetric factorization
        #   of hierarchical matrices with applications. arxiv:1405.0223.
        # Variable naming below follows notation in Algorithm 1 in paper
        W = self.pos_def_matrix.sqrt
        K = self.inner_pos_def_matrix
        U = W.inv @ self.factor_matrix
        L = TriangularMatrix(
            nla.cholesky(U.T @ U.array), lower=True, make_triangular=False)
        I_outer, I_inner = IdentityMatrix(U.shape[0]), np.identity(U.shape[1])
        M = sla.sqrtm(I_inner + L.T @ (K @ L.array))
        X = DenseSymmetricMatrix(L.inv.T @ ((M - I_inner) @ L.inv))
        return W @ SymmetricLowRankUpdateMatrix(U, I_outer, X)

    @property
    def grad_log_abs_det(self):
        return 2 * (self.inv @ (
            self.factor_matrix.array @ self.inner_pos_def_matrix))

    def grad_quadratic_form_inv(self, vector):
        inv_matrix_vector = self.inv @ vector
        return -2 * np.outer(
            inv_matrix_vector,
            self.inner_pos_def_matrix @ (
                self.factor_matrix.T @ inv_matrix_vector))
