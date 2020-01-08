"""Structured matrix classes implementing basic linear algebra operations."""

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import abc


class Matrix(abc.ABC):
    """Base class for matrix-like objects.

    Implements overloads of the matrix multiplication operator `@`, as well as
    the standard multiplication and division operators `*` and `/` when the
    second argument is a scalar quantity.
    """

    __array_priority__ = 1

    def __init__(self, shape):
        """
        Args:
           shape (Tuple[int, int]): Shape of matrix `(num_rows, num_columns)`.
        """
        self._shape = shape

    def __array__(self):
        return self.array

    def __mul__(self, other):
        if np.isscalar(other) or np.ndim(other) == 0:
            if other == 0:
                raise NotImplementedError(
                    'Scalar multiplication by zero not implemented.')
            return self._scalar_multiply(other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if np.isscalar(other) or np.ndim(other) == 0:
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
        if isinstance(other, MatrixProduct):
            return MatrixProduct((self, *other.matrices))
        elif isinstance(other, Matrix):
            return MatrixProduct((self, other))
        else:
            return self._left_matrix_multiply(other)

    def __rmatmul__(self, other):
        if self.shape[0] is not None and other.shape[-1] != self.shape[0]:
            raise ValueError(
                f'Inconsistent dimensions for matrix multiplication: '
                f'{other.shape} and {self.shape}.')
        if isinstance(other, MatrixProduct):
            return MatrixProduct((*other.matrices, self))
        elif isinstance(other, Matrix):
            return MatrixProduct((other, self))
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
    @abc.abstractmethod
    def T(self):
        """Transpose of matrix."""

    @property
    def diagonal(self):
        """Diagonal of matrix as a 1D array."""
        return self.array.diagonal()

    def __str__(self):
        return f'(shape={self.shape})'

    def __repr__(self):
        return type(self).__name__ + str(self)


class ExplicitArrayMatrix(Matrix):
    """Matrix with an explicit array representation."""

    @property
    def array(self):
        return self._array

    def _left_matrix_multiply(self, other):
        return self._array @ other

    def _right_matrix_multiply(self, other):
        return other @ self._array


class ImplicitArrayMatrix(Matrix):
    """Matrix with an implicit array representation."""

    def __init__(self, shape):
        """
        Args:
           shape (Tuple[int, int]): Shape of matrix `(num_rows, num_columns)`.
        """
        super().__init__(shape)
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
    """Matrix implicitly defined as a product of a sequence of matrices."""

    def __init__(self, matrices):
        super().__init__((matrices[0].shape[0], matrices[-1].shape[1]))
        self._matrices = tuple(matrices)

    @property
    def matrices(self):
        return self._matrices

    def _scalar_multiply(self, scalar):
        return MatrixProduct((
            ScaledIdentityMatrix(scalar, self.shape[0]), *self._matrices))

    def _left_matrix_multiply(self, other):
        for matrix in reversed(self._matrices):
            other = matrix @ other
        return other

    def _right_matrix_multiply(self, other):
        for matrix in self._matrices:
            other = other @ matrix
        return other

    @property
    def T(self):
        return MatrixProduct(
            tuple(matrix.T for matrix in reversed(self._matrices)))

    def _construct_array(self):
        return self.matrices[0].array @ MatrixProduct(self.matrices[1:])


class SquareMatrix(Matrix):
    """Base class for matrices with equal numbers of rows and columns."""

    def __init__(self, size):
        """
        Args:
            size (int): Number of rows / columns in matrix.
        """
        super().__init__((size, size))

    @property
    @abc.abstractmethod
    def log_abs_det(self):
        """Logarithm of absolute value of determinant of matrix.

        For matrix representations of metrics it is proportional to the
        logarithm of the density of then Riemannian measure associated with
        metric with respect to the Lebesgue measure.
        """


class InvertibleMatrix(SquareMatrix):
    """Base class for non-singular square matrices."""

    @property
    @abc.abstractmethod
    def inv(self):
        """Inverse of matrix as a `Matrix` object.

        This will not necessarily form an explicit representation of the
        inverse matrix but may instead return a `Matrix` object that implements
        the matrix multiplication operators by solving the linear system
        defined by the original matrix object.
        """


class SymmetricMatrix(SquareMatrix):
    """Base class for square matrices which are equal to their transpose."""

    def __init__(self, size):
        """
        Args:
            size (int): Number of rows / columns in matrix.
        """
        self._eigval = None
        self._eigvec = None
        super().__init__(size)

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

    @property
    def T(self):
        return self

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.eigval)).sum()


class PositiveDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):
    """Base class for positive definite matrices."""

    @property
    @abc.abstractmethod
    def sqrt(self):
        """Square-root of matrix satisfying `matrix == sqrt @ sqrt.T`.

        This will in general not correspond to the unique, if defined,
        symmetric square root of a symmetric matrix but instead may return any
        matrix satisfying the above property.
        """


class IdentityMatrix(PositiveDefiniteMatrix, ImplicitArrayMatrix):
    """Matrix representing identity operator on a vector space.

    Array representation has ones on diagonal elements and zeros elsewhere.
    May be defined with an implicit shape reprsented by `(None, None)` which
    will allow use for subset of operations where shape is not required to be
    known.
    """

    def __init__(self, size=None):
        super().__init__(size)

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

    @property
    def sqrt(self):
        return self

    @property
    def eigvec(self):
        return self

    @property
    def inv(self):
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
        super().__init__(size)

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

    @property
    def inv(self):
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

    @property
    def inv(self):
        return PositiveScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    @property
    def sqrt(self):
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
        self._diagonal = diagonal
        super().__init__(diagonal.size)

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

    @property
    def inv(self):
        return DiagonalMatrix(1. / self.diagonal)

    def _construct_array(self):
        return np.diag(self.diagonal)

    @property
    def grad_log_abs_det(self):
        return 1. / self.diagonal

    def grad_quadratic_form_inv(self, vector):
        return -(self.inv @ vector)**2


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

    @property
    def inv(self):
        return PositiveDiagonalMatrix(1. / self.diagonal)

    @property
    def sqrt(self):
        return PositiveDiagonalMatrix(self.diagonal**0.5)


class TriangularMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Matrix with non-zero values only in lower or upper triangle elements."""

    def __init__(self, array, lower=True):
        """
        Args:
            array (array): 2D array containing lower / upper triangular element
                values of matrix. Any values above (below) diagonal are
                ignored for lower (upper) triangular matrices i.e. when
                `lower == True` (`lower == False`).
            lower (bool): Whether the matrix is lower-triangular (`True`) or
                upper-triangular (`False`).
        """
        super().__init__(array.shape[0])
        self._array = np.tril(array) if lower else np.triu(array)
        self.lower = lower

    def _scalar_multiply(self, scalar):
        return TriangularMatrix(self.array * scalar, self.lower)

    @property
    def inv(self):
        return InverseTriangularMatrix(self.array, lower=self.lower)

    @property
    def T(self):
        return TriangularMatrix(self.array.T, lower=not self.lower)

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.diagonal)).sum()

    def __str__(self):
        return f'(shape={self.shape}, lower={self.lower})'


class InverseTriangularMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Triangular matrix implicitly specified by its inverse."""

    def __init__(self, inverse_array, lower=True):
        """
        Args:
            inverse_array (array): 2D containing values of *inverse* of this
                matrix, with the inverse of a lower (upper) triangular matrix
                being itself lower (upper) triangular. Any values above (below)
                diagonal are ignored for lower (upper) triangular matrices i.e.
                when `lower == True` (`lower == False`).
            lower (bool): Whether the matrix is lower-triangular (`True`) or
                upper-triangular (`False`).
        """
        self._inverse_array = inverse_array
        self.lower = lower
        super().__init__(inverse_array.shape[0])

    def _scalar_multiply(self, scalar):
        return InverseTriangularMatrix(
            self._inverse_array / scalar, self.lower)

    def _left_matrix_multiply(self, other):
        return sla.solve_triangular(
            self._inverse_array, other, lower=self.lower)

    def _right_matrix_multiply(self, other):
        return sla.solve_triangular(
            self._inverse_array, other.T, lower=self.lower, trans=1).T

    @property
    def inv(self):
        return TriangularMatrix(self._inverse_array, lower=self.lower)

    @property
    def T(self):
        return InverseTriangularMatrix(
            self._inverse_array.T, lower=not self.lower)

    def _construct_array(self):
        return self @ np.identity(self.shape[0])

    @property
    def diagonal(self):
        return 1. / self._inverse_array.diagonal()

    @property
    def log_abs_det(self):
        return -self.inv.log_abs_det

    def __str__(self):
        return f'(shape={self.shape}, lower={self.lower})'


class BaseTriangularFactoredDefiniteMatrix(SymmetricMatrix, InvertibleMatrix):

    def __init__(self, size, sign=1):
        if not (sign == 1 or sign == -1):
            raise ValueError('sign must be equal to +1 or -1')
        self._sign = sign
        super().__init__(size=size)

    @property
    def factor(self):
        return self._factor

    @property
    def inv(self):
        return TriangularFactoredDefiniteMatrix(
            factor=self.factor.inv.T, sign=self._sign)

    @property
    def log_abs_det(self):
        return 2 * self.factor.log_abs_det

    def __str__(self):
        return f'(shape={self.shape}, sign={self._sign})'


class TriangularFactoredDefiniteMatrix(
        BaseTriangularFactoredDefiniteMatrix, DifferentiableMatrix,
        ImplicitArrayMatrix):

    def __init__(self, factor, lower=True, sign=1):
        if not isinstance(factor, (TriangularMatrix, InverseTriangularMatrix)):
            factor = TriangularMatrix(factor, lower)
        self._factor = factor
        super().__init__(factor.shape[0], sign=sign)

    def _scalar_multiply(self, scalar):
        return TriangularFactoredDefiniteMatrix(
            factor=abs(scalar)**0.5 * self.factor,
            sign=self._sign * np.sign(scalar))

    def _left_matrix_multiply(self, other):
        return self._sign * (self.factor @ (self.factor.T @ other))

    def _right_matrix_multiply(self, other):
        return self._sign * ((other @ self.factor) @ self.factor.T)

    @property
    def grad_log_abs_det(self):
        return 2 * self.factor.inv.T.array

    def grad_quadratic_form_inv(self, vector):
        inv_factor_vector = self.factor.inv @ vector
        inv_vector = self.inv @ vector
        return -2 * self._sign * np.outer(inv_vector, inv_factor_vector)

    def _construct_array(self):
        return self._sign * (self.factor @ self.factor.array.T)


class TriangularFactoredPositiveDefiniteMatrix(
        TriangularFactoredDefiniteMatrix, PositiveDefiniteMatrix):

    def __init__(self, factor, lower=True):
        super().__init__(factor, lower=lower, sign=1)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return TriangularFactoredPositiveDefiniteMatrix(
                factor=scalar**0.5 * self.factor)
        else:
            return super()._scalar_multiply(scalar)

    @property
    def inv(self):
        return TriangularFactoredPositiveDefiniteMatrix(
            factor=self.factor.inv.T)

    @property
    def sqrt(self):
        return self.factor


class DenseDefiniteMatrix(BaseTriangularFactoredDefiniteMatrix,
                          DifferentiableMatrix, ExplicitArrayMatrix):

    def __init__(self, array, sign=1, factor=None):
        super().__init__(array.shape[0], sign=sign)
        self._array = array
        self._factor = factor

    def _scalar_multiply(self, scalar):
        if (scalar > 0) == (self._sign == 1):
            return DenseDefiniteMatrix(
                scalar * self.array, 1,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)
        else:
            return DenseDefiniteMatrix(
                scalar * self.array, -1,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)

    @property
    def factor(self):
        if self._factor is None:
            self._factor = TriangularMatrix(
                nla.cholesky(self._sign * self._array), True)
        return self._factor

    @property
    def grad_log_abs_det(self):
        return self.inv.array

    def grad_quadratic_form_inv(self, vector):
        inv_metric_vector = self.inv @ vector
        return -np.outer(inv_metric_vector, inv_metric_vector)


class DensePositiveDefiniteMatrix(DenseDefiniteMatrix, PositiveDefiniteMatrix):

    def __init__(self, array, factor=None):
        super().__init__(array=array, sign=1, factor=factor)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return DensePositiveDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)
        else:
            return DenseNegativeDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)

    @property
    def inv(self):
        return TriangularFactoredPositiveDefiniteMatrix(
            factor=self.factor.inv.T)

    @property
    def sqrt(self):
        return self.factor


class DenseNegativeDefiniteMatrix(DenseDefiniteMatrix):

    def __init__(self, array, factor=None):
        super().__init__(array=array, sign=-1, factor=factor)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return DenseNegativeDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)
        else:
            return DensePositiveDefiniteMatrix(
                scalar * self.array,
                None if self._factor is None else
                abs(scalar)**0.5 * self._factor)


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
        super().__init__(array.shape[0])
        self._array = array
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
            self._lu_and_piv = sla.lu_factor(self._array)
            self._lu_transposed = False
        return self._lu_and_piv

    @property
    def log_abs_det(self):
        lu, piv = self.lu_and_piv
        return np.log(np.abs(lu.diagonal())).sum()

    @property
    def T(self):
        lu_and_piv = self.lu_and_piv
        return DenseSquareMatrix(
            self._array.T, lu_and_piv, not self._lu_transposed)

    @property
    def inv(self):
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
        self._inv_array = inv_array
        self._inv_lu_and_piv = inv_lu_and_piv
        self._inv_lu_transposed = inv_lu_transposed
        super().__init__(inv_array.shape[0])

    def _scalar_multiply(self, scalar):
        old_inv_lu, piv = self._inv_lu_and_piv
        # Divide upper-triangle by scalar
        new_inv_lu = old_inv_lu - (scalar - 1) / scalar * np.triu(old_inv_lu)
        return InverseLUFactoredSquareMatrix(
            self._inv_array / scalar, (new_inv_lu, piv),
            self._inv_lu_transposed)

    def _left_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inv_lu_and_piv, other, self._inv_lu_transposed)

    def _right_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inv_lu_and_piv, other.T, not self._inv_lu_transposed).T

    @property
    def log_abs_det(self):
        return -np.log(np.abs(self._inv_lu_and_piv[0].diagonal())).sum()

    def _construct_array(self):
        return self @ np.identity(self.shape[0])

    @property
    def inv(self):
        return DenseSquareMatrix(
            self._inv_array, self._inv_lu_and_piv, self._inv_lu_transposed)

    @property
    def T(self):
        return InverseLUFactoredSquareMatrix(
            self._inv_array.T, self._inv_lu_and_piv,
            not self._inv_lu_transposed)


class OrthogonalMatrix(InvertibleMatrix, ExplicitArrayMatrix):
    """Square matrix with columns and rows that are orthogonal unit vectors."""

    def __init__(self, array):
        """
        Args:
            array (array): Explicit 2D array representation of matrix.
        """
        super().__init__(array.shape[0])
        self._array = array

    def _scalar_multiply(self, scalar):
        return ScaledOrthogonalMatrix(scalar, self.array)

    @property
    def log_abs_det(self):
        return 0

    @property
    def T(self):
        return OrthogonalMatrix(self.array.T)

    @property
    def inv(self):
        return self.T


class ScaledOrthogonalMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Matrix corresponding to orthogonal matrix multiplied by a scalar."""

    def __init__(self, scalar, orth_array):
        self._scalar = scalar
        self._orth_array = orth_array
        super().__init__(orth_array.shape[0])

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

    @property
    def T(self):
        return ScaledOrthogonalMatrix(self._scalar, self._orth_array.T)

    @property
    def inv(self):
        return ScaledOrthogonalMatrix(1 / self._scalar, self._orth_array.T)


class EigendecomposedSymmetricMatrix(
        SymmetricMatrix, InvertibleMatrix, ImplicitArrayMatrix):
    """Symmetric matrix parameterised by its eigendecomposition."""

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
        super().__init__(eigvec.shape[0])
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

    @property
    def inv(self):
        return EigendecomposedSymmetricMatrix(self.eigvec, 1 / self.eigval)

    def _construct_array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for symmetric '
                'eigendecomposed matrix with implicit size.')
        return self @ np.identity(self.shape[0])


class EigendecomposedPositiveDefiniteMatrix(
        EigendecomposedSymmetricMatrix, PositiveDefiniteMatrix):
    """Positive definite matrix parameterised by its eigendecomposition."""

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

    @property
    def inv(self):
        return EigendecomposedPositiveDefiniteMatrix(
            self.eigvec, 1 / self.eigval)

    @property
    def sqrt(self):
        return EigendecomposedSymmetricMatrix(self.eigvec, self.eigval**0.5)


class SoftAbsRegularisedPositiveDefiniteMatrix(
        EigendecomposedPositiveDefiniteMatrix):
    """Matrix transformed to be positive definite by regularising eigenvalues.

    Matrix is parameterised by a symmetric array `symmetric_array`, of which an
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
                form eigenvalue-regularised transformation of.
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
        return EigendecomposedSymmetricMatrix(self.eigvec, grad_eigval)

    def grad_quadratic_form_inv(self, vector):
        num_j_mtx = self.eigval[:, None] - self.eigval[None, :]
        num_j_mtx += np.diag(self.grad_softabs(self.unreg_eigval))
        den_j_mtx = self.unreg_eigval[:, None] - self.unreg_eigval[None, :]
        np.fill_diagonal(den_j_mtx, 1)
        j_mtx = num_j_mtx / den_j_mtx
        e_vct = (self.eigvec.T @ vector) / self.eigval
        return -(
            (self.eigvec @ (np.outer(e_vct, e_vct) * j_mtx)) @ self.eigvec.T)


class SquareBlockDiagonalMatrix(InvertibleMatrix, ImplicitArrayMatrix):
    """Square matrix with non-zero values only in blocks along diagonal."""

    def __init__(self, blocks):
        self._blocks = tuple(blocks)
        sizes = tuple(block.shape[0] for block in self._blocks)
        super().__init__(size=sum(sizes))
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

    @property
    def T(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.T for block in self._blocks))

    @property
    def sqrt(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.sqrt for block in self._blocks))

    @property
    def diag(self):
        return np.concatenate([block.diagonal() for block in self._blocks])

    @property
    def inv(self):
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


class SymmetricBlockDiagonalMatrix(SquareBlockDiagonalMatrix):
    """Symmetric specialisation of `SquareBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be symmetric, i.e.
    `block.T == block`.
    """

    def __init__(self, blocks):
        blocks = tuple(blocks)
        if not all(block is block.T for block in blocks):
            raise ValueError('All blocks must be symmetric')
        super().__init__(blocks)

    def _scalar_multiply(self, scalar):
        return SymmetricBlockDiagonalMatrix(
            tuple(scalar * block for block in self._blocks))

    @property
    def T(self):
        return self


class PositiveDefiniteBlockDiagonalMatrix(
        SymmetricBlockDiagonalMatrix, PositiveDefiniteMatrix):
    """Positive definite specialisation of `SymmetricBlockDiagonalMatrix`.

    All matrix blocks in diagonal are restricted to be positive definite.
    """

    def __init__(self, blocks):
        blocks = tuple(blocks)
        if not all(isinstance(block, PositiveDefiniteMatrix)
                   for block in blocks):
            raise ValueError('All blocks must be positive definite')
        super().__init__(blocks)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveDefiniteBlockDiagonalMatrix(
                tuple(scalar * block for block in self._blocks))
        else:
            return super()._scalar_multiply(scalar)

    @property
    def sqrt(self):
        return SquareBlockDiagonalMatrix(
            tuple(block.sqrt for block in self._blocks))
