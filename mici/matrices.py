"""Structured matrix classes implementing basic linear algebra operations."""

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla


class _AbstractMatrix(object):

    __array_priority__ = 1

    def __init__(self, shape):
        self.shape = shape

    def __array__(self):
        if hasattr(self, 'array'):
            return self.array
        else:
            raise NotImplementedError()

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
        try:
            return self._left_matrix_multiply(other)
        except NotImplementedError:
            return MatrixProduct(self, other)

    def __rmatmul__(self, other):
        if self.shape[0] is not None and other.shape[-1] != self.shape[0]:
            raise ValueError(
                f'Inconsistent dimensions for matrix multiplication: '
                f'{other.shape} and {self.shape}.')
        return self._right_matrix_multiply(other)

    @property
    def array(self):
        return self._array

    def _left_matrix_multiply(self, other):
        return self.array @ other

    def _right_matrix_multiply(self, other):
        return other @ self.array

    def _scalar_multiply(self, other):
        raise NotImplementedError()


class _AbstractSquareMatrix(_AbstractMatrix):

    def __init__(self, size):
        super().__init__((size, size))

    @property
    def log_abs_det_sqrt(self):
        return 0.5 * self.log_abs_det

    @property
    def diagonal(self):
        return self.array.diagonal()


class _AbstractSymmetricMatrix(_AbstractSquareMatrix):

    def __init__(self, size, is_posdef=False, is_negdef=False):
        self._eigval = None
        self._eigvec = None
        self._is_posdef = is_posdef
        self._is_negdef = is_negdef
        super().__init__(size)

    def _compute_eigendecomposition(self):
        self._eigval, eigvec = nla.eigh(self.array)
        self._eigvec = OrthogonalMatrix(eigvec)

    @property
    def eigval(self):
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigval

    @property
    def eigvec(self):
        if self._eigval is None or self._eigvec is None:
            self._compute_eigendecomposition()
        return self._eigvec

    @property
    def T(self):
        return self

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.eigval)).sum()

    @property
    def is_posdef(self):
        return self._is_posdef

    @property
    def is_negdef(self):
        return self._is_negdef


class IdentityMatrix(_AbstractSymmetricMatrix):

    def __init__(self, size=None):
        super().__init__(size, is_posdef=True, is_negdef=False)

    def _scalar_multiply(self, scalar):
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

    @property
    def array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for identity matrix with '
                'implicit size.')
        return np.identity(self.shape[0])

    @property
    def log_abs_det(self):
        return 0.

    @property
    def log_abs_det_sqrt(self):
        return 0.


class ScaledIdentityMatrix(_AbstractSymmetricMatrix):

    def __init__(self, scalar, size=None):
        if scalar == 0:
            raise ValueError('scalar must be non-zero')
        self._scalar = scalar
        super().__init__(size, is_posdef=scalar > 0, is_negdef=scalar < 0)

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
    def sqrt(self):
        if not self.is_posdef:
            raise NotImplementedError(
                'Square-root supported only for positive definite '
                'matrices.')
        return ScaledIdentityMatrix(self._scalar**0.5, self.shape[0])

    @property
    def eigvec(self):
        return IdentityMatrix(self.shape[0])

    @property
    def inv(self):
        return ScaledIdentityMatrix(1 / self._scalar, self.shape[0])

    @property
    def diagonal(self):
        return self._scalar * np.ones(self.shape[0])

    @property
    def array(self):
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
    def grad_log_abs_det_sqrt(self):
        return 0.5 * self.shape[0] / self._scalar

    def grad_quadratic_form_inv(self, vector):
        return -np.sum(vector**2) / self._scalar**2


class PositiveScaledIdentityMatrix(ScaledIdentityMatrix):

    def __init__(self, scalar, size=None):
        if scalar < 0:
            raise ValueError('Scalar multiplier must be positive.')
        super().__init__(scalar, size)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveScaledIdentityMatrix(
                scalar * self._scalar, self.shape[0])
        else:
            return ScaledIdentityMatrix(scalar * self._scalar, self.shape[0])

    @property
    def sqrt(self):
        return PositiveScaledIdentityMatrix(self._scalar**0.5, self.shape[0])


class DiagonalMatrix(_AbstractSymmetricMatrix):

    def __init__(self, diagonal):
        if diagonal.ndim != 1:
            raise ValueError('Specified diagonal must be a 1D array.')
        self._diagonal = diagonal
        super().__init__(diagonal.size, is_posdef=np.all(diagonal > 0),
                         is_negdef=np.all(diagonal < 0))

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

    @property
    def array(self):
        return np.diag(self.diagonal)

    @property
    def grad_log_abs_det_sqrt(self):
        return 0.5 / self.diagonal

    def grad_quadratic_form_inv(self, vector):
        return -(self.inv @ vector)**2


class PositiveDiagonalMatrix(DiagonalMatrix):

    def __init__(self, diagonal):
        if not np.all(diagonal >= 0):
            raise ValueError('Diagonal values must all be positive.')
        super().__init__(diagonal)

    def _scalar_multiply(self, scalar):
        if scalar > 0:
            return PositiveDiagonalMatrix(self.diagonal * scalar)
        else:
            return DiagonalMatrix(self.diagonal * scalar)

    @property
    def sqrt(self):
        return PositiveDiagonalMatrix(self.diagonal**0.5)


class TriangularMatrix(_AbstractSquareMatrix):

    def __init__(self, array, lower=True):
        self._array = array
        self.lower = lower
        super().__init__(array.shape[0])

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


class InverseTriangularMatrix(_AbstractSquareMatrix):

    def __init__(self, inverse_array, lower=True):
        self._inverse_array = inverse_array
        self.lower = lower
        super().__init__(inverse_array.shape[0])

    @property
    def inverse_array(self):
        return self._inverse_array

    def _scalar_multiply(self, scalar):
        return InverseTriangularMatrix(self.inverse_array / scalar, self.lower)

    def _left_matrix_multiply(self, other):
        return sla.solve_triangular(
            self.inverse_array, other, lower=self.lower)

    def _right_matrix_multiply(self, other):
        return sla.solve_triangular(
            self.inverse_array, other.T, lower=self.lower, trans=1).T

    @property
    def inv(self):
        return TriangularMatrix(self.inverse_array, lower=self.lower)

    @property
    def T(self):
        return InverseTriangularMatrix(
            self.inverse_array.T, lower=not self.lower)

    @property
    def array(self):
        return self @ np.identity(self.shape[0])

    @property
    def diagonal(self):
        return 1. / self.inverse_array.diagonal()

    @property
    def log_abs_det(self):
        return -self.inv.log_abs_det


class _AbstractTriangularFactoredDefiniteMatrix(_AbstractSymmetricMatrix):

    def __init__(self, array, factor, sign=1):
        self._array = array
        self._factor = factor
        self._sign = sign
        if array is not None:
            super().__init__(size=array.shape[0], is_posdef=(sign == 1),
                             is_negdef=(sign == -1))
        elif factor is not None:
            super().__init__(size=factor.shape[0], is_posdef=(sign == 1),
                             is_negdef=(sign == -1))
        else:
            raise RuntimeError(
                'At least one of array and factor must be specified.')

    @property
    def factor(self):
        return self._factor

    @property
    def inv(self):
        return TriangularFactoredDefiniteMatrix(
            factor=self.factor.inv.T, sign=self._sign)

    @property
    def sqrt(self):
        if self._sign == 1:
            return self.factor
        else:
            raise NotImplementedError(
                'Square-root not supported for negative definite matrices.')

    @property
    def log_abs_det(self):
        return 2 * self.factor.log_abs_det

    @property
    def log_abs_det_sqrt(self):
        return self.factor.log_abs_det


class TriangularFactoredDefiniteMatrix(
        _AbstractTriangularFactoredDefiniteMatrix):

    def __init__(self, factor, lower=True, sign=1, array=None):
        if not isinstance(factor, (TriangularMatrix, InverseTriangularMatrix)):
            factor = TriangularMatrix(factor, lower)
        super().__init__(array=array, factor=factor, sign=sign)

    def _scalar_multiply(self, scalar):
        return TriangularFactoredDefiniteMatrix(
            factor=abs(scalar)**0.5 * self.factor,
            sign=self._sign * np.sign(scalar))

    def _left_matrix_multiply(self, other):
        return self._sign * (self.factor @ (self.factor.T @ other))

    def _right_matrix_multiply(self, other):
        return self._sign * ((other @ self.factor) @ self.factor.T)

    @property
    def grad_log_abs_det_sqrt(self):
        return self.factor.inv.T.array

    def grad_quadratic_form_inv(self, vector):
        inv_factor_vector = self.factor.inv @ vector
        inv_vector = self.inv @ vector
        return -2 * self._sign * np.outer(inv_vector, inv_factor_vector)

    @property
    def array(self):
        if self._array is None:
            self._array = self._sign * (self.factor @ self.factor.array.T)
        return self._array


class DenseDefiniteMatrix(_AbstractTriangularFactoredDefiniteMatrix):

    def __init__(self, array, sign=1, factor=None):
        super().__init__(array=array, factor=factor, sign=sign)

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
    def grad_log_abs_det_sqrt(self):
        return 0.5 * self.inv.array

    def grad_quadratic_form_inv(self, vector):
        inv_metric_vector = self.inv @ vector
        return -np.outer(inv_metric_vector, inv_metric_vector)


class DensePositiveDefiniteMatrix(DenseDefiniteMatrix):

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


class DenseSquareMatrix(_AbstractSquareMatrix):

    def __init__(self, array, transposed=False, lu_and_piv=None):
        self._array = array
        self._transposed = transposed
        self._lu_and_piv = lu_and_piv
        super().__init__(array.shape[0])

    def _scalar_multiply(self, scalar):
        if self._lu_and_piv is None:
            return DenseSquareMatrix(scalar * self.array, self._transposed)
        else:
            old_lu, piv = self._lu_and_piv
            # Multiply upper-triangle by scalar
            new_lu = old_lu + (scalar - 1) * np.triu(old_lu)
            return DenseSquareMatrix(
                scalar * self.array, self._transposed, (new_lu, piv))

    @property
    def lu_and_piv(self):
        if self._lu_and_piv is None:
            self._lu_and_piv = sla.lu_factor(self.array)
        return self._lu_and_piv

    @property
    def log_abs_det(self):
        return np.log(np.abs(self.lu_and_piv[0].diagonal())).sum()

    @property
    def T(self):
        return DenseSquareMatrix(
            self.array.T, not self._transposed, self.lu_and_piv)

    @property
    def inv(self):
        return InverseLUFactoredSquareMatrix(
            self.array, self._transposed, self.lu_and_piv)


class InverseLUFactoredSquareMatrix(_AbstractSquareMatrix):

    def __init__(self, inverse_array, transposed, inverse_lu_and_piv):
        self._inverse_array = inverse_array
        self._inverse_lu_and_piv = inverse_lu_and_piv
        self._transposed = transposed
        super().__init__(inverse_array.shape[0])

    def _scalar_multiply(self, scalar):
        old_inv_lu, piv = self._inverse_lu_and_piv
        # Divide upper-triangle by scalar
        new_inv_lu = old_inv_lu - (scalar - 1) / scalar * np.triu(old_inv_lu)
        return InverseLUFactoredSquareMatrix(
            self._inverse_array / scalar, self._transposed, (new_inv_lu, piv))

    def _left_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inverse_lu_and_piv, other, 1 if self._transposed else 0)

    def _right_matrix_multiply(self, other):
        return sla.lu_solve(
            self._inverse_lu_and_piv, other.T, 0 if self._transposed else 1).T

    @property
    def log_abs_det(self):
        return -np.log(np.abs(self._inverse_lu_and_piv[0].diagonal())).sum()

    @property
    def array(self):
        return self @ np.identity(self.shape[0])

    @property
    def inv(self):
        return DenseSquareMatrix(
            self._inverse_array, self._transposed, self._inverse_lu_and_piv)

    @property
    def T(self):
        return InverseLUFactoredSquareMatrix(
            self._inverse_array, not self._transposed,
            self._inverse_lu_and_piv)


class OrthogonalMatrix(_AbstractSquareMatrix):

    def __init__(self, array):
        self._array = array
        super().__init__(array.shape[0])

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


class ScaledOrthogonalMatrix(_AbstractSquareMatrix):

    def __init__(self, scalar, orth_array):
        self._scalar = scalar
        self._orth_array = orth_array
        super().__init__(orth_array.shape[0])

    def _scalar_multiply(self, scalar):
        return ScaledOrthogonalMatrix(scalar * self._scalar, self._orth_array)

    @property
    def array(self):
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


class EigendecomposedSymmetricMatrix(_AbstractSymmetricMatrix):

    def __init__(self, eigvec, eigval):
        if isinstance(eigvec, np.ndarray):
            eigvec = OrthogonalMatrix(eigvec)
        super().__init__(eigvec.shape[0], is_posdef=np.all(eigval > 0),
                         is_negdef=np.all(eigval < 0))
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
    def sqrt(self):
        return EigendecomposedSymmetricMatrix(
            self.eigvec, self.eigval**0.5)

    @property
    def inv(self):
        return EigendecomposedSymmetricMatrix(
            self.eigvec, 1 / self.eigval)

    @property
    def array(self):
        if self.shape[0] is None:
            raise RuntimeError(
                'Cannot get array representation for symmetric '
                'eigendecomposed matrix with implicit size.')
        return self @ np.identity(self.shape[0])


class SoftAbsRegularisedPositiveDefiniteMatrix(EigendecomposedSymmetricMatrix):

    def __init__(self, symmetric_array, softabs_coeff):
        """
        Args:
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
    def grad_log_abs_det_sqrt(self):
        return 0.5 * (
            self.eigvec @ DiagonalMatrix(
                self.grad_softabs(self.unreg_eigval) / self.eigval) @
            self.eigvec.T)

    def grad_quadratic_form_inv(self, vector):
        num_j_mtx = self.eigval[:, None] - self.eigval[None, :]
        num_j_mtx += np.diag(self.grad_softabs(self.unreg_eigval))
        den_j_mtx = self.unreg_eigval[:, None] - self.unreg_eigval[None, :]
        np.fill_diagonal(den_j_mtx, 1)
        j_mtx = num_j_mtx / den_j_mtx
        e_vct = (self.eigvec.T @ vector) / self.eigval
        return -(
            (self.eigvec @ (np.outer(e_vct, e_vct) * j_mtx)) @ self.eigvec.T)


class MatrixProduct(_AbstractMatrix):

    def __init__(self, matrices):
        self._matrices = matrices
        super().__init__((matrices[0].shape[0], matrices[-1].shape[1]))

    def _scalar_multiply(self, scalar):
        return MatrixProduct((
            ScaledIdentityMatrix(scalar, self.shape[0]), *self.matrices))

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
            [matrix.T for matrix in reversed(self._matrices)])

    @property
    def array(self):
        return self @ np.identity(self.shape[1])


class SquareBlockDiagonalMatrix(_AbstractSquareMatrix):

    def __init__(self, *blocks):
        sizes = tuple(block.shape[0] for block in blocks)
        super().__init__(size=sum(sizes))
        self._blocks = tuple(blocks)
        self._sizes = sizes
        self._splits = np.cumsum(sizes[:-1])

    @property
    def blocks(self):
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
        return type(self)(*(scalar * block for block in self._blocks))

    @property
    def T(self):
        return SquareBlockDiagonalMatrix(*(block.T for block in self._blocks))

    @property
    def sqrt(self):
        return SquareBlockDiagonalMatrix(
            *(block.sqrt for block in self._blocks))

    @property
    def inv(self):
        return type(self)(*(block.inv for block in self._blocks))

    @property
    def eigval(self):
        return np.concatenate([block.eigval for block in self._blocks])

    @property
    def eigvec(self):
        return SquareBlockDiagonalMatrix(
            *(block.eigvec for block in self._blocks))

    @property
    def array(self):
        return sla.block_diag(self._blocks)

    @property
    def log_abs_det(self):
        return sum(block.log_abs_det for block in self._blocks)


class SymmetricBlockDiagonalMatrix(SquareBlockDiagonalMatrix):

    def __init__(self, *blocks):
        assert all(block is block.T for block in blocks)
        SquareBlockDiagonalMatrix.__init__(self, *blocks)
        self._is_posdef = all(block.is_posdef for block in blocks)
        self._is_negdef = all(block.is_negdef for block in blocks)

    @property
    def is_posdef(self):
        return self._is_posdef

    @property
    def is_negdef(self):
        return self._is_negdef
