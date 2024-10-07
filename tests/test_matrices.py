from copy import copy, deepcopy
from functools import reduce
from itertools import product

import numpy as np
import numpy.linalg as nla
import numpy.testing as npt
import pytest
import scipy.linalg as sla

from mici import matrices

AUTOGRAD_AVAILABLE = True
try:
    import autograd.numpy as anp
    from autograd import grad
    from autograd.core import defvjp, primitive
except ImportError:
    AUTOGRAD_AVAILABLE = False
    import warnings

    warnings.warn("Autograd not available. Skipping gradient tests.", stacklevel=2)

SEED = 3046987125
NUM_SCALAR = 2
SIZES = {1, 10}
ATOL = 1e-10


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(params=SIZES)
def size(request):
    return request.param


@pytest.fixture
def shape_0(size):
    return size


@pytest.fixture
def shape_1(size):
    return size


@pytest.fixture(params=((), (1,), None))
def premultiplier(rng, shape_0, request):
    shape = (request.param if request.param is not None else (shape_0,)) + (shape_0,)
    return rng.standard_normal(shape)


@pytest.fixture(params=((), (1,), None))
def postmultiplier(rng, shape_1, request):
    shape = (shape_1,) + (request.param if request.param is not None else (shape_1,))
    return rng.standard_normal(shape)


@pytest.fixture(params=(1 if i % 2 == 0 else -1 for i in range(NUM_SCALAR)))
def scalar(rng, request):
    # Ensure a mix of positive and negative scalar multipliers
    return request.param * np.abs(rng.standard_normal())


@pytest.fixture
def matrix(matrix_pair):
    return matrix_pair[0]


@pytest.fixture
def np_matrix(matrix_pair):
    return matrix_pair[1]


@pytest.fixture
def vector(rng, shape_0):
    return rng.standard_normal(shape_0)


class MatrixTests:
    def test_self_equality(self, matrix):
        assert matrix == matrix  # noqa: PLR0124

    def test_hashable(self, matrix):
        assert hash(matrix) == hash(matrix)

    def test_copy_equality(self, matrix):
        matrix_copy = copy(matrix)
        assert matrix == matrix_copy
        assert hash(matrix) == hash(matrix_copy)

    def test_deepcopy_equality(self, matrix):
        matrix_copy = deepcopy(matrix)
        assert matrix == matrix_copy
        assert hash(matrix) == hash(matrix_copy)

    def test_shape(self, matrix, np_matrix):
        assert matrix.shape in {(None, None), np_matrix.shape}

    def test_lmult(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose(matrix @ postmultiplier, np_matrix @ postmultiplier)

    def test_rmult(self, matrix, np_matrix, premultiplier):
        npt.assert_allclose(premultiplier @ matrix, premultiplier @ np_matrix)

    def test_neg_lmult(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose((-matrix) @ postmultiplier, -np_matrix @ postmultiplier)

    def test_lmult_rmult_trans(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose(matrix @ postmultiplier, (postmultiplier.T @ matrix.T).T)

    def test_rmult_lmult_trans(self, matrix, np_matrix, premultiplier):
        npt.assert_allclose(premultiplier @ matrix, (matrix.T @ premultiplier.T).T)

    def test_lmult_scalar_lmult(self, matrix, np_matrix, scalar, postmultiplier):
        npt.assert_allclose(
            (scalar * matrix) @ postmultiplier,
            scalar * np_matrix @ postmultiplier,
        )

    def test_rdiv_scalar_lmult(self, matrix, np_matrix, scalar, postmultiplier):
        npt.assert_allclose(
            (matrix / scalar) @ postmultiplier,
            (np_matrix / scalar) @ postmultiplier,
        )

    def test_rmult_scalar_lmult(self, matrix, np_matrix, scalar, postmultiplier):
        npt.assert_allclose(
            (matrix * scalar) @ postmultiplier,
            (np_matrix * scalar) @ postmultiplier,
        )

    def test_lmult_scalar_rmult(self, matrix, np_matrix, scalar, premultiplier):
        npt.assert_allclose(
            premultiplier @ (scalar * matrix),
            premultiplier @ (scalar * np_matrix),
        )

    def test_rmult_scalar_rmult(self, matrix, np_matrix, scalar, premultiplier):
        npt.assert_allclose(
            premultiplier @ (matrix * scalar),
            premultiplier @ (np_matrix * scalar),
        )

    def test_invalid_scalar_mult(self, matrix):
        with pytest.raises(NotImplementedError):
            0 * matrix
        with pytest.raises(TypeError, match="unsupported operand type[(]s[)] for \\*:"):
            None * matrix

    def test_invalid_scalar_div(self, matrix):
        with pytest.raises(NotImplementedError):
            matrix / 0
        with pytest.raises(TypeError, match="unsupported operand type[(]s[)] for /:"):
            matrix / None

    def test_to_string(self, matrix):
        assert isinstance(str(matrix), str)

    def test_repr(self, matrix):
        assert isinstance(repr(matrix), str)


class ExplicitShapeMatrixTests(MatrixTests):
    def test_matrix_inequality_different_shapes(self, matrix):
        different_shape_matrix = matrices.IdentityMatrix(matrix.shape[0] + 1)
        assert matrix != different_shape_matrix
        assert hash(matrix) != hash(different_shape_matrix)

    def test_array(self, matrix, np_matrix):
        npt.assert_allclose(matrix.array, np_matrix, atol=ATOL)

    def test_array_transpose(self, matrix, np_matrix):
        npt.assert_allclose(matrix.T.array, np_matrix.T, atol=ATOL)

    def test_array_transpose_transpose(self, matrix, np_matrix):
        npt.assert_allclose(matrix.T.T.array, np_matrix, atol=ATOL)

    def test_array_numpy(self, matrix, np_matrix):
        npt.assert_allclose(matrix, np_matrix, atol=ATOL)

    def test_diagonal(self, matrix, np_matrix):
        npt.assert_allclose(matrix.diagonal, np_matrix.diagonal(), atol=ATOL)

    def test_lmult_scalar_array(self, matrix, np_matrix, scalar):
        npt.assert_allclose((scalar * matrix).array, scalar * np_matrix, atol=ATOL)

    def test_rmult_scalar_array(self, matrix, np_matrix, scalar):
        npt.assert_allclose((matrix * scalar).array, np_matrix * scalar, atol=ATOL)

    def test_rdiv_scalar_array(self, matrix, np_matrix, scalar):
        npt.assert_allclose((matrix / scalar).array, np_matrix / scalar, atol=ATOL)

    def test_neg_array(self, matrix, np_matrix):
        npt.assert_allclose((-matrix).array, -np_matrix, atol=ATOL)

    def test_invalid_shape_matmul(self, matrix):
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            matrix @ matrices.IdentityMatrix(matrix.shape[1] + 1)
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            matrices.IdentityMatrix(matrix.shape[0] + 1) @ matrix


class SquareMatrixTests(MatrixTests):
    def test_quadratic_form(self, matrix, np_matrix, vector):
        npt.assert_allclose(vector @ matrix @ vector, vector @ np_matrix @ vector)


class ExplicitShapeSquareMatrixTests(SquareMatrixTests, ExplicitShapeMatrixTests):
    def test_log_abs_det(self, matrix, np_matrix):
        npt.assert_allclose(matrix.log_abs_det, nla.slogdet(np_matrix)[1], atol=ATOL)


class SymmetricMatrixTests(SquareMatrixTests):
    def test_symmetry_identity(self, matrix, np_matrix):
        assert matrix is matrix.T

    def test_symmetry_lmult(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose(matrix @ postmultiplier, (postmultiplier.T @ matrix).T)

    def test_symmetry_rmult(self, matrix, np_matrix, premultiplier):
        npt.assert_allclose(premultiplier @ matrix, (matrix @ premultiplier.T).T)


class ExplicitShapeSymmetricMatrixTests(
    SymmetricMatrixTests,
    ExplicitShapeSquareMatrixTests,
):
    def test_symmetry_array(self, matrix, np_matrix):
        npt.assert_allclose(matrix.array, matrix.T.array)

    def test_eigval(self, matrix, np_matrix):
        # Ensure eigenvalues in ascending order
        npt.assert_allclose(np.sort(matrix.eigval), nla.eigh(np_matrix)[0])

    def test_eigvec(self, matrix, np_matrix):
        # Ensure eigenvectors correspond to ascending eigenvalue ordering
        eigval_order = np.argsort(matrix.eigval)
        eigvec = matrix.eigvec.array[:, eigval_order]
        np_eigvec = nla.eigh(np_matrix)[1]
        # Account for eigenvector sign ambiguity when checking for equivalence
        assert np.all(np.isclose(eigvec, np_eigvec) | np.isclose(eigvec, -np_eigvec))


class InvertibleMatrixTests(MatrixTests):
    def test_lmult_inv(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose(
            matrix.inv @ postmultiplier,
            nla.solve(np_matrix, postmultiplier),
        )

    def test_rmult_inv(self, matrix, np_matrix, premultiplier):
        npt.assert_allclose(
            premultiplier @ matrix.inv,
            nla.solve(np_matrix.T, premultiplier.T).T,
        )

    def test_lmult_scalar_inv_lmult(self, matrix, np_matrix, scalar, postmultiplier):
        npt.assert_allclose(
            (scalar * matrix.inv) @ postmultiplier,
            nla.solve(np_matrix / scalar, postmultiplier),
        )

    def test_inv_lmult_scalar_lmult(self, matrix, np_matrix, scalar, postmultiplier):
        npt.assert_allclose(
            (scalar * matrix).inv @ postmultiplier,
            nla.solve(scalar * np_matrix, postmultiplier),
        )

    def test_quadratic_form_inv(self, matrix, np_matrix, vector):
        npt.assert_allclose(
            vector @ matrix.inv @ vector,
            vector @ nla.solve(np_matrix, vector),
        )


class ExplicitShapeInvertibleMatrixTests(
    ExplicitShapeSquareMatrixTests,
    InvertibleMatrixTests,
):
    def test_array_inv(self, matrix, np_matrix):
        npt.assert_allclose(matrix.inv.array, nla.inv(np_matrix), atol=ATOL)

    def test_array_inv_inv(self, matrix, np_matrix):
        npt.assert_allclose(matrix.inv.inv.array, np_matrix, atol=ATOL)

    def test_log_abs_det_inv(self, matrix, np_matrix):
        npt.assert_allclose(
            matrix.inv.log_abs_det,
            -nla.slogdet(np_matrix)[1],
            atol=ATOL,
        )


class PositiveDefiniteMatrixTests(SymmetricMatrixTests, InvertibleMatrixTests):
    def test_pos_def(self, matrix, np_matrix, vector):
        assert vector @ matrix @ vector > 0

    def test_lmult_sqrt(self, matrix, np_matrix, postmultiplier):
        npt.assert_allclose(
            matrix.sqrt @ (matrix.sqrt.T @ postmultiplier),
            np_matrix @ postmultiplier,
        )

    def test_rmult_sqrt(self, matrix, np_matrix, premultiplier):
        npt.assert_allclose(
            (premultiplier @ matrix.sqrt) @ matrix.sqrt.T,
            premultiplier @ np_matrix,
        )

    def test_inv_is_posdef(self, matrix, np_matrix):
        assert isinstance(matrix.inv, matrices.PositiveDefiniteMatrix)

    def test_pos_scalar_multiple_is_posdef(self, matrix, np_matrix):
        assert isinstance(matrix * 2, matrices.PositiveDefiniteMatrix)


class ExplicitShapePositiveDefiniteMatrixTests(
    PositiveDefiniteMatrixTests,
    ExplicitShapeInvertibleMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
):
    def test_sqrt_array(self, matrix, np_matrix):
        npt.assert_allclose((matrix.sqrt @ matrix.sqrt.T).array, np_matrix)


class DifferentiableMatrixTests(MatrixTests):
    if AUTOGRAD_AVAILABLE:

        @pytest.fixture
        def grad_log_abs_det(self, matrix):
            param = self.get_param(matrix)
            return grad(lambda p: anp.linalg.slogdet(self.param_func(p, matrix))[1])(
                param,
            )

        @pytest.fixture
        def grad_quadratic_form_inv(self, matrix):
            param = self.get_param(matrix)
            return lambda v: grad(
                lambda p: v @ anp.linalg.solve(self.param_func(p, matrix), v),
            )(param)

        def test_grad_log_abs_det(self, matrix, grad_log_abs_det):
            # Use non-zero atol to allow for floating point errors in gradients
            # analytically equal to zero
            npt.assert_allclose(matrix.grad_log_abs_det, grad_log_abs_det, atol=ATOL)

        def test_grad_quadratic_form_inv(self, matrix, vector, grad_quadratic_form_inv):
            # Use non-zero atol to allow for floating point errors in gradients
            # analytically equal to zero
            npt.assert_allclose(
                matrix.grad_quadratic_form_inv(vector),
                grad_quadratic_form_inv(vector),
                atol=ATOL,
            )


class TestImplicitIdentityMatrix(SymmetricMatrixTests, InvertibleMatrixTests):
    @pytest.fixture
    def matrix_pair(self, size):
        return matrices.IdentityMatrix(None), np.identity(size)


class TestIdentityMatrix(ExplicitShapePositiveDefiniteMatrixTests):
    @pytest.fixture
    def matrix_pair(self, size):
        return matrices.IdentityMatrix(size), np.identity(size)


class TestImplicitScaledIdentityMatrix(InvertibleMatrixTests, SymmetricMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        scalar = rng.normal()
        return (matrices.ScaledIdentityMatrix(scalar, None), scalar * np.identity(size))


class DifferentiableScaledIdentityMatrixTests(DifferentiableMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        scalar = self.generate_scalar(rng)
        return self.matrix_class(scalar, size), scalar * np.identity(size)

    @staticmethod
    def param_func(param, matrix):
        return param * anp.eye(matrix.shape[0])

    @staticmethod
    def get_param(matrix):
        return matrix._scalar


class TestScaledIdentityMatrix(
    DifferentiableScaledIdentityMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
    ExplicitShapeInvertibleMatrixTests,
):
    matrix_class = matrices.ScaledIdentityMatrix

    @staticmethod
    def generate_scalar(rng):
        return rng.normal()


class TestPositiveScaledIdentityMatrix(
    DifferentiableScaledIdentityMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    matrix_class = matrices.PositiveScaledIdentityMatrix

    @staticmethod
    def generate_scalar(rng):
        return abs(rng.normal())


class DifferentiableDiagonalMatrixTests(DifferentiableMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        diagonal = self.generate_diagonal(size, rng)
        return self.matrix_class(diagonal), np.diag(diagonal)

    @staticmethod
    def param_func(param, _matrix):
        return anp.diag(param)

    @staticmethod
    def get_param(matrix):
        return matrix.diagonal


class TestDiagonalMatrix(
    DifferentiableDiagonalMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
    ExplicitShapeInvertibleMatrixTests,
):
    matrix_class = matrices.DiagonalMatrix

    @staticmethod
    def generate_diagonal(size, rng):
        return rng.standard_normal(size)


class TestPositiveDiagonalMatrix(
    DifferentiableDiagonalMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    matrix_class = matrices.PositiveDiagonalMatrix

    @staticmethod
    def generate_diagonal(size, rng):
        return abs(rng.standard_normal(size))


class TestTriangularMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture(params=(True, False))
    def matrix_pair(self, rng, size, request):
        lower = request.param
        array = rng.standard_normal((size, size))
        tri_array = np.tril(array) if lower else np.triu(array)
        return matrices.TriangularMatrix(tri_array, lower=lower), tri_array


class TestInverseTriangularMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture(params=(True, False))
    def matrix_pair(self, rng, size, request):
        lower = request.param
        array = rng.standard_normal((size, size))
        inv_tri_array = np.tril(array) if lower else np.triu(array)
        return (
            matrices.InverseTriangularMatrix(inv_tri_array, lower=lower),
            nla.inv(inv_tri_array),
        )


class DifferentiableTriangularFactoredDefiniteMatrixTests(DifferentiableMatrixTests):
    @pytest.fixture(params=(True, False))
    def matrix_pair(self, rng, size, sign, request):
        factor_is_lower = request.param
        array = rng.standard_normal((size, size))
        tri_array = sla.cholesky(array @ array.T, factor_is_lower)
        return (
            self.matrix_class(tri_array, sign, factor_is_lower=factor_is_lower),
            sign * tri_array @ tri_array.T,
        )

    @staticmethod
    def param_func(param, matrix):
        param = anp.tril(param) if matrix.factor.lower else anp.triu(param)
        return param @ param.T

    @staticmethod
    def get_param(matrix):
        return matrix.factor.array


class TestTriangularFactoredDefiniteMatrix(
    DifferentiableTriangularFactoredDefiniteMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
    ExplicitShapeInvertibleMatrixTests,
):
    matrix_class = matrices.TriangularFactoredDefiniteMatrix

    @pytest.fixture(params=(+1, -1))
    def sign(self, request):
        return request.param


class TestTriangularFactoredPositiveDefiniteMatrix(
    DifferentiableTriangularFactoredDefiniteMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @staticmethod
    def matrix_class(factor, _sign, factor_is_lower):
        return matrices.TriangularFactoredPositiveDefiniteMatrix(
            factor,
            factor_is_lower=factor_is_lower,
        )

    @pytest.fixture
    def sign(self):
        return 1


class DifferentiableDenseDefiniteMatrixTests(DifferentiableMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size, sign):
        sqrt_array = rng.standard_normal((size, size))
        array = sign * sqrt_array @ sqrt_array.T
        return self.matrix_class(array, is_posdef=(sign == 1)), array

    @staticmethod
    def param_func(param, _matrix):
        return param

    @staticmethod
    def get_param(matrix):
        return matrix.array


class TestDenseDefiniteMatrix(
    DifferentiableDenseDefiniteMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
    ExplicitShapeInvertibleMatrixTests,
):
    matrix_class = matrices.DenseDefiniteMatrix

    @pytest.fixture(params=(+1, -1))
    def sign(self, request):
        return request.param


class TestDensePositiveDefiniteMatrix(
    DifferentiableDenseDefiniteMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @staticmethod
    def matrix_class(array, _is_posdef):
        return matrices.DensePositiveDefiniteMatrix(array)

    @pytest.fixture
    def sign(self):
        return 1


class TestDensePositiveDefiniteProductMatrix(
    DifferentiableMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @pytest.fixture(params=(lambda s: s + 1, lambda s: s * 2))
    def size_inner(self, size, request):
        return request.param(size)

    @pytest.fixture
    def matrix_pair(self, rng, size, size_inner):
        rect_matrix = rng.standard_normal((size, size_inner))
        pos_def_matrix = rng.standard_normal((size_inner, size_inner))
        pos_def_matrix = pos_def_matrix @ pos_def_matrix.T
        array = rect_matrix @ pos_def_matrix @ rect_matrix.T
        return (
            matrices.DensePositiveDefiniteProductMatrix(rect_matrix, pos_def_matrix),
            array,
        )

    @staticmethod
    def param_func(param, matrix):
        return param @ matrix._pos_def_matrix @ param.T

    @staticmethod
    def get_param(matrix):
        return matrix._rect_matrix.array


class TestDenseSquareMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        array = rng.standard_normal((size, size))
        return matrices.DenseSquareMatrix(array), array


class TestInverseLUFactoredSquareMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture(params=[True, False])
    def matrix_pair(self, rng, size, request):
        transposed = request.param
        inverse_array = rng.standard_normal((size, size))
        inverse_lu_and_piv = sla.lu_factor(
            inverse_array.T if transposed else inverse_array,
        )
        array = nla.inv(inverse_array)
        return (
            matrices.InverseLUFactoredSquareMatrix(
                inverse_array,
                inv_lu_and_piv=inverse_lu_and_piv,
                inv_lu_transposed=transposed,
            ),
            array,
        )


class TestDenseSymmetricMatrix(
    ExplicitShapeInvertibleMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        array = rng.standard_normal((size, size))
        array = array + array.T
        return matrices.DenseSymmetricMatrix(array), array


class TestOrthogonalMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        array = nla.qr(rng.standard_normal((size, size)))[0]
        return matrices.OrthogonalMatrix(array), array


class TestScaledOrthogonalMatrix(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        orth_array = nla.qr(rng.standard_normal((size, size)))[0]
        scalar = rng.standard_normal()
        return (
            matrices.ScaledOrthogonalMatrix(scalar, orth_array),
            scalar * orth_array,
        )


class TestEigendecomposedSymmetricMatrix(
    ExplicitShapeInvertibleMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        eigvec = nla.qr(rng.standard_normal((size, size)))[0]
        eigval = rng.standard_normal(size)
        return (
            matrices.EigendecomposedSymmetricMatrix(eigvec, eigval),
            (eigvec * eigval) @ eigvec.T,
        )


class TestEigendecomposedPositiveDefiniteMatrix(
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size):
        eigvec = nla.qr(rng.standard_normal((size, size)))[0]
        eigval = np.abs(rng.standard_normal(size))
        return (
            matrices.EigendecomposedPositiveDefiniteMatrix(eigvec, eigval),
            (eigvec * eigval) @ eigvec.T,
        )


class TestSoftAbsRegularizedPositiveDefiniteMatrix(
    DifferentiableMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @pytest.fixture(params=(0.5, 1.0, 1.5))
    def matrix_pair(self, rng, size, request):
        softabs_coeff = request.param
        sym_array = rng.standard_normal((size, size))
        sym_array = sym_array + sym_array.T
        unreg_eigval, eigvec = np.linalg.eigh(sym_array)
        eigval = unreg_eigval / np.tanh(unreg_eigval * softabs_coeff)
        return (
            matrices.SoftAbsRegularizedPositiveDefiniteMatrix(sym_array, softabs_coeff),
            (eigvec * eigval) @ eigvec.T,
        )

    @staticmethod
    def get_param(matrix):
        eigvec = matrix.eigvec.array
        return (eigvec * matrix.unreg_eigval) @ eigvec.T

    @staticmethod
    def param_func(param, matrix):
        softabs_coeff = matrix._softabs_coeff
        sym_array = (param + param.T) / 2
        unreg_eigval, eigvec = anp.linalg.eigh(sym_array)
        eigval = unreg_eigval / anp.tanh(unreg_eigval * softabs_coeff)
        return (eigvec * eigval) @ eigvec.T


class TestMatrixProduct(ExplicitShapeMatrixTests):
    @pytest.fixture(params=product((2, 4), (True, False)))
    def matrix_pair(self, rng, size, request):
        n_terms, explicit = request.param
        arrays = [
            rng.standard_normal(
                (size if t % 2 == 0 else 2 * size, 2 * size if t % 2 == 0 else size),
            )
            for t in range(n_terms)
        ]
        matrices_ = [matrices.DenseRectangularMatrix(a) for a in arrays]
        np_matrix = nla.multi_dot(arrays)
        if explicit:
            return matrices.MatrixProduct(matrices_), np_matrix
        return reduce(lambda a, b: a @ b, matrices_), np_matrix


class TestSquareMatrixProduct(ExplicitShapeSquareMatrixTests):
    @pytest.fixture(params=(2, 5))
    def matrix_pair(self, rng, size, request):
        n_terms = request.param
        arrays = [rng.standard_normal((size, size)) for _ in range(n_terms)]
        matrix = matrices.SquareMatrixProduct(
            [matrices.DenseSquareMatrix(a) for a in arrays],
        )
        return matrix, nla.multi_dot(arrays)


class TestInvertibleMatrixProduct(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture(params=product((2, 5), (True, False)))
    def matrix_pair(self, rng, size, request):
        n_terms, explicit = request.param
        arrays = [rng.standard_normal((size, size)) for _ in range(n_terms)]
        matrices_ = [matrices.DenseSquareMatrix(a) for a in arrays]
        if explicit:
            matrix = matrices.InvertibleMatrixProduct(matrices_)
        else:
            matrix = reduce(lambda a, b: a @ b, matrices_)
        return matrix, nla.multi_dot(arrays)


class InvertibleBlockMatrixTests(ExplicitShapeInvertibleMatrixTests):
    @pytest.fixture(params=(1, 2, 5))
    def n_block(self, request):
        return request.param

    @pytest.fixture
    def shape_0(self, size, n_block):
        return size * n_block

    @pytest.fixture
    def shape_1(self, size, n_block):
        return size * n_block


class TestSquareBlockDiagonalMatrix(InvertibleBlockMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size, n_block):
        arrays = [rng.standard_normal((size, size)) for _ in range(n_block)]
        return (
            matrices.SquareBlockDiagonalMatrix(
                matrices.DenseSquareMatrix(arr) for arr in arrays
            ),
            sla.block_diag(*arrays),
        )


class TestSymmetricBlockDiagonalMatrix(
    InvertibleBlockMatrixTests,
    ExplicitShapeSymmetricMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size, n_block):
        arrays = [rng.standard_normal((size, size)) for _ in range(n_block)]
        arrays = [arr + arr.T for arr in arrays]
        return (
            matrices.SymmetricBlockDiagonalMatrix(
                matrices.DenseSymmetricMatrix(arr) for arr in arrays
            ),
            sla.block_diag(*arrays),
        )


if AUTOGRAD_AVAILABLE:
    # Define new block_diag primitive and corresponding vector-Jacobian-product

    @primitive
    def block_diag(blocks):
        return sla.block_diag(*blocks)

    def vjp_block_diag(_ans, blocks):
        blocks = tuple(blocks)

        def vjp(g):
            i, j = 0, 0
            vjp_blocks = []
            for block in blocks:
                j += block.shape[0]
                vjp_blocks.append(g[i:j, i:j])
                i = j
            return tuple(vjp_blocks)

        return vjp

    defvjp(block_diag, vjp_block_diag)


class TestPositiveDefiniteBlockDiagonalMatrix(
    InvertibleBlockMatrixTests,
    DifferentiableMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size, n_block):
        arrays = [rng.standard_normal((size, size)) for _ in range(n_block)]
        arrays = [arr @ arr.T for arr in arrays]
        return (
            matrices.PositiveDefiniteBlockDiagonalMatrix(
                matrices.DensePositiveDefiniteMatrix(arr) for arr in arrays
            ),
            sla.block_diag(*arrays),
        )

    @staticmethod
    def get_param(matrix):
        return tuple(block.array for block in matrix._blocks)

    @staticmethod
    def param_func(param, _matrix):
        return block_diag(param)


class TestDenseRectangularMatrix(ExplicitShapeMatrixTests):
    @pytest.fixture(params=SIZES)
    def shape_1(self, request):
        return request.param

    @pytest.fixture
    def matrix_pair(self, rng, shape_0, shape_1):
        array = rng.standard_normal((shape_0, shape_1))
        return matrices.DenseRectangularMatrix(array), array


class TestBlockRowMatrix(ExplicitShapeMatrixTests):
    @pytest.fixture(params=(2, 5))
    def n_block(self, request):
        return request.param

    @pytest.fixture
    def shape_1(self, size, n_block):
        return size * n_block

    @pytest.fixture
    def matrix_pair(self, rng, size, n_block):
        blocks = [rng.standard_normal((size, size)) for _ in range(n_block)]
        return (
            matrices.BlockRowMatrix(
                matrices.DenseSquareMatrix(block) for block in blocks
            ),
            np.hstack(blocks),
        )


class TestBlockColumnMatrix(ExplicitShapeMatrixTests):
    @pytest.fixture(params=(2, 5))
    def n_block(self, request):
        return request.param

    @pytest.fixture
    def shape_0(self, size, n_block):
        return size * n_block

    @pytest.fixture
    def matrix_pair(self, rng, size, n_block):
        blocks = [rng.standard_normal((size, size)) for _ in range(n_block)]
        return (
            matrices.BlockColumnMatrix(
                matrices.DenseSquareMatrix(block) for block in blocks
            ),
            np.vstack(blocks),
        )


class SquareLowRankUpdateMatrixTests(ExplicitShapeSquareMatrixTests):
    @pytest.fixture(
        params=(lambda size: max(1, size // 2), lambda size: max(1, size - 1)),
    )
    def size_inner(self, size, request):
        return request.param(size)


class TestSquareLowRankUpdateMatrix(SquareLowRankUpdateMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size, size_inner):
        left_factor_matrix = rng.standard_normal((size, size_inner))
        right_factor_matrix = rng.standard_normal((size_inner, size))
        inner_square_matrix = rng.standard_normal((size_inner, size_inner))
        square_matrix = rng.standard_normal((size, size))
        return (
            matrices.SquareLowRankUpdateMatrix(
                matrices.DenseRectangularMatrix(left_factor_matrix),
                matrices.DenseRectangularMatrix(right_factor_matrix),
                matrices.DenseSquareMatrix(square_matrix),
                matrices.DenseSquareMatrix(inner_square_matrix),
            ),
            square_matrix
            + left_factor_matrix @ (inner_square_matrix @ right_factor_matrix),
        )


class TestNoInnerMatrixSquareLowRankUpdateMatrix(SquareLowRankUpdateMatrixTests):
    @pytest.fixture
    def matrix_pair(self, rng, size, size_inner):
        left_factor_matrix = rng.standard_normal((size, size_inner))
        right_factor_matrix = rng.standard_normal((size_inner, size))
        square_matrix = rng.standard_normal((size, size))
        return (
            matrices.SquareLowRankUpdateMatrix(
                matrices.DenseRectangularMatrix(left_factor_matrix),
                matrices.DenseRectangularMatrix(right_factor_matrix),
                matrices.DenseSquareMatrix(square_matrix),
            ),
            square_matrix + left_factor_matrix @ right_factor_matrix,
        )


class TestSymmetricLowRankUpdateMatrix(
    ExplicitShapeSymmetricMatrixTests,
    SquareLowRankUpdateMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size, size_inner):
        factor_matrix = rng.standard_normal((size, size_inner))
        inner_symmetric_matrix = rng.standard_normal((size_inner, size_inner))
        inner_symmetric_matrix = inner_symmetric_matrix + inner_symmetric_matrix.T
        symmetric_matrix = rng.standard_normal((size, size))
        symmetric_matrix = symmetric_matrix + symmetric_matrix.T
        return (
            matrices.SymmetricLowRankUpdateMatrix(
                matrices.DenseRectangularMatrix(factor_matrix),
                matrices.DenseSymmetricMatrix(symmetric_matrix),
                matrices.DenseSymmetricMatrix(inner_symmetric_matrix),
            ),
            symmetric_matrix
            + factor_matrix @ (inner_symmetric_matrix @ factor_matrix.T),
        )


class TestPositiveDefiniteLowRankUpdateMatrix(
    SquareLowRankUpdateMatrixTests,
    DifferentiableMatrixTests,
    ExplicitShapePositiveDefiniteMatrixTests,
):
    @pytest.fixture
    def matrix_pair(self, rng, size, size_inner):
        factor_matrix = rng.standard_normal((size, size_inner))
        inner_pos_def_matrix = rng.standard_normal((size_inner, size_inner))
        inner_pos_def_matrix = inner_pos_def_matrix @ inner_pos_def_matrix.T
        pos_def_matrix = rng.standard_normal((size, size))
        pos_def_matrix = pos_def_matrix @ pos_def_matrix.T
        return (
            matrices.PositiveDefiniteLowRankUpdateMatrix(
                matrices.DenseRectangularMatrix(factor_matrix),
                matrices.DensePositiveDefiniteMatrix(pos_def_matrix),
                matrices.DensePositiveDefiniteMatrix(inner_pos_def_matrix),
            ),
            pos_def_matrix + factor_matrix @ (inner_pos_def_matrix @ factor_matrix.T),
        )

    @staticmethod
    def param_func(param, matrix):
        return (
            matrix.pos_def_matrix.array + param @ matrix.inner_pos_def_matrix @ param.T
        )

    @staticmethod
    def get_param(matrix):
        return matrix.factor_matrix.array
