import mici.matrices as matrices
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import numpy.testing as npt
from functools import partial, wraps, reduce

AUTOGRAD_AVAILABLE = True
try:
    import autograd.numpy as anp
    from autograd import grad
    from autograd.core import primitive, defvjp
except ImportError:
    AUTOGRAD_AVAILABLE = False
    import warnings
    warnings.warn(
        'Autograd not available. Skipping gradient tests.')

SEED = 3046987125
NUM_SCALAR = 4
NUM_VECTOR = 4
SIZES = {1, 2, 5, 10}
ATOL = 1e-10


def iterate_over_matrix_pairs(test):

    @wraps(test)
    def iterated_test(self):
        for matrix_pair in self.matrix_pairs.values():
            yield (test, *matrix_pair)

    return iterated_test


def iterate_over_matrix_pairs_vectors(test):

    @wraps(test)
    def iterated_test(self):
        for key, (matrix, np_matrix) in self.matrix_pairs.items():
            for vector in self.vectors[np_matrix.shape[0]]:
                yield test, matrix, np_matrix, vector

    return iterated_test


def iterate_over_matrix_pairs_premultipliers(test):

    @wraps(test)
    def iterated_test(self):
        for key, (matrix, np_matrix) in self.matrix_pairs.items():
            for pre in self.premultipliers[np_matrix.shape[0]]:
                yield test, matrix, np_matrix, pre

    return iterated_test


def iterate_over_matrix_pairs_postmultipliers(test):

    @wraps(test)
    def iterated_test(self):
        for key, (matrix, np_matrix) in self.matrix_pairs.items():
            for post in self.postmultipliers[np_matrix.shape[1]]:
                yield test, matrix, np_matrix, post

    return iterated_test


def iterate_over_matrix_pairs_scalars(test):

    @wraps(test)
    def iterated_test(self):
        for matrix, np_matrix in self.matrix_pairs.values():
            for scalar in self.scalars:
                yield test, matrix, np_matrix, scalar

    return iterated_test


def iterate_over_matrix_pairs_scalars_postmultipliers(test):

    @wraps(test)
    def iterated_test(self):
        for matrix, np_matrix in self.matrix_pairs.values():
            for scalar in self.scalars:
                for post in self.postmultipliers[np_matrix.shape[1]]:
                    yield test, matrix, np_matrix, scalar, post

    return iterated_test


def iterate_over_matrix_pairs_scalars_premultipliers(test):

    @wraps(test)
    def iterated_test(self):
        for matrix, np_matrix in self.matrix_pairs.values():
            for scalar in self.scalars:
                for pre in self.premultipliers[np_matrix.shape[0]]:
                    yield test, matrix, np_matrix, scalar, pre

    return iterated_test


class MatrixTestCase(object):

    def __init__(self, matrix_pairs, rng=None):
        self.matrix_pairs = matrix_pairs
        self.rng = np.random.RandomState(SEED) if rng is None else rng
        # Ensure a mix of positive and negative scalar multipliers
        self.scalars = np.abs(self.rng.standard_normal(NUM_SCALAR))
        self.scalars[NUM_SCALAR // 2:] = -self.scalars[NUM_SCALAR // 2:]
        self.premultipliers = {
            shape_0: self._generate_premultipliers(shape_0)
            for shape_0 in set(m.shape[0] for _, m in matrix_pairs.values())}
        self.postmultipliers = {
            shape_1: self._generate_postmultipliers(shape_1)
            for shape_1 in set(m.shape[1] for _, m in matrix_pairs.values())}

    def _generate_premultipliers(self, size):
        return (
            [self.rng.standard_normal((size,))] +
            [self.rng.standard_normal((s, size)) for s in [1, size, 2 * size]]
        )

    def _generate_postmultipliers(self, size):
        return (
            [self.rng.standard_normal((size,))] +
            [self.rng.standard_normal((size, s)) for s in [1, size, 2 * size]]
        )

    @iterate_over_matrix_pairs
    def test_shape(matrix, np_matrix):
        assert (
            matrix.shape == (None, None) or matrix.shape == np_matrix.shape)

    @iterate_over_matrix_pairs_postmultipliers
    def test_lmult(matrix, np_matrix, post):
        npt.assert_allclose(matrix @ post, np_matrix @ post)

    @iterate_over_matrix_pairs_premultipliers
    def test_rmult(matrix, np_matrix, pre):
        npt.assert_allclose(pre @ matrix, pre @ np_matrix)

    @iterate_over_matrix_pairs_postmultipliers
    def test_neg_lmult(matrix, np_matrix, post):
        npt.assert_allclose((-matrix) @ post, -np_matrix @ post)

    @iterate_over_matrix_pairs_postmultipliers
    def test_lmult_rmult_trans(matrix, np_matrix, post):
        npt.assert_allclose(matrix @ post, (post.T @ matrix.T).T)

    @iterate_over_matrix_pairs_premultipliers
    def test_rmult_lmult_trans(matrix, np_matrix, pre):
        npt.assert_allclose(pre @ matrix, (matrix.T @ pre.T).T)

    @iterate_over_matrix_pairs_scalars_postmultipliers
    def test_lmult_scalar_lmult(matrix, np_matrix, scalar, post):
        npt.assert_allclose(
            (scalar * matrix) @ post, scalar * np_matrix @ post)

    @iterate_over_matrix_pairs_scalars_postmultipliers
    def test_rdiv_scalar_lmult(matrix, np_matrix, scalar, post):
        npt.assert_allclose(
            (matrix / scalar) @ post, (np_matrix / scalar) @ post)

    @iterate_over_matrix_pairs_scalars_postmultipliers
    def test_rmult_scalar_lmult(matrix, np_matrix, scalar, post):
        npt.assert_allclose(
            (matrix * scalar) @ post, (np_matrix * scalar) @ post)

    @iterate_over_matrix_pairs_scalars_premultipliers
    def test_lmult_scalar_rmult(matrix, np_matrix, scalar, pre):
        npt.assert_allclose(
            pre @ (scalar * matrix), pre @ (scalar * np_matrix))

    @iterate_over_matrix_pairs_scalars_premultipliers
    def test_rmult_scalar_rmult(matrix, np_matrix, scalar, pre):
        npt.assert_allclose(
            pre @ (matrix * scalar), pre @ (np_matrix * scalar))


class ExplicitShapeMatrixTestCase(MatrixTestCase):

    @iterate_over_matrix_pairs
    def test_array(matrix, np_matrix):
        npt.assert_allclose(matrix.array, np_matrix)

    @iterate_over_matrix_pairs
    def test_array_transpose(matrix, np_matrix):
        npt.assert_allclose(matrix.T.array, np_matrix.T)

    @iterate_over_matrix_pairs
    def test_array_transpose_transpose(matrix, np_matrix):
        npt.assert_allclose(matrix.T.T.array, np_matrix)

    @iterate_over_matrix_pairs
    def test_array_numpy(matrix, np_matrix):
        npt.assert_allclose(matrix, np_matrix)

    @iterate_over_matrix_pairs
    def test_diagonal(matrix, np_matrix):
        npt.assert_allclose(matrix.diagonal, np_matrix.diagonal())

    @iterate_over_matrix_pairs_scalars
    def test_lmult_scalar_array(matrix, np_matrix, scalar):
        npt.assert_allclose((scalar * matrix).array, scalar * np_matrix)

    @iterate_over_matrix_pairs_scalars
    def test_rmult_scalar_array(matrix, np_matrix, scalar):
        npt.assert_allclose((matrix * scalar).array, np_matrix * scalar)

    @iterate_over_matrix_pairs_scalars
    def test_rdiv_scalar_array(matrix, np_matrix, scalar):
        npt.assert_allclose((matrix / scalar).array, np_matrix / scalar)

    @iterate_over_matrix_pairs
    def test_neg_array(matrix, np_matrix):
        npt.assert_allclose((-matrix).array, -np_matrix)


class SquareMatrixTestCase(MatrixTestCase):

    def __init__(self, matrix_pairs, rng=None):
        super().__init__(matrix_pairs, rng)
        self.vectors = {
            size: self.rng.standard_normal((NUM_VECTOR, size))
            for size in set(m.shape[0] for _, m in matrix_pairs.values())}

    @iterate_over_matrix_pairs_vectors
    def test_quadratic_form(matrix, np_matrix, vector):
        npt.assert_allclose(
            vector @ matrix @ vector, vector @ np_matrix @ vector)


class ExplicitShapeSquareMatrixTestCase(SquareMatrixTestCase):

    @iterate_over_matrix_pairs
    def test_log_abs_det(matrix, np_matrix):
        npt.assert_allclose(
            matrix.log_abs_det, nla.slogdet(np_matrix)[1], atol=ATOL)


class SymmetricMatrixTestCase(SquareMatrixTestCase):

    @iterate_over_matrix_pairs
    def test_symmetry_identity(matrix, np_matrix):
        assert matrix is matrix.T

    @iterate_over_matrix_pairs_postmultipliers
    def test_symmetry_lmult(matrix, np_matrix, post):
        npt.assert_allclose(matrix @ post, (post.T @ matrix).T)

    @iterate_over_matrix_pairs_premultipliers
    def test_symmetry_rmult(matrix, np_matrix, pre):
        npt.assert_allclose(pre @ matrix, (matrix @ pre.T).T)


class ExplicitShapeSymmetricMatrixTestCase(
        SymmetricMatrixTestCase, ExplicitShapeSquareMatrixTestCase):

    @iterate_over_matrix_pairs
    def test_symmetry_array(matrix, np_matrix):
        npt.assert_allclose(matrix.array, matrix.T.array)

    @iterate_over_matrix_pairs
    def test_eigval(matrix, np_matrix):
        # Ensure eigenvalues in ascending order
        npt.assert_allclose(
            np.sort(matrix.eigval), nla.eigh(np_matrix)[0])

    @iterate_over_matrix_pairs
    def test_eigvec(matrix, np_matrix):
        # Ensure eigenvectors correspond to ascending eigenvalue ordering
        eigval_order = np.argsort(matrix.eigval)
        eigvec = matrix.eigvec.array[:, eigval_order]
        np_eigvec = nla.eigh(np_matrix)[1]
        # Account for eigenvector sign ambiguity when checking for equivalence
        assert np.all(
            np.isclose(eigvec, np_eigvec) | np.isclose(eigvec, -np_eigvec))


class InvertibleMatrixTestCase(MatrixTestCase):

    @iterate_over_matrix_pairs_postmultipliers
    def test_lmult_inv(matrix, np_matrix, post):
        npt.assert_allclose(matrix.inv @ post, nla.solve(np_matrix, post))

    @iterate_over_matrix_pairs_premultipliers
    def test_rmult_inv(matrix, np_matrix, pre):
        npt.assert_allclose(pre @ matrix.inv, nla.solve(np_matrix.T, pre.T).T)

    @iterate_over_matrix_pairs_scalars_postmultipliers
    def test_lmult_scalar_inv_lmult(matrix, np_matrix, scalar, post):
        npt.assert_allclose(
            (scalar * matrix.inv) @ post, nla.solve(np_matrix / scalar, post))

    @iterate_over_matrix_pairs_scalars_postmultipliers
    def test_inv_lmult_scalar_lmult(matrix, np_matrix, scalar, post):
        npt.assert_allclose(
            (scalar * matrix).inv @ post, nla.solve(scalar * np_matrix, post))

    @iterate_over_matrix_pairs_vectors
    def test_quadratic_form_inv(matrix, np_matrix, vector):
        npt.assert_allclose(
            vector @ matrix.inv @ vector,
            vector @ nla.solve(np_matrix, vector))


class ExplicitShapeInvertibleMatrixTestCase(
        ExplicitShapeSquareMatrixTestCase, InvertibleMatrixTestCase):

    @iterate_over_matrix_pairs
    def test_array_inv(matrix, np_matrix):
        npt.assert_allclose(matrix.inv.array, nla.inv(np_matrix), atol=ATOL)

    @iterate_over_matrix_pairs
    def test_array_inv_inv(matrix, np_matrix):
        npt.assert_allclose(matrix.inv.inv.array, np_matrix, atol=ATOL)

    @iterate_over_matrix_pairs
    def test_log_abs_det_inv(matrix, np_matrix):
        npt.assert_allclose(
            matrix.inv.log_abs_det, -nla.slogdet(np_matrix)[1], atol=ATOL)


class PositiveDefiniteMatrixTestCase(
        SymmetricMatrixTestCase, InvertibleMatrixTestCase):

    @iterate_over_matrix_pairs_vectors
    def test_pos_def(matrix, np_matrix, vector):
        assert vector @ matrix @ vector > 0

    @iterate_over_matrix_pairs_postmultipliers
    def test_lmult_sqrt(matrix, np_matrix, post):
        npt.assert_allclose(
            matrix.sqrt @ (matrix.sqrt.T @ post), np_matrix @ post)

    @iterate_over_matrix_pairs_premultipliers
    def test_rmult_sqrt(matrix, np_matrix, pre):
        npt.assert_allclose(
            (pre @ matrix.sqrt) @ matrix.sqrt.T, pre @ np_matrix)

    @iterate_over_matrix_pairs
    def test_inv_is_posdef(matrix, np_matrix):
        assert isinstance(matrix.inv, matrices.PositiveDefiniteMatrix)

    @iterate_over_matrix_pairs
    def test_pos_scalar_multiple_is_posdef(matrix, np_matrix):
        assert isinstance(matrix * 2, matrices.PositiveDefiniteMatrix)


class ExplicitShapePositiveDefiniteMatrixTestCase(
        PositiveDefiniteMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase):

    @iterate_over_matrix_pairs
    def test_sqrt_array(matrix, np_matrix):
        npt.assert_allclose((matrix.sqrt @ matrix.sqrt.T).array, np_matrix)


class DifferentiableMatrixTestCase(MatrixTestCase):

    def __init__(self, matrix_pairs, get_param, param_func, rng=None):
        super().__init__(matrix_pairs, rng)
        self.get_param = get_param
        self.param_func = param_func

    if AUTOGRAD_AVAILABLE:

        def grad_log_abs_det(self, matrix):
            param = self.get_param(matrix)
            return grad(
                lambda p: anp.linalg.slogdet(
                    self.param_func(p, matrix))[1])(param)

        def grad_quadratic_form_inv(self, matrix):
            param = self.get_param(matrix)
            return lambda v: grad(
                lambda p: v @ anp.linalg.solve(
                    self.param_func(p, matrix), v))(param)

        def check_grad_log_abs_det(self, matrix, grad_log_abs_det):
            # Use non-zero atol to allow for floating point errors in gradients
            # analytically equal to zero
            npt.assert_allclose(
                matrix.grad_log_abs_det, grad_log_abs_det, atol=1e-10)

        def test_grad_log_abs_det(self):
            for key, (matrix, np_matrix) in self.matrix_pairs.items():
                yield (self.check_grad_log_abs_det, matrix,
                       self.grad_log_abs_det(matrix))

        def check_grad_quadratic_form_inv(
                self, matrix, vector, grad_quadratic_form_inv):
            # Use non-zero atol to allow for floating point errors in gradients
            # analytically equal to zero
            npt.assert_allclose(
                matrix.grad_quadratic_form_inv(vector),
                grad_quadratic_form_inv(vector), atol=1e-10)

        def test_grad_quadratic_form_inv(self):
            for key, (matrix, np_matrix) in self.matrix_pairs.items():
                for vector in self.vectors[np_matrix.shape[0]]:
                    yield (self.check_grad_quadratic_form_inv, matrix, vector,
                           self.grad_quadratic_form_inv(matrix))


class TestImplicitIdentityMatrix(
        SymmetricMatrixTestCase, InvertibleMatrixTestCase):

    def __init__(self):
        super().__init__({sz: (
            matrices.IdentityMatrix(None), np.identity(sz)) for sz in SIZES})


class TestIdentityMatrix(ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__({sz: (
            matrices.IdentityMatrix(sz), np.identity(sz)) for sz in SIZES})


class TestImplicitScaledIdentityMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            scalar = rng.normal()
            matrix_pairs[sz] = (
                matrices.ScaledIdentityMatrix(scalar, None),
                scalar * np.identity(sz))
        super().__init__(matrix_pairs, rng)


class DifferentiableScaledIdentityMatrixTestCase(DifferentiableMatrixTestCase):

    def __init__(self, generate_scalar, matrix_class):
        rng = np.random.RandomState(SEED)
        matrix_pairs = {}
        for sz in SIZES:
            scalar = generate_scalar(rng)
            matrix_pairs[sz] = (
                matrix_class(scalar, sz), scalar * np.identity(sz))

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                return param * anp.eye(matrix.shape[0])

            def get_param(matrix):
                return matrix._scalar

        else:
            param_func, get_param = None, None

        super().__init__(
            matrix_pairs, get_param, param_func, rng)


class TestScaledIdentityMatrix(
        DifferentiableScaledIdentityMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda rng: rng.normal(), matrices.ScaledIdentityMatrix)


class TestPositiveScaledIdentityMatrix(
        DifferentiableScaledIdentityMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda rng: abs(rng.normal()),
            matrices.PositiveScaledIdentityMatrix)


class DifferentiableDiagonalMatrixTestCase(DifferentiableMatrixTestCase):

    def __init__(self, generate_diagonal, matrix_class):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            diagonal = generate_diagonal(sz, rng)
            matrix_pairs[sz] = (matrix_class(diagonal), np.diag(diagonal))

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                return anp.diag(param)

            def get_param(matrix):
                return matrix.diagonal

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestDiagonalMatrix(
        DifferentiableDiagonalMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda sz, rng: rng.standard_normal(sz),
            matrices.DiagonalMatrix)


class TestPositiveDiagonalMatrix(
        DifferentiableDiagonalMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda sz, rng: abs(rng.standard_normal(sz)),
            matrices.PositiveDiagonalMatrix)


class TestTriangularMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for lower in [True, False]:
                array = rng.standard_normal((sz, sz))
                tri_array = np.tril(array) if lower else np.triu(array)
                matrix_pairs[(sz, lower)] = (
                    matrices.TriangularMatrix(tri_array, lower), tri_array)
        super().__init__(matrix_pairs, rng)


class TestInverseTriangularMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for lower in [True, False]:
                array = rng.standard_normal((sz, sz))
                inv_tri_array = np.tril(array) if lower else np.triu(array)
                matrix_pairs[(sz, lower)] = (
                    matrices.InverseTriangularMatrix(inv_tri_array, lower),
                    nla.inv(inv_tri_array))
        super().__init__(matrix_pairs, rng)


class DifferentiableTriangularFactoredDefiniteMatrixTestCase(
        DifferentiableMatrixTestCase):

    def __init__(self, matrix_class, signs):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for factor_is_lower in [True, False]:
                for sign in signs:
                    array = rng.standard_normal((sz, sz))
                    tri_array = sla.cholesky(array @ array.T, factor_is_lower)
                    matrix_pairs[(sz, factor_is_lower, sign)] = (
                        matrix_class(tri_array, sign, factor_is_lower),
                        sign * tri_array @ tri_array.T)

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                param = (
                    anp.tril(param) if matrix.factor.lower
                    else anp.triu(param))
                return param @ param.T

            def get_param(matrix):
                return matrix.factor.array

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestTriangularFactoredDefiniteMatrix(
        DifferentiableTriangularFactoredDefiniteMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        super().__init__(matrices.TriangularFactoredDefiniteMatrix, (+1, -1))


class TestTriangularFactoredPositiveDefiniteMatrix(
        DifferentiableTriangularFactoredDefiniteMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda factor, sign, factor_is_lower:
                matrices.TriangularFactoredPositiveDefiniteMatrix(
                    factor, factor_is_lower),
            (+1,))


class DifferentiableDenseDefiniteMatrixTestCase(DifferentiableMatrixTestCase):

    def __init__(self, matrix_class, signs):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for sign in signs:
                sqrt_array = rng.standard_normal((sz, sz))
                array = sign * sqrt_array @ sqrt_array.T
                matrix_pairs[(sz, sign)] = (
                    matrix_class(array, is_posdef=(sign == 1)), array)

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                return param

            def get_param(matrix):
                return matrix.array

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestDenseDefiniteMatrix(
        DifferentiableDenseDefiniteMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        super().__init__(matrices.DenseDefiniteMatrix, (+1, -1))


class TestDensePositiveDefiniteMatrix(
        DifferentiableDenseDefiniteMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__(
            lambda array, is_posdef:
                matrices.DensePositiveDefiniteMatrix(array), (+1,))


class TestDensePositiveDefiniteProductMatrix(
        DifferentiableMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for dim_0 in SIZES:
            for dim_1 in [dim_0 + 1, dim_0 * 2]:
                rect_matrix = rng.standard_normal((dim_0, dim_1))
                pos_def_matrix = rng.standard_normal((dim_1, dim_1))
                pos_def_matrix = pos_def_matrix @ pos_def_matrix.T
                array = rect_matrix @ pos_def_matrix @ rect_matrix.T
                matrix_pairs[(dim_0, dim_1)] = (
                    matrices.DensePositiveDefiniteProductMatrix(
                        rect_matrix, pos_def_matrix), array)

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                return param @ matrix._pos_def_matrix @ param.T

            def get_param(matrix):
                return matrix._rect_matrix.array

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestDenseSquareMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            array = rng.standard_normal((sz, sz))
            matrix_pairs[sz] = (
                matrices.DenseSquareMatrix(array), array)
        super().__init__(matrix_pairs, rng)


class TestInverseLUFactoredSquareMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for transposed in [True, False]:
                inverse_array = rng.standard_normal((sz, sz))
                inverse_lu_and_piv = sla.lu_factor(
                    inverse_array.T if transposed else inverse_array)
                array = nla.inv(inverse_array)
                matrix_pairs[(sz, transposed)] = (
                    matrices.InverseLUFactoredSquareMatrix(
                        inverse_array, inverse_lu_and_piv, transposed), array)
            super().__init__(matrix_pairs, rng)


class TestDenseSymmetricMatrix(
        ExplicitShapeInvertibleMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            array = rng.standard_normal((sz, sz))
            array = array + array.T
            matrix_pairs[sz] = (
                matrices.DenseSymmetricMatrix(array), array)
        super().__init__(matrix_pairs, rng)


class TestOrthogonalMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            array = nla.qr(rng.standard_normal((sz, sz)))[0]
            matrix_pairs[sz] = (matrices.OrthogonalMatrix(array), array)
            super().__init__(matrix_pairs, rng)


class TestScaledOrthogonalMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            orth_array = nla.qr(rng.standard_normal((sz, sz)))[0]
            scalar = rng.standard_normal()
            matrix_pairs[sz] = (
                matrices.ScaledOrthogonalMatrix(scalar, orth_array),
                scalar * orth_array)
            super().__init__(matrix_pairs, rng)


class TestEigendecomposedSymmetricMatrix(
        ExplicitShapeInvertibleMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            eigvec = nla.qr(rng.standard_normal((sz, sz)))[0]
            eigval = rng.standard_normal(sz)
            matrix_pairs[sz] = (
                matrices.EigendecomposedSymmetricMatrix(eigvec, eigval),
                (eigvec * eigval) @ eigvec.T)
        super().__init__(matrix_pairs, rng)


class TestEigendecomposedPositiveDefiniteMatrix(
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            eigvec = nla.qr(rng.standard_normal((sz, sz)))[0]
            eigval = np.abs(rng.standard_normal(sz))
            matrix_pairs[sz] = (
                matrices.EigendecomposedPositiveDefiniteMatrix(eigvec, eigval),
                (eigvec * eigval) @ eigvec.T)
        super().__init__(matrix_pairs, rng)


class TestSoftAbsRegularisedPositiveDefiniteMatrix(
        DifferentiableMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs, grad_log_abs_dets, grad_quadratic_form_invs = {}, {}, {}
        rng = np.random.RandomState(SEED)
        for sz in SIZES:
            for softabs_coeff in [0.5, 1., 1.5]:
                sym_array = rng.standard_normal((sz, sz))
                sym_array = sym_array + sym_array.T
                unreg_eigval, eigvec = np.linalg.eigh(sym_array)
                eigval = unreg_eigval / np.tanh(unreg_eigval * softabs_coeff)
                matrix_pairs[(sz, softabs_coeff)] = (
                    matrices.SoftAbsRegularisedPositiveDefiniteMatrix(
                        sym_array, softabs_coeff
                    ), (eigvec * eigval) @ eigvec.T)

        if AUTOGRAD_AVAILABLE:

            def get_param(matrix):
                eigvec = matrix.eigvec.array
                return (eigvec * matrix.unreg_eigval) @ eigvec.T

            def param_func(param, matrix):
                softabs_coeff = matrix._softabs_coeff
                sym_array = (param + param.T) / 2
                unreg_eigval, eigvec = anp.linalg.eigh(sym_array)
                eigval = unreg_eigval / anp.tanh(unreg_eigval * softabs_coeff)
                return (eigvec * eigval) @ eigvec.T

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestMatrixProduct(ExplicitShapeMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_terms in [2, 4]:
                for explicit in [True, False]:
                    arrays = [
                        rng.standard_normal((s if t % 2 == 0 else 2 * s,
                                             2 * s if t % 2 == 0 else s))
                        for t in range(n_terms)]
                    matrices_ = [
                        matrices.DenseRectangularMatrix(a) for a in arrays]
                    if explicit:
                        matrix = matrices.MatrixProduct(matrices_)
                    else:
                        matrix = reduce(lambda a, b: a @ b, matrices_)
                    matrix_pairs[(s, n_terms, explicit)] = (
                        matrix, nla.multi_dot(arrays))
        super().__init__(matrix_pairs, rng)


class TestSquareMatrixProduct(ExplicitShapeSquareMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_terms in [2, 5]:
                arrays = [
                    rng.standard_normal((s, s)) for _ in range(n_terms)]
                matrix = matrices.SquareMatrixProduct([
                    matrices.DenseSquareMatrix(a) for a in arrays])
                matrix_pairs[(s, n_terms)] = (matrix, nla.multi_dot(arrays))
        super().__init__(matrix_pairs, rng)


class TestInvertibleMatrixProduct(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_terms in [2, 5]:
                for explicit in [True, False]:
                    arrays = [
                        rng.standard_normal((s, s)) for _ in range(n_terms)]
                    matrices_ = [
                        matrices.DenseSquareMatrix(a) for a in arrays]
                    if explicit:
                        matrix = matrices.InvertibleMatrixProduct(matrices_)
                    else:
                        matrix = reduce(lambda a, b: a @ b, matrices_)
                    matrix_pairs[(s, n_terms, explicit)] = (
                        matrix, nla.multi_dot(arrays))
        super().__init__(matrix_pairs, rng)


class TestSquareBlockDiagonalMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_block in [1, 2, 5]:
                arrays = [rng.standard_normal((s, s)) for _ in range(n_block)]
                matrix_pairs[(s, n_block)] = (
                    matrices.SquareBlockDiagonalMatrix(
                        matrices.DenseSquareMatrix(arr) for arr in arrays),
                    sla.block_diag(*arrays))
        super().__init__(matrix_pairs, rng)


class TestSymmetricBlockDiagonalMatrix(
        ExplicitShapeInvertibleMatrixTestCase,
        ExplicitShapeSymmetricMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_block in [1, 2, 5]:
                arrays = [rng.standard_normal((s, s)) for _ in range(n_block)]
                arrays = [arr + arr.T for arr in arrays]
                matrix_pairs[(s, n_block)] = (
                    matrices.SymmetricBlockDiagonalMatrix(
                        matrices.DenseSymmetricMatrix(arr) for arr in arrays),
                    sla.block_diag(*arrays))
        super().__init__(matrix_pairs, rng)


class TestPositiveDefiniteBlockDiagonalMatrix(
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_block in [1, 2, 5]:
                arrays = [rng.standard_normal((s, s)) for _ in range(n_block)]
                arrays = [arr @ arr.T for arr in arrays]
                matrix_pairs[(s, n_block)] = (
                    matrices.PositiveDefiniteBlockDiagonalMatrix(
                        matrices.DensePositiveDefiniteMatrix(arr)
                        for arr in arrays),
                    sla.block_diag(*arrays))
        super().__init__(matrix_pairs, rng)


class TestPositiveDefiniteBlockDiagonalMatrix(
        DifferentiableMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_block in [1, 2, 5]:
                arrays = [rng.standard_normal((s, s)) for _ in range(n_block)]
                arrays = [arr @ arr.T for arr in arrays]
                matrix_pairs[(s, n_block)] = (
                    matrices.PositiveDefiniteBlockDiagonalMatrix(
                        matrices.DensePositiveDefiniteMatrix(arr)
                        for arr in arrays),
                    sla.block_diag(*arrays))

        if AUTOGRAD_AVAILABLE:

            @primitive
            def block_diag(blocks):
                return sla.block_diag(*blocks)

            def vjp_block_diag(ans, blocks):

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

            def get_param(matrix):
                return tuple(
                    block.array for block in matrix._blocks)

            def param_func(param, matrix):
                return block_diag(param)

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)


class TestDenseRectangularMatrix(ExplicitShapeMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s0 in SIZES:
            for s1 in SIZES:
                if s0 != s1:
                    array = rng.standard_normal((s0, s1))
                    matrix_pairs[(s0, s1)] = (
                        matrices.DenseRectangularMatrix(array), array)
        super().__init__(matrix_pairs, rng)


class TestBlockRowMatrix(ExplicitShapeMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_blocks in [2, 5]:
                blocks = [rng.standard_normal((s, s)) for _ in range(n_blocks)]
                matrix_pairs[(s, n_blocks)] = (
                    matrices.BlockRowMatrix(
                        matrices.DenseSquareMatrix(block) for block in blocks),
                    np.hstack(blocks))
        super().__init__(matrix_pairs, rng)


class TestBlockColumnMatrix(ExplicitShapeMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for s in SIZES:
            for n_blocks in [2, 5]:
                blocks = [rng.standard_normal((s, s)) for _ in range(n_blocks)]
                matrix_pairs[(s, n_blocks)] = (
                    matrices.BlockColumnMatrix(
                        matrices.DenseSquareMatrix(block) for block in blocks),
                    np.vstack(blocks))
        super().__init__(matrix_pairs, rng)


class TestSquareLowRankUpdateMatrix(ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for outer_dim in SIZES:
            for inner_dim in [max(1, outer_dim // 2), max(1, outer_dim - 1)]:
                left_factor_matrix = rng.standard_normal(
                    (outer_dim, inner_dim))
                right_factor_matrix = rng.standard_normal(
                    (inner_dim, outer_dim))
                inner_square_matrix = rng.standard_normal(
                    (inner_dim, inner_dim))
                square_matrix = rng.standard_normal((outer_dim, outer_dim))
                matrix_pairs[(inner_dim, outer_dim)] = (
                    matrices.SquareLowRankUpdateMatrix(
                        matrices.DenseRectangularMatrix(left_factor_matrix),
                        matrices.DenseRectangularMatrix(right_factor_matrix),
                        matrices.DenseSquareMatrix(square_matrix),
                        matrices.DenseSquareMatrix(inner_square_matrix)),
                    square_matrix + left_factor_matrix @ (
                        inner_square_matrix @ right_factor_matrix))
        super().__init__(matrix_pairs, rng)


class TestNoInnerMatrixSquareLowRankUpdateMatrix(
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for outer_dim in SIZES:
            inner_dim = max(1, outer_dim // 2)
            left_factor_matrix = rng.standard_normal(
                (outer_dim, inner_dim))
            right_factor_matrix = rng.standard_normal(
                (inner_dim, outer_dim))
            square_matrix = rng.standard_normal((outer_dim, outer_dim))
            matrix_pairs[(inner_dim, outer_dim)] = (
                matrices.SquareLowRankUpdateMatrix(
                    matrices.DenseRectangularMatrix(left_factor_matrix),
                    matrices.DenseRectangularMatrix(right_factor_matrix),
                    matrices.DenseSquareMatrix(square_matrix)),
                square_matrix + left_factor_matrix @ right_factor_matrix)
        super().__init__(matrix_pairs, rng)


class TestSymmetricLowRankUpdateMatrix(
        ExplicitShapeSymmetricMatrixTestCase,
        ExplicitShapeInvertibleMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for outer_dim in SIZES:
            for inner_dim in [max(1, outer_dim // 2), max(1, outer_dim - 1)]:
                factor_matrix = rng.standard_normal(
                    (outer_dim, inner_dim))
                inner_symmetric_matrix = rng.standard_normal(
                    (inner_dim, inner_dim))
                inner_symmetric_matrix = (
                    inner_symmetric_matrix + inner_symmetric_matrix.T)
                symmetric_matrix = rng.standard_normal((outer_dim, outer_dim))
                symmetric_matrix = symmetric_matrix + symmetric_matrix.T
                matrix_pairs[(inner_dim, outer_dim)] = (
                    matrices.SymmetricLowRankUpdateMatrix(
                        matrices.DenseRectangularMatrix(factor_matrix),
                        matrices.DenseSymmetricMatrix(symmetric_matrix),
                        matrices.DenseSymmetricMatrix(inner_symmetric_matrix)),
                    symmetric_matrix + factor_matrix @ (
                        inner_symmetric_matrix @ factor_matrix.T))
        super().__init__(matrix_pairs, rng)


class TestPositiveDefiniteLowRankUpdateMatrix(
        DifferentiableMatrixTestCase,
        ExplicitShapePositiveDefiniteMatrixTestCase):

    def __init__(self):
        matrix_pairs = {}
        rng = np.random.RandomState(SEED)
        for outer_dim in SIZES:
            for inner_dim in [max(1, outer_dim // 2), max(1, outer_dim - 1)]:
                factor_matrix = rng.standard_normal(
                    (outer_dim, inner_dim))
                inner_pos_def_matrix = rng.standard_normal(
                    (inner_dim, inner_dim))
                inner_pos_def_matrix = (
                    inner_pos_def_matrix @ inner_pos_def_matrix.T)
                pos_def_matrix = rng.standard_normal((outer_dim, outer_dim))
                pos_def_matrix = pos_def_matrix @ pos_def_matrix.T
                matrix_pairs[(inner_dim, outer_dim)] = (
                    matrices.PositiveDefiniteLowRankUpdateMatrix(
                        matrices.DenseRectangularMatrix(factor_matrix),
                        matrices.DensePositiveDefiniteMatrix(pos_def_matrix),
                        matrices.DensePositiveDefiniteMatrix(
                            inner_pos_def_matrix)),
                    pos_def_matrix + factor_matrix @ (
                        inner_pos_def_matrix @ factor_matrix.T))

        if AUTOGRAD_AVAILABLE:

            def param_func(param, matrix):
                return (
                    matrix.pos_def_matrix.array +
                    param @ matrix.inner_pos_def_matrix @ param.T)

            def get_param(matrix):
                return matrix.factor_matrix.array

        else:
            param_func, get_param = None, None

        super().__init__(matrix_pairs, get_param, param_func, rng)
