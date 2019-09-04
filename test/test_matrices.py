import hmc.matrices as matrices
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import numpy.testing as npt
from functools import partial

AUTOGRAD_AVAILABLE = True
try:
    import autograd.numpy as anp
    from autograd import grad
except ImportError:
    AUTOGRAD_AVAILABLE = False
    import warnings
    warnings.warn(
        'Autograd not available. Skipping gradient tests.')

SEED = 3046987125
NUM_SCALAR = 4
SIZES = {1, 2, 5, 10}
ATOL = 1e-10


def generate_premultipliers(rng, size):
    return (
        [rng.standard_normal((size,))] +
        [rng.standard_normal((sz, size)) for sz in [1, 2, size, 2 * size]]
    )


def generate_postmultipliers(rng, size):
    return (
        [rng.standard_normal((size,))] +
        [rng.standard_normal((size, sz)) for sz in [1, 2, size, 2 * size]]
    )


class MatrixTestCase(object):

    def __init__(self):
        self.rng = np.random.RandomState(SEED)
        # Ensure a mix of positive and negative scalar multipliers
        self.scalars = np.abs(self.rng.standard_normal(NUM_SCALAR))
        self.scalars[NUM_SCALAR // 2:] = -self.scalars[NUM_SCALAR // 2:]
        self.premultipliers = {
            sz: generate_premultipliers(self.rng, sz) for sz in SIZES}
        self.postmultipliers = {
            sz: generate_postmultipliers(self.rng, sz) for sz in SIZES}
        self.matrices = {}
        self.np_matrices = {}

    def check_shape(self, size):
        assert (
            self.matrices[size].shape == (None, None) or
            self.matrices[size].shape == self.np_matrices[size].shape)

    def test_shape(self):
        for size in SIZES:
            yield self.check_shape, size

    def check_lmult(self, size, other):
        npt.assert_allclose(
            self.matrices[size] @ other, self.np_matrices[size] @ other)

    def test_lmult(self):
        for size in SIZES:
            for post in self.postmultipliers[size]:
                yield self.check_lmult, size, post

    def check_rmult(self, size, other):
        npt.assert_allclose(
            other @ self.matrices[size], other @ self.np_matrices[size])

    def test_rmult(self):
        for size in SIZES:
            for pre in self.premultipliers[size]:
                yield self.check_rmult, size, pre

    def check_lmult_scalar_lmult(self, size, scalar, other):
        npt.assert_allclose(
            (scalar * self.matrices[size]) @ other,
            scalar * self.np_matrices[size] @ other)

    def test_lmult_scalar_lmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for post in self.postmultipliers[size]:
                    yield self.check_lmult_scalar_lmult, size, scalar, post

    def check_rmult_scalar_lmult(self, size, scalar, other):
        npt.assert_allclose(
            (self.matrices[size] * scalar) @ other,
            scalar * self.np_matrices[size] @ other)

    def test_rmult_scalar_lmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for post in self.postmultipliers[size]:
                    yield self.check_rmult_scalar_lmult, size, scalar, post

    def check_lmult_scalar_rmult(self, size, scalar, other):
        npt.assert_allclose(
            other @ (scalar * self.matrices[size]),
            other @ (scalar * self.np_matrices[size]))

    def test_lmult_scalar_rmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for pre in self.premultipliers[size]:
                    yield self.check_lmult_scalar_rmult, size, scalar, pre

    def check_rmult_scalar_rmult(self, size, scalar, other):
        npt.assert_allclose(
            other @ (self.matrices[size] * scalar),
            other @ (self.np_matrices[size] * scalar))

    def test_rmult_scalar_rmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for pre in self.premultipliers[size]:
                    yield self.check_rmult_scalar_rmult, size, scalar, pre


class ExplicitMatrixTestCase(MatrixTestCase):

    def check_array(self, size):
        npt.assert_allclose(self.matrices[size].array, self.np_matrices[size])

    def test_array(self):
        for size in SIZES:
            yield self.check_array, size

    def check_array_transpose(self, size):
        npt.assert_allclose(
            self.matrices[size].T.array, self.np_matrices[size].T)

    def test_array_transpose(self):
        for size in SIZES:
            yield self.check_array_transpose, size

    def check_array_transpose_transpose(self, size):
        npt.assert_allclose(
            self.matrices[size].T.T.array, self.np_matrices[size])

    def test_array_transpose_transpose(self):
        for size in SIZES:
            yield self.check_array_transpose_transpose, size

    def check_array_numpy(self, size):
        npt.assert_allclose(self.matrices[size], self.np_matrices[size])

    def test_array_numpy(self):
        for size in SIZES:
            yield self.check_array_numpy, size

    def check_diagonal(self, size):
        npt.assert_allclose(
            self.matrices[size].diagonal, self.np_matrices[size].diagonal())

    def test_diagonal(self):
        for size in SIZES:
            yield self.check_diagonal, size

    def check_log_abs_det(self, size):
        npt.assert_allclose(self.matrices[size].log_abs_det,
                            nla.slogdet(self.np_matrices[size])[1], atol=ATOL)

    def test_log_abs_det(self):
        for size in SIZES:
            yield self.check_log_abs_det, size

    def check_log_abs_det_sqrt(self, size):
        npt.assert_allclose(self.matrices[size].log_abs_det_sqrt,
                            nla.slogdet(self.np_matrices[size])[1] / 2,
                            atol=ATOL)

    def test_log_abs_det_sqrt(self):
        for size in SIZES:
            yield self.check_log_abs_det_sqrt, size

    def check_lmult_scalar_array(self, size, scalar):
        npt.assert_allclose(
            (scalar * self.matrices[size]).array,
            scalar * self.np_matrices[size])

    def test_lmult_scalar_array(self):
        for size in SIZES:
            for scalar in self.scalars:
                yield self.check_lmult_scalar_array, size, scalar

    def check_rmult_scalar_array(self, size, scalar):
        npt.assert_allclose(
            (self.matrices[size] * scalar).array,
            scalar * self.np_matrices[size])

    def test_rmult_scalar_array(self):
        for size in SIZES:
            for scalar in self.scalars:
                yield self.check_rmult_scalar_array, size, scalar

    def check_rdiv_scalar_array(self, size, scalar):
        npt.assert_allclose(
            (self.matrices[size] / scalar).array,
            self.np_matrices[size] / scalar)

    def test_rdiv_scalar_array(self):
        for size in SIZES:
            for scalar in self.scalars:
                yield self.check_rdiv_scalar_array, size, scalar


class SymmetricMatrixTestCase(MatrixTestCase):

    def check_symmetry_identity(self, size):
        assert self.matrices[size] is self.matrices[size].T

    def test_symmetry_identity(self):
        for size in SIZES:
            yield self.check_symmetry_identity, size

    def check_symmetry_array(self, size):
        npt.assert_allclose(
            self.matrices[size].array, self.matrices[size].T.array)

    def test_symmetry_array(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_symmetry_array, size

    def check_symmetry_lmult(self, other, size):
        npt.assert_allclose(
            self.matrices[size] @ other, (other.T @ self.matrices[size]).T)

    def test_symmetry_lmult(self):
        for size in SIZES:
            for post in self.postmultipliers[size]:
                yield self.check_symmetry_lmult, post, size

    def check_symmetry_rmult(self, other, size):
        npt.assert_allclose(
            other @ self.matrices[size], (self.matrices[size] @ other.T).T)

    def test_symmetry_rmult(self):
        for size in SIZES:
            for pre in self.premultipliers[size]:
                yield self.check_symmetry_rmult, pre, size

    def check_eigval(self, size):
        # Ensure eigenvalues in ascending order
        npt.assert_allclose(
            np.sort(self.matrices[size].eigval),
            nla.eigh(self.np_matrices[size])[0])

    def test_eigval(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_eigval, size

    def check_eigvec(self, size):
        # Ensure eigenvectors correspond to ascending eigenvalue ordering
        eigval_order = np.argsort(self.matrices[size].eigval)
        eigvec = self.matrices[size].eigvec.array[:, eigval_order]
        np_eigvec = nla.eigh(self.np_matrices[size])[1]
        # Account for eigenvector sign ambiguity when checking for equivalence
        assert np.all(
            np.isclose(eigvec, np_eigvec) | np.isclose(eigvec, -np_eigvec))

    def test_eigvec(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_eigvec, size


class InvertibleMatrixTestCase(MatrixTestCase):

    def check_lmult_inv(self, size, other):
        npt.assert_allclose(
            self.matrices[size].inv @ other,
            nla.solve(self.np_matrices[size], other))

    def test_lmult_inv(self):
        for size in SIZES:
            for post in self.postmultipliers[size]:
                yield self.check_lmult_inv, size, post

    def check_rmult_inv(self, size, other):
        npt.assert_allclose(
            other @ self.matrices[size].inv,
            nla.solve(self.np_matrices[size].T, other.T).T)

    def test_rmult_inv(self):
        for size in SIZES:
            for pre in self.premultipliers[size]:
                yield self.check_rmult_inv, size, pre

    def check_array_inv(self, size):
        npt.assert_allclose(
            self.matrices[size].inv.array,
            nla.inv(self.np_matrices[size]), atol=ATOL)

    def test_array_inv(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_array_inv, size

    def check_array_inv_inv(self, size):
        npt.assert_allclose(
            self.matrices[size].inv.inv.array,
            self.np_matrices[size], atol=ATOL)

    def test_array_inv_inv(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_array_inv_inv, size

    def check_log_abs_det_inv(self, size):
        npt.assert_allclose(self.matrices[size].inv.log_abs_det,
                            -nla.slogdet(self.np_matrices[size])[1], atol=ATOL)

    def test_log_abs_det_inv(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_log_abs_det_inv, size

    def check_log_abs_det_sqrt_inv(self, size):
        npt.assert_allclose(self.matrices[size].inv.log_abs_det_sqrt,
                            -nla.slogdet(self.np_matrices[size])[1] / 2,
                            atol=ATOL)

    def test_log_abs_det_sqrt_inv(self):
        if isinstance(self, ExplicitMatrixTestCase):
            for size in SIZES:
                yield self.check_log_abs_det_sqrt_inv, size

    def check_lmult_scalar_inv_lmult(self, size, scalar, other):
        npt.assert_allclose(
            (scalar * self.matrices[size].inv) @ other,
            nla.solve(self.np_matrices[size] / scalar, other))

    def test_lmult_scalar_inv_lmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for post in self.postmultipliers[size]:
                    yield self.check_lmult_scalar_inv_lmult, size, scalar, post

    def check_inv_lmult_scalar_lmult(self, size, scalar, other):
        npt.assert_allclose(
            (scalar * self.matrices[size]).inv @ other,
            nla.solve(scalar * self.np_matrices[size], other))

    def test_inv_lmult_scalar_lmult(self):
        for size in SIZES:
            for scalar in self.scalars:
                for post in self.postmultipliers[size]:
                    yield self.check_inv_lmult_scalar_lmult, size, scalar, post


class PositiveDefiniteMatrixTestCase(
        SymmetricMatrixTestCase, InvertibleMatrixTestCase):

    def check_lmult_sqrt(self, size, other):
        npt.assert_allclose(
            self.matrices[size].sqrt @ (self.matrices[size].sqrt.T @ other),
            self.np_matrices[size] @ other)

    def test_lmult_sqrt(self):
        for size in SIZES:
            for post in self.postmultipliers[size]:
                yield self.check_lmult_sqrt, size, post

    def check_rmult_sqrt(self, size, other):
        npt.assert_allclose(
            (other @ self.matrices[size].sqrt) @ self.matrices[size].sqrt.T,
            other @ self.np_matrices[size])

    def test_rmult_sqrt(self):
        for size in SIZES:
            for pre in self.premultipliers[size]:
                yield self.check_rmult_sqrt, size, pre


class DifferentiableMatrixTestCase(MatrixTestCase):

    def __init__(self):
        super().__init__()
        self.grad_log_abs_det_sqrts = {}
        self.grad_quadratic_form_invs = {}
        self.vectors = {
            sz: self.rng.standard_normal((NUM_SCALAR, sz)) for sz in SIZES}

    if AUTOGRAD_AVAILABLE:

        def check_grad_log_abs_det_sqrt(self, size):
            npt.assert_allclose(
                self.matrices[size].grad_log_abs_det_sqrt,
                self.grad_log_abs_det_sqrts[size])

        def test_grad_log_abs_det_sqrt(self):
            for size in SIZES:
                yield self.check_grad_log_abs_det_sqrt, size

        def check_grad_quadratic_form_inv(self, size, vector):
            npt.assert_allclose(
                self.matrices[size].grad_quadratic_form_inv(vector),
                self.grad_quadratic_form_invs[size](vector))

        def test_grad_quadratic_form_inv(self):
            for size in SIZES:
                for vector in self.vectors[size]:
                    yield self.check_grad_quadratic_form_inv, size, vector


class TestImplicitIdentityMatrix(PositiveDefiniteMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            self.matrices[sz] = matrices.IdentityMatrix(None)
            self.np_matrices[sz] = np.identity(sz)


class TestIdentityMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            self.matrices[sz] = matrices.IdentityMatrix(sz)
            self.np_matrices[sz] = np.identity(sz)


class TestPositiveScaledIdentityMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase,
        DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            scalar = abs(self.rng.normal())
            self.matrices[sz] = matrices.PositiveScaledIdentityMatrix(
                scalar, sz)
            self.np_matrices[sz] = scalar * np.identity(sz)
            if AUTOGRAD_AVAILABLE:
                self.grad_log_abs_det_sqrts[sz] = grad(
                    lambda s: 0.5 * anp.linalg.slogdet(s * anp.eye(sz))[1])(
                        scalar)
                self.grad_quadratic_form_invs[sz] = partial(
                    grad(lambda s, v: (v / s) @ v), scalar)


class TestScaledIdentityMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase,
        ExplicitMatrixTestCase, DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            scalar = self.rng.normal()
            self.matrices[sz] = matrices.ScaledIdentityMatrix(scalar, sz)
            self.np_matrices[sz] = scalar * np.identity(sz)
            if AUTOGRAD_AVAILABLE:
                self.grad_log_abs_det_sqrts[sz] = grad(
                    lambda s: 0.5 * anp.linalg.slogdet(s * anp.eye(sz))[1])(
                        scalar)
                self.grad_quadratic_form_invs[sz] = partial(
                    grad(lambda s, v: (v / s) @ v), scalar)


class TestImplicitScaledIdentityMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            scalar = self.rng.normal()
            self.matrices[sz] = matrices.ScaledIdentityMatrix(scalar, None)
            self.np_matrices[sz] = scalar * np.identity(sz)


class TestPositiveDiagonalMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase,
        DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        if AUTOGRAD_AVAILABLE:
            grad_log_abs_det_sqrt_func = grad(
                lambda d: 0.5 * anp.linalg.slogdet(anp.diag(d))[1])
            grad_quadratic_form_inv_func = grad(
                lambda d, v: v @ anp.diag(1 / d) @ v)
        for sz in SIZES:
            diagonal = np.abs(self.rng.standard_normal(sz))
            self.matrices[sz] = matrices.PositiveDiagonalMatrix(diagonal)
            self.np_matrices[sz] = np.diag(diagonal)
            if AUTOGRAD_AVAILABLE:
                self.grad_log_abs_det_sqrts[sz] = (
                    grad_log_abs_det_sqrt_func(diagonal))
                self.grad_quadratic_form_invs[sz] = partial(
                    grad_quadratic_form_inv_func, diagonal)


class TestDiagonalMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase,
        ExplicitMatrixTestCase, DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        if AUTOGRAD_AVAILABLE:
            grad_log_abs_det_sqrt_func = grad(
                lambda d: 0.5 * anp.linalg.slogdet(anp.diag(d))[1])
            grad_quadratic_form_inv_func = grad(
                lambda d, v: v @ anp.diag(1 / d) @ v)
        for sz in SIZES:
            diagonal = self.rng.standard_normal(sz)
            self.matrices[sz] = matrices.DiagonalMatrix(diagonal)
            self.np_matrices[sz] = np.diag(diagonal)
            if AUTOGRAD_AVAILABLE:
                self.grad_log_abs_det_sqrts[sz] = (
                    grad_log_abs_det_sqrt_func(diagonal))
                self.grad_quadratic_form_invs[sz] = partial(
                    grad_quadratic_form_inv_func, diagonal)


class TestTriangularMatrix(
        InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            for lower in [True, False]:
                array = self.rng.standard_normal((sz, sz))
                tri_array = np.tril(array) if lower else np.triu(array)
                self.matrices[sz] = matrices.TriangularMatrix(tri_array, lower)
                self.np_matrices[sz] = tri_array


class TestInverseTriangularMatrix(
        InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            for lower in [True, False]:
                array = self.rng.standard_normal((sz, sz))
                inv_tri_array = np.tril(array) if lower else np.triu(array)
                self.matrices[sz] = matrices.InverseTriangularMatrix(
                    inv_tri_array, lower)
                self.np_matrices[sz] = nla.inv(inv_tri_array)


class TestTriangularFactoredDefiniteMatrix(
        InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            for lower in [True, False]:
                for sign in [+1, -1]:
                    array = self.rng.standard_normal((sz, sz))
                    tri_array = np.tril(array) if lower else np.triu(array)
                    self.matrices[sz] = (
                        matrices.TriangularFactoredDefiniteMatrix(
                            tri_array, lower, sign))
                    self.np_matrices[sz] = sign * (tri_array @ tri_array.T)


class TestDenseDefiniteMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase,
        ExplicitMatrixTestCase, DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        if AUTOGRAD_AVAILABLE:
            grad_log_abs_det_sqrt_func = grad(
                lambda a: 0.5 * anp.linalg.slogdet(a)[1])
            grad_quadratic_form_inv_func = grad(
                lambda a, v: v @ anp.linalg.solve(a, v))
        for sz in SIZES:
            for lower in [True, False]:
                for sign in [+1, -1]:
                    sqrt = self.rng.standard_normal((sz, sz))
                    array = sign * sqrt @ sqrt.T
                    self.matrices[sz] = matrices.DenseDefiniteMatrix(
                        array, sign)
                    self.np_matrices[sz] = array
                    if AUTOGRAD_AVAILABLE:
                        self.grad_log_abs_det_sqrts[sz] = (
                            grad_log_abs_det_sqrt_func(array))
                        self.grad_quadratic_form_invs[sz] = partial(
                            grad_quadratic_form_inv_func, array)


class TestDensePositiveDefiniteMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase,
        DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        if AUTOGRAD_AVAILABLE:
            grad_log_abs_det_sqrt_func = grad(
                lambda a: 0.5 * anp.linalg.slogdet(a)[1])
            grad_quadratic_form_inv_func = grad(
                lambda a, v: v @ anp.linalg.solve(a, v))
        for sz in SIZES:
            sqrt = self.rng.standard_normal((sz, sz))
            array = sqrt @ sqrt.T
            self.matrices[sz] = matrices.DensePositiveDefiniteMatrix(array)
            self.np_matrices[sz] = array
            if AUTOGRAD_AVAILABLE:
                self.grad_log_abs_det_sqrts[sz] = (
                    grad_log_abs_det_sqrt_func(array))
                self.grad_quadratic_form_invs[sz] = partial(
                    grad_quadratic_form_inv_func, array)


class TestDenseNegativeDefiniteMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase,
        ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            sqrt = self.rng.standard_normal((sz, sz))
            array = -sqrt @ sqrt.T
            self.matrices[sz] = matrices.DenseNegativeDefiniteMatrix(array)
            self.np_matrices[sz] = array


class TestDenseMatrix(InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            for transposed in [True, False]:
                array = self.rng.standard_normal((sz, sz))
                self.matrices[sz] = matrices.DenseSquareMatrix(
                    array, transposed)
                self.np_matrices[sz] = array.T if transposed else array


class TestInverseLUFactoredSquareMatrix(
        InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            for transposed in [True, False]:
                inverse_array = self.rng.standard_normal((sz, sz))
                inverse_lu_and_piv = sla.lu_factor(inverse_array)
                array = nla.inv(inverse_array)
                self.matrices[sz] = matrices.InverseLUFactoredSquareMatrix(
                    inverse_array, transposed, inverse_lu_and_piv)
                self.np_matrices[sz] = array.T if transposed else array


class TestOrthogonalMatrix(InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            array = nla.qr(self.rng.standard_normal((sz, sz)))[0]
            self.matrices[sz] = matrices.OrthogonalMatrix(array)
            self.np_matrices[sz] = array


class TestScaledOrthogonalMatrix(
        InvertibleMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            orth_array = nla.qr(self.rng.standard_normal((sz, sz)))[0]
            scalar = self.rng.standard_normal()
            self.matrices[sz] = matrices.ScaledOrthogonalMatrix(
                scalar, orth_array)
            self.np_matrices[sz] = scalar * orth_array


class TestEigendecomposedSymmetricMatrix(
        InvertibleMatrixTestCase, SymmetricMatrixTestCase,
        ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            eigvec = nla.qr(self.rng.standard_normal((sz, sz)))[0]
            eigval = self.rng.standard_normal(sz)
            self.matrices[sz] = matrices.EigendecomposedSymmetricMatrix(
                eigvec, eigval)
            self.np_matrices[sz] = (eigvec * eigval) @ eigvec.T


class TestEigendecomposedPositiveDefiniteMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase):

    def __init__(self):
        super().__init__()
        for sz in SIZES:
            eigvec = nla.qr(self.rng.standard_normal((sz, sz)))[0]
            eigval = np.abs(self.rng.standard_normal(sz))
            self.matrices[sz] = matrices.EigendecomposedSymmetricMatrix(
                eigvec, eigval)
            self.np_matrices[sz] = (eigvec * eigval) @ eigvec.T


class TestSoftAbsRegularisedPositiveDefiniteMatrix(
        PositiveDefiniteMatrixTestCase, ExplicitMatrixTestCase,
        DifferentiableMatrixTestCase):

    def __init__(self):
        super().__init__()
        if AUTOGRAD_AVAILABLE:
            def softabs_reg(sym_array, softabs_coeff):
                sym_array = (sym_array + sym_array.T) / 2
                unreg_eigval, eigvec = anp.linalg.eigh(sym_array)
                eigval = unreg_eigval / anp.tanh(unreg_eigval * softabs_coeff)
                return (eigvec * eigval) @ eigvec.T
            grad_log_abs_det_sqrt_func = grad(
                lambda a, s: 0.5 * anp.linalg.slogdet(softabs_reg(a, s))[1])
            grad_quadratic_form_inv_func = grad(
                lambda a, s, v: v @ anp.linalg.solve(softabs_reg(a, s), v))
        for sz in SIZES:
            for softabs_coeff in [0.5, 1., 1.5]:
                sym_array = self.rng.standard_normal((sz, sz))
                sym_array += sym_array.T
                unreg_eigval, eigvec = np.linalg.eigh(sym_array)
                eigval = unreg_eigval / np.tanh(unreg_eigval * softabs_coeff)
                self.matrices[sz] = (
                    matrices.SoftAbsRegularisedPositiveDefiniteMatrix(
                        sym_array, softabs_coeff
                    ))
                self.np_matrices[sz] = (eigvec * eigval) @ eigvec.T
                if AUTOGRAD_AVAILABLE:
                    self.grad_log_abs_det_sqrts[sz] = (
                        grad_log_abs_det_sqrt_func(sym_array, softabs_coeff))
                    self.grad_quadratic_form_invs[sz] = partial(
                        grad_quadratic_form_inv_func, sym_array, softabs_coeff)
