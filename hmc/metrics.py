"""Euclidean and Riemannian metric classes.

For use with Hamiltonian systems with a Gaussian conditional distribution on
the momentum variables corresponding to a quadratic form for the kinetic energy
function.

Using a non-identity metric is equivalent to using an identity metric with a
reparameterisation of the original target distribution on the position
variables. This effective reparameterisation can be used to for example
rescale and decorrelate the distribution on the position variables to improve
the chain mixing performance.
"""

import numpy as np
import scipy.linalg as sla
from hmc.states import cache_in_state, multi_cache_in_state
from hmc.autodiff import autodiff_fallback
from hmc.utils import inherit_docstrings


class _BaseEuclideanMetric(object):
    """Abstract base class for Euclidean metric classes."""

    def quadratic_form_inv(self, vector):
        """Evaluate the quadratic form associated with the inverse metric.

        Args:
            vector (array): Vector to evaluate quadratic form for.

        Returns:
            float: Value of `vector @ inv(metric) @ vector`
        """
        return np.sum(vector * self.lmult_inv(vector))

    def lmult(self, other):
        """Evaluate left-multiplication of argument by metric.

        Args:
            other (array): Array to left-multiply by metric.

        Returns
            array: Result of performing `metric @ other`.
        """
        return NotImplementedError()

    def lmult_inv(self, other):
        """Evaluate left-multiplication of argument by inverse metric.

        Args:
            other (array): Array to left-multiply by inverse metric.

        Returns
            array: Result of performing `inv(metric) @ other`.
        """
        return NotImplementedError()

    def lmult_sqrt(self, other):
        """Evaluate left-multiplication of argument by square-root of metric.

        Args:
            other (array): Array to left-multiply by square-root of metric.

        Returns
            array: Result of performing `sqrtm(metric) @ other`.
        """
        return NotImplementedError()


@inherit_docstrings
class IsotropicEuclideanMetric(_BaseEuclideanMetric):
    """Euclidean metric corresponding to an identity matrix.

    Equivalent to a kinetic energy of the form

        kin_energy(mom) = 0.5 * mom @ mom

    i.e. no rescaling is performed.
    """

    def lmult(self, other):
        return other

    def lmult_inv(self, other):
        return other

    def lmult_sqrt(self, other):
        return other


@inherit_docstrings
class DiagonalEuclideanMetric(_BaseEuclideanMetric):
    """Euclidean metric corresponding to a diagonal matrix.

    Equivalent to a kinetic energy of the form

        kin_energy(mom) = 0.5 * (mom / metric_diagonal) @ mom

    i.e. a per component constant rescaling by metric is performed.
    """

    def __init__(self, metric_diagonal):
        """
        Args:
            metric_diagonal (array): One-dimensional array of positive values
                specifying diagonal elements of metric.
        """
        if not metric_diagonal.ndim == 1:
            raise ValueError('metric_diagonal should be a 1D array.')
        if not np.all(metric_diagonal > 0.):
            raise ValueError('metric_diagonal should be all positive.')
        self.diagonal = metric_diagonal

    def lmult(self, other):
        return (other.T * self.diagonal).T

    def lmult_inv(self, other):
        return (other.T / self.diagonal).T

    def lmult_sqrt(self, other):
        return (other.T * self.diagonal**0.5).T


@inherit_docstrings
class DenseEuclideanMetric(_BaseEuclideanMetric):
    """Euclidean metric corresponding to a dense matrix.

    Equivalent to a kinetic energy of the form

        kin_energy(mom) = 0.5 * mom @ inv(metric) @ mom

    where `inv` indicates the matrix inverse.
    """

    def __init__(self, metric):
        """
        Args:
            metric (array): Two-dimensional array specifying metric. Should be
                symmetric and positive-definite.
        """
        if not metric.ndim == 2:
            raise ValueError('metric should be a two-dimensional array.')
        if not np.allclose(metric, metric.T):
            raise ValueError('metric should be a symmetric matrix (2D array).')
        self.metric = metric
        try:
            self.chol = sla.cholesky(metric, lower=True)
        except sla.LinAlgError as e:
            raise ValueError(
                'metric should be a positive-definite matrix (2D array).')

    def lmult(self, other):
        return self.metric @ other

    def lmult_inv(self, other):
        return sla.cho_solve((self.chol, True), other)

    def lmult_sqrt(self, other):
        return self.chol @ other


class _BaseRiemannianMetric(object):
    """Base class for Riemannian metrics."""

    def lmult(self, state, other):
        """Evaluate left-multiplication of argument by metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.
            other (array): Array to left-multiply by metric.

        Returns:
            array: Result of performing `metric(state.pos) @ other`.
        """
        raise NotImplementedError()

    def lmult_inv(self, state, other):
        """Evaluate left-multiplication of argument by inverse metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.
            other (array): Array to left-multiply by inverse metric.

        Returns:
            array: Result of performing `inv(metric(state.pos)) @ other`.
        """
        raise NotImplementedError()

    def lmult_sqrt(self, state, other):
        """Evaluate left-multiplication of argument by square-root of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.
            other (array): Array to left-multiply by square-root of metric.

        Returns:
            array: Result of performing `sqrtm(metric(state.pos)) @ other`.
        """
        raise NotImplementedError()

    def log_det_sqrt(self, state):
        """Evaluate logarithm of square-root of determinant of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            float: Value of `log(det(metric(state.pos))**0.5)`.
        """
        raise NotImplementedError()

    def grad_log_det_sqrt(self, state):
        """Evaluate gradient of log-determinant square root of metric.

        Derivatives are with respect to position coordinates of state.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Value of `grad(log(det(metric)**0.5))(state.pos)`.
        """
        raise NotImplementedError()

    def quadratic_form_inv(self, state, vector):
        """Evaluate the quadratic form associated with the inverse metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.
            vector (array): Vector to evaluate quadratic form for.

        Returns:
            float: Value of `vector @ inv(metric(state.pos)) @ vector`.
        """
        return np.sum(vector * self.lmult_inv(state, vector))

    def grad_quadratic_form_inv(self, state, vector):
        """Evaluate the gradient of the quadratic form for the inverse metric.

        Derivatives are with respect to position coordinates of state.

        Args:
            state (HamiltonianState): State to evaluate metric at.
            vector (array): Vector to evaluate quadratic form for.

        Returns:
            float: Value of `grad(vector @ inv(metric) @ vector)(state.pos)`.
        """
        raise NotImplementedError()


@inherit_docstrings
class DiagonalRiemannianMetric(_BaseRiemannianMetric):
    """Riemannian metric with non-zero terms only on diagonal."""

    def __init__(self, diagonal_func, vjp_diagonal_func=None):
        """
        Args:
            diagonal_func: Function to evaluate diagonal of metric.
            vjp_diagonal_func: Higher-order function which returns a function
                to evaluate the vector-Jacobian-product for the diagonal of the
                metric evaluated at provided state (position).
        """
        self._diagonal = diagonal_func
        self._vjp_diagonal = autodiff_fallback(
            vjp_diagonal_func, diagonal_func, 'vjp_and_value',
            'vjp_diagonal_func')

    @cache_in_state('pos')
    def diagonal(self, state):
        return self._diagonal(state.pos)

    @multi_cache_in_state(['pos'], ['vjp_diagonal', 'diagonal'])
    def vjp_diagonal(self, state):
        return self._vjp_diagonal(state.pos)

    def lmult(self, state, other):
        return (other.T * self.diagonal(state)).T

    def lmult_inv(self, state, other):
        return (other.T / self.diagonal(state)).T

    def lmult_sqrt(self, state, other):
        return (other.T * self.diagonal(state)**0.5).T

    def log_det_sqrt(self, state):
        return 0.5 * np.log(self.diagonal(state)).sum()

    @cache_in_state('pos')
    def grad_log_det_sqrt(self, state):
        return 0.5 * self.vjp_diagonal(state)(1. / self.diagonal(state))

    def grad_quadratic_form_inv(self, state, vector):
        inv_metric_vector = self.lmult_inv(state, vector)
        return -self.vjp_diagonal(state)(inv_metric_vector**2)


@inherit_docstrings
class _BaseCholeskyRiemannianMetric(_BaseRiemannianMetric):
    """Base class for Cholesky factored Riemannian metrics."""

    def chol(self, state):
        """Evaluate Cholesky factor of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Result of performing `chol(metric(state.pos))`.
        """
        raise NotImplementedError()

    def lmult_inv(self, state, other):
        return sla.cho_solve((self.chol(state), True), other)

    def lmult_sqrt(self, state, other):
        return self.chol(state) @ other

    @cache_in_state('pos')
    def inv(self, state):
        """Evaluate inverse of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Result of performing `inv(metric(state.pos))`.
        """
        return self.lmult_inv(state, np.eye(state.mom.shape[0]))

    @cache_in_state('pos')
    def log_det_sqrt(self, state):
        return np.log(self.chol(state).diagonal()).sum()


@inherit_docstrings
class CholeskyRiemannianMetric(_BaseCholeskyRiemannianMetric):
    """Riemannian metric specified by its Cholesky factor.

    Efficient implementation of a Riemannian metric for which an explicit
    expression for the lower-triangular Cholesky factor of the metric can be
    found.
    """

    def __init__(self, chol_func, vjp_chol_func=None):
        """
        Args:
            chol_func: Function to evaluate lower Cholesky factor of metric.
            vjp_chol_func: Higher-order function which returns a function to
                evaluate the vector-Jacobian-product for the Cholesky of the
                metric evaluated at provided state (position).
        """
        self._chol = chol_func
        self._vjp_chol = autodiff_fallback(
            vjp_chol_func, chol_func, 'vjp_and_value', 'vjp_chol_func')

    @cache_in_state('pos')
    def chol(self, state):
        return self._chol(state.pos)

    @multi_cache_in_state(['pos'], ['vjp_chol', 'chol'])
    def vjp_chol(self, state):
        return self._vjp_chol(state.pos)

    def lmult_inv_chol(self, state, other):
        return sla.solve_triangular(self.chol(state), other, lower=True)

    @cache_in_state('pos')
    def inv_chol(self, state):
        """Evaluate inverse of Cholesky factor of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Result of performing `inv(chol(metric(state.pos)))`.
        """
        return self.lmult_inv_chol(state, np.eye(state.mom.shape[0]))

    @cache_in_state('pos')
    def grad_log_det_sqrt(self, state):
        return self.vjp_chol(state)(self.inv_chol(state).T)

    def grad_quadratic_form_inv(self, state, vector):
        inv_chol_metric_vector = self.lmult_inv_chol(state, vector)
        inv_metric_vector = self.lmult_inv(state, vector)
        return -2 * self.vjp_chol(state)(
            np.outer(inv_metric_vector, inv_chol_metric_vector))


@inherit_docstrings
class DenseRiemannianMetric(_BaseCholeskyRiemannianMetric):
    """General purpose class for dense Riemannian metrics."""

    def __init__(self, metric_func, vjp_metric_func=None):
        """
        Args:
            metric_func: Function to evaluate metric. Should return a positive-
                definite matrix for all arguments.
            vjp_metric_func: Higher-order function which returns a function to
                evaluate the vector-Jacobian-product for the metric evaluated
                at a provided state (position).
        """
        self._value = metric_func
        self._vjp = autodiff_fallback(
            vjp_metric_func, metric_func, 'vjp_and_value', 'vjp_metric_func')

    @cache_in_state('pos')
    def value(self, state):
        """Evaluate metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Result of performing `metric(state.pos)`.
        """
        return self._value(state.pos)

    @multi_cache_in_state(['pos'], ['vjp', 'value'])
    def vjp(self, state):
        return self._vjp(state.pos)

    @cache_in_state('pos')
    def chol(self, state):
        return sla.cholesky(self.value(state), True)

    @cache_in_state('pos')
    def grad_log_det_sqrt(self, state):
        return 0.5 * self.vjp(state)(self.inv(state))

    def grad_quadratic_form_inv(self, state, vector):
        inv_metric_vector = self.lmult_inv(state, vector)
        return -self.vjp(state)(np.outer(inv_metric_vector, inv_metric_vector))


@inherit_docstrings
class SoftAbsRiemannianMetric(_BaseRiemannianMetric):
    """Riemannian metric based on regularisation of potential energy Hessian.

    The metric is computed as

        hess = hess_pot_energy(state.pos)
        eigval, eigvec = eigh(hess)
        metric_eigval = softabs(eigval, softabs_coeff)
        metric = eigvec @ diag(metric_eigval) @ eigvec.T

    where `hess_pot_energy` is a function which computes the Hessian of the
    potential energy function, `eigh` computes the eigendecomposition of a
    Hermitian matrix and `softabs` is a smooth approximation to the absolute
    function with positive regularisation coefficient `softabs_coeff`

       softabs(eigval, softabs_coeff) = eigval / tanh(eigval * softabs_coeff)

    This eigenvalue regularisation ensures the metric is positive-definite
    even when the Hessian is not [1].

    References:

    1.  Betancourt, M., 2013. A general metric for Riemannian manifold
        Hamiltonian Monte Carlo. In *Geometric science of information*
        (pp. 327-334).
    """

    def __init__(self, system, softabs_coeff=1.):
        """
        Args:
            system (HamiltonianSystem): Hamiltonian system specifying potential
                energy function (and associated Hessian) to use to define
                metric.
            softabs_coeff (float): Positive regularisation coefficient for
                smooth approximation to absolute value. As the value tends to
                infinity the approximation becomes increasingly close to the
                absolute function.
        """
        self.system = system
        if softabs_coeff <= 0:
            raise ValueError('softabs_coeff must be positive.')
        self.softabs_coeff = softabs_coeff

    def softabs(self, x):
        """Smooth approximation to absolute function."""
        return x / np.tanh(x * self.softabs_coeff)

    def grad_softabs(self, x):
        """Derivative of smooth approximation to absolute function."""
        return (
            1. / np.tanh(self.softabs_coeff * x) -
            self.softabs_coeff * x / np.sinh(self.softabs_coeff * x)**2)

    @cache_in_state('pos')
    def eig(self, state):
        """Eigendecomposition of metric and potential energy Hessian.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            metric_eigval (array): Eigenvalues of metric.
            hess_eigval (array): Eigenvalues of potential energy Hessian.
            eigvec (array): Eigenvectors of metric (and pot. energy Hessian).
        """
        hess = self.system.hess_pot_energy(state)
        hess_eigval, eigvec = sla.eigh(hess)
        metric_eigval = self.softabs(hess_eigval)
        return metric_eigval, hess_eigval, eigvec

    @cache_in_state('pos')
    def sqrt(self, state):
        """Matrix square-root of metric.

        Args:
            state (HamiltonianState): State to evaluate metric at.

        Returns:
            array: Result of performing `sqrtm(metric(state.pos))`.
        """
        metric_eigval, hess_eigval, eigvec = self.eig(state)
        return eigvec * metric_eigval**0.5

    def lmult_sqrt(self, state, other):
        return self.sqrt(state) @ other

    def lmult_inv(self, state, other):
        metric_eigval, hess_eigval, eigvec = self.eig(state)
        return (eigvec / metric_eigval) @ (eigvec.T @ other)

    @cache_in_state('pos')
    def log_det_sqrt(self, state):
        metric_eigval, hess_eigval, eigvec = self.eig(state)
        return 0.5 * np.log(metric_eigval).sum()

    @cache_in_state('pos')
    def grad_log_det_sqrt(self, state):
        metric_eigval, hess_eigval, eigvec = self.eig(state)
        return 0.5 * self.system.mtp_pot_energy(state)(
            eigvec * self.grad_softabs(hess_eigval) / metric_eigval @ eigvec.T)

    def grad_quadratic_form_inv(self, state, vector):
        metric_eigval, hess_eigval, eigvec = self.eig(state)
        num_j_mtx = metric_eigval[:, None] - metric_eigval[None, :]
        num_j_mtx += np.diag(self.grad_softabs(hess_eigval))
        den_j_mtx = hess_eigval[:, None] - hess_eigval[None, :]
        np.fill_diagonal(den_j_mtx, 1)
        j_mtx = num_j_mtx / den_j_mtx
        e_vct = (eigvec.T @ vector) / metric_eigval
        return -self.system.mtp_pot_energy(state)(
            eigvec @ (np.outer(e_vct, e_vct) * j_mtx) @ eigvec.T)
