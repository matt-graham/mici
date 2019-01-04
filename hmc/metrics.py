"""Euclidean and Riemannian metric tensor classes."""

import numpy as np
import scipy.linalg as sla
from hmc.states import cache_in_state, multi_cache_in_state
from hmc.autodiff import autodiff_fallback


class BaseEuclideanMetric(object):

    def quadratic_form_inv(self, vector):
        return np.sum(vector * self.lmult_inv(vector))

    def lmult(self, rhs):
        return NotImplementedError()

    def lmult_inv(self, rhs):
        return NotImplementedError()

    def lmult_sqrt(self, rhs):
        return NotImplementedError()


class IsotropicEuclideanMetric(BaseEuclideanMetric):

    def lmult(self, other):
        return other

    def lmult_inv(self, other):
        return other

    def lmult_sqrt(self, other):
        return other


class DiagonalEuclideanMetric(BaseEuclideanMetric):

    def __init__(self, metric_diagonal):
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


class DenseEuclideanMetric(BaseEuclideanMetric):

    def __init__(self, matrix):
        self.matrix = matrix
        self.chol = sla.cholesky(matrix, lower=True)

    def lmult(self, other):
        return self.matrix @ other

    def lmult_inv(self, other):
        return sla.cho_solve((self.chol, True), other)

    def lmult_sqrt(self, other):
        return self.chol @ other


class BaseRiemannianMetric(object):

    def lmult_inv(self, state, other):
        raise NotImplementedError()

    def lmult_sqrt(self, state, other):
        raise NotImplementedError()

    def log_det_sqrt(self, state):
        raise NotImplementedError()

    def grad_log_det_sqrt(self, state):
        raise NotImplementedError()

    def quadratic_form_inv(self, state, vector):
        return np.sum(vector * self.lmult_inv(state, vector))

    def grad_quadratic_form_inv(self, state, vector):
        raise NotImplementedError()


class BaseCholeskyRiemannianMetric(BaseRiemannianMetric):

    def __init__(self, metric, vjp_metric):
        self._metric = metric
        self._vjp_metric = autodiff_fallback(
            vjp_metric, metric, 'make_vjp', 'vjp_metric')

    def chol(self, state):
        raise NotImplementedError()

    def lmult_inv(self, state, other):
        return sla.cho_solve((self.chol(state), True), other)

    def lmult_sqrt(self, state, other):
        return self.chol(state) @ other

    @cache_in_state('pos')
    def inv(self, state):
        return self.lmult_inv(state, np.eye(state.n_dim))

    @cache_in_state('pos')
    def log_det_sqrt(self, state):
        return np.log(self.chol(state).diagonal()).sum()


class CholeskyRiemannianMetric(BaseCholeskyRiemannianMetric):

    def __init__(self, chol_func, vjp_chol_func=None):
        self._chol = chol_func
        self._vjp_chol = autodiff_fallback(
            vjp_chol_func, chol_func, 'make_vjp', 'vjp_chol_func')

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
        return self.lmult_inv_chol(state, np.eye(state.n_dim))

    @cache_in_state('pos')
    def grad_log_det_sqrt(self, state):
        return self.vjp_chol(state)(self.inv_chol(state).T)

    def grad_quadratic_form_inv(self, state, vector):
        inv_chol_metric_vector = self.lmult_inv_chol(state, vector)
        inv_metric_vector = self.lmult_inv(state, vector)
        return -2 * self.vjp_chol(state)(
            np.outer(inv_metric_vector, inv_chol_metric_vector))


class DenseRiemannianMetric(BaseCholeskyRiemannianMetric):

    def __init__(self, metric_func, vjp_metric_func=None):
        self._value = metric_func
        self._vjp = autodiff_fallback(
            vjp_metric_func, metric_func, 'make_vjp', 'vjp_metric_func')

    @cache_in_state('pos')
    def value(self, state):
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


class SoftAbsRiemannianMetric(BaseRiemannianMetric):

    def __init__(self, system, softabs_coeff=1.):
        self.system = system
        self.softabs_coeff = softabs_coeff

    def softabs(self, x):
        return x / np.tanh(x * self.softabs_coeff)

    def grad_softabs(self, x):
        return (
            1. / np.tanh(self.softabs_coeff * x) -
            self.softabs_coeff * x / np.sinh(self.softabs_coeff * x)**2)

    @cache_in_state('pos')
    def eig(self, state):
        hess = self.system.hess_pot_energy(state)
        hess_eigval, eigvec = sla.eigh(hess)
        metric_eigval = self.softabs(hess_eigval)
        return metric_eigval, hess_eigval, eigvec

    @cache_in_state('pos')
    def sqrt(self, state):
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
