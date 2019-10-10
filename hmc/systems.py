"""Hamiltonian systems encapsulating energy functions and their derivatives."""

import logging
import numpy as np
from hmc.states import cache_in_state, multi_cache_in_state
from hmc.matrices import (
    IdentityMatrix, PositiveScaledIdentityMatrix, PositiveDiagonalMatrix,
    DenseSquareMatrix, TriangularFactoredDefiniteMatrix,
    DensePositiveDefiniteMatrix, SoftAbsRegularisedPositiveDefiniteMatrix)
from hmc.autodiff import autodiff_fallback


class _HamiltonianSystem(object):
    """Base class for Hamiltonian systems.

    The Hamiltonian function `h` is assumed to have the general form

        h(pos, mom) = h1(pos) + h2(pos, mom)

    where `pos` and `mom` are the position and momentum variables respectively,
    and `h1(pos)` and `h2(pos, mom)` Hamiltonian components. The exact
    Hamiltonian flow for the `h1` component can be computed however depending
    on the form of `h2` the corresponding Hamiltonian flow may or may not be
    simulable.

    By default `h1` is assumed to correspond to the negative logarithm of an
    unnormalised density on the position variables with respect to the Lebesgue
    measure, with the corresponding distribution on the position space being
    the target distribution it is wished to draw approximate samples from.
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (callable): Function which given a position vector
                returns the negative logarithm of an unnormalised probability
                density on the position space with respect to the Lebesgue
                measure, with the corresponding distribution on the position
                space being the target distribution it is wished to draw
                approximate samples from.
            grad_neg_log_dens (callable or None): Function which given a
                position vector returns the derivative of the negative
                logarithm of the unnormalised density specified by
                `neg_log_dens` with respect to the position vector argument.
                Optionally the function may instead return a pair of values
                with the first being the value of the `neg_log_dens` evaluated
                at the passed position vector and the second being the value of
                its derivative with respect to the position argument. If `None`
                is passed (the default) an automatic differentiation fallback
                will be used to attempt to construct the derivative of
                `neg_log_dens` automatically.

        """
        self._neg_log_dens = neg_log_dens
        self._grad_neg_log_dens = autodiff_fallback(
            grad_neg_log_dens, neg_log_dens,
            'grad_and_value', 'grad_neg_log_dens')

    @cache_in_state('pos')
    def neg_log_dens(self, state):
        return self._neg_log_dens(state.pos)

    @multi_cache_in_state(['pos'], ['grad_neg_log_dens', 'neg_log_dens'])
    def grad_neg_log_dens(self, state):
        return self._grad_neg_log_dens(state.pos)

    def h1(self, state):
        return self.neg_log_dens(state)

    def dh1_dpos(self, state):
        return self.grad_neg_log_dens(state)

    def h1_flow(self, state, dt):
        state.mom -= dt * self.dh1_dpos(state)

    def h2(self, state):
        raise NotImplementedError()

    def h2_flow(self, state):
        raise NotImplementedError()

    def h(self, state):
        return self.h1(state) + self.h2(state)

    def dh_dpos(self, state):
        if hasattr(self, 'dh2_dpos'):
            return self.dh1_dpos(state) + self.dh2_dpos(state)
        else:
            return self.dh1_dpos(state)

    def dh_dmom(self, state):
        return self.dh2_dmom(state)

    def sample_momentum(self, state, rng):
        raise NotImplementedError()


class EuclideanMetricSystem(_HamiltonianSystem):
    """Hamiltonian system with fixed covariance Gaussian-distributed momenta.

    The momentum variables are taken to be independent of the position
    variables and with a zero-mean Gaussian marginal distribution with
    covariance specified by a fixed positive-definite matrix (metric tensor),
    so that the `h2` Hamiltonian component is

        h2(pos, mom) = 0.5 * mom @ inv(metric) @ mom

    where `pos` and `mom` are the position and momentum variables respectively,
    and `inv(metric)` is the matrix inverse of the metric tensor.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (callable): Function which given a position vector
                returns the negative logarithm of an unnormalised probability
                density on the position space with respect to the Lebesgue
                measure, with the corresponding distribution on the position
                space being the target distribution it is wished to draw
                approximate samples from.
            metric (Matrix or None): Matrix object corresponding to covariance
                of Gaussian marginal distribution on momentum vector. If `None`
                is passed (the default), the identity matrix will be used.
            grad_neg_log_dens (callable or None): Function which given a
                position vector returns the derivative of the negative
                logarithm of the unnormalised density specified by
                `neg_log_dens` with respect to the position vector argument.
                Optionally the function may instead return a pair of values
                with the first being the value of the `neg_log_dens` evaluated
                at the passed position vector and the second being the value of
                its derivative with respect to the position argument. If `None`
                is passed (the default) it will be attempted to use automatic
                differentiation to construct the derivative of `neg_log_dens`
                automatically.
        """
        super().__init__(neg_log_dens, grad_neg_log_dens)
        self.metric = IdentityMatrix() if metric is None else metric

    @cache_in_state('mom')
    def h2(self, state):
        return 0.5 * state.mom @ self.metric.inv @ state.mom

    @cache_in_state('mom')
    def dh2_dmom(self, state):
        return self.metric.inv @ state.mom

    def h2_flow(self, state, dt):
        state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, state, dt):
        return dt * self.metric.inv, IdentityMatrix(state.pos.size)

    def sample_momentum(self, state, rng):
        return self.metric.sqrt @ rng.standard_normal(state.pos.shape)


class GaussianEuclideanMetricSystem(EuclideanMetricSystem):
    """Euclidean Hamiltonian system with a tractable Gaussian component.

    The momentum variables are taken to be independent of the position
    variables and with a zero-mean Gaussian marginal distribution with
    covariance specified by a fixed positive-definite matrix (metric tensor).

    Additionally the target distribution on the position variables is assumed
    to be defined by an unnormalised density with respect to the standard
    Gaussian measure on the position space (with identity covariance and zero
    mean), with the Hamiltonian component `h1` corresponding to the negative
    logarithm of this density rather than the density with respect to the
    Lebesgue measure on the position space. The Hamiltonian component function
    `h2` is therefore assumed to have the form

         h2(pos, mom) = 0.5 * pos @ pos + 0.5 * mom @ inv(metric) @ mom

    where `pos` and `mom` are the position and momentum variables respectively,
    and `inv(metric)` is the matrix inverse of the metric tensor. In this case
    the Hamiltonian flow due to the quadratic `h2` component can be solved for
    analytically, allowing an integrator to be defined using this alternative
    splitting of the Hamiltonian [1].

    References:

      1. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014. Split
         Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (callable): Function which given a position vector
                returns the negative logarithm of an unnormalised probability
                density on the position space with respect to the standard
                Gaussian measure on the position space, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric (Matrix or None): Matrix object corresponding to covariance
                of Gaussian marginal distribution on momentum vector. If `None`
                is passed (the default), the identity matrix will be used.
            grad_neg_log_dens (callable or None): Function which given a
                position vector returns the derivative of the negative
                logarithm of the unnormalised density specified by
                `neg_log_dens` with respect to its position vector argument.
                Optionally the function may instead return a pair of values
                with the first being the value of the `neg_log_dens` evaluated
                at the passed position vector and the second being the value of
                its derivative with respect to the position argument. If `None`
                is passed (the default) an automatic differentiation backend
                will be used to construct the derivative of `neg_log_dens`
                automatically if available.
        """
        if metric is not None and not (
                isinstance(metric, DiagonalMatrix) or
                isinstance(metric, IdentityMatrix)):
            raise NotImplementedError(
                'Only currently implemented for identity and diagonal metric.')
        super().__init__(neg_log_dens, metric, grad_neg_log_dens)

    def h2(self, state):
        return (0.5 * state.pos @ state.pos +
                0.5 * state.mom @ self.metric.inv @ state.mom)

    @cache_in_state('mom')
    def dh2_dmom(self, state):
        return self.metric.inv @ state.mom

    @cache_in_state('mom')
    def dh2_dpos(self, state):
        return state.pos

    def h2_flow(self, state, dt):
        omega = 1. / self.metric.eigval**0.5
        sin_omega_dt, cos_omega_dt = np.sin(omega * dt), np.cos(omega * dt)
        eigvec_T_pos = self.metric.eigvec.T @ state.pos
        eigvec_T_mom = self.metric.eigvec.T @ state.mom
        state.pos = self.metric.eigvec @ (
            cos_omega_dt * eigvec_T_pos +
            (sin_omega_dt * omega) * eigvec_T_mom)
        state.mom = self.metric.eigvec @ (
            cos_omega_dt * eigvec_T_mom -
            (sin_omega_dt / omega) * eigvec_T_pos)

    def dh2_flow_dmom(self, state, dt):
        omega = 1. / self.metric.eigval**0.5
        sin_omega_dt, cos_omega_dt = np.sin(omega * dt), np.cos(omega * dt)
        return (
            eig_hermitian_matrix(self.metric.eigvec, sin_omega_dt * omega),
            eig_hermitian_matrix(self.metric.eigvec, cos_omega_dt))


class _ConstrainedEuclideanMetricSystem(EuclideanMetricSystem):
    """Base class for Euclidean Hamiltonian systems subject to constraints.

    The system is assumed to be subject to a set of holonomic constraints on
    the position component of the state. These constraints are specified by a
    vector constraint function which takes as argument the position component,
    and which is equal to zero in all components when the constraints are
    satisfied. The constraint function implicitly defines a manifold embedded
    in the position space of constraint satisfying configurations. There are
    also implicitly a set of constraints on the momentum component of the state
    due to the requirment that velocity (momentum pre-multiplied by inverse
    metric) is always tangential to the constraint manifold.
    """

    def __init__(self, neg_log_dens, constr, dens_wrt_hausdorff=True,
                 metric=None, grad_neg_log_dens=None, jacob_constr=None,
                 log_det_sqrt_gram=None, gram_log_det_sqrt_gram=None):
        super().__init__(neg_log_dens=neg_log_dens, metric=metric,
                         grad_neg_log_dens=grad_neg_log_dens)
        self._constr = constr
        self.dens_wrt_hausdorff = dens_wrt_hausdorff
        self._jacob_constr = autodiff_fallback(
            jacob_constr, constr, 'jacobian_and_value', 'jacob_constr')
        if not self.dens_wrt_hausdorff:
            self._mhp_constr = autodiff_fallback(
                mhp_constr, constr, 'mhp_jacobian_and_value', 'mhp_constr')

    @cache_in_state('pos')
    def constr(self, state):
        return self._constr(state.pos)

    @multi_cache_in_state(['pos'], ['jacob_constr', 'constr'])
    def jacob_constr(self, state):
        return self._jacob_constr(state.pos)

    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None):
        raise NotImplementedError()

    @cache_in_state('pos')
    def gram(self, state):
        return self.jacob_constr_inner_product(
            self.jacob_constr(state), self.metric.inv)

    def inv_gram(self, state):
        return self.gram(state).inv

    def log_det_sqrt_gram(self, state):
        return 0.5 * self.gram(state).log_det_abs_sqrt

    def grad_log_det_sqrt_gram(self, state):
        raise NotImplementedError()

    def h1(self, state):
        if self.dens_wrt_hausdorff:
            return self.neg_log_dens(state)
        else:
            return self.neg_log_dens(state) + self.log_det_sqrt_gram(state)

    def dh1_dpos(self, state):
        if self.dens_wrt_hausdorff:
            return self.grad_neg_log_dens(state)
        else:
            return (self.grad_neg_log_dens(state) +
                    self.grad_log_det_sqrt_gram(state))

    def project_onto_cotangent_space(self, mom, state):
        jacob_constr = self.jacob_constr(state)
        gram_mom = self.gram_mom(state)
        # Use parenthesis to force right-to-left evaluation to avoid
        # matrix-matrix products
        mom -= (self.jacob_constr(state).T @ (
                    self.inv_gram(state) @ (
                        self.jacob_constr(state) @ (self.metric.inv @ mom))))

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
        self.project_onto_cotangent_space(mom, state)
        return mom


class DenseConstrainedEuclideanMetricSystem(_ConstrainedEuclideanMetricSystem):
    """Euclidean Hamiltonian systems subject to a dense set of constraints.

    The system is assumed to be subject to a set of holonomic constraints on
    the position component of the state. These constraints are specified by a
    vector constraint function which takes as argument the position component,
    and which is equal to zero in all components when the constraints are
    satisfied. The constraint function implicitly defines a manifold embedded
    in the position space of constraint satisfying configurations. There are
    also implicitly a set of constraints on the momentum component of the state
    due to the requirment that velocity (momentum pre-multiplied by inverse
    metric) is always tangential to the constraint manifold.
    """

    def __init__(self, neg_log_dens, constr, dens_wrt_hausdorff=True,
                 metric=None, grad_neg_log_dens=None, jacob_constr=None,
                 mhp_constr=None):
        super().__init__(neg_log_dens=neg_log_dens, metric=metric,
                         grad_neg_log_dens=grad_neg_log_dens,
                         jacob_constr=jacob_constr)
        if not self.dens_wrt_hausdorff:
            self._mhp_constr = autodiff_fallback(
                mhp_constr, constr, 'mhp_jacobian_and_value', 'mhp_constr')

    @multi_cache_in_state(
        ['pos'], ['mhp_constr', 'jacob_constr', 'constr'])
    def mhp_constr(self, state):
        return self._mhp_constr(state.pos)

    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2):
        if jacob_constr_2 is None or jacob_constr_2 is jacob_constr_1:
            return DensePositiveDefiniteMatrix(
                jacob_constr_1 @ inner_product_matrix @ jacob_constr_1.T)
        else:
            return DenseSquareMatrix(
                jacob_constr_1 @ inner_product_matrix @ jacob_constr_2.T)

    @cache_in_state('pos')
    def grad_log_det_sqrt_gram(self, state):
        return self.mhp_constr(state)(
            self.inv_gram(state) @ self.jacob_constr(state) @ self.metric.inv)


class GaussianDenseConstrainedEuclideanMetricSystem(
        GaussianEuclideanMetricSystem, DenseConstrainedEuclideanMetricSystem):

    def __init__(self, neg_log_dens, constr, dens_wrt_hausdorff=True,
                 metric=None, grad_neg_log_dens=None, jacob_constr=None,
                 mhp_constr=None):
        ConstrainedEuclideanMetricSystem().__init__(
            neg_log_dens=neg_log_dens, metric=metric,
            grad_neg_log_dens=grad_neg_log_dens, jacob_constr=jacob_constr,
            mhp_constr=mhp_constr)


class RiemannianMetricSystem(_HamiltonianSystem):
    """Riemannian Hamiltonian system with a position dependent metric tensor.

    The momentum variables are assumed to have a zero-mean Gaussian conditional
    distribution given the position variables, with covariance specified by a
    position dependent positive-definite metric tensor [1]. Due to the coupling
    between the position and momentum variables in the quadratic form of the
    negative log density of the Gaussian conditional distribution on the
    momentum variables, the Hamiltonian system is non-separable, requiring use
    of a numerical integrator with implicit steps.

    References:

      1. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
         Hamiltonian Monte Varlo methods. Journal of the Royal Statistical
         Society: Series B (Statistical Methodology), 73(2), pp.123-214.
    """

    def __init__(self, neg_log_dens, metric_matrix_class, metric_func,
                 vjp_metric_func=None, grad_neg_log_dens=None):
        self._metric_matrix_class = metric_matrix_class
        self._metric_func = metric
        self._vjp_metric_func = autodiff_fallback(
            vjp_metric_func, metric_func, 'vjp_and_value', 'vjp_metric_func')
        super().__init__(neg_log_dens, grad_neg_log_dens)

    @cache_in_state('pos')
    def metric_func(self, state):
        return self._metric_func(state.pos)

    @multi_cache_in_state(['pos'], ['vjp_metric_func', 'metric_func'])
    def vjp_metric_func(self, state):
        return self._vjp_metric_func(state.pos)

    @cache_in_state('pos')
    def metric(self, state):
        return self._metric_matrix_class(self.metric_func(state))

    def h(self, state):
        return self.h1(state) + self.h2(state)

    def h1(self, state):
        return self.neg_log_dens(state) + self.metric(state).log_det_abs_sqrt

    def dh1_dpos(self, state):
        # Evaluate VJP of metric function before metric as metric value will
        # be computed in forward pass and cached
        vjp_metric = self.vjp_metric(state)
        return (self.grad_neg_log_dens(state) +
                vjp_metric(self.metric(state).grad_log_det_abs_sqrt))

    def h2(self, state):
        return 0.5 * state.mom @ self.metric(state).inv @ state.mom

    def dh2_dpos(self, state):
        # Evaluate VJP of metric function before metric as metric value will
        # be computed in forward pass and cached
        vjp_metric = self.vjp_metric_func(state)
        return 0.5 * vjp_metric(
            self.metric(state).grad_quadratic_form_inv(state.mom))

    def dh2_dmom(self, state):
        return self.metric(state).inv @ state.mom

    def sample_momentum(self, state, rng):
        return self.metric(state).sqrt @ rng.normal(size=state.pos.shape)


class ScalarRiemannianMetricSystem(RiemannianMetricSystem):

    def __init__(self, neg_log_dens, metric_scalar_func,
                 vjp_metric_scalar_func=None, grad_neg_log_dens=None):
        super().__init__(
            neg_log_dens, PositiveScaledIdentityMatrix, metric_scalar_func,
            vjp_metric_scalar_func, grad_neg_log_dens)


class DiagonalRiemannianMetricSystem(RiemannianMetricSystem):

    def __init__(self, neg_log_dens, metric_diagonal_func,
                 vjp_metric_diagonal_func=None, grad_neg_log_dens=None):
        super().__init__(
            neg_log_dens, PositiveDiagonalMatrix, metric_diagonal_func,
            vjp_metric_diagonal_func, grad_neg_log_dens)


class CholeskyFactoredRiemannianMetricSystem(RiemannianMetricSystem):

    def __init__(self, neg_log_dens, metric_chol_func,
                 vjp_metric_chol_func=None, grad_neg_log_dens=None):
        super().__init__(
            neg_log_dens, TriangularFactoredDefiniteMatrix,
            metric_chol_func, vjp_metric_chol_func, grad_neg_log_dens)


class DenseRiemannianMetricSystem(RiemannianMetricSystem):

    def __init__(self, neg_log_dens, metric_func,
                 vjp_metric_func=None, grad_neg_log_dens=None):
        super().__init__(
            neg_log_dens, DensePositiveDefiniteMatrix, metric_func,
            vjp_metric_func, grad_neg_log_dens)


class SoftAbsRiemannianMetricSystem(RiemannianMetricSystem):
    """SoftAbs Riemmanian metric Hamiltonian system.

    Hamiltonian system with a position dependent metric tensor which is
    specified to be an eigenvalue-regularised transformation of the Hessian of
    the potential energy function (with the 'soft-absolute' regularisation
    ensuring all the eigenvalues are strictly positive and so the resulting
    metric tensor is positive definite everywhere).

    References:

    1. Betancourt, M., 2013. A general metric for Riemannian manifold
       Hamiltonian Monte Carlo. In Geometric science of information
       (pp. 327-334).
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None,
                 hess_neg_log_dens=None, mtp_neg_log_dens=None,
                 softabs_coeff=1.):
        self._hess_neg_log_dens = autodiff_fallback(
            hess_neg_log_dens, neg_log_dens, 'hessian_grad_and_value',
            'neg_log_dens')
        self._mtp_neg_log_dens = autodiff_fallback(
            mtp_neg_log_dens, neg_log_dens, 'mtp_hessian_grad_and_value',
            'mtp_neg_log_dens')
        super().__init__(neg_log_dens,
                         SoftAbsRegularisedPositiveDefiniteMatrix,
                         self._hess_neg_log_dens, self._mtp_neg_log_dens,
                         grad_neg_log_dens)

    def metric_func(self, state):
        return self.hess_neg_log_dens(state)

    def vjp_metric_func(self, state):
        return self.mtp_neg_log_dens(state)

    @multi_cache_in_state(
        ['pos'], ['hess_neg_log_dens', 'grad_neg_log_dens', 'neg_log_dens'])
    def hess_neg_log_dens(self, state):
        return self._hess_neg_log_dens(state.pos)

    @multi_cache_in_state(
        ['pos'], ['mtp_neg_log_dens', 'hess_neg_log_dens',
                  'grad_neg_log_dens', 'neg_log_dens'])
    def mtp_neg_log_dens(self, state):
        return self._mtp_neg_log_dens(state.pos)
