"""Hamiltonian systems encapsulating energy functions and their derivatives."""

from abc import ABC, abstractmethod
import logging
import numpy as np
from mici.states import cache_in_state, multi_cache_in_state
from mici.matrices import (
    IdentityMatrix, PositiveScaledIdentityMatrix, PositiveDiagonalMatrix,
    DenseSquareMatrix, TriangularFactoredPositiveDefiniteMatrix,
    DenseDefiniteMatrix, DensePositiveDefiniteMatrix, PositiveDefiniteMatrix,
    EigendecomposedSymmetricMatrix, SoftAbsRegularisedPositiveDefiniteMatrix)
from mici.autodiff import autodiff_fallback


class System(ABC):
    r"""Base class for Hamiltonian systems.

    The Hamiltonian function \(h\) is assumed to have the general form

    \[ h(q, p) = h_1(q) + h_2(q, p) \]

    where \(q\) and \(p\) are the position and momentum variables respectively,
    and \(h_1\) and \(h_2\) Hamiltonian component functions. The exact
    Hamiltonian flow for the \(h_1\) component can be always be computed as it
    depends only on the position variable however depending on the form of
    \(h_2\) the corresponding exact Hamiltonian flow may or may not be
    simulable.

    By default \(h_1\) is assumed to correspond to the negative logarithm of an
    unnormalised density on the position variables with respect to the Lebesgue
    measure, with the corresponding distribution on the position space being
    the target distribution it is wished to draw approximate samples from.
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalised probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                the negative logarithm of the unnormalised density specified by
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a pair of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        self._neg_log_dens = neg_log_dens
        self._grad_neg_log_dens = autodiff_fallback(
            grad_neg_log_dens, neg_log_dens,
            'grad_and_value', 'grad_neg_log_dens')

    @cache_in_state('pos')
    def neg_log_dens(self, state):
        """Negative logarithm of unnormalised density of target distribution.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of computed negative log density.
        """
        return self._neg_log_dens(state.pos)

    @multi_cache_in_state(['pos'], ['grad_neg_log_dens', 'neg_log_dens'])
    def grad_neg_log_dens(self, state):
        """Derivative of negative log density with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `neg_log_dens(state)` derivative with respect to
                `state.pos`.
        """
        return self._grad_neg_log_dens(state.pos)

    def h1(self, state):
        """Hamiltonian component depending only on position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of `h1` Hamiltonian component.
        """
        return self.neg_log_dens(state)

    def dh1_dpos(self, state):
        """Derivative of `h1` Hamiltonian component with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of computed `h1` derivative.
        """
        return self.grad_neg_log_dens(state)

    def h1_flow(self, state, dt):
        """Apply exact flow map corresponding to `h1` Hamiltonian component.

        `state` argument is modified in place.

        Args:
            state (mici.states.ChainState): State to start flow at.
            dt (float): Time interval to simulate flow for.
        """
        state.mom -= dt * self.dh1_dpos(state)

    @abstractmethod
    def h2(self, state):
        """Hamiltonian component depending on momentum and optionally position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of `h2` Hamiltonian component.
        """

    @abstractmethod
    def dh2_dmom(self, state):
        """Derivative of `h2` Hamiltonian component with respect to momentum.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h2(state)` derivative with respect to `state.pos`.
        """

    def h(self, state):
        """Hamiltonian function for system.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of Hamiltonian.
        """
        return self.h1(state) + self.h2(state)

    def dh_dpos(self, state):
        """Derivative of Hamiltonian with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h(state)` derivative with respect to `state.pos`.
        """
        if hasattr(self, 'dh2_dpos'):
            return self.dh1_dpos(state) + self.dh2_dpos(state)
        else:
            return self.dh1_dpos(state)

    def dh_dmom(self, state):
        """Derivative of Hamiltonian with respect to momentum.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `h(state)` derivative with respect to `state.mom`.
        """
        return self.dh2_dmom(state)

    @abstractmethod
    def sample_momentum(self, state, rng):
        """
        Sample a momentum from its conditional distribution given a position.

        Args:
            state (mici.states.ChainState): State defining position to
               condition on.

        Returns:
            mom (array): Sampled momentum.
        """


class EuclideanMetricSystem(System):
    r"""Hamiltonian system with a Euclidean metric on the position space.

    Here Euclidean metric is defined to mean a metric with a fixed positive
    definite matrix representation \(M\). The momentum variables are taken to
    be independent of the position variables and with a zero-mean Gaussian
    marginal distribution with covariance specified by \(M\), so that the
    \(h_2\) Hamiltonian component is

    \[ h_2(q, p) = \frac{1}{2} p^T M^{-1} p \]

    where \(q\) and \(p\) are the position and momentum variables respectively.

    The \(h_1\) Hamiltonian component function is

    \[ h_1(q) = \ell(q) \]

    where \(\ell(q)\) is the negative log (unnormalised) density of
    the target distribution with respect to the Lebesgue measure.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalised probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric (None or array or Matrix): Matrix object corresponding to
                matrix representation of metric on position space and
                covariance of Gaussian marginal distribution on momentum
                vector. If `None` is passed (the default), the identity matrix
                will be used. If a 1D array is passed then this is assumed to
                specify a metric with diagonal matrix representation and the
                array to the matrix diagonal. If a 2D array is passed then this
                is assumed to specify a metric with a dense positive definite
                matrix representation specified by the array.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                the negative logarithm of the unnormalised density specified by
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a pair of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(neg_log_dens, grad_neg_log_dens)
        if metric is None:
            self.metric = IdentityMatrix()
        elif isinstance(metric, np.ndarray):
            if metric.ndim == 1:
                self.metric = PositiveDiagonalMatrix(metric)
            elif metric.ndim == 2:
                self.metric = DensePositiveDefiniteMatrix(metric)
            else:
                raise ValueError('If NumPy ndarray value is used for `metric`'
                                 ' must be either 1D (diagonal matrix) or 2D '
                                 '(dense positive definite matrix)')
        else:
            self.metric = metric

    @cache_in_state('mom')
    def h2(self, state):
        return 0.5 * state.mom @ (self.metric.inv @ state.mom)

    @cache_in_state('mom')
    def dh2_dmom(self, state):
        return self.metric.inv @ state.mom

    def h2_flow(self, state, dt):
        state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        return dt * self.metric.inv, IdentityMatrix(self.metric.shape[0])

    def sample_momentum(self, state, rng):
        return self.metric.sqrt @ rng.standard_normal(state.pos.shape)


class GaussianEuclideanMetricSystem(EuclideanMetricSystem):
    r"""Euclidean Hamiltonian system with a tractable Gaussian component.

    Here Euclidean metric is defined to mean a metric with a fixed positive
    definite matrix representation \(M\). The momentum variables are taken to
    be independent of the position variables and with a zero-mean Gaussian
    marginal distribution with covariance specified by \(M\).

    Additionally the target distribution on the position variables is assumed
    to be defined by an unnormalised density with respect to the standard
    Gaussian measure on the position space (with identity covariance and zero
    mean), with the Hamiltonian component \(h_1\) corresponding to the negative
    logarithm of this density rather than the density with respect to the
    Lebesgue measure on the position space, i.e.

    \[ h_1(q) = \ell(q) - \frac{1}{2} q^T q \]

    where \(q\) is the position and \(\ell(q)\) is the negative log
    (unnormalised) density of the target distribution with respect to the
    Lebesgue measure at \(q\). The Hamiltonian  component function \(h_2\) is
    then assumed to have the form

    \[ h_2(q, p) = \frac{1}{2} q^T q + \frac{1}{2} p^T M^{-1} p \]

    where \(p\) is the momentum. In this case the Hamiltonian flow due to the
    quadratic \(h_2\) component can be solved for analytically, allowing an
    integrator to be defined using this alternative splitting of the
    Hamiltonian [1].

    References:

      1. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014. Split
         Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalised probability density on the position space with
                respect to the standard Gaussian measure on the position space,
                with the corresponding distribution on the position space being
                the target distribution it is wished to draw approximate
                samples from.
            metric (None or array or Matrix): Matrix object corresponding to
                matrix representation of metric on position space and
                covariance of Gaussian marginal distribution on momentum
                vector. If `None` is passed (the default), the identity matrix
                will be used. If a 1D array is passed then this is assumed to
                specify a metric with diagonal matrix representation and the
                array to the matrix diagonal. If a 2D array is passed then this
                is assumed to specify a metric with a dense positive definite
                matrix representation specified by the array.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position vector returns the derivative
                of the negative logarithm of the unnormalised density specified
                by `neg_log_dens` with respect to its position vector argument.
                Optionally the function may instead return a pair of values
                with the first being the value of the `neg_log_dens` evaluated
                at the passed position vector and the second being the value of
                its derivative with respect to the position argument. If `None`
                is passed (the default) an automatic differentiation backend
                will be used to construct the derivative of `neg_log_dens`
                automatically if available.
        """
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

    def dh2_flow_dmom(self, dt):
        omega = 1. / self.metric.eigval**0.5
        sin_omega_dt, cos_omega_dt = np.sin(omega * dt), np.cos(omega * dt)
        return (
            EigendecomposedSymmetricMatrix(
                self.metric.eigvec, sin_omega_dt * omega),
            EigendecomposedSymmetricMatrix(
                self.metric.eigvec, cos_omega_dt))


class ConstrainedEuclideanMetricSystem(EuclideanMetricSystem):
    """Base class for Euclidean Hamiltonian systems subject to constraints.

    The system is assumed to be subject to a set of holonomic constraints on
    the position component of the state. These constraints are specified by a
    vector constraint function which takes as argument the position component,
    and which is equal to zero in all components when the constraints are
    satisfied. The constraint function implicitly defines a manifold embedded
    in the position space of constraint satisfying configurations. There are
    also implicitly a set of constraints on the momentum component of the state
    due to the requirement that velocity (momentum pre-multiplied by inverse
    metric) is always tangential to the constraint manifold.
    """

    def __init__(self, neg_log_dens, constr, metric=None,
                 dens_wrt_hausdorff=True, grad_neg_log_dens=None,
                 jacob_constr=None):
        super().__init__(neg_log_dens=neg_log_dens, metric=metric,
                         grad_neg_log_dens=grad_neg_log_dens)
        self._constr = constr
        self.dens_wrt_hausdorff = dens_wrt_hausdorff
        self._jacob_constr = autodiff_fallback(
            jacob_constr, constr, 'jacobian_and_value', 'jacob_constr')

    @cache_in_state('pos')
    def constr(self, state):
        return self._constr(state.pos)

    @multi_cache_in_state(['pos'], ['jacob_constr', 'constr'])
    def jacob_constr(self, state):
        return self._jacob_constr(state.pos)

    @abstractmethod
    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None):
        """Compute inner product of rows of constraint Jacobian matrices.

        Computes `jacob_constr_1 @ inner_product_matrix @ jacob_constr_2.T`
        potentially exploiting any structure / sparsity in `jacob_constr_1`,
        `jacob_constr_2` and `inner_product_matrix`.

        Args:
            jacob_constr_1 (Matrix): First constraint Jacobian in product.
            inner_product_matrix (Matrix): Positive-definite matrix defining
                inner-product between rows of two constraint Jacobians.
            jacob_constr_2 (None or Matrix): Second constraint Jacobian in
                product. Defaults to `jacob_constr_1` if set to `None`.

        Returns
            Matrix: Object corresponding to computed inner products of
               the constraint Jacobian rows.
        """

    @cache_in_state('pos')
    def gram(self, state):
        """Gram matrix at current position."""
        return self.jacob_constr_inner_product(
            self.jacob_constr(state), self.metric.inv)

    def inv_gram(self, state):
        """Inverse of Gram matrix at current position."""
        return self.gram(state).inv

    def log_det_sqrt_gram(self, state):
        """Value of (half of) log-determinant of Gram matrix."""
        return 0.5 * self.gram(state).log_abs_det

    @abstractmethod
    def grad_log_det_sqrt_gram(self, state):
        """Derivative of (half of) log-determinant of Gram matrix wrt position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `log_det_sqrt_gram(state)` derivative with respect
            to `state.pos`.
        """

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
        """Project a momentum on to the co-tangent space at a position."""
        # Use parenthesis to force right-to-left evaluation to avoid
        # matrix-matrix products
        mom -= (self.jacob_constr(state).T @ (
                    self.inv_gram(state) @ (
                        self.jacob_constr(state) @ (self.metric.inv @ mom))))
        return mom

    def sample_momentum(self, state, rng):
        """Sample a momentum from its conditional at the current position."""
        mom = super().sample_momentum(state, rng)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom


class DenseConstrainedEuclideanMetricSystem(ConstrainedEuclideanMetricSystem):
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

    def __init__(self, neg_log_dens, constr, metric=None,
                 dens_wrt_hausdorff=True, grad_neg_log_dens=None,
                 jacob_constr=None, mhp_constr=None):
        super().__init__(neg_log_dens=neg_log_dens, constr=constr,
                         metric=metric, dens_wrt_hausdorff=dens_wrt_hausdorff,
                         grad_neg_log_dens=grad_neg_log_dens,
                         jacob_constr=jacob_constr)
        if not dens_wrt_hausdorff:
            self._mhp_constr = autodiff_fallback(
                mhp_constr, constr, 'mhp_jacobian_and_value', 'mhp_constr')

    @multi_cache_in_state(
        ['pos'], ['mhp_constr', 'jacob_constr', 'constr'])
    def mhp_constr(self, state):
        return self._mhp_constr(state.pos)

    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None):
        sign = 2 * isinstance(inner_product_matrix, PositiveDefiniteMatrix) - 1
        if jacob_constr_2 is None or jacob_constr_2 is jacob_constr_1:
            return DenseDefiniteMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_1.T),
                sign=sign)
        else:
            return DenseSquareMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_2.T))

    @cache_in_state('pos')
    def grad_log_det_sqrt_gram(self, state):
        # Evaluate MHP of constraint function before Jacobian as Jacobian value
        # will potentially be computed in 'forward' pass and cached
        mhp_constr = self.mhp_constr(state)
        return mhp_constr(
            self.inv_gram(state) @ self.jacob_constr(state) @ self.metric.inv)


class GaussianDenseConstrainedEuclideanMetricSystem(
        GaussianEuclideanMetricSystem, DenseConstrainedEuclideanMetricSystem):

    def __init__(self, neg_log_dens, constr, metric=None,
                 dens_wrt_hausdorff=True, grad_neg_log_dens=None,
                 jacob_constr=None, mhp_constr=None):
        DenseConstrainedEuclideanMetricSystem.__init__(
            self, neg_log_dens=neg_log_dens, constr=constr, metric=metric,
            dens_wrt_hausdorff=dens_wrt_hausdorff,
            grad_neg_log_dens=grad_neg_log_dens, jacob_constr=jacob_constr,
            mhp_constr=mhp_constr)


class RiemannianMetricSystem(System):
    r"""Riemannian Hamiltonian system with a position-dependent metric.

    The position space is assumed to be a Riemannian manifold with a metric
    with position-dependent positive definite matrix-representation \(M(q)\)
    where \(q\) is a position vector. The momentum \(p\) is then taken to have
    a zero-mean Gaussian conditional distribution given the position \(q\),
    with covariance \(M(q)\), i.e. \(p \sim \mathcal{N}(0, M(q))\) [1].

    The \(h_1\) Hamiltonian component is then

    \[ h_1(q) = \ell(q) + \frac{1}{2}\log\left|M(q)\right| \]

    where \(\ell(q)\) is the negative log (unnormalised) density of the target
    distribution with respect to the Lebesgue measure at \(q\). The \(h_2\)
    Hamiltonian component is

    \[ h_2(q, p) = \frac{1}{2} p^T (M(q))^{-1} p. \]

    Due to the coupling between the position and momentum variables in \(h_2\),
    the Hamiltonian system is non-separable, requiring use of a numerical
    integrator with implicit steps.

    References:

      1. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
         Hamiltonian Monte Varlo methods. Journal of the Royal Statistical
         Society: Series B (Statistical Methodology), 73(2), pp.123-214.
    """

    def __init__(self, neg_log_dens, metric_matrix_class, metric_func,
                 vjp_metric_func=None, grad_neg_log_dens=None,
                 metric_kwargs=None):
        self._metric_matrix_class = metric_matrix_class
        self._metric_func = metric_func
        self._vjp_metric_func = autodiff_fallback(
            vjp_metric_func, metric_func, 'vjp_and_value', 'vjp_metric_func')
        self._metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        super().__init__(neg_log_dens, grad_neg_log_dens)

    @cache_in_state('pos')
    def metric_func(self, state):
        return self._metric_func(state.pos)

    @multi_cache_in_state(['pos'], ['vjp_metric_func', 'metric_func'])
    def vjp_metric_func(self, state):
        return self._vjp_metric_func(state.pos)

    @cache_in_state('pos')
    def metric(self, state):
        return self._metric_matrix_class(
            self.metric_func(state), **self._metric_kwargs)

    def h(self, state):
        return self.h1(state) + self.h2(state)

    def h1(self, state):
        return self.neg_log_dens(state) + 0.5 * self.metric(state).log_abs_det

    def dh1_dpos(self, state):
        # Evaluate VJP of metric function before metric as metric value will
        # potentially be computed in forward pass and cached
        vjp_metric = self.vjp_metric_func(state)
        return (self.grad_neg_log_dens(state) +
                0.5 * vjp_metric(self.metric(state).grad_log_abs_det))

    def h2(self, state):
        return 0.5 * state.mom @ self.metric(state).inv @ state.mom

    def dh2_dpos(self, state):
        # Evaluate VJP of metric function before metric as metric value will
        # potentially be computed in forward pass and cached
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
            neg_log_dens, TriangularFactoredPositiveDefiniteMatrix,
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
                         grad_neg_log_dens,
                         metric_kwargs={'softabs_coeff': softabs_coeff})

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
