"""Hamiltonian systems encapsulating energy functions and their derivatives."""

from abc import ABC, abstractmethod
import numpy as np
from mici.states import cache_in_state, cache_in_state_with_aux
import mici.matrices as matrices
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
    unnormalized density on the position variables with respect to the Lebesgue
    measure, with the corresponding distribution on the position space being
    the target distribution it is wished to draw approximate samples from.
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
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
        """Negative logarithm of unnormalized density of target distribution.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            float: Value of computed negative log density.
        """
        return self._neg_log_dens(state.pos)

    @cache_in_state_with_aux('pos', 'neg_log_dens')
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

    where \(\ell(q)\) is the negative log (unnormalized) density of
    the target distribution with respect to the Lebesgue measure.
    """

    def __init__(self, neg_log_dens, metric=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on position
                space and covariance of Gaussian marginal distribution on
                momentum vector. If `None` is passed (the default), the
                identity matrix will be used. If a 1D array is passed then this
                is assumed to specify a metric with positive diagonal matrix
                representation and the array the matrix diagonal. If a 2D array
                is passed then this is assumed to specify a metric with a dense
                positive definite matrix representation specified by the array.
                Otherwise if the value is a subclass of
                `mici.matrices.PositiveDefiniteMatrix` it is assumed to
                directly specify the metric matrix representation.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(neg_log_dens, grad_neg_log_dens)
        if metric is None:
            self.metric = matrices.IdentityMatrix()
        elif isinstance(metric, np.ndarray):
            if metric.ndim == 1:
                self.metric = matrices.PositiveDiagonalMatrix(metric)
            elif metric.ndim == 2:
                self.metric = matrices.DensePositiveDefiniteMatrix(metric)
            else:
                raise ValueError('If NumPy ndarray value is used for `metric`'
                                 ' must be either 1D (diagonal matrix) or 2D '
                                 '(dense positive definite matrix)')
        else:
            self.metric = metric

    @cache_in_state('mom')
    def h2(self, state):
        return 0.5 * state.mom @ self.dh2_dmom(state)

    @cache_in_state('mom')
    def dh2_dmom(self, state):
        return self.metric.inv @ state.mom

    def h2_flow(self, state, dt):
        """Apply exact flow map corresponding to `h2` Hamiltonian component.

        `state` argument is modified in place.

        Args:
            state (mici.states.ChainState): State to start flow at.
            dt (float): Time interval to simulate flow for.
        """
        state.pos += dt * self.dh2_dmom(state)

    def dh2_flow_dmom(self, dt):
        """Derivatives of `h2_flow` flow map with respect to input momentum.

        Args:
            dt (float): Time interval flow simulated for.

        Returns:
            dpos_dmom (mici.matrices.Matrix): Matrix representing derivative
                (Jacobian) of position output of `h2_flow` with respect to the
                value of the momentum component of the initial input state.
            dmom_dmom (mici.matrices.Matrix): Matrix representing derivative
                (Jacobian) of momentum output of `h2_flow` with respect to the
                value of the momentum component of the initial input state.
        """
        return (dt * self.metric.inv,
                matrices.IdentityMatrix(self.metric.shape[0]))

    def sample_momentum(self, state, rng):
        return self.metric.sqrt @ rng.standard_normal(state.pos.shape)


class GaussianEuclideanMetricSystem(EuclideanMetricSystem):
    r"""Euclidean Hamiltonian system with a tractable Gaussian component.

    Here Euclidean metric is defined to mean a metric with a fixed positive
    definite matrix representation \(M\). The momentum variables are taken to
    be independent of the position variables and with a zero-mean Gaussian
    marginal distribution with covariance specified by \(M\).

    Additionally the target distribution on the position variables is assumed
    to be defined by an unnormalized density with respect to the standard
    Gaussian measure on the position space (with identity covariance and zero
    mean), with the Hamiltonian component \(h_1\) corresponding to the negative
    logarithm of this density rather than the density with respect to the
    Lebesgue measure on the position space, i.e.

    \[ h_1(q) = \ell(q) - \frac{1}{2} q^T q \]

    where \(q\) is the position and \(\ell(q)\) is the negative log
    (unnormalized) density of the target distribution with respect to the
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
                unnormalized probability density on the position space with
                respect to the standard Gaussian measure on the position space,
                with the corresponding distribution on the position space being
                the target distribution it is wished to draw approximate
                samples from.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on position
                space and covariance of Gaussian marginal distribution on
                momentum vector. If `None` is passed (the default), the
                identity matrix will be used. If a 1D array is passed then this
                is assumed to specify a metric with positive diagonal matrix
                representation and the array the matrix diagonal. If a 2D array
                is passed then this is assumed to specify a metric with a dense
                positive definite matrix representation specified by the array.
                Otherwise if the value is a subclass of
                `mici.matrices.PositiveDefiniteMatrix` it is assumed to
                directly specify the metric matrix representation.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
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
        return (matrices.EigendecomposedSymmetricMatrix(
                    self.metric.eigvec, sin_omega_dt * omega),
                matrices.EigendecomposedSymmetricMatrix(
                    self.metric.eigvec, cos_omega_dt))


class ConstrainedEuclideanMetricSystem(EuclideanMetricSystem):
    r"""Base class for Euclidean Hamiltonian systems subject to constraints.

    The (constrained) position space is assumed to be a differentiable manifold
    embedded with a \(Q\)-dimensional ambient Euclidean space. The \(Q-C\)
    dimensional manifold \(\mathcal{M}\) is implicitly defined by an equation
    \(\mathcal{M} = \lbrace q \in \mathbb{R}^Q : c(q) = 0 \rbrace\) with
    \(c: \mathbb{R}^Q \to \mathbb{R}^C\) the *constraint function*.

    The ambient Euclidean space is assumed to be equipped with a metric with
    constant positive-definite matrix representation \(M\) which further
    specifies the covariance of the zero-mean Gaussian distribution
    \(\mathcal{N}(0, M)\) on the *unconstrained* momentum (co-)vector \(p\)
    with corresponding \(h_2\) Hamiltonian component defined as

    \[ h_2(q, p) = \frac{1}{2} p^T M^{-1} p. \]

    The time-derivative of the constraint equation implies a further set of
    constraints on the momentum \(q\) with \( \partial c(q) M^{-1} p = 0\)
    at all time points, corresponding to the momentum (velocity) being in the
    co-tangent space (tangent space) to the manifold.

    The target distribution is either assumed to be directly specified with
    unnormalized density \(\exp(-\ell(q))\) with respect to the Hausdorff
    measure on the manifold (under the metric induced from the ambient metric)
    with in this case the \(h_1\) Hamiltonian component then simply

    \[ h_1(q) = \ell(q), \]

    or alternatively it is assumed a prior distribution on the position \(q\)
    with density \(\exp(-\ell(q))\) with respect to the Lebesgue measure on
    the ambient space is specifed and the target distribution is the posterior
    distribution on \(q\) when conditioning on the event \(c(q) = 0\). The
    negative logarithm of the posterior distribution density with respect to
    the Hausdorff measure (and so \(h_1\) Hamiltonian component) is then

    \[
      h_1(q) =
      \ell(q) + \frac{1}{2} \log\left|\partial c(q)M^{-1}\partial c(q)^T\right|
    \]

    with an additional second *Gram matrix* determinant term to give the
    correct density with respect to the Hausdorff measure on the manifold.

    Due to the requirement to enforce the constraints on the position and
    momentum, a constraint-preserving numerical integrator needs to be used
    when simulating the Hamiltonian dynamic associated with the system, e.g.
    `mici.integrators.ConstrainedLeapfrogIntegrator`.

    References:

      1. Lelièvre, T., Rousset, M. and Stoltz, G., 2019. Hybrid Monte Carlo
         methods for sampling probability measures on submanifolds. Numerische
         Mathematik, 143(2), pp.379-421.
      2. Graham, M.M. and Storkey, A.J., 2017. Asymptotically exact inference
         in differentiable generative models. Electronic Journal of Statistics,
         11(2), pp.5105-5164.
    """

    def __init__(self, neg_log_dens, constr, metric=None,
                 dens_wrt_hausdorff=True, grad_neg_log_dens=None,
                 jacob_constr=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the constrained position
                space with respect to the Hausdorff measure on the constraint
                manifold (if `dens_wrt_hausdorff == True`) or alternatively the
                negative logarithm of an unnormalized probability density on
                the unconstrained (ambient) position space with respect to the
                Lebesgue measure. In the former case the target distribution it
                is wished to draw approximate samples from is assumed to be
                directly specified by the density function on the manifold. In
                the latter case the density function is instead taken to
                specify a prior distribution on the ambient space with the
                target distribution then corresponding to the posterior
                distribution when conditioning on the (zero Lebesgue measure)
                event `constr(pos) == 0`. This target posterior distribution
                has support on the differentiable manifold implicitly defined
                by the constraint equation, with density with respect to the
                Hausdorff measure on the manifold corresponding to the ratio of
                the prior density (specified by `neg_log_dens`) and the
                square-root of the determinant of the Gram matrix defined by

                    gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

                where `jacob_constr` is the Jacobian of the constraint function
                `constr` and `metric` is the matrix representation of the
                metric on the ambient space.
            constr (Callable[[array], array]): Function which given a position
                array return as a 1D array the value of the (vector-valued)
                constraint function, the zero level-set of which implicitly
                defines the manifold the dynamic is simulated on.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on
                *unconstrained* position space and covariance of Gaussian
                marginal distribution on *unconstrained* momentum vector. If
                `None` is passed (the default), the identity matrix will be
                used. If a 1D array is passed then this is assumed to specify a
                metric with positive diagonal matrix representation and the
                array the matrix diagonal. If a 2D array is passed then this is
                assumed to specify a metric with a dense positive definite
                matrix representation specified by the array. Otherwise if the
                value is a `mici.matrices.PositiveDefiniteMatrix` subclass it
                is assumed to directly specify the metric matrix
                representation.
            dens_wrt_hausdorff (bool): Whether the `neg_log_dens` function
                specifies the (negative logarithm) of the density of the target
                distribution with respect to the Hausdorff measure on the
                manifold directly (True) or alternatively the negative
                logarithm of a density of a prior distriubtion on the
                unconstrained (ambient) position space with respect to the
                Lebesgue measure, with the target distribution then
                corresponding to the posterior distribution when conditioning
                on the event `const(pos) == 0` (False). Note that in the former
                case the base Hausdorff measure on the manifold depends on the
                metric defined on the ambient space, with the Hausdorff measure
                being defined with respect to the metric induced on the
                manifold from this ambient metric.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the derivative (and value)
                of `neg_log_dens` automatically.
            jacob_constr (
                    None or Callable[[array], array or Tuple[array, array]]):
                Function which given a position array computes the Jacobian
                (matrix / 2D array of partial derivatives) of the output of the
                constraint function `c = constr(q)` with respect to the position
                array argument `q`, returning the computed Jacobian as a 2D
                array `jacob` with

                    jacob[i, j] = ∂c[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the Jacobian and
                the second being the value of `constr` evaluated at the passed
                position array. If `None` is passed (the default) an automatic
                differentiation fallback will be used to attempt to construct a
                function to compute the Jacobian (and value) of `constr`
                automatically.
        """
        super().__init__(neg_log_dens=neg_log_dens, metric=metric,
                         grad_neg_log_dens=grad_neg_log_dens)
        self._constr = constr
        self.dens_wrt_hausdorff = dens_wrt_hausdorff
        self._jacob_constr = autodiff_fallback(
            jacob_constr, constr, 'jacobian_and_value', 'jacob_constr')

    @cache_in_state('pos')
    def constr(self, state):
        """Constraint function at the current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `constr(state.pos)` as 1D array.
        """
        return self._constr(state.pos)

    @cache_in_state_with_aux('pos', 'constr')
    def jacob_constr(self, state):
        """Jacobian of constraint function at the current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of Jacobian of `constr(state.pos)` as 2D array.
        """
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
        """Gram matrix at current position.

        The Gram matrix as a position `q` is defined as

            gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

        where `jacob_constr` is the Jacobian of the constraint function
        `constr` and `metric` is the matrix representation of the metric on the
        ambient space.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mici.matrices.PositiveDefiniteMatrix: Gram matrix as matrix object.
        """
        return self.jacob_constr_inner_product(
            self.jacob_constr(state), self.metric.inv)

    def inv_gram(self, state):
        """Inverse of Gram matrix at current position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mici.matrices.PositiveDefiniteMatrix: Inverse of Gram matrix as
                matrix object.
        """
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
        """Project a momentum on to the co-tangent space at a position.

        Args:
            mom (array): Momentum (co-)vector as 1D array to project on to
                co-tangent space.
            state (mici.states.ChainState): State definining position on the
                manifold to project in to the co-tangent space of.

        Returns:
            array: Projected momentum in the co-tangent space at `state.pos`.
        """
        # Use parenthesis to force right-to-left evaluation to avoid
        # matrix-matrix products
        mom -= (self.jacob_constr(state).T @ (
                    self.inv_gram(state) @ (
                        self.jacob_constr(state) @ (self.metric.inv @ mom))))
        return mom

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
        mom = self.project_onto_cotangent_space(mom, state)
        return mom


class DenseConstrainedEuclideanMetricSystem(ConstrainedEuclideanMetricSystem):
    r"""Euclidean Hamiltonian system subject to a dense set of constraints.

    See `ConstrainedEuclideanMetricSystem` for more details about constrained
    systems.
    """

    def __init__(self, neg_log_dens, constr, metric=None,
                 dens_wrt_hausdorff=True, grad_neg_log_dens=None,
                 jacob_constr=None, mhp_constr=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the constrained position
                space with respect to the Hausdorff measure on the constraint
                manifold (if `dens_wrt_hausdorff == True`) or alternatively the
                negative logarithm of an unnormalized probability density on
                the unconstrained (ambient) position space with respect to the
                Lebesgue measure. In the former case the target distribution it
                is wished to draw approximate samples from is assumed to be
                directly specified by the density function on the manifold. In
                the latter case the density function is instead taken to
                specify a prior distribution on the ambient space with the
                target distribution then corresponding to the posterior
                distribution when conditioning on the (zero Lebesgue measure)
                event `constr(pos) == 0`. This target posterior distribution
                has support on the differentiable manifold implicitly defined
                by the constraint equation, with density with respect to the
                Hausdorff measure on the manifold corresponding to the ratio of
                the prior density (specified by `neg_log_dens`) and the
                square-root of the determinant of the Gram matrix defined by

                    gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

                where `jacob_constr` is the Jacobian of the constraint function
                `constr` and `metric` is the matrix representation of the
                metric on the ambient space.
            constr (Callable[[array], array]): Function which given a position
                array return as a 1D array the value of the (vector-valued)
                constraint function, the zero level-set of which implicitly
                defines the manifold the dynamic is simulated on.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on
                *unconstrained* position space and covariance of Gaussian
                marginal distribution on *unconstrained* momentum vector. If
                `None` is passed (the default), the identity matrix will be
                used. If a 1D array is passed then this is assumed to specify a
                metric with positive diagonal matrix representation and the
                array the matrix diagonal. If a 2D array is passed then this is
                assumed to specify a metric with a dense positive definite
                matrix representation specified by the array. Otherwise if the
                value is a `mici.matrices.PositiveDefiniteMatrix` subclass it
                is assumed to directly specify the metric matrix
                representation.
            dens_wrt_hausdorff (bool): Whether the `neg_log_dens` function
                specifies the (negative logarithm) of the density of the target
                distribution with respect to the Hausdorff measure on the
                manifold directly (True) or alternatively the negative
                logarithm of a density of a prior distriubtion on the
                unconstrained (ambient) position space with respect to the
                Lebesgue measure, with the target distribution then
                corresponding to the posterior distribution when conditioning
                on the event `const(pos) == 0` (False). Note that in the former
                case the base Hausdorff measure on the manifold depends on the
                metric defined on the ambient space, with the Hausdorff measure
                being defined with respect to the metric induced on the
                manifold from this ambient metric.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the derivative (and value)
                of `neg_log_dens` automatically.
            jacob_constr (
                    None or Callable[[array], array or Tuple[array, array]]):
                Function which given a position array computes the Jacobian
                (matrix / 2D array of partial derivatives) of the output of the
                constraint function `c = constr(q)` with respect to the
                position array argument `q`, returning the computed Jacobian as
                a 2D array `jacob` with

                    jacob[i, j] = ∂c[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the Jacobian
                and the second being the value of `constr` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the Jacobian (and value) of
                `neg_log_dens` automatically.
            mhp_constr (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable, array, array]]):
                Function which given a position array returns another function
                which takes a 2D array as an argument and returns the
                *matrix-Hessian-product* (MHP) of the constraint function
                `constr` with respect to the position array argument. The MHP
                is here defined as a function of a `(dim_constr, dim_pos)`
                shaped 2D array `m`

                    mhp(m) = sum(m[:, :, None] * hess[:, :, :], axis=(0, 1))

                where `hess` is the `(dim_constr, dim_pos, dim_pos)` shaped
                vector-Hessian of `c = constr(q)` with respect to `q` i.e. the
                array of second-order partial derivatives of such that

                    hess[i, j, k] = ∂²c[i] / (∂q[j] ∂q[k])

                Optionally the function may instead return a 3-tuple of values
                with the first a function to compute a MHP of `constr`, the
                second a 2D array corresponding to the Jacobian of `constr`,
                and the third the value of `constr`, all evaluated at the
                passed position array. If `None` is passed (the default) an
                automatic differentiation fallback will be used to attempt to
                construct a function which calculates the MHP (and Jacobian and
                value) of `constr` automatically.
        """
        super().__init__(neg_log_dens=neg_log_dens, constr=constr,
                         metric=metric, dens_wrt_hausdorff=dens_wrt_hausdorff,
                         grad_neg_log_dens=grad_neg_log_dens,
                         jacob_constr=jacob_constr)
        if not dens_wrt_hausdorff:
            self._mhp_constr = autodiff_fallback(
                mhp_constr, constr, 'mhp_jacobian_and_value', 'mhp_constr')

    @cache_in_state_with_aux('pos', ('jacob_constr', 'constr'))
    def mhp_constr(self, state):
        return self._mhp_constr(state.pos)

    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None):
        if jacob_constr_2 is None or jacob_constr_2 is jacob_constr_1:
            return matrices.DensePositiveDefiniteMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_1.T))
        else:
            return matrices.DenseSquareMatrix(
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
    r"""Gaussian Euclidean Hamiltonian system st. a dense set of constraints.

    See `ConstrainedEuclideanMetricSystem` for more details about constrained
    systems and `GaussianEuclideanMetricSystem` for Gaussian Euclidean metric
    systems.
    """

    def __init__(self, neg_log_dens, constr, metric=None,
                 grad_neg_log_dens=None, jacob_constr=None, mhp_constr=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the unconstrained (ambient)
                position space with respect to the standard Gaussian measure.
                The density function is taken to specify a prior distribution
                on the ambient space with the target distribution then
                corresponding to the posterior distribution when conditioning
                on the (zero Lebesgue measure) event `constr(pos) == 0`. This
                target posterior distribution has support on the differentiable
                manifold implicitly defined by the constraint equation, with
                density with respect to the Hausdorff measure on the manifold
                corresponding to the ratio of the prior density (specified by
                `neg_log_dens`) and the square-root of the determinant of the
                Gram matrix defined by

                    gram(q) = jacob_constr(q) @ inv(metric) @ jacob_constr(q).T

                where `jacob_constr` is the Jacobian of the constraint function
                `constr` and `metric` is the matrix representation of the
                metric on the ambient space.
            constr (Callable[[array], array]): Function which given a position
                array return as a 1D array the value of the (vector-valued)
                constraint function, the zero level-set of which implicitly
                defines the manifold the dynamic is simulated on.
            metric (None or array or PositiveDefiniteMatrix): Matrix object
                corresponding to matrix representation of metric on
                *unconstrained* position space and covariance of Gaussian
                marginal distribution on *unconstrained* momentum vector. If
                `None` is passed (the default), the identity matrix will be
                used. If a 1D array is passed then this is assumed to specify a
                metric with positive diagonal matrix representation and the
                array the matrix diagonal. If a 2D array is passed then this is
                assumed to specify a metric with a dense positive definite
                matrix representation specified by the array. Otherwise if
                a subclass of `mici.matrices.PositiveDefiniteMatrix` it is
                assumed to directly specify the metric matrix representation.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the derivative (and value)
                of `neg_log_dens` automatically.
            jacob_constr (
                    None or Callable[[array], array or Tuple[array, array]]):
                Function which given a position array computes the Jacobian
                (matrix / 2D array of partial derivatives) of the output of the
                constraint function `c = constr(q)` with respect to the
                position array argument `q`, returning the computed Jacobian as
                a 2D array `jacob` with

                    jacob[i, j] = ∂c[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the Jacobian
                and the second being the value of `constr` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct a function to compute the Jacobian (and value) of
                `neg_log_dens` automatically.
            mhp_constr (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable, array, array]]):
                Function which given a position array returns another function
                which takes a 2D array as an argument and returns the
                *matrix-Hessian-product* (MHP) of the constraint function
                `constr` with respect to the position array argument. The MHP
                is here defined as a function of a `(dim_constr, dim_pos)`
                shaped 2D array `m`

                    mhp(m) = sum(m[:, :, None] * hess[:, :, :], axis=(0, 1))

                where `hess` is the `(dim_constr, dim_pos, dim_pos)` shaped
                vector-Hessian of `c = constr(q)` with respect to `q` i.e. the
                array of second-order partial derivatives of such that

                    hess[i, j, k] = ∂²c[i] / (∂q[j] ∂q[k])

                Optionally the function may instead return a 3-tuple of values
                with the first a function to compute a MHP of `constr`, the
                second a 2D array corresponding to the Jacobian of `constr`,
                and the third the value of `constr`, all evaluated at the
                passed position array. If `None` is passed (the default) an
                automatic differentiation fallback will be used to attempt to
                construct a function which calculates the MHP (and Jacobian and
                value) of `constr` automatically.
        """
        DenseConstrainedEuclideanMetricSystem.__init__(
            self, neg_log_dens=neg_log_dens, constr=constr, metric=metric,
            dens_wrt_hausdorff=False,
            grad_neg_log_dens=grad_neg_log_dens, jacob_constr=jacob_constr,
            mhp_constr=mhp_constr)

    def jacob_constr_inner_product(
            self, jacob_constr_1, inner_product_matrix, jacob_constr_2=None):
        if jacob_constr_2 is None or jacob_constr_2 is jacob_constr_1:
            return matrices.DenseSymmetricMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_1.T))
        else:
            return matrices.DenseSquareMatrix(
                jacob_constr_1 @ (inner_product_matrix @ jacob_constr_2.T))


class RiemannianMetricSystem(System):
    r"""Riemannian Hamiltonian system with a position-dependent metric.

    This class allows for metric matrix representations of any generic type.
    In most cases a specialized subclass such as `DenseRiemannianMetricSystem`,
    `CholeskyFactoredRiemannianMetricSystem`, `DiagonalRiemannianMetricSystem`,
    `ScalarRiemannianMetricSystem` or `SoftAbsRiemannianMetricSystem` will
    provide a simpler method of constructng a system with a metric matrix
    representation of a specific type.

    The position space is assumed to be a Riemannian manifold with a metric
    with position-dependent positive definite matrix-representation \(M(q)\)
    where \(q\) is a position vector. The momentum \(p\) is then taken to have
    a zero-mean Gaussian conditional distribution given the position \(q\),
    with covariance \(M(q)\), i.e. \(p \sim \mathcal{N}(0, M(q))\) [1].

    The \(h_1\) Hamiltonian component is then

    \[ h_1(q) = \ell(q) + \frac{1}{2}\log\left|M(q)\right| \]

    where \(\ell(q)\) is the negative log (unnormalized) density of the target
    distribution with respect to the Lebesgue measure at \(q\). The \(h_2\)
    Hamiltonian component is

    \[ h_2(q, p) = \frac{1}{2} p^T (M(q))^{-1} p. \]

    Due to the coupling between the position and momentum variables in \(h_2\),
    the Hamiltonian system is non-separable, requiring use of a numerical
    integrator with implicit steps when simulating the Hamiltonian dynamic
    associated with the system, e.g.
    `mici.integrators.ImplicitLeapfrogIntegrator`.

    References:

      1. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
         Hamiltonian Monte Varlo methods. Journal of the Royal Statistical
         Society: Series B (Statistical Methodology), 73(2), pp.123-214.
    """

    def __init__(self, neg_log_dens, metric_matrix_class, metric_func,
                 vjp_metric_func=None, grad_neg_log_dens=None,
                 metric_kwargs=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric_matrix_class (type[PositiveDefiniteMatrix]): Class (or
                factory function returning an instance of the class) which
                defines type of matrix representation of metric. The class
                initializer should take a single positional argument which will
                be passed the array outputted by `metric_func`, and which is
                assumed to be a parameter which fully defines the resulting
                matrix (e.g. the diagonal of a `mici.matrices.DiagonalMatrix`).
                The class initializer may also optionally take one or more
                keyword arguments, with the `metric_kwargs` argument used to
                specify the value of these, if any. Together this means the
                metric matrix representation at a position `pos` is constructed
                as

                    metric = metric_matrix_class(
                        metric_func(pos), **metric_kwargs)

                The `mici.matrices.PositiveDefiniteMatrix` subclass should as a
                minimum define `inv`, `log_abs_det`, `grad_log_abs_det`,
                `grad_quadratic_form_inv`, `__matmul__` and `__rmatmul__`
                methods / properties (see documentation of
                `mici.matrices.PositiveDefiniteMatrix` and
                `mici.matrices.DifferentiableMatrix` for definitions of the
                expected behaviour of these methods).
            metric_func (Callable[[array], array]): Function which given a
                position array returns an array containing the parameter value
                of the metric matrix representation passed as the single
                positional argument to the `metric_matrix_class` initializer.
            vjp_metric_func (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable[[array], array], array]]):
                Function which given a position array returns another function
                which takes an array as an argument and returns the
                *vector-Jacobian-product* (VJP) of `metric_func` with respect
                to the position array argument. The VJP is here defined as a
                function of an array `v` (of the same shape as the output of
                `metric_func`) corresponding to

                    vjp(v) = sum(v[..., None] * jacob, tuple(range(v.ndim))

                where `jacob` is the Jacobian of `m = metric_func(q)` wrt `q`
                i.e. the array of partial derivatives of the function such that

                    jacob[..., i] = ∂m[...] / ∂q[i]

                Optionally the function may instead return a 2-tuple of values
                with the first a function to compute a VJP of `metric_func` and
                the second an array containing the value of `metric_func`, both
                evaluated at the passed position array. If `None` is passed
                (the default) an automatic differentiation fallback will be
                used to attempt to construct a function which calculates the
                VJP (and value) of `metric_func` automatically.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
            metric_kwargs (None or Dict[str, object]): An optional dictionary
                of any additional keyword arguments to the initializer of
                `metric_matrix_class`.
        """
        self._metric_matrix_class = metric_matrix_class
        self._metric_func = metric_func
        self._vjp_metric_func = autodiff_fallback(
            vjp_metric_func, metric_func, 'vjp_and_value', 'vjp_metric_func')
        self._metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        super().__init__(neg_log_dens, grad_neg_log_dens)

    @cache_in_state('pos')
    def metric_func(self, state):
        """
        Function computing the parameter of the metric matrix representation.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            array: Value of `metric_func(state.pos)`.
        """
        return self._metric_func(state.pos)

    @cache_in_state_with_aux('pos', 'metric_func')
    def vjp_metric_func(self, state):
        """
        Function constructing a vector-Jacobian-product for `metric_func`.

        The vector-Jacobian-product is here defined as a function of an array
        `v` (of the same shape as the output of `metric_func`) corresponding to

            vjp(v) = sum(v[..., None] * jacob, axis=tuple(range(v.ndim))

        where `jacob` is the Jacobian of `m = metric_func(q)` wrt `q` i.e.
        the array of partial derivatives of the function such that

            jacob[..., i] = ∂m[...] / ∂q[i]

        Args:
            state (mici.states.ChainState): State to compute VJP at.

        Returns:
            Callable[[array], array]: Vector-Jacobian-product function.
        """
        return self._vjp_metric_func(state.pos)

    @cache_in_state('pos')
    def metric(self, state):
        """
        Function computing the metric matrix representation.

        The returned type of this function is that specified by the
        `metric_matrix_class` argument to the initializer.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mici.matrices.PositiveDefiniteMatrix: Metric matrix representation.
        """
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
    """Riemannian-metric system with scaled identity matrix representation.

    Hamiltonian system with a position dependent scaled identity metric matrix
    representation which is specified by a scalar function
    `metric_scalar_function` of the position `q` which outputs a strictly
    positive scalar `s = metric_scalar_func(q)` with the the metric matrix
    representation then taken to be `s * identity(q.shape[0])`.

    See documentation of `RiemannianMetricSystem` for more general details
    about Riemannian-metric Hamiltonian systems.
    """

    def __init__(self, neg_log_dens, metric_scalar_func,
                 vjp_metric_scalar_func=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric_scalar_func (Callable[[array], float]): Function which
                given a position array returns a strictly positive scalar
                corresponding to the parameter value of the scaled identity
                metric matrix representation.
            vjp_metric_scalar_func (None or
                    Callable[[array], Callable[[array], float]] or
                    Callable[[array], Tuple[Callable[[array, float]], float]]):
                Function which given a position array returns another function
                which takes a scalar as an argument and returns the
                *vector-Jacobian-product* (VJP) of `metric_scalar_func` with
                respect to the position array argument. The VJP is here defined
                as a function of a scalar `v`

                    vjp(v) = v * grad

                where `grad` is the `(dim_pos,)` shaped Jacobian (gradient) of
                `s = metric_scalar_func(q)` with respect to `q` i.e. the array
                of partial derivatives of the function such that

                    grad[i] = ∂s / ∂q[i]

                Optionally the function may instead return a 2-tuple of values
                with the first a function to compute a VJP of
                `metric_scalar_func` and the second a float containing the
                value of `metric_scalar_func`, both evaluated at the passed
                position array. If `None` is passed (the default) an automatic
                differentiation fallback will be used to attempt to construct a
                function which calculates the VJP (and value) of
                `metric_scalar_func` automatically.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """

        super().__init__(
            neg_log_dens, matrices.PositiveScaledIdentityMatrix,
            metric_scalar_func, vjp_metric_scalar_func, grad_neg_log_dens)

    @cache_in_state('pos')
    def metric(self, state):
        return self._metric_matrix_class(
            self.metric_func(state), size=state.pos.shape[0])


class DiagonalRiemannianMetricSystem(RiemannianMetricSystem):
    """Riemannian-metric system with diagonal matrix representation.

    Hamiltonian system with a position dependent diagonal metric matrix
    representation which is specified by a vector-valued function
    `metric_diagonal_func` of the position `q` which outputs a 1D array with
    strictly positive elements `d = metric_diagonal_func(q)` with the metric
    matrix representation then taken to be `diag(d)`.

    See documentation of `RiemannianMetricSystem` for more general details
    about Riemannian-metric Hamiltonian systems.
    """

    def __init__(self, neg_log_dens, metric_diagonal_func,
                 vjp_metric_diagonal_func=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric_diagonal_func (Callable[[array], array]): Function which
                given a position array returns a 1D array with strictly
                positive values corresponding to the diagonal values
                (left-to-right) of the diagonal metric matrix representation.
            vjp_metric_diagonal_func (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable[[array], array], array]]):
                Function which given a position array returns another function
                which takes a 1D array as an argument and returns the
                *vector-Jacobian-product* (VJP) of `metric_diagonal_func` with
                respect to the position array argument. The VJP is here defined
                as a function of a 1D array `v`

                    vjp(v) = sum(v[:, None] * jacob[:, :], axis=0)

                where `jacob` is the `(dim_pos, dim_pos)` shaped Jacobian of
                `d = metric_diagonal_func(q)` with respect to `q` i.e. the
                array of partial derivatives of the function such that

                    jacob[i, j] = ∂d[i] / ∂q[j]

                Optionally the function may instead return a 2-tuple of values
                with the first a function to compute a VJP of
                `metric_diagonal_func` and the second a 1D array containing the
                value of `metric_diagonal_func`, both evaluated at the passed
                position array. If `None` is passed (the default) an automatic
                differentiation fallback will be used to attempt to construct a
                function which calculates the VJP (and value) of
                `metric_diagonal_func` automatically.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(
            neg_log_dens, matrices.PositiveDiagonalMatrix, metric_diagonal_func,
            vjp_metric_diagonal_func, grad_neg_log_dens)


class CholeskyFactoredRiemannianMetricSystem(RiemannianMetricSystem):
    """Riemannian-metric system with Cholesky-factored matrix representation.

    Hamiltonian system with a position dependent metric matrix representation
    which is specified by its Cholesky factor by a matrix function
    `metric_chol_func` of the position `q` which outputs a lower-triangular
    matrix `L = metric_chol_func(q)` with the metric matrix representation then
    taken to be `L @ L.T`.

    See documentation of `RiemannianMetricSystem` for more general details
    about Riemannian-metric Hamiltonian systems.
    """

    def __init__(self, neg_log_dens, metric_chol_func,
                 vjp_metric_chol_func=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric_chol_func (Callable[[array], array]): Function which given
                a position array returns a 2D array with zeros above the
                diagonal corresponding to the lower-triangular Cholesky-factor
                of the positive definite metric matrix representation.
            vjp_metric_chol_func (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable[[array], array], array]]):
                Function which given a position array returns another function
                which takes a lower-triangular 2D array as an argument (any
                values in the array above the diagonal are ignored) and returns
                the *vector-Jacobian-product* (VJP) of `metric_chol_func` with
                respect to the position array argument. The VJP is here defined
                as a function of a 2D array `v`

                    vjp(v) = sum(v[:, :, None] * jacob[:, :, :], axis=(0, 1))

                where `jacob` is the `(dim_pos, dim_pos, dim_pos)` shaped
                Jacobian of `L = metric_chol_func(q)` with respect to `q` i.e.
                the array of partial derivatives of the function such that

                    jacob[i, j, k] = ∂L[i, j] / ∂q[k]

                Optionally the function may instead return a 2-tuple of values
                with the first a function to compute a VJP of
                `metric_chol_func` and the second a 2D array containing the
                value of `metric_chol_func`, both evaluated at the passed
                position array. If `None` is passed (the default) an automatic
                differentiation fallback will be used to attempt to construct a
                function which calculates the VJP (and value) of
                `metric_chol_func` automatically.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(
            neg_log_dens, matrices.TriangularFactoredPositiveDefiniteMatrix,
            metric_chol_func, vjp_metric_chol_func, grad_neg_log_dens,
            metric_kwargs={'factor_is_lower': True})


class DenseRiemannianMetricSystem(RiemannianMetricSystem):
    """Riemannian-metric system with dense matrix representation.

    Hamiltonian system with a position dependent metric matrix representation
    which is specified to be a dense matrix function `metric_func` of the
    position `q` which is guaranteed to be positive definite almost-everywhere,
    with `M = metric_func(q)` then the metric matrix representation.

    See documentation of `RiemannianMetricSystem` for more general details
    about Riemannian-metric Hamiltonian systems.
    """

    def __init__(self, neg_log_dens, metric_func,
                 vjp_metric_func=None, grad_neg_log_dens=None):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            metric_func (Callable[[array], array]): Function which given a
                position array returns a 2D array corresponding to the positive
                definite metric matrix representation. The returned matrices
                (2D arrays) are assumed to be positive-definite for all input
                positions and a `LinAlgError` exception may be raised if this
                fails to be the case.
            vjp_metric_func (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable[[array], array], array]]):
                Function which given a position array returns another function
                which takes a 2D array as an argument and returns the
                *vector-Jacobian-product* (VJP) of `metric_func` with respect
                to the position array argument. The VJP is here defined as a
                function of a 2D array `v`

                    vjp(v) = sum(v[:, :, None] * jacob[:, :, :], axis=(0, 1))

                where `jacob` is the `(dim_pos, dim_pos, dim_pos)` shaped
                Jacobian of `M = metric_func(q)` with respect to `q` i.e. the
                array of partial derivatives of the function such that

                    jacob[i, j, k] = ∂M[i, j] / ∂q[k]

                Optionally the function may instead return a 2-tuple of values
                with the first a function to compute a VJP of `metric_func` and
                the second a 2D array containing the value of `metric_func`,
                both evaluated at the passed position array. If `None` is
                passed (the default) an automatic differentiation fallback will
                be used to attempt to construct a function which calculates the
                VJP (and value) of `metric_func` automatically.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
        """
        super().__init__(
            neg_log_dens, matrices.DensePositiveDefiniteMatrix, metric_func,
            vjp_metric_func, grad_neg_log_dens)


class SoftAbsRiemannianMetricSystem(RiemannianMetricSystem):
    """SoftAbs Riemmanian metric Hamiltonian system.

    Hamiltonian system with a position dependent metric matrix representation
    which is specified to be a dense matrix function `metric_func` of the
    position `q` which is guaranteed to be positive definite almost-everywhere,
    with `M = metric_func(q)` then the metric matrix representation.

    Hamiltonian system with a position dependent metric matrix representation
    which is specified to be an eigenvalue-regularized transformation of the
    Hessian of the negative log density function (the symmetric matrix of
    second derivatives the negative log density function with respect to the
    position array components. Specifically if `hess_neg_log_dens` is a
    symmetric 2D square array valued function of the position `q`, with
    `H = hess_neg_log_dens(q)` then if an eigenvalue decomposition of `H` is
    computed, i.e. `eigval, eigvec = eigh(H)`, with `eigval` a 1D array of
    real eigenvalues, and `eigvec` the corresponding 2D array (orthogonal
    matrix) with eigenvectors as columns, then the resulting positive-definite
    metric matrix representation `M` is computed as

        M = eigvec @ diag(softabs(eigval, softabs_coeff)) @ eigvec.T

    with `softabs(x, softabs_coeff) = x / tanh(x * softabs_coeff)` an
    elementwise function which acts as a smooth approximation to the absolute
    function (ensuring all the eigenvalues of `M` are strictly positive) with
    the additional scalar parameter `softabs_coeff` controlling the smoothness
    of the approximation, with `softabs` tending to the piecewise linear `abs`
    function as `softabs_coeff` tends to infinity, and becoming increasingly
    smooth as `softabs_coeff` tends to zero.

    See documentation of `RiemannianMetricSystem` for more general details
    about Riemannian-metric Hamiltonian systems.

    References:

      1. Betancourt, M., 2013. A general metric for Riemannian manifold
         Hamiltonian Monte Carlo. In Geometric science of information
         (pp. 327-334).
    """

    def __init__(self, neg_log_dens, grad_neg_log_dens=None,
                 hess_neg_log_dens=None, mtp_neg_log_dens=None,
                 softabs_coeff=1.):
        """
        Args:
            neg_log_dens (Callable[[array], float]): Function which given a
                position array returns the negative logarithm of an
                unnormalized probability density on the position space with
                respect to the Lebesgue measure, with the corresponding
                distribution on the position space being the target
                distribution it is wished to draw approximate samples from.
            grad_neg_log_dens (
                    None or Callable[[array], array or Tuple[array, float]]):
                Function which given a position array returns the derivative of
                `neg_log_dens` with respect to the position array argument.
                Optionally the function may instead return a 2-tuple of values
                with the first being the array corresponding to the derivative
                and the second being the value of the `neg_log_dens` evaluated
                at the passed position array. If `None` is passed (the default)
                an automatic differentiation fallback will be used to attempt
                to construct the derivative of `neg_log_dens` automatically.
            hess_neg_log_dens (None or
                    Callable[[array], array or Tuple[array, array, float]]):
                Function which given a position array returns the Hessian of
                `neg_log_dens` with respect to the position array argument as a
                2D array. Optionally the function may instead return a 3-tuple
                of values with the first a 2D array containting the Hessian of
                `neg_log_dens`, the second a 1D array containing the gradient
                of `neg_log_dens` and the third the value of `neg_log_dens`,
                all evaluated at the passed position array. If `None` is passed
                (the default) an automatic differentiation fallback will be
                used to attempt to construct a function which calculates the
                Hessian (and gradient and value) of `neg_log_dens`
                automatically.
            mtp_neg_log_dens (None or
                    Callable[[array], Callable[[array], array]] or
                    Callable[[array], Tuple[Callable, array, array, float]]):
                Function which given a position array returns another function
                which takes a 2D array (matrix) as an argument and returns the
                *matrix-Tressian-product* (MTP) of `neg_log_dens` with respect
                to the position array argument. The MTP is here defined as a
                function of a matrix `m` corresponding to

                    mtp(m) = sum(m[:, :, None] * tress[:, :, :], axis=(0, 1))

                where `tress` is the 'Tressian' of `f = neg_log_dens(q)` wrt
                `q` i.e. the 3D array of third-order partial derivatives of the
                scalar-valued function such that

                    tress[i, j, k] = ∂³f / (∂q[i] ∂q[j] ∂q[k])

                Optionally the function may instead return a 4-tuple of values
                with the first a function to compute a MTP of `neg_log_dens`,
                the second a 2D array containing the Hessian of `neg_log_dens`,
                the third a 1D array containing the gradient of `neg_log_dens`
                and the fourth the value of `neg_log_dens`, all evaluated at
                the passed position array. If `None` is passed (the default) an
                automatic differentiation fallback will be used to attempt to
                construct a function which calculates the MTP (and Hesisan and
                gradient and value) of `neg_log_dens` automatically.
            softabs_coeff (float): Positive regularisation coefficient for
                smooth approximation to absolute value used to regularize
                Hessian eigenvalues in metric matrix representation. As the
                value tends to infinity the approximation becomes increasingly
                close to the absolute function.
        """
        self._hess_neg_log_dens = autodiff_fallback(
            hess_neg_log_dens, neg_log_dens, 'hessian_grad_and_value',
            'neg_log_dens')
        self._mtp_neg_log_dens = autodiff_fallback(
            mtp_neg_log_dens, neg_log_dens, 'mtp_hessian_grad_and_value',
            'mtp_neg_log_dens')
        super().__init__(neg_log_dens,
                         matrices.SoftAbsRegularizedPositiveDefiniteMatrix,
                         self._hess_neg_log_dens, self._mtp_neg_log_dens,
                         grad_neg_log_dens,
                         metric_kwargs={'softabs_coeff': softabs_coeff})

    def metric_func(self, state):
        return self.hess_neg_log_dens(state)

    def vjp_metric_func(self, state):
        return self.mtp_neg_log_dens(state)

    @cache_in_state_with_aux('pos', ('grad_neg_log_dens', 'neg_log_dens'))
    def hess_neg_log_dens(self, state):
        """Hessian of negative log density with respect to position.

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            hessian (array): 2D array of `neg_log_dens(state)` second
                derivatives with respect to `state.pos`, with `hessian[i, j]`
                the second derivative of `neg_log_dens(state)` with respect to
                `state.pos[i]` and `state.pos[j]`.
        """
        return self._hess_neg_log_dens(state.pos)

    @cache_in_state_with_aux(
        'pos', ('hess_neg_log_dens', 'grad_neg_log_dens', 'neg_log_dens'))
    def mtp_neg_log_dens(self, state):
        """Generate MTP of negative log density with respect to position.

        The matrix-Tressian-product (MTP) is here defined as a function of a
        matrix `m` corresponding to

            mtp(m) = sum(m[:, :, None] * tress[:, :, :], axis=(0, 1))

        where `tress` is the 'Tressian' of `f = neg_log_dens(q)` with respect
        to `q = state.pos` i.e. the 3D array of third-order partial derivatives
        of the scalar-valued function such that

            tress[i, j, k] = ∂³f / (∂q[i] ∂q[j] ∂q[k])

        Args:
            state (mici.states.ChainState): State to compute value at.

        Returns:
            mtp (Callable[[array], array]): Function which accepts a 2D array
                of shape `(state.pos.shape[0], state.pos.shape[0])` as an
                argument and returns an array of shape `state.pos.shape`
                containing the computed MTP value.
        """
        return self._mtp_neg_log_dens(state.pos)
