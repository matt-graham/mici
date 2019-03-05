"""Hamiltonian systems encapsulating energy functions and their derivatives."""

import logging
import numpy as np
import scipy.linalg as sla
from hmc.states import cache_in_state, multi_cache_in_state
from hmc.metrics import IsotropicEuclideanMetric, SoftAbsRiemannianMetric
from hmc.autodiff import autodiff_fallback


class HamiltonianSystem(object):
    """Base class for Hamiltonian systems."""

    def __init__(self, pot_energy, grad_pot_energy=None):
        self._pot_energy = pot_energy
        self._grad_pot_energy = autodiff_fallback(
            grad_pot_energy, pot_energy, 'grad_and_value', 'grad_pot_energy')

    @cache_in_state('pos')
    def pot_energy(self, state):
        return self._pot_energy(state.pos)

    @multi_cache_in_state(['pos'], ['grad_pot_energy', 'pot_energy'])
    def grad_pot_energy(self, state):
        return self._grad_pot_energy(state.pos)

    def h(self, state):
        raise NotImplementedError()

    def dh_dpos(self, state):
        raise NotImplementedError()

    def dh_dmom(self, state):
        raise NotImplementedError()

    def sample_momentum(self, state, rng):
        raise NotImplementedError()


class SeparableHamiltonianSystem(HamiltonianSystem):
    """Base class for separable Hamiltonian systems.

    Here separable means that the Hamiltonian can be expressed as the sum of
    a term depending only on the position (target) variables, here denoted
    the potential energy, and a second term depending only on the momentum
    variables, here denoted the kinetic energy.
    """

    @cache_in_state('mom')
    def kin_energy(self, state):
        return self._kin_energy(state.mom)

    @cache_in_state('mom')
    def grad_kin_energy(self, state):
        return self._grad_kin_energy(state.mom)

    def h(self, state):
        return self.pot_energy(state) + self.kin_energy(state)

    def dh_dpos(self, state):
        return self.grad_pot_energy(state)

    def dh_dmom(self, state):
        return self.grad_kin_energy(state)

    def _kin_energy(self, mom):
        raise NotImplementedError()

    def _grad_kin_energy(self, mom):
        raise NotImplementedError()

    def sample_momentum(self, state, rng):
        raise NotImplementedError()


class EuclideanMetricSystem(SeparableHamiltonianSystem):
    """Euclidean Hamiltonian system with a fixed metric tensor.

    The momentum variables are taken to be independent of the position
    variables and with a zero-mean Gaussian marginal distribution with
    covariance specified by a fixed positive-definite matrix (metric tensor).
    """

    def __init__(self, pot_energy, metric=None, grad_pot_energy=None):
        self.metric = IsotropicEuclideanMetric() if metric is None else metric
        super().__init__(pot_energy, grad_pot_energy)

    def _kin_energy(self, mom):
        return 0.5 * self.metric.quadratic_form_inv(mom)

    def _grad_kin_energy(self, mom):
        return self.metric.lmult_inv(mom)

    def sample_momentum(self, state, rng):
        return self.metric.lmult_sqrt(rng.normal(size=state.pos.shape))


class IsotropicGaussianSplitSystem(SeparableHamiltonianSystem):
    """Separable Hamiltonian system with isotropic Gaussian component.

    The kinetic energy is assumed to correspond to an isotropic covariance
    Gaussian distribution and the potential energy is assumed to consist of
    the sum of an isotropic Gaussian component and further non-Gaussian term.
    Specifically the Hamiltonian function is assumed to have the form

         h(pos, mom) = h1(pos) + h2(pos, mom)
                     = h1(pos) + 0.5 * sum(pos**2) + 0.5 * sum(mom**2)

    In this case the Hamiltonian flow due to the `h2` component can be solved
    for analytically, allowing use of a split integration scheme [1].

    References:

      1. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014. Split
         Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
    """

    def __init__(self, h1, grad_h1=None):
        self._h1 = h1
        self._grad_h1 = autodiff_fallback(
            grad_h1, h1, 'grad_and_value', 'grad_h1')

    def pot_energy(self, state):
        return self.h1(state) + 0.5 * np.sum(state.pos**2)

    def grad_pot_energy(self, state):
        return self.dh1_dpos(state) + state.pos

    def _kin_energy(self, mom):
        return 0.5 * np.sum(mom**2)

    def _grad_kin_energy(self, mom):
        return mom

    @cache_in_state('pos')
    def h1(self, state):
        return self._h1(state.pos)

    @multi_cache_in_state(['pos'], ['grad_h1', 'h1'])
    def dh1_dpos(self, state):
        return self._grad_h1(state.pos)

    def h2(self, state):
        return 0.5 * np.sum(state.pos**2) + 0.5 * np.sum(state.mom**2)

    def h2_exact_flow(self, state, dt):
        pos_0, mom_0 = state.pos.copy(), state.mom.copy()
        sin_dt, cos_dt = np.sin(dt), np.cos(dt)
        state.pos *= cos_dt
        state.pos += sin_dt * mom_0
        state.mom *= cos_dt
        state.mom -= sin_dt * pos_0
        return state

    def sample_momentum(self, state, rng):
        return rng.normal(size=state.pos.shape)


class BaseEuclideanMetricConstrainedSystem(EuclideanMetricSystem):
    """Base class for Euclidean Hamiltonian systems subject to constraints.

    The system is assumed to be subject to a set of holonomic constraints on
    the position component of the state. These constraints are specified by a
    vector constraint function which takes as argument the position component,
    and which is equal to zero in all components when the constraints are
    satisfied. The constraint function implicitly defines a manifold embedded
    in the position space of constraint satisfying configurations. There are
    also implcitly a set of constraints on the momentum component of the state
    due to the requirment that velocity (momentum pre-multiplied by inverse
    metric) is always tangential to the constraint manifold.
    """

    def constr(self, state):
        raise NotImplementedError()

    def jacob_constr(self, state):
        raise NotImplementedError()

    @cache_in_state('pos')
    def inv_metric_jacob_constr_t(self, state):
        jacob_constr = self.jacob_constr(state)
        return self.metric.lmult_inv(jacob_constr.T)

    @cache_in_state('pos')
    def chol_gram(self, state):
        jacob_constr = self.jacob_constr(state)
        gram = jacob_constr @ self.inv_metric_jacob_constr_t(state)
        return sla.cholesky(gram, lower=True)

    def project_onto_cotangent_space(self, mom, state):
        jacob_constr = self.jacob_constr(state)
        chol_gram = self.chol_gram(state)
        mom -= jacob_constr.T @ sla.cho_solve(
            (chol_gram, True), jacob_constr @ self.metric.lmult_inv(mom))

    def solve_dh_dmom_for_mom(self, dpos_dt):
        return self.metric.lmult(dpos_dt)

    def sample_momentum(self, state, rng):
        mom = super().sample_momentum(state, rng)
        self.project_onto_cotangent_space(mom, state)
        return mom


class EuclideanMetricConstrainedSystem(BaseEuclideanMetricConstrainedSystem):
    """Euclidean metric Hamiltonian system subject to constraints.

    The potential energy here represents the negative log density of the
    invariant distribution on the position component of the state with respect
    to the Hausdorff (area) measure on the constraint manifold. In this class
    the potential energy is assumed to be specified directly with any Jacobian
    determinant factor already included in the function definition.
    """

    def __init__(self, pot_energy, constr, metric=None,
                 grad_pot_energy=None, jacob_constr=None):
        super().__init__(pot_energy=pot_energy, metric=metric,
                         grad_pot_energy=grad_pot_energy)
        self._constr = constr
        self._jacob_constr = autodiff_fallback(
            jacob_constr, constr, 'jacobian_and_value', 'jacob_constr')

    @cache_in_state('pos')
    def constr(self, state):
        return self._constr(state.pos)

    @multi_cache_in_state(['pos'], ['jacob_constr', 'constr'])
    def jacob_constr(self, state):
        return self._jacob_constr(state.pos)


class EuclideanMetricObservedGeneratorSystem(
        BaseEuclideanMetricConstrainedSystem):
    """Hamiltonian system for generative models with observed outputs.

    Here the constraints are assumed to arise from requiring the output of a
    generative model to be equal to known observed values. A generative model
    is here considered to be a function mapping from a vector of inputs with
    a distribution with known density with respect to the Lebesgue measure on
    the input space, to a vector of generated outputs, with the dimension of
    the output space being strictly less than that of the input space. The
    function is termed the generator, and the density on the inputs the input
    density. The posterior distribution on the inputs given observed outputs
    has support on an embedded manifold with an unnnormalised density with
    respect to the Hausdorff measure on the manifold which is equal to the
    product of the input density and the reciprocal of the (generalised)
    Jacobian determinant of the generator function [1].

    References:

      1. Graham, M.M. and Storkey, A.J., 2017a. Asymptotically exact inference
         in differentiable generative models. Electronic Journal of Statistics,
         11(2), pp.5105-5164.
    """

    def __init__(self, neg_log_input_density, generator, obs_output,
                 metric=None, grad_neg_log_input_density=None,
                 jacob_generator=None, mhp_generator=None):
        self._generator = generator
        self.obs_output = obs_output
        super().__init__(
            pot_energy=neg_log_input_density,
            grad_pot_energy=grad_neg_log_input_density,
            metric=metric)
        self._jacob_generator = autodiff_fallback(
            jacob_generator, generator, 'jacobian_and_value', 'jacob_generator'
        )
        self._mhp_generator = autodiff_fallback(
            mhp_generator, generator, 'mhp_jacobian_and_value', 'mhp_generator'
        )

    @cache_in_state('pos')
    def generator(self, state):
        return self._generator(state.pos)

    @multi_cache_in_state(['pos'], ['jacob_generator', 'generator'])
    def jacob_generator(self, state):
        return self._jacob_generator(state.pos)

    @multi_cache_in_state(
        ['pos'], ['mhp_generator', 'jacob_generator', 'generator'])
    def mhp_generator(self, state):
        return self._mhp_generator(state.pos)

    def constr(self, state):
        return self.generator(state) - self.obs_output

    def jacob_constr(self, state):
        return self.jacob_generator(state)

    @cache_in_state('pos')
    def log_det_sqrt_gram(self, state):
        chol_gram = self.chol_gram(state)
        return np.log(chol_gram.diagonal()).sum()

    @cache_in_state('pos')
    def grad_log_det_sqrt_gram(self, state):
        mhp_generator = self.mhp_generator(state)
        jacob_generator = self.jacob_generator(state)
        chol_gram = self.chol_gram(state)
        gram_inv_jacob_generator = sla.cho_solve(
            (chol_gram, True), jacob_generator)
        return self.mhp_generator(state)(gram_inv_jacob_generator)

    @cache_in_state('pos')
    def neg_log_input_density(self, state):
        return self._pot_energy(state.pos)

    @multi_cache_in_state(
        ['pos'], ['grad_neg_log_input_density', 'neg_log_input_density'])
    def grad_neg_log_input_density(self, state):
        return self._grad_pot_energy(state.pos)

    def pot_energy(self, state):
        return (self.neg_log_input_density(state) +
                self.log_det_sqrt_gram(state))

    def grad_pot_energy(self, state):
        return (self.grad_neg_log_input_density(state) +
                self.grad_log_det_sqrt_gram(state))


class RiemannianMetricSystem(HamiltonianSystem):
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

    def __init__(self, pot_energy, metric, grad_pot_energy=None):
        self.metric = metric
        super().__init__(pot_energy, grad_pot_energy)

    def h(self, state):
        return self.h1(state) + self.h2(state)

    def h1(self, state):
        return self.pot_energy(state) + self.metric.log_det_sqrt(state)

    def h2(self, state):
        return 0.5 * self.metric.quadratic_form_inv(state, state.mom)

    def dh1_dpos(self, state):
        return (
            self.grad_pot_energy(state) + self.metric.grad_log_det_sqrt(state))

    def dh2_dpos(self, state):
        return 0.5 * self.metric.grad_quadratic_form_inv(state, state.mom)

    def dh_dpos(self, state):
        return self.dh1_dpos(state) + self.dh2_dpos(state)

    def dh_dmom(self, state):
        return self.metric.lmult_inv(state, state.mom)

    def sample_momentum(self, state, rng):
        return self.metric.lmult_sqrt(state, rng.normal(size=state.pos.shape))


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

    def __init__(self, pot_energy, grad_pot_energy=None, hess_pot_energy=None,
                 mtp_pot_energy=None, softabs_coeff=1.):
        super().__init__(pot_energy, grad_pot_energy)
        self._hess_pot_energy = autodiff_fallback(
            hess_pot_energy, pot_energy, 'hessian_grad_and_value',
            'hess_pot_energy')
        self._mtp_pot_energy = autodiff_fallback(
            mtp_pot_energy, pot_energy, 'mtp_hessian_grad_and_value',
            'mtp_pot_energy')
        self.metric = SoftAbsRiemannianMetric(self, softabs_coeff)

    @multi_cache_in_state(
        ['pos'], ['hess_pot_energy', 'grad_pot_energy', 'pot_energy'])
    def hess_pot_energy(self, state):
        return self._hess_pot_energy(state.pos)

    @multi_cache_in_state(
        ['pos'],
        ['mtp_pot_energy', 'hess_pot_energy', 'grad_pot_energy', 'pot_energy'])
    def mtp_pot_energy(self, state):
        return self._mtp_pot_energy(state.pos)
