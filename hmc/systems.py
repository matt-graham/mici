"""Classes to represent Hamiltonian systems of various types."""

import logging
import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt

autograd_available = True
try:
    from autograd import grad, value_and_grad, jacobian, make_vjp
except ImportError:
    autograd_available = False


class HamiltonianSystem(object):
    """Base class for Hamiltonian systems."""

    def __init__(self, pot_energy, grad_pot_energy=None):
        self._pot_energy = pot_energy
        if grad_pot_energy is None and autograd_available:
            self._grad_pot_energy = grad(pot_energy)
            self._val_and_grad_pot_energy = value_and_grad(pot_energy)
        elif grad_pot_energy is None and not autograd_available:
            raise ValueError('Autograd not available therefore grad_pot_energy'
                             ' must be provided.')
        else:
            self._grad_pot_energy = grad_pot_energy
            self._val_and_grad_pot_energy = None

    def pot_energy(self, state):
        if state.pot_energy is None:
            state.pot_energy = self._pot_energy(state.pos)
        return state.pot_energy

    def grad_pot_energy(self, state):
        if state.grad_pot_energy is None:
            if self._val_and_grad_pot_energy is None:
                state.grad_pot_energy = self._grad_pot_energy(state.pos)
            else:
                state.pot_energy, state.grad_pot_energy = (
                    self._val_and_grad_pot_energy(state.pos))
        return state.grad_pot_energy

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
    a term depending only on the position (target) variables, typically denoted
    the potential energy, and a second term depending only on the momentum
    variables, typically denoted the kinetic energy.
    """

    def kin_energy(self, state):
        if state.kin_energy is None:
            state.kin_energy = self._kin_energy(state.mom)
        return state.kin_energy

    def grad_kin_energy(self, state):
        if state.grad_kin_energy is None:
            state.grad_kin_energy = self._grad_kin_energy(state.mom)
        return state.grad_kin_energy

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


class IsotropicEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with isotropic metric.

    The momenta are taken to be independent of the position variables and with
    a isotropic covariance zero-mean Gaussian marginal distribution.
    """

    def __init__(self, pot_energy, grad_pot_energy=None):
        super().__init__(pot_energy, grad_pot_energy)

    def _kin_energy(self, mom):
        return 0.5 * np.sum(mom**2)

    def _grad_kin_energy(self, mom):
        return mom

    def sample_momentum(self, state, rng):
        return rng.normal(size=state.pos.shape)


class DiagonalEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with diagonal metric.

    The momenta are taken to be independent of the position variables and with
    a zero-mean Gaussian marginal distribution with diagonal covariance matrix.
    """

    def __init__(self, pot_energy, diag_metric, grad_pot_energy=None):
        super().__init__(pot_energy, grad_pot_energy)
        self.diag_metric = diag_metric

    def _kin_energy(self, mom):
        return 0.5 * np.sum(mom**2 / self.diag_metric)

    def _grad_kin_energy(self, mom):
        return mom / self.diag_metric

    def sample_momentum(self, state, rng):
        return self.diag_metric**0.5 * rng.normal(size=state.pos.shape)


class DenseEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with dense metric.

    The momenta are taken to be independent of the position variables and with
    a zero-mean Gaussian marginal distribution with dense covariance matrix.
    """

    def __init__(self, pot_energy, metric, grad_pot_energy=None):
        super().__init__(pot_energy, grad_pot_energy)
        self.metric = metric
        self.chol_metric = sla.cholesky(metric, lower=True)

    def _kin_energy(self, mom):
        return 0.5 * mom @ self._grad_kin_energy(mom)

    def _grad_kin_energy(self, mom):
        return sla.cho_solve((self.chol_metric, True), mom)

    def sample_momentum(self, state, rng):
        return self.chol_metric @ rng.normal(size=state.pos.shape)


class BaseRiemannianMetricHamiltonianSystem(HamiltonianSystem):

    def sqrt_metric(self, state):
        raise NotImplementedError()

    def log_det_sqrt_metric(self, state):
        raise NotImplementedError()

    def grad_log_det_sqrt_metric(self, state):
        raise NotImplementedError()

    def grad_mom_inv_metric_mom(self, state):
        raise NotImplementedError()

    def inv_metric_mom(self, state):
        raise NotImplementedError()

    def h(self, state):
        return self.h1(state) + self.h2(state)

    def h1(self, state):
        return self.pot_energy(state) + self.log_det_sqrt_metric(state)

    def h2(self, state):
        return 0.5 * state.mom @ self.inv_metric_mom(state)

    def dh1_dpos(self, state):
        return (
            self.grad_pot_energy(state) +
            self.grad_log_det_sqrt_metric(state))

    def dh2_dpos(self, state):
        return 0.5 * self.grad_mom_inv_metric_mom(state)

    def dh_dpos(self, state):
        return self.dh1_dpos(state) + self.dh2_dpos(state)

    def dh_dmom(self, state):
        return self.inv_metric_mom(state)

    def sample_momentum(self, state, rng):
        sqrt_metric = self.sqrt_metric(state)
        return sqrt_metric @ rng.normal(size=state.pos.shape)


class BaseCholeskyRiemannianMetricHamiltonianSystem(
        BaseRiemannianMetricHamiltonianSystem):

    def chol_metric(self, state):
        raise NotImplementedError()

    def log_det_sqrt_metric(self, state):
        chol_metric = self.chol_metric(state)
        return np.log(chol_metric.diagonal()).sum()

    def inv_metric_mom(self, state):
        if state.inv_metric_mom is None:
            chol_metric = self.chol_metric(state)
            state.inv_metric_mom = sla.cho_solve(
                (chol_metric, True), state.mom)
        return state.inv_metric_mom

    def sqrt_metric(self, state):
        return self.chol_metric(state)


class DenseRiemannianMetricHamiltonianSystem(
            BaseCholeskyRiemannianMetricHamiltonianSystem):

    def __init__(self, pot_energy, metric, grad_pot_energy=None,
                 vjp_metric=None):
        super().__init__(pot_energy, grad_pot_energy)
        self._metric = metric
        if vjp_metric is None and autograd_available:
            self._vjp_metric = make_vjp(metric)
        elif vjp_metric is None and not autograd_available:
            raise ValueError('Autograd not available therefore vjp_metric'
                             ' must be provided.')
        else:
            self._vjp_metric = vjp_metric

    def grad_log_det_sqrt_metric(self, state):
        inv_metric = self.inv_metric(state)
        return 0.5 * self.vjp_metric(state)(inv_metric)

    def grad_mom_inv_metric_mom(self, state):
        inv_metric_mom = self.inv_metric_mom(state)
        inv_metric_mom_outer = np.outer(inv_metric_mom, inv_metric_mom)
        return -self.vjp_metric(state)(inv_metric_mom_outer)

    def metric(self, state):
        if state.metric is None:
            state.metric = self._metric(state.pos)
        return state.metric

    def chol_metric(self, state):
        if state.chol_metric is None:
            state.chol_metric = sla.cholesky(self.metric(state), True)
        return state.chol_metric

    def inv_metric(self, state):
        if state.inv_metric is None:
            chol_metric = self.chol_metric(state)
            state.inv_metric = sla.cho_solve(
                (chol_metric, True), np.eye(state.n_dim))
        return state.inv_metric

    def vjp_metric(self, state):
        if state.vjp_metric is None:
            state.vjp_metric, state.metric = self._vjp_metric(state.pos)
        return state.vjp_metric


class FactoredRiemannianMetricHamiltonianSystem(
            BaseCholeskyRiemannianMetricHamiltonianSystem):

    def __init__(self, pot_energy, chol_metric, grad_pot_energy=None,
                 vjp_chol_metric=None):
        super().__init__(pot_energy, grad_pot_energy)
        self._chol_metric = chol_metric
        if vjp_chol_metric is None and autograd_available:
            self._vjp_chol_metric = make_vjp(chol_metric)
        elif vjp_chol_metric is None and not autograd_available:
            raise ValueError('Autograd not available therefore vjp_chol_metric'
                             ' must be provided.')
        else:
            self._vjp_chol_metric = vjp_chol_metric

    def grad_log_det_sqrt_metric(self, state):
        inv_chol_metric = self.inv_chol_metric(state)
        return self.vjp_chol_metric(state)(inv_chol_metric.T)

    def grad_mom_inv_metric_mom(self, state):
        chol_metric = self.chol_metric(state)
        inv_chol_metric_mom = sla.solve_triangular(
            chol_metric, state.mom, lower=True)
        inv_metric_mom = self.inv_metric_mom(state)
        inv_metric_mom_outer = np.outer(inv_metric_mom, inv_chol_metric_mom)
        return -2 * self.vjp_chol_metric(state)(inv_metric_mom_outer)

    def chol_metric(self, state):
        if state.chol_metric is None:
            state.chol_metric = self._chol_metric(state.pos)
        return state.chol_metric

    def inv_chol_metric(self, state):
        if state.inv_chol_metric is None:
            chol_metric = self.chol_metric(state)
            state.inv_chol_metric = sla.solve_triangular(
                chol_metric, np.eye(state.n_dim), lower=True)
        return state.inv_chol_metric

    def vjp_chol_metric(self, state):
        if state.vjp_chol_metric is None:
            state.vjp_chol_metric, state.chol_metric = self._vjp_chol_metric(
                state.pos)
        return state.vjp_chol_metric
