"""Classes to represent Hamiltonian systems of various types."""

import logging
import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt

autograd_available = True
try:
    from autograd import grad, value_and_grad, jacobian
except ImportError:
    autograd_available = False


class SeparableHamiltonianSystem(object):
    """Base class for separable Hamiltonian systems.

    Here separable means that the Hamiltonian can be expressed as the sum of
    a term depending only on the position (target) variables, typically denoted
    the potential energy, and a second term depending only on the momentum
    variables, typically denoted the kinetic energy.
    """

    def __init__(self, pot_energy, pot_energy_grad=None):
        self._pot_energy = pot_energy
        if pot_energy_grad is None and autograd_available:
            self._pot_energy_grad = grad(pot_energy)
            self._pot_energy_val_and_grad = value_and_grad(pot_energy)
        elif pot_energy_grad is None and not autograd_available:
            raise ValueError('Autograd not available therefore pot_energy_grad'
                             ' must be provided.')
        else:
            self._pot_energy_grad = pot_energy_grad
            self._pot_energy_val_and_grad = None

    def pot_energy(self, state):
        if state.pot_energy_val is None:
            state.pot_energy_val = self._pot_energy(state.pos)
        return state.pot_energy_val

    def pot_energy_grad(self, state):
        if state.pot_energy_grad is None:
            if self._pot_energy_val_and_grad is None:
                state.pot_energy_grad = self._pot_energy_grad(state.pos)
            else:
                state.pot_energy_val, state.pot_energy_grad = (
                    self._pot_energy_val_and_grad(state.pos))
        return state.pot_energy_grad

    def kin_energy(self, state):
        if state.kin_energy_val is None:
            state.kin_energy_val = self._kin_energy(state.mom)
        return state.kin_energy_val

    def kin_energy_grad(self, state):
        if state.kin_energy_grad is None:
            state.kin_energy_grad = self._kin_energy_grad(state.mom)
        return state.kin_energy_grad

    def h(self, state):
        return self.pot_energy(state) + self.kin_energy(state)

    def dh_dpos(self, state):
        return self.pot_energy_grad(state)

    def dh_dmom(self, state):
        return self.kin_energy_grad(state)

    def _kin_energy(self, mom):
        raise NotImplementedError()

    def _kin_energy_grad(self, mom):
        raise NotImplementedError()

    def sample_momentum(self, state, rng):
        raise NotImplementedError()


class IsotropicEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with isotropic metric.

    The momenta are taken to be independent of the position variables and with
    a isotropic covariance zero-mean Gaussian marginal distribution.
    """

    def __init__(self, pot_energy, pot_energy_grad=None):
        super().__init__(pot_energy, pot_energy_grad)

    def _kin_energy(self, mom):
        return 0.5 * np.sum(mom**2)

    def _kin_energy_grad(self, mom):
        return mom

    def sample_momentum(self, state, rng):
        return rng.normal(size=state.pos.shape)


class DiagonalEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with diagonal metric.

    The momenta are taken to be independent of the position variables and with
    a zero-mean Gaussian marginal distribution with diagonal covariance matrix.
    """

    def __init__(self, pot_energy, diag_metric, pot_energy_grad=None):
        self.diag_metric = diag_metric
        super().__init__(pot_energy, pot_energy_grad)

    def _kin_energy(self, mom):
        return 0.5 * np.sum(mom**2 / self.diag_metric)

    def _kin_energy_grad(self, mom):
        return mom / self.diag_metric

    def sample_momentum(self, state, rng):
        return self.diag_metric**0.5 * rng.normal(size=state.pos.shape)


class DenseEuclideanMetricHamiltonianSystem(SeparableHamiltonianSystem):
    """Euclidean-Gaussian Hamiltonian system with dense metric.

    The momenta are taken to be independent of the position variables and with
    a zero-mean Gaussian marginal distribution with dense covariance matrix.
    """

    def __init__(self, pot_energy, metric, pot_energy_grad=None):
        self.metric = metric
        self.chol_metric = sla.cholesky(metric, lower=True)
        super().__init__(pot_energy, pot_energy_grad)

    def _kin_energy(self, mom):
        return 0.5 * mom @ self._kin_energy_grad(mom)

    def _kin_energy_grad(self, mom):
        return sla.cho_solve((self.chol_metric, True), mom)

    def sample_momentum(self, state, rng):
        return self.chol_metric @ rng.normal(size=state.pos.shape)
