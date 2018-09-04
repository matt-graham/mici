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
