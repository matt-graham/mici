# -*- coding: utf-8 -*-
""" Numpy implementations of unconstrained Euclidean metric HMC samplers. """

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import scipy.linalg as la
from .base import AbstractHmcSampler


class IsotropicHmcSampler(AbstractHmcSampler):
    """Standard unconstrained HMC sampler with identity mass matrix. """

    def __init__(self, energy_func, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        super(IsotropicHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)

    def kinetic_energy(self, pos, cache, mom):
        return 0.5 * mom.dot(mom)

    def simulate_dynamic(self, pos, mom, cache, dt, n_step):
        mom = mom - 0.5 * dt * self.energy_grad(pos)
        pos = pos + dt * mom
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos)
            pos += dt * mom
        mom -= 0.5 * dt * self.energy_grad(pos)
        return pos, mom, None

    def sample_independent_momentum_given_position(pos, cache):
        return self.prng.normal(size=pos.shape[0]).astype(self.dtype)


class EuclideanMetricHamiltonianSampler(object):
    """Standard unconstrained HMC sampler with constant mass matrix. """

    def __init__(self, energy_func, mass_matrix, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        super(EuclideanMetricHamiltonianSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.mass_matrix = mass_matrix
        self.mass_matrix_chol = la.cho_factor(mass_matrix)

    def kinetic_energy(self, pos, cache, mom):
        return 0.5 * mom.dot(la.cho_solve(self.mass_matrix_chol, mom))

    def simulate_dynamic(self, pos, mom, cache, dt, n_step):
        mom = mom - 0.5 * dt * self.energy_grad(pos)
        pos = pos + dt * la.cho_solve(self.mass_matrix_chol, mom)
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos)
            pos += dt * la.cho_solve(self.mass_matrix_chol, mom)
        mom -= 0.5 * dt * self.energy_grad(pos)
        return pos, mom, None

    def sample_independent_momentum_given_position(pos, cache):
        return (self.mass_matrix_chol.dot(
            self.prng.normal(size=pos.shape[0]))).astype(self.dtype)
