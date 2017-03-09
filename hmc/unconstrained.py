# -*- coding: utf-8 -*-
""" Numpy implementations of unconstrained Euclidean metric HMC samplers. """

import numpy as np
import scipy.linalg as la
from .base import AbstractHmcSampler


class IsotropicHmcSampler(AbstractHmcSampler):
    """Standard unconstrained HMC sampler with identity mass matrix. """

    def kinetic_energy(self, pos, mom, cache={}):
        return 0.5 * mom.dot(mom)

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos = pos + dt * mom
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos, cache)
            pos += dt * mom
        mom -= 0.5 * dt * self.energy_grad(pos, cache)
        return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        return self.prng.normal(size=pos.shape[0]).astype(self.dtype)


class EuclideanMetricHmcSampler(IsotropicHmcSampler):
    """Standard unconstrained HMC sampler with constant mass matrix. """

    def __init__(self, energy_func, mass_matrix, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        super(EuclideanMetricHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.mass_matrix = mass_matrix
        self.mass_matrix_chol = la.cho_factor(mass_matrix)

    def kinetic_energy(self, pos, mom, cache={}):
        return 0.5 * mom.dot(la.cho_solve(self.mass_matrix_chol, mom))

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        mom = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos = pos + dt * la.cho_solve(self.mass_matrix_chol, mom)
        for s in range(1, n_step):
            mom -= dt * self.energy_grad(pos, cache)
            pos += dt * la.cho_solve(self.mass_matrix_chol, mom)
        mom -= 0.5 * dt * self.energy_grad(pos, cache)
        return pos, mom, None

    def sample_independent_momentum_given_position(self, pos, cache={}):
        return (self.mass_matrix_chol.dot(
            self.prng.normal(size=pos.shape[0]))).astype(self.dtype)
