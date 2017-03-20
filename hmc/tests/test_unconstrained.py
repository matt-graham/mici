# -*- coding: utf-8 -*-
"""Unconstrained HMC sampler tests."""

import unittest
import scipy.linalg as la
import hmc.unconstrained as uhmc

autograd_available = True
try:
    import autograd.numpy as np
except ImportError:
    autograd_available = False
    import numpy as np


SEED = 1234


def energy_func(pos, cache={}):
    return 0.5 * np.dot(pos, pos)


def energy_grad(pos, cache={}):
    return pos


class IsotropicHmcSamplerTestCase(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(SEED)

    def test_init_with_energy_grad(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        sampler = uhmc.IsotropicHmcSampler(
            energy_func=energy_func,
            energy_grad=energy_grad,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype)
        assert sampler.energy_func == energy_func
        assert sampler.energy_grad == energy_grad
        assert sampler.prng == self.prng
        assert sampler.mom_resample_coeff == mom_resample_coeff
        assert sampler.dtype == dtype

    def test_init_without_energy_grad(self):
        if autograd_available:
            sampler = uhmc.IsotropicHmcSampler(
                energy_func=energy_func, prng=self.prng)
            assert sampler.energy_grad is not None, (
                'Sampler energy_grad not being automatically defined using '
                'Autograd correctly.'
            )
            pos = self.prng.normal(size=5)
            assert np.allclose(sampler.energy_grad(pos), pos), (
                'Sampler energy_grad inconsistent with energy_func.'
            )

    def test_kinetic_energy(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        sampler = uhmc.IsotropicHmcSampler(
            energy_func=energy_func,
            energy_grad=energy_grad,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype)
        for n_dim in [10, 100, 1000]:
            pos, mom = self.prng.normal(size=(2, n_dim,)).astype(dtype)
            k_energy = sampler.kinetic_energy(pos, mom, {})
            assert np.isscalar(k_energy), (
                'kinetic_energy returning non-scalar value.'
            )
            assert np.allclose(k_energy, 0.5 * mom.dot(mom)), (
                'kinetic_energy returning incorrect value.'
            )

    def test_sample_independent_momentum(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        sampler = uhmc.IsotropicHmcSampler(
            energy_func=energy_func,
            energy_grad=energy_grad,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype)
        for n_dim in [10, 100, 1000]:
            pos = self.prng.normal(size=(n_dim,)).astype(dtype)
            mom = sampler.sample_independent_momentum_given_position(
                pos, cache={}
            )
            assert mom.ndim == pos.ndim and mom.shape[0] == pos.shape[0], (
                'Momentum sampling returning incorrect shaped array.'
            )
            assert mom.dtype == pos.dtype, (
                'Momentum sampling returning array with incorrect dtype.'
            )
            assert abs(mom.mean()) < 5. / n_dim**0.5, (
                'Mean of sampled momentum > 5 std. from expected value.'
            )

    def test_dynamic_reversible(self):
        sampler = uhmc.IsotropicHmcSampler(
            energy_func=energy_func, energy_grad=energy_grad,
            prng=self.prng, dtype=np.float64)

        def do_reversible_check(n_dim, dt, n_step):
            pos_0, mom_0 = self.prng.normal(size=(2, n_dim))
            pos_f, mom_f, cache_f = sampler.simulate_dynamic(
                n_step, dt, pos_0, mom_0, {})
            pos_r, mom_r, cache_r = sampler.simulate_dynamic(
                n_step, -dt, pos_f, mom_f, {})
            assert np.allclose(pos_0, pos_r) and np.allclose(mom_0, mom_r), (
                'Hamiltonian dynamic simulation not time-reversible. ' +
                'Initial state\n  {0}\n  {1}\n'.format(pos_0, mom_0) +
                'Reversed state\n  {0}\n  {1}\n'.format(pos_r, mom_r)
            )

        do_reversible_check(2, 0.1, 5)
        do_reversible_check(10, 0.1, 100)


class EuclideanMetricHmcSamplerTestCase(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(SEED)

    def test_init_with_energy_grad(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        mass_matrix = np.array([[5., 3., -1], [3., 4., 2.], [-1., 2., 8.]])
        sampler = uhmc.EuclideanMetricHmcSampler(
            energy_func=energy_func,
            mass_matrix=mass_matrix,
            energy_grad=energy_grad,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype)
        assert sampler.energy_func == energy_func
        assert np.allclose(sampler.mass_matrix, mass_matrix)
        assert np.allclose(
            sampler.mass_matrix_chol.dot(sampler.mass_matrix_chol.T),
            mass_matrix), 'Mass matrix Cholesky factor inconsistent.'
        assert sampler.energy_grad == energy_grad
        assert sampler.prng == self.prng
        assert sampler.mom_resample_coeff == mom_resample_coeff
        assert sampler.dtype == dtype

    def test_kinetic_energy(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        for n_dim in [10, 100, 1000]:
            mass_matrix = self.prng.normal(size=(n_dim, n_dim))
            mass_matrix = mass_matrix.dot(mass_matrix.T)
            mass_matrix_chol = la.cholesky(mass_matrix, lower=True)
            sampler = uhmc.EuclideanMetricHmcSampler(
                energy_func=energy_func,
                mass_matrix=mass_matrix,
                energy_grad=energy_grad,
                prng=self.prng,
                mom_resample_coeff=mom_resample_coeff,
                dtype=dtype)
            pos, mom = self.prng.normal(size=(2, n_dim,)).astype(dtype)
            k_energy = sampler.kinetic_energy(pos, mom, {})
            assert np.isscalar(k_energy), (
                'kinetic_energy returning non-scalar value.'
            )
            assert np.allclose(
                k_energy,
                0.5 * mom.dot(la.cho_solve((mass_matrix_chol, True), mom))), (
                    'kinetic_energy returning incorrect value.'
                )

    def test_sample_independent_momentum(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        for n_dim in [10, 100, 1000]:
            mass_matrix = self.prng.normal(size=(n_dim, n_dim))
            mass_matrix = mass_matrix.dot(mass_matrix.T)
            mass_matrix_chol = la.cholesky(mass_matrix, lower=True)
            sampler = uhmc.EuclideanMetricHmcSampler(
                energy_func=energy_func,
                mass_matrix=mass_matrix,
                energy_grad=energy_grad,
                prng=self.prng,
                mom_resample_coeff=mom_resample_coeff,
                dtype=dtype)
            pos = self.prng.normal(size=(n_dim,)).astype(dtype)
            mom = sampler.sample_independent_momentum_given_position(
                pos, cache={}
            )
            assert mom.ndim == pos.ndim and mom.shape[0] == pos.shape[0], (
                'Momentum sampling returning incorrect shaped array.'
            )
            assert mom.dtype == pos.dtype, (
                'Momentum sampling returning array with incorrect dtype.'
            )
            sum_std = sum(mass_matrix.diagonal()**0.5)
            assert abs(mom.mean()) < 5. * sum_std / n_dim**0.5, (
                'Mean of sampled momentum > 5 std. from expected value.'
            )

    def test_dynamic_reversible(self):

        def do_reversible_check(n_dim, dt, n_step):
            mass_matrix = self.prng.normal(size=(n_dim, n_dim))
            mass_matrix = mass_matrix.dot(mass_matrix.T)
            sampler = uhmc.EuclideanMetricHmcSampler(
                energy_func=energy_func, mass_matrix=mass_matrix,
                energy_grad=energy_grad, prng=self.prng, dtype=np.float64)
            pos_0, mom_0 = self.prng.normal(size=(2, n_dim))
            pos_f, mom_f, cache_f = sampler.simulate_dynamic(
                n_step, dt, pos_0, mom_0, {})
            pos_r, mom_r, cache_r = sampler.simulate_dynamic(
                n_step, -dt, pos_f, mom_f, {})
            assert np.allclose(pos_0, pos_r) and np.allclose(mom_0, mom_r), (
                'Hamiltonian dynamic simulation not time-reversible. ' +
                'Initial state\n  {0}\n  {1}\n'.format(pos_0, mom_0) +
                'Reversed state\n  {0}\n  {1}\n'.format(pos_r, mom_r)
            )

        do_reversible_check(2, 0.1, 5)
        do_reversible_check(10, 0.1, 100)
