# -*- coding: utf-8 -*-
"""Unconstrained HMC sampler tests."""

import unittest
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


class UnconstrainedSamplerTestCase(object):

    def test_init_with_energy_grad(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        sampler = self.test_class(
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
            sampler = self.test_class(
                energy_func=energy_func, prng=self.prng)
            assert sampler.energy_grad is not None, (
                'Sampler energy_grad not being automatically defined using '
                'Autograd correctly.'
            )
            pos = self.prng.normal(size=5)
            assert np.allclose(sampler.energy_grad(pos), pos), (
                'Sampler energy_grad inconsistent with energy_func.'
            )

    def test_dynamic_reversible(self):
        sampler = self.test_class(
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


class IsotropicHmcTestCase(UnconstrainedSamplerTestCase, unittest.TestCase):

    def setUp(self):
        self.test_class = uhmc.IsotropicHmcSampler
        self.prng = np.random.RandomState(SEED)
