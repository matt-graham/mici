# -*- coding: utf-8 -*-
"""Constrained HMC sampler tests."""

import unittest
import hmc.constrained as chmc

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


def constr_func(pos):
    return pos[:2]


@chmc.wrap_constr_jacob_func
def constr_jacob(pos):
    return np.c_[np.eye(2), np.zeros((2, pos.shape[0] - 2))]


class ConstrainedSamplerTestCase(object):

    def test_init_with_constr_jacob(self):
        mom_resample_coeff = 1.
        dtype = np.float64
        tol = 1e-8
        max_iters = 100
        sampler = self.test_class(
            energy_func=energy_func,
            constr_func=constr_func,
            energy_grad=energy_grad,
            constr_jacob=constr_jacob,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype,
            tol=tol,
            max_iters=max_iters)
        assert sampler.energy_func == energy_func
        assert sampler.constr_func == constr_func
        assert sampler.energy_grad == energy_grad
        assert sampler.constr_jacob == constr_jacob
        assert sampler.prng == self.prng
        assert sampler.mom_resample_coeff == mom_resample_coeff
        assert sampler.dtype == dtype
        assert sampler.tol == tol
        assert sampler.max_iters == max_iters

    def test_init_without_grads(self):
        if autograd_available:
            constr_jacob_wrapped = chmc.wrap_constr_jacob_func(constr_jacob)
            mom_resample_coeff = 1.
            dtype = np.float64
            tol = 1e-8
            max_iters = 100
            sampler = self.test_class(
                energy_func=energy_func,
                constr_func=constr_func,
                energy_grad=None,
                constr_jacob=None,
                prng=self.prng,
                mom_resample_coeff=mom_resample_coeff,
                dtype=dtype,
                tol=tol,
                max_iters=max_iters)
            assert sampler.energy_grad is not None, (
                'Sampler energy_grad not being automatically defined using '
                'Autograd correctly.'
            )
            pos = self.prng.normal(size=5)
            de_dpos = sampler.energy_grad(pos, {})
            assert np.allclose(de_dpos, energy_grad(pos)), (
                'Sampler energy_grad inconsistent with energy_func.'
            )
            assert sampler.constr_jacob is not None, (
                'Sampler constr_jacob not being automatically defined using '
                'Autograd correctly.'
            )
            pos = self.prng.normal(size=5)
            dc_dpos = sampler.constr_jacob(pos, False)['dc_dpos']
            assert np.allclose(dc_dpos, constr_jacob(pos, False)['dc_dpos']), (
                'Sampler constr_jacob inconsistent with constr_func.'
            )

    def test_dynamic_reversible(self):
        mom_resample_coeff = 0.
        dtype = np.float64
        tol = 1e-8
        max_iters = 100
        sampler = self.test_class(
            energy_func=energy_func,
            constr_func=constr_func,
            energy_grad=energy_grad,
            constr_jacob=constr_jacob,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype,
            tol=tol,
            max_iters=max_iters)

        def do_reversible_check(n_dim, dt, n_step):
            pos_0, mom_0 = self.prng.normal(size=(2, n_dim))
            constr_dim = sampler.constr_func(pos_0).shape[0]
            rand_basis = self.prng.normal(size=(constr_dim, n_dim))
            pos_0 = chmc.project_onto_constraint_surface(
                pos_0, {'dc_dpos': rand_basis}, sampler.constr_func, tol=tol,
                max_iters=sampler.max_iters, scipy_opt_fallback=True,
                constr_jacob=sampler.constr_jacob)
            cache = sampler.constr_jacob(pos_0, True)
            mom_0 = chmc.project_onto_nullspace(mom_0, cache)
            assert np.max(np.abs(sampler.constr_func(pos_0))) < tol, (
                'Initial position not satisfying constraint function.'
            )
            assert np.max(np.abs(cache['dc_dpos'].dot(mom_0))) < tol, (
                'Initial momentum not in constraint manifold tangent space.'
            )
            pos_f, mom_f, cache_f = sampler.simulate_dynamic(
                n_step, dt, pos_0, mom_0, {})
            assert np.max(np.abs(sampler.constr_func(pos_f))) < tol, (
                'Final position not satisfying constraint function.'
            )
            assert np.max(np.abs(cache_f['dc_dpos'].dot(mom_f))) < tol, (
                'Final momentum not in constraint manifold tangent space.'
            )
            pos_r, mom_r, cache_r = sampler.simulate_dynamic(
                n_step, -dt, pos_f, mom_f, {})
            assert np.allclose(pos_0, pos_r) and np.allclose(mom_0, mom_r), (
                'Hamiltonian dynamic simulation not time-reversible. ' +
                'Initial state\n  {0}\n  {1}\n'.format(pos_0, mom_0) +
                'Reversed state\n  {0}\n  {1}\n'.format(pos_r, mom_r)
            )
        do_reversible_check(5, 0.05, 5)
        do_reversible_check(10, 0.05, 100)


class ConstrainedIsotropicHmcTestCase(ConstrainedSamplerTestCase,
                                      unittest.TestCase):

    def setUp(self):
        self.test_class = chmc.ConstrainedIsotropicHmcSampler
        self.prng = np.random.RandomState(SEED)


class RattleConstrainedIsotropicHmcTestCase(ConstrainedSamplerTestCase,
                                            unittest.TestCase):

    def setUp(self):
        self.test_class = chmc.RattleConstrainedIsotropicHmcSampler
        self.prng = np.random.RandomState(SEED)


class GbabConstrainedIsotropicHmcTestCase(ConstrainedSamplerTestCase,
                                          unittest.TestCase):

    def setUp(self):
        self.test_class = chmc.GbabConstrainedIsotropicHmcSampler
        self.prng = np.random.RandomState(SEED)


class LfGbabConstrainedIsotropicHmcTestCase(ConstrainedSamplerTestCase,
                                            unittest.TestCase):

    def setUp(self):
        self.test_class = chmc.LfGbabConstrainedIsotropicHmcSampler
        self.prng = np.random.RandomState(SEED)
