import numpy as np
import unittest
import hmc.unconstrained as uhmc

autograd_available = True
try:
    import autograd.numpy as ag_np
except ImportError:
    autograd_available = False


class TestIsotropicHMCSampler(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(1234)

    def test_init_with_energy_grad(self):
        energy_func = lambda pos, **cache: 0.5 * pos.dot(pos)
        energy_grad = lambda pos, **cache: pos
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
            energy_func = lambda pos, **cache: 0.5 * ag_np.dot(pos, pos)
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

    def test_dynamic_reversible(self):
        energy_func = lambda pos, **cache: 0.5 * pos.dot(pos)
        energy_grad = lambda pos, **cache: pos
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
