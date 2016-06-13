import numpy as np
import unittest
import hmc.constrained as chmc

autograd_available = True
try:
    import autograd.numpy as ag_np
except ImportError:
    autograd_available = False


class TestConstrainedIsotropicHMCSampler(unittest.TestCase):

    def setUp(self):
        self.prng = np.random.RandomState(1234)

    def test_init_with_constr_jacob(self):
        energy_func = lambda pos, cache={}: 0.5 * pos.dot(pos)
        energy_grad = lambda pos, cache={}: pos
        constr_func = lambda pos: pos[:2]
        constr_jacob = lambda pos: np.c_[np.eye(2),
                                         np.zeros((2, pos.shape[0] - 2))]
        constr_jacob_wrapped = chmc.wrap_constr_jacob_func(constr_jacob)
        mom_resample_coeff = 1.
        dtype = np.float64
        tol = 1e-8
        max_iters = 100
        sampler = chmc.ConstrainedIsotropicHmcSampler(
            energy_func=energy_func,
            constr_func=constr_func,
            energy_grad=energy_grad,
            constr_jacob=constr_jacob_wrapped,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype,
            tol=tol,
            max_iters=max_iters)
        assert sampler.energy_func == energy_func
        assert sampler.constr_func == constr_func
        assert sampler.energy_grad == energy_grad
        assert sampler.constr_jacob == constr_jacob_wrapped
        assert sampler.prng == self.prng
        assert sampler.mom_resample_coeff == mom_resample_coeff
        assert sampler.dtype == dtype
        assert sampler.tol == tol
        assert sampler.max_iters == max_iters

    def test_init_without_energy_grad(self):
        if autograd_available:
            energy_func = lambda pos, cache={}: 0.5 * pos.dot(pos)
            energy_grad = lambda pos, cache={}: pos
            constr_func = lambda pos: pos[:2]
            constr_jacob = lambda pos: np.c_[np.eye(2),
                                             np.zeros((2, pos.shape[0] - 2))]
            constr_jacob_wrapped = chmc.wrap_constr_jacob_func(constr_jacob)
            mom_resample_coeff = 1.
            dtype = np.float64
            tol = 1e-8
            max_iters = 100
            sampler = chmc.ConstrainedIsotropicHmcSampler(
                energy_func=energy_func,
                constr_func=constr_func,
                energy_grad=energy_grad,
                constr_jacob=None,
                prng=self.prng,
                mom_resample_coeff=mom_resample_coeff,
                dtype=dtype,
                tol=tol,
                max_iters=max_iters)
            assert sampler.constr_jacob is not None, (
                'Sampler constr_jacob not being automatically defined using '
                'Autograd correctly.'
            )
            pos = self.prng.normal(size=5)
            dc_dpos = sampler.constr_jacob(pos, False)['dc_dpos']
            assert np.allclose(dc_dpos, constr_jacob(pos)), (
                'Sampler constr_jacob inconsistent with constr_func.'
            )

    def test_dynamic_reversible(self):
        energy_func = lambda pos, cache={}: 0.5 * pos.dot(pos)
        energy_grad = lambda pos, cache={}: pos
        constr_func = lambda pos: pos[:2]
        constr_jacob = lambda pos: np.c_[np.eye(2),
                                         np.zeros((2, pos.shape[0] - 2))]
        constr_jacob_wrapped = chmc.wrap_constr_jacob_func(constr_jacob)
        mom_resample_coeff = 0.
        dtype = np.float64
        tol = 1e-8
        max_iters = 100
        sampler = chmc.ConstrainedIsotropicHmcSampler(
            energy_func=energy_func,
            constr_func=constr_func,
            energy_grad=energy_grad,
            constr_jacob=constr_jacob_wrapped,
            prng=self.prng,
            mom_resample_coeff=mom_resample_coeff,
            dtype=dtype,
            tol=tol,
            max_iters=max_iters)

        def do_reversible_check(n_dim, dt, n_step):
            pos_0, mom_0 = self.prng.normal(size=(2, n_dim))
            pos_0[:2] = 0
            mom_0[:2] = 0
            assert np.max(np.abs(sampler.constr_func(pos_0))) < tol, (
                'Initial position not satisfying constraint function.'
            )
            pos_f, mom_f, cache_f = sampler.simulate_dynamic(
                n_step, dt, pos_0, mom_0, {})
            pos_r, mom_r, cache_r = sampler.simulate_dynamic(
                n_step, -dt, pos_f, mom_f, {})
            assert np.allclose(pos_0, pos_r) and np.allclose(mom_0, mom_r), (
                'Hamiltonian dynamic simulation not time-reversible. ' +
                'Initial state\n  {0}\n  {1}\n'.format(pos_0, mom_0) +
                'Reversed state\n  {0}\n  {1}\n'.format(pos_r, mom_r)
            )
        do_reversible_check(5, 0.05, 5)
        do_reversible_check(10, 0.05, 100)
