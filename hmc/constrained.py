# -*- coding: utf-8 -*-
""" Numpy implementations of constrained Euclidean metric HMC samplers. """

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from .unconstrained import IsotropicHmcSampler

autograd_available = True
try:
    from autograd import jacobian
except ImportError:
    autograd_available = False


class ExceededMaxItersError(Exception):
    """Exception raised when iterative solver exceeds iteration limit."""

    def __init__(self, solver, max_iters, tol, error):
        super(ExceededMaxItersError, self).__init__(
            'Solver ({0}) exceeded maximum number of iterations ({1}) without '
            'converging. Maximum error for convergence: {2}. '
            'Last maximum error: {3}'
            .format(solver, max_iters, tol, error))


class ConstrainedIsotropicHmcSampler(IsotropicHmcSampler):

    def __init__(self, energy_func, constr_func, energy_grad=None,
                 constr_jacob=None, prng=None, mom_resample_coeff=1.,
                 dtype=np.float64, tol=1e-8, max_iters=100):
        super(ConstrainedIsotropicHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.constr_func = constr_func
        if constr_jacob is None and autograd_available:
            constr_jacob = jacobian(constr_func)
            def constr_jacob_func(pos, calc_gram_chol=True):
                jacob = constr_jacob(pos)
                if calc_gram_chol:
                    prod_chol = la.cho_factor(jacob.dot(jacob.T))
                    return jacob, prod_chol
                else:
                    return jacob
            self.constr_jacob = constr_jacob_func
        elif constr_jacob is None and not autograd_available:
            raise ValueError('Autograd not available therefore constraint '
                             'Jacobian must be provided.')
        else:
            self.constr_jacob = constr_jacob
        self.tol = tol
        self.max_iters = max_iters

    def simulate_dynamic(self, pos, mom, cache, dt, n_step):
        if cache is None:
            dc_dpos = self.constr_jacob(pos)
        else:
            dc_dpos = cache
        mom_half = mom - 0.5 * dt * self.energy_grad(pos)
        pos_n = pos + dt * mom_half
        project_onto_constraint_surface(
            pos_n, dc_dpos, self.constr_func, self.tol, self.max_iters,
            scipy_opt_fallback=True, constr_jacob=self.constr_jacob)
        mom_half = (pos_n - pos) / dt
        pos = pos_n
        dc_dpos = self.constr_jacob(pos)
        for s in range(n_step):
            mom_half -= dt * self.energy_grad(pos)
            pos_n = pos + dt * mom_half
            project_onto_constraint_surface(
                pos_n, dc_dpos, self.constr_func, self.tol, self.max_iters,
                scipy_opt_fallback=True, constr_jacob=self.constr_jacob)
            mom_half = (pos_n - pos) / dt
            pos = pos_n
            dc_dpos = self.constr_jacob(pos)
        mom = mom_half - 0.5 * dt * self.energy_grad(pos)
        project_onto_nullspace(mom, dc_dpos)
        return pos, mom, dc_dpos

    def sample_independent_momentum_given_position(self, pos, cache):
        n_dim = pos.shape[0]
        if cache is None:
            dc_dpos = self.constr_jacob(pos)
        else:
            dc_dpos = cache
        mom = self.prng.normal(size=n_dim).astype(self.dtype)
        project_onto_nullspace(mom, dc_dpos)
        return mom


def project_onto_constraint_surface(pos, dc_dpos_prev, constr_func, tol=1e-8,
                                    max_iters=100, scipy_opt_fallback=True,
                                    constr_jacob=None):
    """ Projects a vector on to constraint surface using Newton iteration. """
    converged = False
    iters = 0
    pos_0 = pos * 1.
    if isinstance(dc_dpos_prev, tuple):
        dc_dpos_prev, prod_chol_prev = dc_dpos_prev
    else:
        prod_chol_prev = la.cho_factor(dc_dpos_prev.dot(dc_dpos_prev.T))
    while not converged and iters < max_iters:
        c = constr_func(pos)
        pos -= dc_dpos_prev.T.dot(la.cho_solve(prod_chol_prev, c))
        converged = np.all(np.abs(c) < tol)
        iters += 1
    if iters >= max_iters and scipy_opt_fallback and constr_jacob:
        pos_n = lambda l: pos_0 - dc_dpos_prev.T.dot(l)
        func = lambda l: constr_func(pos_n(l))
        jacob = lambda l: -dc_dpos_prev.dot(constr_jacob(pos_n(l), False).T)
        sol = opt.root(fun=func, x0=np.zeros_like(c), method='hybr', jac=jacob,
                       tol=tol, options={'maxfev': max_iters})
        if sol.success:
            pos = pos_0 - dc_dpos_prev.T.dot(sol.x)
        else:
            raise ExceededMaxItersError('scipy.root(hybrj)', max_iters, tol,
                                        np.max(np.abs(sol.fun[-1])))
    elif iters >= max_iters:
        raise ExceededMaxItersError('numpy symmetric-Newton', max_iters, tol,
                                    np.max(np.abs(c)))


def project_onto_nullspace(vct, mtx):
    """ Projects a vector on to the nullspace of a matrix. """
    if isinstance(mtx, tuple):
        mtx, mtx_prod_chol = mtx
    else:
        mtx_prod_chol = la.cho_factor(mtx.dot(mtx.T))
    mtx_vct = mtx.dot(vct)
    mtx_prod_inv_mtx_vct = la.cho_solve(mtx_prod_chol, mtx_vct)
    vct -= mtx.T.dot(mtx_prod_inv_mtx_vct)
