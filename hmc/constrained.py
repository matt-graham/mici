# -*- coding: utf-8 -*-
""" Numpy implementations of constrained Euclidean metric HMC samplers. """

import logging
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from .unconstrained import IsotropicHmcSampler
from .base import DynamicsError

autograd_available = True
try:
    from autograd import jacobian
except ImportError:
    autograd_available = False

logger = logging.getLogger(__name__)


class ConvergenceError(DynamicsError):
    """Exception raised when iterative solver fails to converge."""

    def __init__(self, solver, max_iters, tol, error):
        super(ConvergenceError, self).__init__(
            'Solver ({0}) did not converge.\n'
            'Maximum number of iterations: {1}\n'
            'Maximum error for convergence: {2}\n'
            'Last maximum error: {3}'
            .format(solver, max_iters, tol, error))


def check_reverse_in_tol(pos, pos_rev, cache, tol):
    """Helper function for checking constrained step is reversible."""
    # Estimate maximum absolute distance in *constraint* space
    c_dist = np.max(abs(cache['dc_dpos'].dot(pos - pos_rev)))
    # Maximum absolute distance in position space -
    # as a heuristic check if this is within sqrt(tol) where tol is the
    # convergence tolerance for projection step in constraint space, see e.g.
    # > The Devil is in the Detail: Hints for Practical Optimisation
    # > Christensen, Hurn and Lindsey (2008)
    p_dist = np.max(abs(pos - pos_rev))
    if c_dist > tol or p_dist > tol**0.5:
        raise NonReversibleStepError(c_dist, p_dist, tol)


class NonReversibleStepError(DynamicsError):
    """Exception raised when reversed step does return to original position."""

    def __init__(self, c_dist, p_dist, tol):
        super(DynamicsError, self).__init__(
            'Non-reversible geodesic step.\n'
            'Maximum absolute distance between positions: {0}\n'
            'Estimated maximum absolute distance in constraint space: {1}\n'
            'Distance tolerance: {2}'
            .format(p_dist, c_dist, tol))


class ConstrainedIsotropicHmcSampler(IsotropicHmcSampler):
    """
    Constrained HMC sampler with identity mass matrix.

    Generates MCMC samples on a manifold embedded in Euclidean space, specified
    as the solution set to `constr_func(pos) = 0` for some vector function of
    the position state `pos`. Based on method presented in [1].

    A hybrid SHAKE [2] / RATTLE [3] integration scheme is used to simulate the
    constrained Hamiltonian dynamics.

    References
    ----------
    [1] Brubaker, Salzmann and Urtasun. A family of MCMC methods on implicitly
        defined manifolds. Proceedings of International Conference on
        Artificial Intelligence and Statistics, 2012.
    [2] Ryckaert, Ciccotti and Berendsen. Numerical integration of Cartesian
        equations of motion of a system with constraints : molecular dynamics
        of n-alkanes. Journal of Computational Physics, 1977
    [3] Andersen. RATTLE: A "velocity" version of the SHAKE algorithm for
        molecular dynamics calculations. Journal of Computational Physics, 1983
    """

    def __init__(self, energy_func, constr_func, energy_grad=None,
                 constr_jacob=None, prng=None, mom_resample_coeff=1.,
                 dtype=np.float64, tol=1e-8, max_iters=100,
                 check_reverse=True, scipy_opt_fallback=True):
        """
        Parameters
        ----------
        energy_func : function(vector, dictionary]) -> scalar
            Function which returns energy (marginal negative log density) of a
            position state. Should also accept a (potentially empty) `cache`
            dictionary argument which correspond to cached intermediate results
            which are fully determined by position state. If no `energy_grad`
            is provided it will be attempted to use Autograd to calculate the
            gradient and so this function should be then be defined using the
            `autograd.numpy` interface.
        constr_func : function(vector) -> vector
            Function which returns the vector-valued constraint function which
            defines the constraint manifold to sample on (i.e. the set of
            position states for which the constraint function is equal to the
            zero vector). If no `constr_jacob` is provided it will be attempted
            to use Autograd to calculate the Jacobian and so this function
            should be then be defined using the `autograd.numpy` interface.
        energy_grad : function(vector, dictionary) -> vector or None
            Function which returns gradient of energy function at a position
            state. Should also accept a (potentially empty) `cache` dictionary
            argument which correspond to cached intermediate results which are
            fully determined by position state. If not provided it will be
            attempted to use Autograd to create a gradient function from the
            provided `energy_func`. In this case any cached results will be
            ignored when calculating the gradient as there is no information
            available of how to propagate the derivatives through them.
        constr_jacob : function(vector, boolean) -> dict or None
            Function calculating the constraint Jacobian, the matrix of partial
            derivatives of constraint function with respect to the position
            state, with dimensions dim(pos) * dim(constr_func(pos)). The
            function should return a dictionary with entry with key `dc_dpos`
            corresponding to the constraint Jacobian. Optionally if the second
            is True it may also calculate the Cholesky decomposition of the
            Gram matrix `dc_dpos.dot(dc_dpos.T)` which should then be stored in
            the returned dictionary with key `gram_chol` in the same format as
            that returned by `scipy.linalg.cho_factor` i.e. a tuple with first
            element the calculated Cholesky decomposition and the second
            element a boolean value indicating if a lower triangular factor
            was computed (True) or upper triangular (False). If not provided it
            will be attempted to use Autograd to create a Jacobian function
            from the provided `constr_func`.
        prng : RandomState
            Pseudo-random number generator. If `None` a new Numpy RandomState
            instance is created.
        mom_resample_coeff : scalar in [0, 1]
            Coefficient specifying degree with which to resample momentum
            after each Metropolis-Hastings Hamiltonian move. If equal to 1 the
            momentum will be independently sampled from its conditional given
            the current position, while if equal to 0 the momentum will be
            left at the value returned from the M-H step. Values in between
            these two extremes will lead to a new momentum which is a weighted
            sun of an independent draw from the conditional and the current
            momentum state.
        dtype : Numpy data type
            Floating point data type to use in arrays storing states.
        tol : float
            Convergence tolerance for iterative solution of projection on to
            constraint manifold - convergence is assumed when
            `max(abs(constr_func(pos_n))) < tol`.
        max_iters : integer
            Maximum number of iterations to perform when solving non-linear
            projection step - if iteration count exceeds this without a
            solution being found a `ConvergenceError` exception is raised.
        check_reverse : boolean
            Whether to check reverse step and projection on to constraint
            manifold returns to the original position (to within floating
            point precision using `numpy.allclose`). This adds around a ~25%
            run time overhead (mainly in re-running the projection for the
            reverse step) but ensures reversibility of the proposal.
            Non-reversible steps are recorded to the logger output so it is
            possible to monitor how often non-reversible steps occur in a
            given sampling run.
        scipy_opt_fallback : boolean
            Whether to fallback to `scipy.opt.root('hybrj')` solver if initial
            quasi-Newton iteration does not converge when solving for
            projection of position on to constraint manifold.
        """
        super(ConstrainedIsotropicHmcSampler, self).__init__(
            energy_func, energy_grad, prng, mom_resample_coeff, dtype)
        self.constr_func = constr_func
        if constr_jacob is None and autograd_available:
            self.constr_jacob = wrap_constr_jacob_func(jacobian(constr_func))
        elif constr_jacob is None and not autograd_available:
            raise ValueError('Autograd not available therefore constraint '
                             'Jacobian must be provided.')
        else:
            self.constr_jacob = constr_jacob
        self.tol = tol
        self.max_iters = max_iters
        self.check_reverse = check_reverse
        self.scipy_opt_fallback = scipy_opt_fallback

    def project_position(self, pos, cache):
        return project_onto_constraint_surface(
            pos, cache, constr_func=self.constr_func, tol=self.tol,
            max_iters=self.max_iters,
            scipy_opt_fallback=self.scipy_opt_fallback,
            constr_jacob=self.constr_jacob)

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        if not any(cache):
            cache.update(self.constr_jacob(pos))
        mom_half = mom - 0.5 * dt * self.energy_grad(pos, cache)
        pos_n = pos + dt * mom_half
        pos_n = self.project_position(pos_n, cache)
        mom_half = (pos_n - pos) / dt
        pos = pos_n
        cache = self.constr_jacob(pos)
        for s in range(n_step):
            mom_half -= dt * self.energy_grad(pos, cache)
            pos_n = pos + dt * mom_half
            pos_n = self.project_position(pos_n, cache)
            cache_n = self.constr_jacob(pos_n)
            mom_half_n = (pos_n - pos) / dt
            if self.check_reverse:
                pos_rev = pos_n - dt * mom_half_n
                pos_rev = self.project_position(pos_rev, cache_n)
                check_reverse_in_tol(pos, pos_rev, cache, self.tol)
            pos, cache, mom_half = pos_n, cache_n, mom_half_n
        mom = mom_half - 0.5 * dt * self.energy_grad(pos, cache)
        mom = project_onto_nullspace(mom, cache)
        return pos, mom, cache

    def sample_independent_momentum_given_position(self, pos, cache):
        n_dim = pos.shape[0]
        if not any(cache):
            cache.update(self.constr_jacob(pos))
        mom = self.prng.normal(size=n_dim).astype(self.dtype)
        mom = project_onto_nullspace(mom, cache)
        return mom


class RattleConstrainedIsotropicHmcSampler(ConstrainedIsotropicHmcSampler):
    """
    Constrained HMC sampler with identity mass matrix.

    Generates MCMC samples on a manifold embedded in Euclidean space, specified
    as the solution set to `constr_func(pos) = 0` for some vector function of
    the position state `pos`. Based on method presented in [1].

    A RATTLE [2] integration scheme is used to simulate the constrained
    Hamiltonian dynamics.

    References
    ----------
    [1] Brubaker, Salzmann and Urtasun. A family of MCMC methods on implicitly
        defined manifolds. Proceedings of International Conference on
        Artificial Intelligence and Statistics, 2012.
    [2] Andersen. RATTLE: A "velocity" version of the SHAKE algorithm for
        molecular dynamics calculations. Journal of Computational Physics, 1983
    """

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        if not any(cache):
            cache.update(self.constr_jacob(pos))
        for s in range(n_step):
            mom_half = mom - 0.5 * dt * self.energy_grad(pos, cache)
            pos_n = pos + dt * mom_half
            pos_n = self.project_position(pos_n, cache)
            cache_n = self.constr_jacob(pos_n)
            mom_half_n = (pos_n - pos) / dt
            mom_n = mom_half_n - 0.5 * dt * self.energy_grad(pos_n, cache_n)
            mom_n = project_onto_nullspace(mom_n, cache_n)
            if self.check_reverse:
                mom_half_rev = mom_n + 0.5 * dt * self.energy_grad(
                    pos_n, cache_n)
                pos_rev = pos_n - dt * mom_half_rev
                pos_rev = self.project_position(pos_rev, cache_n)
                check_reverse_in_tol(pos, pos_rev, cache, self.tol)
            pos, cache, mom = pos_n, cache_n, mom_n
        return pos, mom, cache


class GbabConstrainedIsotropicHmcSampler(ConstrainedIsotropicHmcSampler):
    """
    Constrained HMC sampler with identity mass matrix.

    Generates MCMC samples on a manifold embedded in Euclidean space, specified
    as the solution set to `constr_func(pos) = 0` for some vector function of
    the position state `pos`. Based on method presented in [1].

    A Geodesic-BAB integration scheme is used to simulate the constrained
    Hamiltonian dynamics.

    References
    ----------
    [1] Brubaker, Salzmann and Urtasun. A family of MCMC methods on implicitly
        defined manifolds. Proceedings of International Conference on
        Artificial Intelligence and Statistics, 2012.
    [2] Leimkuhler and Matthews. Efficient molecular dynamics using geodesic
        integration and solve--solute splitting. Proceedings of the Royal
        Society (A), 2016.
    """

    def __init__(self, energy_func, constr_func, energy_grad=None,
                 constr_jacob=None, prng=None, mom_resample_coeff=1.,
                 dtype=np.float64, tol=1e-8, max_iters=100, check_reverse=True,
                 scipy_opt_fallback=True, n_inner_update=10):
        super(GbabConstrainedIsotropicHmcSampler, self).__init__(
            energy_func, constr_func, energy_grad, constr_jacob, prng,
            mom_resample_coeff, dtype, tol, max_iters, check_reverse,
            scipy_opt_fallback)
        self.n_inner_update = n_inner_update

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        if not any(cache):
            cache.update(self.constr_jacob(pos))
        dt_inner = dt / self.n_inner_update
        for s in range(n_step):
            mom_half = mom - 0.5 * dt * self.energy_grad(pos, cache)
            mom_half = project_onto_nullspace(mom_half, cache)
            for i in range(self.n_inner_update):
                pos_n = pos + dt_inner * mom_half
                pos_n = self.project_position(pos_n, cache)
                cache_n = self.constr_jacob(pos_n)
                mom_half_n = (pos_n - pos) / dt_inner
                mom_half_n = project_onto_nullspace(mom_half_n, cache_n)
                if self.check_reverse:
                    pos_rev = pos_n - dt_inner * mom_half_n
                    pos_rev = self.project_position(pos_rev, cache_n)
                    check_reverse_in_tol(pos, pos_rev, cache, self.tol)
                pos, cache, mom_half = pos_n, cache_n, mom_half_n
            mom = mom_half - 0.5 * dt * self.energy_grad(pos, cache)
            mom = project_onto_nullspace(mom, cache)
        return pos, mom, cache


class LfGbabConstrainedIsotropicHmcSampler(GbabConstrainedIsotropicHmcSampler):
    """
    Constrained HMC sampler with identity mass matrix.

    Generates MCMC samples on a manifold embedded in Euclidean space, specified
    as the solution set to `constr_func(pos) = 0` for some vector function of
    the position state `pos`. Based on method presented in [1].

    A leapfrog Geodesic-BAB integration scheme is used to simulate the
    constrained Hamiltonian dynamics.

    References
    ----------
    [1] Brubaker, Salzmann and Urtasun. A family of MCMC methods on implicitly
        defined manifolds. Proceedings of International Conference on
        Artificial Intelligence and Statistics, 2012.
    [2] Leimkuhler and Matthews. Efficient molecular dynamics using geodesic
        integration and solve--solute splitting. Proceedings of the Royal
        Society (A), 2016.
    """

    def simulate_dynamic(self, n_step, dt, pos, mom, cache={}):
        if not any(cache):
            cache.update(self.constr_jacob(pos))
        dt_inner = dt / self.n_inner_update
        for s in range(n_step):
            if s == 0:
                mom_half = mom - 0.5 * dt * self.energy_grad(pos, cache)
            else:
                mom_half = mom_half - dt * self.energy_grad(pos, cache)
            mom_half = project_onto_nullspace(mom_half, cache)
            for i in range(self.n_inner_update):
                pos_n = pos + dt_inner * mom_half
                pos_n = self.project_position(pos_n, cache)
                cache_n = self.constr_jacob(pos_n)
                mom_half_n = (pos_n - pos) / dt_inner
                mom_half_n = project_onto_nullspace(mom_half_n, cache_n)
                if self.check_reverse:
                    pos_rev = pos_n - dt_inner * mom_half_n
                    pos_rev = self.project_position(pos_rev, cache_n)
                    check_reverse_in_tol(pos, pos_rev, cache, self.tol)
                pos, cache, mom_half = pos_n, cache_n, mom_half_n
        mom = mom_half - 0.5 * dt * self.energy_grad(pos, cache)
        mom = project_onto_nullspace(mom, cache)
        return pos, mom, cache


def wrap_constr_jacob_func(constr_jacob):
    """Convenience function to wrap function calculating constraint Jacobian.

    Produces a function which returns a dictionary with entry with key
    `dc_dpos` corresponding to calculated constraint Jacobian and optionally
    also entry with key `gram_chol` for Cholesky decomposition of Gram matrix
    if keyword argument `calc_gram_chol` is True.
    """

    def constr_jacob_wrapper(pos, calc_gram_chol=True):
        jacob = constr_jacob(pos)
        cache = {'dc_dpos': jacob}
        if calc_gram_chol:
            cache['gram_chol'] = la.cho_factor(jacob.dot(jacob.T))
        return cache

    return constr_jacob_wrapper


def project_onto_constraint_surface(pos, cache, constr_func, tol=1e-8,
                                    max_iters=100, scipy_opt_fallback=True,
                                    constr_jacob=None):
    """ Projects a vector on to constraint surface using Newton iteration.

    Parameters
    ----------
    pos : vector
        Position state to project on to constraint manifold (usually result
        of a unconstrained Hamiltonian dynamics step).
    cache : dictionary
        Dictionary of cached constraint Jacobian results for *previous*
        position state (i.e. position state before unconstrained step).
    constr_func : function(vector) -> vector
        Function which returns the vector-valued constraint function which
        defines the constraint manifold to sample on (i.e. the set of position
        states for which the constraint function is equal to the zero vector).
    tol : float
        Convergence tolerance for iterative solution of projection on to
        constraint manifold - convergence is assumed when
        `max(abs(constr_func(pos_n))) < tol`.
    max_iters : integer
        Maximum number of iterations to perform when solving non-linear
        projection step - if iteration count exceeds this without a
        solution being found a `ConvergenceError` exception is raised.
    scipy_opt_fallback : boolean
        Whether to fallback to `scipy.opt.root('hybrj')` solver if initial
        quasi-Newton iteration does not converge.
    constr_jacob : function(vector, boolean) -> dict
        Function calculating the constraint Jacobian, the matrix of partial
        derivatives of constraint function with respect to the position
        state, with dimensions dim(pos) * dim(constr_func(pos)). The
        function should return a dictionary with entry with key `dc_dpos`
        corresponding to the constraint Jacobian. Optionally if the second is
        True it may also calculate the Cholesky decomposition of the Gram
        matrix `dc_dpos.dot(dc_dpos.T)` which should then be stored in the
        returned dictionary with key `gram_chol`.

    Returns
    -------
    pos_n : vector
        Updated position state on constraint manifold
            `pos_n = pos - dc_dpos_prev.T.dot(l)`
        where `l` is chosen such that
            `max(abs(constr_func(pos_n)) < tol`

    Raises
    ------
    ConvergenceError :
        Raised when Newton iteration (and fallback if enabled) do not converge
        within `max_iters`.
    """
    converged = False
    diverging = False
    iters = 0
    pos_0 = pos * 1.
    dc_dpos_prev = cache['dc_dpos']
    if 'gram_chol' not in cache:
        cache['gram_chol'] = la.cho_factor(dc_dpos_prev.dot(dc_dpos_prev.T))
    gram_chol_prev = cache['gram_chol']
    while not converged and not diverging and iters < max_iters:
        c = constr_func(pos)
        if np.any(np.isinf(c)) or np.any(np.isnan(c)):
            diverging = True
            break
        pos -= dc_dpos_prev.T.dot(la.cho_solve(gram_chol_prev, c))
        converged = np.all(np.abs(c) < tol)
        iters += 1
    if diverging:
        logger.info('Quasi-Newton iteration diverged.')
    if not converged and scipy_opt_fallback and constr_jacob:
        logger.info(
            'Quasi-Newton iteration did not converge within max_iters.\n'
            'Last max error: {0}\n'.format(np.max(np.abs(c))) +
            'Falling back to scipy.optimize.root(hybrj).')

        def pos_n(l):
            return pos_0 - dc_dpos_prev.T.dot(l)

        def func(l):
            return constr_func(pos_n(l))

        def jacob(l):
            return -dc_dpos_prev.dot(
                constr_jacob(pos_n(l), False)['dc_dpos'].T)

        # root `tol` parameter is on function inputs not outputs (as used in
        # preceding Newton iteration) therefore use square root of supplied tol
        # see e.g.
        # > The Devil is in the Detail: Hints for Practical Optimisation
        # > Christensen, Hurn and Lindsey (2008)
        sol = opt.root(fun=func, x0=np.zeros_like(c), method='hybr', jac=jacob,
                       tol=tol**0.5, options={'maxfev': max_iters})
        if sol.success:
            pos = pos_0 - dc_dpos_prev.T.dot(sol.x)
        else:
            raise ConvergenceError('scipy.optimize.root(hybrj)', max_iters,
                                   tol, np.max(np.abs(sol.fun[-1])))
    elif not converged:
        raise ConvergenceError('numpy symmetric Quasi-Newton', max_iters, tol,
                               np.max(np.abs(c)))
    return pos


def project_onto_nullspace(mom, cache):
    """ Projects a momentum on to the nullspace of the constraint Jacobian.

    Parameters
    ----------
    mom : vector
        Momentum state to project.
    cache : dictionary
        Dictionary of cached constraint Jacobian results.

    Returns
    -------
    mom : vector
        Momentum state projected into nullspace of constraint Jacobian.
    """
    dc_dpos = cache['dc_dpos']
    if 'gram_chol' not in cache:
        cache['gram_chol'] = la.cho_factor(dc_dpos.dot(dc_dpos.T))
    gram_chol = cache['gram_chol']
    dc_dpos_mom = dc_dpos.dot(mom)
    gram_inv_dc_dpos_mom = la.cho_solve(gram_chol, dc_dpos_mom)
    dc_dpos_pinv_dc_dpos_mom = dc_dpos.T.dot(gram_inv_dc_dpos_mom)
    mom -= dc_dpos_pinv_dc_dpos_mom
    return mom
