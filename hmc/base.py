# -*- coding: utf-8 -*-
""" Hamiltonian dynamics based MCMC samplers. """

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np

autograd_available = True
try:
    from autograd import grad
except ImportError:
    autograd_available = False


class AbstractHmcSampler(object):
    """ Abstract Hamiltonian Monte Carlo sampler base class. """

    def __init__(self, energy_func, energy_grad=None, prng=None,
                 mom_resample_coeff=1., dtype=np.float64):
        """
        Abstract HMC sampler constructor

        Parameters
        ----------
        energy_func : function(vector) -> scalar
            Function which returns energy (marginal negative log density) of a
            position state. If no `energy_grad` is provided and so `autograd`
            is to be used to calculate the gradient this function should be
            defined using the autograd.numpy interface.
        energy_grad : function(vector) -> vector or None
            Function which returns gradient of energy function at a position
            state. If not provided it will be attempted to use `autograd` to
            create a gradient function from the provided `energy_func`.
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
        """
        self.energy_func = energy_func
        if energy_grad is None and autograd_available:
            self.energy_grad = grad(energy_func)
        elif energy_grad is None and not autograd_available:
            raise ValueError('Autograd not available therefore energy gradient'
                             ' must be provided.')
        else:
            self.energy_grad = energy_grad
        self.prng = prng if prng is not None else np.random.RandomState()
        if mom_resample_coeff < 0 or mom_resample_coeff > 1:
                raise ValueError('Momentum resampling coefficient must be in '
                                 '[0, 1]')
        self.mom_resample_coeff = mom_resample_coeff
        self.dtype = dtype

    def kinetic_energy(self, pos, cache, mom):
        """
        Value of 'kinetic' energy term at provided state pair.

        The kinetic energy here is defined as the negative log conditional
        density on the momentum given the position.

        Parameters
        ----------
        pos : vector
            Position state.
        mom : vector
            Momentum state.
        cache : object
            Either a set of cached results which can be deterministically
            calculated from position state (e.g. from previous move) or `None`
            if dynamic does not make use of cached results or no cached results
            are yet available.
        """
        raise NotImplementedError()

    def simulate_dynamic(self, pos, mom, cache, dt, n_step):
        """
        Simulate Hamiltonian dynamics from a given state and return new state.

        Parameters
        ----------
        pos : vector
            Initial position state.
        mom : vector
            Initial momentum state.
        cache : object
            Either a set of cached results which can be deterministically
            calculated from position state (e.g. from previous move) or `None`
            if move does not make use of cached results or no cached results
            are yet available.
        dt : scalar
            Time step for discretised dynamics.
        n_step : positive integer
            Number of time steps to simulate discretised dynamics for.

        Returns
        -------
        pos_n : vector
            Final position state in simulated trajectory.
        mom_n : vector
            Final momentum state in simulated trajectory.
        cache_n : vector
            Any cached results associated with final position state for use in
            further moves.
        """
        raise NotImplementedError()

    def sample_independent_momentum_given_position(pos, cache):
        """
        Sample a momentum independently from the conditional given a position.

        Note: In some cases the momentum will be independently distributed of
        the positions and so will be resampled from its marginal which is
        equal to the conditional in this case. The position argument will be
        ignored in these cases.

        Parameters
        ----------
        pos : vector
            Position state.
        cache : object
            Either a set of cached results which can be deterministically
            calculated from position state (e.g. from previous move) or `None`
            if move does not make use of cached results or no cached results
            are yet available.

        Returns
        -------
        mom : vector
            Mmentum state sampled from conditional.
        """
        raise NotImplementedError()

    def resample_momentum(self, pos, cache, mom):
        """
        Resample momentum state leaving conditional given position invariant.

        Samples a new momentum state given provided momentum and position
        states, with the property that if the original state pair was
        distributed according to the target invariant density of the Markov
        chain, the new state pair consisting of the newly sampled momentum,
        and provided position will also be distributed according to the target
        density.

        More concretely the newly sampled momentum is a weighted sum of the
        provided momentum state and a new indepedent draw from the conditional
        density on the momentum given the position, with the relative weights
        given to the two components determined by the `self.mom_resample_coeff`
        property.

        Parameters
        ----------
        pos : vector
            Position state.
        cache : object
            Either a set of cached results which can be deterministically
            calculated from position state (e.g. from previous move) or `None`
            if move does not make use of cached results or no cached results
            are yet available.
        mom : vector
            Momentum state.

        Returns
        -------
        mom_n : vector
            Newly sampled momentum state.
        """
        if self.mom_resample_coeff == 1:
            return self.sample_independent_momentum_given_position(pos, cache)
        elif self.mom_resample_coeff == 0:
            return mom
        else:
            mom_i = self.sample_independent_momentum_given_position(pos, cache)
            return (self.mom_resample_coeff * mom_i +
                    (1. - self.mom_resample_coeff**2) * mom)

    def hamiltonian(self, pos, cache, mom):
        """
        Hamiltonian (negative log density) of a position-momentum state pair.

        Parameters
        ----------
        pos : vector
            Position state.
        cache : object
            Either a set of cached results which can be deterministically
            calculated from position state (e.g. from previous move) or `None`
            if dynamic does not make use of cached results or no cached results
            are yet available.
        mom : vector
            Momentum state.

        Returns
        -------
        scalar
            Hamiltonian value at specified state pair.
        """
        return self.energy_func(pos) + self.kinetic_energy(pos, mom)

    def get_samples(self, pos, dt, n_step_per_sample, n_sample, mom=None):
        """
        Run HMC sampler and return state samples.

        Parameters
        ----------
        pos : vector
            Initial position state.
        dt : scalar
            Time step for leapfrog integration.
        n_step_per_sample : positive integer or tuple (lower, upper)
            Number of time steps to simulate when proposing a new state pair.
            Either an integer in which case the same number of steps are taken
            for every sample, or a tuple of two integers with the first
            specifying the lower bound and second upper bound for the interval
            to draw the random number of leapfrog steps to take on each sample
            proposal.
        n_sample : positive integer
            Number of MCMC samples to return (including initial state).
        mom : vector
            Optional initial momentum state. If not provided randomly sampled.

        Returns
        -------
        pos_samples : 2D array (n_sample, n_dim)
            Array of MCMC position samples.
        mom_samples : 2D array (n_sample, n_dim)
            Array of MCMC momentum samples.
        acceptance_rate : scalar in [0, 1]
            Proportion of Hamiltonian dynamic proposals accepted.
        """
        n_dim = pos.shape[0]
        pos_samples, mom_samples = np.empty((2, n_sample, n_dim), self.dtype)
        if mom is None:
            mom = self.sample_independent_momentum_given_position(pos, cache)
        pos_samples[0], mom_samples[0], cache = pos, mom, None

        if is_instance(n_step_per_sample, tuple):
            randomise_steps = True
            step_interval_lower, step_interval_upper = n_step_per_sample
            assert step_interval_lower < step_interval_upper
            assert step_interval_lower > 0
        else:
            randomise_steps = False

        hamiltonian_c = self.hamiltonian(pos, cache, mom)
        n_reject = 0

        for s in range(1, n_samples):
            if randomise_steps:
                n_step_per_sample = self.prng.random_integers(
                    step_interval_lower, step_interval_upper)
            pos_p, mom_p, cache_p = self.simulate_dynamic(
                pos_samples[s-1], mom_samples[s-1], cache)
            hamiltonian_p = self.hamiltonian(pos_p, cache_p, mom_p)
            if self.prng.uniform() < np.exp(hamiltonian_c - hamiltonian_p):
                pos_samples[s], mom_samples[s], cache = pos_p, mom_p, cache_p
                hamiltonian_c = hamiltonian_p
            else:
                pos_samples[s] = pos_samples[s-1]
                # negate momentum on rejection to ensure reversibility
                mom_samples[s] = -mom_samples[s-1]
                n_reject += 1
            mom_samples[s] = self.resample_momentum(
                pos_samples[s], cache, mom_samples[s])
            if self.mom_resample_coeff != 0:
                hamiltonian_c = self.hamiltonian(pos_samples[s], cache,
                                                 mom_samples[s])

        return pos_samples, mom_samples, 1. - (n_reject * 1. / n_sample)
