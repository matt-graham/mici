# Hamiltonian Monte Carlo

Implementations of various Hamiltonian dynamics based Markov chain Monte Carlo
(MCMC) samplers in Python. A modular design is used to as far as possible
allowing mixing and matching elements of different proposed extensions to the
original Hybrid Monte Carlo algorithm proposed in Duane et al. (1987).

## Implemented methods

Samplers

  * Static integration time with Metropolis sampling (Duane et al., 1987)
  * Random integration time with Metropolis sampling (Mackenzie, 1989)
  * Correlated momentum updates in Metropolis samplers (Horowitz, 1991)
  * Dynamic integration time with multinomial sampling (Betancourt, 2017)

Hamiltonian systems

  * Euclidean-metric systems - isotropic, diagonal and dense metrics
  * Riemannian-metric systems (Girolami and Calderhead, 2011) inluding the
    log density Hessian based *SoftAbs* metric (Betancourt, 2013)
  * Euclidean-metric systems subject to holonomic constraints (Hartmann and
    Schütte, 2005; Brubaker, Salzmann and Urtasun, 2012; Lelièvre, Rousset and
    Stoltz, 2018) and for inference in differentiable generative models when 
    conditioning on observed outputs (Graham and Storkey, 2017a)

Numerical integrators

  * Explicit leapfrog for separable Hamiltonian systems
  * Implicit leapfrog for non-separable Hamiltonian systems
  * Geodesic leapfrog for constrained Hamiltonian systems
  * 'Split' leapfrog for Hamiltonian systems with an analytically tractable
    component for which the exact flow can be solved (Shahbaba et al., 2014)


## Installation

To install and use the package the minimal requirements are a Python 3.6+
environment with [NumPy](http://www.numpy.org/) (tested with v1.15.0) and
[SciPy](https://www.scipy.org) (tested with v1.1.0) installed.

From a local clone of the repository run `python setup.py install` to install
the package in the current Python environment.

## Optional dependencies

  * [Autograd](https://github.com/HIPS/autograd): if available will be used to 
    automatically compute the required derivatives of the model functions 
    (providing they are specified using functions from the `autograd.numpy` and 
    `autograd.scipy` interfaces).
  * [tqdm](https://github.com/tqdm/tqdm): if available a simple progress bar 
    will be shown during sampling.
  * [Arviz](https://arviz-devs.github.io/arviz/index.html#): if available 
    outputs of a sampling run can be returned in an `arviz.InferenceData` 
    container object, allowing straightforward use of the extensive Arviz 
    visualisation and diagnostic functionality.
  * [multiprocess](https://github.com/uqfoundation/multiprocess) and 
    [dill](https://github.com/uqfoundation/dill): if available
    `multiprocess.Pool` will be used in preference to the in-built
    `mutiprocessing.Pool` for parallelisation as `multiprocess` supports
    serialisation ( via dill) of a much wider range of types, including of
    Autograd generated functions.
  * [RandomGen](https://github.com/bashtage/randomgen): if available the
    `Xorshift1024` random number generator will be used when running multiple
    chains in parallel, with the `jump` method of the object used to 
    reproducibly generate independent substreams.

## Example usage

A simple complete example of using the package to sample from a multivariate
Gaussian distribution with randomly generated parameters is given below. Here an
isotropic Euclidean metric Hamiltonian system is used (corresponding to a
isotropic covariance Gaussian marginal distribution on the momenta) with the
dynamic integration time HMC implementation described in Betancourt (2017),
which is a extension of the NUTS algorithm (Hoffman and Gelman, 2014).

```python
import hmc
import autograd.numpy as np

# Generate random precision and mean parameters for a Gaussian
n_dim = 50
rng = np.random.RandomState(seed=1234)
rnd_eigvec, _ = np.linalg.qr(rng.normal(size=(n_dim, n_dim)))
rnd_eigval = np.exp(rng.normal(size=n_dim) * 2)
prec = (rnd_eigvec / rnd_eigval) @ rnd_eigvec.T
mean = rng.normal(size=n_dim)

# Deine potential energy (negative log density) for the Gaussian target
# distribution (gradient will be automatically calculated using autograd)
def pot_energy(pos):
    pos_minus_mean = pos - mean
    return 0.5 * pos_minus_mean @ prec @ pos_minus_mean

# Specify Hamiltonian system with isotropic Gaussian kinetic energy
system = hmc.systems.EuclideanMetricSystem(pot_energy)

# Hamiltonian is separable therefore use explicit leapfrog integrator
integrator = hmc.integrators.LeapfrogIntegrator(system, step_size=0.15)

# Use dynamic integration-time HMC implementation with multinomial 
# sampling from trajectories
sampler = hmc.samplers.DynamicMultinomialHMC(system, integrator, rng)

# Sample an initial position from zero-mean isotropic Gaussian
init_pos = rng.normal(size=n_dim)

# Sample a Markov chain with 1000 transitions
chains, chain_stats = sampler.sample_chain(1000, init_pos)

# Print RMSE in mean estimate
mean_rmse = np.mean((chains['pos'].mean(0) - mean)**2)**0.5
print(f'Mean estimate RMSE: {mean_rmse}')

# Print average acceptance probability
mean_accept_prob = chain_stats['accept_prob'].mean()
print(f'Mean accept prob: {mean_accept_prob:0.2f}')
```

## References

  1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
     Hybrid Monte Carlo. *Physics letters B*, 195(2), pp.216-222.
  2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
     *Physics Letters B*, 226(3-4), pp.369-371.
  3. Horowitz, A.M., 1991. A generalized guided Monte Carlo algorithm.
     *Physics Letters  B*, 268(CERN-TH-6172-91), pp.247-252.
  4. Neal, R. M., 1994. An improved acceptance procedure for the hybrid Monte
     Carlo algorithm. *Journal of Computational Physics*, 111:194–203.
  5. Hartmann, C. and Schütte, C., 2005. A constrained hybrid Monte‐Carlo
     algorithm and the problem of calculating the free energy in several
     variables. *ZAMM ‐ Journal of Applied Mathematics and Mechanics*, 85(10),
     pp.700-710.
  6. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
     *Handbook of Markov Chain Monte Carlo*, 2(11), p.2.
  7. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
     Hamiltonian Monte Varlo methods. *Journal of the Royal Statistical Society:
     Series B (Statistical Methodology)*, 73(2), pp.123-214.
  8. Brubaker, M., Salzmann, M. and Urtasun, R., 2012. A family of MCMC methods
     on implicitly defined manifolds. In *Artificial intelligence and statistics*
     (pp. 161-172).
 9.  Betancourt, M., 2013. A general metric for Riemannian manifold Hamiltonian
     Monte Carlo. In *Geometric science of information* (pp. 327-334).
 10. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler: adaptively
     setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine
     Learning Research*, 15(1), pp.1593-1623.
 11. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014.
     Split Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
 12. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
     *arXiv preprint arXiv:1701.02434*.
 13. Graham, M.M. and Storkey, A.J., 2017a. Asymptotically exact inference in 
     differentiable generative models. *Electronic Journal of Statistics*, 
     11(2), pp.5105-5164.
 14. Lelièvre, T., Rousset, M. and Stoltz, G., 2018. Hybrid Monte Carlo methods
     for sampling probability measures on submanifolds. *arXiv preprint
     1807.02356*.
