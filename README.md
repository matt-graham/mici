<img src='images/mici-logo-rectangular.svg' width='400px'/>

**Mici** is a Python package providing implementations of Markov chain Monte Carlo (MCMC) methods for approximate inference in probabilistic models, with a particular focus on MCMC methods based on simulating Hamiltonian dynamics on a manifold.

## Features

Key features include

  * implementations of MCMC methods for sampling from distributions on embedded manifolds implicitly-defined by a constraint equation and distributions on Riemannian manifolds with a user-specified metric,
  * a modular design allowing use of a wide range of inference algorithms by mixing and matching different components, making it easy for users to extend the package and use within their own code, 
  * computational efficient inference via transparent caching of the results of expensive operations and intermediate results calculated in derivative computations allowing later reuse without recalculation, 
  * memory efficient inference for large models by memory-mapping chains to disk, allowing long runs on large models without hitting memory issues.

## Installation

To install and use Mici the minimal requirements are a Python 3.6+
environment with [NumPy](http://www.numpy.org/) (tested with v1.15.0) and
[SciPy](https://www.scipy.org) (tested with v1.1.0) installed.

From a local clone of the repository run `python setup.py install` to install
the package in the current Python environment.

If available in the installed Python environment the following additional packages provide extra functionality and features

  * [Autograd](https://github.com/HIPS/autograd): if available will be used to 
    automatically compute the required derivatives of the model functions 
    (providing they are specified using functions from the `autograd.numpy` and 
    `autograd.scipy` interfaces).
  * [tqdm](https://github.com/tqdm/tqdm): if available a simple progress bar 
    will be shown during sampling.
  * [Arviz](https://arviz-devs.github.io/arviz/index.html#): if available 
    outputs of a sampling run can be converted to an `arviz.InferenceData` 
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

## Why Mici?

Mici is named for [Augusta 'Mici' Teller](https://en.wikipedia.org/wiki/Augusta_H._Teller), who along with [Arriana Rosenbluth](https://en.wikipedia.org/wiki/Arianna_W._Rosenbluth) developed the code for the [MANIAC I](https://en.wikipedia.org/wiki/MANIAC_I) computer used in the seminal paper [*Equations of State Calculations by Fast Computing Machines*](https://doi.org/10.1063%2F1.1699114) which introduced the first example of a Markov chain Monte Carlo method. 


## Related projects

Other Python packages for performing MCMC inference include [PyMC3](https://github.com/pymc-devs/pymc3), [PyStan](https://github.com/stan-dev/pystan) (the Python interface to [Stan](http://mc-stan.org/)), [Pyro](https://github.com/pyro-ppl/pyro) / [NumPyro](https://github.com/pyro-ppl/numpyro), [TensorFlow Probability](https://github.com/tensorflow/probability), [emcee](https://github.com/dfm/emcee) and [Sampyl](https://github.com/mcleonard/sampyl).

Unlike PyMC3, PyStan, (Num)Pyro and TensorFlow Probability which are complete probabilistic programming frameworks including functionality for definining a probabilistic model / program, but like emcee and Sampyl, Mici is solely focussed on providing implementations of inference algorithms, with the user expected to be able to define at a minimum a function specifying the negative log (unnormalised) density of the distribution of interest. 

Further while PyStan, (Num)Pyro and TensorFlow Probability all push the sampling loop into external compiled non-Python code, in mici the sampling loop is run directly within Python. This has the consequence that for small models in which the negative log density of the target distribution and other model functions are cheap to evaluate, the interpreter overhead in iterating over the chains in Python can dominate the computational cost, making sampling much slower than packages which outsource the sampling loop to a efficient compiled implementation.

 ## Overview of package
 
 The three main user-facing modules within the `mici` package are the `systems`, `integrators` and `samplers` modules:
 
 [`mici.systems`](mici/systems.py) - Hamiltonian systems encapsulating model functions and their derivatives

   * `EuclideanMetricSystem` - systems with a metric on the position space with a constant matrix representation,
   * `GaussianEuclideanMetricSystem` - systems in which the target distribution is defined by a density with respect to the standard Gaussian measure on the position space allowing analytically solving for flow corresponding to the quadratic components of Hamiltonian (Shahbaba et al., 2014),
   * `RiemannianMetricSystem` - systems with a metric on the position space with a position-dependent matrix representation (Girolami and Calderhead, 2011),
   * `SoftAbsRiemannianMetricSystem`  - system with *SoftAbs* eigenvalue-regularised Hessian of negative log target density as metric matrix representation (Betancourt, 2013),
   * `DenseConstrainedEuclideanMetricSystem` - Euclidean-metric system subject to holonomic constraints (Hartmann and Schütte, 2005; Brubaker, Salzmann and Urtasun, 2012; Lelièvre, Rousset and Stoltz, 2018) with a dense constraint function Jacobian matrix,

[`mici.integrators`](mici/integrators.py) - symplectic integrators for Hamiltonian dynamics

  * `LeapfrogIntegrator` - explicit leapfrog (Störmer-Verlet) integrator for separable Hamiltonian systems (Leimkulher and Reich, 2004),
  * `ImplicitLeapfrogIntegrator` - implicit (or generalised) leapfrog integrator for non-separable Hamiltonian systems (Leimkulher and Reich, 2004),
  * `ConstrainedLeapfrogIntegrator` - constrained leapfrog integrator for Hamiltonian systems subject to holonomic constraints (Andersen, 1983; Leimkuhler and Reich, 1994)

[`mici.samplers`](mici/samplers.py) - MCMC samplers for peforming inference

  * `StaticMetropolisHMC` - Static integration time Hamiltonian Monte Carlo with Metropolis accept step (Duane et al., 1987),
  * `RandomMetropolisHMC` - Random integration time Hamiltonian Monte Carlo with Metropolis accept step (Mackenzie, 1989),
  * `DynamicMultinomialHMC` - Dynamic integration time Hamiltonian Monte Carlo with multinomial sampling from trajectory (Hoffman and Gelman, 2014; Betancourt, 2017).

## Example usage

A simple complete example of using the package to sample from a multivariate
Gaussian distribution with randomly generated parameters is given below. Here a Hamiltonian system with a Euclidean metric with identity matrix representation is used (corresponding to a standard normal distribution on the momenta) with a
dynamic integration time HMC implementation.

```Python
import mici
import autograd.numpy as np

# Generate random precision and mean parameters for a Gaussian
n_dim = 50
rng = np.random.RandomState(seed=1234)
rnd_eigvec, _ = np.linalg.qr(rng.normal(size=(n_dim, n_dim)))
rnd_eigval = np.exp(rng.normal(size=n_dim) * 2)
prec = (rnd_eigvec / rnd_eigval) @ rnd_eigvec.T
mean = rng.normal(size=n_dim)

# Define negative log density for the Gaussian target distribution (gradient 
# will be automatically calculated using autograd)
def neg_log_dens(pos):
    pos_minus_mean = pos - mean
    return 0.5 * pos_minus_mean @ prec @ pos_minus_mean

# Specify Hamiltonian system with default identity metric
system = mici.systems.EuclideanMetricSystem(neg_log_dens)

# Hamiltonian is separable therefore use explicit leapfrog integrator
integrator = mici.integrators.LeapfrogIntegrator(system, step_size=0.15)

# Use dynamic integration-time HMC implementation with multinomial 
# sampling from trajectories
sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)

# Sample an initial position from a zero-mean identity-covariance Gaussian
init_pos = rng.normal(size=n_dim)

# Sample a Markov chain with 1000 transitions
final_state, traces, chain_stats = sampler.sample_chain(1000, init_pos)

# Print RMSE in mean estimate
mean_rmse = np.mean((traces['pos'].mean(0) - mean)**2)**0.5
print(f'Mean estimate RMSE: {mean_rmse}')

# Print average acceptance probability
mean_accept_prob = chain_stats['accept_prob'].mean()
print(f'Mean accept prob: {mean_accept_prob:0.2f}')
```

## References

  1. Andersen, H.C., 1983. RATTLE: A “velocity” version of the SHAKE algorithm 
     for molecular dynamics calculations. *Journal of Computational Physics*, 
     52(1), pp.24-34.
  2. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
     Hybrid Monte Carlo. *Physics letters B*, 195(2), pp.216-222.
  3. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
     *Physics Letters B*, 226(3-4), pp.369-371.
  4. Horowitz, A.M., 1991. A generalized guided Monte Carlo algorithm.
     *Physics Letters  B*, 268(CERN-TH-6172-91), pp.247-252.
  5. Leimkuhler, B. and Reich, S., 1994. Symplectic integration of constrained 
     Hamiltonian systems. *Mathematics of Computation*, 63(208), pp.589-605.
  6. Leimkuhler, B. and Reich, S., 2004. Simulating Hamiltonian dynamics (Vol. 14). 
     *Cambridge University Press*.
  7. Hartmann, C. and Schütte, C., 2005. A constrained hybrid Monte‐Carlo
     algorithm and the problem of calculating the free energy in several
     variables. *ZAMM ‐ Journal of Applied Mathematics and Mechanics*, 85(10),
     pp.700-710.
  8. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
     Hamiltonian Monte Varlo methods. *Journal of the Royal Statistical Society:
     Series B (Statistical Methodology)*, 73(2), pp.123-214.
  9. Brubaker, M., Salzmann, M. and Urtasun, R., 2012. A family of MCMC methods
     on implicitly defined manifolds. In *Artificial intelligence and statistics*
     (pp. 161-172).
 10. Betancourt, M., 2013. A general metric for Riemannian manifold Hamiltonian
     Monte Carlo. In *Geometric science of information* (pp. 327-334).
 11. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler: adaptively
     setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine
     Learning Research*, 15(1), pp.1593-1623.
 12. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014.
     Split Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
 13. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
     *arXiv preprint arXiv:1701.02434*.
 14. Lelièvre, T., Rousset, M. and Stoltz, G., 2018. Hybrid Monte Carlo methods
     for sampling probability measures on submanifolds. *arXiv preprint
     1807.02356*.
