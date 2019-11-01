<img src='images/mici-logo-rectangular.svg' width='400px'/>

<p class='badges'>
  <a href="https://badge.fury.io/py/mici">
    <img src="https://badge.fury.io/py/mici.svg" alt="PyPI version">
  </a>
  <a href="https://matt-graham.github.io/mici/docs">
    <img src="https://img.shields.io/badge/API_docs-grey.svg" alt="API documentation">
  </a>
  <a href="https://zenodo.org/badge/latestdoi/52494384">
    <img src="https://zenodo.org/badge/52494384.svg" alt="DOI">
  </a>
</p>

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

The latest Mici release on PyPI can be installed in the current Python environment by running

```pip install mici```

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

Further while PyStan, (Num)Pyro and TensorFlow Probability all push the sampling loop into external compiled non-Python code, in Mici the sampling loop is run directly within Python. This has the consequence that for small models in which the negative log density of the target distribution and other model functions are cheap to evaluate, the interpreter overhead in iterating over the chains in Python can dominate the computational cost, making sampling much slower than packages which outsource the sampling loop to a efficient compiled implementation.

 ## Overview of package
 
API documentation for the package is available [here](https://matt-graham.github.io/mici/docs). The three main user-facing modules within the `mici` package are the `systems`, `integrators` and `samplers` modules:
 
 [`mici.systems`](https://matt-graham.github.io/mici/docs/systems.html) - Hamiltonian systems encapsulating model functions and their derivatives

   * `EuclideanMetricSystem` - systems with a metric on the position space with a constant matrix representation,
   * `GaussianEuclideanMetricSystem` - systems in which the target distribution is defined by a density with respect to the standard Gaussian measure on the position space allowing analytically solving for flow corresponding to the quadratic components of Hamiltonian (Shahbaba et al., 2014),
   * `RiemannianMetricSystem` - systems with a metric on the position space with a position-dependent matrix representation (Girolami and Calderhead, 2011),
   * `SoftAbsRiemannianMetricSystem`  - system with *SoftAbs* eigenvalue-regularised Hessian of negative log target density as metric matrix representation (Betancourt, 2013),
   * `DenseConstrainedEuclideanMetricSystem` - Euclidean-metric system subject to holonomic constraints (Hartmann and Schütte, 2005; Brubaker, Salzmann and Urtasun, 2012; Lelièvre, Rousset and Stoltz, 2018) with a dense constraint function Jacobian matrix,

[`mici.integrators`](https://matt-graham.github.io/mici/docs/integrators.html) - symplectic integrators for Hamiltonian dynamics

  * `LeapfrogIntegrator` - explicit leapfrog (Störmer-Verlet) integrator for separable Hamiltonian systems (Leimkulher and Reich, 2004),
  * `ImplicitLeapfrogIntegrator` - implicit (or generalised) leapfrog integrator for non-separable Hamiltonian systems (Leimkulher and Reich, 2004),
  * `ConstrainedLeapfrogIntegrator` - constrained leapfrog integrator for Hamiltonian systems subject to holonomic constraints (Andersen, 1983; Leimkuhler and Reich, 1994)

[`mici.samplers`](https://matt-graham.github.io/mici/docs/samplers.html) - MCMC samplers for peforming inference

  * `StaticMetropolisHMC` - Static integration time Hamiltonian Monte Carlo with Metropolis accept step (Duane et al., 1987),
  * `RandomMetropolisHMC` - Random integration time Hamiltonian Monte Carlo with Metropolis accept step (Mackenzie, 1989),
  * `DynamicMultinomialHMC` - Dynamic integration time Hamiltonian Monte Carlo with multinomial sampling from trajectory (Hoffman and Gelman, 2014; Betancourt, 2017).

## Example: sampling on a torus

<img src='images/torus-samples.gif' width='360px'/>

A simple complete example of using the package to compute approximate samples from a distribution on a two-dimensional torus embedded in a three-dimensional space is given below. The computed samples are visualised in the animation above. Here we use `autograd` to automatically construct functions to calculate the required derivatives (gradient of negative log density of target distribution and Jacobian of constraint function) and sample four chains in parallel using `multiprocess`.

```Python
from mici import systems, integrators, samplers
import autograd.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define fixed model parameters
R = 1.0  # toroidal radius ∈ (0, ∞)
r = 0.5  # poloidal radius ∈ (0, R)
α = 0.9  # density fluctuation amplitude ∈ [0, 1)

# Define constraint function such that the set {q : constr(q) == 0} is a torus
def constr(q):
    x, y, z = q.T
    return np.stack([((x**2 + y**2)**0.5 - R)**2 + z**2 - r**2], -1)

# Define negative log density for the target distribution on torus
# (with respect to 2D 'area' measure for torus)
def neg_log_dens(q):
    x, y, z = q.T
    θ = np.arctan2(y, x)
    ϕ = np.arctan2(z, x / np.cos(θ) - R)
    return np.log1p(r * np.cos(ϕ) / R) - np.log1p(np.sin(4*θ) * np.cos(ϕ) * α)

# Specify constrained Hamiltonian system with default identity metric
system = systems.DenseConstrainedEuclideanMetricSystem(neg_log_dens, constr)

# System is constrained therefore use constrained leapfrog integrator
integrator = integrators.ConstrainedLeapfrogIntegrator(system, step_size=0.2)

# Seed a random number generator
rng = np.random.RandomState(seed=1234)

# Use dynamic integration-time HMC implementation as MCMC sampler
sampler = samplers.DynamicMultinomialHMC(system, integrator, rng)

# Sample initial positions on torus using parameterisation (θ, ϕ) ∈ [0, 2π)²
# x, y, z = (R + r * cos(ϕ)) * cos(θ), (R + r * cos(ϕ)) * sin(θ), r * sin(ϕ)
n_chain = 4
θ_init, ϕ_init = rng.uniform(0, 2 * np.pi, size=(2, n_chain))
q_init = np.stack([
    (R + r * np.cos(ϕ_init)) * np.cos(θ_init),
    (R + r * np.cos(ϕ_init)) * np.sin(θ_init),
    r * np.sin(ϕ_init)], -1)

# Define function to extract variables to trace during sampling
def trace_func(state):
    x, y, z = state.pos
    return {'x': x, 'y': y, 'z': z}

# Sample four chains of 2500 samples in parallel
final_states, traces, stats = sampler.sample_chains(
    n_sample=2500, init_states=q_init, n_process=4, trace_funcs=[trace_func])

# Print average accept probability and number of integrator steps per chain
for c in range(n_chain):
    print(f"Chain {c}:")
    print(f"  Average accept prob. = {stats['accept_prob'][c].mean():.2f}")
    print(f"  Average number steps = {stats['n_step'][c].mean():.1f}")

# Visualise concatentated chain samples as animated 3D scatter plot   
fig = plt.figure(figsize=(4, 4))
ax = Axes3D(fig, [0., 0., 1., 1.], proj_type='ortho')
points_3d, = ax.plot(*(np.concatenate(traces[k]) for k in 'xyz'), '.', ms=0.5)
ax.axis('off')
for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
    set_lim((-1, 1))

def update(i):
    angle = 45 * (np.sin(2 * np.pi * i / 60) + 1)
    ax.view_init(elev=angle, azim=angle)
    return (points_3d,)

anim = animation.FuncAnimation(fig, update, frames=60, interval=100, blit=True)
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
