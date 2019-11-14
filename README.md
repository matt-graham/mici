<img src='images/mici-logo-rectangular.svg' width='400px'/>

<p class='badges'>
  <a href="https://badge.fury.io/py/mici">
    <img src="https://badge.fury.io/py/mici.svg" alt="PyPI version"/>
  </a>
  <a href="https://matt-graham.github.io/mici/docs">
    <img src="https://img.shields.io/badge/API_docs-grey.svg" 
         alt="API documentation"/>
  </a>
  <a href="https://zenodo.org/badge/latestdoi/52494384">
    <img src="https://zenodo.org/badge/52494384.svg" alt="DOI"/>
  </a>
</p>

**Mici** is a Python package providing implementations of *Markov chain Monte
Carlo* (MCMC) methods for approximate inference in probabilistic models, with a
particular focus on MCMC methods based on simulating Hamiltonian dynamics on a
manifold.

## Features

Key features include

  * implementations of MCMC methods for sampling from distributions on embedded
    manifolds implicitly-defined by a constraint equation and distributions on
    Riemannian manifolds with a user-specified metric, 
  * a modular design allowing use of a wide range of inference algorithms by 
    mixing and matching different components, making it easy for users to 
    extend the package and use within their own code,  
  * computational efficient inference via transparent caching of the results of 
    expensive operations and intermediate results calculated in derivative 
    computations allowing later reuse without recalculation,  
  * memory efficient inference for large models by memory-mapping chains to 
    disk, allowing long runs on large models without hitting memory issues.

## Installation

To install and use Mici the minimal requirements are a Python 3.6+ environment
with [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org)
installed. The latest Mici release on PyPI (and its dependencies) can be
installed in the current Python environment by running 

```sh
pip install mici
``` 

To instead install the latest development version from the `master` branch on Github run 

```sh
pip install git+https://github.com/matt-graham/mici
```

If available in the installed Python environment the following additional
packages provide extra functionality and features

  * [Autograd](https://github.com/HIPS/autograd): if available Autograd will 
    be used to automatically compute the required derivatives of the model
    functions (providing they are specified using functions from the
    `autograd.numpy` and `autograd.scipy` interfaces). To sample chains in
    parallel using `autograd` functions you also need to install
    [multiprocess](https://github.com/uqfoundation/multiprocess). This will
    cause `multiprocess.Pool` to be used in preference to the in-built
    `mutiprocessing.Pool` for parallelisation as multiprocess supports
    serialisation (via [dill](https://github.com/uqfoundation/dill)) of a much
    wider range of types, including of Autograd generated functions. Both
    Autograd and multiprocess can be installed alongside Mici by running `pip
    install mici[autodiff]`.
  * [RandomGen](https://github.com/bashtage/randomgen): if RandomGen is 
    available the `randomgen.Xorshift1024` random number generator will be used 
    when running multiple chains in parallel, with the `jump` method of the
    object used to reproducibly generate independent substreams. RandomGen can
    be installed alongside Mici by running `pip install mici[randomgen]`.
  * [ArviZ](https://arviz-devs.github.io/arviz/index.html#): if ArviZ is 
    available  outputs of a sampling run can be converted to an
    `arviz.InferenceData` container object using
    `mici.utils.convert_to_arviz_inference_data`, allowing straightforward use
    of the extensive Arviz visualisation and diagnostic functionality.

## Why Mici?

Mici is named for [Augusta 'Mici'
Teller](https://en.wikipedia.org/wiki/Augusta_H._Teller), who along with
[Arianna Rosenbluth](https://en.wikipedia.org/wiki/Arianna_W._Rosenbluth)
developed the code for the [MANIAC I](https://en.wikipedia.org/wiki/MANIAC_I)
computer used in the seminal paper [*Equations of State Calculations by Fast
Computing Machines*](https://doi.org/10.1063%2F1.1699114) which introduced the
first example of a Markov chain Monte Carlo method. 

## Related projects

Other Python packages for performing MCMC inference include
[PyMC3](https://github.com/pymc-devs/pymc3),
[PyStan](https://github.com/stan-dev/pystan) (the Python interface to
[Stan](http://mc-stan.org/)), [Pyro](https://github.com/pyro-ppl/pyro) /
[NumPyro](https://github.com/pyro-ppl/numpyro), [TensorFlow
Probability](https://github.com/tensorflow/probability),
[emcee](https://github.com/dfm/emcee) and
[Sampyl](https://github.com/mcleonard/sampyl).

Unlike PyMC3, PyStan, (Num)Pyro and TensorFlow Probability which are complete
probabilistic programming frameworks including functionality for definining a
probabilistic model / program, but like emcee and Sampyl, Mici is solely
focussed on providing implementations of inference algorithms, with the user
expected to be able to define at a minimum a function specifying the negative
log (unnormalised) density of the distribution of interest. 

Further while PyStan, (Num)Pyro and TensorFlow Probability all push the
sampling loop into external compiled non-Python code, in Mici the sampling loop
is run directly within Python. This has the consequence that for small models
in which the negative log density of the target distribution and other model
functions are cheap to evaluate, the interpreter overhead in iterating over the
chains in Python can dominate the computational cost, making sampling much
slower than packages which outsource the sampling loop to a efficient compiled
implementation.

 ## Overview of package
 
API documentation for the package is available
[here](https://matt-graham.github.io/mici/docs). The three main user-facing
modules within the `mici` package are the `systems`, `integrators` and
`samplers` modules and you will generally need to create an instance of one
class from each module.
 
 [`mici.systems`](https://matt-graham.github.io/mici/docs/systems.html) -
 Hamiltonian systems encapsulating model functions and their derivatives

   * `EuclideanMetricSystem` - systems with a metric on the position space with
      a constant matrix representation,
   * `GaussianEuclideanMetricSystem` - systems in which the target distribution
     is defined by a density with respect to the standard Gaussian measure on 
     the position space allowing analytically solving for flow corresponding to 
     the quadratic components of Hamiltonian 
     ([Shahbaba et al., 2014](#shababa2014split)),
   * `RiemannianMetricSystem` - systems with a metric on the position space 
     with a position-dependent matrix representation 
     ([Girolami and Calderhead, 2011](#girolami2011riemann)),
   * `SoftAbsRiemannianMetricSystem`  - system with *SoftAbs* 
     eigenvalue-regularised Hessian of negative log target density as metric 
     matrix representation ([Betancourt, 2013](#betancourt2013general)),
   * `DenseConstrainedEuclideanMetricSystem` - Euclidean-metric system subject 
     to holonomic constraints 
     ([Hartmann and Schütte, 2005](#hartmann2005constrained); 
      [Brubaker, Salzmann and Urtasun, 2012](#brubaker2012family); 
      [Lelièvre, Rousset and Stoltz, 2018](#lelievre2018hybrid)) 
     with a dense constraint function Jacobian matrix,

[`mici.integrators`](https://matt-graham.github.io/mici/docs/integrators.html) - 
symplectic integrators for Hamiltonian dynamics

  * `LeapfrogIntegrator` - explicit leapfrog (Störmer-Verlet) integrator for 
    separable Hamiltonian systems 
    ([Leimkulher and Reich, 2004](#leimkuhler2004simulating)),
  * `ImplicitLeapfrogIntegrator` - implicit (or generalised) leapfrog 
    integrator for non-separable Hamiltonian systems 
    ([Leimkulher and Reich, 2004](#leimkuhler2004simulating)),
  * `ConstrainedLeapfrogIntegrator` - constrained leapfrog integrator for 
     Hamiltonian systems subject to holonomic constraints 
     ([Andersen, 1983](#andersen1983rattle); 
     [Leimkuhler and Reich, 1994](#leimkuhler1994symplectic)).

[`mici.samplers`](https://matt-graham.github.io/mici/docs/samplers.html) - MCMC
samplers for peforming inference

  * `StaticMetropolisHMC` - Static integration time Hamiltonian Monte Carlo 
     with Metropolis accept step ([Duane et al., 1987](duane1987hybrid)),
  * `RandomMetropolisHMC` - Random integration time Hamiltonian Monte Carlo
    with Metropolis accept step ([Mackenzie, 1989](#mackenzie1989improved)),
  * `DynamicMultinomialHMC` - Dynamic integration time Hamiltonian Monte Carlo
    with multinomial sampling from trajectory 
    ([Hoffman and Gelman, 2014](#hoffman2014nouturn); 
    [Betancourt, 2017](#betancourt2017conceptual)).

## Example: sampling on a torus

<img src='images/torus-samples.gif' width='360px'/>

A simple complete example of using the package to compute approximate samples
from a distribution on a two-dimensional torus embedded in a three-dimensional
space is given below. The computed samples are visualised in the animation
above. Here we use `autograd` to automatically construct functions to calculate
the required derivatives (gradient of negative log density of target
distribution and Jacobian of constraint function), sample four chains in
parallel using `multiprocess` and use `matplotlib` to plot the samples.

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

  1. <a id='andersen1983rattle'></a> Andersen, H.C., 1983. RATTLE: A “velocity” 
     version of the SHAKE algorithm for molecular dynamics calculations. 
     *Journal of Computational Physics*, 52(1), pp.24-34.
     [![DOI:10.1016/0021-9991(83)90014-1](https://zenodo.org/badge/DOI/10.1016/0021-9991(83)90014-1.svg)](https://doi.org/10.1016/0021-9991(83)90014-1)
  2. <a id='duane1987hybrid'></a> Duane, S., Kennedy, A.D., Pendleton, B.J. and 
     Roweth, D., 1987. Hybrid Monte Carlo. *Physics letters B*, 195(2), 
     pp.216-222.
     [![DOI:10.1016/0370-2693(87)91197-X](https://zenodo.org/badge/DOI/10.1016/0370-2693(87)91197-X.svg)](https://doi.org/10.1016/0370-2693(87)91197-X)
  3. <a id='mackenzie1989improved'></a> Mackenzie, P.B., 1989. An improved 
     hybrid Monte Carlo method. *Physics Letters B*, 226(3-4), pp.369-371.
     [![DOI:10.1016/0370-2693(89)91212-4](https://zenodo.org/badge/DOI/10.1016/0370-2693(89)91212-4.svg)](https://doi.org/10.1016/0370-2693(89)91212-4)
  4. <a id='horowitz1991generalized'></a> Horowitz, A.M., 1991. A generalized 
     guided Monte Carlo algorithm. *Physics Letters  B*, 268(CERN-TH-6172-91), 
     pp.247-252.
     [![DOI:10.1016/0370-2693(91)90812-5](https://zenodo.org/badge/DOI/10.1016/0370-2693(91)90812-5.svg)](https://doi.org/10.1016/0370-2693(91)90812-5)
  5. <a id='leimkuhler1994symplectic'></a> Leimkuhler, B. and Reich, S., 1994. 
     Symplectic integration of constrained Hamiltonian systems. *Mathematics of 
     Computation*, 63(208), pp.589-605.
     [![DOI:10.2307/2153284](https://zenodo.org/badge/DOI/10.2307/2153284.svg)](https://doi.org/10.2307/2153284)
  6. <a id='leimkuhler2004simulating'></a> Leimkuhler, B. and Reich, S., 2004. 
     Simulating Hamiltonian dynamics (Vol. 14). *Cambridge University Press*.
     [![DOI:10.1017/CBO9780511614118](https://zenodo.org/badge/DOI/10.1017/CBO9780511614118.svg)](https://doi.org/10.1017/CBO9780511614118)
  7. <a id='hartmann2005constrained'></a> Hartmann, C. and Schütte, C., 2005. A 
     constrained hybrid Monte‐Carlo algorithm and the problem of calculating the 
     free energy in several variables. *ZAMM ‐ Journal of Applied Mathematics and 
     Mechanics*, 85(10), pp.700-710.
     [![DOI:10.1002/zamm.200410218](https://zenodo.org/badge/DOI/10.1002/zamm.200410218.svg)](https://doi.org/10.1002/zamm.200410218)
  8. <a id='girolami2011riemann'></a> Girolami, M. and Calderhead, B., 2011. 
     Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *Journal of 
     the Royal Statistical Society: Series B (Statistical Methodology)*, 73(2), pp.123-214.
     [![DOI:10.1111/j.1467-9868.2010.00765.x](https://zenodo.org/badge/DOI/10.1111/j.1467-9868.2010.00765.x.svg)](https://doi.org/10.1111/j.1467-9868.2010.00765.x)
  9. <a id='brubaker2012family'></a> Brubaker, M., Salzmann, M. and 
     Urtasun, R., 2012. A family of MCMC methods on implicitly defined 
     manifolds. In *Artificial intelligence and statistics* (pp. 161-172).
     [![CiteSeerX:10.1.1.417.6111](http://img.shields.io/badge/CiteSeerX-10.1.1.417.6111-blue.svg)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.417.6111)
 10. <a id='betancourt2013general'></a> Betancourt, M., 2013. A general metric 
     for Riemannian manifold Hamiltonian Monte Carlo. In *Geometric science of 
     information* (pp. 327-334).
     [![DOI:10.1007/978-3-642-40020-9_35](https://zenodo.org/badge/DOI/10.1007/978-3-642-40020-9_35.svg)](https://doi.org/10.1007/978-3-642-40020-9_35)
     [![arXiv:1212.4693](http://img.shields.io/badge/arXiv-1212.4693-B31B1B.svg)](https://arxiv.org/abs/1212.4693)
 11. <a id='hoffman2014nouturn'></a> Hoffman, M.D. and Gelman, A., 2014. The 
     No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte 
     Carlo. *Journal of Machine Learning Research*, 15(1), pp.1593-1623.
     [![CiteSeerX:10.1.1.220.8395](http://img.shields.io/badge/CiteSeerX-10.1.1.220.8395-blue.svg)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.220.8395)
     [![arXiv:1111.4246](http://img.shields.io/badge/arXiv-1111.4246-B31B1B.svg)](https://arxiv.org/abs/1111.4246)
 12. <a id='shababa2014split'></a> Shahbaba, B., Lan, S., Johnson, W.O. and 
     Neal, R.M., 2014. Split Hamiltonian Monte Carlo. *Statistics and 
     Computing*, 24(3), pp.339-349.
     [![DOI:10.1007/s11222-012-9373-1](https://zenodo.org/badge/DOI/10.1007/s11222-012-9373-1.svg)](https://doi.org/10.1007/s11222-012-9373-1)
     [![arXiv:1106.5941](http://img.shields.io/badge/arXiv-1106.5941-B31B1B.svg)](https://arxiv.org/abs/1106.5941)
 13. <a id='betancourt2017conceptual'></a> Betancourt, M., 2017. A conceptual 
     introduction to Hamiltonian Monte Carlo.
     [![arXiv:1701.02434](http://img.shields.io/badge/arXiv-1701.02434-B31B1B.svg)](https://arxiv.org/abs/1701.02434)
 14. <a id='lelievre2018hybrid'></a> Lelièvre, T., Rousset, M. and Stoltz, G., 
     2018. Hybrid Monte Carlo methods for sampling probability measures on 
     submanifolds.
     [![arXiv:1807.02356](http://img.shields.io/badge/arXiv-1807.02356-B31B1B.svg)](https://arxiv.org/abs/1807.02356)
