<div style="text-align: center;" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/matt-graham/mici/main/images/mici-logo-rectangular-light-text.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/matt-graham/mici/main/images/mici-logo-rectangular.svg">
  <img alt="Mici logo" src="https://raw.githubusercontent.com/matt-graham/mici/main/images/mici-logo-rectangular.svg" width="400px">
</picture>
<div style="text-align: center;" align="center">

[![PyPI version](https://badge.fury.io/py/mici.svg)](https://pypi.org/project/mici)
[![Zenodo DOI](https://zenodo.org/badge/52494384.svg)](https://zenodo.org/badge/latestdoi/52494384)
[![Documentation](https://img.shields.io/badge/Sphinx-documentation-blue?logo=sphinx&logoColor=white)](https://matt-graham.github.io/mici/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Test status](https://github.com/matt-graham/mici/actions/workflows/tests.yml/badge.svg)](https://github.com/matt-graham/mici/actions/workflows/tests.yml)
[![Linting status](https://github.com/matt-graham/mici/actions/workflows/linting.yml/badge.svg)](https://github.com/matt-graham/mici/actions/workflows/linting.yml)
[![Docs status](https://github.com/matt-graham/mici/actions/workflows/docs.yml/badge.svg)](https://github.com/matt-graham/mici/actions/workflows/docs.yml)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>
</div>

**Mici** is a Python package providing implementations of _Markov chain Monte
Carlo_ (MCMC) methods for approximate inference in probabilistic models, with a
particular focus on MCMC methods based on simulating Hamiltonian dynamics on a
manifold.

## Features

Key features include

- a modular design allowing use of a wide range of inference algorithms by
  mixing and matching different components, and making it easy to
  extend the package,
- a pure Python code base with minimal dependencies,
  allowing easy integration within other code,
- built-in support for several automatic differentiation frameworks, including
  [JAX](https://jax.readthedocs.io/en/latest/) and
  [Autograd](https://github.com/HIPS/autograd), or the option to supply your own
  derivative functions,
- implementations of MCMC methods for sampling from distributions on embedded
  manifolds implicitly-defined by a constraint equation and distributions on
  Riemannian manifolds with a user-specified metric,
- computationally efficient inference via transparent caching of the results
  of expensive operations and intermediate results calculated in derivative
  computations allowing later reuse without recalculation,
- memory efficient inference for large models by memory-mapping chains to
  disk, allowing long runs on large models without hitting memory issues.

## Installation

To install and use Mici the minimal requirements are a Python 3.10+ environment
with [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org)
installed. The latest Mici release on PyPI (and its dependencies) can be
installed in the current Python environment by running

```sh
pip install mici
```

To instead install the latest development version from the `main` branch on Github run

```sh
pip install git+https://github.com/matt-graham/mici
```

If available in the installed Python environment the following additional
packages provide extra functionality and features

- [ArviZ](https://python.arviz.org/en/latest/index.html): if ArviZ is
  available the traces (dictionary) output of a sampling run can be directly
  converted to an `arviz.InferenceData` container object using
  `arviz.convert_to_inference_data` or implicitly converted by passing the
  traces dictionary as the `data` argument
  [to ArviZ API functions](https://python.arviz.org/en/latest/api/index.html),
  allowing straightforward use of the ArviZ's extensive visualisation and
  diagnostic functions.
- [Autograd](https://github.com/HIPS/autograd): if available Autograd will
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
install mici[autograd]`.
- [JAX](https://jax.readthedocs.io/en/latest/): if available JAX will be used to
  automatically compute the required derivatives of the model functions (providing
  they are specified using functions from the [`jax`
  interface](https://jax.readthedocs.io/en/latest/jax.html)). To sample chains
  parallel using JAX functions you also need to install
  [multiprocess](https://github.com/uqfoundation/multiprocess), though note due to
  JAX's use of multithreading which [is incompatible with forking child
  processes](https://docs.python.org/3/library/os.html#os.fork), this can result in
  deadlock. Both JAX and multiprocess can be installed alongside Mici by running `pip
install mici[jax]`.
- [SymNum](https://github.com/matt-graham/symnum): if available SymNum will be used to
  automatically compute the required derivatives of the model functions (providing
  they are specified using functions from the [`symnum.numpy`
  interface](https://matt-graham.github.io/symnum/symnum.numpy.html)). Symnum can be
  installed alongside Mici by running `pip install mici[symnum]`.

## Why Mici?

Mici is named for [Augusta 'Mici'
Teller](https://en.wikipedia.org/wiki/Augusta_H._Teller), who along with
[Arianna Rosenbluth](https://en.wikipedia.org/wiki/Arianna_W._Rosenbluth)
developed the code for the [MANIAC I](https://en.wikipedia.org/wiki/MANIAC_I)
computer used in the seminal paper [_Equations of State Calculations by Fast
Computing Machines_](https://doi.org/10.1063%2F1.1699114) which introduced the
first example of a Markov chain Monte Carlo method.

## Related projects

Other Python packages for performing MCMC inference include
[PyMC](https://github.com/pymc-devs/pymc),
[PyStan](https://github.com/stan-dev/pystan) (the Python interface to
[Stan](http://mc-stan.org/)), [Pyro](https://github.com/pyro-ppl/pyro) /
[NumPyro](https://github.com/pyro-ppl/numpyro), [TensorFlow
Probability](https://github.com/tensorflow/probability),
[emcee](https://github.com/dfm/emcee),
[Sampyl](https://github.com/mcleonard/sampyl) and
[BlackJAX](https://github.com/blackjax-devs/blackjax).

Unlike PyMC, PyStan, (Num)Pyro and TensorFlow Probability which are complete
probabilistic programming frameworks including functionality for defining a
probabilistic model / program, but like emcee, Sampyl and BlackJAX, Mici is solely
focused on providing implementations of inference algorithms, with the user
expected to be able to define at a minimum a function specifying the negative
log (unnormalized) density of the distribution of interest.

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
[here](https://matt-graham.github.io/mici/). The three main user-facing
modules within the `mici` package are the `systems`, `integrators` and
`samplers` modules and you will generally need to create an instance of one
class from each module.

[`mici.systems`](https://matt-graham.github.io/mici/mici.systems.html) -
Hamiltonian systems encapsulating model functions and their derivatives

- `EuclideanMetricSystem` - systems with a metric on the position space with
  a constant matrix representation,
- `GaussianEuclideanMetricSystem` - systems in which the target distribution
  is defined by a density with respect to the standard Gaussian measure on
  the position space allowing analytically solving for flow corresponding to
  the quadratic components of Hamiltonian
  ([Shahbaba et al., 2014](#shababa2014split)),
- `RiemannianMetricSystem` - systems with a metric on the position space
  with a position-dependent matrix representation
  ([Girolami and Calderhead, 2011](#girolami2011riemann)),
- `SoftAbsRiemannianMetricSystem` - system with _SoftAbs_
  eigenvalue-regularized Hessian of negative log target density as metric
  matrix representation ([Betancourt, 2013](#betancourt2013general)),
- `DenseConstrainedEuclideanMetricSystem` - Euclidean-metric system subject
  to holonomic constraints
  ([Hartmann and Schütte, 2005](#hartmann2005constrained);
  [Brubaker, Salzmann and Urtasun, 2012](#brubaker2012family);
  [Lelièvre, Rousset and Stoltz, 2019](#lelievre2019hybrid))
  with a dense constraint function Jacobian matrix,

[`mici.integrators`](https://matt-graham.github.io/mici/mici.integrators.html) -
symplectic integrators for Hamiltonian dynamics

- `LeapfrogIntegrator` - explicit leapfrog (Störmer-Verlet) integrator for
  separable Hamiltonian systems
  ([Leimkulher and Reich, 2004](#leimkuhler2004simulating)),
- `ImplicitLeapfrogIntegrator` - implicit (or generalized) leapfrog
  integrator for Hamiltonian systems with non-separable component
  ([Leimkulher and Reich, 2004](#leimkuhler2004simulating)),
- `ImplicitMidpointIntegrator` - implicit midpoint
  integrator for general Hamiltonian systems
  ([Leimkulher and Reich, 2004](#leimkuhler2004simulating)),
- `SymmetricCompositionIntegrator` - family of symplectic integrators for Hamiltonians
  that can be split in to components with tractable flow maps, with specific
  two-, three- and four-stage instantations due to [Blanes, Casas and Sanz-Serna (2014)](#blanes2014numerical),
- `ConstrainedLeapfrogIntegrator` - constrained leapfrog integrator for
  Hamiltonian systems subject to holonomic constraints
  ([Andersen, 1983](#andersen1983rattle);
  [Leimkuhler and Reich, 1994](#leimkuhler1994symplectic)).

[`mici.samplers`](https://matt-graham.github.io/mici/mici.samplers.html) - MCMC
samplers for peforming inference

- `StaticMetropolisHMC` - static integration time Hamiltonian Monte Carlo
  with Metropolis accept step ([Duane et al., 1987](duane1987hybrid)),
- `RandomMetropolisHMC` - random integration time Hamiltonian Monte Carlo
  with Metropolis accept step ([Mackenzie, 1989](#mackenzie1989improved)),
- `DynamicSliceHMC` - dynamic integration time Hamiltonian Monte Carlo
  with slice sampling from trajectory, equivalent to the original 'NUTS' algorithm
  ([Hoffman and Gelman, 2014](#hoffman2014nouturn)).
- `DynamicMultinomialHMC` - dynamic integration time Hamiltonian Monte Carlo
  with multinomial sampling from trajectory, equivalent to the current default
  MCMC algorithm in [Stan](https://mc-stan.org/)
  ([Hoffman and Gelman, 2014](#hoffman2014nouturn);
  [Betancourt, 2017](#betancourt2017conceptual)).

## Notebooks

The manifold MCMC methods implemented in Mici have been used in several research projects. Below links are provided to a selection of Jupyter notebooks associated with these projects as demonstrations of how to use Mici and to illustrate some of the settings in which manifold MCMC methods can be computationally advantageous.

<table>
  <tr>
    <th colspan="2"><a href="https://github.com/thiery-lab/manifold_lifting">Manifold lifting: scaling MCMC to the vanishing noise regime</a></th>
  </tr>
  <tr>
    <td>Open non-interactive version with nbviewer</td>
    <td>
      <a href="https://nbviewer.jupyter.org/github/thiery-lab/manifold_lifting/blob/main/notebooks/Two-dimensional_example.ipynb">
        <img src="https://raw.githubusercontent.com/jupyter/design/main/logos/Badges/nbviewer_badge.svg?sanitize=true" width="109" alt="Render with nbviewer"  style="vertical-align:text-bottom" />
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Binder</td>
    <td>
      <a href="https://mybinder.org/v2/gh/thiery-lab/manifold_lifting/main?filepath=notebooks%2FTwo-dimensional_example.ipynb">
        <img src="https://mybinder.org/badge_logo.svg" alt="Launch with Binder"  style="vertical-align:text-bottom"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Google Colab</td>
    <td>
      <a href="https://colab.research.google.com/github/thiery-lab/manifold_lifting/blob/main/notebooks/Two-dimensional_example.ipynb">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
       </a>
    </td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="2"><a href="https://github.com/thiery-lab/manifold-mcmc-for-diffusions">Manifold MCMC methods for inference in diffusion models</a></th>
  </tr>
  <tr>
    <td>Open non-interactive version with nbviewer</td>
    <td>
      <a href="https://nbviewer.jupyter.org/github/thiery-lab/manifold-mcmc-for-diffusions/blob/main/FitzHugh-Nagumo_example.ipynb">
        <img src="https://raw.githubusercontent.com/jupyter/design/main/logos/Badges/nbviewer_badge.svg?sanitize=true" width="109" alt="Render with nbviewer"  style="vertical-align:text-bottom" />
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Binder</td>
    <td>
      <a href="https://mybinder.org/v2/gh/thiery-lab/manifold-mcmc-for-diffusions/main?filepath=FitzHugh-Nagumo_example.ipynb">
        <img src="https://mybinder.org/badge_logo.svg" alt="Launch with Binder"  style="vertical-align:text-bottom"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Open interactive version with Google Colab</td>
    <td>
      <a href="https://colab.research.google.com/github/thiery-lab/manifold-mcmc-for-diffusions/blob/main/FitzHugh-Nagumo_example.ipynb">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
       </a>
    </td>
  </tr>
</table>

## Example: sampling on a torus

<img src='https://raw.githubusercontent.com/matt-graham/mici/main/images/torus-samples.gif' width='360px'/>

A simple complete example of using the package to compute approximate samples from a
distribution on a two-dimensional torus embedded in a three-dimensional space is given
below. The computed samples are visualized in the animation above. Here we use
[SymNum](https://github.com/matt-graham/symnum) to automatically construct functions to
calculate the required derivatives (gradient of negative log density of target
distribution and Jacobian of constraint function), sample four chains in parallel using
`multiprocessing`, use [ArviZ](https://python.arviz.org/en/stable/) to calculate
diagnostics and use [Matplotlib](https://matplotlib.org/) to plot the samples.

```Python
import mici
import numpy as np
import symnum
import symnum.numpy as snp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import arviz

# Define fixed model parameters
R = 1.0  # toroidal radius ∈ (0, ∞)
r = 0.5  # poloidal radius ∈ (0, R)
α = 0.9  # density fluctuation amplitude ∈ [0, 1)

# State dimension
dim_q = 3


# Define constraint function such that the set {q : constr(q) == 0} is a torus
@symnum.numpify(dim_q)
def constr(q):
    x, y, z = q
    return snp.array([((x**2 + y**2) ** 0.5 - R) ** 2 + z**2 - r**2])


# Define negative log density for the target distribution on torus
# (with respect to 2D 'area' measure for torus)
@symnum.numpify(dim_q)
def neg_log_dens(q):
    x, y, z = q
    θ = snp.arctan2(y, x)
    ϕ = snp.arctan2(z, x / snp.cos(θ) - R)
    return snp.log1p(r * snp.cos(ϕ) / R) - snp.log1p(snp.sin(4 * θ) * snp.cos(ϕ) * α)


# Specify constrained Hamiltonian system with default identity metric
system = mici.systems.DenseConstrainedEuclideanMetricSystem(
    neg_log_dens,
    constr,
    backend="symnum",
)

# System is constrained therefore use constrained leapfrog integrator
integrator = mici.integrators.ConstrainedLeapfrogIntegrator(system)

# Seed a random number generator
rng = np.random.default_rng(seed=1234)

# Use dynamic integration-time HMC implementation as MCMC sampler
sampler = mici.samplers.DynamicMultinomialHMC(system, integrator, rng)

# Sample initial positions on torus using parameterisation (θ, ϕ) ∈ [0, 2π)²
# x, y, z = (R + r * cos(ϕ)) * cos(θ), (R + r * cos(ϕ)) * sin(θ), r * sin(ϕ)
n_chain = 4
θ_init, ϕ_init = rng.uniform(0, 2 * np.pi, size=(2, n_chain))
q_init = np.stack(
    [
        (R + r * np.cos(ϕ_init)) * np.cos(θ_init),
        (R + r * np.cos(ϕ_init)) * np.sin(θ_init),
        r * np.sin(ϕ_init),
    ],
    -1,
)


# Define function to extract variables to trace during sampling
def trace_func(state):
    x, y, z = state.pos
    return {"x": x, "y": y, "z": z}


# Sample 4 chains in parallel with 500 adaptive warm up iterations in which the
# integrator step size is tuned, followed by 2000 non-adaptive iterations
final_states, traces, stats = sampler.sample_chains(
    n_warm_up_iter=500,
    n_main_iter=2000,
    init_states=q_init,
    n_process=4,
    trace_funcs=[trace_func],
)

# Print average accept probability and number of integrator steps per chain
for c in range(n_chain):
    print(f"Chain {c}:")
    print(f"  Average accept prob. = {stats['accept_stat'][c].mean():.2f}")
    print(f"  Average number steps = {stats['n_step'][c].mean():.1f}")

# Print summary statistics and diagnostics computed using ArviZ
print(arviz.summary(traces))

# Visualize concatentated chain samples as animated 3D scatter plot
fig, ax = plt.subplots(
    figsize=(4, 4),
    subplot_kw={"projection": "3d", "proj_type": "ortho"},
)
(points_3d,) = ax.plot(*(np.concatenate(traces[k]) for k in "xyz"), ".", ms=0.5)
ax.axis("off")
for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
    set_lim((-1, 1))


def update(i):
    angle = 45 * (np.sin(2 * np.pi * i / 60) + 1)
    ax.view_init(elev=angle, azim=angle)
    return (points_3d,)


anim = animation.FuncAnimation(fig, update, frames=60, interval=100)
plt.show()
```

## References

1. <a id='andersen1983rattle'></a> Andersen, H.C., 1983. RATTLE: A “velocity”
   version of the SHAKE algorithm for molecular dynamics calculations.
   _Journal of Computational Physics_, 52(1), pp.24-34.
   [![DOI:10.1016/0021-9991(83)90014-1](<https://zenodo.org/badge/DOI/10.1016/0021-9991(83)90014-1.svg>)](<https://doi.org/10.1016/0021-9991(83)90014-1>)
2. <a id='duane1987hybrid'></a> Duane, S., Kennedy, A.D., Pendleton, B.J. and
   Roweth, D., 1987. Hybrid Monte Carlo. _Physics letters B_, 195(2),
   pp.216-222.
   [![DOI:10.1016/0370-2693(87)91197-X](<https://zenodo.org/badge/DOI/10.1016/0370-2693(87)91197-X.svg>)](<https://doi.org/10.1016/0370-2693(87)91197-X>)
3. <a id='mackenzie1989improved'></a> Mackenzie, P.B., 1989. An improved
   hybrid Monte Carlo method. _Physics Letters B_, 226(3-4), pp.369-371.
   [![DOI:10.1016/0370-2693(89)91212-4](<https://zenodo.org/badge/DOI/10.1016/0370-2693(89)91212-4.svg>)](<https://doi.org/10.1016/0370-2693(89)91212-4>)
4. <a id='horowitz1991generalized'></a> Horowitz, A.M., 1991. A generalized
   guided Monte Carlo algorithm. _Physics Letters B_, 268(CERN-TH-6172-91),
   pp.247-252.
   [![DOI:10.1016/0370-2693(91)90812-5](<https://zenodo.org/badge/DOI/10.1016/0370-2693(91)90812-5.svg>)](<https://doi.org/10.1016/0370-2693(91)90812-5>)
5. <a id='leimkuhler1994symplectic'></a> Leimkuhler, B. and Reich, S., 1994.
   Symplectic integration of constrained Hamiltonian systems. _Mathematics of
   Computation_, 63(208), pp.589-605.
   [![DOI:10.2307/2153284](https://zenodo.org/badge/DOI/10.2307/2153284.svg)](https://doi.org/10.2307/2153284)
6. <a id='leimkuhler2004simulating'></a> Leimkuhler, B. and Reich, S., 2004.
   Simulating Hamiltonian dynamics (Vol. 14). _Cambridge University Press_.
   [![DOI:10.1017/CBO9780511614118](https://zenodo.org/badge/DOI/10.1017/CBO9780511614118.svg)](https://doi.org/10.1017/CBO9780511614118)
7. <a id='hartmann2005constrained'></a> Hartmann, C. and Schütte, C., 2005. A
   constrained hybrid Monte‐Carlo algorithm and the problem of calculating the
   free energy in several variables. _ZAMM ‐ Journal of Applied Mathematics and
   Mechanics_, 85(10), pp.700-710.
   [![DOI:10.1002/zamm.200410218](https://zenodo.org/badge/DOI/10.1002/zamm.200410218.svg)](https://doi.org/10.1002/zamm.200410218)
8. <a id='girolami2011riemann'></a> Girolami, M. and Calderhead, B., 2011.
   Riemann manifold Langevin and Hamiltonian Monte Carlo methods. _Journal of
   the Royal Statistical Society: Series B (Statistical Methodology)_, 73(2), pp.123-214.
   [![DOI:10.1111/j.1467-9868.2010.00765.x](https://zenodo.org/badge/DOI/10.1111/j.1467-9868.2010.00765.x.svg)](https://doi.org/10.1111/j.1467-9868.2010.00765.x)
9. <a id='brubaker2012family'></a> Brubaker, M., Salzmann, M. and
   Urtasun, R., 2012. A family of MCMC methods on implicitly defined
   manifolds. In _Artificial intelligence and statistics_ (pp. 161-172).
   [![CiteSeerX:10.1.1.417.6111](http://img.shields.io/badge/CiteSeerX-10.1.1.417.6111-blue.svg)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.417.6111)
10. <a id='betancourt2013general'></a> Betancourt, M., 2013. A general metric
    for Riemannian manifold Hamiltonian Monte Carlo. In _Geometric science of
    information_ (pp. 327-334).
    [![DOI:10.1007/978-3-642-40020-9_35](https://zenodo.org/badge/DOI/10.1007/978-3-642-40020-9_35.svg)](https://doi.org/10.1007/978-3-642-40020-9_35)
    [![arXiv:1212.4693](http://img.shields.io/badge/arXiv-1212.4693-B31B1B.svg)](https://arxiv.org/abs/1212.4693)
11. <a id='hoffman2014nouturn'></a> Hoffman, M.D. and Gelman, A., 2014. The
    No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte
    Carlo. _Journal of Machine Learning Research_, 15(1), pp.1593-1623.
    [![CiteSeerX:10.1.1.220.8395](http://img.shields.io/badge/CiteSeerX-10.1.1.220.8395-blue.svg)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.220.8395)
    [![arXiv:1111.4246](http://img.shields.io/badge/arXiv-1111.4246-B31B1B.svg)](https://arxiv.org/abs/1111.4246)
12. <a id='shababa2014split'></a> Shahbaba, B., Lan, S., Johnson, W.O. and
    Neal, R.M., 2014. Split Hamiltonian Monte Carlo. _Statistics and
    Computing_, 24(3), pp.339-349.
    [![DOI:10.1007/s11222-012-9373-1](https://zenodo.org/badge/DOI/10.1007/s11222-012-9373-1.svg)](https://doi.org/10.1007/s11222-012-9373-1)
    [![arXiv:1106.5941](http://img.shields.io/badge/arXiv-1106.5941-B31B1B.svg)](https://arxiv.org/abs/1106.5941)
13. <a id='blanes2014numerical'></a> Blanes, S., Casas, F., & Sanz-Serna, J. M., 2014.
    Numerical integrators for the Hybrid Monte Carlo method.
    _SIAM Journal on Scientific Computing_, 36(4), A1556-A1580.
    [![DOI:10.1137/130932740](https://zenodo.org/badge/DOI/10.1137/130932740.svg)](https://doi.org/10.1137/130932740)
    [![arXiv:1405.3153](http://img.shields.io/badge/arXiv-1405.3153-B31B1B.svg)](https://arxiv.org/abs/1405.3153)
14. <a id='betancourt2017conceptual'></a> Betancourt, M., 2017. A conceptual
    introduction to Hamiltonian Monte Carlo.
    [![arXiv:1701.02434](http://img.shields.io/badge/arXiv-1701.02434-B31B1B.svg)](https://arxiv.org/abs/1701.02434)
15. <a id='lelievre2019hybrid'></a> Lelièvre, T., Rousset, M. and Stoltz, G., 2019. Hybrid Monte Carlo methods for sampling probability measures on
    submanifolds. In _Numerische Mathematik_, 143(2), (pp.379-421).
    [![DOI:10.1007/s00211-019-01056-4](https://zenodo.org/badge/DOI/10.1007/s00211-019-01056-4.svg)](https://doi.org/10.1007/s00211-019-01056-4)
    [![arXiv:1807.02356](http://img.shields.io/badge/arXiv-1807.02356-B31B1B.svg)](https://arxiv.org/abs/1807.02356)
