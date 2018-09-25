# Hamiltonian Monte Carlo

Implementations of various Hamiltonian dynamics based Markov chain Monte Carlo (MCMC) samplers in idiomatic Python code. A modular design is used to as far as possible allowing mixing and matching elements of different proposed extensions to the original Hybrid Monte Carlo algorithm proposed in Duane et al. (1987).

## Dependencies

To install and use the package the minimal requirements are a Python 3 environment with [NumPy](http://www.numpy.org/) and [SciPy](https://www.scipy.org) installed. 

If available [autograd](https://github.com/HIPS/autograd) will be used to automatically compute the required derivatives of the model functions (providing they are specified using functions from the `autograd.numpy` and `autograd.scipy` interfaces). If the [tqdm](https://github.com/tqdm/tqdm) package is available a simple progress bar will be shown during sampling.

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
    Stoltz, 2018) and for inference in differentiable generative models when conditioning on observed outputs (Graham and Storkey, 2017a)

Numerical integrators

  * Explicit leapfrog for separable Hamiltonian systems
  * 'Split' leapfrog for Hamiltonian systems with an analytically tractable
    component for which the exact flow can be solved (Shahbaba et al., 2014)
  * Implicit leapfrog for non-separable Hamiltonian systems
  * Geodesic leapfrog for constrained Hamiltonian systems

## Example usage

A simple complete example of using the package to sample from a multivariate Gaussian distribution with randomly generated parameters is given below. Here an isotropic Euclidean metric Hamiltonian system is used (corresponding to a isotropic covariance Gaussian marginal distribution on the momenta) with the dynamic integration time HMC implementation described in Betancourt (2017), which is a extension of the NUTS algorithm (Hoffman and Gelman, 2014).

```python
import hmc
import autograd.numpy as np
import autograd.scipy.linalg as sla

# Generate random precision and mean parameters for a Gaussian
n_dim = 50
rng = np.random.RandomState(seed=1234)
rnd_eigvec, _ = sla.qr(rng.normal(size=(n_dim, n_dim)))
rnd_eigval = np.exp(rng.normal(size=n_dim) * 2)
prec = (rnd_eigvec / rnd_eigval) @ rnd_eigvec.T
mean = rng.normal(size=n_dim)

# Define potential energy (negative log density) for a Gaussian
# target distribution. The derivative of this function will
# be calculated automatically using autograd
def pot_energy(pos):
    pos_minus_mean = pos - mean
    return 0.5 * pos_minus_mean @ prec @ pos_minus_mean

# Specify Hamiltonian system with isotropic Gaussian kinetic energy
system = hmc.systems.IsotropicEuclideanMetricSystem(pot_energy)

# Hamiltonian is separable therefore use explicit leapfrog integrator
integrator = hmc.integrators.LeapfrogIntegrator(system, step_size=0.15)

# Use dynamic integration-time HMC implementation with multinomial sampling 
# from generated trajectories
sampler = hmc.samplers.DynamicMultinomialHMC(system, integrator, rng)

# Sample an initial state from zero-mean isotropic Gaussian
init_state = hmc.states.HamiltonianState(rng.normal(size=n_dim))

# Sample a Markov chain with 1000 transitions
pos_chain, chain_stats = sampler.sample_chain(1000, init_state)

# Print RMSE in mean estimate
mean_rmse = np.mean((pos_chain.mean(0) - mean)**2)**0.5
print(f'Mean estimate RMSE: {mean_rmse}')

# Print average acceptance probability
mean_accept_prob = chain_stats['accept_prob'].mean()
print(f'Mean accept prob: {mean_accept_prob:0.2f}')
```

## To do

  * Static multinomial sampler (Betancourt, 2017)
  * Dynamic integration time using slice sampling (Hoffman and Gelman, 2014)
  * Dual-averaging step-size adaptation (Hoffman and Gelman, 2014)
  * Windowed sampling (Neal, 1994)
  * Look-Ahead HMC (Sohl-Dickenstein et al., 2014)
  * Exponential integration (Chao et al., 2015)
  * Continuous tempering (Graham and Storkey, 2017b)
  * Non-Gaussian kinetic energies (Livingstone et al., 2017)
  * Magnetic HMC (Lu et al., 2017)
  * Relativistic HMC (Tripuraneni et al., 2017)

## References

  1. Duane, S., Kennedy, A.D., Pendleton, B.J. and Roweth, D., 1987.
     Hybrid Monte Carlo. Physics letters B, 195(2), pp.216-222.
  2. Mackenzie, P.B., 1989. An improved hybrid Monte Carlo method.
     Physics Letters B, 226(3-4), pp.369-371.
  3. Horowitz, A.M., 1991. A generalized guided Monte Carlo algorithm.
     Phys. Lett. B, 268(CERN-TH-6172-91), pp.247-252.
  4. Neal, R. M., 1994. An improved acceptance procedure for the hybrid Monte
     Carlo algorithm. Journal of Computational Physics, 111:194–203.
  6. Hartmann, C. and Schütte, C., 2005. A constrained hybrid Monte‐Carlo
     algorithm and the problem of calculating the free energy in several
     variables. ZAMM ‐ Journal of Applied Mathematics and Mechanics, 85(10),
     pp.700-710.
  7. Neal, R.M., 2011. MCMC using Hamiltonian dynamics.
     Handbook of Markov Chain Monte Carlo, 2(11), p.2.
  8. Girolami, M. and Calderhead, B., 2011. Riemann manifold Langevin and
     Hamiltonian Monte Varlo methods. Journal of the Royal Statistical Society:
     Series B (Statistical Methodology), 73(2), pp.123-214.
  9. Brubaker, M., Salzmann, M. and Urtasun, R., 2012. A family of MCMC methods
     on implicitly defined manifolds. In Artificial intelligence and statistics
     (pp. 161-172).
 10. Betancourt, M., 2013. A general metric for Riemannian manifold Hamiltonian
     Monte Carlo. In Geometric science of information (pp. 327-334).
 11. Hoffman, M.D. and Gelman, A., 2014. The No-U-turn sampler: adaptively
     setting path lengths in Hamiltonian Monte Carlo. Journal of Machine
     Learning Research, 15(1), pp.1593-1623.
 12. Sohl-Dickstein, J., Mudigonda, M. and DeWeese, M., 2014. Hamiltonian Monte
     Carlo Without Detailed Balance. In International Conference on Machine
     Learning (pp. 719-726).
 13. Shahbaba, B., Lan, S., Johnson, W.O. and Neal, R.M., 2014.
     Split Hamiltonian Monte Carlo. Statistics and Computing, 24(3), pp.339-349.
 14. Chao, W.L., Solomon, J., Michels, D. and Sha, F., 2015. Exponential
     integration for Hamiltonian Monte Carlo. In International Conference on
     Machine Learning (pp. 1142-1151).
 15. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
     arXiv preprint arXiv:1701.02434.
 16. Lu, X., Perrone, V., Hasenclever, L., Teh, Y.W. and Vollmer, S., 2017.
     Relativistic Monte Carlo. In Artificial Intelligence and Statistics
     (pp. 1236-1245).
 17. Tripuraneni, N., Rowland, M., Ghahramani, Z. and Turner, R., 2017.
     Magnetic Hamiltonian Monte Carlo. In International Conference on Machine
     Learning (pp. 3453-3461).
 18. Livingstone, S., Faulkner, M.F. and Roberts, G.O., 2017. Kinetic energy
     choice in Hamiltonian/hybrid Monte Carlo. arXiv preprint arXiv:1706.02649.
 18. Graham, M.M. and Storkey, A.J., 2017a. Asymptotically exact inference in differentiable generative models. Electronic Journal of Statistics, 11(2), pp.5105-5164.
 19. Graham, M.M. and Storkey, A.J., 2017b. Continuously tempered Hamiltonian
     Monte Carlo. In Proceedings of the 33rd Conference on Uncertainty in
     Artificial Intelligence.
 20. Lelièvre, T., Rousset, M. and Stoltz, G., 2018. Hybrid Monte Carlo methods
     for sampling probability measures on submanifolds. arXiv preprint
     arXiv:1807.02356.
