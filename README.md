# Hamiltonian Monte Carlo

Implementations of various Hamiltonian dynamics based MCMC samplers in idiomatic Python code. A modular design is used to as far as possible allowing mixing and matching elements of different proposed extensions to the original HMC algorithm proposed in Duane et al. (1987).

Users can either directly specific the required model derivatives or when available [autograd](https://github.com/HIPS/autograd) can be used to construct the required derivative functions using automatic differentiation.

## Dependencies

To install and use the package the minimal requirement is a Python 3 environment with `numpy` and `scipy` installed, with `autograd` an optional additional requirement to allow automatic calculation of model derivative functions.

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
    Stoltz, 2018)

Numerical integrators

  * Explicit leapfrog for separable Hamiltonian systems
  * Implicit leapfrog for non-separable Hamiltonian systems
  * Geodesic leapfrog for constrained Hamiltonian systems

## To do

  * Static multinomial sampler (Betancourt, 2017)
  * Dynamic integration time using slice sampling (Hoffman and Gelman, 2014)
  * Dual-averaging step-size adaptation (Hoffman and Gelman, 2014)
  * Windowed sampling (Neal, 1994)
  * Look-Ahead HMC (Sohl-Dickenstein et al., 2014)
  * Exponential integration (Chao et al., 2015)
  * Continuous tempering (Graham and Storkey, 2017)
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
 13. Chao, W.L., Solomon, J., Michels, D. and Sha, F., 2015. Exponential
     integration for Hamiltonian Monte Carlo. In International Conference on
     Machine Learning (pp. 1142-1151).
 14. Betancourt, M., 2017. A conceptual introduction to Hamiltonian Monte Carlo.
     arXiv preprint arXiv:1701.02434.
 15. Lu, X., Perrone, V., Hasenclever, L., Teh, Y.W. and Vollmer, S., 2017.
     Relativistic Monte Carlo. In Artificial Intelligence and Statistics
     (pp. 1236-1245).
 16. Tripuraneni, N., Rowland, M., Ghahramani, Z. and Turner, R., 2017.
     Magnetic Hamiltonian Monte Carlo. In International Conference on Machine
     Learning (pp. 3453-3461).
 17. Livingstone, S., Faulkner, M.F. and Roberts, G.O., 2017. Kinetic energy
     choice in Hamiltonian/hybrid Monte Carlo. arXiv preprint arXiv:1706.02649.
 18. Graham, M.M. and Storkey, A.J., 2017. Continuously tempered Hamiltonian
     Monte Carlo. In Proceedings of the 33rd Conference on Uncertainty in
     Artificial Intelligence.
 19. Lelièvre, T., Rousset, M. and Stoltz, G., 2018. Hybrid Monte Carlo methods
     for sampling probability measures on submanifolds. arXiv preprint
     arXiv:1807.02356.
