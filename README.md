# Hamiltonian Monte Carlo

Implementations of various Hamiltonian dynamics based MCMC samplers in
`python` including samplers for both unconstrained systems and systems with
holonomic constraints defined by some (potentially non-linear) constraint
function. When available [autograd](https://github.com/HIPS/autograd) can be
used to automatically calculate gradients of the target log density and/or
constraint function Jacobian.

## Dependencies

Python 2.7 with `numpy` and `scipy` installed current minimal configuration,
with `autograd` an optional requirement.
