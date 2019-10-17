#!/usr/bin/env python2

from setuptools import setup

if __name__ == '__main__':
    setup(name='mici',
          description=(
              'MCMC samplers based on simulating Hamiltonian dynamics on a '
              'manifold'),
          author='Matt Graham',
          url='https://github.com/matt-graham/hamiltonian-monte-carlo.git',
          packages=['mici'])
