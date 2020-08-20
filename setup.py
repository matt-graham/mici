import setuptools

setuptools.setup(
    name='mici',
    version='0.1.9',
    author='Matt Graham',
    description=(
        'MCMC samplers based on simulating Hamiltonian dynamics on a manifold'
    ),
    long_description=(
        'Mici is a Python package providing implementations of Markov chain '
        'Monte Carlo (MCMC) methods for approximate inference in probabilistic'
        ' models, with a particular focus on MCMC methods based on simulating '
        'Hamiltonian dynamics on a manifold.'
    ),
    url='https://github.com/matt-graham/mici.git',
    project_urls={
        'Documentation': 'https://matt-graham.github.io/mici/docs'
    },
    packages=['mici'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers'
    ],
    keywords='inference sampling MCMC HMC',
    license='MIT',
    license_file='LICENSE',
    install_requires=['numpy>=1.17', 'scipy>=1.1'],
    python_requires='>=3.6',
    extras_require={
        'autodiff':  ['autograd>=1.3', 'multiprocess>=0.70']
    }
)
