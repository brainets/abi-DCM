from setuptools import setup, find_packages

setup(
    name='abi-DCM',
    version='1.0',
    author='Pedro Garcia-Rodriguez',
    author_email='pedro.garcia-rodriguez@univ-amu.fr, andrea.brovelli@univ-amu.fr',
    packages=find_packages(),
    url='https://github.com/brainets/abi-DCM',
    keywords = ['Bayesian Inference', 'DCM', 'Gradient-Descent', 'Markov Chain Monte Carlo'],
    license='BSD 3-Clause',
    description='A Python library for efficient Bayesian Inference in DCM, including routines for Gradient-Descent and Markov Chain Monte Carlo schemes',
)

