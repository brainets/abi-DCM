# abi-DCM
A Python toolbox for Automated Bayesian Inference in Dynamic Causal Modelling (DCM)

# Description

This repository contains Python code for efficient Bayesian Inference in DCM, including routines for Gradient-Descent and Markov Chain Monte Carlo schemes.

# Dependencies

- [NumPyro](https://num.pyro.ai/)

- [JAX](https://docs.jax.dev/en/latest/)

- [JAXopt](https://jaxopt.github.io)

- [vbjax](https://github.com/ins-amu/vbjax)

- [frites](https://brainets.github.io/frites/)

# Installation

## Download the code from GitHub
git clone https://github.com/brainets/abi-DCM.git $HOME/abi-DCM \
cd $HOME/abi-dcm/

## Create a Python environment
conda env create -f environment.yml \
conda activate abi-DCM \

## Run the code on JupyterLab
cd $HOME/abi-dcm/examples \
jupyter-lab &

# Acknowledgements

This research has been supported by EU’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS 2.0 Project).
![EU logo](./eu_logo.jpg)
