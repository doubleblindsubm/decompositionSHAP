# Code accompanying paper 'Decomposition of the Shapley Value to explain Model and Result'

## Install instructions

- clone the Github respository
- install Conda environment (requires Conda): `conda env create -f environment.yml`

## Files and important functions

- `tests` tests of the implemented code (Note: some tests can fail sometimes because they are stochastic)
- `distributions.py` holds different implementations of assumed data distributions
- `experiments.py` has all code to do all experiments of the paper and generate all results
- `force_dependent.py` holds implementation of force plot visualizations with interventional and dependent effects
    - function `force_dependent_plot(...)` visualizes the given SHAP values and interventional effects in a matplotlib
      force plot. See code documentation for information on arguments.
- `helpers.py` some helpers functions
- `kernel_dependent.py` holds functions that can compute interventional and dependent effects
    - class `DependentKernelExplainer` explains output of any function for any situation by decomposing the Shapley
      value into an interventional and dependent effect. See code documentation for information on constructor
      arguments.
    - function `shap_values(...)` estimates the SHAP values and interventional effects for a set of samples. See code
      documentation for information on arguments.