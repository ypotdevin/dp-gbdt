# Differentially Private Gradient Boosted Decision Trees (DP-GBDTs)

In this repository we implement differentially private Gradient Boosted Decision Trees, DP-GBDTs (technically, we only implemented *regression* trees to far, not decision trees).
Our implementation is largely based on *DPBoost*, an algorithm by Li et al., mentioned in [Privacy-Preserving Gradient Boosting Decision Trees](http://arxiv.org/abs/1911.04209). Additionally to a »vanilla« implementation of DPBoost, our implementations offers a technique we call *tree rejection*. Basically, instead of building the tree ensemble the usual way (i.e. *accepting* each newly generated tree and adding it to the ensemble), we evaluate each newly created tree's contribution to the ensemble. If the contribution is positive, *accept* the tree, otherwise *reject* it.

## Side channel leakage and trusted computing
To further secure our implementation, we aim to make harden it against side channel attacks (work in progress) and enable Intel SGX (work in progress).
The directory `hardened_gbdt` contains the hardened code and the ditectory `hardened_sgx_gbdt` is supposed to contain a hardened and SGX compatible code.

## Requirements
Our implementation is successfully tested on Ubuntu 20.04 and 22.04 systems (in principle it should also work on other Linuxes, Windows and MacOS).
We require `conda` to be present on the target system. Make sure to have the build essentials (e.g. via `sudo apt install build-essential`).

## Installation
1. clone our repository
2. create a conda environment using our `environment.ymv` file:
   ```bash
   cd dp-gbdt
   conda env create -f hardering/hardened_gbdt/environment.yml
   ```
3. activate that environment: `conda activate dp_gbdt_evaluation`
4. build the python extension:
   ```bash
   cd hardening/hardened_gbdt
   python setup.py build_ext --inplace
   ```

5. test the python extension via `python example_main.py`. The output should be
   ```bash
   Mean - RMSE: 3.248705
   Depth first growth - RMSE: 17.326674
   ```

## Usage
### Option 1: Python extension
Import the `dpgbdt.py` module and use the `DPGBDTRegressor` class for regression. Its interface matches the sklearn API. That means the main methods are `fit` and `predict`.

### Option 2: C++ interface
Include `estimator.h` and use the class `Estimator`. It offers the methods `fit` and `predict`, similar to the API of the python extension.

## Limitations
- as mentioned above, currently only regression is supported and not classification
- side channel hardening is work in progress (the vanilla DPBoost is mostly hardened, but the tree rejection is not)
- the integration with Intel SGX is work in progress

## Repeating our evaluaiton
TODO

## Diagnosis
TODO
