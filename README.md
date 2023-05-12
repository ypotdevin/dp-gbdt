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
   cd dp-gbdt/hardened_gbdt
   conda env create -f environment.yml
   ```
3. activate that environment: `conda activate dp_gbdt_evaluation`
4. build the python extension:
   ```bash
   python setup.py build_ext --inplace
   ```

5. test the python extension via `python example_main.py`. The output should be something similar to
   ```bash
   Mean - RMSE: 3.248705
   Depth first growth - RMSE: 17.326674
   ```

## Usage
### Option 1: Python extension
Import the [`dpgbdt.py`](hardened_gbdt/dpgbdt.py) module and use the `DPGBDTRegressor` class for regression. Its interface matches the sklearn API. That means the main methods are `fit` and `predict`.

### Option 2: C++ interface
Include `estimator.h` and use the class `Estimator`. It offers the methods `fit` and `predict`, similar to the API of the python extension.

## Integrating new rejection mechanisms
To add a novel way of evaluating a newly created tree's contribution to the ensemble,
1. extend the class `TreeScorer` located in [`include/gbdt/tree_rejection.h`](hardened_gbdt/include/gbdt/tree_rejection.h) and implement the method `score_tree`
2. (optional) declare that class in [`cpp_estimator.pxd`](hardened_gbdt/cpp_estimator.pxd)
3. (optional) wrap that class in cython code by extending the `PyTreeScorer` class in [`py_estimator.pyx`](hardened_gbdt/py_estimator.pyx)
4. (optional) extend the convenience functions `make_tree_scorer` and `_make_tree_scorer_from_self` in [`dpgbdt.py`](hardened_gbdt/dpgbdt.py) to include your new scorer

## Limitations
- as mentioned above, currently only regression is supported and not classification
- side channel hardening is work in progress (the vanilla DPBoost is mostly hardened, but the tree rejection is not)
- the integration with Intel SGX is work in progress

## Repeating our evaluation
We use the module [`hyperparameter_tuning.py`](hardened_gbdt/hyperparameter_tuning.py) to evaluate DP-GBDT on several datasets (located at [`datasets/real`](datasets/real)). Our approach is similar to sklearn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) and [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV), but we implemented our own version, to have control over arrangement of the random seeds.

All the evaluation experiments we performed are listed in that file. For an overview, have a look at the function `select_experiment`, where they are bundled. We've also implemented a command line interface. See `python hyperparameter_tuning -h` for how to use. Just make sure to have a directory set up at `~/shame/dp-gbdt-evaluation`, or provide a different path via the `--local-dir` argument.
