# Enclave hardening for private ML

This in an optimized implementation of "DPBoost" (Li et al.).

## Versions
To keep the code clean there exist 2 different versions:

`hardened_gbdt`: the hardened version of cpp_gbdt. It reflects the minimum changes that are needed to achieve ε-differential privacy. The code does not run inside an SGX enclave.

`hardened_sgx_gbdt`: the final combination of hardened_gbdt and enclave_gbdt.


## Requirements
tested on a fresh Ubuntu 20.04.2 VM
```bash
sudo apt-get install libspdlog-dev
sudo apt-get install icdiff
```


### Running
- **Running C++ gbdt**
```bash
cd hardening/hardened_gbdt/
make
./run --ensemble-privacy-budget 1.0 \
      --dp-rmse-tree-rejection \
      --rejection-budget 0.01 \
      --error-upper-bound 13.8 \
      --dp-rmse-gamma 1.3 \
      --nb-trees 1 \
      --max_depth 5 \
      --learning-rate 5.01 \
      --l2-lambda 7.7 \
      --l2-threshold 0.3 \
      --dataset abalone \
      --num-samples 4177 \
      --log-level debug \
      --seed 42 \
      --results-file out.csv
```

### Command Line Interface (CLI)
| Group | Parameter                           | Type         | Meaning                                                               |
| ----- | ----------------------------------- | ------------ | --------------------------------------------------------------------- |
| 0     | --ensemble-privacy-budget           | float        | The privacy budget used for the ensemble (the boosted trees) creation |
| 1a    | --dp-rmse-tree-rejection            | boolean flag | Perform DP-rMSE tree rejection (using Cauchy distribution)            |
| 1a    | --rejection-budget                  | float        | The per-loop DP budget for the tree rejection mechanism               |
| 1a    | --dp-rmse-gamma                     | float        | Gamma defines the custom distribution to sample from                  |
| 1a    | --error-upper-bound                 | float        | An upper bound to the prediction errors, needed for sensitivity calculation (larger values will be clipped; large values will increase the sensitivity and require more noise, low values do not reflect the true prediction errors) |
| 1b    | --no-tree-rejection                 | boolean flag | Perform no rejection (keep every tree)                                |
| 1c    | --quantile-rejection                | boolean flag | Perform leaky tree rejection via quantiles                            |
| 1c    | --quantile-rejection-q              | float        | The quantile to use (0 <= q <= 1)                                     |
| 1d    | --quantile-combination-rejection    | boolean flag | Perform leaky tree rejection via a convex combination of quantiles    |
| 1d    | --quantile-combination-rejection-q0 | float        | The *first* quantile to use (may be up to 4, depending on how many are provided) |
| 1d    | --quantile-combination-rejection-w0 | float        | The relative weight of the *first* quantile (the weights will be normalized anyway, i.e. post processing will make them sum up to 1.0) |
…
| 1d    | --quantile-combination-rejection-q4 | float        | The *fifth* quantile to use                                            |
| 1d    | --quantile-combination-rejection-w4 | float        | The relative weight of the *fifth* quantile                            |
| 1e    | --dp-laplace-rmse                   | boolean flag | Perform approx.-DP-rMSE tree rejection (using Laplace distribution)    |
| 1e    | --rejection-budget                  | float        | The per-loop DP budget for the tree rejection mechanism                |
| 1e    | --rejection-failure-prob            | float        | The delta from the definition of approximate DP                        |
| 1e    | --error-upper-bound                 | float        | An upper bound to the prediction errors, needed for sensitivity calculation (larger values will be clipped; large values will increase the sensitivity and require more noise, low values do not reflect the true prediction errors) |
| 2     | --nb-trees                | int          | The (maximal) number of trees an ensemble may contain (is affected by tree rejection) |
…

Note: Groups with same number but different letter are mutually exclusive.

## Limitations
- the C++ implementations can only do **regression** (no classification).
- There are still small DP problems, such as
  - init\_score is leaking information about what values are present in a dataset
  - the partial (and not complete) use of constant-time floating point operations
