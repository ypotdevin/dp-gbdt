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


## Limitations
- the C++ implementations can only do **regression** (no classification).
- There are still small DP problems, such as
  - init\_score is leaking information about what values are present in a dataset
  - the partial (and not complete) use of constant-time floating point operations
