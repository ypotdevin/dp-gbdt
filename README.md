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

## CPP-DP-GBDT

Components:
- **main.cpp**
If you just want to play around with different parameters, different logging levels etc. this is the place to do it. Use `make`, then `./run`.
Note, this version does not use multithreading to make stuff obvious to debug. So it's naturally slower.
- **benchmark.cpp**
this component demonstrates the potential speed of the CPP implementation. It e.g. takes advantage of multithreading. To use it, adjust _benchmark.cpp_ according to your needs, compile the project with `make fast`, then do `./run --bench`.
- **evaluation.cpp**
this component allows running the model successively with multiple privacy budgets. It will create a .csv in the results/ directory. In there you can use _plot.py_ to create plots from these files. To compile and run, use `make fast`, then `./run --eval`. Be aware, running the code with _use_dp=false_ resp. _privacy_budget=0_ is much slower than using dp (because it uses **all** sample rows for each tree).
- **verification.cpp**
this is just to show that our algorithm results are consistent with the python implementation. (you can run this with _verify.sh_). It works by running both implementations without randomness, and then comparing intermediate values.


### Running
- **Running C++ gbdt**
```bash
cd hardening/hardened_gbdt/
make
./run --ensemble-privacy-budget 1.0 \
      --optimization-privacy-budget 0.01 \
      --gamma 1.3 \
      --nb-trees 1 \
      --max_depth 5 \
      --learning-rate 5.01 \
      --l2-lambda 7.7 \
      --l2-threshold 0.3 \
      --dataset abalone \
      --num-samples 4177 \
      --error-upper-bound 13.8 \
      --log-level debug \
      --results-file out.csv
```

## Limitations

- the C++ implementations can only do **regression** (no classification).
- There are still small DP problems, such as
  - init\_score is leaking information about what values are present in a dataset
  - the partial (and not complete) use of constant-time floating point operations
