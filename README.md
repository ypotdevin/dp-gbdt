# Enclave hardening for private ML

This in an optimized implementation of

## Versions

To keep the code clean there exist 2 different versions:

`hardened_gbdt`: the hardened version of cpp_gbdt. It reflects the minimum changes that are needed to achieve ε-differential privacy. The code does not run inside an SGX enclave.

`hardened_sgx_gbdt`: the final combination of hardened_gbdt and enclave_gbdt.



## Requirements
tested on a fresh Ubuntu 20.04.2 VM
```bash
sudo apt-get install libspdlog-dev
sudo apt-get install icdiff
sudo apt install python3-pip
python3 -m pip install -r code/python_gbdt/requirements.txt
```
and add this to your .bashrc
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/.../code/python_gbdt
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
- **Running the verification script to compare Python and C++**
```bash
cd code/
./verify.sh
```
- **Running C++ gbdt**
```bash
cd code/cpp_gbdt/
make
./run
(./run --verify)
(./run --eval)
(./run --bench)
```

## Limitations

- the C++ implementations can only do **regression** (no classification).
- There are still small DP problems, such as
  - init\_score is leaking information about what values are present in a dataset
  - the partial (and not complete) use of constant-time floating point operations
