FROM ubuntu:20.04

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt -y install libspdlog-dev icdiff git g++ make

RUN git clone https://github.com/ypotdevin/dp-gbdt.git

WORKDIR /dp-gbdt/hardening/hardened_gbdt
RUN make
CMD ["./run", \
    "--ensemble-privacy-budget", "1.0", \
    "--optimization-privacy-budget", "0.01", \
    "--gamma", "1.3", \
    "--nb-trees", "1", \
    "--max_depth", "5", \
    "--learning-rate", "5.0", \
    "--l2-lambda", "7.7", \
    "--l2-threshold", "0.3", \
    "--dataset", "abalone", \
    "--num-samples", "4177", \
    "--error-upper-bound", "13.8", \
    "--seed", "42", \
    "--log-level", "debug", \
    "--results-file", "out.csv"]