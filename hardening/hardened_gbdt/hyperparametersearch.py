# inspired by https://medium.com/capital-one-tech/bashing-the-bash-replacing-shell-scripts-with-python-d8d201bc0989

# span a hyperparameter grid, suited to the specific case (like
# dp-optimizing, leaky optimizing or no optimizing), pick randomly and
# uniformly configurations from that space (up to a certain number) and
# then run the DP-GBDT-Programm via CLI with the sampled configuration.

import multiprocessing
import subprocess
import typing
from itertools import chain, product

import numpy as np
from scipy import stats
from sklearn.model_selection import ParameterSampler
from sklearn.utils.fixes import loguniform

Setting = typing.Dict[str, typing.Any]
Settings = typing.Iterable[Setting]


class Uniform:
    def __init__(self, lb: float = 0.0, ub: float = 1.0):
        self.lb = lb
        self.ub = ub

    def rvs(self, random_state=None):
        return stats.uniform.rvs(
            loc=self.lb, scale=self.ub - self.lb, size=1, random_state=random_state
        )[0]


def varying_settings_abalone(num_settings, seed) -> Settings:
    grid = {
        # generic
        "--gamma": loguniform(1 + 1e-2, 1e2),
        "--nb-trees": [1, 2, 5, 10, 20, 50],
        "--max-depth": [1, 2, 5, 10, 20],
        "--learning-rate": loguniform(1e-2, 1e1),
        "--l2-lambda": loguniform(1e-2, 1e1),
        "--l2-threshold": loguniform(1e-1, 2.9 * 1e1),
        # abalone specific
        "--dataset": ["abalone"],
        "--num-samples": [4177],
        "--error-upper-bound": Uniform(lb=3.0, ub=40.0),
    }
    yield from ParameterSampler(grid, n_iter=num_settings, random_state=seed)


def round_floats(settings: Settings, digits=3) -> Setting:
    keys = [
        "--gamma",
        "--learning-rate",
        "--l2-lambda",
        "--l2-threshold",
        "--error-upper-bound",
    ]
    for setting in settings:
        for key in keys:
            setting[key] = round(setting[key], digits)
        yield setting


def add_repetitions(settings: Settings, num_repetitions: int, rng) -> Settings:
    for setting in settings:
        seeds = rng.integers(2**30 - 1, size=num_repetitions)
        for seed in seeds:
            assert "--seed" not in setting.keys()
            yield {
                "--seed": seed,
                **setting,
            }


def add_dp_opt_settings(settings: Settings) -> Settings:
    ep_budgets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    op_budgets = [0.01, 0.05, 1.0, 5.0]
    for setting in settings:
        for (epb, opb) in product(ep_budgets, op_budgets):
            assert "--ensemble-privacy-budget" not in setting.keys()
            assert "--optimization-privacy-budget" not in setting.keys()
            yield {
                "--ensemble-privacy-budget": epb,
                "--optimization-privacy-budget": opb,
                **setting,
            }


def add_nodp_opt_setting(settings: Settings, variant: str) -> Settings:
    ep_budgets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for setting in settings:
        for epb in ep_budgets:
            assert "--ensemble-privacy-budget" not in setting.keys()
            assert variant not in setting.keys()
            _setting = {
                "--ensemble-privacy-budget": epb,
                variant: "",
                **setting,
            }
            _setting.pop("--gamma")
            _setting.pop("--error-upper-bound")
            yield _setting


def add_leaky_opt_settings(settings: Settings) -> Settings:
    yield from add_nodp_opt_setting(settings, "--leaky-optimization")


def add_no_opt_settings(settings: Settings) -> Settings:
    yield from add_nodp_opt_setting(settings, "--no-optimization")


def add_filenames(settings: Settings, template) -> Settings:
    for i, setting in enumerate(settings):
        filename = template.format(i)
        assert "--results-file" not in setting.keys()
        assert "log_filename" not in setting.keys()
        yield {
            "--results-file": filename + ".csv",
            "log_filename": filename + ".log",
            **setting,
        }


def settings_loop(
    num_settings: int, num_repetitions: int, templates: typing.Dict[str, str], rng
) -> Settings:
    # using 2 ** 32 - 1 would apparently lead to std::out_of_range in
    # the C++ application
    seed = rng.integers(2**30 - 1)
    settings = varying_settings_abalone(num_settings, seed)
    settings = round_floats(settings)
    settings = add_repetitions(settings, num_repetitions=num_repetitions, rng=rng)
    materialized_settings = list(settings)

    no_opt_settings = add_no_opt_settings(materialized_settings)
    no_opt_settings = add_filenames(no_opt_settings, templates["no_opt_template"])

    leaky_opt_settings = add_leaky_opt_settings(materialized_settings)
    leaky_opt_settings = add_filenames(
        leaky_opt_settings, templates["leaky_opt_template"]
    )

    dp_opt_settings = add_dp_opt_settings(materialized_settings)
    dp_opt_settings = add_filenames(dp_opt_settings, templates["dp_opt_template"])

    return chain(no_opt_settings, leaky_opt_settings, dp_opt_settings)


def to_command_line(setting: Setting) -> typing.List[str]:
    args = " ".join(chain(*[[k, str(v)] for (k, v) in setting.items()]))
    command_line = f"./run --log-level debug {args}"
    return command_line


def run_benchmark(setting: Setting):
    log_filename = setting.pop("log_filename")
    command_line = to_command_line(setting)
    # Somehow it does not work to capture the output of the C++ program
    # in process.stdout (I don't know why) ...
    # However, the workaround via shell=True and shell output
    # redirection does what it should.
    command_line += f" > {log_filename}"
    subprocess.run(args=command_line, shell=True, check=True)


def get_filenames(settings: Settings) -> typing.List[str]:
    csv_names = [setting["--results-file"] for setting in settings]
    log_names = ["{}.log".format(csv_name.split(".csv")[0]) for csv_name in csv_names]
    return log_names


def main():
    rng = np.random.default_rng()
    templates = dict(
        no_opt_template="evaluation/abalone/no-opt_{}",
        leaky_opt_template="evaluation/abalone/leaky-opt_{}",
        dp_opt_template="evaluation/abalone/dp-opt_{}",
    )

    num_cores = 56
    settings = settings_loop(10 * num_cores, 2, templates, rng)
    with multiprocessing.Pool(num_cores) as p:
        p.map(run_benchmark, settings)


if __name__ == "__main__":
    main()
