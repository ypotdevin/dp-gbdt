# inspired by https://medium.com/capital-one-tech/bashing-the-bash-replacing-shell-scripts-with-python-d8d201bc0989

# span a hyperparameter grid, suited to the specific case (like
# dp-optimizing, leaky optimizing or no optimizing), pick randomly and
# uniformly configurations from that space (up to a certain number) and
# then run the DP-GBDT-Programm via CLI with the sampled configuration.

import argparse
import multiprocessing
import os
import socket
import subprocess
import typing
from itertools import chain, product
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import ParameterSampler
from sklearn.utils.fixes import loguniform

Setting = typing.Dict[str, typing.Any]
Settings = Iterable[Setting]


class Uniform:
    def __init__(self, lb: float = 0.0, ub: float = 1.0):
        self.lb = lb
        self.ub = ub

    def rvs(self, random_state=None):
        return stats.uniform.rvs(
            loc=self.lb, scale=self.ub - self.lb, size=1, random_state=random_state
        )[0]


def basic_setting_space_abalone(num_settings, seed) -> Settings:
    """A generous search space to look for well performing
    hyperparameters (on the abalone dataset).

    This space may be adapted for DP-rMSE tree rejection, leaky tree
    rejection and also disabled tree rejection.
    """
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


def basic_quantile_combination_space(num_qs, num_settings, seed) -> Settings:
    grid = {
        # generic
        "--nb-trees": [1, 2, 5, 10, 20, 50],
        "--max-depth": [1, 2, 5, 10, 20],
        "--learning-rate": loguniform(1e-2, 1e1),
        "--l2-lambda": loguniform(1e-2, 1e1),
        "--l2-threshold": loguniform(1e-1, 2.9 * 1e1),
    }
    for i in range(num_qs):
        grid[f"--quantile-combination-rejection-q{i}"] = Uniform(lb=0.5, ub=1.0)
        grid[f"--quantile-combination-rejection-w{i}"] = Uniform(lb=0.0, ub=1.0)
    yield from ParameterSampler(grid, n_iter=num_settings, random_state=seed)


def basic_dp_rmse_space_abalone(num_settings, seed) -> Settings:
    grid = {
        # generic
        "--dp-rmse-gamma": [2.0],
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


def basic_approx_dp_rmse_laplace_space_abalone(delta, num_settings, seed) -> Settings:
    grid = {
        # generic
        "--nb-trees": [1, 2, 5, 10, 20, 50],
        "--max-depth": [1, 2, 5, 10, 20],
        "--learning-rate": loguniform(1e-2, 1e1),
        "--l2-lambda": loguniform(1e-2, 1e1),
        "--l2-threshold": loguniform(1e-1, 2.9 * 1e1),
        "--rejection-failure-prob": [delta],
        # abalone specific
        "--dataset": ["abalone"],
        "--num-samples": [4177],
        "--error-upper-bound": Uniform(lb=3.0, ub=40.0),
    }
    yield from ParameterSampler(grid, n_iter=num_settings, random_state=seed)


def basic_leaky_space(num_settings, seed) -> Settings:
    grid = {
        "--nb-trees": [1, 2, 5, 10, 20, 50],
        "--max-depth": [1, 2, 5, 10, 20],
        "--learning-rate": loguniform(1e-2, 1e1),
        "--l2-lambda": loguniform(1e-2, 1e1),
        "--l2-threshold": loguniform(1e-1, 2.9 * 1e1),
    }
    yield from ParameterSampler(grid, n_iter=num_settings, random_state=seed)


def round_floats(settings: Settings, digits=3) -> Setting:
    keys = (
        [
            "--dp-rmse-gamma",
            "--learning-rate",
            "--l2-lambda",
            "--l2-threshold",
            "--error-upper-bound",
            "--quantile-rejection-q",
        ]
        + [f"--quantile-combination-rejection-q{i}" for i in range(5)]
        + [f"--quantile-combination-rejection-w{i}" for i in range(5)]
        + [f"--quantile-linear-combination-rejection-q{i}" for i in range(5)]
        + [f"--quantile-linear-combination-rejection-c{i}" for i in range(5)]
    )
    for setting in settings:
        for key in keys:
            if key in setting:
                setting[key] = round(setting[key], digits)
        yield setting


def add_repetitions(settings: Settings, num_repetitions: int, rng) -> Settings:
    for setting in settings:
        seeds = rng.integers(2 ** 30 - 1, size=num_repetitions)
        for seed in seeds:
            assert "--seed" not in setting.keys()
            yield {
                "--seed": seed,
                **setting,
            }


def _mult_by_flag(flag: str, settings: Settings) -> Settings:
    for setting in settings:
        assert flag not in setting.keys()
        yield {
            flag: "",
            **setting,
        }


def _mult_by_single_hyperparam(
    label: str, values: Iterable[any], settings: Settings
) -> Settings:
    for setting in settings:
        for value in values:
            assert label not in setting.keys()
            yield {
                label: value,
                **setting,
            }


def add_ensemble_privacy_budgets(
    budgets: Iterable[float], settings: Settings, label: str = None
) -> Settings:
    if label is None:
        label = "--ensemble-privacy-budget"
    yield from _mult_by_single_hyperparam(label, budgets, settings)


def add_rejection_privacy_budgets(
    budgets: Iterable[float], settings: Settings, label: str = None
) -> Settings:
    if label is None:
        label = "--rejection-budget"
    yield from _mult_by_single_hyperparam(label, budgets, settings)


def add_quantile_rejection(
    qs: Iterable[float], settings: Settings, label: str = None
) -> Settings:
    if label is None:
        label = "--quantile-rejection-q"
    settings = _mult_by_flag("--quantile-rejection", settings)
    settings = _mult_by_single_hyperparam(label, qs, settings)
    return settings


def add_quantile_linear_combination_rejection(
    qss: Iterable[Iterable[float]],
    coefficientss: Iterable[Iterable[float]],
    settings: Settings,
) -> Settings:
    settings = _mult_by_flag("--quantile-linear-combination-rejection", settings)
    for (qs, coefficients) in zip(qss, coefficientss):
        for (i, (q, coefficient)) in enumerate(zip(qs, coefficients)):
            settings = _mult_by_single_hyperparam(
                f"--quantile-linear-combination-rejection-q{i}", [q], settings
            )
            settings = _mult_by_single_hyperparam(
                f"--quantile-linear-combination-rejection-c{i}", [coefficient], settings
            )
        yield from settings


def add_abalone(settings: Settings) -> Settings:
    settings = _mult_by_single_hyperparam("--num-samples", [4177], settings)
    settings = _mult_by_single_hyperparam("--dataset", ["abalone"], settings)
    return settings


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
    seed = rng.integers(2 ** 30 - 1)
    settings = basic_setting_space_abalone(num_settings, seed)
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


def to_command_line(setting: Setting) -> str:
    args = " ".join(chain(*[[k, str(v)] for (k, v) in setting.items()]))
    command_line = f"./run --log-level info {args}"
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


def csv_to_settings(
    filename: str,
    columns: Iterable[str] = None,
    renaming: typing.Dict[str, str] = None,
) -> Settings:
    """
    Parameters
    ----------
    filename: str
        The path (including filename and extension) to the csv file
        containing benchmark (or hyperparameter optimization) results.
    columns: iterable of str
        The columns to extract hyperparameters from (this may not be
        enough to yield complete settings).
    renaming: a mapping of str to str
        A translation of column labels to setting hyperparameter (e.g.
        `param_ensemble_privacy_budget` to `--ensemble-privacy-budget`).
    """
    if columns is None:
        columns = [
            "param_ensemble_privacy_budget",
            "param_l2_lambda",
            "param_l2_threshold",
            "param_learning_rate",
            "param_max_depth",
            "param_nb_trees",
        ]
    if renaming is None:
        renaming = dict(
            param_ensemble_privacy_budget="--ensemble-privacy-budget",
            param_l2_lambda="--l2-lambda",
            param_l2_threshold="--l2-threshold",
            param_learning_rate="--learning-rate",
            param_max_depth="--max-depth",
            param_nb_trees="--nb-trees",
        )
    df = pd.read_csv(filename)
    df = df[columns]
    df.rename(renaming, axis=1, inplace=True)
    for (_, row) in df.iterrows():
        yield row.to_dict()


def quantile_convex_combination_settings(num_cores, eval_dir, rng):
    settings = basic_quantile_combination_space(
        num_qs=3, num_settings=2 * num_cores, seed=rng.integers(2 ** 30 - 1)
    )
    settings = _mult_by_flag("--quantile-combination-rejection", settings)
    settings = add_ensemble_privacy_budgets([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], settings)
    settings = add_abalone(settings)
    settings = round_floats(settings)
    settings = add_repetitions(settings, 2, rng)
    settings = add_filenames(settings, eval_dir + "leaky-quantile-combination-opt_{}")
    return settings


def quantile_linear_combination_settings(num_cores, eval_dir, rng):
    settings = basic_leaky_space(
        num_settings=2 * num_cores, seed=rng.integers(2 ** 30 - 1),
    )
    settings = add_quantile_linear_combination_rejection(
        qss=[[0.5, 0.85, 0.95],],
        coefficientss=[[-0.30757628, 0.32697374, 0.41003781],],
        settings=settings,
    )
    settings = round_floats(settings)
    settings = add_ensemble_privacy_budgets([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], settings)
    settings = add_abalone(settings)
    settings = add_repetitions(settings, 2, rng)
    settings = add_filenames(
        settings, eval_dir + "quantile-linear-combination-leaky-opt_{}"
    )
    return settings


def quantile_linear_combination_settings2(num_cores, eval_dir, rng):
    settings = basic_leaky_space(
        num_settings=2 * num_cores, seed=rng.integers(2 ** 30 - 1),
    )
    settings = add_quantile_linear_combination_rejection(
        qss=[[0.5, 0.95],],
        coefficientss=[[0.53619529, 0.33035797],],
        settings=settings,
    )
    settings = round_floats(settings)
    settings = add_ensemble_privacy_budgets([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], settings)
    settings = add_abalone(settings)
    settings = add_repetitions(settings, 2, rng)
    settings = add_filenames(
        settings, eval_dir + "quantile-linear-combination-leaky-opt_{}"
    )
    return settings


def approx_dp_rmse_laplace_settings(num_cores, eval_dir, rng):
    settings = basic_approx_dp_rmse_laplace_space_abalone(
        delta=1e-4, num_settings=2 * num_cores, seed=rng.integers(2 ** 30 - 1),
    )
    settings = _mult_by_flag("--dp-laplace-rmse-rejection", settings)
    settings = add_ensemble_privacy_budgets([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], settings)
    settings = add_rejection_privacy_budgets([0.01, 0.05, 1.0, 5.0], settings)
    settings = round_floats(settings)
    settings = add_repetitions(settings, 2, rng)
    settings = add_filenames(settings, eval_dir + "approx-laplace-dp-opt_{}")
    return settings


def dp_rmse_settings(num_cores, eval_dir, rng):
    settings = basic_dp_rmse_space_abalone(
        num_settings=2 * num_cores, seed=rng.integers(2 ** 30 - 1),
    )
    settings = _mult_by_flag("--dp-rmse-tree-rejection", settings)
    settings = add_ensemble_privacy_budgets([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], settings)
    settings = add_rejection_privacy_budgets([0.01, 0.05, 1.0, 5.0], settings)
    settings = round_floats(settings)
    settings = add_repetitions(settings, 2, rng)
    settings = add_filenames(settings, eval_dir + "dp-rmse_{}")
    return settings


def main():
    parser = argparse.ArgumentParser(
        description="Perform random search over the hyperparameter space."
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=4,
        help="The number of CPU cores to use for hyperparameter search "
        "(affects the number of tested hyperparameter combinations).",
    )
    args = parser.parse_args()

    hostname = socket.gethostname()
    eval_dir = f"evaluation/{hostname}/quantile-linear-comb-leaky2/abalone/"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    rng = np.random.default_rng()

    settings = quantile_linear_combination_settings2(args.num_cores, eval_dir, rng)

    with multiprocessing.Pool(args.num_cores) as p:
        p.map(run_benchmark, settings, chunksize=1)


if __name__ == "__main__":
    main()
