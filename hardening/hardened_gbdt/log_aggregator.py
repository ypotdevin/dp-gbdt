# ypo@informatik.uni-kiel.de

import argparse
import pathlib
import sys
import zipfile
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lark import Lark, Transformer

import log_parser

Lines = Iterable[str]


def stripped_lines(filename: str) -> Lines:
    "Read lines from file and remove leading and trailing whitespace."
    with open(filename, "r") as file:
        for line in file:
            yield line.strip()


def filter_lines(substring: str, lines: Lines) -> Lines:
    return [line for line in lines if substring in line]


def parse_lines(lines: Lines, parser: Lark, transformer: Transformer) -> Iterable[Any]:
    return [transformer.transform(parser.parse(line)) for line in lines]


def fast_parse(lines: Lines, parser: Lark) -> Iterable[Any]:
    return [parser.parse(line) for line in lines]


def _parse_diag_lines(lines: Lines) -> Iterable[Any]:
    return parse_lines(
        lines, log_parser.diagnosis_parser(), log_parser.DiagnosisToDict()
    )


def _parse_diag_lines2(filter_str: str, lines: Lines) -> pd.DataFrame:
    lines = filter_lines(filter_str, lines)
    diag_statistics = _parse_diag_lines(lines)
    df = pd.DataFrame(diag_statistics)
    return df


def parse_rmses(lines: Lines) -> pd.DataFrame:
    return _parse_diag_lines2("### diagnosis value 01 ###", lines)


def parse_rmse_approximations(lines: Lines) -> pd.DataFrame:
    return _parse_diag_lines2("### diagnosis value 02 ###", lines)


def parse_quantiles(lines: Lines) -> pd.DataFrame:
    lines = filter_lines("### diagnosis value 03 ###", lines)
    quantile_dict_list = _parse_diag_lines(lines)
    quantile_lists = [d["quantiles"] for d in quantile_dict_list]
    df = pd.DataFrame(
        quantile_lists,
        columns=["q={q:.2f}".format(q=q) for q in np.linspace(0.5, 1.0, 11)],
    )
    return df


def parse_smooth_sens(lines: Lines) -> pd.DataFrame:
    return _parse_diag_lines2("### diagnosis value 04 ###", lines)


def parse_maximizer_local_sens(lines: Lines) -> pd.DataFrame:
    return _parse_diag_lines2("### diagnosis value 05 ###", lines)


def parse_maximizer_k(lines: Lines) -> pd.DataFrame:
    return _parse_diag_lines2("### diagnosis value 06 ###", lines)


def parse_error_vectors(lines: Lines) -> np.ndarray:
    lines = filter_lines("### diagnosis value 18 ###", lines)
    lines = [line.split("### - ")[1] for line in lines]
    tree_indices_and_error_vectors = fast_parse(
        lines, log_parser.fast_diagnosis_parser()
    )
    arr = np.zeros(
        shape=(
            len(tree_indices_and_error_vectors),  # type: ignore
            len(tree_indices_and_error_vectors[0]["absolute_errors"]),  # type: ignore
        )
    )
    for (i, tree_idx_and_error_vector) in enumerate(tree_indices_and_error_vectors):
        # The tree indices may not be strictly consecutive, as DP argmax
        # may fail for same trees. Therefore the tree index is not a
        # suitable array index.
        arr[i] = tree_idx_and_error_vector["absolute_errors"]
    return arr


def parse_tree_idx(lines: Lines) -> pd.DataFrame:
    lines = filter_lines("[ info] [            train] Building dp-tree-", lines)
    tree_indices_and_sample_nums = parse_lines(
        lines, log_parser.tree_idx_parser(), log_parser.TreeIdxToDict()
    )
    df = pd.DataFrame(tree_indices_and_sample_nums)
    return df


def parse_tree_acceptence(lines: Lines) -> pd.DataFrame:
    lines = filter_lines("### diagnosis value 07 ###", lines)
    acceptance_and_stuff = parse_lines(
        lines, log_parser.tree_acceptence_parser(), log_parser.TreeAcceptenceToDict()
    )
    df = pd.DataFrame(acceptance_and_stuff)
    return df


def quantile_log_to_df(lines: Lines) -> pd.DataFrame:
    df = pd.concat(
        [
            parse_rmses(lines),
            parse_rmse_approximations(lines),
            parse_quantiles(lines),
            # parse_tree_idx(lines),
            parse_tree_acceptence(lines),
        ],
        axis=1,
    )
    return df


def approx_dp_rmse_log_to_df(lines: Lines) -> pd.DataFrame:
    df = pd.concat(
        [
            parse_rmses(lines),
            parse_rmse_approximations(lines),
            parse_tree_idx(lines),
            parse_smooth_sens(lines),
            parse_maximizer_local_sens(lines),
            parse_maximizer_k(lines),
        ],
        axis=1,
    )
    return df


def add_job_id(filename: str, df: pd.DataFrame) -> pd.DataFrame:
    "This expects `filename` to be like f'evaluation/{hostname}/{approximation_method}/{dataset}/{prefix}_{execution_number}.log'"
    p = pathlib.Path(filename)
    assert p.parts[0] == "evaluation"
    hostname = p.parts[1]
    assert "ignis" in hostname
    filename = p.stem
    execution_number = filename.split("_")[1]
    job_id = f"{hostname}_{execution_number}"
    return df.assign(job_id=job_id)


def aggregate_logs(
    to_df: Callable[[Lines], pd.DataFrame], files: Iterable[str], num_worker: int = 8
) -> pd.DataFrame:
    def job(arg):
        lines = list(stripped_lines(arg))
        df = to_df(lines)
        df = add_job_id(arg, df)
        return df

    dfs = Parallel(n_jobs=num_worker)(delayed(job)(file) for file in files)
    return pd.concat(dfs)  # type: ignore


def error_vector_dispatch(args: argparse.Namespace) -> None:
    arrays = {}
    if args.zipped:
        with zipfile.ZipFile(args.log_files[0], "r") as zfile:
            for log in zfile.namelist():
                with zfile.open(log, "r") as logfile:
                    stripped_lines = (line.decode().lstrip() for line in logfile)
                    name = pathlib.Path(log).name
                    arrays[name] = parse_error_vectors(stripped_lines)
    else:
        for log in args.log_files:
            with open(log, "r") as logfile:
                stripped_lines = (line.lstrip() for line in logfile)
                name = pathlib.Path(log).name
                arrays[name] = parse_error_vectors(stripped_lines)
    np.savez(args.out_path, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="log aggregator")
    subparsers = parser.add_subparsers()
    error_vector_parser = subparsers.add_parser(
        "error-vectors", help="extract error vectors + tree index from log files"
    )
    error_vector_parser.add_argument(
        "out_path",
        type=str,
        metavar="OUT_PATH",
        help="the path of the output file (containing the error vector arrays)",
    )
    error_vector_parser.add_argument(
        "log_files",
        type=str,
        nargs="+",
        metavar="LOG_FILE",
        help="the log file to extract tree indices and error vectors from",
    )
    error_vector_parser.add_argument(
        "--zipped",
        help="if set, provide only a single .zip file instead of one, or more, log files",
        action="store_true",
    )
    error_vector_parser.set_defaults(dispatch_func=error_vector_dispatch)
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])
    if hasattr(args, "dispatch_func"):
        args.dispatch_func(args)
