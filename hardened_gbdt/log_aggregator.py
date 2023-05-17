# ypo@informatik.uni-kiel.de
"""
"""
import argparse
import itertools
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


def stripped_lines_from_file(filename: str) -> Lines:
    "Read lines from file and remove leading and trailing whitespace."
    with open(filename, "r", encoding="utf-8") as file:
        return list(stripped_lines(file))


def stripped_lines(lines) -> Lines:
    for line in lines:
        if hasattr(line, "decode"):
            yield line.decode().strip()
        else:
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


def _parse_diag_lines3(filter_str: str, lines: Lines) -> pd.DataFrame:
    """Only match lines containing `filter_str` and regarding those
    lines, apply the equations parser only to what comes after it
    (basically discarding everything up to and including `filter_str`).
    """
    lines = filter_lines(filter_str, lines)
    lines = (line.split(filter_str, 1)[1] for line in lines)
    diag_statistics = fast_parse(lines, log_parser.fast_diagnosis_parser())
    df = pd.DataFrame(diag_statistics)
    return df


def parse_rmses(lines: Lines) -> pd.DataFrame:
    """parse the rmse values from a given log"""
    return _parse_diag_lines3(
        "### diagnosis value 01 ### - rmse of absolute differences ", lines
    )


def parse_rmse_approximations(lines: Lines) -> pd.DataFrame:
    """parse the approximated (by a tree scoring mechanism chosen at run
    time) rmse from a given log"""
    return _parse_diag_lines3("### diagnosis value 02 ### - current ", lines)


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
    for i, tree_idx_and_error_vector in enumerate(tree_indices_and_error_vectors):
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


def parse_tree_acceptance(lines: Lines) -> pd.DataFrame:
    lines = filter_lines("### diagnosis value 07 ###", lines)
    acceptance_and_stuff = parse_lines(
        lines, log_parser.tree_acceptence_parser(), log_parser.TreeAcceptenceToDict()
    )
    df = pd.DataFrame(acceptance_and_stuff)
    return df


def parse_acceptance_matching(lines: Lines) -> pd.DataFrame:
    lines = list(lines)
    df = pd.concat(
        [
            parse_rmses(lines),
            parse_rmse_approximations(lines),
        ],
        axis=1,
    )
    df = df.rename(dict(rmse="rmse_incl_tree", rmse_approx="dp_rmse_incl_tree"), axis=1)

    df["dp_rmse_excl_tree"] = pd.Series(
        index=df["dp_rmse_incl_tree"].index,
        data=list(
            itertools.accumulate(
                df["dp_rmse_incl_tree"],
                lambda x, y: x if x < y else y,
                initial=float("inf"),
            )
        )[:-1],
    )
    df["rmse_excl_tree"] = pd.Series(
        index=df["rmse_incl_tree"].index,
        data=list(
            itertools.accumulate(
                df["rmse_incl_tree"],
                lambda x, y: x if x < y else y,
                initial=float("inf"),
            )
        )[:-1],
    )

    def _latest_accepted_tree(excl: pd.Series, incl: pd.Series):
        min_idx = None
        for i, (e1, e2) in enumerate(zip(excl, incl)):
            if e1 > e2:
                min_idx = i
            yield min_idx

    df["latest_accepted_dp_tree"] = pd.Series(
        index=df["dp_rmse_excl_tree"].index,
        data=_latest_accepted_tree(df["dp_rmse_excl_tree"], df["dp_rmse_incl_tree"]),
    )

    df["dp_tree_accepted"] = df["dp_rmse_incl_tree"] < df["dp_rmse_excl_tree"]
    df["contrafactual_rmse_excl_tree"] = (
        df["rmse_incl_tree"][df["latest_accepted_dp_tree"]]
        .reset_index(drop=True)
        .shift(periods=1, fill_value=float("inf"))
    )

    df["leaky_tree_accepted"] = (
        df["rmse_incl_tree"] < df["contrafactual_rmse_excl_tree"]
    )
    df["matching?"] = df["dp_tree_accepted"] == df["leaky_tree_accepted"]
    df = df[
        [
            "dp_rmse_excl_tree",
            "dp_rmse_incl_tree",
            "dp_tree_accepted",
            "latest_accepted_dp_tree",
            "rmse_excl_tree",
            "contrafactual_rmse_excl_tree",
            "rmse_incl_tree",
            "leaky_tree_accepted",
            "matching?",
        ]
    ]
    return df


def extract_dataframes_from_zip(
    zip_filename: str,
    extractor: Callable[[Lines], pd.DataFrame],
) -> Iterable[pd.DataFrame]:
    """Apply a dataframe extracting function to all logs in a .zip file.

    Args:
        zip_filename (str): the .zip file containing the log files
        extractor (Callable[[Lines], pd.DataFrame]): the dataframe
        extracting function

    Returns:
        Iterable[pd.DataFrame]: the dataframes extracted from the logs
        within the .zip file
    """
    with zipfile.ZipFile(zip_filename, "r") as zfile:
        for log in zfile.namelist():
            with zfile.open(log, "r") as logfile:
                strppd_lines = stripped_lines(logfile)
                yield extractor(strppd_lines)


def dataframes_to_zip(
    dfs: Iterable[pd.DataFrame], archive_names: Iterable[str], zip_filename: str
) -> None:
    """Write several dataframes as .csv to a zip archive."""
    with zipfile.ZipFile(zip_filename, "w") as zfile:
        for df, fname in zip(dfs, archive_names):
            zfile.writestr(fname, df.to_csv())


def df_extractor_dispatch(args: argparse.Namespace) -> None:
    df_parsers = dict(
        acceptance_matching=parse_acceptance_matching,
    )
    extractor = df_parsers[args.df_extractor]
    if args.zipped:
        dfs = extract_dataframes_from_zip(args.log_file, extractor)
        with zipfile.ZipFile(args.log_file, "r") as zfile:
            archive_names = zfile.namelist()
        archive_names = [
            str(pathlib.Path(arch_name).with_suffix(".csv"))
            for arch_name in archive_names
        ]
        dataframes_to_zip(dfs, archive_names, args.out_path)
    else:
        lines = stripped_lines_from_file(args.log_file)
        df = extractor(lines)
        df.to_csv(args.out_path)


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

    df_parser = subparsers.add_parser(
        "dataframe", help="extract a dataframe from a log"
    )
    df_parser.add_argument(
        "df_extractor",
        type=str,
        choices=["acceptance_matching"],
        metavar="DF_EXTRACTOR",
        help="which Dataframe extracting function to apply to the log file",
    )
    df_parser.add_argument(
        "out_path",
        type=str,
        metavar="OUT_PATH",
        help="the path of the output .csv file (or .zip of .csv files)",
    )
    df_parser.add_argument(
        "log_file",
        type=str,
        metavar="LOG_FILE",
        help="the log file (or .zip of log files) to extract dataframes from",
    )
    df_parser.add_argument(
        "--zipped",
        help="if set, LOG_FILE is interpreted as .zip file",
        action="store_true",
    )
    df_parser.set_defaults(dispatch_func=df_extractor_dispatch)
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
