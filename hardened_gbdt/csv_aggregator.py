import glob
import re
from collections import ChainMap
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
from lark import Lark, Transformer


def job_id(filename: str) -> str:
    """Extract the job_id from a filename.

    The job_id combines the execution number and the job location. For
    example, "ignis2_64" is a valid job_id telling us, that the job with
    number 64 was executed on the host ignis2.

    Args:
        filename (str): a filename obeying the format
        "evaluation/{hostname}/…/…_{i}.csv" (without {}), where
        {hostname} is a simple string and {i} is an integer representing
        the number of the job.

    Returns:
        str: the string "{hostname}_{i}" (without {}), where hostname
        and i are substituted correspondingly.
    """
    p = Path(filename)
    assert p.parts[0] == "evaluation"
    hostname = p.parts[1]
    assert "ignis" in hostname
    filename = p.stem
    execution_number = filename.split("_")[1]
    job_id = f"{hostname}_{execution_number}"
    return job_id


def stack(csv_pattern: str, sep: str) -> pd.DataFrame:
    """Read multiple csv files and stack the corresponding dataframes
    vertically.

    Args:
        csv_pattern (str): a pattern describing the considered multiple
        single-line csv files.
        sep (str): the separating character used by the csv files.

    Returns:
        pd.DataFrame: the stacked dataframe containing all read data
        frames, plus the new column `job_id`.
    """
    candidates = glob.glob(csv_pattern)
    dfs = [
        pd.read_csv(file, sep=sep).assign(job_id=job_id(file)) for file in candidates
    ]
    df = pd.concat(dfs, ignore_index=True)
    return df


def neg_filter(df, not_like: str, axis) -> pd.DataFrame:
    """Only keep labels from axis, which satisfy `not_like in label == False`."""
    pattern = r"^(?!.*" + re.escape(not_like) + r").*$"
    return df.filter(regex=pattern, axis=axis)


def unpivot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Discard train scores and switch from wide format to tall format.

    The k test scores (from k-fold cross validation) are stored in the
    columns split_0_test_score to split_(k-1)_test_score. Transform to
    tall format by introducing columns `score` and `split` and using
    the split index as a variable and the score as a value.

    Args:
        df (pd.DataFrame): the dataframe containing the split test
        scores in wide format.

    Returns:
        pd.DataFrame: The dataframe containing the split test
        scores in tall format.
    """
    df = neg_filter(df, "train_score", axis=1)  # discard the training splits
    test_split_columns = [col for col in df.columns if "test_score" in col]
    other_columns = [col for col in df.columns if "test_score" not in col]
    df = df.melt(
        id_vars=other_columns,
        value_vars=test_split_columns,
        var_name="split",
        value_name="score",
    )
    return df


def unified_settings(csv_filename_pattern: str) -> pd.DataFrame:
    """Merge multiple single-line csv files to one large dataframe.

    Args:
        csv_filename_pattern (str): a pattern describing the considered
        multiple single-line csv files.

    Returns:
        pd.DataFrame: The result of merging all csv files.
    """
    df = stack(csv_filename_pattern, sep=";")
    df = unpivot_scores(df)
    return df


def constructor_parser():
    """Parse strings representing (simple) constructors.

    In this case, a constructor as an identifier followed by
    parentheses, which may enclose keyword arguments."""

    grammar = r"""
        ?constructor: identifier "(" kwargs ")"
        ?kwargs: [equation ("," equation)*]
        ?equation: identifier "=" value
        ?value: NUMBER
            | string
            | list
            | constructor
        identifier : CNAME
        string: ESCAPED_STRING
        list: "[" value ("," value)* "]"

        %import common.ESCAPED_STRING
        %import common.CNAME
        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS
        %ignore WS
    """
    diag_parser = Lark(grammar, start="constructor")
    return diag_parser


class ConstructorToDict(Transformer):
    """Convert the parsed AST to a deep/nested dictionary repr."""

    def constructor(self, args):
        (identifier, kwargs) = args
        return dict(identifier=identifier, kwargs=kwargs)

    def kwargs(self, eqs):
        return dict(ChainMap(*eqs))

    def equation(self, eq):
        ident, val = eq
        return {ident[:]: val}

    def string(self, s):
        (s,) = s
        return s[1:-1]  # to remove the double quotes around the string

    def identifier(self, i):
        (i,) = i
        return i[:]

    def NUMBER(self, n):
        return float(n)

    list = list


def _flatten_cons(prefix, cons_dict):
    assert "identifier" in cons_dict and "kwargs" in cons_dict
    return {
        f"{prefix}{cons_dict['identifier']}_{k}": v
        for (k, v) in _flatten_kwargs(cons_dict["kwargs"]).items()
    }


def _flatten_kwargs(kwargs):
    d = {}
    for (k, v) in kwargs.items():
        if any([isinstance(v, Number), isinstance(v, str), isinstance(v, list)]):
            d[k] = v
        elif isinstance(v, dict):
            d = {**_flatten_cons(f"{k}_", v), **d}
        else:
            raise ValueError("Unexpected kwarg value")
    return d


def flatten_dict(deep_dict) -> dict[str, Any]:
    """Flatten a nested constructor representation.

    Args:
        deep_dict (_type_): the nested constructor representation.

    Returns:
        dict[str, Any]: the flattened constructor representation, where
        every "simple" keyword argument (the key) is fully qualified by
        the belonging constructor. The value is the keyword argument's
        value.

    Note:
        This implementation may have problems if some keyword argument
        has a list as value, which itself may contain more complex types
        than just numbers or strings (another constructor for example).
    """
    return _flatten_cons("", deep_dict)


def extract_tree_rejector_params(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Extract (recursively) all keyword arguments from a column of
    constructor strings.

    Then add to the dataframe for each parameter a column containing
    the parameter values.

    Args:
        df (pd.DataFrame): a dataframe df with constructor strings at
        df[label].
        label (str): the target column's label
    Returns:
        pd.DataFrame: a new dataframe with the additional columns.
    """
    transformer = ConstructorToDict()
    parser = constructor_parser()
    flat_conses = []
    for cons in df[label]:
        parsed = parser.parse(cons)
        transformed = transformer.transform(parsed)
        flattened = flatten_dict(transformed)
        flat_conses.append(flattened)
    _df = pd.DataFrame(flat_conses)
    return pd.concat([df, _df], axis=1)


def mean_std_scores(df: pd.DataFrame, excluded_keys: list[str]) -> pd.DataFrame:
    """Aggregate score (count, mean, std) over the excluded keys.

    Args:
        df (pd.DataFrame): the dataframe to calculate mean and std
        scores for.
        excluded_keys (list[str]): The columns to aggregate over (all
        other columns are used as keys when grouping. It is necessary
        that this includes "score".

    Returns:
        pd.DataFrame: a dataframe containing only the remaining columns
        and the new columns "count", "mean_score" and "std_score" (i.e.
        not the excluded keys).
    """
    remaining_columns = [col for col in df.columns if col not in excluded_keys]
    grouped = df.groupby(remaining_columns, dropna=False)
    agged = grouped.agg(
        count=("score", "count"),
        mean_score=("score", "mean"),
        std_score=("score", "std"),
    ).reset_index()
    _sorted = agged.sort_values(by="mean_score")
    return _sorted


def high_score_settings(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Find for each unique combination of keys the setting with best
    "mean_score".

    Args:
        df (pd.DataFrame): a dataframe containing at least the columns
        `keys` and "mean_score".
        keys (list[str]): a set of key labels for which to search for
        unique value combinations.

    Returns:
        pd.DataFrame: a dataframe with the same set of columns as
        before, but the rows compressed to one per unique combination
        of keys - the one having the lowest "mean_score".
    """
    just_scores = df.groupby(keys).agg(mean_score=("mean_score", "min")).reset_index()
    scores_with_features = pd.merge(
        just_scores, df, on=keys + ["mean_score"], how="inner"
    )
    scores_with_features.sort_values(by="mean_score", inplace=True)
    return scores_with_features
