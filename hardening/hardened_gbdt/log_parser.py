from lark import Lark, Transformer


def diagnosis_parser():
    # `clutter` matches everything up to (and including) the first
    # occasion of ###.
    grammar = r"""
        ?line: clutter "diagnosis" "value" NUMBER "###" "-" equation
        ?clutter: /^(.*?)###/
        ?equation: IDENTIFIER "=" value
        ?value: list
            | NUMBER

        ?list : "[" [value ("," value)*] "]"

        %import common.CNAME -> IDENTIFIER
        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS
        %ignore WS
    """
    diag_parser = Lark(grammar, start="line")
    return diag_parser


class DiagnosisToDict(Transformer):
    def line(self, args):
        return args[-1]  # ignore everything but the `equation` part

    def equation(self, eq):
        ident, val = eq
        return {ident[:]: val}

    def NUMBER(self, n):
        return float(n)

    list = list


def tree_idx_parser():
    # `clutter` matches everything up to (and including) the first
    # occasion of `Building`.
    grammar = r"""
        ?line: clutter "dp-tree-" NUMBER "using" NUMBER "samples..."
        ?clutter: /^(.*?)Building/

        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS
        %ignore WS
    """
    diag_parser = Lark(grammar, start="line")
    return diag_parser


class TreeIdxToDict(Transformer):
    def line(self, args):
        [_, tree_idx, num_samples] = args  # ignore first item (clutter)
        return dict(tree_idx=tree_idx, num_samples=num_samples)

    def NUMBER(self, n):
        return float(n)

    list = list


def tree_acceptence_parser():
    grammar = r"""
        ?line: clutter1 WORD "decision" "tree" NUMBER clutter1 ":" NUMBER clutter2 ":" NUMBER
        ?clutter1: /(.*?)ensemble/
        ?clutter2: /(.*?)left/

        %import common.SIGNED_NUMBER -> NUMBER
        %import common.CNAME -> WORD
        %import common.WS
        %ignore WS
    """
    diag_parser = Lark(grammar, start="line")
    return diag_parser


class TreeAcceptenceToDict(Transformer):
    def line(self, args):
        [_, is_included, tree_idx, _, ensemble_size, _, num_instances_left] = args
        return dict(
            tree_idx=tree_idx,
            is_included=is_included,
            ensemble_size=ensemble_size,
            num_instances_left=num_instances_left,
        )

    def WORD(self, w):
        if w == "includes":
            return True
        elif w == "excludes":
            return False

    def NUMBER(self, n):
        return int(float(n))
