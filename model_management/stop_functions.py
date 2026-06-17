from typing import Dict, List, Tuple
import math

LispType = float | str | List["LispType"]
LispVars = List[Tuple[str, LispType]]


class StopFunction:
    """Represents an expression that determines if training should stop.

    WARNING! THIS IMPLEMENTATION DOES NOT CHECK THE INPUTTED CODE.
    Any call to a `StopFunction` must catch the following exceptions:
    - TypeError
    - ValueError
    - IndexError
    Many of these will occur from user errors, but some are due to the fact that
    my type checker cannot properly consider the union types used :/

    An odd extention of the Lisp implementation from
    https://zserge.com/posts/langs-lisp/
    modified for this use case.
    """

    def __init__(self, code: str):
        self._code = code
        self._parsed = StopFunction.parse(StopFunction.lex(self._code))

    def __call__(self, **kwargs: float) -> bool:
        value = StopFunction.eval(self._parsed, StopFunction.convert_kwargs(kwargs))
        return not StopFunction.is_nil(value)

    @staticmethod
    def execute(code: str, **kwargs: float) -> bool:
        return StopFunction.eval(
            StopFunction.parse(StopFunction.lex(code)),
            StopFunction.convert_kwargs(kwargs),
        )

    @staticmethod
    def convert_kwargs(kwargs: Dict[str, float]) -> LispVars:
        globals = []
        for entry in kwargs.items():
            globals.append(entry)
        return globals

    # Very simple lexer, split by parens and whitespace
    @staticmethod
    def lex(code: str) -> List[str]:
        return code.replace("(", " ( ").replace(")", " ) ").split()

    # A simple parser: build nested lists from nested parenthesis
    @staticmethod
    def parse(tokens: List[str]) -> LispType:
        t = tokens.pop(0)
        if t == "(":
            sexp = []
            while tokens[0] != ")":
                sexp.append(StopFunction.parse(tokens))
            tokens.pop(0)
            return sexp
        try:
            return float(t)
        except ValueError:
            return t

    #
    # Add (x, y) pair to list L:
    #
    #   pairlis([3], [4], []) -> [[3, 4]]
    #   pairlis([1, 2, 3], [4, 5, 6], [[7, 8]]) -> [[1, 4], [2, 5], [3, 6], [7, 8]]
    #
    @staticmethod
    def pairlis(x: List[str], y: List[LispType], L: LispVars) -> LispVars:
        return L if not x else [(x[0], y[0])] + StopFunction.pairlis(x[1:], y[1:], L)

    #
    # Find value in associated list L by key x:
    #
    #   L = [["foo", 12], ["bar", 42], ["baz", 123]]
    #   assoc("bar", L) -> 42
    #
    @staticmethod
    def assoc(x: str, L: LispVars) -> LispType:
        return (
            [] if not L else L[0][1] if L[0][0] == x else StopFunction.assoc(x, L[1:])
        )

    #
    # Atom is not a list, or an empty list (nil)
    #   atom([]) -> t
    #   atom(42) -> t
    #   atom([42, 'a']) -> []
    #
    @staticmethod
    def atom(x: LispType) -> LispType:
        return "t" if (type(x) is not list) or len(x) == 0 else []

    @staticmethod
    def is_nil(x: LispType) -> bool:
        return (type(x) is list) and len(x) == 0

    #
    #   apply[fn;x;a] =
    #        [atom[fn] → [eq[fn;CAR] → caar[x];
    #                    eq[fn;CDR] → cdar[x];
    #                    eq[fn;CONS] → cons[car[x];cadr[x]];
    #                    eq[fn;ATOM] → atom[car[x]];
    #                    eq[fn;EQ] → eq[car[x];cadr[x]];
    #                    T → apply[eval[fn;a];x;a]];
    #        eq[car[fn];LAMBDA] → eval[caddr[fn]; pairlis[cadr[fn];x;a]]];
    #
    @staticmethod
    def apply(f, args, L: LispVars):
        if f == "atom":
            return StopFunction.atom(args[0])
        elif f == "car":
            return args[0][0]
        elif f == "cdr":
            return args[0][1:]
        elif f == "cons":
            return [args[0]] + args[1]
        elif f == "eq":
            return "t" if StopFunction.atom(args[0]) and args[0] == args[1] else []
        elif f == "+":
            return args[0] + args[1]
        elif f == "-":
            return args[0] - args[1]
        elif f == "*":
            return args[0] * args[1]
        elif f == "/":
            return args[0] / args[1]
        elif f == "**":
            return args[0] ** args[1]
        elif f == "log":
            return math.log(args[0], args[1])
        elif f == "<":
            return "t" if args[0] < args[1] else []
        elif f == "<=":
            return "t" if args[0] <= args[1] else []
        elif f == ">":
            return "t" if args[0] > args[1] else []
        elif f == ">=":
            return "t" if args[0] >= args[1] else []
        elif f == "or":
            return args[1] if StopFunction.is_nil(args[0]) else args[0]
        elif f == "and":
            return args[0] if StopFunction.is_nil(args[0]) else args[1]
        elif f == "not":
            return "t" if StopFunction.is_nil(args[0]) else []
        else:  # StopFunction.is_nil(f):
            raise ValueError(f"No such lisp function `{f}`.")
        # elif (type(f) is list) and f[0] == "lambda":
        #     return StopFunction.eval(f[2], StopFunction.pairlis(f[1], args, L))
        # else:
        #     return StopFunction.apply(StopFunction.eval(f, L), args, L)

    # Evaluate "cond"
    @staticmethod
    def evcon(x: List[List[LispType]], L: LispVars) -> LispType:
        if len(x) == 0:
            return []
        if StopFunction.eval(x[0][0], L):
            return StopFunction.eval(x[0][1], L)
        return StopFunction.evcon(x[1:], L)

    # Evaluate list of lambda arguments
    @staticmethod
    def evlis(x: List[LispType], L: LispVars) -> List[LispType]:
        return [StopFunction.eval(x[0], L)] + StopFunction.evlis(x[1:], L) if x else []

    #
    # McCarthy's eval from the paper:
    #
    #   eval[e;a] =
    #      [atom[e] → cdr[assoc[e;a]];
    #       atom[car[e]] →
    #             [eq[car[e],QUOTE] → cadr[e];
    #              eq[car[e];COND] → evcon[cdr[e];a];
    #              T → apply[car[e];evlis[cdr[e];a];a]];
    #       T → apply[car[e];evlis[cdr[e];a];a]]
    #
    # ...with a few additions: nil, t, symbols and label
    #
    @staticmethod
    def eval(x, L: LispVars):
        if x == "nil":
            return []
        elif x == "t" or type(x) is float:
            return x
        elif type(x) is str:
            return StopFunction.assoc(x, L)
        elif type(x) is list and x[0] == "quote":
            return x[1]
        elif type(x) is list and x[0] == "cond":
            return StopFunction.evcon(x[1:], L)
        # elif type(x) is list and x[0] == "label":
        #     L.insert(0, (x[1], x[2]))
        #     return x[1]
        else:
            return StopFunction.apply(x[0], StopFunction.evlis(x[1:], L), L)
