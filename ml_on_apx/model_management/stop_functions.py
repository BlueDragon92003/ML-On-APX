"""Functions to evaluate whether or not model training should stop."""

import math
from typing import Dict, List, Tuple

from ml_on_apx.logging import log_call
from ml_on_apx.model_management import _MODEL

LispType = float | str | List["LispType"]
LispVars = List[Tuple[str, LispType]]

_STOP_FUNCTIONS = "stopfn" @ _MODEL


class StopFunction:
    """Represents an expression that determines if training should stop.

    WARNING! THIS IMPLEMENTATION DOES NOT VERIFY THE INPUTTED CODE.
    Any call to a `StopFunction` must catch the EvaluationErrors this class defines.

    An odd extention of the Lisp implementation from
    https://zserge.com/posts/langs-lisp/, modified for this use case.

    ## Function API

    This dialect of lisp expects single-expression programs. The result of a program
    will be interpreted as a boolean value. Truthy values (non-nil) indicate that the
    training process will stop. Falsy values (nil, and empty list) indicate that the
    program should continue to execute.

    As noted before, this implementation is not type-checked at compile time. Training
    will stop early if the stop function encounters an error during evaluation.

    All functions have a valency. Any time such function is called, it will process
    exactly that many arguments, or raise an InvalidArgumentNumberError if there are too
    few. Arguments beyond the valency are currently ignored; it is best practice not to
    use them.

    Functions also expect certain types, and reject if the types are incorrect. The
    three types encountered in this program are Numeric, List, and "t" (a truethy atom).
    The indicated type must be provided to a function to avoid an
    InvalidArgumentTypeError.

    The "Boolean" type is, for arguments, any of these, as they are all either truthy or
    falsy. In return types, Boolean indicates the function will provide the true atom
    "t" or nil.

    The "Atomic" type represents all types that are non-empty lists.

    If a string is given indicating a missing function, a MissingFunctionError is
    raised. If a string is given that does not resolve to a valid variable or function,
    depending on its location within a list, an UnevaluateableError is raised.

    As programs are single expressions, no user-defined functions are available.

    A list of functions, their valencies, and their return values can be found below.

    ### atom
        Valency: 1 [Any]
        Return: Boolean

        True if the first argument is Atomic.

    ### car
        Valency: 1 [List<T: Any>]
        Return: T

        The first item from the provided list.

    ### cdr
        Valency: 1 [List<T: Any>]
        Return: List<T>

        The list with the first item removed.

    ### cons
        Valency: 2 [Any, List<Any>]
        Return: List<Any>

        The first item prepended to the given list as an item.

    ### eq
        Valency: 2 [Atomic, Any]
        Return: Boolean

        True if the first argument is atomic and equal to the second.

    ### +
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The sum of the two arguments.

    ### -
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The second argument subtracted from the first.

    ### *
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The product of the two arguments.

    ### /
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The first argument divided by the second.

        Raises an InvalidArgumentTypeError if the second argument is 0.

    ### **
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The first argument raised to the power of the second.

    ### log
        Valency: 2 [Numeric, Numeric]
        Return: Numeric

        The log of the first argument, with a base of the second.

        Raises an InvalidArgumentTypeError if the second argument is nonpositive.

    ### <
        Valency: 2 [Numeric, Numeric]
        Return: Boolean

        True if first argument is strictly less than the second.

    ### <=
        Valency: 2 [Numeric, Numeric]
        Return: Boolean

        True if first argument is less than or equal to the second.

    ### >
        Valency: 2 [Numeric, Numeric]
        Return: Boolean

        True if first argument is strictly greater than the second.

    ### >=
        Valency: 2 [Numeric, Numeric]
        Return: Boolean

        True if first argument is greater than or equal to the second.

    ### or
        Valency: n [Any*]
        Return: Any

        The first truthy argument, or nil if they are all falsy.

    ### and
        Valency: n [Any*]
        Return: Boolean

        The first falsy argument, or True if they are all truthy.

    ### not
        Valency: 1 [Any]
        Return: Boolean

        True if first argument is falsy, nil otherwise

    """

    class EvaluationError(Exception):
        """Errors in stop function evaluation."""

    class InvalidArgumentTypeError(EvaluationError):
        """If an argument is of the incorrect type."""

    class InvalidArgumentNumberError(EvaluationError):
        """If there are too few arguments provided."""

    class MissingFunctionError(EvaluationError):
        """If a function does not exist."""

    class UnevaluateableError(EvaluationError):
        """If a function does not exist."""

    def __init__(self, code: str) -> None:
        """Create a new stop function.

        Propogates:
            IndexError: From `parse`.

        Args:
            code (str): The lisp code underneath the stop function.

        """
        self._code = code
        self._parsed = StopFunction.parse(StopFunction.lex(self._code))

    @log_call(action_type="call" > _STOP_FUNCTIONS)
    def __call__(self, **kwargs: list[float]) -> bool:
        """Call the stop function.

        Args:
            kwargs (float): Variables the stop function has access to.

        Propogates:
            UnevaluatableError: From `eval`.

        Returns:
            bool: Whether or not the training job should stop.

        """
        value = StopFunction.eval(self._parsed, StopFunction.convert_kwargs(kwargs))
        return not StopFunction.is_nil(value)

    @staticmethod
    def execute(code: str, **kwargs: list[float]) -> bool:
        """Execute a one-off stop function.

        Args:
            code (str): The stop function to execute.
            kwargs (float): Variables the stop function has access to.

        Propogates:
            UnevaluatableError: From `eval`.
            IndexError: From `parse`.

        Returns:
            bool: The result of the stop function.

        """
        return StopFunction.is_nil(
            StopFunction.eval(
                StopFunction.parse(StopFunction.lex(code)),
                StopFunction.convert_kwargs(kwargs),
            )
        )

    @staticmethod
    @log_call(action_type="kwargs" > _STOP_FUNCTIONS)
    def convert_kwargs(kwargs: Dict[str, list[float]]) -> LispVars:
        """Convert a keyword argument dict to the interpreter's variable format.

        Args:
            kwargs (Dict[str, float]): A mapping from variable names to numbers.

        Returns:
            LispVars: A structure that is used by the interpreter for variables.

        """
        globals = []
        for entry in kwargs.items():
            globals.append(entry)
        return globals

    # Very simple lexer, split by parens and whitespace
    @staticmethod
    @log_call(action_type="lex" > _STOP_FUNCTIONS)
    def lex(code: str) -> List[str]:
        """Process the code string into individual tokens.

        Args:
            code (str): The code to process

        Returns:
            List[str]: The language tokens the code is composed of.

        """
        return code.replace("(", " ( ").replace(")", " ) ").split()

    # A simple parser: build nested lists from nested parenthesis
    @staticmethod
    @log_call(action_type="parse" > _STOP_FUNCTIONS)
    def parse(tokens: List[str]) -> LispType:
        """Parse tokens into an executable lisp AST.

        Args:
            tokens (List[str]): The tokens to parse.

        Raises:
            IndexError if the list of tokens to parse is empty.

        Returns:
            LispType: The AST to execute.

        """
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
    # Find value in associated list L by key x:
    #
    #   L = [["foo", 12], ["bar", 42], ["baz", 123]]
    #   assoc("bar", L) -> 42
    #
    @staticmethod
    def assoc(x: str, local_vars: LispVars) -> LispType:
        """Get the value of a variable.

        Args:
            x (str): The variable to get the value of.
            local_vars (LispVars): The local vars to pull from.

        Returns:
            LispType: The value of the variable.

        """
        return (
            []
            if not local_vars
            else local_vars[0][1]
            if local_vars[0][0] == x
            else StopFunction.assoc(x, local_vars[1:])
        )

    #
    # Atom is not a list, or an empty list (nil)
    #   atom([]) -> t
    #   atom(42) -> t
    #   atom([42, 'a']) -> []
    #
    @staticmethod
    def atom(x: LispType) -> LispType:
        """Return a lisp boolean if the value is atomic.

        Args:
            x (LispType): The value to determine.

        Returns:
            LispType: The lisp boolean (`t` or `nil`) if x was atomic.

        """
        return "t" if (type(x) is not list) or len(x) == 0 else []

    @staticmethod
    def is_nil(x: LispType) -> bool:
        """Return a python boolean if the value is nil (an empty list).

        Args:
            x (LispType): The value to test.

        Returns:
            bool: True if `x` is nil.

        """
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
    def apply(f: str, args: list[LispType], _local_vars: LispVars) -> LispType:
        """Apply a function `f` to the arguments.

        Args:
            f (str): The function to apply.
            args (LispType): The arguments of the function.
            _local_vars (LispVars): The local variables to read from. (Unused)

        Raises:
            EvaluationError: If the arguments are incorrect or if the function does not
                exist.
            ValueError: If, during log evaluation, the base is non-positive.
            ZeroDivisionError: If, during division, the denominator is 0.

        Returns:
            LispType: The result of the function.

        """
        if f == "atom":
            return StopFunction.atom(args[0])
        elif f == "car":
            if len(args) <= 0:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not list or len(args[0]) <= 0:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0][0]
        elif f == "cdr":
            if len(args) <= 0:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not list or len(args[0]) <= 1:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0][1:]
        elif f == "cons":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[1]) is not list:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return [args[0]] + args[1]
        elif f == "eq":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            return "t" if StopFunction.atom(args[0]) and args[0] == args[1] else []
        elif f == "+":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0] + args[1]
        elif f == "-":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0] - args[1]
        elif f == "*":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0] * args[1]
        elif f == "/":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0] / args[1]
        elif f == "**":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return args[0] ** args[1]
        elif f == "log":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float or args[1] <= 0:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return math.log(args[0], args[1])
        elif f == "<":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return "t" if args[0] < args[1] else []
        elif f == "<=":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return "t" if args[0] <= args[1] else []
        elif f == ">":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return "t" if args[0] > args[1] else []
        elif f == ">=":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            if type(args[0]) is not float or type(args[1]) is not float:
                raise StopFunction.InvalidArgumentTypeError(f, args, _local_vars)
            return "t" if args[0] >= args[1] else []
        elif f == "or":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            return args[1] if StopFunction.is_nil(args[0]) else args[0]
        elif f == "and":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            return args[0] if StopFunction.is_nil(args[0]) else args[1]
        elif f == "not":
            if len(args) <= 1:
                raise StopFunction.InvalidArgumentNumberError(f, len(args))
            return "t" if StopFunction.is_nil(args[0]) else []
        else:  # StopFunction.is_nil(f):
            raise StopFunction.MissingFunctionError(f)
        # elif (type(f) is list) and f[0] == "lambda":
        #     return StopFunction.eval(f[2], StopFunction.pairlis(f[1], args, L))
        # else:
        #     return StopFunction.apply(StopFunction.eval(f, L), args, L)

    # Evaluate "cond"
    @staticmethod
    def evcon(x: List[List[LispType]], local_vars: LispVars) -> LispType:
        """Evaluate a lisp condition function.

        Args:
            x (List[List[LispType]]): A list of lisp conditions branches.
            local_vars (LispVars): The local variables available.

        Propogates:
            UnevaluatableError: From `eval`.

        Returns:
            LispType: The result of the evaluated second element of the first condition
                branch where the second element evaluates to non-nil.

        """
        if len(x) == 0:
            return []
        if StopFunction.eval(x[0][0], local_vars):
            return StopFunction.eval(x[0][1], local_vars)
        return StopFunction.evcon(x[1:], local_vars)

    # Evaluate list of lambda arguments
    @staticmethod
    def evlis(x: List[LispType], local_vars: LispVars) -> List[LispType]:
        """Evaluate a list of arguments for a function.

        Args:
            x (List[LispType]): The arguments provided to the function
            local_vars (LispVars): The local variables available.

        Propogates:
            UnevaluateableTokenError: From `eval`.

        Returns:
            List[LispType]: The results from evaluating the arguments.

        """
        return (
            [
                StopFunction.eval(x[0], local_vars),
                *StopFunction.evlis(x[1:], local_vars),
            ]
            if x
            else []
        )

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
    def eval(x: LispType, local_vars: LispVars) -> LispType:
        """Evaluate a token.

        Args:
            x (LispType): The token to evaluate.
            local_vars (LispVars): The local variable available.

        Raises:
            UnevaluateableError: If the AST node cannot be evaluated.

        Returns:
            LispType: The value of the evaluted token.

        """
        if x == "nil":
            return []
        elif x == "t" or type(x) is float:
            return x
        elif type(x) is str:
            return StopFunction.assoc(x, local_vars)
        elif type(x) is list and x[0] == "quote":
            return x[1]
        elif type(x) is list and x[0] == "cond":
            return StopFunction.evcon(x[1:], local_vars)
        # elif type(x) is list and x[0] == "label":
        #     L.insert(0, (x[1], x[2]))
        #     return x[1]
        elif type(x) is list:
            return StopFunction.apply(
                x[0], StopFunction.evlis(x[1:], local_vars), local_vars
            )
        else:
            raise StopFunction.UnevaluateableError(x)
