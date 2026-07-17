"""Tests for the StopFunction class."""

import unittest

from ml_on_apx.model_management.stop_functions import (
    WAHR,
    LispType,
    StopFunction,
)


def simple_eval(expr: str) -> LispType:
    """Evaluate and get the lisp return value of the expression.

    Args:
        expr (str): The expression.

    Returns:
        LispType: The return value of the expression.

    """
    return StopFunction.eval(StopFunction.parse(StopFunction.lex(expr)), [])


class TestsStopFunction(unittest.TestCase):
    """Tests for the StopFunction class."""

    def test_stop_function__atom__valency_check(self) -> None:
        """Test if the `atom` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(atom)")

    def test_stop_function__atom(self) -> None:
        """Test the `atom` function works."""
        self.assertEqual(WAHR, simple_eval("(atom nil)"))
        self.assertEqual(WAHR, simple_eval("(atom 0.5)"))
        self.assertEqual(WAHR, simple_eval("(atom (and 1 2))"))
        self.assertEqual([], simple_eval("(atom (quote (1 2 3)))"))

    def test_stop_function__car__valency_check(self) -> None:
        """Test if the `car` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(car)")

    def test_stop_function__car__type_check(self) -> None:
        """Test if the `car` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(car nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(car 0.5)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(car (and 1 2))")

    def test_stop_function__car(self) -> None:
        """Test the `car` function works."""
        self.assertEqual(1.0, simple_eval("(car (quote (1 2 3)))"))

    def test_stop_function__cdr__valency_check(self) -> None:
        """Test if the `cdr` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(cdr)")

    def test_stop_function__cdr__type_check(self) -> None:
        """Test if the `cdr` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(cdr nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(cdr 0.5)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(cdr (and 1 2))")

    def test_stop_function__cdr(self) -> None:
        """Test the `cdr` function works."""
        self.assertEqual([2, 3], simple_eval("(cdr (quote (1 2 3)))"))
        self.assertEqual([], simple_eval("(cdr (quote (1)))"))

    def test_stop_function__cons__valency_check(self) -> None:
        """Test if the `cons` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(cons)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(cons 1)")

    def test_stop_function__cons__type_check(self) -> None:
        """Test if the `cons` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(cons 2 1)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(cons 2 (and 2 3))")

    def test_stop_function__cons(self) -> None:
        """Test the `cons` function works."""
        self.assertEqual([2], simple_eval("(cons 2 nil)"))
        self.assertEqual([2, 3], simple_eval("(cons 2 (cons 3 nil))"))

    def test_stop_function__eq__valency_check(self) -> None:
        """Test if the `eq` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(eq)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(eq 1)")

    def test_stop_function__eq(self) -> None:
        """Test the `eq` function works."""
        self.assertEqual(WAHR, simple_eval("(eq 2 2)"))
        self.assertEqual(WAHR, simple_eval("(eq nil nil)"))
        self.assertEqual([], simple_eval("(eq nil 2)"))
        self.assertEqual([], simple_eval("(eq (cons 1 nil) (cons 1 nil))"))
        self.assertEqual(WAHR, simple_eval("(eq (and 1 2) (and 2 3))"))

    def test_stop_function__add__valency_check(self) -> None:
        """Test if the `+` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(+)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(+ 1)")

    def test_stop_function__add__type_check(self) -> None:
        """Test if the `+` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ 3 (and 2 3))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ (and 2 3)) 3")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ 2 (cons 1 nil))")
        with self.assertRaises(ValueError):
            simple_eval("(+ (cons 1 nil) 2)")

    def test_stop_function__add(self) -> None:
        """Test the `+` function works."""
        self.assertEqual(3, simple_eval("(+ 1 2)"))
        self.assertEqual(1, simple_eval("(+ -1 2)"))

    def test_stop_function__sub__valency_check(self) -> None:
        """Test if the `-` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(-)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(- 1)")

    def test_stop_function__sub__type_check(self) -> None:
        """Test if the `sub` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- 3 (and 2 3))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- (and 2 3)) 3")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- (cons 1 nil) 2)")

    def test_stop_function__sub(self) -> None:
        """Test the `-` function works."""
        self.assertEqual(-1, simple_eval("(- 1 2)"))
        self.assertEqual(-3, simple_eval("(- -1 2)"))

    def test_stop_function__mult__valency_check(self) -> None:
        """Test if the `mult` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(*)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(* 1)")

    def test_stop_function__mult__type_check(self) -> None:
        """Test if the `mult` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 3 (and 2 3))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* (and 2 3)) 3")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* (cons 1 nil) 2)")

    def test_stop_function__mult(self) -> None:
        """Test the `mult` function works."""
        self.assertEqual(3, simple_eval("(* 1 2)"))
        self.assertEqual(1, simple_eval("(* -1 2)"))

    def test_stop_function__div__valency_check(self) -> None:
        """Test if the `div` function's valency check works."""
        # TODO

    def test_stop_function__div__type_check(self) -> None:
        """Test if the `div` function's type checks work."""
        # TODO

    def test_stop_function__div__value_check(self) -> None:
        """Test the `div` function's value checks work."""
        # TODO

    def test_stop_function__div(self) -> None:
        """Test the `div` function works."""
        # TODO

    def test_stop_function__exp__valency_check(self) -> None:
        """Test if the `exp` function's valency check works."""
        # TODO

    def test_stop_function__exp__type_check(self) -> None:
        """Test if the `exp` function's type checks work."""
        # TODO

    def test_stop_function__exp(self) -> None:
        """Test the `exp` function works."""
        # TODO

    def test_stop_function__log__valency_check(self) -> None:
        """Test if the `log` function's valency check works."""
        # TODO

    def test_stop_function__log__type_check(self) -> None:
        """Test if the `log` function's type checks work."""
        # TODO

    def test_stop_function__log__value_check(self) -> None:
        """Test the `log` function's value checks work."""
        # TODO

    def test_stop_function__log(self) -> None:
        """Test the `log` function works."""
        # TODO

    def test_stop_function__lt__valency_check(self) -> None:
        """Test if the `lt` function's valency check works."""
        # TODO

    def test_stop_function__lt__type_check(self) -> None:
        """Test if the `lt` function's type checks work."""
        # TODO

    def test_stop_function__lt(self) -> None:
        """Test the `lt` function works."""
        # TODO

    def test_stop_function__le__valency_check(self) -> None:
        """Test if the `le` function's valency check works."""
        # TODO

    def test_stop_function__le__type_check(self) -> None:
        """Test if the `le` function's type checks work."""
        # TODO

    def test_stop_function__le(self) -> None:
        """Test the `le` function works."""
        # TODO

    def test_stop_function__gt__valency_check(self) -> None:
        """Test if the `gt` function's valency check works."""
        # TODO

    def test_stop_function__gt__type_check(self) -> None:
        """Test if the `gt` function's type checks work."""
        # TODO

    def test_stop_function__gt(self) -> None:
        """Test the `gt` function works."""
        # TODO

    def test_stop_function__ge__valency_check(self) -> None:
        """Test if the `ge` function's valency check works."""
        # TODO

    def test_stop_function__ge__type_check(self) -> None:
        """Test if the `ge` function's type checks work."""
        # TODO

    def test_stop_function__ge(self) -> None:
        """Test the `ge` function works."""
        # TODO

    def test_stop_function__or__valency_check(self) -> None:
        """Test if the `or` function's valency check works."""
        # TODO

    def test_stop_function__or__type_check(self) -> None:
        """Test if the `or` function's type checks work."""
        # TODO

    def test_stop_function__or__value_check(self) -> None:
        """Test the `or` function's value checks work."""
        # TODO

    def test_stop_function__or(self) -> None:
        """Test the `or` function works."""
        # TODO

    def test_stop_function__and__valency_check(self) -> None:
        """Test if the `and` function's valency check works."""
        # TODO

    def test_stop_function__and__type_check(self) -> None:
        """Test if the `and` function's type checks work."""
        # TODO

    def test_stop_function__and__value_check(self) -> None:
        """Test the `and` function's value checks work."""
        # TODO

    def test_stop_function__and(self) -> None:
        """Test the `and` function works."""
        # TODO

    def test_stop_function__not__valency_check(self) -> None:
        """Test if the `not` function's valency check works."""
        # TODO

    def test_stop_function__not__type_check(self) -> None:
        """Test if the `not` function's type checks work."""
        # TODO

    def test_stop_function__not__value_check(self) -> None:
        """Test the `not` function's value checks work."""
        # TODO

    def test_stop_function__not(self) -> None:
        """Test the `not` function works."""
        # TODO
