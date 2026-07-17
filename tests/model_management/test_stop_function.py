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
        """Test if the `atom` function works."""
        self.assertEqual(WAHR, simple_eval("(atom nil)"))
        self.assertEqual(WAHR, simple_eval("(atom 0.5)"))
        self.assertEqual(WAHR, simple_eval("(atom (not nil))"))
        self.assertEqual([], simple_eval("(atom (cons 1 (cons 2 (cons 3 nil))))"))

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
            simple_eval("(car (not nil))")

    def test_stop_function__car(self) -> None:
        """Test if the `car` function works."""
        self.assertEqual(1.0, simple_eval("(car (cons 1 (cons 2 (cons 3 nil))))"))

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
            simple_eval("(cdr (not nil))")

    def test_stop_function__cdr(self) -> None:
        """Test if the `cdr` function works."""
        self.assertEqual([2, 3], simple_eval("(cdr (cons 1 (cons 2 (cons 3 nil))))"))
        self.assertEqual([], simple_eval("(cdr (cons 1 nil))"))

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
            simple_eval("(cons 2 (not nil))")

    def test_stop_function__cons(self) -> None:
        """Test if the `cons` function works."""
        self.assertEqual([2], simple_eval("(cons 2 nil)"))
        self.assertEqual([2, 3], simple_eval("(cons 2 (cons 3 nil))"))

    def test_stop_function__eq__valency_check(self) -> None:
        """Test if the `eq` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(eq)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(eq 1)")

    def test_stop_function__eq(self) -> None:
        """Test if the `eq` function works."""
        self.assertEqual(WAHR, simple_eval("(eq 2 2)"))
        self.assertEqual(WAHR, simple_eval("(eq nil nil)"))
        self.assertEqual([], simple_eval("(eq nil 2)"))
        self.assertEqual([], simple_eval("(eq (cons 1 nil) (cons 1 nil))"))
        self.assertEqual(WAHR, simple_eval("(eq (not nil) (not nil))"))

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
            simple_eval("(+ 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(+ (cons 1 nil) 2)")

    def test_stop_function__add(self) -> None:
        """Test if the `+` function works."""
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
            simple_eval("(- 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(- (cons 1 nil) 2)")

    def test_stop_function__sub(self) -> None:
        """Test if the `-` function works."""
        self.assertEqual(-1, simple_eval("(- 1 2)"))
        self.assertEqual(-3, simple_eval("(- -1 2)"))

    def test_stop_function__mult__valency_check(self) -> None:
        """Test if the `*` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(*)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(* 1)")

    def test_stop_function__mult__type_check(self) -> None:
        """Test if the `*` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(* (cons 1 nil) 2)")

    def test_stop_function__mult(self) -> None:
        """Test if the `*` function works."""
        self.assertEqual(4, simple_eval("(* 2 2)"))
        self.assertEqual(-2, simple_eval("(* -1 2)"))

    def test_stop_function__div__valency_check(self) -> None:
        """Test if the `div` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(/)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(/ 1)")

    def test_stop_function__div__type_check(self) -> None:
        """Test if the `/` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(/ (cons 1 nil) 2)")

    def test_stop_function__div__value_check(self) -> None:
        """Test if the `/` function's value checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentValueError):
            simple_eval("(/ 32 0)")

    def test_stop_function__div(self) -> None:
        """Test if the `/` function works."""
        self.assertEqual(0.5, simple_eval("(/ 1 2)"))
        self.assertEqual(-2, simple_eval("(/ 2 -1)"))

    def test_stop_function__exp__valency_check(self) -> None:
        """Test if the `**` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(**)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(** 1)")

    def test_stop_function__exp__type_check(self) -> None:
        """Test if the `**` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(** (cons 1 nil) 2)")

    def test_stop_function__exp(self) -> None:
        """Test if the `**` function works."""
        self.assertEqual(1, simple_eval("(** 1 2)"))
        self.assertEqual(0.25, simple_eval("(** 4 -1)"))

    def test_stop_function__log__valency_check(self) -> None:
        """Test if the `log` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(log)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(log 1)")

    def test_stop_function__log__type_check(self) -> None:
        """Test if the `log` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(log (cons 1 nil) 2)")

    def test_stop_function__log__value_check(self) -> None:
        """Test if the `log` function's value checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentValueError):
            simple_eval("(log 32 0)")
        with self.assertRaises(StopFunction.InvalidArgumentValueError):
            simple_eval("(log 32 -0.25)")
        with self.assertRaises(StopFunction.InvalidArgumentValueError):
            simple_eval("(log 0 2)")
        with self.assertRaises(StopFunction.InvalidArgumentValueError):
            simple_eval("(log -5 2)")

    def test_stop_function__log(self) -> None:
        """Test if the `log` function works."""
        self.assertEqual(0, simple_eval("(log 1 2)"))
        self.assertEqual(0.5, simple_eval("(log 2 4)"))
        self.assertEqual(-0.5, simple_eval("(log 0.5 4)"))

    def test_stop_function__lt__valency_check(self) -> None:
        """Test if the `<` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(<)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(< 1)")

    def test_stop_function__lt__type_check(self) -> None:
        """Test if the `<` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(< (cons 1 nil) 2)")

    def test_stop_function__lt(self) -> None:
        """Test if the `<` function works."""
        self.assertEqual(WAHR, simple_eval("(< 1 2)"))
        self.assertEqual([], simple_eval("(< 2 1)"))
        self.assertEqual([], simple_eval("(< 1 1)"))

    def test_stop_function__le__valency_check(self) -> None:
        """Test if the `<=` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(<=)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(<= 1)")

    def test_stop_function__le__type_check(self) -> None:
        """Test if the `<=` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(<= (cons 1 nil) 2)")

    def test_stop_function__le(self) -> None:
        """Test if the `<=` function works."""
        self.assertEqual(WAHR, simple_eval("(<= 1 2)"))
        self.assertEqual([], simple_eval("(<= 2 1)"))
        self.assertEqual(WAHR, simple_eval("(<= 1 1)"))

    def test_stop_function__gt__valency_check(self) -> None:
        """Test if the `>` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(>)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(> 1)")

    def test_stop_function__gt__type_check(self) -> None:
        """Test if the `>` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(> (cons 1 nil) 2)")

    def test_stop_function__gt(self) -> None:
        """Test if the `>` function works."""
        self.assertEqual([], simple_eval("(> 1 2)"))
        self.assertEqual(WAHR, simple_eval("(> 2 1)"))
        self.assertEqual([], simple_eval("(> 1 1)"))

    def test_stop_function__ge__valency_check(self) -> None:
        """Test if the `>=` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(>=)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(>= 1)")

    def test_stop_function__ge__type_check(self) -> None:
        """Test if the `>=` function's type checks work."""
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= 2 nil)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= nil 2)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= 3 (not nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= (not nil) 3)")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= 2 (cons 1 nil))")
        with self.assertRaises(StopFunction.InvalidArgumentTypeError):
            simple_eval("(>= (cons 1 nil) 2)")

    def test_stop_function__ge(self) -> None:
        """Test if the `>=` function works."""
        self.assertEqual([], simple_eval("(>= 1 2)"))
        self.assertEqual(WAHR, simple_eval("(>= 2 1)"))
        self.assertEqual(WAHR, simple_eval("(>= 1 1)"))

    def test_stop_function__or__valency_check(self) -> None:
        """Test if the `or` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(or)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(or 1)")

    def test_stop_function__or(self) -> None:
        """Test if the `or` function works."""
        self.assertEqual(1, simple_eval("(or 1 2)"))
        self.assertEqual(2, simple_eval("(or nil 2)"))
        self.assertEqual([], simple_eval("(or nil nil)"))
        self.assertEqual([1], simple_eval("(or nil (cons 1 nil))"))
        self.assertEqual(WAHR, simple_eval("(or nil (not nil))"))

    def test_stop_function__and__valency_check(self) -> None:
        """Test if the `and` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(and)")
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(and 1)")

    def test_stop_function__and(self) -> None:
        """Test if the `and` function works."""
        self.assertEqual(2, simple_eval("(and 1 2)"))
        self.assertEqual([], simple_eval("(and nil 2)"))
        self.assertEqual([1], simple_eval("(and 1 (cons 1 nil))"))
        self.assertEqual(WAHR, simple_eval("(and 1 (not nil))"))

    def test_stop_function__not__valency_check(self) -> None:
        """Test if the `not` function's valency check works."""
        with self.assertRaises(StopFunction.InvalidArgumentNumberError):
            simple_eval("(not)")

    def test_stop_function__not(self) -> None:
        """Test if the `not` function works."""
        self.assertEqual([], simple_eval("(not 1)"))
        self.assertEqual(WAHR, simple_eval("(not nil)"))
        self.assertEqual([], simple_eval("(not (cons 1 nil))"))
        self.assertEqual([], simple_eval("(not (not nil))"))
