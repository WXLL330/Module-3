"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiply two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The product of the two numbers.

    """
    return a * b


def id(x: float) -> float:
    """Returns the input value unchanged.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The same input value.

    """
    return x


def add(a: float, b: float) -> float:
    """Adds two floating-point numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The sum of the two numbers.

    """
    return a + b


def neg(x: float) -> float:
    """Returns the negation of the input value.

    Args:
    ----
        x (float): The input value to be negated.

    Returns:
    -------
        float: The negated value of the input.

    """
    return -x


def lt(a: float, b: float) -> float:
    """Compare two float values and return True if the first value is less than the second value.

    Args:
    ----
        a (float): The first value to compare.
        b (float): The second value to compare.

    Returns:
    -------
        float: 1.0 if `a` is less than `b`, otherwise 0.0.

    """
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Compares two floating-point numbers for equality.

    Args:
    ----
        a (float): The first number to compare.
        b (float): The second number to compare.

    Returns:
    -------
        float: 1.0 if the numbers are equal, 0.0 otherwise.

    """
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        a (float): The first number.
        b (float): The second number.

    Returns:
    -------
        float: The maximum of the two numbers.

    """
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Determines if two floating-point numbers are close to each other within a small tolerance.

    Args:
    ----
        a (float): The first floating-point number.
        b (float): The second floating-point number.

    Returns:
    -------
        float: True if the absolute difference between `a` and `b` is less than 0.01, otherwise False.

    """
    return abs(a - b) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid of x.

    The sigmoid function is defined as 1 / (1 + exp(-x)) for x >= 0,
    and exp(x) / (1 + exp(x)) for x < 0. This implementation is numerically
    stable and avoids overflow issues.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of the input value.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:  # noqa: D417
    """Applies the ReLU (Rectified Linear Unit) function to the input.

    The ReLU function is defined as:
        ReLU(x) = max(0, x)

    Parameters
    ----------
        x (float): The input value.

    Returns
    -------
        float: The output of the ReLU function, which is the input value if it is positive, or 0 otherwise.

    """
    return x if x > 0 else 0.0


EPS = 1e-10


def log(x: float) -> float:
    """Computes the natural logarithm of a given number.

    Args:
    ----
        x (float): The number to compute the natural logarithm for.

    Returns:
    -------
        float: The natural logarithm of the input number.

    """
    return math.log(x+EPS)


def exp(x: float) -> float:
    """Computes the exponential of a given number.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of the input value.

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the gradient of the logarithm function with respect to its input.

    Args:
    ----
        x (float): The input value to the logarithm function.
        y (float): The second arg for backward.

    Returns:
    -------
        float: The gradient of the logarithm function with respect to x.

    """
    if abs(x) < EPS:
        x = EPS
    return y * (1.0 / x)


def inv(x: float) -> float:  # noqa: D417
    """Computes the multiplicative inverse of a given number.

    Parameters
    ----------
        x (float): The number to invert.

    Returns
    -------
        float: The multiplicative inverse of the input number.

    Raises
    ------
        ZeroDivisionError: If the input number is zero.

    """
    # try:
    #     return 1.0 / x
    # except ZeroDivisionError:
    #     raise ZeroDivisionError("division by zero")
    if abs(x) < EPS:
        x = EPS
    return 1.0 / x

def inv_back(x: float, y: float) -> float:
    """Computes the gradient of the inverse function with respect to its input.

    Given the function f(x) = 1/x, the derivative f'(x) = -1/(x^2). This function
    calculates the gradient of the inverse function with respect to its input, scaled
    by the provided gradient `y`.

    Args:
    ----
        x (float): The input value for which the gradient is computed.
        y (float): The second arg for backward.

    Returns:
    -------
        float: The gradient of the inverse function with respect to `x`, scaled by `y`.

    """
    if abs(x) < EPS:
        x = EPS
    return y * (-1.0 / (x**2))


def relu_back(x: float, y: float) -> float:  # noqa: D417
    """Computes the gradient of the ReLU function with respect to its input.

    The ReLU function is defined as:
        ReLU(x) = max(0, x)

    The gradient of ReLU is:
        - y if x > 0
        - 0 otherwise

    Parameters
    ----------
    x (float): The input value to the ReLU function.
    y (float): The gradient of the loss with respect to the output of the ReLU function.

    Returns
    -------
    float: The gradient of the loss with respect to the input of the ReLU function.

    """
    return y if x > 0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Applies a given function to each element in an iterable of floats.

    Args:
    ----
        fn (Callable[[float], float]): A function that takes a float and returns a float.
        ls (Iterable[float]): An iterable of floats to which the function will be applied.

    Returns:
    -------
        Iterable[float]: A new iterable with the function applied to each element.

    """
    new_ls = []
    for num in ls:
        new_ls.append(fn(num))
    return new_ls


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Applies a binary function to pairs of elements from two input iterables and returns a new iterable with the results.

    Args:
    ----
        fn (Callable[[float, float], float]): A function that takes two float arguments and returns a float.
        ls1 (Iterable[float]): The first iterable of floats.
        ls2 (Iterable[float]): The second iterable of floats.

    Returns:
    -------
        Iterable[float]: A new iterable containing the results of applying the function to corresponding elements of the input iterables.

    """
    new_ls = []
    for x, y in zip(ls1, ls2):
        new_ls.append(fn(x, y))
    return new_ls


def reduce(
    fn: Callable[[float, float], float], ls: Iterable[float], start: float
) -> float:
    """Apply a binary function cumulatively to the items of a sequence, from left to right,
    so as to reduce the sequence to a single value.

    Args:
    ----
        fn (Callable[[float, float], float]): A binary function that takes two floats and returns a float.
        ls (Iterable[float]): An iterable of floats to be reduced.
        start (float): The initial value to start the reduction.

    Returns:
    -------
        float: The reduced value after applying the binary function cumulatively.

    """
    result = start
    for num in ls:
        result = fn(result, num)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Applies the neg function to each element in the given iterable.

    Args:
    ----
        ls (Iterable[float]): An iterable of float numbers.

    Returns:
    -------
        Iterable[float]: An iterable where each element is the negation of the corresponding element in the input iterable.

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements of two lists.

    Args:
    ----
        ls1 (Iterable[float]): The first list of floats.
        ls2 (Iterable[float]): The second list of floats.

    Returns:
    -------
        Iterable[float]: A new iterable containing the sums of the corresponding elements of ls1 and ls2.

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Calculate the sum of a list of floating-point numbers.

    Args:
    ----
        ls (Iterable[float]): An iterable containing floating-point numbers to be summed.

    Returns:
    -------
        float: The sum of the numbers in the iterable.

    """
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in the given iterable.

    Args:
    ----
        ls (Iterable[float]): An iterable of floating-point numbers.

    Returns:
    -------
        float: The product of all elements in the iterable.

    """
    return reduce(mul, ls, 1.0)
