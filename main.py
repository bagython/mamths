from inspect import getsource
from math import ceil, log2
from typing import Callable


def forg(x):
    return x**3 - 27


def fof(x):
    return x**2


def rootfinder(
    function: Callable[[float], float], interval: tuple[float, float], error: float
) -> float:
    a, b = interval

    codigits = len(
        str(bisection_worst_case_steps(interval, error))
    )  # pad iteration number

    count = 0
    while b - a > error:
        c = (a + b) / 2

        if function(c) == 0:
            break

        if function(c) * function(a) < 0:
            b = c
        elif function(c) * function(b) < 0:
            a = c
        else:
            raise Exception(
                "erm the interval is not such that there is a singular root"
            )
        print(f"{count:<{codigits}} {c:.{16}f}", function(c))
        count += 1
    return (a + b) / 2


def bisection_worst_case_steps(interval: tuple[float, float], epsilon: float) -> int:
    a, b = interval
    return ceil(log2((b - a) / epsilon))


def main():
    epsilon = 0.0000000000001
    print(rootfinder(forg, (0, 5), epsilon))
    # print(rootfinder(fof, (-1, 1), epsilon))  # nuh uh

    print("homework: ")
    eps = 0.01
    interval = (1, 5)

    def f(x):
        return (x**2) - 4

    print(getsource(f))  # :D
    print(
        f"expected worst case iterations: {bisection_worst_case_steps(interval, eps)}"
    )
    print("iteration | root approximation | f(c) value")
    print("root:", rootfinder(f, interval, eps))


if __name__ == "__main__":
    main()
