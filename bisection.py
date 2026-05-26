from math import ceil, log2
from typing import Callable


def bisection(
    function: Callable[[float], float],
    interval: tuple[float, float],
    error: float,
    max_iter: int = 500,
) -> tuple[float, list[float]]:
    a, b = interval
    history: list[float] = []
    for _ in range(max_iter):
        history.append(b - a)
        if b - a <= error:
            break
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
    return (a + b) / 2, history


def bisection_worst_case_steps(interval: tuple[float, float], epsilon: float) -> int:
    a, b = interval
    return ceil(log2((b - a) / epsilon))
