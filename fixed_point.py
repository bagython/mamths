from typing import Callable


# Fixed-point iteration for f(x) = 0 via g(x) = alpha * f(x) + x.
# Converges near a root r when |1 + alpha * f'(r)| < 1.
def fixed_point(
    function: Callable[[float], float],
    interval: tuple[float, float],
    alpha: float,
    error: float,
    initial_approx: float | None = None,
    max_iter: int = 500,
) -> tuple[float | None, list[float]]:
    f = function
    a, b = interval
    if initial_approx is None:
        x = (a + b) / 2
    else:
        x = initial_approx

    def _g(x: float) -> float:
        return alpha * f(x) + x  # hmm

    history: list[float] = []
    for _ in range(max_iter):
        diff = abs(x - (x := _g(x)))  # :P
        history.append(diff)
        if diff < error:
            return x, history
        if abs(x) > 1e8:
            return None, history
    return x, history
