from typing import Callable


def bisection(
    function: Callable[[float], float], interval: tuple[float, float], error: float
) -> tuple[float, int]:
    a, b = interval
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
        count += 1
    return (a + b) / 2, count


def fixedpoint(
    function: Callable[[float], float],
    interval: tuple[float, float],
    alpha: float,  # kinda wack to pass it in
    error: float,
    initial_approx: float | None = None,
    max_iter: int = 100000,
) -> tuple[float, int]:
    f = function
    a, b = interval
    if initial_approx is None:
        x = (a + b) / 2
    else:
        x = initial_approx

    def _g(x: float) -> float:
        return alpha * f(x) + x  # hmm

    count = 0
    while True:
        x_new = _g(x)
        count += 1
        if abs(x_new - x) < error:
            return x_new, count
        if abs(x_new) > 1e8 or count >= max_iter:  # diverged / gave up
            return x_new, count
        x = x_new


def main():
    def f(x):
        return (x**3) - 27  # Alina's function, single root at x = 3

    root = 3.0
    eps = 0.000000001

    # several experiments: vary alpha, interval and starting point, comparing
    # iterations against bisection on the same interval. near the root the
    # fixed-point rate is |1 + alpha * f'(3)| = |1 + 27 * alpha|, so:
    #   alpha = -1/27   -> rate ~ 0    -> much faster than bisection
    #   alpha = -0.0185 -> rate ~ 0.5  -> about the same as bisection
    #   alpha = -0.005  -> rate ~ 0.87 -> much slower than bisection
    experiments = [
        ("fixed point ~ bisection", (1, 4), 2.5, -0.0185),
        ("fixed point much slower", (0, 5), 4.0, -0.005),
        ("fixed point much faster", (1, 6), 4.5, -1 / 27),
    ]

    for label, interval, x0, alpha in experiments:
        rb, ib = bisection(f, interval, eps)
        rf, if_ = fixedpoint(f, interval, alpha, eps, initial_approx=x0)
        print(f"{label}: interval={interval} x0={x0} alpha={alpha:.5f}")
        print(f"  bisection  : root={rb:.10f} err={abs(rb - root):.2e} iters={ib}")
        print(f"  fixed point: root={rf:.10f} err={abs(rf - root):.2e} iters={if_}")


if __name__ == "__main__":
    main()
