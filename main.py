from typing import Callable


def f(x):
    return x**3 - 27


def fof(x):
    return x**2


def rootfinder(
    function: Callable[[float], float], xvals: tuple[float, float], error: float
):
    a, b = xvals
    c = (a + b) / 2
    count = 0
    while b - a > error:
        c = (a + b) / 2
        if function(c) * function(a) < 0:
            b = c
        elif function(c) * function(b) < 0:
            a = c
        else:
            raise Exception(
                "erm the interval is not such that there is a singular root"
            )
        print(count, c, function(c))
        count += 1
    return c


def main():
    epsilon = 0.0000000000001
    print(rootfinder(f, (0, 5), epsilon))
    print(rootfinder(fof, (-1, 1), epsilon))  # nuh uh


if __name__ == "__main__":
    main()
