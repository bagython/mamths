import math

import matplotlib.pyplot as plt
import numpy as np

from bisection import bisection
from fixed_point import fixed_point
from newton import newton

EPS = 1e-12


def sign_change_brackets(f, a, b, n):
    xs = np.linspace(a, b, n + 1)
    vals = [f(x) for x in xs]
    out = []
    for i in range(n):
        if vals[i] == 0:
            out.append((xs[i], xs[i]))
        elif vals[i] * vals[i + 1] < 0:
            out.append((xs[i], xs[i + 1]))
    return out


# Each experiment is a different function. alpha is the single fixed-point step
# (g = alpha*f + x); where it diverges out of a bracket the algorithm is marked
# "diverged" instead of reporting a root it did not actually find there.
EXPERIMENTS = [
    {
        "name": "quadratic",
        "title": "f(x) = x^2 - 4",
        "f": lambda x: x**2 - 4,
        "df": lambda x: 2 * x,
        "interval": (-3.0, 3.5),
        "alpha": -0.1,
        "subdivisions": 23,
    },
    {
        "name": "trig",
        "title": "f(x) = cos(x) - x",
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1,
        "interval": (0.0, 1.5),
        "alpha": 1.0,
        "subdivisions": 21,
    },
    {
        "name": "cubic",
        "title": "f(x) = (x - 1)^3",
        "f": lambda x: (x - 1) ** 3,
        "df": lambda x: 3 * (x - 1) ** 2,
        "interval": (-2.0, 2.7),
        "alpha": -0.2,
        "subdivisions": 23,
    },
]


def _fmt(x):
    return "None" if x is None else f"{x:.10f}"


def run(exp):
    f, df = exp["f"], exp["df"]
    a, b = exp["interval"]
    alpha = exp["alpha"]

    brackets = sign_change_brackets(f, a, b, exp["subdivisions"])
    print(f"\n{exp['title']}: {len(brackets)} bracket(s) in [{a}, {b}]  (alpha={alpha})")

    fig, ax = plt.subplots(figsize=(10, 6))
    seen = set()

    def add(history, label, color, style):
        if not history:
            return
        lbl = label if label not in seen else None
        if lbl:
            seen.add(lbl)
        ax.semilogy(
            range(1, len(history) + 1),
            history,
            label=lbl,
            color=color,
            linestyle=style,
            linewidth=1.8,
        )

    for lo, hi in brackets:
        r_bis, h_bis = bisection(f, (lo, hi), EPS)
        r_new, h_new = newton(f, df, (lo, hi), EPS)
        r_fp, h_fp = fixed_point(f, (lo, hi), alpha, EPS)
        # only trust fixed-point if it actually reached eps and stayed in bracket
        fp_converged = bool(h_fp) and h_fp[-1] < EPS
        fp_ok = (
            r_fp is not None and fp_converged and lo - 1e-6 <= r_fp <= hi + 1e-6
        )

        print(f"  bracket [{lo:.4f}, {hi:.4f}]:")
        print(f"    bisection  : root={_fmt(r_bis)}  iters={len(h_bis)}")
        print(f"    newton     : root={_fmt(r_new)}  iters={len(h_new)}")
        print(
            f"    fixed-point: "
            + (f"root={_fmt(r_fp)}  iters={len(h_fp)}" if fp_ok else "diverged")
        )

        add(h_bis, "bisection", "tab:blue", "-")
        add(h_new, "newton", "tab:orange", "--")
        if fp_ok:
            add(h_fp, "fixed-point", "tab:green", ":")

    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    ax.set_title(f"convergence: {exp['title']}  (alpha={alpha})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"convergence_{exp['name']}.png", dpi=120)
    plt.close(fig)


def main():
    for exp in EXPERIMENTS:
        run(exp)


if __name__ == "__main__":
    main()
