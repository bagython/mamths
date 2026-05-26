import matplotlib.pyplot as plt
import numpy as np

from bisection import bisection
from fixed_point import fixed_point
from newton import newton


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


def main():
    # f(x) = (x-1)(x+2)(x-3) = x^3 - 2x^2 - 5x + 6  -- roots at -2, 1, 3
    def F(x):
        return x**3 - 2 * x**2 - 5 * x + 6

    def DF(x):
        return 3 * x**2 - 4 * x - 5

    A = -4.0
    B = 5.0
    EPS = 1e-12
    ALPHA = -0.05  # step size for fixed-point iteration (g = alpha*f + x)
    N_SUBDIVISIONS = 37  # for locating sign-change brackets

    brackets = sign_change_brackets(F, A, B, N_SUBDIVISIONS)
    print(f"Found {len(brackets)} sign-change bracket(s) in [{A}, {B}]")

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_labels = set()

    def add_plot(history, label, color, style):
        if not history:
            return
        lbl = label if label not in plotted_labels else None
        if lbl:
            plotted_labels.add(lbl)
        ax.semilogy(
            range(1, len(history) + 1),
            history,
            label=lbl,
            color=color,
            linestyle=style,
            linewidth=1.8,
        )

    for a, b in brackets:
        x0 = (a + b) / 2

        r_bis, h_bis = bisection(F, (a, b), EPS)
        r_new, h_new = newton(F, DF, (a, b), EPS)
        r_fp, h_fp = fixed_point(F, (a, b), ALPHA, EPS)

        print(f"\nbracket [{a:.4f}, {b:.4f}]  (x0={x0:.4f}):")
        print(f"  bisection  : root={_fmt(r_bis)}  iters={len(h_bis)}")
        print(f"  newton     : root={_fmt(r_new)}  iters={len(h_new)}")
        print(
            f"  fixed-point: root={_fmt(r_fp)}   iters={len(h_fp)}"
            + ("  (diverged)" if r_fp is None else "")
        )

        add_plot(h_bis, "bisection", "tab:blue", "-")
        add_plot(h_new, "newton", "tab:orange", "--")
        if r_fp is not None:
            add_plot(h_fp, "fixed-point", "tab:green", ":")

    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    ax.set_title(f"convergence: bisection vs newton vs fixed-point (α={ALPHA})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=120)
    plt.show()


def _fmt(x):
    return "None" if x is None else f"{x:.10f}"


if __name__ == "__main__":
    main()
