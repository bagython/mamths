import csv

import matplotlib.pyplot as plt
import numpy as np

from gauss import gaussian_elimination
from jacobi import jacobi_naive
from seidel import gauss_seidel

EPS = 1e-10
MAX_ITER = 10000
DIMENSIONS = range(3, 51)


def solve_gauss(A, b):
    x = gaussian_elimination(A.tolist(), b.tolist())
    return np.asarray(x, dtype=float), 1, True


def solve_jacobi(A, b):
    x, iters = jacobi_naive(A.tolist(), b.tolist(), EPS, MAX_ITER)
    return np.asarray(x, dtype=float), iters, iters < MAX_ITER


def solve_seidel(A, b):
    n = len(b)
    rows = [{j: float(A[i][j]) for j in range(n)} for i in range(n)]
    x, iters, _ = gauss_seidel(rows, b.tolist(), max_iters=MAX_ITER, tol=EPS)
    return np.asarray(x, dtype=float), iters, iters < MAX_ITER


def random_matrix(n, low=-5.0, high=5.0, dominant=False, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(low, high, (n, n))
    if dominant:
        for i in range(n):
            off = np.sum(np.abs(A[i])) - abs(A[i, i])
            A[i, i] = off + rng.uniform(1.0, 2.0)
    return A


def hilbert_matrix(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


def make_rhs(A, x_test):
    return A @ x_test


def residual(A, x, b):
    return float(np.linalg.norm(A @ x - b))


def abs_error(x, x_test):
    return float(np.linalg.norm(x - x_test))


EXPERIMENTS = [
    {
        "key": "random",
        "label": "RandMatr",
        "title": "Random matrix, no diagonal dominance, entries in [-5, 5]",
        "make": lambda n: random_matrix(n, dominant=False, seed=n),
        "algorithms": [("Gauss", solve_gauss)],
    },
    {
        "key": "hilbert",
        "label": "Hilbert",
        "title": "Hilbert matrix",
        "make": lambda n: hilbert_matrix(n),
        "algorithms": [("Gauss", solve_gauss), ("Seidel", solve_seidel)],
    },
    {
        "key": "diagdominant",
        "label": "DiagDominant",
        "title": "Random matrix with diagonal dominance",
        "make": lambda n: random_matrix(n, dominant=True, seed=n),
        "algorithms": [
            ("Gauss", solve_gauss),
            ("Jacobi", solve_jacobi),
            ("Seidel", solve_seidel),
        ],
    },
]


def run_experiment(exp):
    series = {
        name: {"n": [], "residual": [], "abs_error": []}
        for name, _ in exp["algorithms"]
    }
    rows = []
    for n in DIMENSIONS:
        A = exp["make"](n)
        x_test = np.ones(n)
        b = make_rhs(A, x_test)
        for name, solver in exp["algorithms"]:
            try:
                x, iters, _ = solver(A, b)
                res = residual(A, x, b)
                err = abs_error(x, x_test)
            except ZeroDivisionError:
                iters, res, err = 1, float("nan"), float("nan")
            rows.append(
                {
                    "Problem type": exp["label"],
                    "Algorithm": name,
                    "Dimension": n,
                    "Iterations": iters,
                    "Residue": res,
                    "Abs. error": err,
                }
            )
            series[name]["n"].append(n)
            series[name]["residual"].append(res)
            series[name]["abs_error"].append(err)
    return rows, series


def write_csv(all_rows, path="results.csv"):
    fields = [
        "Problem type",
        "Algorithm",
        "Dimension",
        "Iterations",
        "Residue",
        "Abs. error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def plot_experiment(exp, series):
    fig, (ax_res, ax_err) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for name in series:
        s = series[name]
        ax_res.semilogy(s["n"], s["residual"], marker="o", markersize=3, label=name)
        ax_err.semilogy(s["n"], s["abs_error"], marker="o", markersize=3, label=name)

    ax_res.set_ylabel("residual  |Ax - b|")
    ax_res.set_title(exp["title"])
    ax_res.grid(True, which="both", alpha=0.3)
    ax_res.legend()

    ax_err.set_ylabel("abs. error  |x - x_test|")
    ax_err.set_xlabel("dimension n")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend()

    fig.tight_layout()
    fig.savefig(f"experiment_{exp['key']}.png", dpi=120)
    plt.close(fig)


def main():
    all_rows = []
    for exp in EXPERIMENTS:
        print(f"Running: {exp['title']}")
        rows, series = run_experiment(exp)
        all_rows.extend(rows)
        plot_experiment(exp, series)
        print(f"  -> experiment_{exp['key']}.png")

    write_csv(all_rows)
    print(f"Wrote results.csv ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
