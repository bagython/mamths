import time

import numpy as np


def jacobi_naive(A, b, eps, max_iter=100000):
    n = len(b)
    x = [0.0 for _ in range(n)]
    for k in range(1, max_iter + 1):
        x_new = [0.0 for _ in range(n)]
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


# Matrix form: x^(k+1) = D^-1 (b - (L + U) x^(k))
def jacobi_matrix(A, b, eps, max_iter=100000):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    D = np.diag(A)
    LU = A - np.diag(D)
    x = np.zeros_like(b)
    for k in range(1, max_iter + 1):
        x_new = (b - LU @ x) / D
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k
        x = x_new
    return x, max_iter


def make_tridiagonal(n, diag=2.04, off=-1.0, seed=0):
    A = (
        np.diag(np.full(n, diag))
        + np.diag(np.full(n - 1, off), 1)
        + np.diag(np.full(n - 1, off), -1)
    )
    rng = np.random.default_rng(seed)
    b = rng.uniform(-1.0, 1.0, n)
    return A, b


def main():
    # 6x + 2y - 1z = 4
    # 1x + 5y + 1z = 3
    # 2x + 1y + 4z = 27
    A = (
        (6, 2, -1),
        (1, 5, 1),
        (2, 1, 4),
    )
    b = (4, 3, 27)
    eps = 1e-6

    x_n, it_n = jacobi_naive(A, b, eps)
    x_m, it_m = jacobi_matrix(A, b, eps)
    print("3x3 sanity check:")
    print(f"  naive  ({it_n} iter): {x_n}")
    print(f"  matrix ({it_m} iter): {x_m.tolist()}")

    n = 100
    A_big, b_big = make_tridiagonal(n)
    A_list = A_big.tolist()
    b_list = b_big.tolist()

    t0 = time.perf_counter()
    x_n, it_n = jacobi_naive(A_list, b_list, eps)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_m, it_m = jacobi_matrix(A_big, b_big, eps)
    t_matrix = time.perf_counter() - t0

    max_err = np.max(np.abs(np.array(x_n) - x_m))
    print(f"\n{n}x{n} benchmark, eps={eps}:")
    print(f"  naive  : {it_n} iterations in {t_naive:.4f}s")
    print(f"  matrix : {it_m} iterations in {t_matrix:.4f}s")
    print(f"  speedup: {t_naive / t_matrix:.1f}x")
    print(f"  max |x_naive - x_matrix|: {max_err:.2e}")


if __name__ == "__main__":
    main()
