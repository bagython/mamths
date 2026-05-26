def jacobi_naive(A, b, eps):
    n = len(b)
    x = [0.0 for _ in range(n)]
    while True:
        x_new = [0.0 for _ in range(n)]
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x_new[i] = (b[i] - sigma) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
            return x_new
        x = x_new


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

    print(jacobi_naive(A, b, eps))


if __name__ == "__main__":
    main()
