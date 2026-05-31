def gaussian_elimination(A, b):
    n = len(b)
    M = [list(row) for row in A]
    x = list(b)

    for i in range(n):
        pivot = M[i][i]
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            for k in range(i, n):
                M[j][k] -= factor * M[i][k]
            x[j] -= factor * x[i]

    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x


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

    x = gaussian_elimination(A, b)
    print(x)


if __name__ == "__main__":
    main()
