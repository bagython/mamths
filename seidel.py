import random


def generate_sparse_matrix(
    n: int, off_diagonals_per_row=5, seed=0
) -> list[dict[int, float]]:
    rng = random.Random(seed)
    matrix = []
    for i in range(n):
        row = {}
        off_sum = 0.0
        for _ in range(off_diagonals_per_row):
            j = rng.randrange(n)
            if j == i or j in row:
                continue
            val = rng.uniform(-1.0, 1.0)
            row[j] = val
            off_sum += abs(val)
        row[i] = off_sum + rng.uniform(1.0, 2.0)  # diagonally dominant -> converges
        matrix.append(row)
    return matrix


def matvec(matrix: list[dict[int, float]], x: list[float]):
    result = [0.0] * len(matrix)
    for i, row in enumerate(matrix):
        s = 0.0
        for j, val in row.items():
            s += val * x[j]
        result[i] = s
    return result


def gauss_seidel(
    matrix: list[dict[int, float]], b: list[float], max_iters=1000, tol=1e-8
):
    n = len(matrix)
    x = [0.0] * n

    max_diff = 0.0

    for it in range(1, max_iters + 1):
        max_diff = 0.0
        for i in range(n):
            row = matrix[i]
            diag = row[i]
            s = b[i]
            for j, val in row.items():
                if j != i:
                    s -= val * x[j]
            x_new = s / diag  # x_i = (b_i - sum_{j!=i} a_ij x_j) / a_ii
            diff = abs(x_new - x[i])
            if diff > max_diff:
                max_diff = diff
            x[i] = x_new  # in-place, later rows use fresh value

        if max_diff < tol:
            return x, it, max_diff

    return x, max_iters, max_diff


def main():
    # play with different sizes and sparsity; for each the known solution is
    # 1..n and b is built from it, so we can report the recovered max error.
    print(f"{'n':>8} {'off/row':>8} {'nnz':>10} {'iters':>6} {'max_err':>10}")
    for n in (1_000, 10_000, 100_000):
        for off in (3, 8):
            matrix = generate_sparse_matrix(n, off_diagonals_per_row=off, seed=42)
            nnz = sum(len(row) for row in matrix)
            x_true = [float(i + 1) for i in range(n)]  # known solution 1..n
            b = matvec(matrix, x_true)
            x, iters, _ = gauss_seidel(matrix, b, max_iters=1000, tol=1e-10)
            max_err = max(abs(x[i] - x_true[i]) for i in range(n))
            print(f"{n:>8} {off:>8} {nnz:>10} {iters:>6} {max_err:>10.2e}")


if __name__ == "__main__":
    main()
