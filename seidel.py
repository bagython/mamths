import random


def generate_sparse_matrix(n, off_diagonals_per_row=5, seed=0):
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


def matvec(matrix, x):
    result = [0.0] * len(matrix)
    for i, row in enumerate(matrix):
        s = 0.0
        for j, val in row.items():
            s += val * x[j]
        result[i] = s
    return result


def gauss_seidel(matrix, b, max_iters=1000, tol=1e-8):
    n = len(matrix)
    x = [0.0] * n

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
    n = 100_000
    print(f"Generating random sparse {n} x {n} matrix...")
    matrix = generate_sparse_matrix(n, off_diagonals_per_row=5, seed=42)
    nnz = sum(len(row) for row in matrix)
    print(f"  non-zero entries: {nnz} (~{nnz / n:.1f} per row)")

    x_true = [float(i + 1) for i in range(n)]  # known solution 1..n
    b = matvec(matrix, x_true)

    print("Solving with Gauss-Seidel...")
    x, iters, last_diff = gauss_seidel(matrix, b, max_iters=1000, tol=1e-10)

    max_err = max(abs(x[i] - x_true[i]) for i in range(n))
    print(f"  iterations:        {iters}")
    print(f"  last update diff:  {last_diff:.2e}")
    print(f"  max |x - x_true|:  {max_err:.2e}")

    assert max_err < 1e-5, f"solution did not match: max error {max_err}"
    print("PASS: Gauss-Seidel recovered the known solution.")


if __name__ == "__main__":
    main()
