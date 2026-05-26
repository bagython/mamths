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


print(jacobi_naive(A, b, eps))

# Input: initial guess x(0) to the solution, (diagonal dominant) matrix A, right-hand side vector b, convergence criterion
# Output: solution when convergence is reached
# Comments: pseudocode based on the element-based formula above

# k = 0
# while convergence not reached do
#     for i := 1 step until n do
#         σ = 0
#         for j := 1 step until n do
#             if j ≠ i then
#                 σ = σ + aij xj(k)
#             end
#         end
#         xi(k+1) = (bi − σ) / aii
#     end
#     increment k
# end
