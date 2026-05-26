import matplotlib.pylab as plt

x = [0.0]
y = [0.0]
z = [0.0]

# 6x + 2y - 1z = 4
# 1x + 5y + 1z = 3
# 2x + 1y + 4z = 27
sle = (
    (6, 2, -1, 4),
    (1, 5, 1, 3),
    (2, 1, 4, 27),
)

for i in range(1, 100):
    xi = (sle[0][-1] - sle[0][1] * y[i - 1] - sle[0][2] * z[i - 1]) / sle[0][0]
    yi = (sle[1][-1] - sle[1][0] * x[i - 1] - sle[1][2] * z[i - 1]) / sle[1][1]
    zi = (sle[2][-1] - sle[2][0] * x[i - 1] - sle[2][1] * y[i - 1]) / sle[2][2]

    # Add latest value to history
    x.append(xi)
    y.append(yi)
    z.append(zi)
print(x[-1], y[-1], z[-1])

A = (
    (6, 2, -1),
    (1, 5, 1),
    (2, 1, 4),
)
b = (
    4,
    3,
    27,
)
x = [0.0 for i in range(len(b))]
eps = 1e-6

n = len(b)
while True:
    x_new = [0.0 for _ in range(n)]
    for i in range(n):
        sigma = 0.0
        for j in range(n):
            if j != i:
                sigma += A[i][j] * x[j]
        x_new[i] = (b[i] - sigma) / A[i][i]
    if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
        x = x_new
        break
    x = x_new
print(x)

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

# Plot History of values
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.plot(z, label="z")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend(loc=1)
plt.show()
