import matplotlib.pylab as plt

x = [0.0]
y = [0.0]
z = [0.0]

# 6x + 2y - 1z = 4
# 1x + 5y + 1z = 3
# 2x + 1y + 4z = 27


for i in range(1, 100):
    xi = (4 - (2 * y[i - 1]) - (-1 * z[i - 1])) / 6
    yi = (3 - (1 * x[i - 1]) - (1 * z[i - 1])) / 5
    zi = (27 - (2 * x[i - 1]) - (1 * y[i - 1])) / 4

    # Add latest value to history
    x.append(xi)
    y.append(yi)
    z.append(zi)
print(x[-1], y[-1], z[-1])
# Plot History of values
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.plot(z, label="z")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend(loc=1)
plt.show()
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
