from matrix import Matrix


def solvelinear(A: Matrix, b: Matrix):
    Ai = A.inverse()
    return Ai * b


def main():
    A = Matrix(((1, 1, 1), (1, 2, -1), (1, -1, 3)))
    b = Matrix(((3,), (2,), (3,)))

    x = solvelinear(A, b)

    print(x)


if __name__ == "__main__":
    main()
