# Implement a Matrix class with the following attributes:
# - entries: this will be a list containing the entries of the matrix

# implement the following methods:
# __str__: to print the matrix
# __getitem__(row_idx, col_idx): returns the entry at the given row index and column index
# __add__(other: Matrix) -> Matrix: adds the matrices (THIS SHOULD CHECK THEY're compatible for addition, and raise an error if they're not)
# __mul__(other: Matrix) -> Matrix: multiplies the matrices (THIS SHOULD CHECK THEY're compatible for multiplication, and if they're not it should raise an ERROR)
# __mul__(scalar: float) -> Matrix: multiplies with a scalar

# anything else you'd like to add

# TEST IT WITH THE MATRICES FROM THE EXERCISES IN THE CLASS!

from typing import Iterable, Self

import pytest


class Matrix:
    def __init__(self, data: tuple[tuple[float, ...], ...]):
        self.data = data
        self.rows = len(data)
        self.columns = len(data[0])
        # assumes there is at least one row and every row is same width :/

    @classmethod
    def identity(cls, size: int) -> Self:
        return cls(
            tuple(tuple((1 if i == j else 0) for j in range(size)) for i in range(size))
        )

    def __str__(self) -> str:
        """print the matrix"""
        return "\n".join(str(row) for row in self.data)

    def __getitem__(self, idxs: tuple[int, int]):
        """return entry at given row index and column index"""
        row_idx, col_idx = idxs
        return self.data[row_idx][col_idx]

    def __add__(self, other: "Matrix") -> "Matrix":
        """add matrix to another matrix"""
        nm = []
        for row in range(self.rows):
            nmrow = []
            for entry in range(self.columns):
                nmrow.append(self[row, entry] + other[row, entry])
            nm.append(nmrow)

        return Matrix(tuple(tuple(thing) for thing in nm))

    def __mul__(self, other: "Matrix | float | int") -> "Matrix":
        """multiply matrix by another matrix OR a scalar"""

        if type(other) is Matrix:
            if self.columns == other.rows:
                # 2x2
                # a b   e f   ae+bg af+bh
                #     x     =
                # c d   g h   ce+dg cf+dh
                # ((a, b), (c, d)) * ((e, f), (g, h)) = ((a * e + b * g, a * f + b * h), (c * e + d * g, c * f + d * h))
                # self * other = out

                # self = Matrix(((0, -2), (-2, -5)))
                # other = Matrix(((6, -6), (3, 0)))
                # out = (
                #     (
                #         self[0, 0] * other[0, 0] + self[0, 1] * other[1, 0],
                #         self[0, 0] * other[0, 1] + self[0, 1] * other[1, 1],
                #     ),
                #     (
                #         self[1, 0] * other[0, 0] + self[1, 1] * other[1, 0],
                #         self[1, 0] * other[0, 1] + self[1, 1] * other[1, 1],
                #     ),
                # )

                # abds = [[0.0] * 2 for _ in range(2)]
                # # out2 = ((a, b), (c, d))
                # for i in range(2):
                #     for j in range(2):
                #         abds[i][j] = self[i, 0] * other[0, j] + self[i, 1] * other[1, j]

                k = self.columns

                abds2 = [[0.0] * other.columns for _ in range(self.rows)]

                for i in range(self.rows):
                    for j in range(other.columns):
                        abds2[i][j] = sum(
                            self[i, index] * other[index, j] for index in range(k)
                        )

                return Matrix(tuple(tuple(thing) for thing in abds2))
            else:
                raise Exception("uhh cant multiply these bro")
        elif type(other) is float or type(other) is int:
            return Matrix(tuple(tuple(other * h for h in row) for row in self.data))
        else:
            raise Exception("mate what")

    def __iter__(self) -> Iterable[tuple]:  # for unpacking with * and iterating rows
        return iter(self.data)

    def __eq__(self, other) -> bool:
        if self.data == other.data:
            return True
        else:
            return False

    def transpose(self) -> "Matrix":
        return Matrix(
            tuple(
                tuple(row[column] for row in self.data)
                for column in range(self.columns)
            )
        )

    def determinant(self):
        pass

    def inverse(self) -> "Matrix":
        """Returns the inverse of the matrix using Gauss-Jordan elimination."""
        if self.rows != self.columns:
            raise Exception("Only square matrices have inverses.")

        n = self.rows
        # Create augmented matrix [A | I]
        augmented = [
            list(self.data[i]) + [(1.0 if i == j else 0.0) for j in range(n)]
            for i in range(n)
        ]

        for i in range(n):
            # Find pivot
            pivot_row = i
            while pivot_row < n and abs(augmented[pivot_row][i]) < 1e-12:
                pivot_row += 1

            if pivot_row == n:
                raise Exception("Matrix is singular and cannot be inverted.")

            # Swap rows
            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]

            # Scale pivot row to 1
            pivot_val = augmented[i][i]
            augmented[i] = [x / pivot_val for x in augmented[i]]

            # Eliminate other rows
            for j in range(n):
                if i != j:
                    factor = augmented[j][i]
                    augmented[j] = [
                        augmented[j][k] - factor * augmented[i][k]
                        for k in range(2 * n)
                    ]

        # Extract the right half
        inv_data = tuple(tuple(row[n:]) for row in augmented)
        return Matrix(inv_data)


# matrices from homework: numbered by problem number and left/right, accordingly
# or uhh tuple pairs of (left, right) i guess that works

matrices1 = ((0, 2), (-2, -5)), ((6, -6), (3, 0))
matrices2 = ((6), (-3)), ((-5), (4))
matrices8 = ((3, 2, 5), (2, 3, 1)), ((4, 5, -5), (5, -1, 6))  # undefined product


class TestMatrix:
    def test_str(self):
        assert str(Matrix(matrices8[0])) == """(3, 2, 5)\n(2, 3, 1)"""

    def test_getitem(self):
        m1 = Matrix(matrices1[0])
        assert m1.data == ((0, 2), (-2, -5))

    def test_add(self):
        added1 = Matrix(matrices1[0]) + Matrix(matrices1[1])
        assert added1.data == ((6, -4), (1, -5))

    def test_mul_matrix(self):
        m3 = Matrix(matrices1[0]) * Matrix(matrices1[1])
        assert m3.data == ((6, 0), (-27, 12))
        with pytest.raises(Exception) as e_info:
            xips = Matrix(matrices8[0]) * Matrix(matrices8[1])

    def test_mul_scalar(self):
        m3 = Matrix(((4, 0), (1, -9))) * 2
        assert m3.data == ((8, 0), (2, -18))

    def test_transpose(self):
        m1 = Matrix(matrices8[0])
        m2 = Matrix(matrices1[0])
        m3 = Matrix(matrices1[1])
        assert m1.transpose().data == ((3, 2), (2, 3), (5, 1))
        assert (m2 * m3).transpose().data == (m3.transpose() * m2.transpose()).data
        assert (m2 + m3).transpose().data == (m2.transpose() + m3.transpose()).data

    def test_identity(self):
        assert Matrix.identity(4).data == (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )

    def test_inverse(self):
        # Test 2x2
        m = Matrix(((1, 2), (3, 4)))
        inv = m.inverse()
        # Inverse of [[1, 2], [3, 4]] is [[-2, 1], [1.5, -0.5]]
        assert inv.data[0][0] == pytest.approx(-2.0)
        assert inv.data[0][1] == pytest.approx(1.0)
        assert inv.data[1][0] == pytest.approx(1.5)
        assert inv.data[1][1] == pytest.approx(-0.5)

        # Test Identity property: M * M^-1 = I
        res = m * inv
        identity = Matrix.identity(2)
        for i in range(2):
            for j in range(2):
                assert res[i, j] == pytest.approx(identity[i, j])


def main():
    print("Hello from matrix-class!")


if __name__ == "__main__":
    main()
