from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from tool.matrix_descriptor import MatrixDescriptor


class MatrixClass:
    def __init__(self) -> None:
        self.num_rows = 2
        self.num_cols = 3
        self._value = np.zeros((self.num_rows, self.num_cols))

    value = MatrixDescriptor("num_rows", "num_cols")


class VectorClass:
    def __init__(self) -> None:
        self.num_rows = 1
        self.num_cols = 3
        self._value = np.zeros((self.num_rows, self.num_cols))

    value = MatrixDescriptor("num_rows", "num_cols")


class TestMatrixDescriptor:

    @pytest.fixture
    def matrix(self) -> MatrixClass:
        return MatrixClass()

    @pytest.fixture
    def vector(self) -> VectorClass:
        return VectorClass()

    def test_matrix_initialization(self, matrix: MatrixClass) -> None:
        assert matrix.value.shape == (2, 3)

    def test_matrix_assignment(self, matrix: MatrixClass) -> None:
        new_matrix = [[1, 2.5, 3], [4.5, 5, 6.5]]
        matrix.value = new_matrix
        assert np.all(matrix.value == np.array(new_matrix))

    def test_matrix_single_row_squeeze(self, vector: VectorClass) -> None:
        new_vector = [1, 2, 3]
        vector.value = new_vector
        assert np.all(vector.value == np.array(new_vector))

    def test_matrix_wrong_shape(self, matrix: MatrixClass) -> None:
        with pytest.raises(AssertionError):
            matrix.value = [
                [1, 2],
                [3, 4],
            ]  # Wrong shape (2x2 instead of 2x3)

    def test_private_attribute_exists(self, matrix: MatrixClass) -> None:
        assert hasattr(matrix, "_value")
