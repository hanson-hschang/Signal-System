import numpy as np
import pytest

from ss.tool.descriptor import TensorDescriptor


class VectorClass:
    def __init__(self) -> None:
        self._num_dim_1 = 3
        self._value = np.zeros(self._num_dim_1)

    value = TensorDescriptor("_num_dim_1")


class MatrixClass:
    def __init__(self) -> None:
        self._num_dim_1 = 2
        self._num_dim_2 = 3
        self._value = np.zeros((self._num_dim_1, self._num_dim_2))

    value = TensorDescriptor("_num_dim_1", "_num_dim_2")


class TestTensorDescriptor:

    @pytest.fixture
    def vector(self) -> VectorClass:
        return VectorClass()

    @pytest.fixture
    def matrix(self) -> MatrixClass:
        return MatrixClass()

    def test_vector_assignment(self, vector: VectorClass) -> None:
        new_vector = [1, 2, 3]
        vector.value = new_vector
        assert np.all(vector.value == np.array(new_vector))

    def test_matrix_initialization(self, matrix: MatrixClass) -> None:
        assert matrix.value.shape == (2, 3)

    def test_matrix_assignment(self, matrix: MatrixClass) -> None:
        new_matrix = [[1, 2.5, 3], [4.5, 5, 6.5]]
        matrix.value = new_matrix
        assert np.all(matrix.value == np.array(new_matrix))

    def test_matrix_wrong_shape(self, matrix: MatrixClass) -> None:
        with pytest.raises(AssertionError):
            matrix.value = [
                [1, 2],
                [3, 4],
            ]  # Wrong shape (2x2 instead of 2x3)

    def test_private_attribute_exists(self, matrix: MatrixClass) -> None:
        assert hasattr(matrix, "_value")
