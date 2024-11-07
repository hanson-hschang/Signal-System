import pytest
import numpy as np
from numpy.typing import NDArray
from typing import Any

from tool.matrix_descriptor import MatrixDescriptor

class TestMatrixDescriptor:
    class MatrixClass:
        num_rows = 2
        num_cols = 3
        matrix = MatrixDescriptor("num_rows", "num_cols")
        
        def __init__(self):
            self._matrix = np.zeros((self.num_rows, self.num_cols))

    class VectorClass:
        num_rows = 1
        num_cols = 3
        vector = MatrixDescriptor("num_rows", "num_cols")
        
        def __init__(self):
            self._vector = np.zeros((self.num_rows, self.num_cols))

    def test_matrix_initialization(self):
        test_obj = self.MatrixClass()
        expected = np.zeros((2, 3))
        assert np.all(test_obj.matrix == expected)

    def test_matrix_assignment(self):
        test_obj = self.MatrixClass()
        new_matrix = [[1, 2.5, 3], [4.5, 5, 6.5]]
        test_obj.matrix = new_matrix
        assert np.all(test_obj.matrix == np.array(new_matrix))

    def test_matrix_single_row_squeeze(self):
        test_obj = self.VectorClass()
        test_obj.vector = [1, 2, 3]
        assert np.all(test_obj.vector == np.array([1, 2, 3]))

    def test_matrix_wrong_shape(self):
        test_obj = self.MatrixClass()
        with pytest.raises(AssertionError):
            test_obj.matrix = [[1, 2], [3, 4]]  # Wrong shape (2x2 instead of 2x3)

    def test_private_attribute_exists(self):
        test_obj = self.MatrixClass()
        assert hasattr(test_obj, "_matrix")
