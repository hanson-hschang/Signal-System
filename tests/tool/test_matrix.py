import pytest
import numpy as np
from numpy.typing import NDArray
from typing import Any

# Assuming the MatrixDescriptor is in a file named matrix_descriptor.py
from tool.matrix_descriptor import MatrixDescriptor

class TestClass:
    matrix = MatrixDescriptor("rows", "cols")
    
    def __init__(self) -> None:
        self.rows = 3
        self.cols = 2
        self._matrix = np.zeros((3, 2))

def test_matrix_descriptor_initialization() -> None:
    descriptor = MatrixDescriptor("rows", "cols")
    assert descriptor.name_of_row_numbers == "rows"
    assert descriptor.name_of_column_numbers == "cols"

def test_set_name() -> None:
    descriptor = MatrixDescriptor("rows", "cols")
    descriptor.__set_name__(TestClass, "matrix")
    assert descriptor.name == "matrix"
    assert descriptor.private_name == "_matrix"

def test_get_2d_matrix() -> None:
    test_obj = TestClass()
    test_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    test_obj._matrix = test_matrix
    
    result = test_obj.matrix
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, test_matrix)

def test_get_1d_matrix() -> None:
    class SingleColumnTest:
        matrix = MatrixDescriptor("rows", "cols")
        
        def __init__(self) -> None:
            self.rows = 3
            self.cols = 1
            self._matrix = np.zeros((3, 1))
    
    test_obj = SingleColumnTest()
    test_matrix = np.array([[1.0], [2.0], [3.0]])
    test_obj._matrix = test_matrix
    
    result = test_obj.matrix
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, test_matrix.squeeze())
    assert result.shape == (3,)

def test_set_2d_matrix() -> None:
    test_obj = TestClass()
    test_matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    
    test_obj.matrix = test_matrix
    assert isinstance(test_obj._matrix, np.ndarray)
    assert np.array_equal(test_obj._matrix, np.array(test_matrix))
    assert test_obj._matrix.dtype == np.float64

def test_set_1d_matrix() -> None:
    class SingleColumnTest:
        matrix = MatrixDescriptor("rows", "cols")
        
        def __init__(self) -> None:
            self.rows = 3
            self.cols = 1
            self._matrix = np.zeros((3, 1))
    
    test_obj = SingleColumnTest()
    test_matrix = [1.0, 2.0, 3.0]
    
    test_obj.matrix = test_matrix
    assert isinstance(test_obj._matrix, np.ndarray)
    assert test_obj._matrix.shape == (3, 1)
    assert np.array_equal(test_obj._matrix, np.array(test_matrix).reshape(-1, 1))
