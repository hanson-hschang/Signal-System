from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ss.utility.assertion import is_positive_integer


def get_estimation_model(
    emission_matrix: NDArray,
) -> Any:
    @njit(cache=True)  # type: ignore
    def estimation_model(
        estimated_state: NDArray,
        emission_matrix: NDArray[np.float64] = emission_matrix,
    ) -> NDArray:
        estimation = estimated_state @ emission_matrix
        return estimation

    return estimation_model


def softmax(vector: NDArray, temperature: float = 3.0) -> NDArray:
    exp_vector = np.exp(vector / temperature)
    normalized_vector: NDArray = exp_vector / np.sum(exp_vector)
    return normalized_vector


def get_probability_matrix(
    nrows: int, ncols: int, temperature: float = 3
) -> NDArray:
    assert is_positive_integer(nrows), (
        "The number of rows must be a positive integer. "
        f"The given {nrows = }."
    )
    assert is_positive_integer(ncols) and (ncols >= 2), (
        "The number of columns must be a positive integer with minimum value of 2. "
        f"The given {ncols = }."
    )
    min_number_of_non_zero_elements = max(2, ncols // 3)
    probability_matrix = np.empty((nrows, ncols))
    for r in range(nrows):
        probability_vector = np.zeros(ncols)

        # Decide how many non-zero elements to generate
        number_of_non_zero_elements = np.random.randint(
            min_number_of_non_zero_elements, ncols + 1
        )

        # Generate the indices of the non-zero elements
        indices = np.random.choice(
            ncols, number_of_non_zero_elements, replace=False
        )

        # Generate the values of the non-zero elements
        values = np.random.rand(number_of_non_zero_elements)
        values = softmax(values, temperature)

        # Assign the values to the corresponding indices
        probability_vector[indices] = values

        # Assign the probability vector to the probability matrix
        probability_matrix[r, :] = probability_vector
    return probability_matrix
