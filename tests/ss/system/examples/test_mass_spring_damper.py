import numpy as np
import pytest

from ss.system.examples.mass_spring_damper import MassSpringDamperSystem


class TestMassSpringDamperSystem:

    @pytest.fixture
    def system(self) -> MassSpringDamperSystem:
        """Create multiple mass spring damper systems"""
        return MassSpringDamperSystem(
            number_of_connections=2,
        )

    def test_system_initialization(
        self, system: MassSpringDamperSystem
    ) -> None:
        np.testing.assert_allclose(
            system.state_space_matrix_A,
            [[0, 0, 1, 0], [0, 0, 0, 1], [-2, 1, -2, 1], [1, -1, 1, -1]],
        )

    def test_create_multiple_systems(
        self, system: MassSpringDamperSystem
    ) -> None:
        multi_system = system.create_multiple_systems(number_of_systems=2)
        assert multi_system.number_of_systems == 2
