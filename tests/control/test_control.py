import numpy as np
import pytest

from ss.control import Controller


class TestController:

    @pytest.fixture
    def controller(self) -> Controller:
        return Controller(
            control_dim=2,
            number_of_systems=1,
        )

    @pytest.fixture
    def multi_system_controller(self) -> Controller:
        return Controller(
            control_dim=2,
            number_of_systems=3,
        )

    def test_controller(self, controller: Controller) -> None:
        """Test the controller"""
        controller.compute_control()
        assert controller.control.shape == (2,)
        assert np.allclose(controller.control, np.zeros((1, 2)))
        controller.control = np.array([1.0, 2.0])
        assert np.allclose(controller.control, np.array([1.0, 2.0]))
        controller.control = np.array([[1.0, 2.0]])
        assert np.allclose(controller.control, np.array([1.0, 2.0]))
        with pytest.raises(AssertionError):
            controller.control = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(AssertionError):
            controller.control = np.array([[1.0], [2.0]])
        with pytest.raises(AssertionError):
            controller.control = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_multi_system_controller(
        self, multi_system_controller: Controller
    ) -> None:
        multi_system_controller.compute_control()
        assert multi_system_controller.control.shape == (3, 2)
        assert np.allclose(multi_system_controller.control, np.zeros((3, 2)))
        multi_system_controller.control = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        assert np.allclose(
            multi_system_controller.control,
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ),
        )
        with pytest.raises(AssertionError):
            multi_system_controller.control = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(AssertionError):
            multi_system_controller.control = np.array([[1.0], [2.0]])
        with pytest.raises(AssertionError):
            multi_system_controller.control = np.array([[1.0, 2.0], [3.0, 4.0]])
