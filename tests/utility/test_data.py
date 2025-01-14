import numpy as np
import pytest

from ss.utility.data import Data, MetaData, MetaInfo


class TestMetaData:

    @pytest.fixture
    def meta_data(self) -> MetaData:
        return MetaData(
            key1=np.array([1, 2, 3]),
            key2=MetaData(
                key3=np.array([4, 5, 6]),
            ),
        )

    def test_meta_data(self, meta_data: MetaData) -> None:
        assert np.allclose(meta_data["key1"], [1, 2, 3])
        assert np.allclose(meta_data["key2"]["key3"], [4, 5, 6])


class TestMetaInfo:

    @pytest.fixture
    def meta_info(self) -> MetaInfo:
        return MetaInfo(
            key1=1,
            key2="hello",
            key3=2.5,
        )

    def test_meta_info(self, meta_info: MetaInfo) -> None:
        assert meta_info["key1"] == 1
        assert meta_info["key2"] == "hello"
        assert meta_info["key3"] == 2.5


class TestData:

    @pytest.fixture
    def data(self) -> Data:
        return Data(
            signal_trajectory={
                "time": [0, 1, 2, 3],
                "signal": [3, 1, 4, 1],
            },
            meta_data=MetaData(
                key1=np.array([1, 2, 3]),
                key2=MetaData(
                    key3=np.array([4, 5, 6]),
                ),
            ),
            meta_info=MetaInfo(
                key1=1,
                key2="hello",
                key3=2.5,
            ),
        )

    def test_data(self, data: Data) -> None:
        assert "time" in data
        assert np.allclose(data["signal"], [3, 1, 4, 1])
        del data["signal"]
        assert "signal" not in data
        assert data.keys() == {"time"}
        assert np.allclose(data.meta_data["key1"], [1, 2, 3])
        assert np.allclose(data.meta_data["key2"]["key3"], [4, 5, 6])
        assert data.meta_info["key1"] == 1
        assert data.meta_info["key2"] == "hello"
        assert data.meta_info["key3"] == 2.5
