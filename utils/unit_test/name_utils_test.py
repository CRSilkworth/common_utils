import pytest
from typing import List
from utils.type_utils import Position, PositionDict


def test_position_dict():
    # Test valid PositionDict
    pos_dict: PositionDict = {"x": 10, "y": 20}
    assert pos_dict["x"] == 10
    assert pos_dict["y"] == 20

    # Test invalid PositionDict (should raise error if type checking is enforced)
    with pytest.raises(KeyError):
        pos_dict["z"]


def test_position():
    # Test valid Position tuple
    pos: Position = (10, 20)
    assert pos[0] == 10
    assert pos[1] == 20

    # Test invalid Position tuple (should raise error if type checking is enforced)
    with pytest.raises(IndexError):
        _ = pos[2]


def test_position_list():
    pos_list: List[Position] = [(10, 20), (30, 40), (50, 60)]
    assert len(pos_list) == 3
    assert pos_list[0] == (10, 20)
    assert pos_list[1] == (30, 40)
    assert pos_list[2] == (50, 60)

    # Test invalid Position in list (should raise error if type checking is enforced)
    with pytest.raises(IndexError):
        _ = pos_list[0][2]


def test_position_dict_conversion():
    # Convert from PositionDict to Position and back
    pos_dict: PositionDict = {"x": 10, "y": 20}
    pos: Position = (pos_dict["x"], pos_dict["y"])
    assert pos == (10, 20)

    # Convert back to PositionDict
    pos_dict_converted: PositionDict = {"x": pos[0], "y": pos[1]}
    assert pos_dict_converted == pos_dict


if __name__ == "__main__":
    pytest.main()
