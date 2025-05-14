import pytest
from utils.position_utils import (
    bounding_box,
    bounding_box_tuples,
    place_in_grid,
    place_in_spiral,
    make_tree,
)


def test_bounding_box():
    positions = [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
        {"x": 0, "y": 5},
    ]
    assert bounding_box(positions) == ((0, 3), (2, 5))

    positions = [{"x": 1, "y": 2}]
    assert bounding_box(positions) == ((1, 1), (2, 2))

    assert bounding_box([]) == (None, None)


def test_bounding_box_tuples():
    positions = [
        (1, 2),
        (3, 4),
        (0, 5),
    ]
    assert bounding_box_tuples(positions) == ((0, 3), (2, 5))

    positions = [(1, 2)]
    assert bounding_box_tuples(positions) == ((1, 1), (2, 2))

    assert bounding_box_tuples([]) == ((None, None), (None, None))


def test_place_new_comps():
    num_new_comps = 4
    comp_positions = [(10, 10), (20, 20)]
    margin = 50
    expected_positions = [(15, 70), (65, 70), (15, 120), (65, 120)]
    actual_positions = place_in_grid(num_new_comps, comp_positions, margin)
    assert actual_positions == expected_positions

    num_new_comps = 1
    expected_positions = [(0, 0)]
    actual_positions = place_in_grid(num_new_comps, None, margin)
    assert actual_positions == expected_positions


def test_place_new_values():
    comp_center = (0, 0)
    num_values = 0
    num_new_values = 3
    margin = 20
    result = place_in_spiral(comp_center, num_values, num_new_values, margin)
    assert len(result) == num_new_values
    assert result[0] == (0, 0)  # Center point

    num_values = 3
    num_new_values = 3
    result = place_in_spiral(comp_center, num_values, num_new_values, margin)
    assert len(result) == num_new_values


def test_make_tree():
    edges = [
        ("root", "child1"),
        ("root", "child2"),
        ("child1", "grandchild1"),
        ("child1", "grandchild2"),
    ]
    root_id = "root"
    root_pos = (0, 0)
    margin = 100
    expected_positions = {
        "root": (0, 0),
        "child1": (0, 100),
        "child2": (100, 100),
        "grandchild1": (0, 200),
        "grandchild2": (100, 200),
    }
    result = make_tree(edges, root_id, root_pos, margin)
    assert result == expected_positions

    # Test with directed = False
    result = make_tree(edges, root_id, root_pos, margin, directed=False)
    assert result == expected_positions


if __name__ == "__main__":
    pytest.main()
