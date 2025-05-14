from typing import List, Optional, Tuple, Dict
import math
import networkx as nx
import numpy as np
from utils.type_utils import Position, PositionDict


def bounding_box(
    positions: List[PositionDict],
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Calculate the bounding box for a list of position dictionaries.

    Args:
        positions (List[PositionDict]): A list of dictionaries, each containing 'x' and
            'y' keys representing positions.

    Returns:
        Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]: The bounding box
            coordinates as ((min_x, max_x), (min_y, max_y)).
    """
    if not positions:
        return None, None
    if len(positions) == 1:
        position = positions[0]
        return (position["x"], position["x"]), (position["y"], position["y"])

    min_x, max_x = None, None
    min_y, max_y = None, None
    for position in positions:
        if min_x is None or position["x"] < min_x:
            min_x = position["x"]
        if max_x is None or position["x"] > max_x:
            max_x = position["x"]

        if min_y is None or position["y"] < min_y:
            min_y = position["y"]
        if max_y is None or position["y"] > max_y:
            max_y = position["y"]
    return (min_x, max_x), (min_y, max_y)


def bounding_box_tuples(
    positions: List[Position],
) -> Tuple[Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int]]]:
    """
    Calculate the bounding box for a list of positions represented as tuples or lists.

    Args:
        positions (List[Position]): A list of positions, each represented as a tuple or
            list of (x, y) coordinates.

    Returns:
        Tuple[Tuple[Optional[int], Optional[int]], Tuple[Optional[int],
            Optional[int]]]: The bounding box coordinates as ((min_x, max_x),
            (min_y, max_y)).
    """
    if not positions:
        return (None, None), (None, None)
    if len(positions) == 1:
        position = positions[0]
        return (position[0], position[0]), (position[1], position[1])

    min_x, max_x = None, None
    min_y, max_y = None, None
    for position in positions:
        if min_x is None or position[0] < min_x:
            min_x = position[0]
        if max_x is None or position[0] > max_x:
            max_x = position[0]

        if min_y is None or position[1] < min_y:
            min_y = position[1]
        if max_y is None or position[1] > max_y:
            max_y = position[1]
    return (min_x, max_x), (min_y, max_y)


def place_in_grid(
    num_new: int,
    node_positions: Optional[List[Position]] = None,
    margin: int = 50,
    center: Position = (0, 0),
) -> List[Position]:
    """
    Generate positions for new nodes arranged in a grid.

    Args:
        num_new (int): The number of new nodes to place.
        node_positions (Optional[List[Position]]): Existing node positions to base
            the new placements on. Defaults to None.
        margin (int): The margin between nodes. Defaults to 50.

    Returns:
        List[Position]: List of positions for the new nodes.
    """
    if not num_new:
        return []
    width = math.ceil(math.sqrt(num_new))
    new_positions = [
        (margin * (i % width), margin * (i // width)) for i in range(num_new)
    ]

    if not node_positions:
        return [(p[0] + center[0], p[1] + center[1]) for p in new_positions]

    (min_x, max_x), (min_y, max_y) = bounding_box_tuples(node_positions)

    center_x = int(max_x + min_x) // 2
    center_x = min_x if center_x == 0 else center_x

    new_positions = [(x + center_x, y + max_y + margin) for x, y in new_positions]
    return new_positions


def place_in_spiral(
    parent_center: Position, num_nodes: int, num_new: int, margin: int = 20
) -> List[Position]:
    """
    Generate positions for new nodes around a central parent in a spiral pattern.

    Args:
        parent_center (Position): The central position around which nodes are placed.
        num_nodes (int): The total number of nodes already placed.
        num_new (int): The number of new nodes to place.
        margin (int): The margin between nodes. Defaults to 20.

    Returns:
        List[Position]: List of positions for the new nodes.
    """
    points = []
    a = 0
    b = 1.2 * margin / (2 * np.pi)
    theta = 0
    r = a

    points.append((int(parent_center[0]), int(parent_center[1])))

    for _ in range(1, num_new + num_nodes):
        theta += margin / (np.sqrt(b**2 + r**2))
        r = a + b * theta
        points.append(
            (
                int(parent_center[0] + r * np.cos(theta)),
                int(parent_center[1] + r * np.sin(theta)),
            )
        )

    return points[num_nodes:]


def make_tree(
    edges: List[Tuple[str, str]],
    root_id: str,
    root_pos: Position,
    margin: int = 100,
    directed: bool = True,
) -> Dict[str, Position]:
    """
    Create a layout for a tree graph with nodes placed in levels.

    Args:
        edges (List[Tuple[str, str]]): List of edges in the graph, each represented as
            a tuple (parent, child).
        root_id (str): The ID of the root node.
        root_pos (Position): The position of the root node.
        margin (int): The margin between nodes. Defaults to 100.
        directed (bool): Whether the graph is directed. Defaults to True.

    Returns:
        Dict[str, Position]: Dictionary of node IDs and their positions.
    """
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)

    pos = {}
    levels = {}

    def dfs(node: str, level: int = 0, parent: Optional[str] = None):
        """
        Depth-first search to determine the levels of nodes.

        Args:
            node (str): Current node ID.
            level (int): Current level in the tree.
            parent (Optional[str]): Parent node ID to avoid traversing back to the
                parent.
        """
        if node in levels:
            return
        levels[node] = level
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            if (
                directed or neighbor != parent
            ):  # Avoid traversing back to the parent node
                dfs(neighbor, level + 1, node)

    # Perform DFS to determine the levels of each node
    dfs(root_id)

    # Group nodes by their levels
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)

    # Place nodes in x, y positions
    y = root_pos[1]
    for level in sorted(level_nodes.keys()):
        x = root_pos[0]
        for node in level_nodes[level]:
            pos[node] = (x, y)
            x += margin
        y += margin
    return pos
