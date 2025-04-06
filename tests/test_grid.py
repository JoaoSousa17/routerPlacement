from routers.grid import Grid, CellKind, Coord


def parse_grid(*text: str):
    tiles = [[CellKind(ch) for ch in line] for line in text]
    return Grid(tiles)


def test_grid_dims():
    grid = parse_grid(
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
        ".....",
    )

    assert grid.width == 5
    assert grid.height == 6


def test_grid_clamp():
    grid = parse_grid(
        "..",
        "..",
    )

    assert grid.clamp_coord(Coord(1, 1)) == Coord(1, 1)

    assert grid.clamp_coord(Coord(-1, -1)) == Coord(0, 0)
    assert grid.clamp_coord(Coord(100, 100)) == Coord(1, 1)
    assert grid.clamp_coord(Coord(-1, 100)) == Coord(0, 1)
    assert grid.clamp_coord(Coord(100, -1)) == Coord(1, 0)


def test_grid_rect_bounds():
    grid = parse_grid(
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
    )

    cells = []
    kinds = []

    for cell, kind in grid.iter_rectangle_by_bounds(Coord(1, 1), Coord(3, 3)):
        cells.append(cell)
        kinds.append(kind)

    assert cells == [
        Coord(1, 1),
        Coord(2, 1),
        Coord(3, 1),
        Coord(1, 2),
        Coord(2, 2),
        Coord(3, 2),
        Coord(1, 3),
        Coord(2, 3),
        Coord(3, 3),
    ]

    assert kinds == [
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Target,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
    ]

    cells = []
    kinds = []

    for cell, kind in grid.iter_rectangle_by_bounds(Coord(1, 1), Coord(-1, -1)):
        cells.append(cell)
        kinds.append(kind)

    assert cells == [
        Coord(0, 0),
        Coord(1, 0),
        Coord(0, 1),
        Coord(1, 1),
    ]

    assert kinds == [
        CellKind.Target,
        CellKind.Target,
        CellKind.Target,
        CellKind.Wall,
    ]


def test_grid_rect_radius():
    grid = parse_grid(
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
    )

    cells = []
    kinds = []

    for cell, kind in grid.iter_rectangle_by_radius(Coord(2, 2), 1):
        cells.append(cell)
        kinds.append(kind)

    assert cells == [
        Coord(1, 1),
        Coord(2, 1),
        Coord(3, 1),
        Coord(1, 2),
        Coord(2, 2),
        Coord(3, 2),
        Coord(1, 3),
        Coord(2, 3),
        Coord(3, 3),
    ]

    assert kinds == [
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Target,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Wall,
    ]

    cells = []
    kinds = []

    # +---+
    # |.#.|
    # |##.|
    # |...|
    # +---+

    for cell, kind in grid.iter_rectangle_by_radius(Coord(4, 4), 2):
        cells.append(cell)
        kinds.append(kind)

    assert cells == [
        Coord(2, 2),
        Coord(3, 2),
        Coord(4, 2),
        Coord(2, 3),
        Coord(3, 3),
        Coord(4, 3),
        Coord(2, 4),
        Coord(3, 4),
        Coord(4, 4),
    ]

    assert kinds == [
        CellKind.Target,
        CellKind.Wall,
        CellKind.Target,
        CellKind.Wall,
        CellKind.Wall,
        CellKind.Target,
        CellKind.Target,
        CellKind.Target,
        CellKind.Target,
    ]


def test_neighbors():
    grid = parse_grid(
        ".....",
        ".###.",
        ".#.#.",
        ".###.",
        ".....",
        ".....",
    )

    cells = set()

    for n in grid.neighbours(Coord(2, 2)):
        cells.add(n)

    assert cells == {
        Coord(1, 1),
        Coord(2, 1),
        Coord(3, 1),
        Coord(3, 2),
        Coord(3, 3),
        Coord(3, 3),
        Coord(2, 3),
        Coord(1, 3),
        Coord(1, 2),
    }

    cells = set()

    for n in grid.neighbours(Coord(0, 0)):
        cells.add(n)

    assert cells == {
        Coord(1, 0),
        Coord(1, 1),
        Coord(0, 1),
    }
