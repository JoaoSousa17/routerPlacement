from typing import FrozenSet, Generator, Tuple, Iterable, TypeAlias
from numpy.typing import NDArray

from routers.spatial import Coord
from routers.utils import clamp

from dataclasses import dataclass
from pathlib import Path
from enum import IntEnum

import numpy as np


__all__ = ["CellKind", "Grid", "Problem"]


class CellKind(IntEnum):
    Void = 0
    Target = 1
    Wall = 2

    @classmethod
    def FromASCII(cls, ch: str):
        ascii_map = {
            "-": cls.Void,
            ".": cls.Target,
            "#": cls.Wall,
        }

        try:
            return ascii_map[ch]
        except KeyError:
            raise ValueError(f"Invalid tile '{ch}'")


Mask: TypeAlias = NDArray[np.bool]


class Grid:

    width: int
    height: int

    _cells: NDArray[np.uint8]
    _prefix_sum: NDArray[np.int64]

    _nonwalls: FrozenSet[Coord]
    _targets: FrozenSet[Coord]

    @classmethod
    def FromASCII(cls, lines: Iterable[str]):
        try:
            cells = [[CellKind.FromASCII(ch) for ch in line] for line in lines]
        except Exception as e:
            raise RuntimeError(f"Couldn't parse grid: {e}") from e

        return cls(cells)

    def __init__(self, tiles: Iterable[Iterable[CellKind]]):
        self._cells = np.asarray(tiles, dtype=np.uint8).transpose()
        self.width = self._cells.shape[0]
        self.height = self._cells.shape[1]

        if self.height == 0 or self.width == 0:
            raise ValueError("Grid cannot be empty")

        walls: NDArray[np.uint8] = (self._cells == CellKind.Wall).astype(np.uint8)

        self._prefix_sum = np.zeros((self.width + 1, self.height + 1), dtype=np.int64)
        self._prefix_sum[1:, 1:] = walls.cumsum(axis=0).cumsum(axis=1).astype(np.int64)

        nonwalls = []
        targets = []

        for coord, cell in self:
            if cell == CellKind.Target:
                targets.append(coord)

            if cell != CellKind.Wall:
                nonwalls.append(coord)

        self._nonwalls = frozenset(nonwalls)
        self._targets = frozenset(targets)

    def nonwall_cells(self):
        return self._nonwalls

    def target_cells(self):
        return self._targets

    def __repr__(self):
        return f"<Grid [w:{self.width}, h:{self.height}]>"

    def __getitem__(self, coord: Coord) -> CellKind:
        return CellKind(self._cells[coord.x, coord.y])

    def __iter__(self):
        for (x, y), cell in np.ndenumerate(self._cells):
            yield Coord(x, y), CellKind(cell)

    def matrix(self):
        return self._cells.copy()

    def sees(self, a: Coord, b: Coord):
        tl, br = Coord.topleft_bottomright(a, b)
        br = br.down(1).right(1)

        return (
            self._prefix_sum[br.x, br.y]
            - self._prefix_sum[br.x, tl.y]
            - self._prefix_sum[tl.x, br.y]
            + self._prefix_sum[tl.x, tl.y]
        ) == 0

    def clamp_coord(self, coord: Coord) -> Coord:
        return Coord(
            clamp(coord.x, 0, self.width - 1),
            clamp(coord.y, 0, self.height - 1),
        )

    def neighbours(self, coord: Coord):
        for neighbour in coord.neighbours():
            if self.is_within(neighbour):
                yield neighbour

    def nonwall_neighbours(self, coord: Coord):
        for neigh in self.neighbours(coord):
            if self[neigh] != CellKind.Wall:
                yield neigh

    def is_within(self, coord: Coord) -> bool:
        return (0 <= coord.x < self.width) and (0 <= coord.y < self.height)

    def _iter_bounds_unchecked(
        self,
        top_left: Coord,
        bottom_right: Coord,
    ) -> Generator[Tuple[Coord, CellKind], None, None]:
        subgrid = self._cells[
            top_left.x : bottom_right.x + 1,
            top_left.y : bottom_right.y + 1,
        ]

        for (x, y), cell in np.ndenumerate(subgrid):
            yield Coord(top_left.x + x, top_left.y + y), CellKind(cell)

    def iter_rectangle_by_bounds(
        self,
        coord1: Coord,
        coord2: Coord,
    ) -> Generator[Tuple[Coord, CellKind], None, None]:
        tl, br = Coord.topleft_bottomright(coord1, coord2)

        yield from self._iter_bounds_unchecked(
            self.clamp_coord(tl),
            self.clamp_coord(br),
        )

    def iter_rectangle_by_radius(self, center: Coord, radius: int):
        assert radius >= 0

        top_left = self.clamp_coord(center.up(radius).left(radius))
        bottom_right = self.clamp_coord(center.down(radius).right(radius))

        yield from self._iter_bounds_unchecked(top_left, bottom_right)

    def non_walls_within_radius(self, center: Coord, radius: int):
        for coord, cell in self.iter_rectangle_by_radius(center, radius):
            if cell != CellKind.Wall:
                yield coord

    def visible_targets(self, center: Coord, radius: int):
        for coord, cell in self.iter_rectangle_by_radius(center, radius):
            if cell == CellKind.Target and self.sees(coord, center):
                yield coord

    def visible_targets_mask(
        self,
        center: Coord,
        radius: int,
    ) -> Mask:

        diameter = 2 * radius + 1
        origin = center.upleft(radius)
        mask = np.zeros((diameter, diameter), dtype=np.bool)

        xs = []
        ys = []

        for coord, cell in self.iter_rectangle_by_radius(center, radius):
            if cell == CellKind.Target and self.sees(coord, center):
                xs.append(coord.x - origin.x)
                ys.append(coord.y - origin.y)

        if xs and ys:
            mask[xs, ys] = 1

        return mask

    def mask_to_coords(self, mask: Mask, center: Coord):

        radius = mask.shape[0] // 2
        origin = center.upleft(radius)

        for (x, y), tf in np.ndenumerate(mask):
            if tf:
                yield Coord(origin.x + x, origin.y + y)


@dataclass(frozen=True)
class Problem:
    grid: Grid
    router_radius: int

    cable_price: int
    router_price: int
    budget: int

    backbone: Coord

    @classmethod
    def ParseInput(cls, path: Path):

        with open(path) as file:
            lines_iter = iter(file.read().splitlines())

        try:
            height, width, router_range = map(int, next(lines_iter).split())
            cable_price, router_price, budget = map(int, next(lines_iter).split())
            by, bx = map(int, next(lines_iter).split())
            backbone = Coord(bx, by)

            assert router_price >= 0
            assert cable_price >= 0
            assert budget >= 0

        except Exception as e:
            raise RuntimeError(f"Invalid input file header: {e}") from e

        grid = Grid.FromASCII(lines_iter)

        if grid.width != width or grid.height != height:
            raise RuntimeError(f"Invalid input file: dimensions do not match.")

        if not grid.is_within(backbone):
            raise RuntimeError(f"Invalid input file, backbone is not within the grid")

        return cls(
            grid,
            router_range,
            cable_price,
            router_price,
            budget,
            backbone,
        )

    def router_diameter(self):
        return 1 + 2 * self.router_radius

    def cost_for(self, num_routers: int, num_cables: int):
        for_routers = num_routers * self.router_price
        for_cables = num_cables * self.cable_price

        return for_routers + for_cables

    def coverage_mask(self, coord: Coord):
        return self.grid.visible_targets_mask(coord, self.router_radius)

    def coverage_at(self, coord: Coord):
        yield from self.grid.visible_targets(coord, self.router_radius)
