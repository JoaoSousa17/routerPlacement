from numpy.typing import NDArray
from typing import Collection, List, Set, cast, Tuple, Iterable

from routers.run import Progress
from routers.solution import Solution
from routers.spatial import Coord
from routers.problem import Problem, CellKind, Grid
from routers.solve import worst_routers

from scipy.ndimage import label
from scipy.signal import convolve2d

import numpy as np
import random


__all__ = [
    "find_clusters",
    "openness_heatmap",
    "coords_to_matrix",
    "empty_clusters",
]


def coords_to_matrix(coords: Iterable[Coord], grid: Grid) -> NDArray[np.bool]:
    mat = np.zeros((grid.width, grid.height), dtype=np.bool)

    if coords:
        xs, ys = zip(*coords)
        mat[xs, ys] = 1

    return mat


def find_clusters(mask: NDArray[np.bool], progress: Progress) -> List[Set[Coord]]:

    labeled_clusters, num_clusters = cast(Tuple[NDArray, int], label(mask))
    clusters = []

    for cluster_id in progress.range(num_clusters, "Collecting clustered points"):
        cluster_coords = np.argwhere(labeled_clusters == (cluster_id + 1))
        clusters.append({Coord(x, y) for x, y in cluster_coords})

    return clusters


def openness_heatmap(problem: Problem, progress: Progress) -> NDArray[np.float64]:
    progress.only_title("Calculating openness")
    openness_matrix = (problem.grid.matrix() == CellKind.Target).astype(np.uint32)

    kernel_size = problem.router_diameter()
    kernel_openness = np.ones((kernel_size, kernel_size), dtype=np.int32)

    openness_map = cast(
        NDArray[np.uint32],
        convolve2d(
            openness_matrix,
            kernel_openness,
            mode="same",
            boundary="fill",
            fillvalue=0,
        ),
    )

    normalized = openness_map / (kernel_size**2)
    return normalized


def empty_clusters(solution: Solution, progress: Progress) -> List[Set[Coord]]:

    progress.only_title("Finding empty clusters")
    empty = coords_to_matrix(solution.not_covered, solution.problem.grid)
    clusters = find_clusters(empty, progress)

    return clusters


def find_continuous_spaces(problem: Problem, th: float, progress: Progress):
    progress.only_title("Identifying continuous areas")
    subprogress = progress.spawn()
    openness = openness_heatmap(problem, subprogress)

    mask = (openness >= th).astype(bool)
    clusters = find_clusters(mask, subprogress)
    subprogress.despawn()

    return openness, clusters


def gridify(problem: Problem, cluster: Collection[Coord], offset: Coord = Coord(0, 0)):
    if not len(cluster):
        return []

    as_array = np.asarray([[x, y] for x, y in cluster])
    min_x, min_y = as_array.min(axis=0)
    max_x, max_y = as_array.max(axis=0)

    spots = []

    ox = offset.x % problem.router_radius
    oy = offset.y % problem.router_radius

    for x in range(min_x + ox, max_x, problem.router_diameter()):
        for y in range(min_y + oy, max_y, problem.router_diameter()):
            coord = Coord(x, y)

            if coord in cluster and problem.grid[coord] != CellKind.Wall:
                spots.append(coord)

    return spots


def pick_worst_router(solution: Solution):
    if not solution.routers:
        return None

    return next(iter(worst_routers(solution)))


def pick_random_router(solution: Solution):
    if not solution.routers:
        return None

    return random.choice(list(solution.routers))


def pick_random_movable(solution: Solution):
    if not solution.movable_routers:
        return None

    return random.choice(list(solution.movable_routers))


def pick_random_spot(solution: Solution):
    candidates = solution.not_covered or solution.problem.grid.target_cells()
    candidates -= set(solution.routers)

    if not candidates:
        return None

    return random.choice(list(candidates))


def pick_random_shift(
    solution: Solution,
    origin: Coord,
    radius: int = 0,
):
    if not radius:
        radius = solution.problem.router_radius

    candidates = set(solution.problem.grid.non_walls_within_radius(origin, radius))
    candidates -= set(solution.routers)

    if not candidates:
        return None

    return random.choice(list(candidates))
