from typing import List, Set

from routers.solvers.common import find_continuous_spaces, gridify
from routers.run import Progress, SolutionResult, HeatmapResult, ClustersResult
from routers.solution import Solution
from routers.spatial import Coord

import random


def fill_continuous_spaces(solution: Solution, progress: Progress):
    continuous_th = 0.85
    openness, clusters = find_continuous_spaces(
        solution.problem,
        continuous_th,
        progress,
    )

    yield HeatmapResult(openness, continuous_th)
    yield ClustersResult(clusters)

    for solution in fill_grid(solution, progress, clusters):
        yield SolutionResult(solution, True)


def fill_grid(
    solution: Solution,
    progress: Progress,
    clusters: List[Set[Coord]],
    offset: Coord = Coord(0, 0),
):
    subprogress = progress.spawn()

    for cluster in progress.iter_collection(clusters, "Placing grids"):
        for spot in subprogress.iter_collection(
            gridify(
                solution.problem,
                cluster,
                offset,
            )
        ):
            if spot not in solution.routers:
                new_solution = solution.add_router(spot)

                if not new_solution.is_valid():
                    subprogress.finalize()
                    progress.finalize()
                    progress.despawn()
                    return

                solution = new_solution
                yield solution


def fill_exhaustively(solution: Solution, progress: Progress):

    est_routers = int(solution.unspent_budget() / solution.problem.router_price)

    if not est_routers:
        return

    filled = set()
    spots = list(solution.not_covered)
    random.shuffle(spots)

    for spot in progress.iter_collection(spots, "Filling empty spots"):
        if spot in filled:
            continue

        filled_mask = solution.problem.grid.visible_targets_mask(
            spot,
            int(solution.problem.router_radius * 1.5),
        )

        submask_start = (filled_mask.shape[0] // 2) - solution.problem.router_radius
        submask_end = submask_start + solution.problem.router_diameter()
        router_submask = filled_mask[
            submask_start:submask_end, submask_start:submask_end
        ]

        with_router = solution.add_router_with_mask(spot, router_submask)

        if not with_router.is_valid():
            break

        filled.update(solution.problem.grid.mask_to_coords(filled_mask, spot))
        solution = with_router

        yield SolutionResult(solution, True)

    progress.finalize()
