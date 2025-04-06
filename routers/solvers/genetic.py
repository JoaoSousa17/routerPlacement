from typing import Collection, Dict, List, Literal, Sequence, Tuple, cast

from routers.run import ClustersResult, HeatmapResult, Progress, SolutionResult
from routers.problem import Mask, Problem
from routers.solution import Solution
from routers.solve import rewire_routers_mst, worst_routers
from routers.solvers.common import find_continuous_spaces
from routers.solvers.initial import fill_exhaustively, fill_grid
from routers.solvers.mutations import (
    random_add,
    random_jump,
    remove_random,
    remove_worst,
    shift_random,
    worst_jump,
)
from routers.spatial import ConnectionsMap, Coord, CoordCoverageList, CoordDSU, HashGrid

from itertools import combinations
import random


def generate_population(problem: Problem, size: int, progress: Progress):

    openness_th = 0.85
    openness, clusters = find_continuous_spaces(problem, openness_th, progress)

    yield HeatmapResult(openness, openness_th)
    yield ClustersResult(clusters)

    population: List[Solution] = []

    for _ in progress.range(size, "Generating starting population"):
        offset_x = random.randint(1, problem.router_radius)
        offset_y = random.randint(1, problem.router_radius)

        solution = Solution.Empty(problem)
        grid_offset = Coord(offset_x, offset_y)

        for solution in fill_grid(solution, progress.spawn(), clusters, grid_offset):
            yield SolutionResult(solution, False)

        for result in rewire_routers_mst(solution, progress.spawn()):
            solution = result.solution
            yield SolutionResult(solution, False)

        for result in fill_exhaustively(solution, progress.spawn()):
            solution = result.solution
            yield SolutionResult(solution, False)

        for result in rewire_routers_mst(solution, progress.spawn()):
            solution = result.solution
            yield SolutionResult(solution, False)

        progress.despawn()
        population.append(solution)

    return population


def mst(routers: Collection[Coord]):
    edges: List[Tuple[Coord, Coord, int]] = []

    for a, b in combinations(routers, 2):
        edges.append((a, b, Coord.chebyshev_distance(a, b)))

    edges.sort(key=lambda x: x[2])
    dsu = CoordDSU(routers)

    connections = ConnectionsMap()
    n_connections = 0

    for a, b, _ in edges:

        if dsu.find(a) != dsu.find(b):
            dsu.union(a, b)
            connections.mutable_connect(a, [b])
            n_connections += 1

            if n_connections == len(routers) - 1:
                break

    return connections


def cross_solutions(
    problem: Problem,
    solution_a: Solution,
    solution_b: Solution,
    at: int,
    axis: Literal[0, 1],
):

    child_a_routers = {}
    child_b_routers = {}

    child_a_scores = {}
    child_b_scores = {}

    child_a_outdated = set()
    child_b_outdated = set()

    for router, mask in solution_a.routers.items():
        if router[axis] < at:
            child_a_routers[router] = mask
            child_a_scores[router] = solution_a.router_scores[router]

            if router[axis] > (at - problem.router_diameter()):
                child_a_outdated.add(router)

        else:
            child_b_routers[router] = mask
            child_b_scores[router] = solution_a.router_scores[router]

            if router[axis] < (at + problem.router_diameter() - 1):
                child_b_outdated.add(router)

    for router, mask in solution_b.routers.items():
        if router[axis] < at:
            child_b_routers[router] = mask
            child_b_scores[router] = solution_b.router_scores[router]

            if router[axis] > (at - problem.router_diameter()):
                child_b_outdated.add(router)

        else:
            child_a_routers[router] = mask
            child_a_scores[router] = solution_b.router_scores[router]

            if router[axis] < (at + problem.router_diameter() - 1):
                child_a_outdated.add(router)

    for outdated in solution_a.outdated_scores:
        if outdated[axis] < at:
            child_a_outdated.add(outdated)
        else:
            child_b_outdated.add(outdated)

    for outdated in solution_b.outdated_scores:
        if outdated[axis] < at:
            child_b_outdated.add(outdated)
        else:
            child_a_outdated.add(outdated)

    def routers_to_coverage(routers: Dict[Coord, Mask]):
        for router, mask in routers.items():
            yield from problem.grid.mask_to_coords(mask, router)

    not_covered = frozenset(problem.grid.target_cells())

    child_a_hash_grid = HashGrid(problem.router_diameter()).add_point(*child_a_routers)
    child_a_coverage = CoordCoverageList(routers_to_coverage(child_a_routers))
    child_a_not_covered = not_covered.difference(child_a_coverage)

    child_b_hash_grid = HashGrid(problem.router_diameter()).add_point(*child_b_routers)
    child_b_coverage = CoordCoverageList(routers_to_coverage(child_b_routers))
    child_b_not_covered = not_covered.difference(child_b_coverage)

    child_a = Solution(
        problem,
        child_a_routers,
        child_a_hash_grid,
        mst([problem.backbone, *child_a_routers]),
        child_a_coverage,
        child_a_not_covered,
        frozenset(child_a_routers),
        child_a_scores,
        frozenset(child_a_outdated),
    )

    child_b = Solution(
        problem,
        child_b_routers,
        child_b_hash_grid,
        mst([problem.backbone, *child_b_routers]),
        child_b_coverage,
        child_b_not_covered,
        frozenset(child_b_routers),
        child_b_scores,
        frozenset(child_b_outdated),
    )

    return child_a, child_b


def tournament_selection(population: Sequence[Solution], k=3):
    assert 2 <= k <= len(population)

    selected = random.sample(population, k)
    selected.sort(key=lambda s: s.fitness(), reverse=True)  # higher fitness = better

    return selected[0], selected[1]


mutations = [
    # shift_random,
    remove_worst,
    remove_random,
    random_add,
    worst_jump,
    random_jump,
]


def genetic_algorithm(
    size: int,
    iters: int,
    tournament: int,
    cross_probab: float,
    mut_probab: float,
):

    def run(solution: Solution, progress: Progress):
        population = yield from generate_population(solution.problem, size, progress)

        for _ in progress.range(iters, "Running generations"):
            subprogress = progress.spawn("New generation...")
            new_population = []

            while len(new_population) < len(population):
                subprogress.n_out_of(len(new_population), len(population))
                parent_a, parent_b = tournament_selection(population, tournament)

                if random.random() < cross_probab:
                    axis = cast(Literal[0, 1], random.randint(0, 1))

                    if axis == 0:
                        at = random.randint(0, solution.problem.grid.width)
                    else:
                        at = random.randint(0, solution.problem.grid.height)

                    subprogress.spawn("Crossing...")
                    child_a, child_b = cross_solutions(
                        solution.problem,
                        parent_a,
                        parent_b,
                        at,
                        axis,
                    )

                    subprogress.despawn()

                else:
                    child_a, child_b = parent_a, parent_b

                if random.random() < mut_probab:
                    try_child_a = random.choice(mutations)(child_a)
                    child_a = try_child_a or child_a
                if random.random() < mut_probab:
                    try_child_b = random.choice(mutations)(child_b)
                    child_b = try_child_b or child_b

                subprogress.spawn("Pruning...")
                for worst_router in worst_routers(child_a):
                    if child_a.is_valid():
                        break

                    child_a = child_a.remove_router(worst_router)

                for worst_router in worst_routers(child_b):
                    if child_b.is_valid():
                        break

                    child_b = child_b.remove_router(worst_router)

                subprogress.despawn()
                new_population.extend([child_a, child_b])

            population = new_population[:size]
            yield SolutionResult(max(population, key=lambda s: s.fitness()), True)

        progress.despawn()

    return run
