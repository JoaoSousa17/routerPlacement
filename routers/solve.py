from collections.abc import Iterable
from pygame import math
from routers.spatial import ConnectionsMap, Coord, CoordDSU
from dataclasses import replace
from routers.solution import Solution
from routers.run import  Progress,  SolutionResult
from typing import Callable, List, Tuple, TypeAlias 
from itertools import combinations

import numpy as np
import math
import heapq
import random


def evaluate_routers(solution: Solution, progress: Progress | None = None):
    iterator = (
        progress.iter_collection(
            solution.outdated_scores,
            "Evaluating routers",
        )
        if progress
        else solution.outdated_scores
    )

    for router in iterator:
        solution = solution.update_router_score(router)

    return solution


def worst_routers(solution: Solution, progress: Progress | None = None):
    solution = evaluate_routers(solution, progress)
    from_worst = [(sc, r) for r, sc in solution.router_scores.items()]
    heapq.heapify(from_worst)

    def iterate(solution: Solution):
        while from_worst:
            score, suspect = heapq.heappop(from_worst)

            if suspect in solution.outdated_scores:
                solution = solution.update_router_score(suspect)
                new_score = solution.router_scores[suspect]

                if new_score != score:
                    heapq.heappush(from_worst, (new_score, suspect))
                    continue

            yield suspect

            solution = solution.remove_router(suspect)

    return iterate(solution)


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


def pick_random_spot_best_of(solution: Solution, best_of: int = 10):
    candidates = solution.not_covered or solution.problem.grid.target_cells()
    candidates -= set(solution.routers)

    if not candidates:
        return None

    spots = random.sample(list(candidates), min(len(candidates), best_of))
    best_spot = None
    best_fitness = 0

    for spot in spots:
        attempt = solution.add_router(spot)

        if attempt.fitness() > best_fitness:
            best_spot = spot
            best_fitness = attempt.fitness()

    return best_spot


def pick_random_shift(
    solution: Solution,
    origin: Coord,
    radius: int = 0,
    best_of: int = 10,
):
    if not radius:
        radius = solution.problem.router_radius

    candidates = set(solution.problem.grid.non_walls_within_radius(origin, radius))
    candidates -= set(solution.routers)

    if not candidates:
        return None

    spots = random.sample(list(candidates), min(len(candidates), best_of))
    best_spot = None
    best_fitness = 0

    for spot in spots:
        attempt = solution.add_router(spot)

        if attempt.fitness() > best_fitness:
            best_spot = spot
            best_fitness = attempt.fitness()

    return best_spot


NeighbourFunc: TypeAlias = Callable[[Solution], Solution | None]


def neigh_func_expensive(solution: Solution):

    wanna_place = len(solution.not_covered) / solution.average_router_coverage()
    can_afford = solution.approx_can_afford_routers()

    if can_afford < 1.0 or wanna_place <= 0.001:
        add_ratio = 0
    else:
        add_ratio = min(can_afford / wanna_place, 1)

    add_chance = int(20 * add_ratio)

    REMOVE_TH = 0.96

    if solution.percent_covered() < REMOVE_TH:
        remove_ratio = 0
    else:
        remove_ratio = (solution.percent_covered() - REMOVE_TH) / (1 - REMOVE_TH)

    remove_chance = int(5 * remove_ratio)
    remaining_chances = 100 - add_chance - remove_chance

    jump_chance = int(0.5 * remaining_chances)
    shift_chance = int(0.4 * remaining_chances)
    smart_shift_chance = remaining_chances - jump_chance - shift_chance

    options = ["add", "remove", "jump", "shift", "smart_shift"]
    weights = [add_chance, remove_chance, jump_chance, shift_chance, smart_shift_chance]

    selected = random.choices(options, weights=weights)[0]
    new_solution = None

    if selected == "add":
        spot = pick_random_spot_best_of(solution, 10)

        if spot:
            new_solution = solution.add_router(spot)

    elif selected == "remove":
        remove_from = pick_worst_router(solution)

        if remove_from:
            new_solution = solution.remove_router(remove_from)

    elif selected == "jump":
        remove_from = pick_worst_router(solution)
        place_to = pick_random_spot(solution)

        if remove_from and place_to:
            new_solution = solution.jump(remove_from, place_to)

    elif selected == "shift":
        shifted_router = pick_random_movable(solution)

        if shifted_router:
            dest = pick_random_shift(solution, shifted_router, best_of=10)

            if dest:
                new_solution = solution.shift_router_to(shifted_router, dest)

    elif selected == "smart_shift":
        shifted_router = pick_random_movable(solution)

        if shifted_router:
            smart_shift(solution, shifted_router)

    else:
        raise RuntimeError(f"Unknown operation {selected}")

    if new_solution and new_solution.is_valid():
        return new_solution
    else:
        return None


def neigh_func_cheap(solution: Solution):

    options = ["add", "remove", "jump", "shift"]
    weights = [5, 1, 36, 48]
    # weights = [5, 1, 94, 0]
    selected = random.choices(options, weights=weights)[0]
    new_solution = None

    if selected == "add":
        spot = pick_random_spot_best_of(solution, 3)

        if spot:
            new_solution = solution.add_router(spot)

    elif selected == "remove":
        remove_from = pick_worst_router(solution)

        if remove_from:
            new_solution = solution.remove_router(remove_from)

    elif selected == "jump":
        remove_from = pick_random_router(solution)
        place_to = pick_random_spot(solution)

        if remove_from and place_to:
            new_solution = solution.jump(remove_from, place_to)

    elif selected == "shift":
        shifted_router = pick_random_movable(solution)

        if shifted_router:
            dest = pick_random_shift(solution, shifted_router, 4, 1)

            if dest:
                new_solution = solution.shift_router_to(shifted_router, dest)

    else:
        raise RuntimeError(f"Unknown operation {selected}")

    if new_solution and new_solution.is_valid():
        return new_solution
    else:
        return None


def simulated_annealing(iterations: int, rewire_every: int, neigh_func: NeighbourFunc):

    def run(solution: Solution, progress: Progress):
        temperature = 1000

        score = solution.fitness()
        best_solution = solution
        best_score = score

        for n_iter in progress.range(iterations, "Simulated annealing"):

            if n_iter % rewire_every == 0:
                for result in rewire_routers_mst(solution, progress):
                    if result.is_best:
                        solution = result.solution

                    yield result

            temperature = temperature * 0.999
            next_solution = neigh_func(solution)

            if next_solution is None:
                continue

            next_score = next_solution.fitness()

            if next_score > score:
                solution = next_solution
                score = next_score

            else:
                probab = math.exp((next_score - score) / temperature)

                if random.random() < probab:
                    solution = next_solution
                    score = next_score

            if score > best_score:
                best_solution = solution
                best_score = score
                yield SolutionResult(solution, True)

            else:
                yield SolutionResult(solution, False)

        yield SolutionResult(best_solution, True)

    return run


def threshold_percent(percent: float):

    def th(solution: Solution, _):
        return solution.percent_covered() < percent

    return th


def threshold_useless(min_score: int):

    def th(solution: Solution, router: Coord):
        cs, _ = solution.eval_router(router)
        return cs > min_score

    return th


def prune_and_shuffle(threshold_func: Callable[[Solution, Coord], bool]):

    def run(solution: Solution, progress: Progress):
        progress.title = "Pruning"
        progress.percentage = -1

        worst_router = pick_worst_router(solution)

        while worst_router and not threshold_func(solution, worst_router):
            neights = solution.close_routers(worst_router)
            solution = solution.remove_router(worst_router)

            for router in neights:
                solution = smart_shift(solution, router)
                yield SolutionResult(solution, True)

            worst_router = pick_worst_router(solution)

    return run


def prune_worst_routers(threshold_func: Callable[[Solution, Coord], bool]):

    def run(solution: Solution, progress: Progress):
        progress.title = "Pruning"
        progress.percentage = -1

        for worst_router in worst_routers(solution, progress):
            if threshold_func(solution, worst_router):
                break

            solution = solution.remove_router(worst_router)
            yield SolutionResult(solution, True)

    return run


def smart_shift(solution: Solution, router: Coord, radius: int = 0):
    if radius == 0:
        radius = solution.problem.router_radius
    else:
        radius = radius

    movable_to = list(
        set(solution.problem.grid.non_walls_within_radius(router, radius))
        - set(solution.routers)
        - {router}
    )

    random.shuffle(movable_to)
    same_quality: Tuple[Coord, Solution] | None = None

    for cell in movable_to:
        with_shifts = solution.shift_router_to(router, cell)

        if not with_shifts.is_valid():
            continue

        if with_shifts.fitness() > solution.fitness():
            solution = with_shifts
            return solution

        elif with_shifts.fitness() == solution.fitness():
            same_quality = cell, with_shifts

        if same_quality is not None:
            cell, solution = same_quality

            return solution

    return replace(
        solution,
        movable_routers=solution.movable_routers - {router},
    )

    # Possible modification for the smart_shift function


# def smart_shift(solution: Solution, router: Coord, radius: int = 0):
# 1. **Radius Assignment**:
# If radius is 0, it is assigned the default value from the problem's router_radius.
# If radius is not 0, it remains the provided value.
# if radius == 0:
# radius = solution.problem.router_radius

# 2. **Movable Cells**:
# Get a list of valid cells within the radius that are not routers or the current router.
# movable_to = list(
# set(solution.problem.grid.non_walls_within_radius(router, radius))
# - solution.routers
# - {router}
# )

# Shuffle the cells to introduce randomness in the movement decision.
# random.shuffle(movable_to)

# same_quality: Tuple[Coord, Solution] | None = None

# 3. **Fitness Calculation**:
# Avoid recalculating fitness multiple times by storing it at the start.
# current_fitness = solution.fitness()  # Avoids repeated calls to fitness()

# 4. **Loop Through Movable Cells**:
# Check if moving the router to any of the possible cells leads to a better solution.
# for cell in movable_to:
# with_shifts = solution.shift_router_to(router, cell)

# Skip invalid solutions.
# if not with_shifts.is_valid():
# continue

# 5. **Fitness Comparison**:
# Compare the fitness of the new solution with the current one.
# shifted_fitness = with_shifts.fitness()

# if shifted_fitness > current_fitness:
# If the shifted solution has better fitness, return it.
# return with_shifts

# 6. **Handling Same Quality Solutions**:
# If the fitness is equal to the current fitness and same_quality hasn't been set yet, store it.
# elif shifted_fitness == current_fitness and same_quality is None:
# same_quality = (cell, with_shifts)

# 7. **Return Best Same Quality Solution**:
# If a solution with the same fitness was found, return the best one.
# if same_quality is not None:
# _, best = same_quality
# return best

# 8. **Return Solution with Router Replaced**:
# If no better solution was found, return the solution after replacing the router.
# return replace(
# solution,
# movable_routers=solution.movable_routers - {router},
# )


def smart_shuffle(times: int, radius: int = 0):

    def run(solution: Solution, progress: Progress):

        for _ in progress.range(times, "Shifting routers"):
            if not solution.movable_routers:
                progress.finalize()
                break

            router = random.choice(list(solution.movable_routers))
            solution = smart_shift(solution, router, radius)

            yield SolutionResult(solution, True)

    return run


def rewire_routers_mst(initial: Solution, progress: Progress):
    edges: List[Tuple[Coord, Coord, int]] = []
    points = {*initial.routers, initial.problem.backbone}

    for a, b in progress.iter_est(
        combinations(points, 2),
        math.comb(len(points), 2),
        "Considering all edges",
    ):
        edges.append((a, b, Coord.chebyshev_distance(a, b)))

    solution = replace(initial, connections=ConnectionsMap())
    yield SolutionResult(solution, False)

    edges.sort(key=lambda x: x[2])
    dsu = CoordDSU(points)
    connections = 0

    for a, b, _ in progress.iter_collection(edges, "Wiring routers"):

        if dsu.find(a) != dsu.find(b):
            dsu.union(a, b)
            connections += 1

            new_connect = solution.connections.connect(a, [b])
            solution = replace(solution, connections=new_connect)

            if not solution.is_valid():
                yield SolutionResult(initial, True)
                return

            yield SolutionResult(solution, False)

            if connections == len(points) - 1:
                break

    if solution.fitness() > initial.fitness():
        yield SolutionResult(solution, True)
    else:
        yield SolutionResult(initial, True)


# def fill_empty(solution: Solution, progress: Progress):
#     spots, cl = identify_empty_spots(solution)
#     yield cl
#
#     for spot in progress.iter_collection(spots, "Filling empty spots"):
#         new_solution = solution.add_router(spot)
#
#         if new_solution.is_valid():
#             solution = new_solution
#             yield SolutionResult(solution, True)


# def identify_empty_spots(solution: Solution):
#
#     empty = coords_to_matrix(solution.not_covered, solution.problem.grid)
#     labeled_clusters, num_clusters = cast(Tuple[NDArray, int], label(empty))
#
#     spots = []
#
#     for cluster_id in range(1, num_clusters + 1):
#         cluster_coords = np.argwhere(labeled_clusters == cluster_id)
#         spots.append(find_medoid([Coord(x, y) for x, y in cluster_coords]))
#
#     return spots, ClustersResult(labeled_clusters, num_clusters)


def find_medoid(points: Iterable[Coord]):
    points = np.array(points)
    best_index = None
    best_total_distance = float("inf")

    for i, candidate in enumerate(points):
        distances = np.linalg.norm(points - candidate, axis=1)
        total_distance = np.sum(distances)

        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_index = i

    return Coord(*points[best_index])
