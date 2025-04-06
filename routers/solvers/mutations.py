from routers.solution import Solution

from routers.solvers.common import (
    pick_random_shift,
    pick_worst_router,
    pick_random_router,
    pick_random_spot,
    pick_random_movable,
)


def shift_random(solution: Solution):
    shifted_router = pick_random_movable(solution)

    if shifted_router:
        dest = pick_random_shift(solution, shifted_router, 4)

        if dest:
            return solution.shift_router_to(shifted_router, dest)


def remove_worst(solution: Solution):
    remove_from = pick_worst_router(solution)

    if remove_from:
        return solution.remove_router(remove_from)


def remove_random(solution: Solution):
    remove_from = pick_random_router(solution)

    if remove_from:
        return solution.remove_router(remove_from)


def random_add(solution: Solution):
    spot = pick_random_spot(solution)

    if spot:
        return solution.add_router(spot)


def worst_jump(solution: Solution):
    remove_from = pick_worst_router(solution)
    place_to = pick_random_spot(solution)

    if remove_from and place_to:
        return solution.jump(remove_from, place_to)


def random_jump(solution: Solution):
    remove_from = pick_random_router(solution)
    place_to = pick_random_spot(solution)

    if remove_from and place_to:
        return solution.jump(remove_from, place_to)
