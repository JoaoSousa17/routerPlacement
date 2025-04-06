from dataclasses import dataclass, replace
from typing import Collection, FrozenSet, TypeAlias, Dict, Tuple
from routers.spatial import ConnectionsMap, Coord, CoordCoverageList, HashGrid
from routers.problem import Mask, Problem

__all__ = ["Solution"]

Score: TypeAlias = Tuple[int, int]


@dataclass(frozen=True)
class Solution:

    problem: Problem

    routers: Dict[Coord, Mask]
    router_placements: HashGrid
    connections: ConnectionsMap

    cell_coverage: CoordCoverageList
    not_covered: FrozenSet[Coord]

    movable_routers: FrozenSet[Coord]
    router_scores: Dict[Coord, Score]
    outdated_scores: FrozenSet[Coord]
    # neighbouring_routers: ConnectionsMap

    @classmethod
    def Empty(cls, problem: Problem):
        return cls(
            problem=problem,
            routers=dict(),
            router_placements=HashGrid(problem.router_radius * 2 + 1),
            connections=ConnectionsMap(),
            cell_coverage=CoordCoverageList(),
            not_covered=frozenset(problem.grid.target_cells()),
            movable_routers=frozenset(),
            router_scores={},
            outdated_scores=frozenset(),
            # neighbouring_routers=ConnectionsMap(),
        )

    def percent_covered(self):
        return len(self.cell_coverage) / len(self.problem.grid.target_cells())

    def average_router_price(self):
        return self.cost() / len(self.routers)

    def average_router_coverage(self):
        return len(self.cell_coverage) / len(self.routers)

    def approx_can_afford_routers(self):
        return self.unspent_budget() / self.average_router_price()

    def unspent_budget(self):
        return self.problem.budget - self.cost()

    def fitness(self):
        from_coverage = 1000 * len(self.cell_coverage)
        from_unspent = self.problem.budget - self.cost()

        return from_coverage + from_unspent

    def cost(self):
        total_cables = self.connections.total_lens()
        total_cables += len(self.routers)

        # Router placed on top of the backbone shouldn't require a cable
        if self.problem.backbone in self.routers:
            total_cables -= 1

        return self.problem.cost_for(len(self.routers), total_cables)

    def is_valid(self):
        return self.cost() < self.problem.budget

    def update_router_score(self, router: Coord):
        updated_scores = self.router_scores.copy()
        updated_scores[router] = self.eval_router(router)
        updated_outdated = self.outdated_scores - {router}

        return replace(
            self,
            router_scores=updated_scores,
            outdated_scores=updated_outdated,
        )

    def eval_router(self, router: Coord):
        coverage_score = 0
        total_coverage = 0

        for covered in self._router_coverage(router):
            total_coverage += 1

            if self.cell_coverage[covered] == 1:
                coverage_score += 1

        return coverage_score, total_coverage

    def close_routers(self, router: Coord):
        return self.router_placements.within_radius(
            router, self.problem.router_radius * 2
        )

    def _router_coverage(self, router: Coord):
        return self.problem.grid.mask_to_coords(self.routers[router], router)

    def add_router(self, router: Coord):
        if router in self.routers:
            RuntimeError(f"Router {router} already exists")

        mask = self.problem.coverage_mask(router)
        return self.add_router_with_mask(router, mask)

    def add_router_with_mask(self, router: Coord, mask: Mask):
        if router in self.routers:
            RuntimeError(f"Router {router} already exists")

        covered_coords = list(self.problem.grid.mask_to_coords(mask, router))
        updated_coverage, newly_covered_cells = self.cell_coverage.add(covered_coords)

        updated_not_covered_cells = self.not_covered - newly_covered_cells
        updated_placements = self.router_placements.add_point(router)

        connects_to = self.router_placements.closest(router, self.problem.backbone)
        updated_connections = self.connections.connect(router, [connects_to])

        updated_scores = self.router_scores.copy()
        updated_scores[router] = len(newly_covered_cells), len(covered_coords)

        close_routers = self.close_routers(router)

        updated_movable = self.movable_routers.union(close_routers) | {router}
        updated_outdated_scores = self.outdated_scores.union(close_routers)
        # updated_neighbours = self.neighbouring_routers.connect(router, close_routers)

        new_routers = self.routers.copy()
        new_routers[router] = mask

        return Solution(
            problem=self.problem,
            routers=new_routers,
            router_placements=updated_placements,
            connections=updated_connections,
            cell_coverage=updated_coverage,
            not_covered=updated_not_covered_cells,
            movable_routers=updated_movable,
            router_scores=updated_scores,
            outdated_scores=updated_outdated_scores,
            # neighbouring_routers=updated_neighbours,
        )

    def remove_router(self, router: Coord):
        if router not in self.routers:
            RuntimeError(f"Router {router} does not exist")

        coverage = self._router_coverage(router)
        new_coverage, now_not_covered = self.cell_coverage.remove(coverage)

        new_not_covered = self.not_covered | now_not_covered
        new_connect, affected = self.connections.remove_point(router)
        new_placed = self.router_placements.remove_point(router)

        self._rewire_mutably(new_connect, affected)

        new_scores = self.router_scores.copy()
        del new_scores[router]

        # new_close, affected = self.neighbouring_routers.remove_point(router)
        new_eval = self.outdated_scores.union(self.close_routers(router)) - {
            router
        }

        new_routers = self.routers.copy()
        del new_routers[router]

        return Solution(
            problem=self.problem,
            routers=new_routers,
            router_placements=new_placed,
            connections=new_connect,
            cell_coverage=new_coverage,
            not_covered=new_not_covered,
            movable_routers=self.movable_routers - {router},
            router_scores=new_scores,
            outdated_scores=new_eval,
            # neighbouring_routers=new_close,
        )

    def jump(self, router: Coord, dest: Coord):
        if router not in self.routers:
            raise RuntimeError(f"Cannot jump router {router} that does not exist")

        temp_solution = self.remove_router(router)
        return temp_solution.add_router(dest)

    def shift_router_to(self, router: Coord, dest: Coord):
        if router not in self.routers:
            raise RuntimeError(f"Cannot shift router {router} that does not exist")

        if dest in self.routers:
            raise RuntimeError(f"Cannot shift router to an occupied spot {dest}")

        coverage_before = self._router_coverage(router)

        updated_coverage, newly_not_covered = self.cell_coverage.remove(coverage_before)
        updated_not_covered = self.not_covered | newly_not_covered

        mask = self.problem.coverage_mask(dest)
        coverage_after = list(self.problem.grid.mask_to_coords(mask, dest))
        updated_coverage, newly_covered = updated_coverage.add(coverage_after)
        updated_not_covered = updated_not_covered - newly_covered

        updated_scores = self.router_scores.copy()
        updated_scores[dest] = len(newly_covered), len(coverage_after)
        del updated_scores[router]

        # updated_neighs, oudated_scores = self.neighbouring_routers.remove_point(router)
        updated_outdated_scores = self.outdated_scores.union(
            self.close_routers(router)
        ) - {router}

        updated_connections, was_connected_to = self.connections.remove_point(router)
        updated_connections.mutable_connect(dest, was_connected_to)

        updated_placements = self.router_placements.remove_point(router)
        routers_close_to_dest = updated_placements.within_radius(
            dest, self.problem.router_radius * 2
        )

        updated_outdated_scores = updated_outdated_scores.union(routers_close_to_dest)
        updated_placements = updated_placements.add_point(dest)
        new_movable = self.movable_routers.union(routers_close_to_dest)
        new_movable -= {router}
        new_movable |= {dest}

        new_routers = self.routers.copy()
        new_routers[dest] = mask
        del new_routers[router]

        return Solution(
            problem=self.problem,
            routers=new_routers,
            router_placements=updated_placements,
            connections=updated_connections,
            cell_coverage=updated_coverage,
            not_covered=updated_not_covered,
            movable_routers=new_movable,
            router_scores=updated_scores,
            outdated_scores=updated_outdated_scores,
            # neighbouring_routers=updated_neighs,
        )

    def _rewire_mutably(self, connects: ConnectionsMap, coords: Collection[Coord]):

        if not len(coords):
            print("Cannot rewire 0 disconnection points")
            return

        elif len(coords) == 1:
            return

        elif len(coords) == 2:
            a, b = coords
            connects.mutable_connect(a, [b])
            return

        to_update = set(coords)
        curr = self.problem.backbone.closest_chebyshev(to_update)
        to_update.remove(curr)

        while to_update:
            next = curr.closest_chebyshev(to_update)
            connects.mutable_connect(curr, [next])
            to_update.remove(next)
            curr = next

    def place_cables(self):
        cables = set(self.routers) - {self.problem.backbone}

        for a, b in self.connections:
            cables.update(Coord.line_between(a, b))

        return cables
