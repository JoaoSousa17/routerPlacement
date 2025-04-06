from typing import (
    Collection,
    Dict,
    FrozenSet,
    Iterable,
    NamedTuple,
    List,
    Set,
    Tuple,
    TypeAlias,
)

import heapq


__all__ = [
    "Coord",
    "CoordDSU",
    "CoordCoverageList",
    "HashGrid",
]


class Coord(NamedTuple):

    x: int
    y: int

    @classmethod
    def iter_rect(cls, a: "Coord", b: "Coord"):
        tl, br = cls.topleft_bottomright(a, b)

        for y in range(tl.y, br.y + 1):
            for x in range(tl.x, br.x + 1):
                yield cls(x, y)

    @staticmethod
    def chebyshev_distance(a: "Coord", b: "Coord") -> int:
        return max(abs(a.x - b.x), abs(a.y - b.y))

    @staticmethod
    def topleft_bottomright(a: "Coord", b: "Coord") -> Tuple["Coord", "Coord"]:
        top_left = Coord(
            min(a.x, b.x),
            min(a.y, b.y),
        )

        bottom_right = Coord(
            max(a.x, b.x),
            max(a.y, b.y),
        )

        return top_left, bottom_right

    @staticmethod
    def line_between(start: "Coord", end: "Coord") -> List["Coord"]:

        dx = abs(end.x - start.x)
        dy = abs(end.y - start.y)

        step_x = 1 if start.x < end.x else -1
        step_y = 1 if start.y < end.y else -1

        points: List[Coord] = []
        error = dx - dy

        curr_x = start.x
        curr_y = start.y

        while True:
            error_twice = 2 * error

            if error_twice > -dy:
                error -= dy
                curr_x += step_x

            if error_twice < dx:
                error += dx
                curr_y += step_y

            if curr_x == end.x and curr_y == end.y:
                break

            points.append(Coord(curr_x, curr_y))

        return points

    def sort_by_chebyshev(self, others: Iterable["Coord"]) -> List[Tuple[int, "Coord"]]:
        with_distances = [(self.chebyshev_to(coord), coord) for coord in others]
        return sorted(with_distances, key=lambda dist_cood: dist_cood[0])

    def chebyshev_to(self, other) -> int:
        return self.chebyshev_distance(self, other)

    def closest_chebyshev(self, others: Iterable["Coord"]):
        return min(others, key=self.chebyshev_to)

    def up(self, by: int = 1) -> "Coord":
        return Coord(self.x, self.y - by)

    def left(self, by: int = 1) -> "Coord":
        return Coord(self.x - by, self.y)

    def upleft(self, by: int = 1) -> "Coord":
        return Coord(self.x - by, self.y - by)

    def right(self, by: int = 1) -> "Coord":
        return Coord(self.x + by, self.y)

    def upright(self, by: int = 1) -> "Coord":
        return Coord(self.x + by, self.y - by)

    def down(self, by: int = 1) -> "Coord":
        return Coord(self.x, self.y + by)

    def downright(self, by: int = 1) -> "Coord":
        return Coord(self.x + by, self.y + by)

    def downleft(self, by: int = 1) -> "Coord":
        return Coord(self.x - by, self.y + by)

    def neighbours(self) -> Iterable["Coord"]:
        yield self.upleft()
        yield self.up()
        yield self.upright()
        yield self.right()
        yield self.downright()
        yield self.down()
        yield self.downright()
        yield self.right()

    def __repr__(self) -> str:
        return f"<x: {self.x}, y: {self.y}>"


class CoordDSU:
    def __init__(self, vertices: Collection[Coord]):
        self.parent = {vert: vert for vert in vertices}
        self.rank = {vert: 1 for vert in vertices}

    def find(self, vert: Coord):
        if self.parent[vert] != vert:
            self.parent[vert] = self.find(self.parent[vert])

        return self.parent[vert]

    def union(self, a: Coord, b: Coord):
        s1 = self.find(a)
        s2 = self.find(b)

        if s1 != s2:
            if self.rank[s1] < self.rank[s2]:
                self.parent[s1] = s2
            elif self.rank[s1] > self.rank[s2]:
                self.parent[s2] = s1
            else:
                self.parent[s2] = s1
                self.rank[s1] += 1


class CoordCoverageList:

    _data: Dict[Coord, int]

    def __init__(
        self, initials: Iterable[Coord] | Dict[Coord, int] | None = None
    ) -> None:
        if initials is None:
            self._data = {}
        elif isinstance(initials, dict):
            self._data = initials.copy()
        else:
            self._data = {}

            for coord in initials:
                self._data[coord] = self._data.get(coord, 0) + 1

    def __contains__(self, coord: object):
        return coord in self._data

    def __getitem__(self, coord: Coord):
        return self._data[coord]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def add(self, coords: Iterable[Coord]):
        new_list = CoordCoverageList(self._data)
        new_entries: Set[Coord] = set()

        for coord in coords:
            if coord not in new_list:
                new_list._data[coord] = 1
                new_entries.add(coord)

            else:
                new_list._data[coord] += 1

        return new_list, new_entries

    def remove(self, coords: Iterable[Coord]):
        new_list = CoordCoverageList(self._data)
        removed_entries: Set[Coord] = set()

        for coord in coords:
            if coord not in new_list:
                raise RuntimeError(f"Cell {coord} was not covered")

            new_list._data[coord] -= 1

            if new_list._data[coord] <= 0:
                removed_entries.add(coord)
                del new_list._data[coord]

        return new_list, removed_entries


class ConnectionsMap:

    _total_lens: int
    _connections: Dict[Coord, Tuple[Tuple[Coord, int], ...]]

    def __init__(self):
        self._connections = {}
        self._total_lens = 0

    def __getitem__(self, point: Coord):
        return self._connections.get(point, tuple())

    def __iter__(self):
        seen = set()

        for a, connections in self._connections.items():
            for b, _ in connections:
                edge = frozenset({a, b})

                if edge not in seen:
                    seen.add(edge)
                    yield (a, b)

    def is_connected(self, a: Coord, b: Coord):
        if a not in self._connections:
            return False

        for connected, _ in self._connections[a]:
            if connected == b:
                return True

        return False

    def total_lens(self):
        return self._total_lens

    def mutable_connect(self, a: Coord, to: Iterable[Coord]):
        for coord in to:
            dist_between = Coord.chebyshev_distance(a, coord) - 1

            self._total_lens += dist_between
            self._remember_directed_connection(a, coord, dist_between)
            self._remember_directed_connection(coord, a, dist_between)

    def connect(self, a: Coord, to: Iterable[Coord]):
        new = type(self)()
        new._connections = self._connections.copy()
        new._total_lens = self._total_lens
        new.mutable_connect(a, to)

        return new

    def remove_point(self, point: Coord):
        new = type(self)()
        new._connections = self._connections.copy()
        new._total_lens = self._total_lens
        affected: List[Coord] = []

        if point not in new._connections:
            return new, affected

        for to, dist in new._connections[point]:
            without_point = (cd for cd in new._connections[to] if cd[0] != point)
            affected.append(to)

            new._connections[to] = tuple(without_point)
            new._total_lens -= dist

        del new._connections[point]
        return new, affected

    def _remember_directed_connection(self, a: Coord, b: Coord, dist: int):
        if a in self._connections:
            self._connections[a] += ((b, dist),)
        else:
            self._connections[a] = ((b, dist),)


Buckets: TypeAlias = Dict[Coord, FrozenSet[Coord]]


class HashGrid:

    _bucket_size: int
    _buckets: Buckets

    def __init__(self, bucket_size: int, buckets: Buckets | None = None):
        self._bucket_size = bucket_size
        self._buckets = buckets or {}

    def _bucket_key(self, coord: Coord) -> Coord:
        return Coord(
            coord.x // self._bucket_size,
            coord.y // self._bucket_size,
        )

    def add_point(self, *points: Coord) -> "HashGrid":
        new = type(self)(self._bucket_size, self._buckets.copy())

        for point in points:
            key = self._bucket_key(point)
            new._buckets[key] = new._buckets.get(key, frozenset()) | {point}

        return new

    def remove_point(self, point: Coord) -> "HashGrid":
        key = self._bucket_key(point)

        new = type(self)(self._bucket_size, self._buckets.copy())
        new_bucket = new._buckets[key] - {point}

        if new_bucket:
            new._buckets[key] = new_bucket
        else:
            del new._buckets[key]

        return new

    def within_radius(self, point: Coord, radius: int) -> List[Coord]:
        tl = self._bucket_key(point.upleft(radius))
        br = self._bucket_key(point.downright(radius))

        results: List[Coord] = []

        for key in Coord.iter_rect(tl, br):
            if key not in self._buckets:
                continue

            for candidate in self._buckets[key]:
                if point.chebyshev_to(candidate) <= radius:
                    results.append(candidate)

        return results

    def _bucket_bounds(self, bucket: Coord) -> Tuple[Coord, Coord]:
        min_x = bucket.x * self._bucket_size
        max_x = min_x + self._bucket_size - 1
        min_y = bucket.y * self._bucket_size
        max_y = min_y + self._bucket_size - 1

        return Coord(min_x, min_y), Coord(max_x, max_y)

    def _min_dist_to_bucket(self, point: Coord, bucket: Coord) -> int:
        qx, qy = point
        (min_x, min_y), (max_x, max_y) = self._bucket_bounds(bucket)

        dx = 0 if min_x <= point.x <= max_x else min(abs(qx - min_x), abs(qx - max_x))
        dy = 0 if min_y <= point.y <= max_y else min(abs(qy - min_y), abs(qy - max_y))

        return max(dx, dy)

    def closest(self, point: Coord, best: Coord | None = None) -> Coord:
        qx, qy = point

        best_distance = point.chebyshev_to(best) if best else 0
        best_point: Coord | None = best

        pq: List[Tuple[int, Coord]] = []

        start_bucket = self._bucket_key(point)
        start_dist = self._min_dist_to_bucket(point, start_bucket)

        heapq.heappush(pq, (start_dist, start_bucket))
        visited: Set[Coord] = set()

        while pq:
            curr_dist, bucket = heapq.heappop(pq)

            if best_distance and curr_dist > best_distance:
                break

            if bucket in visited:
                continue

            visited.add(bucket)

            for candidate in self._buckets.get(bucket, set()):
                dist = max(abs(candidate.x - qx), abs(candidate.y - qy))

                if not best_distance or dist < best_distance:
                    best_distance = dist
                    best_point = candidate

            for neigh in bucket.neighbours():
                if neigh in visited:
                    continue

                neigh_dist = self._min_dist_to_bucket(point, neigh)

                if not best_distance or neigh_dist <= best_distance:
                    heapq.heappush(pq, (neigh_dist, neigh))

        assert best_point is not None
        return best_point
