from collections import deque
from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray
from routers.solution import Solution
from routers.measure import Measurements
from typing import (
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    TypeAlias,
    TypeVar,
    Union,
)

from threading import Event, Thread
from routers.spatial import Coord
import time


T = TypeVar("T")


@dataclass
class Progress:

    title: str | None = None
    percentage: int = -1
    subprogress: Optional["Progress"] = None

    def only_title(self, title: str):
        self.title = title
        self.percentage = -1

    def finalize(self):
        self.percentage = 100

    def range(self, size: int, desc: str | None = None):
        return self.iter_collection(range(size), desc)

    def n_out_of(self, n: int, out_of: int):
        self.percentage = (n * 100) // out_of

    def iter_collection(self, collection: Collection[T], desc: str | None = None):
        return self.iter_est(collection, len(collection), desc)

    def iter_est(self, iter: Iterable[T], len: int, desc: str | None = None):
        if len == 0:
            return

        self.title = desc
        self.percentage = 0

        delta = 100 / len
        progress = 0.0

        for item in iter:
            progress += delta
            self.percentage = int(progress)

            yield item

    def spawn(self, title: str | None = None):
        self.subprogress = Progress(title=title)
        return self.subprogress

    def despawn(self):
        self.subprogress = None


IsBest: TypeAlias = bool


class StageResultKind(Enum):

    Solution = 0
    Heatmap = 1
    Clusters = 2


@dataclass(frozen=True)
class HeatmapResult:

    heatmap: NDArray
    threshold: float
    kind: Literal[StageResultKind.Heatmap] = StageResultKind.Heatmap


@dataclass(frozen=True)
class ClustersResult:

    clusters: List[Set[Coord]]
    kind: Literal[StageResultKind.Clusters] = StageResultKind.Clusters


@dataclass(frozen=True)
class SolutionResult:

    solution: Solution
    is_best: bool
    kind: Literal[StageResultKind.Solution] = StageResultKind.Solution


YieldedResult: TypeAlias = Union[HeatmapResult, ClustersResult, SolutionResult]
Stage: TypeAlias = Callable[[Solution, Progress], Iterable[YieldedResult]]


class StageJob:

    progress: Progress

    best_solution: Solution
    recent_solution: Solution
    recent_heatmap: HeatmapResult | None
    recent_clusters: ClustersResult | None

    _start_time: float
    _wait_time: float
    _wait_start: float
    _end_time: float

    _stage: Stage
    _cancel_event: Event
    _finish_event: Event
    _running_flag: Event
    _finished: bool

    def __init__(self, stage: Stage, solution: Solution, on_finish: Event) -> None:

        self._stage = stage
        self.progress = Progress()

        self.best_solution = solution
        self.recent_solution = solution
        self.recent_heatmap = None
        self.recent_clusters = None

        self._cancel_event = Event()
        self._running_flag = Event()
        self._running_flag.set()
        self._finish_event = on_finish
        self._finished = False

        self._wait_time = 0.0
        self._wait_start = 0.0
        self._start_time = 0.0
        self._end_time = 0.0

        thread = Thread(target=self._execute_stage)
        thread.daemon = True
        thread.start()

    def toggle_pause(self):
        if self._running_flag.is_set():
            self._running_flag.clear()
        else:
            self._running_flag.set()

    def cancel(self):
        self._running_flag.set()

        if not self._finish_event.is_set():
            self._cancel_event.set()

    def is_cancelling(self):
        return self._cancel_event.is_set()

    def is_finished(self):
        return self._finished

    def time(self):
        if self._finished:
            return (self._end_time - self._start_time) - self._wait_time

        # Is not waiting
        elif self._wait_start == 0.0:
            return (time.perf_counter() - self._start_time) - self._wait_time

        # Is waiting
        else:
            return (self._wait_start - self._start_time) - self._wait_time

    def _execute_stage(self):
        self._start_time = time.perf_counter()

        for result in self._stage(self.best_solution, self.progress):

            if result.kind == StageResultKind.Heatmap:
                self.recent_heatmap = result

            elif result.kind == StageResultKind.Clusters:
                self.recent_clusters = result

            elif result.kind == StageResultKind.Solution:
                self.recent_solution = result.solution

                if result.is_best:
                    if result.solution.fitness() >= self.best_solution.fitness():
                        self.best_solution = result.solution
            else:
                raise RuntimeError("Unknown result!")

            if self._cancel_event.is_set():
                break

            if not self._running_flag.is_set():
                self._wait_start = time.perf_counter()
                self._running_flag.wait()
                self._wait_time += time.perf_counter() - self._wait_start
                self._wait_start = 0.0

        self._end_time = time.perf_counter()
        self._finished = True
        self._finish_event.set()


class QueueJobPlanner:

    running_job: Optional[StageJob]
    solution: Solution
    total_time: float

    _stages: deque[Stage | None]
    _job_done: Event
    _job_unpaused: Event
    _measure: Measurements | None

    def __init__(self, initial: Solution, measurements: Measurements | None = None):

        self.total_time = 0.0
        self.running_job = None
        self.solution = initial
        self._stages = deque()
        self._job_done = Event()
        self._measure = measurements

    def enqueue_stage(self, stage: Stage | None):
        self._stages.append(stage)

    def skip(self):
        if self.running_job is not None:
            self.running_job.cancel()
            self.total_time += self.running_job.time()

    def skip_all(self):
        self._stages.clear()
        self.skip()

    def toggle_pause(self):
        if self.running_job is not None:
            self.running_job.toggle_pause()

    def update(self):

        result = None

        if self.running_job is not None and self._job_done.is_set():
            self.solution = self.running_job.best_solution
            self.running_job, result = None, self.running_job
            self.total_time += result.time()
            self._job_done.clear()

        if self.running_job is not None:
            return result

        if not self._stages:
            return result

        stage = self._stages.popleft()

        if stage is None:
            if self._measure:
                self._measure.finish(self.total_time, self.solution.fitness())

        else:
            self.running_job = StageJob(stage, self.solution, self._job_done)

        return result
