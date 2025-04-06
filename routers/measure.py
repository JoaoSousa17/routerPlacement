from pathlib import Path
from routers.problem import Problem
import csv


HEADERS = [
    "Name",
    "Budget",
    "Target cells",
    "Router radius",
    "Router price",
    "Cable price",
    "Iterations",
    "Population size",
    "Tournament size",
    "Score",
    "Time",
]


class Measurements:

    _dest: Path
    _name: str
    _problem: Problem

    _iters: int
    _tournamet: int
    _population: int

    def __init__(self, dest: Path, name: str, problem: Problem):

        self._name = name
        self._problem = problem
        self._dest = dest

        if not dest.exists():
            with open(dest, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HEADERS)

    def start(self, iters: int, tournament=0, population=0):
        self._iters = iters
        self._tournamet = tournament
        self._population = population

    def finish(self, time: float, score: int):
        with open(self._dest, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self._name,
                    self._problem.budget,
                    len(list(self._problem.grid.target_cells())),
                    self._problem.router_radius,
                    self._problem.router_price,
                    self._problem.cable_price,
                    self._iters,
                    self._population,
                    self._tournamet,
                    score,
                    round(time, 3),
                ]
            )
