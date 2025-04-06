from dataclasses import asdict
from pathlib import Path
import click
import pygame

from routers.gui import GUI, WINDOW_HEIGHT, WINDOW_WIDTH
from routers.measure import Measurements
from routers.problem import Problem
from routers.run import QueueJobPlanner
from routers.solution import Solution
from routers.solve import (
    neigh_func_cheap,
    simulated_annealing,
    smart_shuffle,
)
from routers.solvers.genetic import genetic_algorithm
from routers.utils import tabulate


def run_genetic(
    problem: Problem,
    iters: int,
    population: int,
    tournamet: int,
    measure: Measurements | None,
):
    if measure:
        measure.start(iters, tournamet, population)

    planner = QueueJobPlanner(Solution.Empty(problem), measure)
    planner.enqueue_stage(genetic_algorithm(population, iters, tournamet, 0.9, 0.2))
    planner.enqueue_stage(None)

    return planner


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "-a",
    "--algorithm",
    type=click.Choice(["simulated_annealing", "local_search", "genetic"]),
    required=True,
    help="Algorithm to use",
)
@click.option(
    "-i",
    "--iterations",
    type=int,
    default=1000,
    help="Number of iterations",
)
@click.option(
    "-p",
    "--population-size",
    type=int,
    default=50,
    help="Population size (for genetic algorithm)",
)
@click.option(
    "-t",
    "--tournament-size",
    type=int,
    default=5,
    help="Tournament size (for genetic algorithm)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=False,
    help="Dest .csv for measurements",
)
def cli(
    input: str,
    algorithm: str,
    iterations: int,
    population_size: int,
    tournament_size: int,
    output: str | None,
):

    path = Path(input)
    problem = Problem.ParseInput(path)
    tabulate(asdict(problem), f"Loaded {path.name}")

    if output is not None:
        measure = Measurements(Path(output), path.stem, problem)
    else:
        measure = None

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    planner = run_genetic(
        problem,
        iterations,
        population_size,
        tournament_size,
        measure,
    )

    gui = GUI(screen, problem, planner)
    gui.run_wait()


if __name__ == "__main__":
    cli()
