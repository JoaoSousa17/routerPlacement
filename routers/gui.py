from enum import Enum
from typing import Iterable, Tuple
from routers.problem import CellKind, Problem
from routers.run import (
    HeatmapResult,
    Progress,
    QueueJobPlanner,
    ClustersResult,
    StageJob,
)
from routers.spatial import Coord
from routers.solve import Solution
from pygame import Surface
import pygame.freetype
import pygame


COLORS = {
    "Empty": (255, 255, 255),
    "Wall": (50, 50, 50),
    "Target": (255, 100, 180),
    "Covered": (200, 255, 0, 100),
    "Wire": (0, 180, 100, 150),
    "Backbone": (0, 255, 255),
    "Router": (255, 0, 0),
    "Movable": (0, 255, 255),
}


class GridPainter:

    _pixels_per_cell: int
    _problem: Problem

    COLORS = {
        CellKind.Void: COLORS["Empty"],
        CellKind.Target: COLORS["Target"],
        CellKind.Wall: COLORS["Wall"],
    }

    def __init__(self, problem: Problem, pixels_per_cell: int = 1):
        assert pixels_per_cell > 0

        self._pixels_per_cell = pixels_per_cell
        self._problem = problem

    def _coord_to_pixel(self, coord: Coord):
        return (coord.x * self._pixels_per_cell, coord.y * self._pixels_per_cell)

    def _coord_to_rect(self, coord: Coord):
        xpix, ypix = self._coord_to_pixel(coord)
        return pygame.Rect(xpix, ypix, self._pixels_per_cell, self._pixels_per_cell)

    def _new_surface(self, flags: int = 0):
        surface_size = (
            self._problem.grid.width * self._pixels_per_cell,
            self._problem.grid.height * self._pixels_per_cell,
        )

        return pygame.Surface(surface_size, flags)

    def prerender_grid(self):
        surface = self._new_surface()

        for y in range(self._problem.grid.height):
            for x in range(self._problem.grid.width):
                grid_coord = Coord(x, y)
                color = self.COLORS[self._problem.grid[grid_coord]]
                pygame.draw.rect(surface, color, self._coord_to_rect(grid_coord))

        pygame.draw.rect(
            surface,
            COLORS["Backbone"],
            self._coord_to_rect(self._problem.backbone),
        )

        return surface

    def render_soulution(self, solution: Solution):
        return self._render_grid_overlay(
            (COLORS["Covered"], solution.cell_coverage),
            (COLORS["Wire"], solution.place_cables()),
            (COLORS["Router"], solution.routers),
            (COLORS["Movable"], solution.movable_routers),
        )

    def _render_grid_overlay(self, *ccss: Tuple[Tuple[int, ...], Iterable[Coord]]):
        surface = self._new_surface(pygame.SRCALPHA)

        for color, coords in ccss:
            for coord in list(coords):
                pygame.draw.rect(surface, color, self._coord_to_rect(coord))

        return surface

    def render_heatmap(self, heatmap: HeatmapResult):
        surface = self._new_surface(pygame.SRCALPHA)
        max_val = heatmap.heatmap.max()

        if max_val == 0:
            return surface

        for coord, _ in self._problem.grid:
            x, y = coord
            val = heatmap.heatmap[x, y]

            if val <= 0:
                continue

            intensity = val / max_val

            if intensity < heatmap.threshold:
                ratio = intensity / heatmap.threshold
                r = 0
                g = int(255 * ratio)
                b = int(255 * (1 - ratio))

            else:
                ratio = (intensity - heatmap.threshold) / (1 - heatmap.threshold)
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                b = 0

            alpha = int(50 + 205 * intensity)
            color = (r, g, b, alpha)
            pygame.draw.rect(surface, color, self._coord_to_rect(coord))

        return surface

    def render_clusters(self, clusters: ClustersResult):
        surface = self._new_surface(pygame.SRCALPHA)
        palette = [
            (255, 0, 0, 150),  # red
            (0, 255, 0, 150),  # green
            (0, 0, 255, 150),  # blue
            (255, 255, 0, 150),  # yellow
            (255, 0, 255, 150),  # magenta
            (0, 255, 255, 150),  # cyan
            (255, 128, 0, 150),  # orange
            (128, 0, 255, 150),  # violet
            (0, 128, 128, 150),  # teal
            (128, 128, 0, 150),  # olive
        ]

        for i, cluster in enumerate(clusters.clusters):
            color = palette[i % len(palette)]

            for coord in cluster:
                pygame.draw.rect(surface, color, self._coord_to_rect(coord))

        return surface


def blit_to_bouding_surface(original: pygame.Surface, bouding: pygame.Surface):
    scale_factor = min(
        bouding.get_width() / original.get_width(),
        bouding.get_height() / original.get_height(),
    )

    new_width = int(original.get_width() * scale_factor)
    new_height = int(original.get_height() * scale_factor)

    init_coord = (
        max(0, (bouding.get_width() - new_width) / 2),
        max(0, (bouding.get_height() - new_height) / 2),
    )

    scaled_surface = pygame.transform.scale(
        original,
        (new_width, new_height),
    )

    bouding.blit(
        scaled_surface,
        init_coord,
    )


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60

SIDEBAR_WIDTH = 300

PROGRESS_BAR_MARGIN = 2
PROGRESS_BAR_HEIGHT = 30

CELL_SIZE = 1


class SpecialDisplay(Enum):
    Normal = 0
    Heatmap = 1
    Clusters = 2


class GUI:

    _screen: Surface
    _main_area: Surface
    _sidebar_area: Surface

    _problem: Problem
    _planner: QueueJobPlanner
    _painter: GridPainter

    _background: pygame.Surface

    _painted_solution: Solution
    _solution_surf: pygame.Surface

    _painted_heatmap: HeatmapResult | None
    _heatmap_surf: pygame.Surface | None

    _painted_clusters: ClustersResult | None
    _clusters_surf: Surface | None

    _show_special: SpecialDisplay

    def __init__(
        self,
        screen: Surface,
        problem: Problem,
        planner: QueueJobPlanner,
    ) -> None:

        self._screen = screen
        self._main_area = pygame.Surface((WINDOW_WIDTH - SIDEBAR_WIDTH, WINDOW_HEIGHT))
        self._sidebar_area = pygame.Surface((SIDEBAR_WIDTH, WINDOW_HEIGHT))

        self._problem = problem
        self._planner = planner
        self._painter = GridPainter(problem, CELL_SIZE)

        self._background = self._painter.prerender_grid()

        self._painted_solution = planner.solution
        self._solution_surf = self._painter.render_soulution(self._painted_solution)

        self._painted_heatmap = None
        self._heatmap_surf = None

        self._painted_clusters = None
        self._clusters_surf = None

        self._show_special = SpecialDisplay.Normal

    def _outlined_text(self, text: str, size: int, outline: int):

        font = pygame.font.Font(None, size)
        font.set_bold(True)

        outline_surf = font.render(text, True, (0, 0, 0))
        outline_size = outline_surf.get_size()

        text_surf = pygame.Surface(
            (
                outline_size[0] + outline * 2,
                outline_size[1] + 2 * outline,
            ),
            pygame.SRCALPHA,
        )

        text_rect = text_surf.get_rect()

        offsets = [
            (ox, oy)
            for ox in range(-outline, 2 * outline, outline)
            for oy in range(-outline, 2 * outline, outline)
            if ox != 0 or ox != 0
        ]

        for ox, oy in offsets:
            px, py = text_rect.center

            text_surf.blit(
                outline_surf,
                outline_surf.get_rect(center=(px + ox, py + oy)),
            )

        inner_text = font.render(text, True, (255, 255, 255)).convert_alpha()
        text_surf.blit(inner_text, inner_text.get_rect(center=text_rect.center))

        return text_surf

    def _draw_progress_bar(self, progress: Progress):

        bar_surface = Surface((SIDEBAR_WIDTH, PROGRESS_BAR_HEIGHT))
        fill_width = SIDEBAR_WIDTH - 2 * PROGRESS_BAR_MARGIN
        fill_height = PROGRESS_BAR_HEIGHT - 2 * PROGRESS_BAR_MARGIN

        if progress.percentage < 0:
            fill_color = (50, 50, 255)

        else:
            fill_width = int(fill_width * (progress.percentage / 100))
            fill_color = (50, 255, 50)

        fill_rect = pygame.Rect(
            PROGRESS_BAR_MARGIN,
            PROGRESS_BAR_MARGIN,
            fill_width,
            fill_height,
        )

        bar_surface.fill((50, 50, 50))
        pygame.draw.rect(bar_surface, fill_color, fill_rect)

        if progress.percentage >= 0:
            perc_surface = self._outlined_text(f"{progress.percentage}%", 21, 2)
            perc_rect = perc_surface.get_rect(
                midleft=(
                    SIDEBAR_WIDTH - 2 * PROGRESS_BAR_MARGIN - perc_surface.get_width(),
                    PROGRESS_BAR_HEIGHT // 2,
                )
            )

            bar_surface.blit(perc_surface, perc_rect)

        if progress.title:
            title_surface = self._outlined_text(progress.title, 21, 2)
            title_rect = title_surface.get_rect(
                center=(
                    SIDEBAR_WIDTH // 2,
                    PROGRESS_BAR_HEIGHT // 2,
                )
            )

            bar_surface.blit(title_surface, title_rect)

        return bar_surface

    def _draw_sidebar(self):
        font = pygame.font.Font(None, 24)

        score_text = font.render(
            f"Score: {self._painted_solution.fitness()}",
            True,
            (255, 255, 255),
        )

        routers_text = font.render(
            f"Routers: {len(self._painted_solution.routers)}",
            True,
            (255, 255, 255),
        )

        coverage_text = font.render(
            f"Coverage: {self._painted_solution.percent_covered() * 100:.2f}%",
            True,
            (255, 255, 255),
        )

        budget_text = font.render(
            f"Cost: {self._painted_solution.cost()} / {self._painted_solution.problem.budget}",
            True,
            (255, 255, 255),
        )

        time_text = font.render(
            f"Time: {self._planner.total_time:.2f} s",
            True,
            (255, 255, 255),
        )

        self._sidebar_area.blit(score_text, (5, 5))
        self._sidebar_area.blit(routers_text, (5, 10 + score_text.get_height()))

        self._sidebar_area.blit(
            coverage_text,
            (5, 15 + score_text.get_height() + routers_text.get_height()),
        )

        self._sidebar_area.blit(
            budget_text,
            (
                5,
                20
                + score_text.get_height()
                + routers_text.get_height()
                + coverage_text.get_height(),
            ),
        )

        self._sidebar_area.blit(
            time_text,
            (
                5,
                25
                + score_text.get_height()
                + routers_text.get_height()
                + coverage_text.get_height()
                + budget_text.get_height(),
            ),
        )

        progress = self._planner.running_job and self._planner.running_job.progress
        vpos = WINDOW_HEIGHT - PROGRESS_BAR_HEIGHT

        while progress is not None:
            bar = self._draw_progress_bar(progress)
            self._sidebar_area.blit(bar, (0, vpos))

            vpos -= PROGRESS_BAR_HEIGHT
            progress = progress.subprogress

    def _paint(self):
        self._screen.fill((255, 255, 255))
        self._main_area.fill(COLORS["Empty"])
        self._sidebar_area.fill((20, 20, 20))
        self._draw_sidebar()

        blit_to_bouding_surface(self._background, self._main_area)

        if self._solution_surf is not None:
            blit_to_bouding_surface(self._solution_surf, self._main_area)

        if (
            self._show_special == SpecialDisplay.Heatmap
            and self._heatmap_surf is not None
        ):
            blit_to_bouding_surface(self._heatmap_surf, self._main_area)

        elif (
            self._show_special == SpecialDisplay.Clusters
            and self._clusters_surf is not None
        ):
            blit_to_bouding_surface(self._clusters_surf, self._main_area)

        self._screen.blit(self._sidebar_area, (0, 0))
        self._screen.blit(self._main_area, (SIDEBAR_WIDTH, 0))
        pygame.display.flip()

    def _update_painted_solution(self, solution: Solution):
        if solution is self._painted_solution:
            return

        self._painted_solution = solution
        self._solution_surf = self._painter.render_soulution(solution)

    def _update_painted_heatmap(self, heatmap: HeatmapResult | None):
        if not heatmap or heatmap is self._painted_heatmap:
            return

        self._painted_heatmap = heatmap
        self._heatmap_surf = self._painter.render_heatmap(heatmap)

    def _update_painted_clusters(self, clusters: ClustersResult | None):
        if not clusters or clusters is self._painted_clusters:
            return

        self._painted_clusters = clusters
        self._clusters_surf = self._painter.render_clusters(clusters)

    def _update_solution(self, job: StageJob):

        if job.is_finished():
            self._update_painted_solution(job.best_solution)
        else:
            self._update_painted_solution(job.recent_solution)

        self._update_painted_heatmap(job.recent_heatmap)
        self._update_painted_clusters(job.recent_clusters)

    def _toggle_special(self, special: SpecialDisplay):
        if self._show_special == special:
            self._show_special = SpecialDisplay.Normal

        else:
            self._show_special = special

    def run_wait(self):
        self._paint()

        clock = pygame.time.Clock()
        running = True

        while running:

            finished_job = self._planner.update()

            if finished_job:
                self._update_solution(finished_job)
            elif self._planner.running_job:
                self._update_solution(self._planner.running_job)

            self._paint()
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self._toggle_special(SpecialDisplay.Heatmap)
                    elif event.key == pygame.K_c:
                        self._toggle_special(SpecialDisplay.Clusters)
                    elif event.key == pygame.K_BACKSPACE:
                        self._planner.skip()
                    elif event.key == pygame.K_SPACE:
                        self._planner.toggle_pause()
                    elif event.key == pygame.K_ESCAPE:
                        self._planner.skip_all()

        pygame.quit()
