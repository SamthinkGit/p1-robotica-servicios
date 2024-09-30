from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Iterator
from enum import Enum
import math
import heapq
import matplotlib.pyplot as plt
from icecream import ic

# =========== CONSTANTS =================
ROTATION = math.pi
SHIFT_X = 100
SHIFT_Y = -100
CELL_SIZE = 3


class Colors(Enum):
    BLACK: int = 1
    RED: int = 128
    YELLOW: int = 130
    GREEN: int = 131
    VIOLET: int = 134
    WHITE: int = 127


# =========== Classes =================
@dataclass
class Cell:

    # Grid coordinates
    i: int
    j: int

    # Pixel coordinates
    x0: int
    y0: int
    center_x: Optional[int] = None
    center_y: Optional[int] = None
    content: Optional[np.ndarray] = None

    # Adjacent cells
    bottom: Optional["Cell"] = field(default=None, repr=False)
    top: Optional["Cell"] = field(default=None, repr=False)
    left: Optional["Cell"] = field(default=None, repr=False)
    right: Optional["Cell"] = field(default=None, repr=False)
    _from: Optional["Cell"] = field(default=None, repr=False)

    def __post_init__(self):
        if self.center_x is None:
            self.center_x = self.x0 + (CELL_SIZE // 2)

        if self.center_y is None:
            self.center_y = self.y0 + (CELL_SIZE // 2)

    def fill(self, color: int = Colors.YELLOW.value, forced: bool = False):
        for row in self.content:
            for idx, value in enumerate(row):
                if value == Colors.WHITE.value or forced:
                    row[idx] = color

    def __hash__(self) -> int:
        return hash((self.x0, self.y0))

    @property
    def occupied(self):
        return Colors.BLACK.value in self.content

    def __contains__(self, coords: tuple[int]):
        return (
            coords[0] >= self.x0
            and coords[0] < self.x0 + CELL_SIZE
            and coords[1] >= self.y0
            and coords[1] < self.y0 + CELL_SIZE
        )

    def __lt__(self, other: "Cell") -> bool:
        return id(self) < id(other)

    def __eq__(self, other):
        return isinstance(other, Cell) and self.x0 == other.x0 and self.y0 == other.y0

    @classmethod
    def distance(cls, cell_1: "Cell", cell_2: "Cell"):
        return abs(cell_2.center_x - cell_1.center_x) + abs(
            cell_2.center_y - cell_1.center_y
        )


class Grid:
    _cells: Optional[list[list[Cell]]] = None
    _matrix: Optional[np.ndarray] = None

    def load_matrix(self, matrix: np.ndarray) -> None:
        self._matrix = matrix
        self._cells = []

        # Building _cells matrix
        for grid_i, mat_j_shifted in enumerate(
            range(CELL_SIZE, matrix.shape[0], CELL_SIZE)
        ):
            grid_row = []

            for grid_j, mat_i_shifted in enumerate(
                range(CELL_SIZE, matrix.shape[1], CELL_SIZE)
            ):

                cell = Cell(
                    i=grid_i,
                    j=grid_j,
                    x0=mat_i_shifted - CELL_SIZE,
                    y0=mat_j_shifted - CELL_SIZE,
                )
                cell.content = matrix[
                    cell.x0 : cell.x0 + CELL_SIZE, cell.y0 : cell.y0 + CELL_SIZE  # noqa
                ]
                grid_row.append(cell)
            self._cells.append(grid_row)

        # Updating cells with their correspondent adjacent
        for cell in self:
            if (
                cell.i == 0
                or cell.j == 0
                or cell.i == len(self._cells[0]) - 1
                or cell.j == len(self._cells) - 1
            ):
                continue

            try:
                cell.bottom = self[cell.i, cell.j + 1]
                cell.top = self[cell.i, cell.j - 1]
                cell.left = self[cell.i - 1, cell.j]
                cell.right = self[cell.i + 1, cell.j]
            except IndexError:
                pass

    def compute_path(self, source: Cell, target: Cell) -> list[Cell]:
        queue = []
        heapq.heappush(queue, (0, source))
        cost = {source: 0}
        closed_set = set()

        while queue:
            _, current = heapq.heappop(queue)

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == target:
                path = [current]
                while current != source:
                    current = current._from
                    path.append(current)
                path.reverse()
                return path

            for neighbor in [current.left, current.bottom, current.right, current.top]:
                if neighbor is None or neighbor.occupied:
                    continue

                new_cost = cost[current]

                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + Cell.distance(neighbor, target)
                    heapq.heappush(queue, (priority, neighbor))
                    neighbor._from = current

        return None

    def __getitem__(self, idx: tuple[int]) -> Cell | list[Cell]:
        if len(idx) != 2:
            raise ValueError("Grid only can be accesed with double index.")
        return self._cells[idx[0]][idx[1]]

    def __iter__(self) -> Iterator[Cell]:
        for row in self._cells:
            for cell in row:
                yield cell
        return
    

if __name__ == "__main__":

    mock_image = np.zeros([100, 100], dtype=int)
    mock_image[15:40, 15:50] = 1

    grid = Grid()
    grid.load_matrix(mock_image)
    src = grid[20, 20]
    dst = grid[1, 1]
    src.fill(4, forced=True)
    dst.fill(4, forced=True)

    cell = grid[1, 1]
    path = grid.compute_path(src, dst)
    for cell in path:
        cell.fill(3, forced=True)

    plt.imshow(mock_image, cmap="viridis", interpolation="none")
    plt.colorbar()
    plt.show()
