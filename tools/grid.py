from dataclasses import dataclass
import numpy as np
from typing import Optional, Iterator
from enum import Enum


class Colors(Enum):
    BLACK: int = 127
    RED: int = 128
    YELLOW: int = 130
    GREEN: int = 131
    VIOLET: int = 134
    WHITE: int = 0


CELL_SIZE = 7


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

    def __post_init__(self):
        if self.center_x is None:
            self.center_x = self.x0 + (CELL_SIZE // 2)

        if self.center_y is None:
            self.center_y = self.y0 + (CELL_SIZE // 2)

    def fill(self, color: int = Colors.YELLOW.value):
        for row in self.content:
            for idx, value in enumerate(row):
                if value == Colors.WHITE.value:
                    row[idx] = color

    @property
    def occupied(self):
        return self.content[CELL_SIZE // 2][CELL_SIZE // 2] == Colors.BLACK.value

    def __contains__(self, coords: tuple[int]):
        return (
            coords[0] >= self.x0
            and coords[0] < self.x0 + CELL_SIZE
            and coords[1] >= self.y0
            and coords[1] < self.y0 + CELL_SIZE
        )


class Grid:
    _cells: list[list[Cell]] = None

    def load_matrix(self, matrix: np.ndarray) -> None:
        self._cells = []

        for grid_j, mat_j_shifted in enumerate(
            range(CELL_SIZE, matrix.shape[0], CELL_SIZE)
        ):
            grid_row = []

            for grid_i, mat_i_shifted in enumerate(
                range(CELL_SIZE, matrix.shape[1], CELL_SIZE)
            ):

                cell = Cell(
                    i=grid_i,
                    j=grid_j,
                    x0=mat_i_shifted - CELL_SIZE,
                    y0=mat_j_shifted - CELL_SIZE,
                )
                cell.content = matrix[
                    cell.x0: cell.x0 + CELL_SIZE, cell.y0: cell.y0 + CELL_SIZE
                ]
                grid_row.append(cell)
            self._cells.append(grid_row)

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
    mock_image = np.zeros([30, 25], dtype=int)
    mock_image[10:20, 10:20] = 1

    grid = Grid()
    grid.load_matrix(mock_image)

    outter_pixels = CELL_SIZE // 2
    inner_pixels = outter_pixels - 1

    for idx, cell in enumerate(grid, start=2):
        mock_image[
            cell.center_y - outter_pixels: cell.center_y + outter_pixels + 1,
            cell.center_x - outter_pixels: cell.center_x + outter_pixels + 1,
        ] = idx//2
        mock_image[
            cell.center_y - inner_pixels: cell.center_y + inner_pixels + 1,
            cell.center_x - inner_pixels: cell.center_x + inner_pixels + 1,
        ] = 0
        mock_image[cell.center_y, cell.center_x] = 5

    print(mock_image)
    print((0, 0) in grid[0, 0])
    print((6, 6) in grid[0, 0])
    print((7, 6) in grid[0, 0])
    print(grid[0, 0].occupied)
    print(grid[0,0])
    print(grid[2,0])
