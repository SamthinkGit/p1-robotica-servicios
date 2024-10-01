import GUI  # noqa
import HAL  # noqa
import heapq
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Iterator, Iterable, Callable
from enum import Enum
import math
import cv2
import time

# =========== CONSTANTS =================
ROTATION = math.pi
SHIFT_X = 100
SHIFT_Y = -100
CELL_SIZE = 15
SCALE_Y = 1
SCALE_X = 1
ROTATION_ACCURACY = 2
ROTATION_SMOOTH = 2
NAVIGATION_SMOOTH = 0.005
FORWARDING_ROTATION_FORCE = 0.2
MAXIMUM_SPEED = 2
ERROR_DISTANCE = 30


class Colors(Enum):
    BLACK: int = 0
    RED: int = 128
    YELLOW: int = 130
    GREEN: int = 131
    VIOLET: int = 134
    WHITE: int = 127


# =========== Classes =================
class ProcessManager:

    current_task: Optional[Callable] = None
    state: int = float("-inf")
    _edging: bool = True
    _edge_calls: list[Callable] = []
    _force_next_state: bool = False

    def running(
        self,
        func: Callable,
        *args,
        state: float = 0,
        **kwargs,
    ) -> bool:
        """
        Executes a given function as a task if the current state allows it to
        start. It manages task execution and state transitions based on the current process state.

        :param func: The function to be executed as a task.
        :param args: Arguments to pass to the task function.
        :param state: The state required to run the task.
        :return: True if the task is running, False otherwise.

        Example usage:
        ```python

        # First define a bucle that defines the action
        def my_func(arg1, arg2) -> Iterable:
            for i in range(5):
              print(arg1 + arg2)
              yield
            return

        # Then you can easily manage that bucle inside a loop as if it were a
        normal function. Note that it will return True on success:
        processManager = ProcessManager()
        while True:
            if processManager.running(my_func, arg_1=3, arg_2=4, state=0)
                continue

            if processManager.running(my_func, arg_1=2, arg_2=5, state=1)
                continue
            exit()
        ```
        """

        if isinstance(state, Enum):
            state = state.value

        if self._force_next_state:
            self._force_next_state = False
            return False

        if state < self.state:
            return False

        # Check if the task is starting
        if state > self.state:
            self._edging = False
            self.state = state
            self.current_task = func(*args, **kwargs)

        # Execute the task
        try:
            next(self.current_task)
        except StopIteration:
            self._edging = True
            return False

        return True

    def edging(self, state: Optional[int] = None) -> bool:
        """
        Checks if the process manager is currently edging, which means
        it is transitioning between states. It can also check for a
        specific state if provided.

        :param state: An optional state to check against the current state.
        :return: True if edging is active for the given state, otherwise False.
        """
        if state is not None:
            return self._edging and self.state == state

        return self._edging

    def log(self):
        """
        Outputs the current state, task, and edging status of the
        process manager for debugging purposes. This includes information
        about the current task being executed and the active state.
        """
        print(
            {
                "current_task": self.current_task,
                "state": self.state,
                "edging": self._edging,
            }
        )

    def at_edge(self, func: Callable) -> None:
        """
        Registers a callback function to be executed when the state
        transitions to an edge state. This allows for custom actions during
        state changes.

        :param func: The callback function to be called at the edge.
        """

        self._edge_calls.append(func)

    def set_next_state(self):
        """
        Sets the next state to be executed and triggers any registered
        edge callbacks. This method is called to advance the state of
        the process manager.
        """
        self._edging = True
        for call in self._edge_calls:
            call()
        self._force_next_state = True

    def change_state(self, state: int | Enum):
        """
        Changes the current state of the process manager. This allows
        for the state to be updated based on the robot's current
        operational context.

        :param state: The new state to transition to, which can be an
                      integer or an enumeration.
        """
        if isinstance(state, Enum):
            state = state.val

        self._edging = True
        for call in self._edge_calls:
            call()
        self._edging = True
        self.state = state

    def flush(self) -> None:
        """
        Resets the current task and state of the process manager. This
        effectively clears the task queue and prepares the manager for
        a new set of tasks.
        """

        self.state = -float("inf")
        self.current_task = None


class Map:

    _robot2map_mat: np.ndarray = None

    def __init__(self, path: str) -> None:
        rgb = GUI.getMap(path)
        self.map: np.ndarray = np.where(
            np.sum(rgb, axis=2) > 0, Colors.WHITE.value, Colors.BLACK.value
        )
        self._path = path

    def define_transformation(self, robot2map_mat: np.ndarray):
        self._robot2map_mat = robot2map_mat

    def robot2map(self, coords: np.ndarray) -> list[int]:
        result = self._robot2map_mat @ coords
        return [int(result[0] * SCALE_X + SHIFT_X), int(result[1] * SCALE_Y + SHIFT_Y)]

    def flush(self):
        self.__init__(self._path)

    def show(self):
        GUI.showNumpy(self.map)

    def add_keypoint(
        self, x: int, y: int, color: int = Colors.GREEN.value, size: int = 10
    ):
        self.map[y - size : y + size, x - size : x + size] = color  # noqa


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


# =========== Some Util Functions=================


class Navigation:

    @staticmethod
    def get_current_map_coords(mapping: Map) -> list[float, float]:
        x_robot = HAL.getPose3d().x
        y_robot = HAL.getPose3d().y
        robot_cords = np.array([x_robot, y_robot, 1]).reshape(-1, 1)
        return mapping.robot2map(robot_cords)

    @staticmethod
    def wait(seconds: float) -> Iterable:
        print(f"Waiting {seconds}s...")
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            yield
        return

    @staticmethod
    def navigateTo(
        mapping: Map, target_x: float, target_y: float, skip_rotation: bool = False
    ) -> Iterable:

        current_x, current_y = Navigation.get_current_map_coords(mapping)
        current_yaw = HAL.getPose3d().yaw
        target_yaw = Navigation._compute_yaw(current_x, current_y, target_x, target_y)

        # Rotate
        if not skip_rotation:
            while round(current_yaw, ROTATION_ACCURACY) != round(
                target_yaw, ROTATION_ACCURACY
            ):
                rotation = (target_yaw - current_yaw) / ROTATION_SMOOTH
                HAL.setW(rotation)
                current_yaw = HAL.getPose3d().yaw
                print(f"Rotating from [{current_yaw}] to [{target_yaw}]: {rotation}")
                yield

        # Forward
        error = float("inf")

        while error > ERROR_DISTANCE:
            current_x, current_y = Navigation.get_current_map_coords(mapping)
            current_yaw = HAL.getPose3d().yaw

            mapping.add_keypoint(current_x, current_y, Colors.RED.value)
            mapping.add_keypoint(target_x, target_y)

            error = np.sqrt(
                (abs(target_x - current_x) ** 2) + (abs(target_y - current_y) ** 2)
            )
            velocity = min(error * NAVIGATION_SMOOTH, MAXIMUM_SPEED)

            # Fix going to the right
            if current_yaw < -2.90:
                current_yaw = np.pi+0.2

            direction = -1 if target_yaw < current_yaw else 1
            rotation = FORWARDING_ROTATION_FORCE * direction

            print(
                f"Going to {[target_x, target_y]} from {[current_x, current_y]}: {round(velocity,2)}m/s. Rot: {rotation} -> "
                f"[{round(error,2)}m {round(current_yaw,2)}º]"
            )

            HAL.setW(rotation)
            HAL.setV(velocity)
            yield

        HAL.setW(0)
        HAL.setV(0)

        return

    @staticmethod
    def _compute_yaw(x0: float, y0: float, x1: float, y1: float):
        x = x1 - x0
        y = y1 - y0
        rad = -(np.arctan2(x, y) + np.pi / 2)
        if rad < -np.pi:
            rad += 2 * np.pi

        # We change from upper-left based space to bottom-left space
        rad = -rad

        return rad


# =========== Defining Coordinate Space =================
robot_points = np.array(
    [[5.1316, 5.5555], [-2.527, 5.5310], [-0.9999, 1.500]], dtype=np.float32
)
map_points = np.array([[48, 996], [850, 956], [570, 670]], dtype=np.float32)
M = cv2.getAffineTransform(robot_points, map_points)

# =========== Initialization =================
print("Starting...")
processManager = ProcessManager()
mapping = Map(
    "/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png"
)
mapping.define_transformation(M)

grid = Grid()
grid.load_matrix(mapping.map)

# src = grid[5, 10]
# dst = grid[5, 50]

# path = grid.compute_path(src, dst)
# for cell in path:
#     cell.fill(Colors.YELLOW.value)

# src.fill(Colors.GREEN.value)
# dst.fill(Colors.RED.value)


while True:

    mapping.show()
    mapping.flush()

    if processManager.running(Navigation.wait, seconds=2, state=1):
        continue

    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+40,
        target_y=570,
        state=2,
    ):
        continue

    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+80,
        target_y=570,
        skip_rotation=True,
        state=3,
    ):
        continue

    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+120,
        target_y=570,
        skip_rotation=True,
        state=4,
    ):
        continue

    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+160,
        target_y=570,
        skip_rotation=True,
        state=5,
    ):
        continue

    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+160,
        target_y=570-40,
        skip_rotation=False,
        state=6,
    ):
        continue
    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+160,
        target_y=570-80,
        skip_rotation=True,
        state=7,
    ):
        continue
    if processManager.running(
        Navigation.navigateTo,
        mapping=mapping,
        target_x=670+160,
        target_y=570-120,
        skip_rotation=True,
        state=8,
    ):
        continue