import GUI  # noqa
import HAL  # noqa
import heapq
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Iterator, Iterable, Callable
from enum import Enum
import math
import time

# =========== CONSTANTS =================
ROTATION = math.pi
SHIFT_X = 40
SHIFT_Y = -20
CELL_SIZE = 15
SCALE = 108.1
ROTATION_ACCURACY = 1
ROTATION_SMOOTH = 2
NAVIGATION_SMOOTH = 0.035
FORWARDING_ROTATION_FORCE = 0.5
MAXIMUM_SPEED = 1.5
ERROR_DISTANCE = 50
RECOVERY_DISTANCE = 80


class Colors(Enum):
    BLACK: int = 0
    RED: int = 128
    YELLOW: int = 130
    GREEN: int = 131
    VIOLET: int = 134
    BLUE: int = 132
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
        x, y = coords

        vec = np.array([x, y, 0, 1])
        result = M @ vec * SCALE
        return [int(result[1] - SHIFT_X), int(result[0] - SHIFT_Y)]

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
    content: Optional[np.ndarray] = field(default=None, repr=False)

    # Adjacent cells
    bottom: Optional["Cell"] = field(default=None, repr=False)
    top: Optional["Cell"] = field(default=None, repr=False)
    left: Optional["Cell"] = field(default=None, repr=False)
    right: Optional["Cell"] = field(default=None, repr=False)
    _from: Optional["Cell"] = field(default=None, repr=False)

    # Cleaning specialized attributes
    clean: bool = False

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
        return Colors.BLACK.value in self.content or Colors.RED.value in self.content

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

    def dilate_walls(self):
        for cell in self:
            if (
                Colors.BLACK.value in cell.content
                or Colors.BLACK.value in cell.left.content
                or Colors.BLACK.value in cell.right.content
                or Colors.BLACK.value in cell.top.content
                or Colors.BLACK.value in cell.bottom.content
            ):
                cell.fill(Colors.RED.value)

    def get_current_cell(self, mapping: Map):
        x, y = Navigation.get_current_map_coords(mapping)
        return self[x // CELL_SIZE, y // CELL_SIZE]

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
        return mapping.robot2map([x_robot, y_robot])

    @staticmethod
    def wait(seconds: float) -> Iterable:
        print(f"[Wait] Sleeping {seconds}s...")
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            yield
        return

    @staticmethod
    def navigate_to(
        mapping: Map,
        target_x: float,
        target_y: float,
        skip_rotation: bool = False,
        logging: bool = False,
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
                if logging:
                    print(
                        f"Rotating from [{current_yaw}] to [{target_yaw}]: {rotation}"
                    )
                yield

        # Forward
        error = float("inf")

        while error > ERROR_DISTANCE:

            current_x, current_y = Navigation.get_current_map_coords(mapping)
            current_yaw = HAL.getPose3d().yaw
            target_yaw = Navigation._compute_yaw(
                current_x, current_y, target_x, target_y
            )

            mapping.add_keypoint(target_x, target_y, Colors.VIOLET.value)

            error = np.sqrt(
                (abs(target_x - current_x) ** 2) + (abs(target_y - current_y) ** 2)
            )

            velocity = min(error * NAVIGATION_SMOOTH, MAXIMUM_SPEED)
            rotation = (target_yaw - current_yaw) / ROTATION_SMOOTH

            if logging:
                print(
                    f"Going to {[target_x, target_y]} with {target_yaw} from {[current_x, current_y]}: "
                    f"{round(velocity,2)}m/s. Rot: {rotation} -> [{round(error,2)}m {round(current_yaw,2)}º]"
                )

            HAL.setW(rotation)
            HAL.setV(velocity)
            yield

        HAL.setW(0)
        HAL.setV(0)
        return

    @staticmethod
    def find_first_unclean_cell(grid: Grid):
        for cell in grid:
            if not cell.clean and not cell.occupied:
                return cell
        else:
            return None

    @staticmethod
    def route(grid: Grid, mapping: Map, target_cell: Cell) -> Iterable:

        temp_process_manager = ProcessManager()
        current_cell = grid.get_current_cell(mapping)
        path = grid.compute_path(current_cell, target_cell)
        for cell in path:
            cell.fill(Colors.GREEN.value)

        idx = 0
        while idx < len(path):
            if temp_process_manager.running(
                Navigation.navigate_to,
                mapping=mapping,
                target_x=path[idx].center_y,
                target_y=path[idx].center_x,
                state=idx + 1,
            ):
                yield
                continue
            idx += 1
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

    @staticmethod
    def bsa(grid: Grid, mapping: Map) -> Iterable:

        temp_proccess_manager = ProcessManager()
        unblocking_cell: Optional[Cell] = grid.get_current_cell(mapping)
        blocked = False

        # We select a number for incrementing the states of the proccess manager
        # each time a new cell is selected
        state = 0

        while True:

            yield

            current_cell = grid.get_current_cell(mapping)
            # Get the a valid cell if blocked:
            if blocked:
                unblocking_cell = Navigation.find_first_unclean_cell(grid)
                print(
                    f"[BSA] Found new cell at {[unblocking_cell.center_x, unblocking_cell.center_y]}"
                )

            # If there is no valid cell, exit
            if unblocking_cell is None:
                print("[BSA] Algorithm completed, exit.")
                break

            # Go to the start cell
            if temp_proccess_manager.running(
                Navigation.route, grid, mapping, unblocking_cell, state=1
            ):
                continue

            if temp_proccess_manager.edging(state=1):
                print("[BSA] New room reached, cleaning started.")
                state = 1

            # Select the next cell
            if temp_proccess_manager.edging():

                # Selecting cell
                if not current_cell.left.occupied and not current_cell.left.clean:
                    next_cell = current_cell.left

                elif not current_cell.bottom.occupied and not current_cell.bottom.clean:
                    next_cell = current_cell.bottom

                elif not current_cell.right.occupied and not current_cell.right.clean:
                    next_cell = current_cell.right

                elif not current_cell.top.occupied and not current_cell.top.clean:
                    next_cell = current_cell.top
                else:
                    next_cell = None

                state += 1

                # If there is no valid cell, you are blocked
                if next_cell is None:
                    print("[BSA] Room completed, searching for new rooms...")
                    blocked = True
                    temp_proccess_manager.flush()
                    continue

            if Cell.distance(current_cell, next_cell) > RECOVERY_DISTANCE:
                print("[BSA] Navigation failure, starting recovery")
                blocked = True
                temp_proccess_manager.flush()

            # Go to the next cell
            if temp_proccess_manager.running(
                Navigation.navigate_to,
                mapping=mapping,
                target_x=next_cell.center_y,
                target_y=next_cell.center_x,
                state=state,
            ):
                continue

        return


# =========== Defining Coordinate Space =================
alpha = -1.57
tx = 3.68
ty = 5.68
tz = 0
M = np.array(
    [
        [np.cos(alpha), -np.sin(alpha), 0, tx],
        [np.sin(alpha), np.cos(alpha), 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ]
)

# =========== Initialization =================
print('\n'*10)
print("="*30 + " BSA " + "="*30)

processManager = ProcessManager()
mapping = Map(
    "/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png"
)
mapping.define_transformation(M)

grid = Grid()
grid.load_matrix(mapping.map)
grid.dilate_walls()

while True:
    pass

    mapping.show()
    x, y = Navigation.get_current_map_coords(mapping)
    current_cell = grid.get_current_cell(mapping)
    current_cell.clean = True

    mapping.add_keypoint(x, y, Colors.BLUE.value)

    if processManager.running(Navigation.wait, seconds=2, state=1):
        continue

    if processManager.running(Navigation.bsa, grid, mapping, state=2):
        continue
