"""
BSA Vacuum Cleaner Coverage
=================================

[IMPORTANT] This Python script has been created from scratch by Sam, with appropriate
references where necessary. Docstrings have been generated with the assistance of
AI for clarity and documentation purposes.

This module implements a BSA algorithmic system for a robotic vacuum cleaner. It
provides classes and functions to manage the robot's process states, map the
environment, and navigate through a grid-based representation of the area. The
system supports various navigation strategies, including pathfinding and state
management during cleaning operations.
"""


"""
==================================================================

[IMPORTANT NOTE]: Due to the size and number of comments and code, 
important functions are indicated with this identifier: 

# 🔥 >>>>    CORE FUNCTION    <<<< 🔥

Make your life easier by going direclty to those functions :D

==================================================================
"""

import GUI  # noqa
import HAL  # noqa
import heapq
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Iterator, Iterable, Callable, Literal
from enum import Enum
import math
import time
import random

# =========== CONSTANTS =================
EXACT_TRACE: bool = False              # Draw exactly where the robot cleans
ROTATION = math.pi                     # Rotation of the map
SHIFT_X = 15                           # Shift on X axis
SHIFT_Y = -35                          # Shift on Y axis
CELL_SIZE = 15                         # Size in pixel of each cell
SCALE = 104.1                          # Scale of the map

DISTANCE_RECOMPUTE = 3                 # Robot intermediate cell removal (efficiency)
ROTATION_ACCURACY = 0.1                # Rotation error
ROTATION_SMOOTH = 0.5                  # Force to rotate the robot
NAVIGATION_SMOOTH = 0.035              # Reactivity when forwarding
FORWARDING_ROTATION_FORCE = 0.5        # Force to rotate when forwarding

MAXIMUM_SPEED = 0.3                    # Maximum robot velocity
ERROR_DISTANCE = 20                    # Error distance in pixels to target cell
RECOVERY_DISTANCE = 80                 # Error distance in pixels to raise a recovery
RECOVERY_TIME = 20                     # Time between cells to raise a recovery


class Colors(Enum):
    BLACK: int = 0
    RED: int = 128
    ORANGE: int = 129
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

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    def define_transformation(self, robot2map_mat: np.ndarray):
        """
        Defines the transformation matrix that relates robot coordinates
        to map coordinates.

        :param robot2map_mat: The transformation matrix to be applied.
        """
        self._robot2map_mat = robot2map_mat

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    def robot2map(self, coords: np.ndarray) -> list[int]:
        """
        Transforms robot coordinates into map coordinates using the defined
        transformation matrix.

        :param coords: The coordinates of the robot in its own frame.
        :return: The corresponding coordinates in the map frame.
        """
        x, y = coords

        vec = np.array([x, y, 0, 1])
        result = M @ vec * SCALE
        return [int(result[1] - SHIFT_X), int(result[0] - SHIFT_Y)]

    def flush(self):
        """
        Resets the map to its initial state by reloading it from the
        specified path.
        """
        self.__init__(self._path)

    def show(self):
        GUI.showNumpy(self.map)

    def add_keypoint(
        self, x: int, y: int, color: int = Colors.GREEN.value, size: int = 10
    ):
        """
        Adds a keypoint to the map at specified coordinates with a given
        color and size.

        :param x: The x-coordinate of the keypoint.
        :param y: The y-coordinate of the keypoint.
        :param color: The color of the keypoint to be displayed.
        :param size: The size of the keypoint to be displayed.
        """
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
        """
        Post-initialization method that calculates the center coordinates
        of the cell if they are not provided.
        """
        if self.center_x is None:
            self.center_x = self.x0 + (CELL_SIZE // 2)

        if self.center_y is None:
            self.center_y = self.y0 + (CELL_SIZE // 2)

    def fill(self, color: int = Colors.YELLOW.value, forced: bool = False):
        """
        Fills the cell's content with a specified color. It can forcibly
        fill the cell even if it is not clean.

        :param color: The color to fill the cell with.
        :param forced: If True, forces the filling even if the cell is clean.
        """
        for row in self.content:
            for idx, value in enumerate(row):
                if value == Colors.WHITE.value or forced:
                    row[idx] = color

    def __hash__(self) -> int:
        """[DEPRECATED]"""
        return hash((self.x0, self.y0))


    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @property
    def occupied(self):
        return Colors.BLACK.value in self.content or Colors.RED.value in self.content

    def __contains__(self, coords: tuple[int]):
        """
        Checks if the given coordinates lie within the boundaries of the cell.

        :param coords: The coordinates to check.
        :return: True if the coordinates are within the cell, False otherwise.
        """
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
        """
        Calculates the Manhattan distance between two cells.

        :param cell_1: The first cell.
        :param cell_2: The second cell.
        :return: The computed distance between the two cells.
        """
        return abs(cell_2.center_x - cell_1.center_x) + abs(
            cell_2.center_y - cell_1.center_y
        )


class Grid:
    _cells: Optional[list[list[Cell]]] = None
    _matrix: Optional[np.ndarray] = None

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    def load_matrix(self, matrix: np.ndarray) -> None:
        """
        Loads a matrix representation of the grid and builds the cells
        based on the matrix data.

        :param matrix: The matrix representing the grid layout.
        """
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

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    def dilate_walls(self):
        """
        Expands the walls in the grid by filling adjacent cells with a
        specific color if they are occupied.
        """
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
        """
        Retrieves the current cell in the grid based on the robot's
        coordinates from the mapping.

        :param mapping: The mapping object to use for retrieving the robot's
                        coordinates.
        :return: The current cell where the robot is located.
        """
        x, y = Navigation.get_current_map_coords(mapping)
        return self[x // CELL_SIZE, y // CELL_SIZE]

    def get_cell(self, x: int, y: int):
        """
        Retrieves a specific cell from the grid based on its pixel coordinates.

        :param x: The x-coordinate of the cell.
        :param y: The y-coordinate of the cell.
        :return: The cell corresponding to the specified coordinates.
        """
        return self[x // CELL_SIZE, y // CELL_SIZE]

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    def compute_path(self, source: Cell, target: Cell) -> list[Cell]:
        """
        Computes the shortest path from the source cell to the target cell
        using a pathfinding algorithm. Uses an A* similar approach.

        :param source: The starting cell.
        :param target: The target cell.
        :return: A list of cells representing the path, or None if no path exists.
        """
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


class Navigation:

    @staticmethod
    def get_current_map_coords(mapping: Map) -> list[float, float]:
        """
        Gets the current coordinates of the robot in the map frame.

        :param mapping: The mapping object to retrieve coordinates from.
        :return: The robot's current coordinates in the map frame.
        """
        x_robot = HAL.getPose3d().x
        y_robot = HAL.getPose3d().y
        return mapping.robot2map([x_robot, y_robot])

    @staticmethod
    def wait(seconds: float) -> Iterable:
        """
        Pauses execution for a specified number of seconds, yielding control
        back to the caller during the wait.

        :param seconds: The amount of time to wait in seconds.
        :return: An iterable that yields while waiting.
        """
        print(f"[Wait] Sleeping {seconds}s...")
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            yield
        return

    @staticmethod
    def wait_for_hal() -> Iterable:
        """
        Waits for the HAL (hardware abstraction layer) to become active,
        yielding control back to the caller while waiting.

        :return: An iterable that yields while waiting for HAL activation.
        """
        print(f"[Wait] Waiting for HAL to be active")
        while HAL.getPose3d().x == 0:
            yield
        print(f"[Wait] Wait completed")
        return

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @staticmethod
    def shortest_angle_distance_radians(a, b):
        """
        Calculates the shortest angular distance in radians between two angles.

        :param a: The first angle in radians.
        :param b: The second angle in radians.
        :return: The shortest distance in radians.
        """
        a = a % (2 * math.pi)
        b = b % (2 * math.pi)

        diff = b - a
        if diff > math.pi:
            diff -= 2 * math.pi
        elif diff < -math.pi:
            diff += 2 * math.pi

        return diff

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @staticmethod
    def navigate_to(
        mapping: Map,
        target_x: float,
        target_y: float,
        skip_rotation: bool = False,
        logging: bool = False,
        keypoints: bool = True,
    ) -> Iterable:
        """
        [Short Navigation Algorithm] Navigates the robot to a specified target
        location on the map, adjusting for rotation and movement as necessary.

        :param mapping: The mapping object to use for navigation.
        :param target_x: The target x-coordinate to navigate to.
        :param target_y: The target y-coordinate to navigate to.
        :param skip_rotation: If True, skips the rotation adjustment.
        :param logging: If True, logs navigation details to the console.
        :param keypoints: If True, marks keypoints along the path.
        :return: An iterable that yields while navigating.
        """

        current_x, current_y = Navigation.get_current_map_coords(mapping)
        current_yaw = HAL.getPose3d().yaw
        target_yaw = Navigation._compute_yaw(current_x, current_y, target_x, target_y)
        rotation = float("inf")

        # Rotate
        if not skip_rotation:
            while abs(rotation) > ROTATION_ACCURACY:
                rotation = (
                    Navigation.shortest_angle_distance_radians(current_yaw, target_yaw)
                    * ROTATION_SMOOTH
                )
                HAL.setW(rotation)
                current_yaw = HAL.getPose3d().yaw
                if logging:
                    print(
                        f"Rotating from [{current_yaw}] to [{target_yaw}]: {rotation}"
                    )
                yield

        # Forward
        error = float("inf")

        if keypoints:
            target_cell = grid.get_cell(target_x, target_y)
            target_cell.fill(Colors.VIOLET.value, forced=True)

        while error > ERROR_DISTANCE:

            current_x, current_y = Navigation.get_current_map_coords(mapping)
            current_yaw = HAL.getPose3d().yaw
            target_yaw = Navigation._compute_yaw(
                current_x, current_y, target_x, target_y
            )

            error = np.sqrt(
                (abs(target_x - current_x) ** 2) + (abs(target_y - current_y) ** 2)
            )

            velocity = min(error * NAVIGATION_SMOOTH, MAXIMUM_SPEED)
            rotation = (
                Navigation.shortest_angle_distance_radians(current_yaw, target_yaw)
                * ROTATION_SMOOTH
            )
            if abs(rotation) < ROTATION_ACCURACY:
                rotation = 0

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
        """
        [DEPRECATED] Finds the first unclean cell in the grid that is not occupied.

        :param grid: The grid to search for unclean cells.
        :return: The first unclean cell found, or None if all are clean.
        """
        for cell in grid:
            if not cell.clean and not cell.occupied:
                return cell
        else:
            return None

    @staticmethod
    def find_closest_unclean_cell(grid: Grid, mapping: Map):
        """
        Finds the closest unclean cell in the grid to the robot's current
        position.

        :param grid: The grid to search for unclean cells.
        :param mapping: The mapping object to determine the robot's position.
        :return: The closest unclean cell found, or None if none exist.
        """
        dist = float("inf")
        current_cell = grid.get_current_cell(mapping)
        result = None

        for cell in grid:
            if (
                not cell.clean
                and not cell.occupied
                and Cell.distance(current_cell, cell) < dist
            ):
                result = cell
                dist = Cell.distance(current_cell, cell)

        return result

    @staticmethod
    def get_unclean_cells(grid: Grid):
        """
        Retrieves a list of all unclean cells in the grid that are not occupied.

        :param grid: The grid to search for unclean cells.
        :return: A list of unclean cells, or None if all are clean.
        """
        result = []
        for cell in grid:
            if not cell.clean and not cell.occupied:
                result.append(cell)

        if len(result) == 0:
            return None
        return result

    @staticmethod
    def route(grid: Grid, mapping: Map, target_cell: Cell) -> Iterable:
        """
        [DEPRECATED] Executes a routing operation to navigate the robot to a target cell
        in the grid.
        Note that this function can be improved by not navigating to each cell, but
        simplifying the route first.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :param target_cell: The target cell to reach.
        :return: An iterable that yields while routing to the target cell.
        """

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
                keypoints=False,
                state=idx + 1,
            ):
                yield
                continue
            idx += 1
        return

    @staticmethod
    def _compute_yaw(x0: float, y0: float, x1: float, y1: float):
        """
        Computes the yaw angle required to face from one coordinate to another.

        :param x0: The starting x-coordinate.
        :param y0: The starting y-coordinate.
        :param x1: The target x-coordinate.
        :param y1: The target y-coordinate.
        :return: The computed yaw angle in radians.
        """
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
        """
        [DEPRECATED] Implements the BSA algorithm to navigate
        through the grid while performing cleaning operations.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :return: An iterable that yields during the BSA process.
        """

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
                unblocking_cell = Navigation.find_closest_unclean_cell(grid, mapping)
                start_time = time.perf_counter()
                print(
                    f"[BSA] Found new cell at {[unblocking_cell.center_x, unblocking_cell.center_y]}"
                )
                blocked = False

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
                if not current_cell.right.occupied and not current_cell.right.clean:
                    next_cell = current_cell.right

                elif not current_cell.top.occupied and not current_cell.top.clean:
                    next_cell = current_cell.top

                elif not current_cell.bottom.occupied and not current_cell.bottom.clean:
                    next_cell = current_cell.bottom

                elif not current_cell.left.occupied and not current_cell.left.clean:
                    next_cell = current_cell.left
                else:
                    next_cell = None

                state += 1
                start_time = time.perf_counter()

                # If there is no valid cell, you are blocked
                if next_cell is None:
                    print("[BSA] Room completed, searching for new rooms...")
                    blocked = True
                    temp_proccess_manager.flush()
                    continue

            if time.perf_counter() - start_time > RECOVERY_TIME:
                start_time = time.perf_counter()
                print("[BSA] Navigation failure, starting recovery")
                blocked = True
                temp_proccess_manager.flush()
                HAL.setV(0)
                time.sleep(1)

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

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @staticmethod
    def bsa_v2(grid: Grid, mapping: Map) -> Iterable:
        """
        An improved version of the BSA algorithm that incorporates planning
        and navigation to unclean cells in a more efficient manner.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :return: An iterable that yields during the BSA process.
        """
        temp_proccess_manager = ProcessManager()
        path: list[Cell | Literal["block", "plan"]] = ["plan"]

        # We select a number for incrementing the states of the proccess manager
        # each time a new cell is selected
        state = 0

        while True:

            yield

            current_cell = grid.get_current_cell(mapping)

            if path[0] == "plan":
                print("[BSA] Planning...")
                path = []
                start_time = time.perf_counter()
                state = 1
                temp_proccess_manager.flush()
                plan = Navigation.bsa_planification(grid, mapping)
                if plan is not None:
                    path = Navigation.simplify_route(plan)
                path.append("block")

                if plan is not None:
                    for cell in plan:
                        if isinstance(cell, Cell):
                            cell.fill(Colors.YELLOW.value, forced=True)

            if path[0] == "block":
                print("[BSA] Robot Blocked. Searching a new cell...")
                start_time = time.perf_counter()
                target_cell = Navigation.find_closest_unclean_cell(grid, mapping)

                if target_cell is None:
                    print("[BSA] Algorithm completed, exit.")
                    break

                path_to_target = grid.compute_path(current_cell, target_cell)
                path = Navigation.simplify_route(path_to_target)
                path.append("plan")

                for cell in path_to_target:
                    if isinstance(cell, Cell):
                        cell.fill(Colors.YELLOW.value, forced=True)

                print(
                    f"[BSA] Found new cell at {[target_cell.center_x, target_cell.center_y]}"
                )

            if time.perf_counter() - start_time > RECOVERY_TIME:
                print("[BSA] Navigation failure, starting recovery")
                HAL.setV(0)
                path = ["block"]
                continue

            if temp_proccess_manager.running(
                Navigation.navigate_to,
                mapping=mapping,
                target_x=path[0].center_y,
                target_y=path[0].center_x,
                state=state,
            ):
                continue

            start_time = time.perf_counter()
            state += 1
            path[0].clean = True
            path.pop(0)

        return

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @staticmethod
    def bsa_planification(grid: Grid, mapping: Map) -> Optional[list[Cell]]:
        """
        Plans a route through the grid by identifying unclean cells and
        returning a list of cells to visit.

        :param grid: The grid to plan the route through.
        :param mapping: The mapping object to use for navigation.
        :return: A list of cells to visit, or None if no route can be planned.
        """
        result: list[Cell] = []
        current_cell = grid.get_current_cell(mapping)

        # Computing BSA search
        while True:

            if not current_cell.right.occupied and not current_cell.right.clean:
                result.append(current_cell.right)
                current_cell = result[-1]
                current_cell.clean = True
                continue

            elif not current_cell.top.occupied and not current_cell.top.clean:
                result.append(current_cell.top)
                current_cell = result[-1]
                current_cell.clean = True
                continue

            elif not current_cell.bottom.occupied and not current_cell.bottom.clean:
                result.append(current_cell.bottom)
                current_cell = result[-1]
                current_cell.clean = True
                continue

            elif not current_cell.left.occupied and not current_cell.left.clean:
                result.append(current_cell.left)
                current_cell = result[-1]
                current_cell.clean = True
                continue
            else:
                break

        if len(result) == 0:
            return None

        for cell in result:
            cell.clean = False

        return result

    # ===================================
    # 🔥 >>>>    CORE FUNCTION    <<<< 🔥
    # ===================================
    @staticmethod
    def simplify_route(path: list[Cell]):
        """
        Simplifies a given route by removing unnecessary intermediate cells.

        :param path: The original list of cells representing a route.
        :return: A simplified list of cells with unnecessary cells removed.
        """

        directions: list[Literal["right", "left", "top", "bottom"]] = []
        result = path.copy()

        for i in range(len(path) - 1):

            if path[i + 1] is path[i].right:
                directions.append("right")
            elif path[i + 1] is path[i].top:
                directions.append("top")
            elif path[i + 1] is path[i].left:
                directions.append("left")
            else:
                directions.append("bottom")

        for i in range(1, len(path) - 2, DISTANCE_RECOMPUTE):
            if directions[i] == directions[i - 1]:
                result[i] = None

        return [cell for cell in result if cell is not None]


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
print("\n" * 10)
print("=" * 30 + " BSA " + "=" * 30)

processManager = ProcessManager()
mapping = Map(
    "/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png"
)
mapping.define_transformation(M)

grid = Grid()
grid.load_matrix(mapping.map)
grid.dilate_walls()

# ================ MAIN =================
while True:
    pass

    mapping.show()
    x, y = Navigation.get_current_map_coords(mapping)
    current_cell = grid.get_current_cell(mapping)
    current_cell.clean = True

    # Activate me for adding the exact keypoints where the robot is cleaning.
    # Deactivated by default
    if EXACT_TRACE:
        mapping.add_keypoint(x, y, Colors.BLUE.value)
    else:
        current_cell.fill(Colors.BLUE.value, forced=True)

    if processManager.running(Navigation.wait_for_hal, state=1):
        continue

    # Use this as an example navigation test
    # if processManager.running(Navigation.route, grid, mapping, grid[8, 12], state=2):
    #     continue

    if processManager.running(Navigation.bsa_v2, grid, mapping, state=3):
        continue

# Version: 011
