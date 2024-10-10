```python
"""
BSA (Behavioral State Automation)
=================================

This module implements a Behavioral State Automation (BSA) system for a robotic
vacuum cleaner. It provides classes and functions to manage the robot's process
states, map the environment, and navigate through a grid-based representation
of the area. The system supports various navigation strategies, including pathfinding
and state management during cleaning operations.

The module relies on external libraries for graphical interface (GUI) and hardware
abstraction layer (HAL) to interact with the robot's sensors and actuators. It
utilizes concepts from computer science and robotics, such as state machines,
grid navigation algorithms, and data structures to facilitate efficient movement
and task execution.
"""

    @staticmethod
    def running(
        self,
        func: Callable,
        *args,
        state: float = 0,
        **kwargs,
    ) -> bool:
        """
        Executes a given function as a task if the current state allows it to
        start. It manages task execution and state transitions based on the
        current process state.

        :param func: The function to be executed as a task.
        :param args: Arguments to pass to the task function.
        :param state: The state required to run the task.
        :return: True if the task is running, False otherwise.
        """
    
    def edging(self, state: Optional[int] = None) -> bool:
        """
        Checks if the process manager is currently edging, which means
        it is transitioning between states. It can also check for a
        specific state if provided.

        :param state: An optional state to check against the current state.
        :return: True if edging is active for the given state, otherwise False.
        """

    def log(self):
        """
        Outputs the current state, task, and edging status of the
        process manager for debugging purposes. This includes information
        about the current task being executed and the active state.
        """

    def at_edge(self, func: Callable) -> None:
        """
        Registers a callback function to be executed when the state
        transitions to an edge state. This allows for custom actions
        during state changes.

        :param func: The callback function to be called at the edge.
        """

    def set_next_state(self):
        """
        Sets the next state to be executed and triggers any registered
        edge callbacks. This method is called to advance the state of
        the process manager.
        """

    def change_state(self, state: int | Enum):
        """
        Changes the current state of the process manager. This allows
        for the state to be updated based on the robot's current
        operational context.

        :param state: The new state to transition to, which can be an
                      integer or an enumeration.
        """

    def flush(self) -> None:
        """
        Resets the current task and state of the process manager. This
        effectively clears the task queue and prepares the manager for
        a new set of tasks.
        """

    def define_transformation(self, robot2map_mat: np.ndarray):
        """
        Defines the transformation matrix that relates robot coordinates
        to map coordinates.

        :param robot2map_mat: The transformation matrix to be applied.
        """

    def robot2map(self, coords: np.ndarray) -> list[int]:
        """
        Transforms robot coordinates into map coordinates using the defined
        transformation matrix.

        :param coords: The coordinates of the robot in its own frame.
        :return: The corresponding coordinates in the map frame.
        """

    def flush(self):
        """
        Resets the map to its initial state by reloading it from the 
        specified path.
        """

    def show(self):
        """
        Displays the current map using the graphical interface.
        """

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

    def __post_init__(self):
        """
        Post-initialization method that calculates the center coordinates
        of the cell if they are not provided.
        """

    def fill(self, color: int = Colors.YELLOW.value, forced: bool = False):
        """
        Fills the cell's content with a specified color. It can forcibly
        fill the cell even if it is not clean.

        :param color: The color to fill the cell with.
        :param forced: If True, forces the filling even if the cell is clean.
        """

    @property
    def occupied(self):
        """
        Checks if the cell is occupied by any object, indicated by
        specific colors in its content.

        :return: True if the cell is occupied, False otherwise.
        """

    def __contains__(self, coords: tuple[int]):
        """
        Checks if the given coordinates lie within the boundaries of the cell.

        :param coords: The coordinates to check.
        :return: True if the coordinates are within the cell, False otherwise.
        """

    def __lt__(self, other: "Cell") -> bool:
        """
        Less-than comparison method for sorting cells, based on their IDs.

        :param other: The other cell to compare with.
        :return: True if this cell's ID is less than the other cell's ID.
        """

    def __eq__(self, other):
        """
        Equality comparison method to check if two cells are the same.

        :param other: The other cell to compare with.
        :return: True if both cells are equal, False otherwise.
        """

    @classmethod
    def distance(cls, cell_1: "Cell", cell_2: "Cell"):
        """
        Calculates the Manhattan distance between two cells.

        :param cell_1: The first cell.
        :param cell_2: The second cell.
        :return: The computed distance between the two cells.
        """

    def load_matrix(self, matrix: np.ndarray) -> None:
        """
        Loads a matrix representation of the grid and builds the cells
        based on the matrix data.

        :param matrix: The matrix representing the grid layout.
        """

    def dilate_walls(self):
        """
        Expands the walls in the grid by filling adjacent cells with a
        specific color if they are occupied.
        """

    def get_current_cell(self, mapping: Map):
        """
        Retrieves the current cell in the grid based on the robot's
        coordinates from the mapping.

        :param mapping: The mapping object to use for retrieving the robot's
                        coordinates.
        :return: The current cell where the robot is located.
        """

    def get_cell(self, x: int, y: int):
        """
        Retrieves a specific cell from the grid based on its pixel coordinates.

        :param x: The x-coordinate of the cell.
        :param y: The y-coordinate of the cell.
        :return: The cell corresponding to the specified coordinates.
        """

    def compute_path(self, source: Cell, target: Cell) -> list[Cell]:
        """
        Computes the shortest path from the source cell to the target cell
        using a pathfinding algorithm.

        :param source: The starting cell.
        :param target: The target cell.
        :return: A list of cells representing the path, or None if no path exists.
        """

    def __getitem__(self, idx: tuple[int]) -> Cell | list[Cell]:
        """
        Gets a cell or a list of cells from the grid using a double index.

        :param idx: The index to access the grid cells.
        :return: The cell at the specified index.
        :raises ValueError: If the index is not a double index.
        """

    def __iter__(self) -> Iterator[Cell]:
        """
        Iterates over the cells in the grid for easy traversal.

        :return: An iterator over the cells in the grid.
        """

    @staticmethod
    def get_current_map_coords(mapping: Map) -> list[float, float]:
        """
        Gets the current coordinates of the robot in the map frame.

        :param mapping: The mapping object to retrieve coordinates from.
        :return: The robot's current coordinates in the map frame.
        """

    @staticmethod
    def wait(seconds: float) -> Iterable:
        """
        Pauses execution for a specified number of seconds, yielding control
        back to the caller during the wait.

        :param seconds: The amount of time to wait in seconds.
        :return: An iterable that yields while waiting.
        """

    @staticmethod
    def wait_for_hal() -> Iterable:
        """
        Waits for the HAL (hardware abstraction layer) to become active,
        yielding control back to the caller while waiting.

        :return: An iterable that yields while waiting for HAL activation.
        """

    @staticmethod
    def shortest_angle_distance_radians(a, b):
        """
        Calculates the shortest angular distance in radians between two angles.

        :param a: The first angle in radians.
        :param b: The second angle in radians.
        :return: The shortest distance in radians.
        """

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
        Navigates the robot to a specified target location on the map,
        adjusting for rotation and movement as necessary.

        :param mapping: The mapping object to use for navigation.
        :param target_x: The target x-coordinate to navigate to.
        :param target_y: The target y-coordinate to navigate to.
        :param skip_rotation: If True, skips the rotation adjustment.
        :param logging: If True, logs navigation details to the console.
        :param keypoints: If True, marks keypoints along the path.
        :return: An iterable that yields while navigating.
        """

    @staticmethod
    def find_first_unclean_cell(grid: Grid):
        """
        Finds the first unclean cell in the grid that is not occupied.

        :param grid: The grid to search for unclean cells.
        :return: The first unclean cell found, or None if all are clean.
        """

    @staticmethod
    def find_closest_unclean_cell(grid: Grid, mapping: Map):
        """
        Finds the closest unclean cell in the grid to the robot's current
        position.

        :param grid: The grid to search for unclean cells.
        :param mapping: The mapping object to determine the robot's position.
        :return: The closest unclean cell found, or None if none exist.
        """

    @staticmethod
    def get_unclean_cells(grid: Grid):
        """
        Retrieves a list of all unclean cells in the grid that are not occupied.

        :param grid: The grid to search for unclean cells.
        :return: A list of unclean cells, or None if all are clean.
        """

    @staticmethod
    def route(grid: Grid, mapping: Map, target_cell: Cell) -> Iterable:
        """
        Executes a routing operation to navigate the robot to a target cell
        in the grid.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :param target_cell: The target cell to reach.
        :return: An iterable that yields while routing to the target cell.
        """

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

    @staticmethod
    def bsa(grid: Grid, mapping: Map) -> Iterable:
        """
        Implements the Behavioral State Automation algorithm to navigate
        through the grid while performing cleaning operations.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :return: An iterable that yields during the BSA process.
        """

    @staticmethod
    def bsa_v2(grid: Grid, mapping: Map) -> Iterable:
        """
        An improved version of the BSA algorithm that incorporates planning
        and navigation to unclean cells in a more efficient manner.

        :param grid: The grid to navigate through.
        :param mapping: The mapping object to use for navigation.
        :return: An iterable that yields during the BSA process.
        """

    @staticmethod
    def bsa_planification(grid: Grid, mapping: Map) -> Optional[list[Cell]]:
        """
        Plans a route through the grid by identifying unclean cells and
        returning a list of cells to visit.

        :param grid: The grid to plan the route through.
        :param mapping: The mapping object to use for navigation.
        :return: A list of cells to visit, or None if no route can be planned.
        """

    @staticmethod
    def simplify_route(path: list[Cell]):
        """
        Simplifies a given route by removing unnecessary intermediate cells.

        :param path: The original list of cells representing a route.
        :return: A simplified list of cells with unnecessary cells removed.
        """
```