"""
Practice 1 - Blind Vacuum Cleaner
==========================================

[IMPORTANT] This Python script has been created from scratch by Sam, with appropriate
references where necessary. Docstrings have been generated with the assistance of
AI for clarity and documentation purposes.
"""
import GUI  # noqa
import HAL  # noqa
import random
from typing import Iterable
from typing import Callable
from typing import Optional
from enum import Enum, auto
import time

# =====================================================
# Constants
# =====================================================

TURNBACK_TIME = 7
TURN_VELOCITY = 0.5
LINEAR_VELOCITY = 1


class Bumper(Enum):
    NORMAL: int = 0
    CRASH: int = 1

# =====================================================
# Process Manager
# =====================================================

class ProcessManager:

    current_task: Optional[Callable] = None
    state: int = 0
    _edging: bool = True
    _edge_calls: list[Callable] = []
    _force_next_state: bool = False

    def running(
        self,
        func: Callable,
        *args,
        state: float = float("-inf"),
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
            print(f"State {States(state).name} joined")
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

# =====================================================
# Navigation Utils
# =====================================================

class Navigation:

    @staticmethod
    def rotate(degrees: float, forward: float = 0) -> Iterable:
        """
        Rotates the robot by a specified number of degrees while
        optionally moving forward.

        :param degrees: The angle in degrees to rotate.
        :param forward: Optional forward movement speed during rotation.

        [Compatible with ProcessManager]
        """
        timer = abs(degrees) * TURNBACK_TIME / 180
        return Navigation._rotate_time(timer, forward, invert=(degrees < 0))

    @classmethod
    def _rotate_time(cls, timer, forward, invert: bool = False) -> Iterable:
        """
        Handles the actual rotation logic for a specified duration. 
        Sets the robot's velocity based on the direction of rotation 
        and manages the yield control until the rotation is complete.

        :param timer: The time duration for the rotation.
        :param forward: The forward speed during rotation.
        :param invert: Whether to reverse the turning direction.
        :return: An iterable that yields control during the rotation.
        """

        start = time.perf_counter()
        turn_vel = -TURN_VELOCITY if invert else TURN_VELOCITY

        while time.perf_counter() - start < timer:
            if forward != 0:
                HAL.setV(forward)
            HAL.setW(turn_vel)
            yield

        HAL.setV(0)
        HAL.setW(0)
        return

    @staticmethod
    def wait(seconds: float) -> Iterable:
        """
        Makes the robot wait for a specified number of seconds, 
        yielding control during the wait time. This can be useful 
        for timing operations or delays in task execution.

        :param seconds: The time to wait in seconds.
        :return: An iterable that yields control during the wait.
        """

        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            yield

        return

    @staticmethod
    def forward(seconds: Optional[float] = None, invert: bool = False) -> Iterable:
        """
        Moves the robot forward at a constant speed for a specified
        duration. If no duration is provided, the robot continues
        moving until stopped.

        :param seconds: Optional duration to move forward.
        :param invert: Whether to move in the reverse direction.
        :return: An iterable that yields control during the movement.
        """

        vel = -LINEAR_VELOCITY if invert else LINEAR_VELOCITY

        if seconds is None:
            while True:
                HAL.setV(vel)
                yield

        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            HAL.setV(vel)
            yield

        HAL.setV(0)
        return


# =====================================================
# Initialization
# =====================================================

print("=" * 20 + " START " + "=" * 30)
HAL.setV(0)
HAL.setW(0)

processManager = ProcessManager()
class States(Enum):
    RETRY: int = auto()
    SEARCH_WALL: int = auto()
    RETURN: int = auto()
    ADVANCE: int = auto()
    TURN: int = auto()


# =====================================================
# Main Loop
# =====================================================
rotation = 180
retrying = False
while True:
    """
    This section implements the state graph defined at: https://samthinkgit.github.io/mobile-robotics-blog/.
    For more information read the documentations at the main blog.
    """

    if retrying:
        if processManager.running(Navigation.rotate, degrees=rotation, state=States.RETRY):
            continue
        else:
            retrying = False

    # Search Wall
    if processManager.running(Navigation.forward, state=States.SEARCH_WALL):
        if HAL.getBumperData().state == Bumper.CRASH.value:
            current_dist = 0.3
            processManager.set_next_state()
            HAL.setV(0)
        continue

    # Backward Some Meters
    if processManager.running(Navigation.forward, seconds=1.5, invert=True, state=States.RETURN):
        continue

    # Forward
    if processManager.running(Navigation.forward, seconds=current_dist, state=States.ADVANCE):
        if HAL.getBumperData().state == Bumper.CRASH.value:
            processManager.flush()
            retrying = True
            rotation = random.randint(120, 230)
            HAL.setV(0)
        continue

    if processManager.edging(state=States.ADVANCE.value):
        current_dist += 0.15

    # Rotate
    if processManager.running(Navigation.rotate, degrees=90, state=States.TURN):
        continue

    processManager.change_state(States.ADVANCE.value - 1)
