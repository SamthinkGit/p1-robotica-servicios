import GUI  # noqa
import HAL  # noqa
import numpy as np
from enum import Enum
import math
import cv2

ROTATION = math.pi
SHIFT_X = 100
SHIFT_Y = -100


class Colors(Enum):
    BLACK: int = 127
    RED: int = 128
    YELLOW: int = 130
    GREEN: int = 131
    VIOLET: int = 134
    WHITE: int = 0


class Map:

    _robot2map_mat: np.ndarray = None

    def __init__(self, path: str) -> None:
        rgb = GUI.getMap(path)
        self.map: np.ndarray = np.where(
            np.sum(rgb, axis=2) > 0, Colors.BLACK.value, Colors.WHITE.value
        )
        self._path = path

    def define_transformation(self, robot2map_mat: np.ndarray):
        self._robot2map_mat = robot2map_mat

    def robot2map(self, coords: np.ndarray) -> list[int]:
        result = self._robot2map_mat @ coords
        return [int(result[0] + SHIFT_X), int(result[1] + SHIFT_Y)]

    def flush(self):
        self.__init__(self._path)

    def show(self):
        GUI.showNumpy(self.map)

    def add_keypoint(
        self, x: int, y: int, color: int = Colors.GREEN.value, size: int = 10
    ):
        self.map[y - size : y + size, x - size : x + size] = color  # noqa


# =========== Defining Coordinate Space =================
robot_points = np.array(
    [[5.1316, 5.5555], [-2.527, 5.5310], [-0.9999, 1.500]], dtype=np.float32
)
map_points = np.array([[48, 996], [850, 956], [570, 670]], dtype=np.float32)
M = cv2.getAffineTransform(robot_points, map_points)

# =========== Initialization =================
mapping = Map(
    "/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png"
)
mapping.define_transformation(M)


# =========== Main Loop =================
print("=" * 30 + " STARTING " + "=" * 30)
while True:
    x, y = [HAL.getPose3d().x, HAL.getPose3d().y]
    robot_cords = np.array([x, y, 1]).reshape(-1, 1)
    x_map, y_map = mapping.robot2map(robot_cords)
    print(f"Current coords: ({x_map}, {y_map})")
    mapping.add_keypoint(x_map, y_map)
    mapping.show()
    mapping.flush()
