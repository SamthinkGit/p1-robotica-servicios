import GUI
import HAL
import numpy as np
from enum import Enum


class Colors(Enum):
    BLACK: int = 127
    WHITE: int = 0


class Map:

    def __init__(self, path: str) -> None:
        rgb = GUI.getMap(path)
        self.map = np.where(
            np.sum(rgb, axis=2) > 0, Colors.BLACK.value, Colors.WHITE.value
        )

    def show(self):
        GUI.showNumpy(self.map)


mapman = Map(
    "/RoboticsAcademy/exercises/static/exercises/vacuum_cleaner_loc_newmanager/resources/images/mapgrannyannie.png"
)
mapman.show()

while True:
    # Enter iterative code!
    HAL.setV(0.2)
