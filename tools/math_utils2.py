import numpy as np


def get_degrees(x0, y0, x1, y1):
    x = x1 - x0
    y = y1 - y0

    rad = -(np.arctan2(x, y) + np.pi / 2)

    if rad < -np.pi:
        rad += 2*np.pi

    return np.rad2deg(rad)
    
def _compute_yaw(x0: float, y0: float, x1: float, y1: float):
    x = x1 - x0
    y = y1 - y0
    rad = -(np.arctan2(x, y) + np.pi / 2)
    if rad < -np.pi:
        rad += 2 * np.pi
    return rad

if __name__ == "__main__":
    a = _compute_yaw(0, 0, 10, 10)
    b = _compute_yaw(0, 0, -10, 10)
    c = _compute_yaw(0, 0, -10, -10)
    d = _compute_yaw(0, 0, 10, -10)

    print(f"-135º: {a} ")
    print(f"-45º: {b} ")
    print(f"45º: {c} ")
    print(f"135º: {d} ")
