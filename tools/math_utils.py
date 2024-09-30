import numpy as np
import cv2


if __name__ == "__main__":

    points_A = np.array([[5.1316, 5.5555], [-2.527, 5.5310], [0, 0]], dtype=np.float32)
    points_B = np.array([[996, 48], [996, 850], [670, 570]], dtype=np.float32)

    M = cv2.getAffineTransform(points_A, points_B)

    example_coordinate = np.array([5.13, 5.5555, 1]).reshape(-1, 1)
    REFLECT = np.array([[-1, 0], [0, 1]])

    coords = M @ example_coordinate
    sanitization = coords.transpose() @ REFLECT

    print(np.round(sanitization, 1))
