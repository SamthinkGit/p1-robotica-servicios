import numpy as np

array = np.array(
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [255, 255, 255], [0, 0, 0]],
        [[0, 0, 0], [255, 255, 255], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ]
)

if __name__ == "__main__":
    print(np.where(np.sum(array, axis=2) > 0, 255, 0))
    # values = map_mock[np.where]
    # print(values)
