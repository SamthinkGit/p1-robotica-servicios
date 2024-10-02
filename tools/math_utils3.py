import math


def shortest_angle_distance_radians(a, b):
    a = a % (2 * math.pi)
    b = b % (2 * math.pi)

    diff = b - a
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -math.pi:
        diff += 2 * math.pi

    return -diff


if __name__ == "__main__":
    print(shortest_angle_distance_radians(math.pi, -2.78))
    print(shortest_angle_distance_radians(math.pi, 2.78))

    print(shortest_angle_distance_radians(0, -1))
    print(shortest_angle_distance_radians(0, 1))