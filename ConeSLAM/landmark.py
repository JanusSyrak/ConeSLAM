import numpy as np
import math


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Landmark:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.cov = np.identity(2) * 0.1
        self.colorcount = [0, 0, 0, 0, 0]

        self.classification_arr = []    # Keeps track of distances and class [class, meters]


    def addClassification(self, color_index, distance):
        self.classification_arr.append([color_index, distance])

    def XY(self):
        return np.array([self.x, self.y])

    def updatePosition(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    def getRBA(self, state):
        dx = self.x - state[0, 2]
        dy = self.y - state[1, 2]
        d = math.sqrt((dx ** 2) + (dy ** 2))

        yaw = math.acos(state[0, 0])
        zHat = np.array(
            [d,
             pi_2_pi(math.atan2(dy, dx) - yaw)])

        return zHat

