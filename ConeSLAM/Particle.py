import numpy as np
import math


class Particle:
    def __init__(self, LM_SIZE=2, w=1):
        self.x = 0
        self.y = 0
        self.yaw = math.pi/2
        self.cov = np.identity(3)
        self.lm = np.empty((0, LM_SIZE))
        self.transformations = []
        self.P = np.eye(3)

        self.w = w

    def print(self):
        print("[", str(self.x), ", ", str(self.y), ", ", str(self.yaw), "]")
        print("Particle weight: " + str(self.w))
        print("Printing particle landmarks: ")
        for idx, lm in enumerate(self.lm):
            print("landmark ", str(idx), ": (", str(lm.x), ",", str(lm.y), ")")


    def getHomogeneousTransform(self):
        T = np.array([[math.cos(self.yaw), -math.sin(self.yaw), self.x],
                     [math.sin(self.yaw), math.cos(self.yaw), self.y],
                     [0, 0, 1]])
        return T

    # Input: a single measurement, z (range, angle, azimuth)
    # Output: get the absolute position of the observed measurement
    def getAbsolutePosition(self, z):
        T = self.getHomogeneousTransform()
        # x and y relative to the particle
        x = z[0] * math.cos(z[1])
        y = z[0] * math.sin(z[1])

        obs_homog = np.array([[1, 0, x],
                             [0, 1, y],
                             [0, 0, 1]])

        homog_abs = T @ obs_homog

        return np.transpose(homog_abs[:2, 2])

    def updateState(self, transform):
        self.s = transform
        self.x = transform[0, 3]
        self.y = transform[1, 3]
        self.yaw = math.acos(transform[0, 0])
