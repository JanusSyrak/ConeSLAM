from numpy import genfromtxt
import numpy as np
import os

class CamParams:
    def __init__(self, path = "..\calib"):
        self.path = path
        self.K = self.getK()
        self.P = self.getP()
        self.fov = self.getFov()
        self.radial_dist = self.getRadial()
        self.tang_dist = self.getTangential()

    def getRadial(self):
        radial_dist = np.array([-0.2410, 0.0931])
        return radial_dist

    def getTangential(self):
        tang_dist = np.array([0.002, -0.0007])
        return tang_dist

    def getK(self):
        path = os.path.join(self.path, "K_airport.csv")
        K = genfromtxt(path, delimiter=",")
        return K
    
    def getP(self):
        path = os.path.join(self.path, "projection_airport.csv")
        K = genfromtxt(path, delimiter=",")
        return K

    def getFov(self):
        path = os.path.join(self.path, "fov_new.csv")
        K = genfromtxt(path, delimiter=",")
        return K
