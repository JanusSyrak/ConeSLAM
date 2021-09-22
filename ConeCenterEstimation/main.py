import os
import open3d as o3d
import cv2
import numpy as np
from numpy import genfromtxt
import math
import time

from measurement import *

from matplotlib import pyplot as plt

def get_cone_centers():
    reference_pt = np.array([-0.919675, 9.72914, 0.17058])
    reference_angle = math.atan2(reference_pt[1],reference_pt[0]) - math.pi/2
    #0: background, 1: blue-cone, 2: end-cone, 3: start-cone, 4: yellow-cone

    ground_truth = np.arange(1, 10, 0.2)

    csv_path = "../data/csv_airport"
    
    for file in sorted(os.listdir(csv_path)):
        start = time.time()
        indx = file[:-4]
        
        measurement = Measurement(indx)
        measurement.projectLidarToImage()
        measurement.calculateCenterOfCones()
        centers = measurement.data3D.center_cones
        print(indx)
        print(centers)
        f = open("C:/Users/stebb/OneDrive/Desktop/skoli/mast/SLAM/fs-SLAM/data/cone_centers_airport_valid_lenscanline_timingtest.txt", "a")

        for cones in centers:
            f.write(str(cones[0]) + " " + str(cones[1]) + " " + str(cones[2]) + " " + str(int(indx) / 4) + "\n")
        #    f.write(str(cones[0]) + " " + str(cones[1]) + " " + str(cones[2]) + " " + str(int(indx) / 1) + "\n")

        end = time.time()
        t = open("C:/Users/stebb/OneDrive/Desktop/skoli/mast/SLAM/fs-SLAM/data/sync_timing.txt", "a")
        t.write(str(end-start) + " " + str(len(centers)) + "\n")

def main():
    csv_path = "../data/csv"
    get_cone_centers()

if __name__ == '__main__':
    main()

