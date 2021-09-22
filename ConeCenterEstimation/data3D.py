import open3d as o3d
import math
import numpy as np

class Data3D:
    def __init__(self, data, fov):
        self.angles = data[:, -1]
        self.points = data[:, :3]
        self.fov = fov
        self.clustered_cones = []
        self.filtered_points = []
        self.center_cones = []
        self.class_indices = []
    
    def filterFOV(self):
        points = self.points
        fov = self.fov
        fov_x = fov[0]
        fov_y = fov[1]
        filtered_pts = []
        for idx, pt in enumerate(points):
            bearing = math.atan2(pt[0], pt[1])
            azimuth = math.atan(math.sqrt(pt[0] ** 2 + pt[1] ** 2)/ pt[2])
            if abs(bearing) < fov_x/2 and abs(azimuth) - math.pi/2 < fov_y/2:
                filtered_pts.append(np.array([pt[0], pt[1], pt[2]]))

        return np.array(filtered_pts)
