import cv2
import numpy as np

class ConeMask:
    def __init__(self, path, cam_params):
        self.cam_params = cam_params
        self.mask = self.getMask(path)

    def showMask(self):
        maskvis = self.mask * 255/5
        cv2.imshow("Current Mask", maskvis)
        cv2.waitKey(0)

    def getMask(self, path):
        mask = cv2.imread(path, 0)
        mask = self.rectify_image(mask)
        return mask

    def rectify_image(self, mask):
        tang_dist = self.cam_params.tang_dist
        radial_dist = self.cam_params.radial_dist
        K = self.cam_params.K

        # OpenCV takes radial and tangential distortion in as a vector
        dist_vector = np.asarray([radial_dist[0], radial_dist[1], tang_dist[0], tang_dist[1]])
        img_rectified = cv2.undistort(mask, K, dist_vector)
        return img_rectified

        
