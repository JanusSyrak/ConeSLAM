
from cone_mask import *
from cam_params import *
from data3D import *
from matplotlib import pyplot as plt
import open3d as o3d
from copy import deepcopy

class Measurement:
    def __init__(self, indx):
        self.indx = indx
        self.mask_path = "../data/masks_airport/"
        self.img_path = "../data/imgs_airport/"
        self.pcd_path = "../data/pcds/"
        self.csv_path = "../data/csv_airport/"
        self.image_type = ".jpg"
        self.cam_params = self.getCamParams()
        self.data3D = self.getPCDFromCSV()
        
        self.mask = self.getMask()
        self.clusters = []

    def getCamParams(self):
        cam_params = CamParams()
        return cam_params

    def getImage(self):
        return cv2.imread(self.img_path + str(self.indx) + self.image_type)


    def showImage(self):
        im = self.getImage()
        cv2.imshow("image: " + str(self.indx), im)
        cv2.waitKey()

    def getPCDFromCSV(self):
        data = np.genfromtxt(os.path.join(self.csv_path, str(self.indx) + ".csv"), delimiter=',')
        fov = self.cam_params.fov
        return Data3D(data[1:,:], fov)

    def getMask(self):
        path = os.path.join(self.mask_path, str(int(self.indx) + 0))
        path = path + self.image_type
        mask = ConeMask(path, self.cam_params)
        return mask

    def makeHomogenious(self, pts):
        points = np.asarray(pts)
        homo_points = np.hstack([points,np.ones((points.shape[0], 1))])
        return homo_points

    def transformPoints(self, pts):
        transformed_points = np.transpose(self.cam_params.P @ np.transpose(pts))
        return transformed_points

    def filterPointsOutsideImg(self, pcd, dim):
        arr = []    # Array of all the points in the point cloud that are on the image
        indx = []   # Array of all the indices on the original point cloud that are within the image
        for idx, pc in enumerate(pcd):
            pc[0:3] = pc[0:3]/pc[2]
            if pc[0] < 0 or pc[1] < 0:
                continue

            xp = int(pc[0])
            yp = int(pc[1])
            
            if xp < 0 or xp >= dim[1] or yp < 0 or yp >= dim[0]:
                continue
            arr.append([xp, yp])
            indx.append(idx)
        return np.asarray(arr), indx

    def filterRange(self, arr, max_range):
        filtered_points = []
        for pt in arr:
            dist = math.sqrt(pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)
            if dist < max_range:
                filtered_points.append(pt)
        
        return filtered_points

    def sortPoints(self, points):
        new_points = []
        for point in points:
            bearing = self.calculateBearing(point)
            azimuth = self.calculateAzimuth(point)
            new_points.append(np.array([point[0], point[1], point[2], bearing, azimuth]))

        new_points = np.asarray(new_points)
        new_points = new_points[np.lexsort((new_points[:,4], new_points[:,3]))]
        return new_points

    def projectLidarToImage(self):

        points = self.data3D.points
        # sorted_points are all the LiDAR points sorted, to make the reconstruction easier
        sorted_points = self.sortPoints(points)
        self.data3D.points = sorted_points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        mask = self.mask.mask
        filtered_fov = self.data3D.filterFOV()
        
        max_range = 14
        filtered_fov = self.filterRange(filtered_fov, max_range)
        f = open("number_of_points.txt", "a")
        f.write(str(len(filtered_fov)) + " ")

        homogenious_pts = self.makeHomogenious(filtered_fov)
        transformed_points = self.transformPoints(homogenious_pts)   # 2D transformed points from the point cloud
        transformed_points, filtered_indx = self.filterPointsOutsideImg(transformed_points, mask.shape)

        colored_3d = np.empty((0,3))
        colored_2d = []                 # 2D locations of points that overlap with the masks
        all_3d_in_2d = []               # All 3D lidar points warped in 2D
        color_array = np.zeros((points.shape))
        class_indices = []              # Classes of the points in color_3D array
        cone_color = [] 
        for i, pc in enumerate(transformed_points):
            xp = pc[0]
            yp = pc[1]

            all_3d_in_2d.append((xp,yp))
            if mask[yp, xp] == 0: 
                continue

            colored_2d.append((xp, yp))
            cone_color.append(mask[yp,xp])

            if mask[yp,xp] == 1:
                colored_3d = np.append(colored_3d, [filtered_fov[filtered_indx[i]]], axis = 0)
                color_array[filtered_indx[i], :] = [1, 0.3, 0]
                class_indices.append(1)
            elif mask[yp,xp] == 2:
                colored_3d = np.append(colored_3d, [filtered_fov[filtered_indx[i]]], axis = 0)
                color_array[filtered_indx[i], :] = [1, 0, 1]
                class_indices.append(2)
            elif mask[yp, xp] == 3:
                colored_3d = np.append(colored_3d, [filtered_fov[filtered_indx[i]]], axis = 0)
                color_array[filtered_indx[i], :] = [0, 0, 1]
                class_indices.append(3)
            elif mask[yp, xp] == 4:
                colored_3d = np.append(colored_3d, [filtered_fov[filtered_indx[i]]], axis = 0)
                color_array[filtered_indx[i], :] = [0, 1, 1]
                class_indices.append(4)

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(colored_3d)
        self.clusterPoints(pcd,cone_color)


    # To see where the LiDAR points are relative to the camera, this function can be used
    def show_3d_warped_2d(self, pt_arr):
        im = self.getImage()
        im_rect = self.rectify_image(im)
        for point in pt_arr:
            cv2.circle(im_rect, point, 2, (0, 255, 255), 2)

        cv2.imshow("ruchawi", im_rect)
        cv2.waitKey()

    def rectify_image(self, image):
        tang_dist = self.cam_params.tang_dist
        radial_dist = self.cam_params.radial_dist
        K = self.cam_params.K

        # OpenCV takes radial and tangential distortion in as a vector
        dist_vector = np.asarray([radial_dist[0], radial_dist[1], tang_dist[0], tang_dist[1]])
        img_rectified = cv2.undistort(image, K, dist_vector	)
        return img_rectified

    def calculateAzimuth(self, pt):
        azimuth = math.atan2(pt[1], pt[0])
        return azimuth

    def calculateBearing(self,pt):
        bearing = math.pi/2 - math.atan2(math.sqrt(pt[0] ** 2 + pt[1] ** 2), pt[2])
        bearing = 180*bearing/math.pi
        return round(bearing)

    def clusterPoints(self, pcd, cone_color):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.3, min_points=7, print_progress=False))
        
        if len(labels) == 0:
            self.data3D.center_cones = []
            return
        max_label = labels.max()

        clustered_points = []
        clustered_points_labels = []
        # Add empty arrays for each cluster
        for i in range(max_label + 1):
            arr = []
            
            clustered_points.append(arr)
        
        for i in range(len(pcd.points)):
            if labels[i] >= 0:
                # x, y, z, color
                pt = pcd.points[i]
                bearing = self.calculateBearing(pt)
                cluster_pt = np.hstack((pt, np.array(cone_color[i])))
                cluster_pt = np.hstack((cluster_pt, np.array(bearing)))
                clustered_points[labels[i]].append(cluster_pt)      # xyz, color and bearing

                self.data3D.filtered_points.append(pt)


        self.data3D.filtered_points = np.asarray(self.data3D.filtered_points)
        self.data3D.clustered_cones = clustered_points

        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    def plotCenterOfCones(self):
        all_points = self.data3D.points
        filtered_points = self.data3D.filtered_points
        centers = self.data3D.center_cones
        plt.plot(filtered_points[:,0], filtered_points[:,1], 'ro',markersize=1)
        plt.plot(centers[:,0], centers[:,1], 'g^')
        plt.show()

    def calculateCenterOfCones(self):
        clusters = self.data3D.clustered_cones
        
        cluster_array = []
        for idx, cluster in enumerate(clusters):
            sorted_by_line = []
            current_line = []
            np_cluster = np.asarray(cluster)
            sorted_cluster = np_cluster[np_cluster[:,-1].argsort()]
            current_angle = sorted_cluster[0,-1]
            for np_c in sorted_cluster:
                if np_c[-1] == current_angle:
                    current_line.append(np_c)
                else:
                    sorted_by_line.append(current_line)
                    current_line = []
                    current_line.append(np_c)
                    current_angle = np_c[-1]

            current_line = np.asarray(current_line)
            sorted_by_line.append(current_line)
            cluster_array.append(sorted_by_line)


        # This cluster_array can be used, loop throug clusters -> loop through scan lines -> loop through points
        center_array = []
        for idx, cluster in enumerate(cluster_array):

            # Find the cone color:
            classes = [0, 0, 0, 0, 0]

            for scanline in cluster:
                for point in scanline:
                    classvalue = int(point[3])
                    if classvalue > 4:
                        classvalue = 4
                    classes[classvalue] += 1

            current_class = np.argmax(classes)

            sumx = 0
            sumy = 0
            max_dist = 0

            # If there is only one scanline in the cluster, it should not be included
            if len(cluster) == 1:
                continue

            reconstructed_scanline, distance, valid = self.reconstructScanLine(cluster[-1][0])
       
            # No cone should have a greater distance than 0.2 (the cones have been measured)
            if distance > 0.2 or valid == False or len(reconstructed_scanline) < 3:
                continue


            for pt in reconstructed_scanline:
                
                sumx += pt[0]
                sumy += pt[1]
                current_dist = math.sqrt(pt[0] ** 2 + pt[1] ** 2)
                if max_dist < current_dist:
                    max_dist = current_dist
           
            avg_x = sumx/len(reconstructed_scanline)
            avg_y = sumy/len(reconstructed_scanline)
            distavg = math.sqrt(avg_x ** 2 + avg_y ** 2)

            #max_dist = distavg
            scale = max_dist / distavg

            centerx = avg_x * scale
            centery = avg_y * scale
            center_array.append(np.asarray([centerx, centery, current_class]))

        self.data3D.center_cones = np.asarray(center_array)

    
    def reconstructScanLine(self,point):
        # Get the closest point
        min_dist = 100000
        min_index = -1

        for idx, pt in enumerate(self.data3D.points):
            current_dist = math.sqrt((point[0] - pt[0]) ** 2 + (point[1] - pt[1]) ** 2 + (point[2] - pt[2]) ** 2)
            if current_dist < min_dist:
                min_dist = current_dist
                min_index = idx

        idx = min_index
        scanline = [self.data3D.points[min_index]]

        leftmost_pt = deepcopy(point)
        while True:
            # Count down the scanline
            previous_pt = self.data3D.points[idx]
            idx -= 1
            if idx < 0:
                break
            current_pt = self.data3D.points[idx]

            threshold = 0.1
            dist = math.sqrt((current_pt[0] - previous_pt[0]) ** 2 + (current_pt[1] - previous_pt[1]) ** 2 + (current_pt[2] - previous_pt[2]) ** 2)
            if dist < threshold:
                scanline.append(current_pt)
                leftmost_pt = deepcopy(current_pt)
            else:
                break
        left_of_cone = self.data3D.points[idx-1]
        idx = min_index
        rightmost_pt = deepcopy(point)
        while True:
            # Count down the scanline
            previous_pt = self.data3D.points[idx]
            idx += 1
            if idx >= len(self.data3D.points):
                break
            current_pt = self.data3D.points[idx]

            threshold = 0.1
            dist = math.sqrt((current_pt[0] - previous_pt[0]) ** 2 + (current_pt[1] - previous_pt[1]) ** 2 + (current_pt[2] - previous_pt[2]) ** 2)
            if dist < threshold:
                scanline.append(current_pt)
                rightmost_pt = deepcopy(current_pt)

            else:
                break
        right_of_cone = self.data3D.points[idx+1]

        distance = math.sqrt((leftmost_pt[0] - rightmost_pt[0]) ** 2 + (leftmost_pt[1] - rightmost_pt[1]) ** 2 + (leftmost_pt[2] - rightmost_pt[2]) ** 2)
       
        # Check if the distance to the left and right of the cones are less than the point on the cone
        point_dis = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[3] ** 2)
        left_dis = math.sqrt(left_of_cone[0] ** 2 + left_of_cone[1] ** 2 + left_of_cone[3] ** 2)
        right_dis = math.sqrt(right_of_cone[0] ** 2 + right_of_cone[1] ** 2 + right_of_cone[3] ** 2)

        scanline = np.asarray(scanline)
        scanline = scanline[np.argsort(scanline[:, 4])]

        # Angle difference between the leftmost point on the scanline and the next poin to the left
        left_of_cone_angle_diff = scanline[0,4] - left_of_cone[4]   
        # Angle difference between the leftmost point on the scanline and the next poin to the left
        right_of_cone_angle_diff = right_of_cone[4] - scanline[-1,4]
        valid = True
        # If the point on the left of the leftmost point on the scanline is closer than the cone, and the angle is close then the cone is truncated
        if point_dis > left_dis and left_of_cone_angle_diff < 0.05:
            valid = False

        if point_dis > right_dis and right_of_cone_angle_diff < 0.05:
            valid = False
        
        return np.asarray(scanline), distance, valid
