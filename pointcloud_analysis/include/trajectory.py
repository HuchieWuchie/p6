import numpy as np
import scipy as sp
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from pointCloud import PointCloud

class Trajectory(PointCloud):
    def __init__(self, npArr, debug=False):
        self.npArr = npArr
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.npArr)
        #self.getParameters() #if running annotate, then outcomment this line.

    def save(self, nrWeld, name=""):
        f= open("trajectories/%s-trajectory-%d.txt" % (name, nrWeld),"w+")
        for point in np.asarray(self.pcd.points):
            for i in range(3):
                f.write("%f" % point[i])
                if i != 2:
                    f.write(" ")
            if nrWeld == 0:
                f.write(" 0 -20 0")
            elif nrWeld == 1:
                f.write(" 0 20 0")
            f.write("\n")
        f.close()

    def trajectory_to_image(self, i_max, y_range, resolution, x_max, span=0.5):
        matrix = np.zeros(i_max*y_range)
        matrix = np.reshape(matrix, (y_range, -1))
        span = resolution * span

        for i in tqdm(range(len(self.pointCloudLine))):
            for j in range(i_max):
                value = x_max - (j * resolution)
                result = np.where(np.logical_and(self.pointCloudLine[i][:][:,0] < value+span, self.pointCloudLine[i][:][:,0] > value-span))
                if len(result[0]) != 0:
                    index = result[0][0]
                    matrix[i, j] = self.pointCloudLine[i][index][2]

        return matrix

    def unNormalize(self, max, min):
        npArr = []
        for point in np.asarray(self.pcd.points):
            #print("z_norm: ", point[2], " unNormalized: ", (point[2]*max)-(point[2]*min)+min)
            point[2] = (point[2]*max)-(point[2]*min)+min
            npArr.append(point)
        npArr = np.array(npArr)
        return npArr

    def smooth_outliers(self, npArray):
        npArray = np.array(npArray)
        npArray = sp.ndimage.filters.median_filter(npArray, 5)
        return npArray

    def create_polynomial(self, trajectory_x, trajectory_y,  trajectory_z, stepsize, y_range, y_min, POLY_DEGREE = 6):
        print(y_range, y_min)
        x = trajectory_x
        y = trajectory_y
        z = trajectory_z


        xcoeff = np.polyfit(x,y,POLY_DEGREE)
        ycoeff = np.polyfit(x,z,3)

        fx = np.poly1d(xcoeff)
        fy = np.poly1d(ycoeff)

        self.xx_value = []
        self.xy_value = []
        new_y = []
        new_z = []
        for i in range(y_range):
            x_value = (stepsize * i) + y_min
            new_y.append(fx(x_value))
            self.xx_value.append(x_value)
            new_z.append(fy(x_value))
            self.xy_value.append(x_value)

        return new_y, new_z


    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
