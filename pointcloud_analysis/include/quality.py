import sys
sys.path.insert(0, 'include/')

import numpy as np
import scipy as sp
import open3d as o3d
import math
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
from trajectory import Trajectory
from scipy import optimize
from numpy import diff
from scipy.signal import butter, filtfilt

npArr = []
traj = Trajectory(npArr)
class Quality:
    def __init__(self, array, pcdarray):
        #initiate values
        self.array = array
        self.FS = len(array)
        self.pcdarray = pcdarray

        #smooth z values and place back in array
        self.smooth_array = array

        #smooth_z = self.moving_Avg(array[:][:,2], 33)
        self.smooth(cutoff=10)
        smooth_z = self.z_smoothvalue
        for i in range(len(array)):
            self.smooth_array[i][2] = smooth_z[i]
        self.smooth_array1 = np.array(self.smooth_array)

        #finding the z minimum of the smoothed peened weld-toe "self.z_smoothmin"
        self.z_smoothvalue = []
        for points in self.smooth_array:
            self.z_smoothvalue.append(points[2])
        self.z_smoothvalue = np.array(self.z_smoothvalue)
        self.z_smoothmin = np.min(self.z_smoothvalue)

        #finding the z minimum of the peened weld-toe "self.z_min"
        z_value = []
        for points in array:
            z_value.append(points[2])
        z_value = np.array(z_value)
        self.z_min = np.min(z_value)

        #finding the first and last point in the peened weld-toe array format: [x,y,z] "self.first_index" "self.last_index"
        self.first_index = array[:][0]
        self.last_index = array[:][-1]

        self.get_derivative()
        self.derivative_min_max()
        self.find_minmax_curvature()

    def get_depth(self):

        depth =  math.sqrt(((self.midpoint[0]-self.xinter)**2)+((self.midpoint[2]-self.zinter)**2))
        return depth

    def plot_stuff(self, array):
        smooth_z = self.z_smoothvalue
        plt.plot(array[:][:,0], smooth_z)
        plt.plot(self.curvature[:,0],self.curvature[:,2])
        plt.show()

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def smooth_outliers(self, npArray, kernal = 3):
        npArray = np.array(npArray)
        npArray = sp.ndimage.filters.median_filter(npArray, kernal)
        return npArray

    def moving_Avg(self, mylist, N=3):
        cumsum, moving_aves = [0], []

        for i, x in enumerate(mylist, 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=N:
                moving_ave = (cumsum[i] - cumsum[i-N])/N
                #can do stuff with moving_ave here
                moving_aves.append(moving_ave)
        for i in range(N-2):
            moving_aves.insert(0,moving_aves[0])
            moving_aves.insert(-1,moving_aves[-1])
        moving_aves = np.array(moving_aves)
        return moving_aves

    def calc_R(self, x,y, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f(self, c, x, y):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(x, y, *c)
        return Ri - Ri.mean()

    def leastsq_circle(self, x,y):
        # coordinates of the barycenter
        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(self.f, center_estimate, args=(x,y))
        xc, yc = center
        Ri       = self.calc_R(x, y, *center)
        R        = Ri.mean()
        residu   = np.sum((Ri - R)**2)
        return xc, yc, R, residu

    def plot_data_circle(self, x,y, xc, yc, R):
        f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
        plt.axis('equal')

        theta_fit = np.linspace(-3.14, 3.14, 180)

        x_fit = xc + R*np.cos(theta_fit)
        y_fit = yc + R*np.sin(theta_fit)
        plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
        plt.plot([xc], [yc], 'bD', mec='y', mew=1)
        plt.xlabel('x')
        plt.ylabel('y')
        # plot data
        plt.plot(x, y, 'r-.', label='data', mew=1)
        plt.legend(loc='best',labelspacing=0.1 )
        plt.grid()
        plt.text(xc,yc, R)
        plt.title('Least Squares Circle')

    def get_derivative(self):
        self.dydx = diff(self.smooth_array[:,2])/diff(self.smooth_array[:,0])

    def derivative_min_max(self):
        arrStart = self.dydx[0:int(len(self.dydx)*0.4)]
        arrEnd = self.dydx[int(len(self.dydx)*0.6):len(self.dydx)]
        fullArr = np.concatenate((arrStart,arrEnd),axis=0)
        derivmin = np.min(fullArr)
        derivmax = np.max(fullArr)
        resultMin = np.where(self.dydx == derivmin)
        self.indexMin = resultMin[0][0]
        resultMax = np.where(self.dydx == derivmax)
        self.indexMax = resultMax[0][0]


    def find_minmax_curvature(self):
        self.curvature = self.smooth_array[self.indexMax:self.indexMin]
        return self.curvature

    def get_width_and_midpoint(self):
        x1 = self.curvature[0][0]
        y1 = self.curvature[0][2]
        x2 = self.curvature[-1][0]
        y2 = self.curvature[-1][2]
        self.dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
        self.midpoint = [(x2+x1)/2, self.curvature[0][1], (y2+y1)/2]
        return self.dist

    def widthline(self):
        x1 = self.curvature[0][0]
        y1 = self.curvature[0][2]
        x2 = self.curvature[-1][0]
        y2 = self.curvature[-1][2]
        m = (y1-y2)/(x1-x2)
        x = np.linspace(x1, x2, 100)
        #y-y2 = m*(x-x2)
        b = (x1*y2 - x2*y1)/(x1-x2)
        y=m*x + b
        return [x, y]


    def pline(self):
        x1 = self.curvature[0][0]
        y1 = self.curvature[0][2]
        x2 = self.curvature[-1][0]
        y2 = self.curvature[-1][2]
        m = (x1-x2)/(y1-y2)
        x = np.linspace(x2, x1, len(self.curvature))
        #y-y2 = m*(x-x2)
        #b = (x1*y2 - x2*y1)/(x1-x2)
        y=-m*(x-self.midpoint[0]) + self.midpoint[2]
        return [x, y]

    def find_intercept(self, array):
        x = self.curvature[:][:,0]
        f = array[1]
        g = self.curvature[:][:,2]

        idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
        self.xinter = x[idx]
        self.zinter = f[idx]
        return x[idx],f[idx]

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def smooth(self, cutoff, order=5):
        b, a = self.butter_lowpass(cutoff, self.FS, order=order)
        y = filtfilt(b, a, self.array[:,2])
        self.z_smoothvalue = y

    def smooth2(self, cutoff, fs, array, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, array)
        return y

    def find_traject(self, xOffset, file = "trajectory"):

        #file = file.replace('quality', '')
        #file = ('trajectories/' + file + 'weld-trajectory-')
        
        if self.midpoint[0] < 0:
            file = (file + '0.txt')
        elif self.midpoint[0] > 0:
            file = (file + '1.txt')

        trajectory = np.loadtxt(file)
        idx = np.where(trajectory[:][:, 1] == round(self.midpoint[1],1))
        #print(trajectory[:][:, 1])
        print(round(self.midpoint[1],1))
        print(len(idx[0]))
        if len(idx[0]) != 0:

            idx = idx[0][0]

        error = trajectory[idx][0]- self.midpoint[0] - xOffset

        return error
