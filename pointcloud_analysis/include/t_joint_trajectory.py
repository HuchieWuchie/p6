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

class t_joint_trajectory:
    def __init__(self, arg):
        super t_joint_trajectory, self).__init__()
        self.arg = arg

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def smooth2(self, cutoff, fs, array, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, array)
        return y

    def linefit(self, x1,y1,x2,y2):
        m = ((y1-y2)/(x1-x2))
        x = np.linspace(x1, x2, 100)
        #y-y2 = m*(x-x2)
        b = (x1*y2 - x2*y1)/(x1-x2)
        y=m*x+b
        array=np.array([x,y])
        return [x,y]

    def rotatepoints(self, angle, x,y, ox,oy):
        qx = ox+np.cos(np.radians(angle))*(x-ox)+np.sin(np.radians(angle))*(y-oy)
        qy = oy+np.cos(np.radians(angle))*(y-oy)-np.sin(np.radians(angle))*(x-ox)
        return [qx, qy]

    def getangle(self, a,b,c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return angle
