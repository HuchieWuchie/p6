import time
start_time = time.time()
import sys
sys.path.insert(0, '/home/karl/P6/Code/p6-2/p6/pointcloud_analysis/include')

import copy
import numpy as np
import scipy as sp
from pointCloud import PointCloud
from pointCloudLine import PointCloudLine
from quality import Quality
from net import Net
from helperFunctions import *
import matplotlib.pyplot as plt
from numpy import fft
from scipy.signal import butter, filtfilt
import open3d as o3d
from trajectory import Trajectory
import heapq
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import filters
import os



FILE_NAME = "T-joint_attempt_1.txt"
PIT_ANGLE = 80#int(input('Define PIT indentation angle in degrees:'))

os.chdir("t-scans/T_joint_scans/")
print("Loading point cloud...")


if os.path.exists(FILE_NAME) == True:
    print(FILE_NAME)
    npArr = np.loadtxt(fname = FILE_NAME)
    npArr = np.delete(npArr, 3, 1)
    pcd = PointCloud(npArr, type='NPARR')
    print("Preprocessing point cloud...")
    pcd.deleteZPoints(0)
    #pcd.setOffset(0, 0, -120)
    #pcd.pcd = o3d.geometry.PointCloud.remove_statistical_outlier(pcd.pcd, 1000, 1)[0]
    pcd.getParameters()
    #np.save(processed_file, np.asarray(pcd.pcd.points))
else:
    print('Cannot access file')
    exit()


#pcd.pcd = o3d.geometry.PointCloud.remove_statistical_outlier(pcd.pcd, 1000, 1)[0]
pcd.getPointCloudLines() # Construct arrays of point cloud lines
img = pcd.pcd_to_image()
npArr = pcd.img_to_pcd(img)
pcd = PointCloud(npArr, type='NPARR')
pcd.flip()
pcd.getPointCloudLines() # Construct arrays of point cloud lines
line = pcd.pointCloudLine


#To visualize the pointcloud uncomment the next line

#o3d.visualization.draw_geometries([pcd.pcd])


#functions used for the program

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def smooth2(cutoff, fs, array, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, array)
    return y

def linefit(x1,y1,x2,y2):
    m = ((y1-y2)/(x1-x2))
    x = np.linspace(x1, x2, 100)
    b = (x1*y2 - x2*y1)/(x1-x2)
    y=m*x+b
    array=np.array([x,y])
    return [x,y]

def rotatepoints(angle, x,y, ox,oy):
    qx = ox+np.cos(np.radians(angle))*(x-ox)+np.sin(np.radians(angle))*(y-oy)
    qy = oy+np.cos(np.radians(angle))*(y-oy)-np.sin(np.radians(angle))*(x-ox)
    return [qx, qy]

def getangle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def smooth_outliers(npArray, kernal = 3):
    npArray = np.array(npArray)
    npArray = sp.ndimage.filters.median_filter(npArray, kernal)
    return npArray

trajectory_left = []
trajectory_right = []

for i in tqdm(range(len(line))):

    line_z = smooth2(23,len(line[i][:][:,2]),line[i][:][:,2], 5)
    #line_z = line[i][:][:,2]
    line_x = line[i][:][:,0]

    gradient = diff(line_z)/diff(line_x)
    gradient_line = np.delete(line_x, -1)

    curvature = diff(gradient)/diff(gradient_line)
    curvature_line = np.delete(gradient_line, -1)

    curv1max = np.max(curvature[1000:-300])
    res = np.where(curvature == curv1max)
    first_max_idx = res[0][0]

    curv2max = np.max(curvature[0:800])
    res2 = np.where(curvature == curv2max)
    second_max_idx = res2[0][0]

    linefit_right = linefit(line_x[second_max_idx], line_z[second_max_idx], line_x[150], line_z[150])
    linefit_right_x = linefit_right[0]
    linefit_right_z = linefit_right[1]

    linefit_left = linefit(line_x[first_max_idx], line_z[first_max_idx], line_x[1995], line_z[1995])
    linefit_left_x = linefit_left[0]
    linefit_left_z = linefit_left[1]

    horizontal_line_point_right = np.array([line_x[second_max_idx-50],line_z[second_max_idx]])
    rotateline_right = []

    horizontal_line_point_left = np.array([line_x[first_max_idx+50],line_z[first_max_idx]])
    rotateline_left = []

    for j in range(100):
        rotateline_left.append(rotatepoints(PIT_ANGLE, linefit_left_x[j], linefit_left_z[j], linefit_left_x[0], linefit_left_z[0]))
        rotateline_right.append(rotatepoints(-PIT_ANGLE, linefit_right_x[j], linefit_right_z[j], linefit_right_x[0], linefit_right_z[0]))


    rotateline_left = np.array(rotateline_left)
    rotateline_left_x = rotateline_left[:][:,0]
    rotateline_left_z = rotateline_left[:][:,1]

    rotateline_right = np.array(rotateline_right)
    rotateline_right_x = rotateline_right[:][:,0]
    rotateline_right_z = rotateline_right[:][:,1]

    left_angle = np.degrees(getangle(horizontal_line_point_left,[line_x[first_max_idx],line_z[first_max_idx]], rotateline_left[-1]))

    right_angle = np.degrees(getangle(horizontal_line_point_right,[line_x[second_max_idx],line_z[second_max_idx]], rotateline_right[-1]))

    trajectory_left.append([line[i][:][first_max_idx,0], line[i][:][first_max_idx,1], line[i][:][first_max_idx,2], left_angle])
    trajectory_right.append([line[i][:][second_max_idx,0], line[i][:][second_max_idx,1], line[i][:][second_max_idx,2], right_angle])

    plt.rcParams.update({'font.size': 22})
    if i>420:
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        axs[0].set_title('Curvature of pointcloud line')
        axs[0].plot(curvature_line,curvature)
        axs[0].plot(curvature_line[first_max_idx], curvature[first_max_idx],'bo')
        axs[0].plot(curvature_line[second_max_idx], curvature[second_max_idx],'bo')
        axs[1].plot(gradient_line,gradient)
        axs[1].set_title('Gradient of pointcloud line')
        axs[2].plot(line[i][:][:,0],line[i][:][:,2])
        axs[2].set_title('Pointcloud line')
        axs[2].plot(line[i][:][first_max_idx,0], line[i][:][first_max_idx,2], 'bo')
        axs[2].plot(line[i][:][second_max_idx,0], line[i][:][second_max_idx,2], 'bo')
        #axs[2].plot(rotateline_left_x, rotateline_left_z,'y')
        #axs[2].plot(linefit_left[0], linefit_left[1],'g')
        #axs[2].plot(horizontal_line_point_left[0],horizontal_line_point_left[1],'ro')
        #axs[2].plot(rotateline_right_x, rotateline_right_z,'y')
        #axs[2].plot(linefit_right[0], linefit_right[1],'g')
        #axs[2].plot(horizontal_line_point_right[0],horizontal_line_point_right[1],'ro')
        #plt.set_title('Pointcloud line')
        #plt.plot(line[i][:][:,0],line[i][:][:,2])
        plt.show()

trajectory_left = np.array(trajectory_left)
trajectory_right = np.array(trajectory_right)

trajectory_left[:,0] = smooth_outliers(trajectory_left[:,0],7)
trajectory_right[:,0] = smooth_outliers(trajectory_right[:,0],7)

os.chdir("/home/karl/P6/Code/p6-2/p6/pointcloud_analysis/trajectories")
with open('trajectory_left.txt', 'w') as filehandle:
    for line in trajectory_left:
        filehandle.write('%s\n' % str(line)[1:-1])
with open('trajectory_right.txt', 'w') as filehandle:
    for line in trajectory_right:
        filehandle.write('%s\n' % str(line)[1:-1])
print("--- %s seconds ---" % (time.time() - start_time))
