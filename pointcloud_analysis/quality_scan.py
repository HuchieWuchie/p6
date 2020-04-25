import sys
sys.path.insert(0, 'include/')

import copy
import numpy as np
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
import time
start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))

FILE_NAME = "04-01-test-scan-5-after.txt"
FILE_NAME2 = "04-01-test-scan-5.txt"
sys.path.append("/home/karl/P6/Code/p6/pointcloud_analysis/build/")
os.chdir("/home/karl/P6/Code/p6/pointcloud_analysis/build/")
os.system("./program "+FILE_NAME+" roi save_false")
os.system("./program "+FILE_NAME2+" trajectories save_false")

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


#pcd.pcd = o3d.geometry.PointCloud.remove_statistical_outlier(pcd.pcd, 1000, 1)[0]
pcd.getPointCloudLines() # Construct arrays of point cloud lines
img = pcd.pcd_to_image()
npArr = pcd.img_to_pcd(img)
pcd = PointCloud(npArr, type='NPARR')
pcd.getPointCloudLines() # Construct arrays of point cloud lines

weld_toe = []

weld_toe_0 = np.loadtxt("weldseam0.txt")
weld_toe_1 = np.loadtxt("weldseam1.txt")

weld_toe_0 = Trajectory(weld_toe_0)
weld_toe_1 = Trajectory(weld_toe_1)


weld_toe.append(weld_toe_1)
weld_toe.append(weld_toe_0)



weld_toe[0].flip()
weld_toe[0].setColor('r')

weld_toe[1].flip()
weld_toe[1].setColor('g')
pcd.flip()

#o3d.visualization.draw_geometries([pcd.pcd, weld_toe[0].pcd, weld_toe[1].pcd])
#o3d.visualization.draw_geometries([pcd.pcd, weld_toe[0].pcd]) #left
#o3d.visualization.draw_geometries([pcd.pcd, weld_toe[1].pcd])

pcd.getPointCloudLines()
weld_toe[0].getPointCloudLines()

weld_toe[1].getPointCloudLines()
for i in range(len(weld_toe[0].pointCloudLine)):
    weld_toe[0].pointCloudLine[i] = np.flip(weld_toe[0].pointCloudLine[i],0)
for i in range(len(weld_toe[1].pointCloudLine)):
    weld_toe[1].pointCloudLine[i] = np.flip(weld_toe[1].pointCloudLine[i],0)

raw_weldtoe_left = copy.deepcopy(weld_toe[0].pointCloudLine)
raw_weldtoe_right = copy.deepcopy(weld_toe[1].pointCloudLine)

#o3d.visualization.draw_geometries([pcd.pcd, weld_toe[0].pcd])
#o3d.visualization.draw_geometries([pcd.pcd, weld_toe[1].pcd])

# Karls rocker algorithm from here.

peenDepth_left = []
peenDepth_right = []
peenRadius_left = []
peenRadius_right = []
peenWidth_left = []
peenWidth_right = []
trajectpredict = []
deviation_l, deviation_r = [], []
line=[]
perpline_left = []
perpline_right = []
intercept_left = []
intercept_right = []
for i in range(len(weld_toe[0].pointCloudLine)):

    try:
        #print(i)
        quality_left = Quality(weld_toe[0].pointCloudLine[i], pcd.pointCloudLine[i])
        quality_right = Quality(weld_toe[1].pointCloudLine[i], pcd.pointCloudLine[i])
        peenWidth_left.append([quality_left.get_width_and_midpoint(),i])
        peenWidth_right.append([quality_right.get_width_and_midpoint(),i])
        perpline_left.append(quality_left.pline())
        perpline_right.append(quality_right.pline())
        intercept_left.append(quality_left.find_intercept(quality_left.pline()))
        intercept_right.append(quality_right.find_intercept(quality_right.pline()))
        peenDepth_left.append([quality_left.get_depth(),i])
        peenDepth_right.append([quality_right.get_depth(),i])

        #quality_left.plot_stuff(weld_toe[0].pointCloudLine[i])
        #quality_right.plot_stuff(weld_toe[1].pointCloudLine[i])

        deviation_l.append([quality_left.find_traject(xOff_l), i])
        deviation_r.append([quality_right.find_traject(xOff_r), i])
    except Exception as e:
        pass
        #print(e)


peenDepth_left = np.array(peenDepth_left)
peenDepth_right = np.array(peenDepth_right)
peenRadius_left = np.array(peenRadius_left)
peenRadius_right = np.array(peenRadius_right)
peenWidth_left = np.array(peenWidth_left)
peenWidth_right = np.array(peenWidth_right)
trajectpredict = np.array(trajectpredict)
deviation_l = np.array(deviation_l)
deviation_r = np.array(deviation_r)

fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].set_title('Depth Left Side')
axs[0].plot(peenDepth_left[:][:,1], quality_right.smooth_outliers(peenDepth_left[:][:,0],5))
axs[0].hlines(0.1, 0, peenDepth_left[-1][1],colors = 'r', linestyles = 'dashed')
axs[0].hlines(0.6, 0, peenDepth_left[-1][1],colors = 'r', linestyles = 'dashed')
axs[0].set(ylabel='Depth (mm)')
axs[0].grid()

axs[1].set_title('Width Left Side')
axs[1].plot(peenWidth_left[:][:,1], quality_right.smooth_outliers(peenWidth_left[:][:,0],5))
axs[1].hlines(3, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[1].hlines(6, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[1].set(ylabel='Width (mm)')
axs[1].grid()

axs[2].set_title('Trajectory Deviation Left')
axs[2].plot(deviation_l[:][:,1], quality_right.smooth_outliers(deviation_l[:][:,0],5))
axs[2].hlines(-0.7121, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[2].hlines(0.7121, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[2].set(xlabel='y-value (mm)', ylabel='Error (mm)')
axs[2].grid()

fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].set_title('Depth Right Side')
axs[0].plot(peenDepth_right[:][:,1], quality_right.smooth_outliers(peenDepth_right[:][:,0],5))
axs[0].hlines(0.1, 0, peenDepth_left[-1][1],colors = 'r', linestyles = 'dashed')
axs[0].hlines(0.6, 0, peenDepth_left[-1][1],colors = 'r', linestyles = 'dashed')
axs[0].set(ylabel='Depth (mm)')
axs[0].grid()

axs[1].set_title('Width Right Side')
axs[1].plot(peenWidth_right[:][:,1], quality_right.smooth_outliers(peenWidth_right[:][:,0],5))
axs[1].hlines(3, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[1].hlines(6, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[1].set(ylabel='Width (mm)')
axs[1].grid()

axs[2].set_title('Trajectory Deviation Right')
axs[2].plot(deviation_r[:][:,1], quality_right.smooth_outliers(deviation_r[:][:,0],5))
axs[2].hlines(-0.7349, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[2].hlines(0.7349, 0, peenDepth_left[-1][1],colors = 'r',linestyles = 'dashed')
axs[2].set(xlabel='y-value (mm)',ylabel='Error (mm)')
axs[2].grid()
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
