import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from scipy.signal import butter, filtfilt
import time
import heapq

class PointCloudLine:
    def __init__(self, points, stepSize=1):
        self.x = []
        self.y = []
        self.z = []
        self.getPoints(points)
        self.FS = self.x.size
        self.stepSize = stepSize
        self.b_smooth = False

    def getPoints(self, points):
        k = 0

        for data in points:
            self.x.append(data[0])
            self.y.append(data[1])
            self.z.append(data[2])
            k += 1

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

    def getFeatureVector(self, index, border_x, x=False, y=False, z=False):
        feature_size = (2*border_x)+1
        if index > border_x and index < (len(self.x)-border_x):
            feature_vector = []
            for i in range(feature_size):
                idx = (index - border_x) + i
                if x == True:
                    feature_vector.append(self.x[idx])
                if y == True:
                    feature_vector.append(self.y[idx])
                if z == True:
                    feature_vector.append(self.z[idx])

            feature_vector = np.array(feature_vector)
            return feature_vector

    def getFeatureVectorZOnly(self, index, step_size):
        feature_size = (2*step_size)+1
        if index > step_size and index < (len(self.x)-step_size):
            feature_vector = []
            for i in range(feature_size):
                idx = (index - step_size) + i
                feature_vector.append(self.z[idx])
            feature_vector = np.array(feature_vector)
            return feature_vector


    def plot(self):
        fig, ax = plt.subplots(1, 1)

        ax[0].plot(self.x,self.z)
        ax[0].set_xlabel('width (mm)')
        ax[0].set_ylabel('height')
        ax[0].grid()
        ax[0].plot(self.x_left, self.z_left+0.1, 'ro')
        ax[0].plot(self.x_right, self.z_right+0.1, 'ro')

        plt.show()

    def getDerivative(self, threshold = 0.000, stepSize = 1):
        gradiant = []
        for k in range(self.x.size-stepSize):
            if abs(self.z[k]-self.z[k+stepSize]) > threshold: #0.003
                gradiant.append(self.z[k]-self.z[k+self.stepSize])
            else:
                gradiant.append(0)
        gradiant = np.array(gradiant)
        return gradiant

    def getEdge(self):

        z_max = np.amin(self.z)
        index = np.where(self.z == z_max)
        return np.array([self.x[index][0], self.y[0], self.z[index][0]])


    def medianFilter(self, kernelSize):
        for i in range(len(self.x)-kernelSize):
            if i>kernelSize-1:
                npArray = []
                for k in range((2*kernelSize)+1):
                    npArray.append(self.z[(i-kernelSize)+k])
                npArray = np.array(npArray)

                self.z[i] = np.median(npArray)

    def findZero(self, thresh):
        npArray = []
        self.x_grad_minmax = []
        self.z_grad_minmax = []
        for i in range(len(self.gradiant2)):
            if self.gradiant2[i] < thresh and self.gradiant2[i] > -thresh:
                npArray.append(i)
                self.x_grad_minmax.append(self.x[i])
                self.z_grad_minmax.append(self.gradiant[i])
