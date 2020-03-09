import numpy as np
import cv2
import matplotlib.pyplot as plt
 
class videoProcess:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
 
    def release(self):
        if not self.cap: return
        self.cap.release()
        self.cap = None
 
    def segmentColor(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
        #HSV hue saturation value                   green               red
        lower_green = np.array([1,100,0])         #175, 90, 150     #1, 150, 0
        upper_green = np.array([50,255,190])       #180, 255, 255    #8, 255, 190
 
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask
 
    def morphOpenClose(self, mask):
        kernel = np.ones((5,5),np.uint8)
        opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing_mask = cv2.morphologyEx(opening_mask, cv2.MORPH_CLOSE, kernel)
        self.morphedFrame = closing_mask
        return self.morphedFrame
 
    def getColorPoints(self, frame):
        h = frame.shape[0]
        w = frame.shape[1]
        self.colorPointlist=[]
        for y in range(0, h):
            for x in range(0, w):
                if frame[y][x] > 0:
                    self.colorPointlist.append([y,x])
        return self.colorPointlist
 
    def getDepthPoints(self):
 
        return 0
 
    def depthP2globalP(self):
 
        return 0