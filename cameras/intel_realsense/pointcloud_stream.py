## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################
import PIL
from PIL import Image
import requests
from io import BytesIO
from PIL import ImageFilter
from PIL import ImageEnhance
from IPython.display import display
# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import matplotlib.pyplot as plt
import video_process_class as vpc
import sys

def pixel_to_point(depth, x, y, intrinsics):
    z = depth.get_distance(x, y);
    x = z * (x - intrinsics.ppx) / intrinsics.fx
    y = z * (y - intrinsics.ppy) / intrinsics.fy
    return(np.array([x, y, z]))


#Deproject(depth, 100, 100, intrinsics);


# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_stream = rs.video_stream_profile (profile.get_stream (rs.stream.depth))
intrinsics = depth_stream.get_intrinsics ()


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

img_counter = 0

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #Into numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        hsv=cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_green=np.array([0,100,100])
        higher_green=np.array([255,255,255])
        mask=cv2.inRange(hsv,lower_green,higher_green)
        color_image=cv2.bitwise_and(color_image, color_image,mask=mask)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('Align RGB Depth RedRecognizer Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align RGB Depth RedRecognizer Example', images)
        key = cv2.waitKey(1)


        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        elif key == 32:
            # SPACE pressed

            color_numpy = np.asanyarray(color_image)
            indexes = np.where(color_numpy != [0, 0, 0])
            y_indexes = indexes[0]
            x_indexes = indexes[1]

            pointcloud = []
            for i in range(len(y_indexes)):
                pointcloud.append(pixel_to_point(aligned_depth_frame, x_indexes[i], y_indexes[i], intrinsics))
            pointcloud = np.array(pointcloud)

            pointcloud_name = "data/pointcloud_{}.txt".format(img_counter)

            np.savetxt(pointcloud_name, pointcloud)
            print("{} written!".format(pointcloud_name))
            img_counter += 1

finally:
        pipeline.stop()
