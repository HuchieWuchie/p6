from os import listdir
from os.path import isfile, join
import re
import numpy as np
import open3d as o3d

def get_data_list():
    files = [f for f in listdir("data/") if isfile(join("data/", f))]
    data_list = []
    for file in files:
        digit = re.sub('[^0-9]','', file)
        data_list.append(digit)

    data_list = list(set(data_list)) # Removes duplicates
    data_list = np.array(data_list)
    data_list = np.sort(data_list) # Sort in ascending order

    return(list(data_list))

def pixel_to_point(x, y, depth, intrinsics):
    z = depth[y, x]
    x = z * (x - intrinsics.cx) / intrinsics.fx
    y = z * (y - intrinsics.cy) / intrinsics.fy
    return(np.array([x, y, z]))

def compute_pointcloud(rgb, depth, intrinsics):

    rgbArr = np.asanyarray(rgb)
    indexes = np.where(rgbArr != [0, 0, 0])
    y_indexes = indexes[0]
    x_indexes = indexes[1]

    pointcloud = []
    colorArr = []

    for i in range(0, len(y_indexes), 3):
        point = pixel_to_point(x_indexes[i], y_indexes[i], depth, intrinsics)
        if np.array_equal(point, np.array([0,0,0])) != True:
            pointcloud.append(point)
            colorArr.append(rgb[y_indexes[i], x_indexes[i]])
    pointcloudArr = np.array(pointcloud)
    colorArr = np.array(colorArr)/255
    return pointcloudArr, colorArr

class Cone:

    def __init__(self, width, height, depth):

        self.width = width
        self.height = height
        self.depth = depth

        self.vertices = np.array([[-self.width/2, -self.height/2, -self.depth], [self.width/2, -self.height/2, -self.depth], [self.width/2, self.height/2, -self.depth], [-self.width/2, self.height/2, -self.depth], [0, 0, 0]])

        self.construct_mesh()

    def construct_mesh(self):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vertices)
        pcd.estimate_normals()
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        mesh_box = o3d.geometry.TriangleMesh.create_box(self.width, self.height, depth = 0.1)
        mesh_box.translate(np.array([-self.width/2, -self.height/2, -self.depth]))

        self.mesh = mesh_box + mesh

    def rotate_x(self, theta):
        # Returns a x-axis rotation matrix

        theta = np.radians(theta)

        c = np.cos(theta)
        s = np.sin(theta)

        rotx = np.array([[1, 0 ,0], [0, c, -s], [0, s, c]])
        self.mesh.rotate(rotx, False)

    def rotate_y(self, theta):
        # Returns an y-axis rotation matrix

        theta = np.radians(theta)

        c = np.cos(theta)
        s = np.sin(theta)

        roty = np.array([[c, 0 ,s], [0, 1, 0], [-s, 0, c]])
        self.mesh.rotate(roty, False)


    def rotate_z(self, theta):
        # Returns a z-axis rotation matrix

        theta = np.radians(theta)

        c = np.cos(theta)
        s = np.sin(theta)

        rotz = np.array([[c, -s ,0], [s, c, 0], [0, 0, 1]])
        self.mesh.rotate(rotz, False)

    def translate(self, vec):
        # Adds transformation vector to all geometry coordinates

        self.mesh.translate(vec)

class Scanning_trajectory:

    def __init__(self, npArr):

        self.trajectory = npArr
        self.cones = []
        for i in range(len(npArr)):
            cone = Cone(25, 25, 50)
            cone.rotate_x(npArr[i][3])
            cone.rotate_y(npArr[i][4])
            cone.rotate_z(npArr[i][5])
            cone.translate(npArr[i][0::2])
            cone.mesh.paint_uniform_color((1,0,1))
            self.cones.append(cone)




class Intrinsics:

    def __init__(self,filename =""):
        file = open(filename, 'r')
        output = file.read()
        output = output.split(" ")
        self.fx = float(output[0])
        self.fy = float(output[1])
        self.cx = float(output[2])
        self.cy = float(output[3])
