
import math
import open3d as o3d
import copy
from math import cos, sin, radians, pi
from matplotlib import pyplot as plt
import numpy as np
import transformations as tf
from scipy.spatial.transform import Rotation as R
import time
def xyz_fixed_rotation(cart):
    dX = cart[0]
    dY = cart[1]
    dZ = cart[2]
    zC, zS = ang(cart[3])
    yC, yS = ang(cart[4])
    xC, xS = ang(cart[5])

    x_tr_r = np.array((dX, dY, dZ, 1))
    T_x_r = tf.translation_matrix(x_tr_r)
    # I=tf.identity_matrix()
    Rot_x_r = tf.rotation_matrix(cart[3], (1, 0, 0))
    Rot_y_r = tf.rotation_matrix(cart[4], (0, 1, 0))
    Rot_z_r = tf.rotation_matrix(cart[5], (0, 0, 1))
    return np.matrix.round(((T_x_r).dot(Rot_z_r).dot(Rot_y_r).dot(Rot_x_r)), 3)

def ang(angle):
    r = radians(angle)
    r = angle
    return cos(r), sin(r)

def matrix(cart):
    dX = cart[0]
    dY = cart[1]
    dZ = cart[2]
    zC, zS = ang(cart[3])
    yC, yS = ang(cart[4])
    xC, xS = ang(cart[5])
    Translate_matrix = np.array([[1, 0, 0, dX],
                                 [0, 1, 0, dY],
                                 [0, 0, 1, dZ],
                                 [0, 0, 0, 1]])
    Rotate_X_matrix = np.array([[1, 0, 0, 0],
                                [0, xC, -xS, 0],
                                [0, xS, xC, 0],
                                [0, 0, 0, 1]])
    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                                [0, 1, 0, 0],
                                [-yS, 0, yC, 0],
                                [0, 0, 0, 1]])
    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                                [zS, zC, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return np.dot(Translate_matrix, np.dot(Rotate_Z_matrix, np.dot(Rotate_Y_matrix, Rotate_X_matrix)))



# the following is modified from examples/Python/Advanced/global_registration.py
# http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    #used to draw frame.
    #frame=o3d.geometry.TriangleMesh.create_coordinate_frame(
    #    size=0.6, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([source_temp,frame])
    #print(source)

def transform_cloud(source, transformation):
    source_temp = copy.deepcopy(source)
    return source_temp.transform(transformation)

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def calculate_fpfh(pcd, voxel_size):

    radius_normal = voxel_size * 2#original 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5#original 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def prepare_dataset(voxel_size,path_file=0):
   source = o3d.io.read_point_cloud("CE-JACKET-REDUCED-V2-PCD-25000.pcd")
    npArr = np.asarray(source.points)*0.001
    source.clear()
    source.points = o3d.utility.Vector3dVector(npArr)
    target = o3d.io.read_point_cloud(path_file)

    source.estimate_normals()
    target.estimate_normals()
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(2000000, 2000))#800000,2000 er solid on 0
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
    distance_threshold = voxel_size * 0.4 # original = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result

def perform_algorithm(voxel_size,path_to_file_target,transform):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, path_file=path_to_file_target)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    draw_registration_result(source, target,
                             result_ransac.transformation)

    result_icp = refine_registration(source_down, target, source_fpfh, target_fpfh,
                                    voxel_size,result_ransac)
    fitness_score=result_icp.fitness

    draw_registration_result(source, target, result_icp.transformation)

    transform_CJ = result_icp.transformation
    #transform_JC = np.linalg.inv(transform_CJ)

    #transform_target_OJ = xyz_fixed_rotation([0, 0, 0, 1.56, 0, 0])

    rotation1= np.asarray([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,  0,   0,   1]])
    rotation=np.linalg.inv(rotation1)
    transform_helper=np.linalg.inv(np.dot(np.dot(matrix([0, 0, 0, 0, -pi/2, 0]),matrix([0, 0, 0, pi/2, 0, 0])),matrix([0, 0, 0, 0, 0, 0])))
    trans_temp=np.dot(rotation,transform_CJ)
    transform_OJ = np.dot(transform, trans_temp)
    #print(transform_target_OJ)
    output_matrix = np.matrix.round(transform_OJ, 3)
    #print(output_matrix)
    pos_vector = output_matrix[0:3, 3:4]
    rotation_matrix_output = output_matrix[0:3, 0:3]
    #object_rot=R.from_matrix(rotation_matrix_output)
    #angles=object_rot.as_euler('zyx', degrees=True)
    #print("out_pu_matrix", output_matrix)
    #print("angles",angles)
    #print("pos_vector", pos_vector)
    angles=1
    pow(pos_vector[0]-0, 4)
    distance_orientation=1#abs(angles[0]-0)+abs(angles[1]-0)+abs(angles[2]-90)
    eucledian_distance = math.sqrt(pow(pos_vector[0]-0, 2)+pow(pos_vector[1]-0, 2)+pow(pos_vector[2]-0, 2))
    return fitness_score,eucledian_distance,distance_orientation,pos_vector,angles


    ##rotations

if __name__ == "__main__":
    t = time.time()
    voxel_size = 0.19
    path_files = []
    path_files.append("Pointcloudspointcloud_0_120520.txt.pcd")
    path_files.append("Pointcloudspointcloud_1_120520.txt.pcd")
    path_files.append("Pointcloudspointcloud_2_120520.txt.pcd")
    path_files.append("Pointcloudspointcloud_3_120520.txt.pcd")
    #path_files.append("Pointcloudspointcloud_0_120520.txt_noise_0.020000.pcd")
    #path_files.append("Pointcloudspointcloud_1_120520.txt_noise_0.020000.pcd")
    #path_files.append("Pointcloudspointcloud_3_120520.txt_noise_0.020000.pcd")
    #path_files.append("Pointcloudspointcloud_4_120520.txt_noise_0.020000.pcd")
    transform = []
    transform.append(xyz_fixed_rotation([0,2.5,3.0,0,0.45,-1]))
    transform.append(xyz_fixed_rotation([2.0,2.7,3.0,0,0.45,-1.56]))
    transform.append(xyz_fixed_rotation([5.0,-2.0,3.0,0,0.43,2.56]))
    transform.append(xyz_fixed_rotation([0.0,-3.7,3.0,0,0.45,1.1]))
    #running 1 iteration of the algoritm
    fitness, distance_points, orientation_sum, pos, angles = perform_algorithm(voxel_size, path_files[0], transform[0])

    #actual algorit performing with testing
    if False:
        path_files=np.array(path_files)
        transform=np.asarray(transform)
        poses=[]
        dist_array=[]
        ori_array=[]
        angles_array=[]
        j=0
        sum_points=0
        for i in range(0,10):
            for i in range(0,4):
                #print(i)
                skip=False
                fitness,distance_points,orientation_sum,pos,angles=perform_algorithm(voxel_size,path_files[i],transform[i])
                ##if not matching with required fitness score, we redo the algorithm.
                if fitness < 0.6:
                    #temp_voxel=voxel_size
                    fitness, distance_points, orientation_sum, pos,angles = perform_algorithm(voxel_size, path_files[i],transform[i])
                if fitness < 0.6:
                    fitness, distance_points, orientation_sum, pos,angles = perform_algorithm(voxel_size, path_files[i], transform[i])
                if fitness < 0.6:
                    fitness, distance_points, orientation_sum, pos,angles = perform_algorithm(voxel_size, path_files[i], transform[i])
                if fitness < 0.6:
                    fitness, distance_points, orientation_sum, pos,angles = perform_algorithm(voxel_size, path_files[i], transform[i])
                else:
                    skip=True

                if skip and distance_points<0.1:
                    j=j+1
                    sum_points=sum_points+distance_points
                    print("fitness",fitness)
                    #print("orientation",orientation_sum)
                    dist_array.append(distance_points)
                    ori_array.append(orientation_sum)
                    #print("distance",distance_points)
                    poses.append(pos)
                    angles_array.append(angles)
                else:
                    dist_array.append(0)
                    ori_array.append(0)
                    poses.append([0,0,0])
                    angles_array.append([0,0,90])
                    print("skip")
        print(j)
        #used for timing.
        elapsed = time.time() - t


    #registration::RegistrationResult with fitness = 0.361345, inlier_rmse = 0.032050, and correspondence_set size of 129
    #object_rot = R.from_matrix(rotation1[0:3,0:3])
    #angles = object_rot.as_euler('zyx', degrees=True)
    #print(np.matrix.round(rotation1[0:3,0:3],3))


"""
Transform for tests
       [x,y,z,roll,pitch,yaw]
    0. [0,2.5,3.0,0,0.45,-1]
    1. [2.0,2.7,3.0,0,0.45,-1.56]
    2. [5.0,0.0,3.0,0,0.43,3.14]
    3. [5.0,-2.0,3.0,0,0.43,2.56]
    4. [0.0,3.7,3.0,0,0.45,1.1]
"""



    #transform_JC = np.dot(np.linalg.inv(transform_OJ), transform_OC_00)
"""
    #calculate them differently
    transform_OC_00 = xyz_fixed_rotation([0.6, 2.5, 3.2, 0, 0.51, -1.56])
    transform_OC_01 = xyz_fixed_rotation([-2.5, -0.5, 3.2, 0, 0.46, 0])
    transform_OC_02 = xyz_fixed_rotation([1, -3.8, 3.2, 0, 0.5, 1.56])
    
    #print(transform_OJ)
    #print(transform_OJ)#4.078
    print(np.linalg.inv(transform_OC_00))#4.105
    transform_JC=np.dot(np.linalg.inv(transform_OJ),transform_OC_00)
    #transform_JC1=np.dot(transform_OJ,transform_OC_01)
    transform_CJ1=np.linalg.inv(transform_JC)
    transform_temp=matrix([0, 0, 0, 0, -pi/2, 0])
    transform_temp1=matrix([0, 0, 0, 0, 0, pi/2])
    transform=np.dot(transform_temp,transform_temp1)
    transform=np.dot(transform,transform_CJ1)
    print(transform_CJ)
    print(transform)
    #print(np.dot(np.linalg.inv(transfor
    # m_OJ),transform_CJ))
    #print(np.dot(transform_OJ,transform_CJ))
    #print(np.dot(np.linalg.inv(transform_02), transform_object))
    #print(transformation_matrix)
"""


##Tranformations


"""
[ 0.009 -1.     0.005  0.6  ]
 [ 0.873  0.011  0.488  2.5  ]
 [-0.488  0.     0.873  3.2  ]
 [ 0.     0.     0.     1.   ]]
 
[[ 0.896  0.     0.444 -2.5  ]
 [ 0.     1.     0.    -0.5  ]
 [-0.444  0.     0.896  3.2  ]
 [ 0.     0.     0.     1.   ]]
 
[[ 0.009 -1.     0.005  1.   ]
 [ 0.878  0.011  0.479 -3.8  ]
 [-0.479  0.     0.878  3.2  ]
 [ 0.     0.     0.     1.   ]]
 
[[-8.78e-01 -2.00e-03 -4.79e-01  4.80e+00]
 [ 1.00e-03 -1.00e+00  1.00e-03 -5.00e-01]
 [-4.79e-01  0.00e+00  8.78e-01  3.20e+00]
 [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00]
 Newer
 [ 0.009 -1.     0.005  0.6  ]
 [ 0.873  0.011  0.488  2.5  ]
 [-0.488  0.     0.873  3.2  ]
 [ 0.     0.     0.     1.   ]]
 
[[ 0.896  0.     0.444 -2.5  ]
 [ 0.     1.     0.    -0.5  ]
 [-0.444  0.     0.896  3.2  ]
 [ 0.     0.     0.     1.   ]
 
 [ 0.009 -1.     0.005  1.   ]
 [ 0.878  0.011  0.479 -3.8  ]
 [-0.479  0.     0.878  3.2  ]
 [ 0.     0.     0.     1.   ]]
 
[[-8.78e-01 -2.00e-03 -4.79e-01  4.80e+00]
 [ 1.00e-03 -1.00e+00  1.00e-03 -5.00e-01]
 [-4.79e-01  0.00e+00  8.78e-01  3.20e+00]
 [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00]

90 around y
180 around x

From source
3.66097573
0.5
-1.75728435
From pose estimation
-0.534515819
1.75191455
3.62040898


From source
3.66
0.5
-1.75
-3.66]
 -1.75]
 0.5]
From pose estimation
-0.5
1.75
3.66

-3.66]
 -1.75]
 0.5]

[[ 0.89604301 -0.44402131 -0.00488423  3.66097573]
 [-0.          0.011      -1.          0.5       ]
 [ 0.44402131  0.89604301  0.00985647 -1.75728435]
 [ 0.          0.          0.          1.        ]]
[[-8.97451624e-01  3.86592668e-03  4.41095951e-01 -3.62111817e+00]
 [ 4.41095831e-01 -9.29796803e-04  8.97459528e-01 -1.75211773e+00]
 [ 3.87964234e-03  9.99992095e-01 -8.70796490e-04  5.33797260e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
 
 
 
 
 Rotation matrix
    [[0.1250   -0.9534    0.2747  0]
     [0.0553   -0.2697   -0.9613  0]
     [0.9906    0.1353    0.0190  0]
     0  0   0   1]]
"""
