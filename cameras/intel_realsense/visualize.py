import numpy as np
import open3d as o3d

FILE_NAME_VERTS = "data/pointcloud_01.txt"
npArr = np.loadtxt(fname = FILE_NAME_VERTS)
verts = o3d.geometry.PointCloud()
verts.points = o3d.utility.Vector3dVector(npArr)
o3d.visualization.draw_geometries([verts])
