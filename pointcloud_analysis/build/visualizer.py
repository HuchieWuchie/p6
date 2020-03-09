import numpy as np
import open3d as o3d

FILE_NAME_VERTS = "pointcloud.txt"
npArr = np.loadtxt(fname = FILE_NAME_VERTS)
verts = o3d.geometry.PointCloud()
verts.points = o3d.utility.Vector3dVector(npArr)
o3d.visualization.draw_geometries([verts])

## Visualize raw
FILE_NAME_RAW = "04-01-test-scan-2.txt"
npArr = np.loadtxt(fname = FILE_NAME_RAW)
npArr = np.delete(npArr, 3, 1)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(npArr)
npArr = np.asarray(pcd.points)
npArr = np.delete(npArr, np.where(npArr[:][:,2] == 0)[0], 0)
pcd.clear()
pcd.points = o3d.utility.Vector3dVector(npArr)
#o3d.visualization.draw_geometries([pcd])

## Visualize normalized

FILE_NAME_NORMALIZED = "Normalized.txt"
npArr = np.loadtxt(fname = FILE_NAME_NORMALIZED)
normalized = o3d.geometry.PointCloud()
normalized.points = o3d.utility.Vector3dVector(npArr)
#o3d.visualization.draw_geometries([normalized])

## Visualize grid

FILE_NAME_GRID = "grid.txt"
npArr = np.loadtxt(fname = FILE_NAME_GRID)
grid = o3d.geometry.PointCloud()
grid.points = o3d.utility.Vector3dVector(npArr)
#o3d.visualization.draw_geometries([grid])

## Visualize roi

FILE_NAME_ROI = "regions_of_interest.txt"
npArr = np.loadtxt(fname = FILE_NAME_ROI)
ROI = o3d.geometry.PointCloud()
ROI.points = o3d.utility.Vector3dVector(npArr)
npArr[:][:,2] = npArr[:][:,2] - 0.3
ROI.paint_uniform_color([1, 0.5, 0.8])
#o3d.visualization.draw_geometries([ROI])
#o3d.visualization.draw_geometries([normalized, ROI])
o3d.visualization.draw_geometries([grid, ROI])

## Visualize weld seams

FILE_NAME_WELD0 = "weldseam0.txt"
FILE_NAME_WELD1 = "weldseam1.txt"
npArr = np.loadtxt(fname = FILE_NAME_WELD0)
npArr[:][:,2] = npArr[:][:,2] - 0.01
weld0 = o3d.geometry.PointCloud()
weld0.points = o3d.utility.Vector3dVector(npArr)
weld0.paint_uniform_color([1, 0, 1])


npArr = np.loadtxt(fname = FILE_NAME_WELD1)
npArr[:][:,2] = npArr[:][:,2] - 0.01
weld1 = o3d.geometry.PointCloud()
weld1.points = o3d.utility.Vector3dVector(npArr)
weld1.paint_uniform_color([1, 0, 1])
#o3d.visualization.draw_geometries([normalized, weld0, weld1])
o3d.visualization.draw_geometries([ROI, weld0, weld1])
o3d.visualization.draw_geometries([grid, weld0, weld1])

#Visualize trajectories

FILE_NAME_TRAJ0 = "trajectory_0.txt"
FILE_NAME_TRAJ1 = "trajectory_1.txt"
npArr = np.loadtxt(fname = FILE_NAME_TRAJ0)
npArr[:][:,2] = npArr[:][:,2] - 0.05
trajectory0 = o3d.geometry.PointCloud()
trajectory0.points = o3d.utility.Vector3dVector(npArr)
trajectory0.paint_uniform_color([0, 0, 1])

npArr = np.loadtxt(fname = FILE_NAME_TRAJ1)
npArr[:][:,2] = npArr[:][:,2] - 0.05
trajectory1 = o3d.geometry.PointCloud()
trajectory1.points = o3d.utility.Vector3dVector(npArr)
trajectory1.paint_uniform_color([0, 0, 1])


o3d.visualization.draw_geometries([pcd, trajectory0, trajectory1])
#o3d.visualization.draw_geometries([weld0, weld1, trajectory0, trajectory1])
#o3d.visualization.draw_geometries([grid, trajectory0, trajectory1])
