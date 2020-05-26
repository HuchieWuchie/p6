import numpy as np
import open3d as o3d
from tqdm import tqdm
from skimage import data
from skimage.measure import label, regionprops

class PointCloud:
    def __init__(self, pcd_values, type = 'PCD'):
        if type == 'PCD':
            self.pcd = o3d.io.read_point_cloud(pcd_values)
        elif type == 'NPARR':
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pcd_values)
        self.getParameters()

    def deleteZPoints(self, value):
        """
        Delete all z values equal to given 'value'
        """
        npArr = []
        for point in np.asarray(self.pcd.points):
            if point[2] != value:
                point[2] = point[2]
                point[2] = abs(point[2])
                npArr.append(point)
        npArr = np.array(npArr)
        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(npArr)

    def setOffset(self, xOff, yOff, zOff):
        npArr = []
        for point in np.asarray(self.pcd.points):
            point[0] += xOff
            point[1] += yOff
            point[2] += zOff
            npArr.append(point)
        npArr = np.array(npArr)
        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(npArr)

    def normalize(self):
        """
        All z-values are set to between 0 and 1
        """
        npArr = []
        pointcloud = np.asarray(self.pcd.points)
        localMin = np.amin(pointcloud[:][:,2])
        localMax = np.amax(pointcloud[:][:,2])
        for point in np.asarray(self.pcd.points):
            point[2] = (point[2]-localMin)/(localMax-localMin)
            npArr.append(point)
        npArr = np.array(npArr)
        return npArr

    def unNormalize(self):
        npArr = []
        for point in np.asarray(self.pcd.points):
            point[2] = (point[2]*self.z_max)-(point[2]*self.z_min)+self.z_min
            npArr.append(point)
        npArr = np.array(npArr)
        return npArr


    def setColor(self, color):
        if color == 'r':
            self.pcd.paint_uniform_color([1, 0, 0])
        if color == 'g':
            self.pcd.paint_uniform_color([0, 1, 0])
        if color == 'b':
            self.pcd.paint_uniform_color([0, 0, 1])
        if color == 'w':
            self.pcd.paint_uniform_color([1, 1, 1])
        if color == 'black':
            self.pcd.paint_uniform_color([0, 0, 0])

    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd])

    def getPointCloudLines(self):
        """
        Computes the point cloud lines as a numpy array. format [i][j][xyz]
        """
        npArr = []
        self.pointCloudLine = []
        i = 0
        y = 0
        holder = 0
        for point in tqdm(np.asarray(self.pcd.points)):
            y = point[1]
            if i == 0:
                holder = point[1]
                i = i + 1

            if y != holder:
                holder = y
                line = np.array(npArr)
                self.pointCloudLine.append(np.array(line))
                npArr.clear()
            npArr.append(point)

        self.pointCloudLine = np.array(self.pointCloudLine)

    def size_threshold_pointCloudLine(self, threshold = 0.7):
        npArr = []
        for line in self.pointCloudLine:
            if len(line) > (self.i_max*threshold):
                for point in line:
                    npArr.append(np.array(point))
        npArr = np.array(npArr)
        return npArr


    def pcd_from_pointCloudLines(self, npArr):

        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(npArr)
        self.getPointCloudLines()
        self.getParameters()

    def getParameters(self):
        """
        Computes the parameters of a point cloud, eg. largest and smallest x val
        """
        holder = 0
        pointcloud = np.asarray(self.pcd.points)
        self.z_values = [pointcloud[:][:,2]]
        self.z_min = np.amin(pointcloud[:][:,2])
        self.z_max = np.amax(pointcloud[:][:,2])
        self.y_max = np.amax(pointcloud[:][:,1])
        self.y_min = np.amin(pointcloud[:][:,1])
        self.x_max = np.amax(pointcloud[:][:,0])
        self.x_min = np.amin(pointcloud[:][:,0])
        self.i_max, i, self.y_range, line, k = 0, 0, -1, [], 0
        for points in tqdm(self.pcd.points):
            if holder != points[1]:
                npLine = np.array(line)
                if len(npLine) != 0:
                    if self.x_min > np.amin(npLine):
                        self.x_min = np.amin(npLine)
                    if self.x_max < np.amax(npLine):
                        self.x_max = np.amax(npLine)
                    if self.i_max < i:
                        self.i_max = i
                line[:] = []
                i = 0
                k = 1
                self.y_range += 1
                holder = points[1]
            if holder == points[1]:
                line.append(points[0])
                i += 1

        self.x_resolution = (abs(self.x_min)+self.x_max)/self.i_max
        self.y_resolution = (self.y_max-self.y_min)/self.y_range


    def pcd_to_image(self, span=0.5):
        """
        Returns a 2D matrix of the point cloud where all the values of the matrix
        are the z values
        """
        self.getParameters()
        matrix = np.zeros(self.i_max*self.y_range)
        matrix = np.reshape(matrix, (self.y_range, -1))

        for i in tqdm(range(len(matrix))):
            for j in range(self.i_max):
                value = self.x_max - (j * self.x_resolution)
                x_near = self.find_nearest(self.pointCloudLine[i][:][:,0], value)
                result = np.where(self.pointCloudLine[i][:][:,0] == x_near)


                if len(result[0]) != 0:
                    index = result[0][0]
                    matrix[i, j] = self.pointCloudLine[i][index][2]
        return matrix

    def mask_image(self, img, npArr, span=0.5):
        """
        masks a 2D image given a numpy array
        """
        matrix = np.zeros(self.i_max*self.y_range)
        matrix = np.reshape(matrix, (self.y_range, -1))
        yArr = []
        for i in range(self.y_range):
            yArr.append(round(self.pointCloudLine[i][0][1],1))
        yArr = np.array(yArr)
        for i in tqdm(range(len(npArr))):
            try:
                z = npArr[i][2]
                y = round(npArr[i][1],1)
                y_index = np.where(yArr == y)
                #if len(y_index[0]):
                y_index = y_index[0][0]
                z_near = self.find_nearest(img[y_index, :], z)
                x_index = np.where(img[y_index, :] == z_near)
                x_index = x_index[0][0]

                matrix[y_index, x_index] = z
            except Exception as e:
                pass

        return matrix

    def img_to_pcd(self, matrix):
        """
        Returns a numpy array [x, y, z] from a 2D matrix input [i, j] = z
        """
        npArr = []
        for i in range(self.y_range):
            for j in range(self.i_max):
                npArr.append([self.x_max-(j*self.x_resolution), (i*self.y_resolution)+self.y_min, matrix[i,j]])
        npArr = np.array(npArr)
        return npArr

    def mask_pcd(self, mask, border=0):
        """
        Returns a point cloud with the values from the 2D mask + a border.
        """
        npArr = []
        j, i = 0, 0
        for line in mask:
            xStart = np.amin(line[:][:,1])
            xEnd = np.amax(line[:][:,1])
            x_len = abs(abs(xEnd)-abs(xStart))+(2*border)

            if border != 0:
                for k in range(x_len):
                    index_x = (xStart - border) + k
                    try:
                        npArr.append(self.pointCloudLine[line[0][0]][index_x])
                    except Exception as e:
                        pass

            else:
                npArr.append(self.pointCloudLine[mask[i][0]][mask[i][1]])
            j +=1
            i +=1

        npArr = np.array(npArr)

        return npArr

    def thresArea(self, label_image, thres):
        """
        Returns list of regions with areas larger than given threshold
        """
        blob = []
        for region in regionprops(label_image):
            if region.area > thres:
                blob.append(region)
        return blob

    def getMinXRegion(self, regions):
        """
        returns the the label of the region with the lowest x centroid value
        """
        x_min = 2000
        region_left = 0
        for region in regions:
            if x_min > region.centroid[1]:
                x_min = region.centroid[1]
                region_left = region

        return region_left

    def getMaxXRegion(self, regions):
        """
        returns the the label of the region with the highest x centroid value
        """
        x_max = 0
        region_right = 0
        for region in regions:
            if x_max < region.centroid[1]:
                x_max = region.centroid[1]
                region_right = region
        return region_right

    def fillEmptyPointCloudLines(self, region, pointCloudLine):

        yArr = []
        holder = -1
        for point in region:
            y = point[0]
            if holder != y:
                holder = y
                yArr.append(y)
        yArr = np.array(yArr)
        if len(pointCloudLine) > len(yArr):
            xMin = np.amin(region[:][:,1])
            xMax = np.amax(region[:][:,1])
            yIdx = []
            for i in range(len(pointCloudLine)):
                result = np.where(yArr[:] == i)
                if len(result[0]) == 0:
                    yIdx.append(i)
            yIdx = np.array(yIdx)
            npArr = []
            for i in range(len(yIdx)):
                xArr = np.arange(xMin, xMax)
                xArr = xArr.reshape(-1,1)
                yArr = np.full((xMax-xMin), yIdx[i])
                yArr = yArr.reshape(-1,1)
                arr = np.hstack((yArr, xArr))
                npArr.append(np.array(arr))
            npArr = np.array(npArr)
            npArr = npArr.reshape(-1, 2)
            region = np.concatenate((region, npArr), axis = 0)
            return region
        else:
            return region


    def getMask(self, regions, pointCloudLine):
        """
        returns coordinates for the welding regions based on the min/max x centroid value
        """
        weldRegions = []
        weldRegions.append(self.getMaxXRegion(regions))
        weldRegions.append(self.getMinXRegion(regions))
        for i in range(len(weldRegions)):
            xMin = np.amin(np.array(weldRegions[i].coords[:,1]))
            xMax = np.amax(np.array(weldRegions[i].coords[:,1]))
            centroid = weldRegions[i].centroid[1]
            weldRegions[i] = np.array(weldRegions[i].coords)

            #Check if more blobs belong to the identified regions
            for region in regions:
                if region.centroid[1] < xMax and region.centroid[1] > xMin:
                    if region.centroid[1] == centroid:
                        pass
                    else:
                        weldRegions[i] = np.concatenate((weldRegions[i], np.array(region.coords)), axis=0)

        for i in range(len(weldRegions)):
            weldRegions[i] = self.fillEmptyPointCloudLines(weldRegions[i], pointCloudLine)
        return weldRegions

    def flip(self):
        """
        Flips all z values.
        """
        npArr = []
        for point in np.asarray(self.pcd.points):
            point[2] = -point[2]
            npArr.append(point)
        npArr = np.array(npArr)
        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(npArr)

    def maskToLines(self, mask):
        npArr, maskLines = [], []
        y, holder = 0, -1,
        for point in mask:
            y = point[0]
            if y != holder:
                holder = y
                if len(npArr) != 0:
                    line = np.array(npArr)
                    maskLines.append(line)
                npArr.clear()
            try:
                npArr.append(np.array(point))
            except Exception as e:
                pass

        maskLines = np.array(maskLines)
        return maskLines

    def find_nearest(self, array, value):
        """
        Finds nearest value in a numpy array
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
