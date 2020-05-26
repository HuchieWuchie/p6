import numpy as np
from numpy import diff
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm

from pointCloud import PointCloud
from pointCloudLine import PointCloudLine
from quality import Quality
from net import Net


with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

def pcdLoader(file_name = "", afterPIT=False, load=True):
    file_name_npy = (file_name + cfg['pcd_file'].get("ext_npy"))
    file_name_pcd = (file_name + cfg['pcd_file'].get("ext_pcd"))
    file_name_txt = (file_name + cfg['pcd_file'].get("ext_txt"))
    if afterPIT == False:
        processed_file = (cfg['pcd_file'].get("path_processed") + file_name_npy)
        unprocessed_file_pcd = (cfg['pcd_file'].get("path_unprocessed") + file_name_pcd)
        unprocessed_file_txt = (cfg['pcd_file'].get("path_unprocessed") + file_name_txt)

    else:
        processed_file = (cfg['after_file'].get("path_processed") + file_name_npy)
        unprocessed_file_pcd = (cfg['after_file'].get("path_unprocessed") + file_name_pcd)
        unprocessed_file_txt = (cfg['after_file'].get("path_unprocessed") + file_name_txt)

    if load==True:
        if os.path.exists(processed_file) == True:
            print(processed_file)
            npArr = np.load(processed_file)
            pcd = PointCloud(npArr, type = 'NPARR')
            pcd.getParameters()
            return pcd
        else:
            print(file_name)
            print("Couldn't find processed point cloud.")
    else:
        if os.path.exists(unprocessed_file_txt) == True:
            print(unprocessed_file_txt)
            npArr = np.loadtxt(fname = unprocessed_file_txt)
            npArr = np.delete(npArr, 3, 1)
            pcd = PointCloud(npArr, type='NPARR')
            print("Preprocessing point cloud...")
            pcd.deleteZPoints(0)
            #pcd.setOffset(0, 0, -120)
            #pcd.pcd = o3d.geometry.PointCloud.remove_statistical_outlier(pcd.pcd, 1000, 1)[0]
            pcd.getParameters()
            np.save(processed_file, np.asarray(pcd.pcd.points))
            return pcd


        elif os.path.exists(unprocessed_file_pcd) == True:
            print(unprocessed_file_pcd)
            pcd = PointCloud(unprocessed_file_pcd, type='PCD')
            print("Preprocessing point cloud...")
            pcd.deleteZPoints(0) # Delete all points with z = 0
            #pcd.setOffset(0, 0, -120)
            #pcd.pcd = o3d.geometry.PointCloud.remove_statistical_outlier(pcd.pcd, 1000, 1)[0]
            pcd.getParameters()
            np.save(processed_file, np.asarray(pcd.pcd.points))
            return pcd

        else:
            print(file_name)
            print("Couldnt find unprocessed point cloud.")
            raise
            return False

def roiLoader(file_name = ""):
    if os.path.exists(file_name) == True:
        print("Loading regions of interest already found.")
        ROI = np.load(file_name)
        return ROI

def getROI(pcdNormalized, net, FILE_NAME="", afterPIT=False,):
    ROI = []
    print("Finding regions of interest...")
    for points in tqdm(pcdNormalized.pointCloudLine):
        pcl = PointCloudLine(points, 1)

        for i in range(len(pcl.x)):
            if i > cfg['model_after'].get("step_size") and i < (len(pcl.x)- cfg['model_after'].get("step_size")):

                x = pcl.getFeatureVectorZOnly(i, cfg['model_after'].get("step_size"))
                x = torch.from_numpy(x)


                if net.predict_threshold(x.float(), cfg['model_after'].get("confidence_threshold")) == 1:
                    ROI.append(np.array([pcl.x[i], pcl.y[0], pcl.z[i]]))
    ROI = np.array(ROI)
    if afterPIT == True:
        print((cfg['after_file'].get("ROI") + FILE_NAME + cfg['after_file'].get("ext_npy"), " saved."))
        np.save((cfg['after_file'].get("ROI") + FILE_NAME + cfg['after_file'].get("ext_npy")), ROI)
    else:
        np.save((cfg['pcd_file'].get("ROI") + FILE_NAME + cfg['pcd_file'].get("ext_npy")), ROI)
    return ROI


def loadClassifier():
    print("Loading classifier...")
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
        net = Net(cfg['model'].get("feature_size"), device=device).to(device)

    else:
        device = torch.device("cpu")
        net = Net(cfg['model'].get("feature_size"), device=device).to(device)
        print("Running on the CPU")

    net.load_state_dict(torch.load((cfg['model'].get("path")+cfg['model'].get("model_name")), map_location=torch.device('cpu')))
    net = net.float()
    return net
