import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
# 加载点云数据
np_data = np.load("/data2/gaoyuming/.cache/datasets/dataset-data/grasp_label/000_labels.npz")
# 将二维点云转化为三维点云
# points = np.hstack((points,np.zeros((points.shape[0],1))))
# 创建点云对象
print(np_data.shape)
pcd = o3d.geometry.PointCloud()
# 将点云数据添加到点云对象中
pcd.points = o3d.utility.Vector3dVector(np_data)
# 可视化点云
o3d.io.write_point_cloud("/data2/gaoyuming/.cache/graspnet-baseline-main/view/new3.pcd", pcd)