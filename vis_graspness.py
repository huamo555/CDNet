import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
import numpy as np
import torch
import sys
from pointnet2_utils import furthest_point_sample
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image

data_path = "/data3/gaoyuming/project/datasets/datasets/dataset-data/"
scene_id = 'scene_0164'
ann_id = '0000'
camera_type = 'realsense'
color = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'rgb', ann_id + '.png')), dtype=np.float32) /255
# depth = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'depth', ann_id + '.png')))
# depth = np.array(Image.open(os.path.join(data_path, 'depth_dan', scene_id, camera_type, ann_id + '.png')))
depth = np.array(Image.open("/data3/gaoyuming/Graspgan_quan/depthguji_quan_huigui_512/logs/0703_re_512_test/scene_0164/realsense/depth/0000.png"))
seg = np.array(Image.open(os.path.join(data_path, 'scenes', scene_id, camera_type, 'label', ann_id + '.png')))
meta = scio.loadmat(os.path.join(data_path, 'scenes', scene_id, camera_type, 'meta', ann_id + '.mat'))
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
depth_mask = (depth > 0)
camera_poses = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
align_mat = np.load(os.path.join(data_path, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
trans = np.dot(align_mat, camera_poses[int(ann_id)])
workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=True, outlier=0.02)

color = color 

mask = (depth_mask)
point_cloud = point_cloud[mask]
color = color[mask]
seg = seg[mask]

""" graspness_full = np.load(os.path.join(data_path, 'graspness_label', scene_id, camera_type, ann_id + '.npy')).squeeze()
graspness_full[seg == 0] = 0.
print('graspness full scene: ', graspness_full.shape, (graspness_full > 0.1).sum())
color[graspness_full > 0.1] = [0., 1., 0.] """


cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))

o3d.io.write_point_cloud("/data3/gaoyuming/Graspgan_quan/depthguji_quan_huigui_512/view/quan_0164.pcd", cloud) 




""" import open3d as o3d
import numpy as np


# 读取.pcd文件
pcd = o3d.io.read_point_cloud("C:/Users/Administrator/Desktop/s4.pcd")
num_points = np.asarray(pcd.points).shape[0]

# 设置灰色
gray_color = [0.7, 0.7, 0.7]  # RGB颜色值，范围为[0, 1]

# 设置所有点的颜色
pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(gray_color), (num_points, 1)))

# 绘制点云

# 显示点云
o3d.visualization.draw_geometries([pcd])  """