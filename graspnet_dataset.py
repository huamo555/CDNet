""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask


class GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label

        if split == 'train':
            self.sceneIds = list(range(0,100))
        elif split == 'train1':
            self.sceneIds = list(range(30, 31))
        elif split == 'all':
            self.sceneIds = list(range(0, 190))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test1':
            self.sceneIds = list(range(156, 157))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.depthpath = []
        self.rgbpath = []
        # self.real_depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.real_depthpath_clean = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                # self.real_depthpath.append(os.path.join(root, 'depth', x, camera, str(img_num).zfill(4)+'.png'))
                self.real_depthpath_clean.append(os.path.join(root, 'depth_clean', x, camera, str(img_num).zfill(4)+'.png'))
                self.rgbpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.graspnesspath.append(os.path.join(root, 'graspness_label', x, camera, str(img_num).zfill(4) + '.npy'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):


        ret_dict = {
                    }
        return ret_dict

    def get_data_label(self, index):
        rgb = np.array(Image.open(self.rgbpath[index]))
        real_depth_clean = np.array(Image.open(self.real_depthpath_clean[index]))  # my
        depth = np.array(Image.open(self.depthpath[index]))  # src
        seg = np.array(Image.open(self.labelpath[index]))
        objectness_label = seg.copy()

        objectness_label[objectness_label > 1] = 1

        neg_zero_mask = np.where(real_depth_clean == 0, 0, 1).astype(np.uint8)
        yuan_zero_mask_TRAIN = np.where(depth == 0, 0, 1).astype(np.uint8)
        yuan_zero_mask_TEST= np.where(depth == 0, 1, 0).astype(np.uint8)
        # label_mask = np.where(label != 0, 1, 0).astype(np.uint8)

        if (depth > 899).any():  # 检查是否有元素大于 999
            depth[depth > 899] = 899

        # if (real_depth_clean > 899).any():  # 检查是否有元素大于 999
            # real_depth_clean[real_depth_clean > 899] = 899
            
        # real_depth_min = real_depth_clean.min()
        # real_depth_max = real_depth_clean.max() 

        num_classes = 900
        # depth_class_labels = ((real_depth_clean / 900) * (num_classes))

        ret_dict = {
                    'depth': depth.astype(np.int64),
                    'real_depth': real_depth_clean.astype(np.int64),
                    'rgb': rgb.astype(np.int64),
                    'loss_mask': neg_zero_mask.astype(np.int64),
                    'yuan_zero_mask_TRAIN': yuan_zero_mask_TRAIN.astype(np.int64),
                    'yuan_zero_mask_TEST': yuan_zero_mask_TEST.astype(np.int64),
                    # 'depth_min': real_depth_min.astype(np.int64),
                    # 'depth_max': real_depth_max.astype(np.int64),
                    # 'depth_class_labels': depth_class_labels.astype(np.int64),
                    'objectness_label': objectness_label.astype(np.int64),
                    }
        return ret_dict




