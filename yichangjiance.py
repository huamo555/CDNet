from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import torch
from multiprocessing import Pool
from models.DFNet import DFNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
# parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=19998, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution') # 表示体素的大小。cfgs.voxel_size是一个配置参数，用于指定体素的分辨率。
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()

def process_scene(scene_id, img_num ,camera):
    # 仅处理指定场景和图像编号范围的函数
    yichang_depth_path = "/home/data3/gaoyuming/datasets/dataset-data/scenes/{}/{}/depth/{}.png"
    quan_depth_path = "/home/data3/gaoyuming/datasets/dataset-data/depth_clean_dan0/{}/{}/{}.png"

    yichang_depth_path = yichang_depth_path.format(scene_id, camera, "0005")
    yichang_depth = np.array(Image.open(yichang_depth_path))
    quan_depth_path = quan_depth_path.format(scene_id, camera, "0005")
    quan_depth = np.array(Image.open(quan_depth_path))

    mask = np.zeros_like(yichang_depth)
    mask[quan_depth != 0 ] = 1
    print(mask.sum())

    threshold = 30
    # 创建一个与 yichang_depth 相同大小的张量，用于标记异常值
    anomaly_mask = np.zeros_like(yichang_depth)
    difference = np.abs(yichang_depth - quan_depth)
    # 将差异超过阈值的位置标记为1
    anomaly_mask[(difference > threshold) &  quan_depth != 0 ] = 1
    print(anomaly_mask.sum())

    one_mask = np.ones_like(yichang_depth)
    one_mask[(difference > threshold) &  quan_depth != 0 ] = 0
    chuli_depth = yichang_depth * anomaly_mask
    print(one_mask.sum())

    chuli_depth_uint8 = chuli_depth.astype(np.int32)
    chuli_depth_image0005 = Image.fromarray(chuli_depth_uint8, 'I')

    chuli_depth_image0005.save("/home/data3/gaoyuming/mutilview_project/graspness_depthguji/view/chuli_depth_image0005.png")

    print("1")

    """ depth_map_uint8 = reconstructed_depth.astype(np.int32)
    depth_image = Image.fromarray(depth_map_uint8, 'I')
    depth_save_path = depth_save_path_template.format(scene_id, camera, str(img_num).zfill(4))
    directory = os.path.dirname(depth_save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    depth_image.save(depth_save_path) """

def main():
    root = "/home/data3/gaoyuming/datasets/dataset-data/"
    camera = 'realsense'
    sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in range(0, 190)]
    for scene_id in sceneIds:
        for img_num in range(256):
            process_scene(scene_id, img_num, camera)


    num_processes = 40  # 根据你的CPU核心数来设置进程数 # 
    # 将场景分配给不同的进程
    for scene_id in sceneIds:
        pool = Pool(num_processes)
        for img_num in range(256):  # 假设每个场景有256张图片
            pool.apply_async(process_scene, args=(scene_id, img_num, camera))
        pool.close()
        pool.join()

if __name__ == "__main__":
    main() 