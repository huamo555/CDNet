import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.DFNet import DFNet
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=19998, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution') # 表示体素的大小。cfgs.voxel_size是一个配置参数，用于指定体素的分辨率。
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mask0_1 = np.zeros((720, 1280), dtype=int)

# 填充掩码
for i in range(720):
    for j in range(1280):
        # 使用按位异或操作来交替填充1和0
        mask0_1[i, j] = (i ^ j) & 1

mask0_1 = torch.tensor(mask0_1)
mask0_1 = mask0_1.to(device)
print(mask0_1.device)
mask0_1 = mask0_1.unsqueeze(0)
mask_inverted = 1 - mask0_1

def inference():
    # 创建了一个名为test_dataset的数据集对象，用于在GraspNet数据集上进行测试
    # remove_outlier=True：表示是否移除离群点。如果设置为True，会进行离群点的移除操作
    # augment=False：表示是否进行数据增强。如果设置为True，会对数据进行增强操作，如旋转、平移或缩放等
    # load_label=False：表示是否加载标签信息。如果设置为True，会加载样本的标签信息，否则只加载点云数据
    test_dataset = GraspNetDataset(cfgs.dataset_root, split='all', camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=True)
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    # 创建一个数据加载器（test_dataloader），用于批量加载测试数据。根据提供的配置参数，设置批量大小、是否打乱数据、工作进程数量等
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn)
    print('Test dataloader length: ', len(test_dataloader))
    # Init the model 初始化模型（net）并将其移动到可用的设备（GPU或CPU）上
    net_DF = DFNet()
    net_DF.to(device)

    # Load checkpoint 加载预训练的模型参数（checkpoint）
    print(cfgs.checkpoint_path)
    checkpoint = torch.load(cfgs.checkpoint_path)
    # net_DF.load_state_dict(checkpoint['model_state_dict'])
    net_DF.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    
    # 批量处理的间隔（batch_interval）和模型的评估模式（net.eval()）
    net_DF.eval()
    # 遍历测试数据加载器，逐批加载数据（batch_data），
    # 将其移动到设备上，并进行前向传播。使用模型对数据进行推断，得到抓取预测结果（grasp_preds）
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            mask = mask0_1 & (batch_data["yuan_zero_mask_TRAIN"] == 1) & (batch_data["objectness_label"] == 1)
            batch_data['rgb'] = batch_data['rgb'].permute(0, 3, 1, 2)   # torch.Size([2, 720, 1280, 3]) -> torch.Size([2, 3, 720, 1280])
            # zero_depth = torch.zeros_like(batch_data_label['depth'])
            res, obj = net_DF(batch_data['rgb'], batch_data['depth'])
            _, predicted_class_indices = torch.max(res, dim=1)
            
            final = predicted_class_indices 
            final_depth = final * mask 

            mask_yuan = mask_inverted & (batch_data["yuan_zero_mask_TRAIN"] == 1) & (batch_data["objectness_label"] == 1)

            no_objmask = (batch_data["objectness_label"] == 0)

            final_depth = final_depth + mask_yuan * batch_data["depth"] + no_objmask * batch_data["depth"]

            print("11")
            final_depth = final_depth.cpu()
            final_depth = final_depth.numpy()
            final_depth = final_depth[0]
            reconstructed_depth = np.zeros((720, 1280))

            reconstructed_depth= final_depth 
            depth_map_uint8 = reconstructed_depth.astype(np.int32)
            depth_image = Image.fromarray(depth_map_uint8, 'I')
            print(depth_image)

        # Dump results for evaluation
        # 对每个样本进行处理。将抓取预测结果转换为NumPy数组，并创建一个GraspGroup对象（gg）来表示抓取
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera, "depth")
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.png')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            depth_image.save(save_path)
            print("保存成功")



if __name__ == '__main__':   
    inference()

