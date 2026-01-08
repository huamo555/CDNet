import torch.nn as nn
import torch
import torch.nn.functional as F
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import numpy as np
import open3d as o3d

### 分类 + 高斯
criterion = nn.CrossEntropyLoss(reduction='none')

# def point_pipei(pred_mask_pred, gt, data_dict, mask):
#     seg = data_dict['seg']
#     intrinsic = data_dict['intrinsic']
#     factor_depth = data_dict['factor_depth']
#     intrinsic = intrinsic.squeeze(0)
#     factor_depth = factor_depth.squeeze(0)

#     mask_no0 = seg != 0
#     gt0  = gt.squeeze(0)
#     mask0 = mask.squeeze(0)
#     pred_mask_pred0 = pred_mask_pred.squeeze(0)

#     gt0 = gt0.cpu().numpy()
#     intrinsic = intrinsic.cpu().numpy()
#     mask0 = mask0.cpu().numpy()
#     mask_no0 = mask_no0.cpu().numpy()
#     pred_mask_pred0 = pred_mask_pred0.cpu().numpy()

#     factor_depth = factor_depth.cpu().numpy()
#     camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
#     cloud_gt = create_point_cloud_from_depth_image(gt0, camera, organized=True)
#     cloud_pred = create_point_cloud_from_depth_image(pred_mask_pred0, camera, organized=True)
#     mask_no0 = mask_no0.squeeze(0)
#     mask_f = mask_no0 & mask0
#     cloud_gt = cloud_gt[mask_f]
#     cloud_pred = cloud_pred[mask_f]
#     seg = seg.squeeze(0)
#     seg_mask = seg[mask_f]
#     if len(cloud_gt) >= 20000:
#             idxs = np.random.choice(len(cloud_gt), 20000, replace=False)
#     else:
#             idxs1 = np.arange(len(cloud_gt))
#             idxs2 = np.random.choice(len(cloud_gt), 20000 - len(cloud_gt), replace=True)
#             idxs = np.concatenate([idxs1, idxs2], axis=0)

#     cloud_gt = cloud_gt[idxs]
#     cloud_pred = cloud_pred[idxs]
#     seg_mask_f = seg_mask[idxs]
#     cloud_gt = torch.from_numpy(cloud_gt)
#     cloud_pred = torch.from_numpy(cloud_pred)
#     return cloud_gt, cloud_pred, seg_mask_f


# def get_loss(data_dict, mask):
#     # objectness_loss, data_dict = compute_objectness_loss(data_dict)
#     pred = data_dict['pred'] 
#     gt = data_dict['real_depth']
#     loss = criterion(pred + data_dict['depth'], gt) 

#     loss = loss[mask]
#     loss = loss.mean()
#     mask_bool = mask.bool()
#     pred_mask_pred = torch.argmax(pred, 1)
#     pred_v = pred_mask_pred & mask_bool
#     loss_v = masked_anti_smooth_loss(pred_v)
#     print(loss_v)

#     data_dict['pred_acc'] = (torch.abs(gt[mask_bool] - pred_mask_pred[mask_bool].long()) < 5).float().mean()

#     cloud_gt, cloud_pred, seg_mask_f = point_pipei(pred_mask_pred, gt, data_dict, mask)
#     unique_values = torch.unique(seg_mask_f)
#     losses = []  # 用于存储每个损失值的列表
#     for i, value in enumerate(unique_values):
#         # 找到值为 value 的元素的索引
#         indices = torch.where(seg_mask_f == value)
#         cloud_pred_m = cloud_pred[indices].unsqueeze(0)
#         cloud_gt_m = cloud_gt[indices].unsqueeze(0)
#         cloud_pred_m.requires_grad_(True)
#         cloud_gt_m.requires_grad_(True)
#         total_loss = enhanced_chamfer_loss_only(cloud_pred_m, cloud_gt_m)
#         losses.append(total_loss)  # 存储需要梯度的张
    
#     total_loss = torch.sum(torch.stack(losses))
#     print("点云相似性", total_loss)
#     print("交叉熵", loss)
#     # print("平滑", loss_v)
#     # loss_all = loss * 20 + total_loss * 200 + loss_v
#     loss_all = loss * 20 + total_loss * 200 
#     return loss_all, data_dict

# def chamfer_distance(pred, gt):
#     # pred/gt: (B, N, 3)/(B, M, 3)
#     dist = torch.cdist(pred, gt)  # (B, N, M)
#     min_dist1 = torch.min(dist, dim=2)[0]  # pred->gt
#     min_dist2 = torch.min(dist, dim=1)[0]  # gt->pred
#     return (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()


# def enhanced_chamfer_loss_only(pred_pc, gt_pc):
   
#     # 计算Chamfer距离
#     dist = pred_pc.unsqueeze(2) - gt_pc.unsqueeze(1)  # (B, N, M, 3)
#     dist_matrix = torch.sum(dist**2, dim=-1)  # (B, N, M)
    
#     # 最近邻搜索
#     min_dist1, idx1 = torch.min(dist_matrix, dim=2)  # pred->gt (B, N)
#     min_dist2, idx2 = torch.min(dist_matrix, dim=1)  # gt->pred (B, M)
    
#     chamfer_loss = min_dist1.mean() + min_dist2.mean()
    
#     total_loss = chamfer_loss 
    
#     return total_loss


# def masked_anti_smooth_loss(pred_depth, margin=3, eps=1e-6):
#     """
#     改进版反平滑损失，排除深度值为0的区域
#     pred_depth: [B, 1, H, W]
#     margin: 最小允许差异阈值
#     eps: 防止除零的小量
#     """
#     # 创建有效掩码（深度非零区域）
#     if pred_depth.dim() == 3:
#         pred_depth = pred_depth.unsqueeze(1)  # 添加通道维度 [B,1,H,W]
#     valid_mask = (pred_depth > eps).float()  # [B,1,H,W]
    
#     # 水平方向掩码（相邻像素均有效）
#     mask_x = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]  # [B,1,H,W-1]
#     # 垂直方向掩码（相邻像素均有效）
#     mask_y = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]  # [B,1,H-1,W]

#     # 计算绝对值差异
#     diff_x = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
#     diff_y = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])

#     # 应用掩码过滤无效区域
#     valid_diff_x = diff_x * mask_x  # 无效区域的差异将被置零
#     valid_diff_y = diff_y * mask_y

#     # 仅在有掩码的区域计算惩罚
#     loss_x = torch.sum(torch.relu(margin - valid_diff_x)) / (torch.sum(mask_x) + eps)
#     loss_y = torch.sum(torch.relu(margin - valid_diff_y)) / (torch.sum(mask_y) + eps)

#     return (loss_x + loss_y) * 0.5






def get_loss(data_dict, mask):
    # objectness_loss, data_dict = compute_objectness_loss(data_dict)
    pred = data_dict['pred']  # torch.Size([2, 200, 720, 1280])
    gt = data_dict['real_depth']
    #loss1 = criterion(pred, gt) 
    #loss1 = loss1[mask]
    #loss1 = loss1.mean()
    loss = gaussian_weighted_loss(pred, gt, mask, num_classes=600, sigma=2)
    #loss = loss + loss1
    loss = loss
    mask_bool = mask.bool()
    pred_mask_pred = torch.argmax(pred, 1)
    data_dict['pred_acc'] = (torch.abs(gt[mask_bool] - pred_mask_pred[mask_bool].long()) < 5).float().mean()
    return loss, data_dict

# def compute_objectness_loss(end_points):
#     criterion = nn.CrossEntropyLoss(reduction='mean')
#     objectness_score = end_points['objectness_score']
#     objectness_label = end_points['objectness_label']
#     # objectness_label = objectness_label.unsqueeze(1)
#     loss = criterion(objectness_score, objectness_label)

#     end_points['loss/stage1_objectness_loss'] = loss

#     objectness_pred = torch.argmax(objectness_score, 1)
#     end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
#     end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
#         objectness_pred == 1].float().mean()
#     end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
#         objectness_label == 1].float().mean()
#     return loss, end_points


def gaussian_weighted_loss(pred, target, mask, num_classes, sigma):
    """
    高斯加权分类损失
    :param pred: 模型预测 (B, num_classes, 720, 1280)，每个位置的类别预测分布
    :param target: 真实类别 (B, 720, 1280)，每个位置的真实类别索引
    :param num_classes: 总类别数
    :param sigma: 高斯核的标准差，控制权重分布
    :return: 损失值 (标量)
    """
    # 获取预测的形状
    B, num_classes, H, W = pred.shape

    # 展开 target 维度，方便与类别索引计算权重
    target_expanded = target.unsqueeze(1).float()  # (B, 1, 720, 1280)

    # 类别索引 (0 到 num_classes-1)
    class_indices = torch.arange(num_classes, device=pred.device).view(1, num_classes, 1, 1).expand(1, num_classes, 720, 1280).float()
    # 计算高斯权重
    weights = torch.exp(-0.5 * ((class_indices - target_expanded) / sigma) ** 2)  # (B, num_classes, 720, 1280)
    # 计算log softmax（对数概率分布）
    log_probs = F.log_softmax(pred, dim=1)  # (B, num_classes, 720, 1280)
    # 计算加权交叉熵损失
    # loss = -(weights * log_probs).sum(dim=1).mean()  # 按权重求和后取平均
    loss = -(weights * log_probs).sum(dim=1)
    loss = loss[mask]
    loss = loss.mean()
    return loss




