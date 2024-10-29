import torch.nn as nn
import torch


""" 
    ### 回归
def get_loss(data_dict):
    loss, data_dict = compute_custom_masked_mse_loss(data_dict)
    data_dict['loss/overall_loss'] = loss
    return loss, data_dict


def _l2(pred, gt):
        return (pred - gt) ** 2
def safe_mean(data, mask, default_res = 0.0):
    masked_data = data[mask]
    return torch.tensor(default_res).to(masked_data.device) if masked_data.numel() == 0 else masked_data.mean()

def compute_custom_masked_mse_loss(data_dict):
       
        pred = data_dict['pred']
        gt = data_dict['real_depth']
        mask = data_dict['loss_mask']
        # loss = safe_mean(_l2(pred, gt), mask)
        loss = _l2(pred, gt)
        loss = loss * mask
        data_dict['stage1_objectness_acc'] = (pred == gt.long()).float().mean()
        return loss, data_dict """


### 分类
criterion = nn.CrossEntropyLoss(reduction='none')
def get_loss(data_dict, mask):

    objectness_loss, data_dict = compute_objectness_loss(data_dict)
    pred = data_dict['pred']  # torch.Size([2, 200, 720, 1280])
    # gt = data_dict['depth']
    gt = data_dict['real_depth']

    loss = criterion(pred, gt)  # torch.Size([2, 1000, 720, 1280])   torch.Size([2, 720, 1280]) -> torch.Size([2, 720, 1280]) 
    mask_bool = mask.bool()

    gt_mask = gt * mask
    pred_mask = pred * mask

    pred_mask_pred = torch.argmax(pred_mask, 1)
    data_dict['pred_acc'] = (torch.abs(gt_mask[mask_bool] - pred_mask_pred[mask_bool].long()) < 5).float().mean()

    loss = loss[mask_bool] # 要转为bool
    loss_mean = torch.mean(loss)
    # 总loss为深度loss + 物体loss
    loss = loss_mean + objectness_loss

    print("重建准确率是", data_dict['pred_acc'])
    # loss = loss_mean
    return loss, data_dict


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    # objectness_label = objectness_label.unsqueeze(1)
    loss = criterion(objectness_score, objectness_label)

    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points