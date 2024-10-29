import torch.nn as nn
import torch
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_loss_novel(end_points,result_obj,result_graspness_score):

    final_weigh = torch.empty(0,1024, device=device)
    
    for i in range(2):
        index_points_dict = {}
        score_n=[]
        scaled_weights_tensor = torch.zeros_like(result_graspness_score[i], device=device)

        for ii, index  in enumerate(result_obj[i]):
            index = index.item()
            if index not in index_points_dict:
                index_points_dict[index] = []
            index_points_dict[index].append(ii)

        for key, value in index_points_dict.items():
                      
            G_score = result_graspness_score[i][value]
        
            G_score_mean = torch.mean(G_score)
           
            #G_score_median = torch.median(G_score)
            score_n.append(G_score_mean)

        num = 0
        score_n_tensor = torch.tensor(score_n, device=device)
        for key, value in index_points_dict.items():
            
            max_score_n = torch.max(score_n_tensor)
            min_score_n = torch.min(score_n_tensor)
            mean_score_n= torch.mean(score_n_tensor)
            # 权重缩放
            scaled_weights =  1 - math.log(score_n_tensor[num]/max_score_n)
            scaled_weights_tensor[value] = scaled_weights
            
            num = num + 1
        scaled_weights_tensor = scaled_weights_tensor.unsqueeze(0) 

        final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)   
        
        



    objectness_loss, end_points = compute_objectness_loss(end_points) # 抓取点
    graspness_loss, end_points = compute_graspness_loss(end_points)   # 抓取graspness得分
    
    view_loss, end_points = compute_view_graspness_loss(end_points,final_weigh)   # 抓取方向
    score_loss, end_points = compute_score_loss(end_points,final_weigh)           # 抓取最终得分
    width_loss, end_points = compute_width_loss(end_points,final_weigh)           # 抓取宽度
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss

    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):  # 是不是物体
    # 首先创建一个交叉熵损失函数criterion，通过nn.CrossEntropyLoss(reduction='mean')实例化
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 从end_points中获取目标性得分（objectness_score）和目标性标签（objectness_label）
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']

    # objectness_score 预测值
    # objectness_label 真实值
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss # 并将其赋值给变量loss

    # 使用torch.argmax(objectness_score, 1)获取目标性预测值（经过softmax后的类别概率最大值对应的类别索引）
    objectness_pred = torch.argmax(objectness_score, 1)
    # 计算正确率，精确度，召回率
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspness_loss(end_points): # 抓取读损失
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()

    
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points,final_weigh): # 抓取读视角损失
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)

    criterion = nn.SmoothL1Loss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).expand_as(loss))
    weighted_mean_loss = torch.mean(weighted_loss)
    end_points['loss/stage2_view_loss'] = weighted_mean_loss
    return weighted_mean_loss, end_points

 
def compute_score_loss(end_points,final_weigh): # 最终得分
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).unsqueeze(-1).expand_as(loss))
    weighted_mean_loss = torch.mean(weighted_loss)
    end_points['loss/stage3_score_loss'] = weighted_mean_loss
    return weighted_mean_loss, end_points


def compute_width_loss(end_points,final_weigh): # 爪子宽度 
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).unsqueeze(-1).expand_as(loss))

    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    weighted_loss = weighted_loss[loss_mask].mean()
    
    end_points['loss/stage3_width_loss'] = weighted_loss
    return weighted_loss, end_points
