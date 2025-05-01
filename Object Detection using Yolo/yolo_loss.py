import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        xy = boxes[:, :2] / self.S 
        wh = boxes[:, 2:] / 2      
        
        xy_1 = xy - wh
        xy_2 = xy + wh
        return torch.cat([xy_1, xy_2], dim=1).clamp(0,1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        box_target_xyxy = self.xywh2xyxy(box_target)
        
        best_ious = torch.zeros(box_target.size(0), device=box_target.device)
        best_boxes = torch.zeros_like(pred_box_list[0])
        
        for i in range(self.B):
            pred_box_xyxy = self.xywh2xyxy(pred_box_list[i][:, :4])
            ious = compute_iou(pred_box_xyxy, box_target_xyxy).diag()
            mask = ious > best_ious
            best_ious[mask] = ious[mask]
            best_boxes[mask] = pred_box_list[i][mask]
        
        return best_ious.unsqueeze(1), best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        loss = F.mse_loss(classes_pred[has_object_map], classes_target[has_object_map], reduction='sum')
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        loss = 0
        no_obj_mask = ~has_object_map
        
        for i in range(self.B):
            conf_scores = pred_boxes_list[i][:, :, :, 4] 
            no_obj_conf = conf_scores[no_obj_mask]
            target = torch.zeros_like(no_obj_conf)
            loss += F.mse_loss(no_obj_conf, target, reduction='sum')
        
        return self.l_noobj * loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        loss = F.mse_loss(box_pred_conf, box_target_conf.detach(), reduction='sum')
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        wh_loss = F.mse_loss(torch.sqrt(torch.abs(box_pred_response[:, 2:])), torch.sqrt(torch.abs(box_target_response[:, 2:])), reduction='sum')
        
        return self.l_coord * (xy_loss + wh_loss)

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        total_loss = 0.0
        pred_boxes = pred_tensor[:, :, :, :self.B*5].reshape(N, self.S, self.S, self.B, 5)
        pred_cls   = pred_tensor[:, :, :, self.B*5:]

        pred_boxes_list = [pred_boxes[:, :, :, i, :].contiguous() for i in range(self.B)]
        
        # compcute classification loss 
        cls_loss    = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)
        total_loss += cls_loss
        
        # compute no-object loss 
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)
        total_loss += no_obj_loss
        
        selector = has_object_map.reshape(-1)
        box_target_response = target_boxes.reshape(-1, 4)[selector]

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation 
        pred_boxes_response = [pred_boxes_list[i].reshape(-1, 5)[selector] for i in range(self.B)]

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_response, box_target_response)
        
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects 
        reg_loss = self.get_regression_loss(best_boxes[:, :4], box_target_response)
        total_loss += reg_loss
        
        # compute contain_object_loss 
        conf_loss = self.get_contain_conf_loss(best_boxes[:, 4:], best_ious)
        total_loss += conf_loss

        # compute final loss
        total_loss *= 1/N
        
        # construct return loss_dict 
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=conf_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict