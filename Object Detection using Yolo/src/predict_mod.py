import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from src.config import VOC_CLASSES, VOC_IMG_MEAN, YOLO_IMG_DIM


def decoder(pred, min_confidence_threshold=0.2, nms_threshold=0.5):
    """
    Decode YOLO predictions.

    Args:
        pred (tensor): shape [1, S, S, B*5 + C]
        min_confidence_threshold (float): Minimum objectness * class prob to keep box
        nms_threshold (float): IoU threshold for non-maximum suppression

    Returns:
        boxes, cls_indexs, probs
    """
    grid_num = pred.squeeze().shape[0]  # 14 for ResNet, 7 for VGG
    assert pred.squeeze().shape[0] == pred.squeeze().shape[1]
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1.0 / grid_num
    pred = pred.squeeze(0)  # [S, S, B*5 + C]

    obj_conf1 = pred[:, :, 4].unsqueeze(2)
    obj_conf2 = pred[:, :, 9].unsqueeze(2)
    obj_confidences = torch.cat((obj_conf1, obj_conf2), 2)

    mask1 = obj_confidences > min_confidence_threshold
    mask2 = obj_confidences == obj_confidences.max()
    mask = (mask1 + mask2).gt(0)

    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    final_score = (contain_prob * max_prob)[0]
                    if float(final_score) > min_confidence_threshold:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(final_score)

    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)
        probs = torch.stack(probs, dim=0)
        cls_indexs = torch.stack(cls_indexs, dim=0)

    keep = nms(boxes, probs, threshold=nms_threshold)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.4):
    """
    Non-Maximum Suppression.

    Args:
        bboxes (tensor): [N, 4]
        scores (tensor): [N]
        threshold (float): IoU threshold

    Returns:
        keep (list of indices)
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item() if order.numel() > 1 else order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.LongTensor(keep)

@torch.inference_mode()
def predict_image(model, image_input, root_img_directory="", min_conf=0.25):
    """
    Predict objects in a single image or image frame.

    Args:
        model: Trained YOLO model
        image_input: Either image filename (string) or RGB image as NumPy array
        root_img_directory: Root directory if using image filename
        min_conf: Minimum confidence to keep a detection

    Returns:
        list of [(x1, y1), (x2, y2), class_name, source_id, prob]
    """
    result = []

    if isinstance(image_input, str):
        image_path = os.path.join(root_img_directory, image_input)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        source_id = image_input
    else:
        image = image_input 
        source_id = "frame"

    h, w, _ = image.shape

    img = cv2.resize(image, (YOLO_IMG_DIM, YOLO_IMG_DIM))
    img = img - np.array(VOC_IMG_MEAN, dtype=np.float32)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to("mps")

    pred = model(img_tensor).cpu()
    boxes, cls_indexs, probs = decoder(pred, min_confidence_threshold=min_conf)

    for i, box in enumerate(boxes):
        prob = float(probs[i])
        if prob < min_conf:
            continue

        x1 = int(max(0, min(box[0], 1.0)) * w)
        x2 = int(max(0, min(box[2], 1.0)) * w)
        y1 = int(max(0, min(box[1], 1.0)) * h)
        y2 = int(max(0, min(box[3], 1.0)) * h)

        cls_index = int(cls_indexs[i])
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], source_id, prob])

    return result