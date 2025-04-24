import torch
import os
import random
import numpy as np

def set_all_seeds(seed=42):
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def move_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_cpu(item) for item in data]
    else:
        return data
    
def parse_example(example):
    image_id = example["image_id"]
    image = example["image"]
    width = example["width"]
    height = example["height"]
    bbox_id = example["objects"]["bbox_id"]
    category = example["objects"]["category"]
    bbox = example["objects"]["bbox"]
    area = example["objects"]["area"]
    
    return image_id, image, width, height, bbox_id, category, bbox, area


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two bounding boxes.
    box1, box2: (x_min, y_min, x_max, y_max)
    """
    box1 = torch.tensor(box1, dtype=torch.float32)
    box2 = torch.tensor(box2, dtype=torch.float32)

    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    inter_width = max(0, x_max - x_min)
    inter_height = max(0, y_max - y_min)
    intersection = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou

def xyxy2xywh(boxes):
    if boxes.dim() == 1:  
        x_min, y_min, x_max, y_max = boxes
    elif boxes.dim() == 2:  
        x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    else:
        raise ValueError(f"Invalid input shape {boxes.shape}, expected [4] or [N, 4].")

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    converted_boxes = torch.stack((x_center, y_center, width, height), dim=-1)

    return converted_boxes

def denormalize(tensor):
    """
    Denormalizes a tensor using the provided mean and std.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor