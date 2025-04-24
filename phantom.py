import torch
from base_attack import BaseAttack
from transformers import DetrConfig, AutoImageProcessor, DetrForObjectDetection
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from PIL import Image
import requests
import torch.nn.functional as F
import pdb
import torch.nn as nn
import torchvision
import numpy as np
import time
from model_zoo import load_from_pretrained
from utils import set_all_seeds
set_all_seeds(0)

lambda_1 = 1
lambda_2 = 10
epsilon = 60


class Phantom(BaseAttack):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_train_loss = 0.0
        self.current_max_objects_loss = 0.0
        self.current_orig_classification_loss = 0.0
        self.min_bboxes_added_preds_loss = 0.0

    
    def set_iou_param(self, nms_conf_thres, nms_iou_thres):
        self.nms_conf_thres = nms_conf_thres
        self.nms_iou_thres = nms_iou_thres
        
    # def update_bx(self):
    #     pass

        
    def run_attack(self, image, img_id):
        
        self.set_iou_param(0.25, 0.45)
        self.init_input(image, img_id)
        self.generate_bx()
        adv_patch = self.bx
        for it in range(self.it_num):
            
            adv_patch, applied_patch = self.fastGradientSignMethod(adv_patch, self.img_tensor)
            perturbation = adv_patch - self.bx
            if it == 0:
                perturbation += torch.normal(mean=0.0, std=0.1, size=perturbation.size()).to(self.device)
            norm = torch.sum(torch.square(perturbation))
            norm = torch.sqrt(norm)
            factor = min(1, epsilon / norm.item()) 
            adv_patch = (torch.clip(self.bx + perturbation * factor, 0.0, 1.0))

            self.inference(applied_patch)
            self.logger(it)
            self.update_buffer(applied_patch, it)
            
        self.save_with_thread_pool(max_workers=256)
        self.write_log()
        
        self.clean_flag = True
        torch.cuda.empty_cache()
        
        
    def fastGradientSignMethod(self, adv_patch, images, epsilon=0.3):
        # adv_patch -> self.bx
        applied_patch = torch.clamp(images[:] + adv_patch, 0, 1)
        data_grad = self.loss_function_gradient(applied_patch, images, adv_patch) 
        sign_data_grad = data_grad.sign()
        perturbed_patch = adv_patch - epsilon * sign_data_grad
        perturbed_patch_c = torch.clamp(perturbed_patch, -0.04, 0.04).detach()
        return perturbed_patch_c, applied_patch
    
    def loss_function_gradient(self, applied_patch, images, adv_patch):
        """
            applied_patch -> bx + img
            init_images -> img
            adcv_patch -> bx
        """
        init_images = self.img_tensor.clone()
        iou = IoU(conf_threshold=self.nms_conf_thres, 
                  iou_threshold=self.nms_iou_thres, 
                  img_size=self.target_size[0], 
                  device=self.device)        
        self.inference(applied_patch)
        max_objects_loss = self.max_objects()

        bboxes_area_loss = bboxes_area(self.boxes, self.target_size)
        
        # combined:
        #   shape: [num_of_dets, 85]
        
        #          [x_min, y_min, x_max, y_max, 
        #           score, 
        #           class_0_conf,
        #           class_1_conf,
        #           ...,
        #           class_79_conf]
        
        iou_loss = iou(self.cl_combined.unsqueeze(0).detach(), 
                       self.combined.unsqueeze(0))
        
        loss = max_objects_loss * lambda_1
        
        if not torch.isnan(iou_loss):
            loss += (iou_loss * (1 - lambda_1))
            self.current_orig_classification_loss += ((1 - lambda_1) * iou_loss.item())

        if not torch.isnan(bboxes_area_loss):
            loss += (bboxes_area_loss * lambda_2)
            
        loss = loss.sum().to(self.device)

        self.model.zero_grad()
        data_grad = torch.autograd.grad(loss, adv_patch)[0]
        return data_grad
    
    def max_objects(self, target_class = None):

        # RT-DeTR does not have objectness score, so use probability instead
        prob = self.prob[:len(self.scores)].contiguous().unsqueeze(0)
        x2 = prob.clone()
        conf, j = x2.max(2, keepdim=False)
        all_target_conf = x2[:, :, target_class]
        conf_thres = 0.25
        under_thr_target_conf = all_target_conf[conf < conf_thres]
        zeros = torch.zeros(under_thr_target_conf.size()).to(self.device)
        zeros.requires_grad = True
        x3 = torch.maximum(-under_thr_target_conf + conf_thres, zeros)
        mean_conf = torch.sum(x3, dim=0) 
        return mean_conf
        

class IoU(nn.Module):
    def __init__(self, conf_threshold, iou_threshold, img_size, device) -> None:
        super(IoU, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

    def forward(self, output_clean, output_patch):
        batch_loss = []

        gn = torch.tensor(self.img_size)[[1, 0, 1, 0]]
        gn = gn.to(self.device)
        pred_clean_bboxes = non_max_suppression(output_clean, self.conf_threshold, self.iou_threshold, classes=None,
                                                max_det=1000)
        patch_conf = 0.001
        pred_patch_bboxes = non_max_suppression(output_patch, patch_conf, self.iou_threshold, classes=None,
                                                max_det=30000)

        # print final amount of predictions
        final_preds_batch = 0
        for img_preds in non_max_suppression(output_patch, self.conf_threshold, self.iou_threshold, classes=None,
                                             max_det=30000):
            final_preds_batch += len(img_preds)

        for (img_clean_preds, img_patch_preds) in zip(pred_clean_bboxes, pred_patch_bboxes):  # per image

            for clean_det in img_clean_preds:
                clean_clss = clean_det[5]

                clean_xyxy = torch.stack([clean_det])  # .clone()
                clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(
                    self.device)

                img_patch_preds_out = img_patch_preds[img_patch_preds[:, 5].view(-1) == clean_clss]

                patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(self.device)

                if len(clean_xyxy_out) != 0:
                    target = self.get_iou(patch_xyxy_out, clean_xyxy_out)
                    if len(target) != 0:
                        target_m, _ = target.max(dim=0)
                    else:
                        target_m = torch.zeros(1).to(self.device)

                    batch_loss.append(target_m)

        one = torch.tensor(1.0).to(self.device)
        if len(batch_loss) == 0:
            return one

        return (one - torch.stack(batch_loss).mean())

    def get_iou(self, bbox1, bbox2):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
            bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

        inter = self.intersect(bbox1, bbox2)
        area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                  (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                  (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x_out = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x_out.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x_out = torch.cat((x_out, v), 0)

        # If none remain process next image
        if not x_out.shape[0]:
            continue

        # Compute conf
        x_out[:, 5:] = x[xc[xi]][:, 5:] * x[xc[xi]][:, 4:5]#x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x_out[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x_out[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x_out = torch.cat((box[i], x_out[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x_out[:, 5:].max(1, keepdim=True)
            x_out = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x_out = x_out[(x_out[:, 5:6] == torch.tensor(classes, device=x_out.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x_out.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x_out = x_out[x_out[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x_out[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x_out[:, :4] + c, x_out[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x_out[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def bboxes_area(bboxes, image_size, conf_thres=0.25):

    t_loss = 0.0
    not_nan_count = 0

    for b in bboxes:  # image index, image inference
        x_min, y_min, x_max, y_max = b
        w = torch.clamp(x_max - x_min, min=2, max=4096)
        h = torch.clamp(y_max - y_min, min=2, max=4096)
        
        img_loss = (w * h).mean() / (image_size[0][0] * image_size[0][0])
        if not torch.isnan(img_loss):
            t_loss += img_loss
            not_nan_count += 1

    if not_nan_count == 0:
        t_loss_f = torch.tensor(torch.nan)
    else:
        t_loss_f = t_loss / not_nan_count

    return t_loss_f

        
def single_test(num_q = 1000, 
                img_id = 0, 
                url = None,
                device = None):
    print("======================== SINGLE TEST START ========================")
    if not url:
        url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
        
    image = Image.open(requests.get(url, stream=True).raw)

    thres = 0.25

    for i in range(3):
        model, image_processor = load_from_pretrained(i, device)
        model.config.num_queries = 1000

        phantom = Phantom(
            model = model,
            image_processor = image_processor,
            it_num = 100,
            conf_thres = thres,
            target_idx = None,
            output_dir = f"./output_rt_detr/phantom_test_{i}",
            device = device
        )
        for j in range(1):
        # phantom.set_iou_param(0.25, 0.45)
        # phantom.init_input(image, img_id)
        # phantom.generate_bx()
            phantom.run_attack(image, img_id)
    print("======================== SINGLE TEST  END  ========================")
        
def multi_test(num_q = 1000,
               val_size = 100,
               device = None
               ):
    print("======================== MULTI TEST START ========================")

    from datasets import load_dataset
    import random
    import utils
    from tqdm import tqdm
    
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random_indices = random.sample(range(len(coco_data)), val_size)
    coco_data = coco_data.select(random_indices)
    
    for i in range(3):
        i=2
        model, image_processor = load_from_pretrained(i, device)
        model.config.num_queries = 1000
        phantom = Phantom(
            model = model,
            image_processor = image_processor,
            it_num = 10,
            conf_thres = 0.25,
            target_idx = None,
            output_dir = f"./results/phantom_test_{i}",
            device = device
        )
    
        for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc=f"running: multi test"):
            image_id, image, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
            phantom.run_attack(image, image_id)
            # import pdb; pdb.set_trace()
    print("======================== MULTI TEST  END  ========================")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # single_test(device = device)
    
    multi_test(device)