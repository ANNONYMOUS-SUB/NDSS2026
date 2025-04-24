from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
from collections import defaultdict
from model_zoo import load_from_pretrained
from tqdm import tqdm
import random
from datasets import load_dataset
import utils
import json

def calculate_metrics(results):
    """
    Calculate COCO-style metrics from detection results.
    Works with HuggingFace dataset format ground truth.
    
    Args:
        results: List of dictionaries containing predictions and ground truth
                Each dict has keys: pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    
    Returns:
        Dictionary containing mAP and AR metrics
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    
    ### test code
    
    print(f"Number of images: {len(results)}")
    total_preds = sum(len(r['pred_boxes']) for r in results)
    total_gts = sum(len(r['gt_boxes']) for r in results)
    print(f"Total predictions: {total_preds}")
    print(f"Total ground truths: {total_gts}")
    
    # 检查是否有空的预测或GT
    empty_preds = sum(1 for r in results if len(r['pred_boxes']) == 0)
    empty_gts = sum(1 for r in results if len(r['gt_boxes']) == 0)
    print(f"Images with no predictions: {empty_preds}")
    print(f"Images with no ground truths: {empty_gts}")
    
    
    
    # Initialize COCO gt structure
    coco_gt = COCO()
    coco_gt.dataset = {
        'images': [],
        'annotations': [],
        'categories': [{'id': i} for i in range(80)]  # COCO classes (0-79)
    }
    
    # Initialize detection list
    coco_dt = []
    
    # Convert format
    ann_id = 1
    for img_id, result in enumerate(results, 1):
        # Add image info
        img_info = {
            'id': img_id,
            'width': result['image_size'][0],  # Normalized coordinates
            'height': result['image_size'][1]
        }
        coco_gt.dataset['images'].append(img_info)
        
        # Add ground truth
        gt_boxes = result['gt_boxes']
        gt_labels = result['gt_labels']
        
        for box, cat_id in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(cat_id),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0
            }
            coco_gt.dataset['annotations'].append(ann)
            ann_id += 1
        
        # Add predictions
        pred_boxes = result['pred_boxes']
        pred_scores = result['pred_scores']
        pred_labels = result['pred_labels']
        
        for box, score, cat_id in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            det = {
                'image_id': img_id,
                'category_id': int(cat_id),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            }
            coco_dt.append(det)
    
    # Create index
    coco_gt.createIndex()
    
    # Create COCO detection object
    coco_dt = coco_gt.loadRes(coco_dt)
    
    # Create evaluator
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.params.maxDets = [1, 10, 100]    # Run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP': cocoEval.stats[0],  # AP @[ IoU=0.50:0.95 ]
        'mAP_50': cocoEval.stats[1],  # AP @[ IoU=0.50 ]
        'mAP_75': cocoEval.stats[2],  # AP @[ IoU=0.75 ]
        'AR_max_100': cocoEval.stats[8]  # AR @[ IoU=0.50:0.95 | maxDets=100 ]
    }
    
    return metrics

if __name__ == "__main__":
    results = defaultdict(dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_q = 1000
    val_size = 1000
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    coco_data = load_dataset("detection-datasets/coco", split="val")

    random_indices = random.sample(range(len(coco_data)), val_size)
    coco_data = coco_data.select(random_indices)
    for i in range(3):
        print(f"Model index: {i}")
        model_results = []
        
        for c in range(1, 20):
            conf = c * 0.05
            print(f"\nEvaluating confidence threshold: {conf:.2f}")
            
            # Load model
            model, image_processor = load_from_pretrained(i, device)
            model.config.num_queries = num_q
            
            # Store predictions for current confidence threshold
            curr_predictions = []
            
            # Process each image and update evaluator
            for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc="Processing COCO data"):
                image_id, image, width, height, bbox_id, category, gt_boxes, _ = utils.parse_example(example)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                # Get model predictions
                inputs = image_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Post-process predictions
                target_size = torch.tensor([image.size[::-1]])
                predictions = image_processor.post_process_object_detection(
                    outputs, 
                    threshold=conf,
                    target_sizes=target_size
                )[0]
                
                # Move predictions to CPU
                predictions = utils.move_to_cpu(predictions)
                
                # Format predictions for evaluation
                pred_boxes = predictions["boxes"].cpu().numpy()
                pred_scores = predictions["scores"].cpu().numpy()
                pred_labels = predictions["labels"].cpu().numpy()
                
                gt_boxes = example["objects"]["bbox"]  # Already in x1,y1,x2,y2 format
                gt_labels = example["objects"]["category"]
                
                model_results.append({
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'pred_labels': pred_labels,
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'image_size' : [width, height]
                })
                
            # Calculate metrics
            metrics = calculate_metrics(model_results)
            results[i][conf] = metrics
            
            print(f"Results for confidence threshold {conf:.2f}:")
            print(f"mAP@[.50:.95]: {metrics['mAP']:.3f}")
            print(f"mAP@.50: {metrics['mAP_50']:.3f}")
            print(f"mAP@.75: {metrics['mAP_75']:.3f}")
            print(f"AR@100: {metrics['AR_max_100']:.3f}")
            
            model_results = []
    
    # Save final results
    with open("evaluate.json", "w") as f:
        json.dump(results, f, indent=4)