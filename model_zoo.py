import torch
import requests
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

import numpy as np
import random
import pdb
"""
    MODEL ZOO:
    
    Requirements
        1.	The model must be an object detection model.
        2.	The model must be transformer-based.
        3.	It must allow setting $num_queries$ in the model configuration without requiring retraining.
        4.	The model must be publicly accessible and reproducible.

    Due to these considerations, all the models we used are from Hugging Face and belong to the RT-DETR family.
    
    Available Models
        1.  RT_DETR_r50vd [huggingface](https://huggingface.co/PekingU/rtdetr_r50vd) [paper](https://arxiv.org/abs/2304.08069)
        2.  RT_DETR_r50vd_coco_o365 [huggingface](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) [paper](https://arxiv.org/abs/2304.08069)
        3.  RT_DETR_v2_r50vd [huggingface](https://huggingface.co/PekingU/rtdetr_v2_r50vd) [paper](https://arxiv.org/html/2407.17140v1)
        
    
"""

from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Define a model zoo dictionary
MODEL_ZOO = {
    0: "PekingU/rtdetr_r50vd",
    1: "PekingU/rtdetr_r50vd_coco_o365",
    2: "PekingU/rtdetr_v2_r50vd"
}

def get_model_name(ckpt):
    """Retrieve model name from the zoo given an integer ID or directly validate a string checkpoint."""
    if isinstance(ckpt, int):
        if ckpt not in MODEL_ZOO:
            raise ValueError(f"Invalid checkpoint ID: {ckpt}. Available IDs: {list(MODEL_ZOO.keys())}")
        return MODEL_ZOO[ckpt], ckpt
    elif isinstance(ckpt, str) and ckpt in MODEL_ZOO.values():
        return ckpt, list(MODEL_ZOO.keys())[list(MODEL_ZOO.values()).index(ckpt)]
    else:
        raise ValueError(
            f"Invalid checkpoint: {ckpt}. \n"
            f"Available options:\n"
            f"IDs: {list(MODEL_ZOO.keys())}\n"
            f"Names: {list(MODEL_ZOO.values())}\n"
            "Please choose a valid ID or checkpoint name."
        )

def load_from_pretrained(ckpt=None, num_q=1000, device=None):
    print(f"loading model on device: {device}")
    """Load a model and image processor from the model zoo."""
    ckpt, ckpt_id = get_model_name(ckpt)

    print(f"Initializing checkpoint: {ckpt}, ID: {ckpt_id}")

    # Load the correct model based on the checkpoint ID
    if ckpt_id in {0, 1}:
        model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    elif ckpt_id == 2:
        model = RTDetrV2ForObjectDetection.from_pretrained(ckpt).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint ID: {ckpt_id}")

    image_processor = RTDetrImageProcessor.from_pretrained(ckpt)
    model = model.eval()
    model.config.num_queries = num_q
    return model, image_processor
        
if __name__ == "__main__":
    # url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # thres = 0.25
    # num_q = 1000
    # for i in range(3):
    #     model, image_processor = load_from_pretrained(i, device=device)
    #     model.config.num_queries = num_q
        
    #     input = image_processor(images=image, return_tensors="pt").to(device)
    #     img_tensor = input["pixel_values"]
    #     target_size = [img_tensor.shape[2:] for _ in range(1)]
    #     result = model(img_tensor)
    #     logits = result.logits[0]
        
    #     output = image_processor.post_process_object_detection(result, 
    #                                                         threshold = thres, 
    #                                                         target_sizes = target_size)[0]
    #     scores = output["scores"]
    #     print(f"== * ==")
    #     # print(f"model ID {i}, \nshape: {logits.shape}, num_class{model.config}")
    #     # print(f"model ID {i}, \nshape: {scores.shape}")
    #     print(model.class_embed if hasattr(model, 'class_embed') else "cannot find class_embed")
    #     print(f"== * ==")
    def ultra_fast_save(tensor, filepath):
        tensor_cpu = tensor.detach().cpu().contiguous()
        shape = tensor_cpu.shape

        shape_dims = np.array([len(shape)], dtype=np.int8).tobytes()
        shape_header = np.array(shape, dtype=np.int64).tobytes()

        dtype_str = np.dtype(tensor_cpu.numpy().dtype).str.encode('ascii')
        dtype_length = np.array([len(dtype_str)], dtype=np.int8).tobytes()

        data_bytes = tensor_cpu.numpy().tobytes()

        with open(filepath, 'wb') as f:
            fd = f.fileno()
            try:
                os.write(fd, shape_dims)
                os.write(fd, shape_header)
                os.write(fd, dtype_length)
                os.write(fd, dtype_str)
                os.write(fd, data_bytes)
                os.fsync(fd)
            except Exception as e:
                print(f"Failed to save tensor to {filepath}: {e}")
                os.remove(filepath)
                
    import utils
    from utils import set_all_seeds
    from datasets import load_dataset
    import torch
    from tqdm import tqdm
    import os
    
    set_all_seeds(0)
    save_dir = "./saved/clean"
    os.makedirs(save_dir, exist_ok=True)
    
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random_indices = random.sample(range(len(coco_data)), 1000)
    coco_data = coco_data.select(random_indices)
    batch_size = len(coco_data) 
    
    ckpt = 0
    num_q = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, image_processor = load_from_pretrained(ckpt, num_q, device=device)
    
    for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc=f"running"):
        image_id, img, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
        img_tensor = utils.denormalize(img_tensor)
        target_size = [img_tensor.shape[2:] for _ in range(1)]
        
        path = os.path.join(save_dir, f"img_id_{str(image_id)}.pt")
        ultra_fast_save(img_tensor, path)
        
        
        
