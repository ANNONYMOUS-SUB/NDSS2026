import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import numpy as np
import os
import glob
sys.path.append("../")
sys.path.append("../../")
from model_zoo import load_from_pretrained


def load_jpg_as_tensor(filepath, device=None):
    img = Image.open(filepath)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    _, processor = load_from_pretrained(ckpt=0, device=device)
    img_tensor = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    img_tensor = denormalize(img_tensor)
    target_size = [img_tensor.shape[2:] for _ in range(1)]
    return img_tensor, target_size


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
            

if __name__ == "__main__":
    input_dir = "./test_src"
    output_dir = "./test_src"
    os.makedirs(output_dir, exist_ok=True)
    
    paths = sorted(glob.glob(f"{input_dir}/*.jpg"))
    print("found {} image files".format(len(paths)))
    for i, p in enumerate(paths):
        print(f"Processing {i+1}/{len(paths)}: {p}")
        img_tensor, _ = load_jpg_as_tensor(p)
        
        # Save the tensor
        output_path = os.path.join(output_dir, os.path.basename(p).replace('.jpg', '.pt'))
        ultra_fast_save(img_tensor, output_path)
        
        print(f"Saved tensor to {output_path}")