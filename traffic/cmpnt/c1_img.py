import sys
sys.path.append("../")
from tqdm import tqdm
from multiprocessing import Process, Queue, Event
import torch
import glob
import time
from legacy_flops import FLOPs_DECORATOR, write_profile, write_profile_lt
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import multiprocessing as mp
import logging
import random
import gc
from scipy import ndimage

import pynvml
import json


random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def input_smoothing(x, width=3, height=-1, kernel_size=None):
    """
    Simple input smoothing defense using median filtering
    
    Parameters:
        x: NumPy array input, with pixel values between 0-1
        width: Width of the sliding window (number of pixels)
        height: Height of the window. Same as width by default
        
    Returns:
        Smoothed array
    """
    # Ensure x is float32
    x = x.astype(np.float32)
    
    # Set default height if not specified
    if height == -1:
        height = width
    
    # Determine dimensions
    ndim = x.ndim
    
    if ndim == 2:  # Single channel 2D image
        # For 2D array, shape is (H, W)
        smoothed = ndimage.filters.median_filter(x, size=(width, height), mode='reflect')
    elif ndim == 3:  # Multi-channel 2D image or single-channel 3D image
        if x.shape[2] <= 4:  # Assuming RGB or RGBA image with shape (H, W, C)
            smoothed = np.zeros_like(x)
            for c in range(x.shape[2]):
                smoothed[:, :, c] = ndimage.filters.median_filter(x[:, :, c], size=(width, height), mode='reflect')
        else:  # Possibly 3D data
            smoothed = ndimage.filters.median_filter(x, size=(width, height, width), mode='reflect')
    elif ndim == 4:  # Batch of images with shape (B, H, W, C)
        smoothed = np.zeros_like(x)
        for b in range(x.shape[0]):
            for c in range(x.shape[3]):
                # Using the exact format from median_filter_py
                temp_input = x[b:b+1, :, :, c:c+1]  # Shape becomes (1, H, W, 1)
                temp_output = ndimage.filters.median_filter(temp_input, size=(1, width, height, 1), mode='reflect')
                smoothed[b, :, :, c] = temp_output[0, :, :, 0]
    else:
        # For other dimensions, use a general approach
        filter_size = tuple([width if i < ndim else 1 for i in range(ndim)])
        smoothed = ndimage.filters.median_filter(x, size=filter_size, mode='reflect')
    
    # Ensure values remain in 0-1 range
    smoothed = np.clip(smoothed, 0, 1)
    return smoothed
        

class imgStream(Process):
    def __init__(self, img2od_queue, device=None):
        super().__init__(name="ImageStream")
        self.img2od_queue = img2od_queue
        self.stop_event = Event()
        self.device = device
        
    def set_config(self, src_folder_path = None, fps = 30, profile_save_path = None, eval_size = 50, if_defense = False):
        self.src_folder_path = src_folder_path
        self.fps = fps
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        self.eval_size = eval_size
        self.if_defense = if_defense
        
    def enable_profile(self, flag):
        self.flag = flag
        logger.info(f"{self.__class__.__name__:<12} : internal profiling set to {self.flag}")

    def run(self):
        if self.flag:
            self.profile_run()
        else:
            self._run()
        
    def profile_run(self):    
        from torch.profiler import profile, record_function, ProfilerActivity
        from torch.profiler import schedule
        # torch.cuda.synchronize()   
        torch.cuda.empty_cache() 
        gc.collect()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=False,
                     record_shapes=False
                     ) as prof:
            with record_function(f"{self.__class__.__name__}"):
                # torch.cuda.synchronize()
                self._run()
                # torch.cuda.synchronize()   
        # torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        write_profile(prof, self.profile_save_path)
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            _paths = sorted(glob.glob(f"{self.src_folder_path}/*.pt"))
            print(self.src_folder_path)
            paths = random.sample(_paths, min(self.eval_size, len(_paths)))
            logger.info(f"{self.__class__.__name__:<12} : found {len(paths)} files, sampling {self.eval_size}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            for p in paths:
                if self.stop_event.is_set():
                    break
                
                # data_tensor = self.ultra_fast_load(p, device=self.device)
                data_nparray = self.ultra_fast_load(p, device=None)
                
                if self.if_defense:
                    data_nparray = input_smoothing(data_nparray)
                    # data_nparray = add_gaussian_noise(data_nparray, sigma=0.1)
                # data_nparray = add_gaussian_noise(data_nparray, sigma=0.1)
                
                while self.img2od_queue.full():
                    time.sleep(0.01)
                # if __name__ == "__main__":
                #     print(data_nparray.shape)
                self.img2od_queue.put(data_nparray)
                self.count += 1
                
                time.sleep(1/self.fps)  
                
                torch.cuda.empty_cache()
                gc.collect()
                del data_nparray
                
                
            logger.info(f"{self.__class__.__name__:<12} : completed, sending END signal")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown")

            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)

            self.shutdown()
                
    def shutdown(self):
        while self.img2od_queue.full():
            time.sleep(0.01)
        self.img2od_queue.put(None)
        # self.img2od_queue.close()
        self.stop_event.set()
        
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    # torch.cuda.synchronize()
                except:
                    pass
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
        
                        
    def ultra_fast_load(self, filepath, device=None):
        with open(filepath, 'rb') as f:
            # Step 1: read number of shape dimensions (1 byte)
            shape_dim = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 2: read the shape (8 bytes per dimension)
            shape = np.frombuffer(f.read(8 * shape_dim), dtype=np.int64)

            # Step 3: read dtype string length (1 byte)
            dtype_len = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 4: read the dtype string
            dtype_str = f.read(dtype_len).decode('ascii')  # e.g. '<f4'

            # Step 5: interpret the rest of the file as data
            np_dtype = np.dtype(dtype_str)
            tensor_data = f.read()

            np_array = np.frombuffer(tensor_data, dtype=np_dtype).copy().reshape(shape)
            
            # return np_array
        
            tensor = torch.from_numpy(np_array)

            # Optional: move to device
            if device is not None:
                tensor = tensor.to(device)
                return tensor
            else:
                return np_array

            
if __name__ == "__main__":
    import time 
    import torch
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()
    tmp_queue = mp.Queue()
    instance = imgStream(tmp_queue, device=device)
    instance.set_config(src_folder_path="../../saved/clean", fps=30, profile_save_path="../approach", eval_size=100)
    instance.enable_profile(False)
    instance.run()
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds")
    instance.shutdown()

    
    # import pynvml
    # import time
    # import torch
    # def measure_gpu(device):
    #     pynvml.nvmlInit()
    #     device_id = 0 if device.index is None else device.index
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    #     t1 = time.time()
    #     power_list = []
    #     tmp_queue = mp.Queue()
    #     instance = imgStream(tmp_queue, device=device)
    #     instance.set_config(src_folder_path="../test_src", fps=30, profile_save_path="../profile", eval_size=10)
    #     instance.enable_profile(False)
    #     instance.run()
    #     instance.shutdown()
    #     del tmp_queue
    #     power = pynvml.nvmlDeviceGetPowerUsage(handle)
    #     power_list.append(power)
    #     t2 = time.time()
    #     latency = t2 - t1
    #     s_energy = sum(power_list) / len(power_list) * latency
    #     energy = s_energy / (10 ** 6)
    #     pynvml.nvmlShutdown()
    #     return latency, energy
    
    #     count = 0.0
    #     start_time = time.perf_counter()
    #     pynvml.nvmlInit()
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)
        
    #     end_time = time.perf_counter()
    #     time_elapsed = end_time - start_time
    #     power = pynvml.nvmlDeviceGetPowerUsage(handle)
    #     energy = ( power * time_elapsed ) / (1e6)
    #     content = {
    #         "count" : self.count,
    #         "time" : self.time_elapsed,
    #         "energy" : energy
    #     }
    
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # latency, energy = measure_gpu(device) # watts
    # print(latency, energy)