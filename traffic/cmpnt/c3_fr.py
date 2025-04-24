from multiprocessing import Process, Queue, Event
from queue import Empty
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from PIL import Image
import torch
import sys
sys.path.append("../")
from legacy_flops import FLOPs_DECORATOR, write_profile
import torch.nn.functional as F
import logging
import numpy as np
import os
import time
import gc

import pynvml
import json



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class frStream(Process):
    def __init__(self, od2fr_queue, fr2kr_queue, device=None):
        super().__init__(name="FRStream")
        self.od2fr_queue = od2fr_queue
        self.fr2kr_queue = fr2kr_queue
        self.stop_event = Event()
        self.device = device 
        
    def set_config(self, model_id="vggface2", profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

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
            self.p_time = 0.0
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self.facenet = InceptionResnetV1(pretrained=f"{self.model_id}").eval().to(self.device)
            logger.info(f"{self.__class__.__name__:<12} : FR model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                try:
                    data = self.od2fr_queue.get(block=False)
                    if data is None:  # End signal
                        # self.od2fr_queue.join_thread()
                        break
                except Empty:
                    continue
                
                time_1 = time.perf_counter()
                # Process the image
                data_tensor = torch.from_numpy(data).to(self.device)
                padded_image = self.facenet_padding(data_tensor)
                
                with torch.no_grad():
                    face_embedding = self.facenet(padded_image).cpu().numpy()
                    
                while self.fr2kr_queue.full():
                    time.sleep(0.01)
                self.fr2kr_queue.put(face_embedding)
                self.count += 1
                # time.sleep(23.296313150203787 / 479 * 0.05)

                time_2 = time.perf_counter()
                self.p_time += (time_2 - time_1)
                
                torch.cuda.empty_cache()
                gc.collect()
                
                del data, data_tensor, padded_image, face_embedding
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : interruptted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy,
                "p_time" : self.p_time,
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)
            self.shutdown()
            
            
    def shutdown(self):
        while self.fr2kr_queue.full():
            time.sleep(0.01)
        self.fr2kr_queue.put(None)
        # self.fr2kr_queue.close()
        self.stop_event.set()
        
        try:
            if hasattr(self, 'facenet'):
                try:
                    self.facenet = self.facenet.to("cpu")
                except:
                    pass
                del self.facenet
                self.facenet = None
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
            
    def facenet_padding(self, image, min_size=160):
        # Get current dimensions
        if image.dim() == 4:
            image = image.squeeze(0)
            
        _, height, width = image.shape
        
        # Check if padding is needed
        if height < min_size or width < min_size:
            # Calculate padding
            pad_height = max(0, min_size - height)
            pad_width = max(0, min_size - width)
            
            # Calculate padding for each side
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            image = F.pad(image.unsqueeze(0), 
                        (pad_left, pad_right, pad_top, pad_bottom), 
                        mode='constant', value=0)
            image = image.squeeze(0)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image