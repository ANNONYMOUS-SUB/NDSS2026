import sys
sys.path.append("../")
sys.path.append("../../")
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
from queue import Empty
from model_zoo import load_from_pretrained
import gc

import pynvml
import json



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class odStream(Process):
    def __init__(self, img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device=None):
        super().__init__(name="ODStream")
        self.img2od_queue = img2od_queue
        self.od2fr_queue = od2fr_queue
        self.od2lpr_queue = od2lpr_queue
        self.od2cap_queue = od2cap_queue
        self.stop_event = Event()
        self.device = device
        
        
    def set_config(self, model_id, profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def shutdown(self):
        while self.od2fr_queue.full() or self.od2lpr_queue.full() or self.od2cap_queue.full(): 
            time.sleep(0.01)
            
        self.od2fr_queue.put(None)  
        self.od2lpr_queue.put(None) 
        self.od2cap_queue.put(None) 
        
        # self.od2fr_queue.close()
        # self.od2lpr_queue.close()
        # self.od2cap_queue.close()  

        self.stop_event.set()
        
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
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
            self.p_time = 0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            self.model, self.processor = load_from_pretrained(ckpt=self.model_id, 
                                                            num_q=1000, 
                                                            device=self.device)
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : OD model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                
                try:
                    data = self.img2od_queue.get(block=False)
                    if data is None:  # End signal
                        # self.img2od_queue.join_thread()
                        break
                except Empty:
                    continue
                
                # data_tensor = torch.from_numpy(data).to(self.device)
                time_1 = time.perf_counter()
                if isinstance(data, np.ndarray):
                    # Convert numpy array to tensor and ensure it's float32
                    data_tensor = torch.from_numpy(data).float().to(self.device)
                elif isinstance(data, torch.Tensor):
                    # Convert existing tensor to float32 if needed
                    data_tensor = data.float().to(self.device)
                    
                with torch.no_grad():
                    preds = self.model(data_tensor) 
                    output = self.processor.post_process_object_detection(
                        preds,
                        threshold=0.25,
                        target_sizes=[data_tensor.shape[2:]]
                    )[0]
                    self.count += 1
                
                _, labels, boxes = output["scores"], output["labels"], output["boxes"]

                face_indices = (labels == 0).nonzero(as_tuple=True)[0]
                plate_indices = (labels == 2).nonzero(as_tuple=True)[0]
                
                if len(face_indices) > 0:
                    for idx in face_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            while self.od2fr_queue.full():
                                time.sleep(0.01)
                            self.od2fr_queue.put(cropped_np)
                            del cropped_np
                
                if len(plate_indices) > 0:
                    for idx in plate_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            while self.od2lpr_queue.full():
                                time.sleep(0.01)
                            self.od2lpr_queue.put(cropped_np)
                            del cropped_np
                                
                if len(face_indices) > 0 or len(plate_indices) > 0:
                    all_indices = torch.cat([face_indices, plate_indices]) if len(face_indices) > 0 and len(plate_indices) > 0 else face_indices if len(face_indices) > 0 else plate_indices
                    merged_box = self.merge_bbox(all_indices, boxes)
                    if self.valid_bbox(merged_box):
                        cropped_np = self.crop_box(data_tensor, merged_box)  
                        while self.od2cap_queue.full():
                            time.sleep(0.01)
                        self.od2cap_queue.put(cropped_np)
                        del cropped_np
                    del  all_indices, merged_box
                        
                # time.sleep(7.326506948797032 / 100 * 0.05)
                time_2 = time.perf_counter()
                self.p_time += (time_2 - time_1)
                
                torch.cuda.empty_cache()
                gc.collect()
                del data_tensor, preds, output, labels, boxes, face_indices, plate_indices
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : interrupted by user")
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

    def crop_box(self, data_tensor, box=None):
        try:
            if box is not None:
                x1, y1, x2, y2 = box
                height, width = data_tensor.shape[2], data_tensor.shape[3]
                cropped_tensor = data_tensor[:, :, int(y1):int(y2), int(x1):int(x2)].clone()
            else:
                cropped_tensor = data_tensor.clone()
            cropped_np = cropped_tensor.cpu().numpy()
            return cropped_np
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in cropping box: {str(e)}")
    
    def denormalize(self, tensor):
        """
        Denormalizes a tensor using the provided mean and std.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        return tensor

    def merge_bbox(self, all_indices, boxes):
        # Get coordinates for all boxes
        all_boxes = boxes[all_indices]
        
        # Find the min/max coordinates to create a single large bounding box
        x_min = torch.min(all_boxes[:, 0])
        y_min = torch.min(all_boxes[:, 1])
        x_max = torch.max(all_boxes[:, 2])
        y_max = torch.max(all_boxes[:, 3])
        
        # Create merged bounding box
        merged_box = torch.tensor([x_min, y_min, x_max, y_max]).to(self.device)
    
        return merged_box
    
    def valid_bbox(self, box):
        try:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return False
            if ((x2 - x1) < 7) or ((y2 - y1) < 7): # ensure larger than kernel size
                return False
            return True
        except Exception as e:
            return False


class var_odStream(Process):
    def __init__(self, img2od_queue, od2fr_queue, od2lpr_queue, device=None):
        super().__init__(name="ODStream")
        self.img2od_queue = img2od_queue
        self.od2fr_queue = od2fr_queue
        self.od2lpr_queue = od2lpr_queue
        self.stop_event = Event()
        self.device = device
        
        
    def set_config(self, model_id, profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def shutdown(self):
        while self.od2fr_queue.full() or self.od2lpr_queue.full(): 
            time.sleep(0.01)
            
        self.od2fr_queue.put(None)  
        self.od2lpr_queue.put(None) 
        
        # self.od2fr_queue.close()
        # self.od2lpr_queue.close()

        self.stop_event.set()
        
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
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
            
            self.model, self.processor = load_from_pretrained(ckpt=self.model_id, 
                                                            num_q=1000, 
                                                            device=self.device)
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : OD model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                
                try:
                    data = self.img2od_queue.get(block=False)
                    if data is None:  # End signal
                        # self.img2od_queue.join_thread()
                        break
                except Empty:
                    continue
                
                # data_tensor = torch.from_numpy(data).to(self.device)
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).to(self.device)
                elif isinstance(data, torch.Tensor):
                    data_tensor = data.to(self.device)
                    
                with torch.no_grad():
                    preds = self.model(data_tensor) 
                    output = self.processor.post_process_object_detection(
                        preds,
                        threshold=0.25,
                        target_sizes=[data_tensor.shape[2:]]
                    )[0]
                    self.count += 1
                
                _, labels, boxes = output["scores"], output["labels"], output["boxes"]

                face_indices = (labels == 0).nonzero(as_tuple=True)[0]
                plate_indices = (labels == 2).nonzero(as_tuple=True)[0]
                
                if len(face_indices) > 0:
                    for idx in face_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            while self.od2fr_queue.full():
                                time.sleep(0.01)
                            self.od2fr_queue.put(cropped_np)
                
                if len(plate_indices) > 0:
                    for idx in plate_indices:
                        if self.valid_bbox(boxes[idx]):
                            cropped_np = self.crop_box(data_tensor, boxes[idx])
                            while self.od2lpr_queue.full():
                                time.sleep(0.01)
                            self.od2lpr_queue.put(cropped_np)
                        
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
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

    def crop_box(self, data_tensor, box=None):
        try:
            if box is not None:
                x1, y1, x2, y2 = box
                height, width = data_tensor.shape[2], data_tensor.shape[3]
                cropped_tensor = data_tensor[:, :, int(y1):int(y2), int(x1):int(x2)].clone()
            else:
                cropped_tensor = data_tensor.clone()
            cropped_np = cropped_tensor.cpu().numpy()
            return cropped_np
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in cropping box: {str(e)}")
    
    def denormalize(self, tensor):
        """
        Denormalizes a tensor using the provided mean and std.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        return tensor

    def merge_bbox(self, all_indices, boxes):
        # Get coordinates for all boxes
        all_boxes = boxes[all_indices]
        
        # Find the min/max coordinates to create a single large bounding box
        x_min = torch.min(all_boxes[:, 0])
        y_min = torch.min(all_boxes[:, 1])
        x_max = torch.max(all_boxes[:, 2])
        y_max = torch.max(all_boxes[:, 3])
        
        # Create merged bounding box
        merged_box = torch.tensor([x_min, y_min, x_max, y_max]).to(self.device)
    
        return merged_box
    
    def valid_bbox(self, box):
        try:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return False
            if ((x2 - x1) < 7) or ((y2 - y1) < 7): # ensure larger than kernel size
                return False
            return True
        except Exception as e:
            return False


class var2_odStream(Process):
    def __init__(self, img2od_queue, od2cap_queue, device=None):
        super().__init__(name="ODStream")
        self.img2od_queue = img2od_queue
        self.od2cap_queue = od2cap_queue
        self.stop_event = Event()
        self.device = device
        
        
    def set_config(self, model_id, profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def shutdown(self):
        while self.od2cap_queue.full(): 
            time.sleep(0.01)
            
        self.od2cap_queue.put(None) 
        
        # self.od2fr_queue.close()
        # self.od2lpr_queue.close()
        # self.od2cap_queue.close()  

        self.stop_event.set()
        
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
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
            self.p_time = 0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            self.model, self.processor = load_from_pretrained(ckpt=self.model_id, 
                                                            num_q=1000, 
                                                            device=self.device)
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : OD model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                
                try:
                    data = self.img2od_queue.get(block=False)
                    if data is None:  # End signal
                        # self.img2od_queue.join_thread()
                        break
                except Empty:
                    time.sleep(0.01)
                    continue
                time_1 = time.perf_counter()
                # data_tensor = torch.from_numpy(data).to(self.device)
                if isinstance(data, np.ndarray):
                    # Convert numpy array to tensor and ensure it's float32
                    data_tensor = torch.from_numpy(data).float().to(self.device)
                elif isinstance(data, torch.Tensor):
                    # Convert existing tensor to float32 if needed
                    data_tensor = data.float().to(self.device)
                    
                with torch.no_grad():
                    preds = self.model(data_tensor) 
                    output = self.processor.post_process_object_detection(
                        preds,
                        threshold=0.25,
                        target_sizes=[data_tensor.shape[2:]]
                    )[0]
                    self.count += 1
                
                _, labels, boxes = output["scores"], output["labels"], output["boxes"]

                face_indices = (labels == 0).nonzero(as_tuple=True)[0]
                plate_indices = (labels == 2).nonzero(as_tuple=True)[0]
                                
                if len(face_indices) > 0 or len(plate_indices) > 0:
                    all_indices = torch.cat([face_indices, plate_indices]) if len(face_indices) > 0 and len(plate_indices) > 0 else face_indices if len(face_indices) > 0 else plate_indices
                    merged_box = self.merge_bbox(all_indices, boxes)
                    if self.valid_bbox(merged_box):
                        cropped_np = self.crop_box(data_tensor, merged_box)  
                        while self.od2cap_queue.full():
                            time.sleep(0.01)
                        self.od2cap_queue.put(cropped_np)
                        del cropped_np
                    del  all_indices, merged_box
                    
                time_2 = time.perf_counter()
                self.p_time += (time_2 - time_1)
                
                torch.cuda.empty_cache()
                gc.collect()
                del data_tensor, preds, output, labels, boxes, face_indices, plate_indices
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : interrupted by user")
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

    def crop_box(self, data_tensor, box=None):
        try:
            if box is not None:
                x1, y1, x2, y2 = box
                height, width = data_tensor.shape[2], data_tensor.shape[3]
                cropped_tensor = data_tensor[:, :, int(y1):int(y2), int(x1):int(x2)].clone()
            else:
                cropped_tensor = data_tensor.clone()
            cropped_np = cropped_tensor.cpu().numpy()
            return cropped_np
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in cropping box: {str(e)}")
    
    def denormalize(self, tensor):
        """
        Denormalizes a tensor using the provided mean and std.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        return tensor

    def merge_bbox(self, all_indices, boxes):
        # Get coordinates for all boxes
        all_boxes = boxes[all_indices]
        
        # Find the min/max coordinates to create a single large bounding box
        x_min = torch.min(all_boxes[:, 0])
        y_min = torch.min(all_boxes[:, 1])
        x_max = torch.max(all_boxes[:, 2])
        y_max = torch.max(all_boxes[:, 3])
        
        # Create merged bounding box
        merged_box = torch.tensor([x_min, y_min, x_max, y_max]).to(self.device)
    
        return merged_box
    
    def valid_bbox(self, box):
        try:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                return False
            if ((x2 - x1) < 7) or ((y2 - y1) < 7): # ensure larger than kernel size
                return False
            return True
        except Exception as e:
            return False

if __name__ == "__main__":
    pass