from multiprocessing import Process, Queue, Event
import time
from queue import Empty
import logging
import json
from datetime import datetime
import os
import pynvml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueueWatch(Process):
    def __init__(self, queues, queue_names, processes, ps_path):
        super().__init__(name="QueueWatch")
        self.queues = queues
        self.queue_names = queue_names
        self.stop_event = Event()
        self.processes = processes
        self.ps_path = ps_path
        self.log_entries = []
        
    def set_config(self, buffer_size=6, sleep_time=5):
        self.buffer_size = buffer_size
        self.qsize_buffer = [0] * (self.buffer_size - 1) 
        self.qsize_buffer.append(1)
        self.sleep_time = sleep_time
        
    def run(self):
        pynvml.nvmlInit()
        self.device_id = 0 
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        count = 0
        max_count = 120
        power_list = []
        while not self.stop_event.is_set():
            string = ""
            qsize_sum = 0
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "queues": {},
                "total_size": 0
            }
            
            for queue, name in zip(self.queues, self.queue_names):
                qsize = queue.qsize()
                sub_string = name + ": " + str(qsize)
                string += sub_string + " | "
                qsize_sum += queue.qsize()
                
                log_entry["queues"][name] = qsize
                pass
            
            log_entry["total_size"] = qsize_sum
            self.log_entries.append(log_entry)
            
            self.qsize_buffer.pop(0)
            self.qsize_buffer.append(qsize_sum)

            logger.info(string)
            time.sleep(self.sleep_time)
                
            if count <= max_count:
                count += 1
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                power_list.append(power)
            else:
                print("="*80 + "> Joule/second", sum(power_list) / len(power_list))
                
            if sum(self.qsize_buffer) == 0:
                logger.info(f"All queues empty, job done")
                
                if self.ps_path:
                    try:
                        os.makedirs(os.path.dirname(self.ps_path), exist_ok=True)
                        dest_path = os.path.join(self.ps_path, "QUEUEWATCH.json")
                        with open(dest_path, 'w') as f:
                            json.dump(self.log_entries, f, indent=2)
                        logger.info(f"Queue logs written to {dest_path}")
                    except Exception as e:
                        logger.error(f"Failed to write to {dest_path}: {e}")
                        
                print("="*80 + "> Joule/second", sum(power_list) / len(power_list))
                self.stop_event.set()
            

            