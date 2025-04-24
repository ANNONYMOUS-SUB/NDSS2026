from cmpnt.c1_img import imgStream
from cmpnt.c2_det import odStream, var_odStream, var2_odStream
from cmpnt.c3_fr import frStream
from cmpnt.c4_lpr import lprStream
from cmpnt.c5_cap import capStream
from cmpnt.c6_kr import krStream
from cmpnt.c7_udp import udpStream
from cleanup import cleanup_resources
import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
import logging
import argparse
import json
import glob
from legacy_flops import summary

parser = argparse.ArgumentParser(description="Traffic Monitoring Pipeline")
parser.add_argument("--model_id", type=int, default=0, help="Model ID for object detection")
parser.add_argument('--algorithm', type=str, default=None, choices=["overload", 
                                                                    "slowtrack", 
                                                                    "phantom", 
                                                                    "teaspoon", 
                                                                    "clean",
                                                                    "weighted",
                                                                    "unweighted"], help="algorithm not found")
parser.add_argument('--target_idx', type=int, nargs='+', default=None, help="List of numbers, unavailable for baseline")
parser.add_argument("--ps_path", type=str, default="./profile_var_2", help="Path to save profile data")
parser.add_argument("--eval_size", type=int, default=100, help="num of images to evaluate")
parser.add_argument("--profiling", action="store_true", help="use internal pytorch profiler")
parser.add_argument("--watch", type=float, default=1.0, help="use internal pytorch profiler")
parser.add_argument("--defense", action="store_true", help="apply defense mechanism")
args = parser.parse_args()

if args.target_idx:
    target_indices = ('_'.join(map(str, args.target_idx)))
else:
    target_indices = "none"
    
base_dir = "../saved"
model_id = args.model_id
algorithm = args.algorithm
eval_size = args.eval_size
profiling = args.profiling
watch_interval = args.watch



if algorithm in ["tea_400","tea_100","norm","area","norm_and_area","eps_2","eps_8"]:
    base_dir = "../ablation/saved"

if algorithm == "clean":
    input_dir = "../saved/clean"
    ps_path = os.path.join(args.ps_path, "clean")
elif algorithm == "weighted":
    input_dir = "../case_study/adv_weighted"
    ps_path = os.path.join(args.ps_path, "weighted")
elif algorithm == "unweighted":
    input_dir = "../case_study/adv_unweighted"
    ps_path = os.path.join(args.ps_path, "unweighted")
elif algorithm is None:
    input_dir = "./test_src"
    ps_path = "./test_profile"
else:
    ps_path = os.path.join(args.ps_path, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    input_dir = os.path.join(base_dir, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    
# print(f"Input directory: {input_dir}")
# print(f"Profile save path: {ps_path}")

os.makedirs(ps_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

            
if __name__ == "__main__":
    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    title = "Running: Traffic Monitoring Pipeline VAR 2"
    subtitle = f"Algorithm: {algorithm}, Model ID: {model_id}"
    subsubtitle = f"Target Indices: {target_indices}"
    
    print("\n\n" + "=" * 80)
    print("|" + " ".center(78) + "|")
    print("|" + title.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subsubtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + date_time.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("=" * 80 + "\n\n")
    
    start_time = time.perf_counter()
    
    cuda_device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {cuda_device_count}")
    cuda_idx = cuda_device_count - 1
    
    # torch.cuda.synchronize()

    mp.set_start_method("spawn")
    
    ms = 1000
    
    img2od_queue = Queue(maxsize=ms)
    od2fr_queue = Queue(maxsize=ms)
    od2lpr_queue = Queue(maxsize=ms)
    od2cap_queue = Queue(maxsize=ms)
    fr2kr_queue = Queue(maxsize=ms)
    lpr2kr_queue = Queue(maxsize=ms)
    cap2udp_queue = Queue(maxsize=ms)
    kr2udp_queue = Queue(maxsize=ms)
    
    od2fr_queue.put(None)
    od2lpr_queue.put(None)
    lpr2kr_queue.put(None)
    fr2kr_queue.put(None)
    kr2udp_queue.put(None)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # img_stream = imgStream(img2od_queue, "cuda:0")
    # od_stream = var2_odStream(img2od_queue, od2cap_queue, "cuda:1")  # Changed: None instead of od2cap_queue
    # # fr_stream = frStream(od2fr_queue, fr2kr_queue, "cuda:2")
    # # lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, "cuda:3")
    # cap_stream = capStream(od2cap_queue, cap2udp_queue, "cuda:4")
    # # kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, "cuda:5")
    # udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, "cuda:6")  # Changed: None instead of cap2udp_queue
    
    img_stream = imgStream(img2od_queue, device)
    od_stream = var2_odStream(img2od_queue, od2cap_queue, device)  # Changed: None instead of od2cap_queue
    # fr_stream = frStream(od2fr_queue, fr2kr_queue, "cuda:2")
    # lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, "cuda:3")
    cap_stream = capStream(od2cap_queue, cap2udp_queue, device)
    # kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, "cuda:5")
    udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, device)  # Changed: None instead of cap2udp_queue
    
    img_stream.set_config(src_folder_path = input_dir, fps = 30, profile_save_path=ps_path, eval_size=eval_size, if_defense=args.defense)
    od_stream.set_config(model_id=model_id, profile_save_path=ps_path)
    # fr_stream.set_config(profile_save_path=ps_path)
    # lpr_stream.set_config(profile_save_path=ps_path)
    cap_stream.set_config(profile_save_path=ps_path)
    # kr_stream.set_config(embedding_path="./face_embeddings", profile_save_path=ps_path)
    udp_Stream.set_config(profile_save_path=ps_path)

    
    processes = [img_stream, od_stream, cap_stream, udp_Stream]  # Removed cap_stream
    queues = [img2od_queue, od2cap_queue, cap2udp_queue]  # Removed od2cap_queue and cap2udp_queue
    queue_names = ["img2od", "od2cap_queue","cap2udp", "cap2udp"]  # Removed "od2cap" and "cap2udp"

    for sub_p in processes:
        sub_p.enable_profile(profiling)
        
    from queue_watch import QueueWatch
    queue_watcher = QueueWatch(queues, queue_names, processes, ps_path)
    queue_watcher.set_config(sleep_time=watch_interval)
    queue_watcher.start()
    
    for p in processes:
        p.start()
        time.sleep(0.1)  # Optional: small delay to ensure all processes start properly
    
    try:
        for p in processes:
            p.join()
            time.sleep(0.1)  # Optional: small delay to ensure all processes finish properly
        queue_watcher.join()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user in MAIN process")
    except Exception as e:
        logger.error(f"Error in MAIN process: {str(e)}")
    finally:
        logger.info("Reach FINALLY block in MAIN process")
        cleanup_resources(processes, queues)
        logger.info("Cleaned up resources in MAIN process")
        
    end_time = time.perf_counter()
    
    time.sleep(5)
    # torch.cuda.synchronize()
    time.sleep(5)
    torch.cuda.empty_cache()
    
    # summary(ps_path, start_time, end_time)
    