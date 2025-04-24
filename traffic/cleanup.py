import multiprocessing as mp
import time
import logging
from typing import List, Dict, Union, Any, Optional
from queue import Empty
from multiprocessing import queues

def cleanup_resources(processes: List[mp.Process] = None, 
                      queues: List[mp.Queue] = None,
                      pool: Optional[mp.Pool] = None,
                      timeout: float = 5.0,
                      logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Cleans up multiprocessing resources including processes, queues, and optionally a process pool.
    
    Args:
        processes: List of multiprocessing.Process objects to terminate.
        queues: List of multiprocessing.Queue objects to drain and close.
        pool: Optional multiprocessing.Pool to shut down.
        timeout: Maximum time (seconds) to wait for each process termination.
        logger: Optional logger for logging cleanup operations.
        
    Returns:
        Dictionary summarizing cleanup results:
            - terminated_processes: list of process names terminated.
            - failed_processes: list of process names that failed to terminate.
            - drained_items: total number of items drained from queues.
            - closed_queues: number of queues successfully closed.
            - pool_cleanup: status message for pool cleanup.
    """
    import time
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s'
        )
        logger = logging.getLogger(__name__)   
         
    results = {
        "terminated_processes": [],
        "failed_processes": [],
        "drained_items": 0,
        "closed_queues": 0,
        "pool_cleanup": None
    }
    
    processes = processes or []
    queues = queues or []
    
    logger.info(f"Starting cleanup of {len(processes)} processes.")
    for i, process in enumerate(processes):
        if process is None:
            continue
        process_name = getattr(process, 'name', f'Process-{i}')
        if not process.is_alive():
            logger.debug(f"Process {process_name} is not running, skipping termination.")
            results["terminated_processes"].append(process_name)
            continue
        
        logger.info(f"Attempting to terminate process {process_name}.")
        try:
            process.terminate()
            start_time = time.time()
            while process.is_alive() and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            if process.is_alive():
                logger.warning(f"Process {process_name} did not terminate gracefully, attempting kill.")
                if hasattr(process, 'kill'):
                    process.kill()
                    process.join(timeout=1.0)
            if process.is_alive():
                logger.error(f"Failed to terminate process {process_name}.")
                results["failed_processes"].append(process_name)
            else:
                logger.info(f"Successfully terminated process {process_name}.")
                results["terminated_processes"].append(process_name)
        except Exception as e:
            logger.error(f"Error terminating process {process_name}: {e}")
            results["failed_processes"].append(process_name)
    
    logger.info(f"Draining and closing {len(queues)} queues.")
    for i, queue in enumerate(queues):
        logger.info(f"Starting cleanup for queue {i}, queue size: {queue.qsize()}.")    
        if queue is None:
            continue
        # import pdb; pdb.set_trace()
        try:
            # drained = 0
            # while True:
            #     try:
            #         queue.get_nowait()
            #         drained += 1
            #     except mp.queues.Empty:  # Specific exception for empty queue
            #         break
            #     except Exception as e:
            #         logger.error(f"Unexpected error while draining queue: {str(e)}")
            #         break
            # results["drained_items"] += drained
            queue.close()
            queue.join_thread()
            results["closed_queues"] += 1
            logger.debug(f"Queue {i} closed successfully.")
        except Exception as e:
            logger.error(f"Error cleaning queue {i}: {e}")
    
    if pool is not None:
        logger.info("Cleaning up process pool.")
        try:
            pool.close()
            pool.join(timeout)
            # If the pool is still not closed, force termination
            if pool._state != mp.pool.CLOSE:
                pool.terminate()
                pool.join(1.0)
            results["pool_cleanup"] = "Process pool cleaned up successfully."
            logger.info("Process pool cleaned up successfully.")
        except Exception as e:
            results["pool_cleanup"] = f"Failed to clean process pool: {e}"
            logger.error(f"Error cleaning process pool: {e}")
    
    logger.info("Cleanup complete.")
    return results

