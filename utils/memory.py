import os
import psutil
import torch
import logging

class MemoryTracker:
    def __init__(self):
        self.logger = logging.getLogger('memory_tracker')
    
    def get_gpu_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        return 0
    
    def get_cpu_memory_usage(self) -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2  # Convert to MB
    
    def log_memory_usage(self, step: int, prefix: str = ""):
        cpu_mem = self.get_cpu_memory_usage()
        gpu_mem = self.get_gpu_memory_usage()
        message = f"{prefix}Step {step} - CPU: {cpu_mem:.2f} MB, GPU: {gpu_mem:.2f} MB"
        self.logger.info(message)