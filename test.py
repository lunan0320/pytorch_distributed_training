import torch
import torch.distributed as dist
import os
import time

print(os.environ)
print("|| MASTER_ADDR:",os.environ["MASTER_ADDR"],
     "|| MASTER_PORT:",os.environ["MASTER_PORT"],
     "|| LOCAL_RANK:",os.environ["LOCAL_RANK"],
     "|| RANK:",os.environ["RANK"], 
     "|| WORLD_SIZE:",os.environ["WORLD_SIZE"])
print()

dist.init_process_group('nccl')
time.sleep(30)
dist.destroy_process_group()
