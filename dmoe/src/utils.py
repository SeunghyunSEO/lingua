import os
from typing import Optional

import torch
import torch.distributed as dist


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank: {rank}, world size: {world_size}")
    return rank, world_size

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

def _get_torch_dtype(fp16: bool, bf16: bool) -> Optional[torch.dtype]:
    if fp16:
        return torch.float16
    elif bf16:
        return torch.bfloat16
    return None