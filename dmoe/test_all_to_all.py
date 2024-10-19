'''
https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
'''
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.utils import (
    init_dist,
    print_message_with_master_process
)

def test_comm():
    rank, world_size = init_dist()

    input = list(
        (torch.arange(4, device=f'cuda:{rank}') + rank * 4).chunk(4)
    )
    print(f'(input) (rank {rank}): {input}')
    output = list(
        (torch.empty([world_size], dtype=torch.int64, device=f'cuda:{rank}')).chunk(4)
    )
    print(f'(output before all_to_all) (rank {rank}): {output}')
    dist.all_to_all(
        output, 
        input,
        group=None,
        async_op=False,
    )
    print(f'(output after all_to_all) (rank {rank}): {output}')

    print_message_with_master_process(rank, '*****'*20)
    dist.barrier()
    print()

if __name__ == "__main__":
    test_comm()