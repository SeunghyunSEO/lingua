import os
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, FSDPShampooConfig
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata

from transformers import AutoTokenizer
from model import Transformer
from utils import (
    set_seed, 
    init_dist, 
    cleanup_distributed, 
    print_message_with_master_process,
)

def test_shampoo():
    rank, world_rank, world_size, device = init_dist()
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = 1024, 8, 4
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to(device)


    from torch.distributed.device_mesh import init_device_mesh
    DP, SHARD = (
        int(os.environ["DP"]), 
        int(os.environ["SHARD"]), 
    )
    print_message_with_master_process(rank, f"{world_size} {DP} {SHARD}")
    assert (world_size == DP*SHARD)
    mesh_2d = init_device_mesh(
        "cuda", 
        (DP, SHARD), 
        mesh_dim_names=("replicate", "shard"),
    )
    if DP*SHARD != 1:
        model = FSDP(
            model, 
            device_mesh=mesh_2d, 
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD
        )
    print_message_with_master_process(rank, f'''
    model: {model}
    ''')

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        epsilon=1e-12,
        weight_decay=1e-05,
        max_preconditioner_dim=8192,
        use_decoupled_weight_decay=True,
        
        precondition_frequency=20,
        start_preconditioning_step=20,

        grafting_config=AdamGraftingConfig(
            beta2=0.999,
            epsilon=1e-12,
        ),
        distributed_config=FSDPShampooConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model),
        ),
    )

    sent = "i love shampoo so much"
    input_ids = tokenizer(sent, return_tensors='pt').to(device)
    x = input_ids['input_ids']
    labels = input_ids['input_ids'][:, 1:]
    model.train()

    num_iter = 100
    log_interval = 10
    for i in range(num_iter):
        pred = model(x)
        loss = F.cross_entropy(pred[..., :-1, :].flatten(0, 1), labels.flatten(0, 1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1)%log_interval == 0:
            print_message_with_master_process(rank, f'({i+1}th step) loss: {loss}')

    cleanup_distributed()

if __name__ == "__main__":
    test_shampoo()