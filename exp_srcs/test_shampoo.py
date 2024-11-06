'''
https://github.com/facebookresearch/optimizers/blob/89fd01d609019ec5bced42e32be3cdccfa59bab6/distributed_shampoo/examples/fsdp_cifar10_example.py#L83-L129
https://github.com/facebookresearch/optimizers/blob/89fd01d609019ec5bced42e32be3cdccfa59bab6/distributed_shampoo/examples/fully_shard_cifar10_example.py#L106-L147
https://github.com/facebookresearch/optimizers/blob/89fd01d609019ec5bced42e32be3cdccfa59bab6/distributed_shampoo/examples/hsdp_cifar10_example.py#L86-L144

https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/distributed_shampoo.py#L499
https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_fully_shard_distributor.py#L18
https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_fsdp_distributor.py#L33
https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_hsdp_distributor.py#L44
https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy
'''

import os
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, FSDPShampooConfig, FullyShardShampooConfig, HSDPShampooConfig
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata

from transformers import AutoTokenizer
from model import Transformer
from utils import (
    set_seed, 
    init_dist, 
    cleanup_distributed, 
    print_message_with_master_process,
)

# USE_FSDP1=True
USE_FSDP1=False

def test_shampoo():
    rank, world_rank, world_size, device = init_dist()
    DP, SHARD = (
        int(os.environ["DP"]), 
        int(os.environ["SHARD"]), 
    )
    USE_FSDP = DP*SHARD != 1
    USE_HSDP = (DP>1) and (SHARD>1)
    print_message_with_master_process(rank, f'''
    world_size: {world_size} 
    DP: {DP} 
    SHARD: {SHARD}
    USE_FSDP: {USE_FSDP}
    USE_FSDP1: {USE_FSDP1}
    USE_HSDP: {USE_HSDP}
    ''')

    set_seed()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = 1024, 8, 4
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to(device)

    if (not USE_FSDP1) or (USE_HSDP):
        from torch.distributed.device_mesh import init_device_mesh
        assert (world_size == DP*SHARD)
        fsdp_mesh = init_device_mesh(
            "cuda", 
            (DP, SHARD), 
            mesh_dim_names=("replicate", "shard"),
        )

    if USE_FSDP:
        if USE_FSDP1 and (not USE_HSDP):
            model = FSDP(
                model,
                use_orig_params=True,
            )
        elif USE_FSDP1 and USE_HSDP:
            model = FSDP(
                model, 
                device_mesh=fsdp_mesh, 
                use_orig_params=True,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD
            )
        else:
            param_dtype, reduce_dtype = torch.bfloat16, torch.float32
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
            fsdp_config = {"mesh": fsdp_mesh, "mp_policy": mp_policy}

            for layer_id, residual_block in enumerate(model.model.h):
                reshard_after_forward = int(layer_id) < len(model.model.h) - 1
                fully_shard(
                    residual_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            fully_shard(model, **fsdp_config, reshard_after_forward=True)
            
    print_message_with_master_process(rank, f'''
    model: {model}
    ''')

    if USE_FSDP and USE_FSDP1 and (not USE_HSDP):
        distributed_config=FSDPShampooConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model),
        )
    if USE_FSDP and USE_FSDP1 and USE_HSDP:
        distributed_config=HSDPShampooConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model),
            device_mesh=fsdp_mesh,
        )
    elif USE_FSDP and (not USE_FSDP1):
        distributed_config = FullyShardShampooConfig()
    else:
        distributed_config = None

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        epsilon=1e-12,

        weight_decay=1e-05,
        use_decoupled_weight_decay=True,
        
        max_preconditioner_dim=8192,

        # precondition_frequency=1,
        # start_preconditioning_step=-1,
        precondition_frequency=20,
        start_preconditioning_step=20,
        # precondition_frequency=100,
        # start_preconditioning_step=100,

        grafting_config=AdamGraftingConfig(
            beta2=0.999,
            epsilon=1e-12,
        ),
        distributed_config=distributed_config,
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