# torch native parallelism

## dist pdb tracer for debugging

```python
from multiprocessing_pdb import MultiprocessingPdb
Tra = MultiprocessingPdb().set_trace

def dummy_code_block(...):
    Tra()
```

## make sure your dist setting is correct

```bash
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23459
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
utils.py
```


## run scripts for torch/nccl all to all comm

```bash
# cd /path/to/dir/lingua/exp_srcs
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23459 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_all_to_all.py
```

```python
rank: 0, world size: 2
rank: 1, world size: 2

    (input) (rank 1): [tensor([0, 1, 2, 3, 4], device='cuda:0'), tensor([10, 11, 12, 13, 14, 15], device='cuda:1')]
    (input_splits) (rank 1): [[2, 3], [3, 3]]
    (input_splitted) (rank 1): [tensor([10, 11, 12], device='cuda:1'), tensor([13, 14, 15], device='cuda:1')]
    (output before all_to_all) (rank 1): [tensor([0, 0, 0], device='cuda:1'), tensor([0, 0, 0], device='cuda:1')]

    (input) (rank 0): [tensor([0, 1, 2, 3, 4], device='cuda:0'), tensor([10, 11, 12, 13, 14, 15], device='cuda:1')]
    (input_splits) (rank 0): [[2, 3], [3, 3]]
    (input_splitted) (rank 0): [tensor([0, 1], device='cuda:0'), tensor([2, 3, 4], device='cuda:0')]
    (output before all_to_all) (rank 0): [tensor([0, 0], device='cuda:0'), tensor([0, 0, 0], device='cuda:0')]
    
    (output after all_to_all) (rank 1): [tensor([2, 3, 4], device='cuda:1'), tensor([13, 14, 15], device='cuda:1')]
    (output after all_to_all) (rank 0): [tensor([0, 1], device='cuda:0'), tensor([10, 11, 12], device='cuda:0')]
```


## 3d parallelism test

### FSDP

```bash
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23459
```

```bash
# cd /path/to/dir/lingua/exp_srcs
export DP=1 &&\
export SHARD=2 &&\
export TP=1
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```

<details>

```python
    mesh_3d: DeviceMesh('cuda', [[[0], [1]]], mesh_dim_names=('replicate', 'shard', 'tp'))
    fsdp_mesh: DeviceMesh('cuda', [[0, 1]], mesh_dim_names=('replicate', 'shard'))
    tp_mesh: DeviceMesh('cuda', [0], mesh_dim_names=('tp',))
    replicate_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd5641c87b0>
    shard_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd5641c8670>
    tp_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd5641c8530>
    

    model: FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (model): ModuleDict(
      (wte): Embedding(50257, 1024)
      (h): ModuleList(
        (0-3): 4 x ResidualBlock(
          (ln1): LayerNorm()
          (attn): Attention(
            (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (ln2): LayerNorm()
          (mlp): MLP(
            (ffn1): Linear(in_features=1024, out_features=4096, bias=False)
            (act): GELU(approximate='none')
            (ffn2): Linear(in_features=4096, out_features=1024, bias=False)
          )
        )
      )
      (ln): LayerNorm()
    )
    (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
  )
)
    
0th step loss: 11.174592018127441
1th step loss: 0.5179086923599243
2th step loss: 16.637855529785156
3th step loss: 15.226019859313965
4th step loss: 11.38132095336914
5th step loss: 10.103214263916016
```

</details>

![DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_True](./assets/images/DP_1_SHARD_2_TP_1_USE_LOSS_PARALLEL_False.png)


### DP only

```bash
export DP=2 &&\
export SHARD=1 &&\
export TP=1
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```

<details>

```python
    mesh_3d: DeviceMesh('cuda', [[[0]], [[1]]], mesh_dim_names=('replicate', 'shard', 'tp'))
    fsdp_mesh: DeviceMesh('cuda', [[0], [1]], mesh_dim_names=('replicate', 'shard'))
    tp_mesh: DeviceMesh('cuda', [0], mesh_dim_names=('tp',))
    replicate_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd1787c0130>
    shard_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd16dcc71b0>
    tp_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7fd1787c0a30>
    
/mnt/chatbot30TB/shseo/venv/lingua/lib/python3.10/site-packages/torch/distributed/fsdp/_init_utils.py:444: UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.HYBRID_SHARD since the world size is 1.
  warnings.warn(

    model: FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (model): ModuleDict(
      (wte): Embedding(50257, 1024)
      (h): ModuleList(
        (0-3): 4 x ResidualBlock(
          (ln1): LayerNorm()
          (attn): Attention(
            (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (ln2): LayerNorm()
          (mlp): MLP(
            (ffn1): Linear(in_features=1024, out_features=4096, bias=False)
            (act): GELU(approximate='none')
            (ffn2): Linear(in_features=4096, out_features=1024, bias=False)
          )
        )
      )
      (ln): LayerNorm()
    )
    (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
  )
)
    
/mnt/chatbot30TB/shseo/venv/lingua/lib/python3.10/site-packages/torch/distributed/fsdp/_init_utils.py:444: UserWarning: FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.HYBRID_SHARD since the world size is 1.
  warnings.warn(
0th step loss: 11.174592018127441
1th step loss: 0.5180912017822266
2th step loss: 16.640453338623047
3th step loss: 15.22722053527832
4th step loss: 11.383994102478027
5th step loss: 10.128301620483398
```

</details>

![DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_True](./assets/images/DP_2_SHARD_1_TP_1_USE_LOSS_PARALLEL_False.png)


### TP only

```bash
export DP=1 &&\
export SHARD=1 &&\
export TP=2
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```

<details>

```python
    mesh_3d: DeviceMesh('cuda', [[[0, 1]]], mesh_dim_names=('replicate', 'shard', 'tp'))
    fsdp_mesh: DeviceMesh('cuda', [[0]], mesh_dim_names=('replicate', 'shard'))
    tp_mesh: DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',))
    replicate_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f90901c01b0>
    shard_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f90901c1c30>
    tp_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f90901c1c70>
    

    model: Transformer(
  (model): ModuleDict(
    (wte): Embedding(50257, 1024)
    (h): ModuleList(
      (0-3): 4 x ResidualBlock(
        (ln1): LayerNorm()
        (attn): Attention(
          (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
          (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
        )
        (ln2): LayerNorm()
        (mlp): MLP(
          (ffn1): Linear(in_features=1024, out_features=4096, bias=False)
          (act): GELU(approximate='none')
          (ffn2): Linear(in_features=4096, out_features=1024, bias=False)
        )
      )
    )
    (ln): LayerNorm()
  )
  (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
)
    
0th step loss: DTensor(local_tensor=11.174638748168945, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
1th step loss: DTensor(local_tensor=0.5185606479644775, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
2th step loss: DTensor(local_tensor=16.62112808227539, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
3th step loss: DTensor(local_tensor=15.390058517456055, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
4th step loss: DTensor(local_tensor=11.48178482055664, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
5th step loss: DTensor(local_tensor=9.973380088806152, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
```

</details>

![DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_True](./assets/images/DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_True.png)


### TP w/o loss parallel

<details>

```python
0th step loss: 11.174637794494629
1th step loss: 0.5185607075691223
2th step loss: 16.62110137939453
3th step loss: 15.38946533203125
4th step loss: 11.482044219970703
5th step loss: 9.972816467285156
```

</details>

![DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_True](./assets/images/DP_1_SHARD_1_TP_2_USE_LOSS_PARALLEL_False.png)


### 2d parallel in 8 gpus

```bash
export WORLD_SIZE=8 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23459
export DP=2 &&\
export SHARD=2 &&\
export TP=2
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_mesh.py
```

```python
rank: 0, world_rank: 0, world size: 8, device: cuda:0
rank: 6, world_rank: 6, world size: 8, device: cuda:6
rank: 5, world_rank: 5, world size: 8, device: cuda:5
rank: 4, world_rank: 4, world size: 8, device: cuda:4
rank: 2, world_rank: 2, world size: 8, device: cuda:2
rank: 3, world_rank: 3, world size: 8, device: cuda:3
rank: 7, world_rank: 7, world size: 8, device: cuda:7
rank: 1, world_rank: 1, world size: 8, device: cuda:1   

    mesh_3d: DeviceMesh('cuda', [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=('replicate', 'shard', 'tp'))
    fsdp_mesh: DeviceMesh('cuda', [[0, 2], [4, 6]], mesh_dim_names=('replicate', 'shard'))
    tp_mesh: DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',))
    replicate_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f047c3142f0>
    shard_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f047c3149b0>
    tp_group: <torch.distributed.distributed_c10d.ProcessGroup object at 0x7f047c314bb0>

    model: FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (model): ModuleDict(
      (wte): Embedding(50257, 1024)
      (h): ModuleList(
        (0-3): 4 x ResidualBlock(
          (ln1): LayerNorm()
          (attn): Attention(
            (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (ln2): LayerNorm()
          (mlp): MLP(
            (ffn1): Linear(in_features=1024, out_features=4096, bias=False)
            (act): GELU(approximate='none')
            (ffn2): Linear(in_features=4096, out_features=1024, bias=False)
          )
        )
      )
      (ln): LayerNorm()
    )
    (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
  )
)

0th step loss: DTensor(local_tensor=11.174638748168945, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
1th step loss: DTensor(local_tensor=0.5185606479644775, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
2th step loss: DTensor(local_tensor=16.62112808227539, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
3th step loss: DTensor(local_tensor=15.390058517456055, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
4th step loss: DTensor(local_tensor=11.48178482055664, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
5th step loss: DTensor(local_tensor=9.973380088806152, device_mesh=DeviceMesh('cuda', [0, 1], mesh_dim_names=('tp',)), placements=(Replicate(),))
```

# MoE

- dmoe
    - megablocks
        - [moe.py](https://github.com/databricks/megablocks/blob/main/megablocks/layers/moe.py)
        - [dmoe.py](https://github.com/databricks/megablocks/blob/main/megablocks/layers/lingua/dmoe.py#L18)
    - llm foundry
        - [layers/ffn.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/layers/ffn.py#L470-L509)
        - [test_dmoe.py](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/tests/models/layers/test_dmoe.py#L71)
        - [moe init](https://github.com/mosaicml/llm-foundry/blob/e6e74a24db234a74010f64f72cbd15bfa4ffda1c/llmfoundry/models/utils/param_init_fns.py#L341-L404)
    - olmo
        - [olmoe](https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/model.py#L680-L690)
        - [parallelism](https://github.com/allenai/OLMo/blob/sewon-olmoe/scripts/train.py#L188-L225)
        - [profiler](https://github.com/allenai/OLMo/blob/sewon-olmoe/olmo/train.py#L1225-L1262)

    - megatron integration
        - [megatron PR 1](https://github.com/NVIDIA/Megatron-LM/pull/287)
        - [megatron PR 2](https://github.com/NVIDIA/Megatron-LM/pull/288)
        - [stanford-futuredata/Megatron-LM](https://github.com/stanford-futuredata/Megatron-LM/tree/3a9e3d8de308e6f6398b59d16a8bd7177374f121)

- torch native trainer references
    - [pytorch/torchtune](https://github.com/pytorch/torchtune)
    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan)
        - [memory_profiler](https://github.com/pytorch/torchtitan/blob/main/docs/memory_profiler.md)

- megatron naive moe
    - [Megatron-LM/megatron/core/transformer/moe](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
    - [Megatron-LM/docs/llama_mistral.md](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama_mistral.md)
    - [Megatron-LM/examples/export/ptq_and_trtllm_export](https://github.com/NVIDIA/Megatron-LM/tree/772faca1f8d5030621b738cbd8e8bb2d8d28f6e6/examples/export/ptq_and_trtllm_export)
    - [mixtral inference example](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mixtral)


## install core dependencies

```bash
python -m pip install --upgrade pip

# megablocks
pip install megablocks==0.6.1

# TE
git clone https://github.com/NVIDIA/TransformerEngine
cd TransformerEngine
git checkout release_v1.7
git submodule update --init --recursive
export NVTE_FRAMEWORK=pytorch 
pip install .
cd ..
```

```bash
cd /path/to/dir/multiprocessing_pdb &&\
pip install -e .
```

## run scripts for dmoe test (WIP)

```bash
# cd /path/to/dir/lingua/exp_srcs
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23459 &&\
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_dmoe.py
```

```python

```


# shampoo

```bash
git clone https://github.com/facebookresearch/optimizers
cd optimizers && pip install -e . && cd ..
python -c "from distributed_shampoo.distributed_shampoo import DistributedShampoo;
shampoo_optimizer = DistributedShampoo;
print(shampoo_optimizer)"
```

```bash
# cd /path/to/dir/lingua/exp_srcs
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458
export DP=1 &&\
export SHARD=2 &&\
export TP=1
torchrun --nproc_per_node=$WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
test_shampoo.py
```

- warnings

```
[rank0]: ValueError: Invalid start_preconditioning_step value: 20. Must be >= precondition_frequency=100.
```

```
start_preconditioning_step set to -1. Setting start_preconditioning_step equal to precondition frequency 100 by default.
```

- actual logs

```python
rank: 1, world_rank: 1, world size: 2, device: cuda:1
rank: 0, world_rank: 0, world size: 2, device: cuda:0

    model: FullyShardedDataParallel(
  (_fsdp_wrapped_module): Transformer(
    (model): ModuleDict(
      (wte): Embedding(50257, 1024)
      (h): ModuleList(
        (0-3): 4 x ResidualBlock(
          (ln1): LayerNorm()
          (attn): Attention(
            (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): Linear(in_features=1024, out_features=1024, bias=False)
          )
          (ln2): LayerNorm()
          (mlp): MLP(
            (ffn1): Linear(in_features=1024, out_features=4096, bias=False)
            (act): GELU(approximate='none')
            (ffn2): Linear(in_features=4096, out_features=1024, bias=False)
          )
        )
      )
      (ln): LayerNorm()
    )
    (lm_head): Linear(in_features=1024, out_features=50257, bias=False)
  )
)

- if freq=100, start_preconditioning=100, iter=100

```python
start_preconditioning_step set to -1. Setting start_preconditioning_step equal to precondition frequency 100 by default.
(10th step) loss: 0.0012735219206660986
(20th step) loss: 0.00038887240225449204
(30th step) loss: 0.0002023066335823387
(40th step) loss: 0.00014369595737662166
(50th step) loss: 0.00011663915938697755
(60th step) loss: 0.00010105434193974361
(70th step) loss: 9.017760748974979e-05
(80th step) loss: 8.183371392078698e-05
(90th step) loss: 7.500954961869866e-05
(100th step) loss: 6.913893594173715e-05
```

- if freq=20, start_preconditioning=20, iter=100 

```python
(10th step) loss: 0.0012735219206660986
(20th step) loss: 0.00038887240225449204
(30th step) loss: 3.665658823592821e-06
(40th step) loss: 1.639122501728707e-06
(50th step) loss: 1.043079237206257e-06
(60th step) loss: 8.046615107559774e-07
(70th step) loss: 7.152547709665669e-07
(80th step) loss: 6.854525622657093e-07
(90th step) loss: 5.960458224762988e-07
(100th step) loss: 5.36441234544327e-07
```

## further things to followup

[muon](https://github.com/KellerJordan/modded-nanogpt)