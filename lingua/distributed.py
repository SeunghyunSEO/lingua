# Copyright (c) Meta Platforms, Inc. and affiliates.

import atexit
import contextlib
from itertools import chain
import logging
import multiprocessing as mp
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache, partial, reduce
from typing import List, Optional, Tuple, Union

import torch
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# for no recompute ops
import xformers.ops

from lingua.float8 import convert_linears_to_fp8

logger = logging.getLogger()

# for selective AC
default_no_recompute_ops = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.c10d_functional.reduce_scatter_tensor.default,
    torch.ops.xformers_flash.flash_fwd.default,
    torch.ops.xformers.efficient_attention_forward_cutlass.default,
}


@dataclass
class DistributedArgs:
    dp_shard: int = (
        1  # In how many shard to split the model weight. Typically number gpu in a node.
    )
    dp_replicate: int = (
        1  # How many times to replicate the model weight. Typically number of nodes.
    )
    tp_size: int = 1
    # loss_parallel: bool = True
    loss_parallel: bool = False
    enable_async_tp: bool = False

    ## TODO
    pp_size: int = 1

    selective_activation_checkpointing: bool = False
    compile: bool = False
    fsdp_type: str = "no_shard"
    model_dtype: str = "bf16"
    float8_recipe: Optional[str] = None
    float8_filter: str = r"layers\.[0-9]+\."

    matmul_allow_tf32: bool = False
    allow_bf16_reduced_precision_reduction = True
    detect_anomaly: bool = False

    compile_cache_size_limit: int = 8

    spawn_method: str = "forkserver"


@dataclass
class EnvironmentArgs:
    # Use GNU openMP (GOMP) instead of Intel OpenMP [Intel Math Kernel Library (MKL)]
    MKL_SERVICE_FORCE_INTEL: str = "GNU"
    OMP_NUM_THREADS: str = "1"
    MKL_NUM_THREADS: str = "1"
    # faster intra-node collectives, seems to be a cluster specific flag
    ENABLE_INTRA_NODE_COMM: str = "1"
    # avoids OOMs with long context
    TORCH_NCCL_AVOID_RECORD_STREAMS: str = "1"
    # increasing NCCL timeout time before having some NCCL error 22 should give a 16s timeout
    NCCL_IB_TIMEOUT: str = "22"
    NCCL_DEBUG: str = "INFO"
    TORCH_NCCL_ASYNC_ERROR_HANDLING: str = "1"


def get_device_mesh(distributed_args: DistributedArgs):
    tp_size = distributed_args.tp_size
    dp_replicate = distributed_args.dp_replicate
    dp_shard = distributed_args.dp_shard
    pp_size = distributed_args.pp_size

    assert (
        dp_replicate * dp_shard * tp_size == get_world_size()
    ), f"dp_replicate * dp_shard * tp_size ({dp_replicate} * {dp_shard} * {tp_size}) != world_size ({get_world_size()})"

    dims = []
    names = []
    if dp_replicate >= 1:
        dims.append(dp_replicate)
        names.append("dp_replicate")
    if dp_shard > 1 or distributed_args.fsdp_type == "no_shard":
        dims.append(dp_shard)
        names.append("dp_shard")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")

    ## TODO
    if pp_size > 1:
        dims.append(pp_size)
        names.append("pp")
        
    dims = tuple(dims)
    names = tuple(names)

    return init_device_mesh("cuda", mesh_shape=dims, mesh_dim_names=names)


def dist_max(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.MAX, group=mesh.get_group() if mesh else None)
    return tensor


def dist_mean(x: Union[int, float], mesh: DeviceMesh = None):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.AVG, group=mesh.get_group() if mesh else None)
    return tensor


def dist_mean_dict(x):
    r = dict()
    for k in x:
        r[k] = dist_mean(x[k])
        r[k] = r[k].item() if (r[k].dim() == 0) else r[k].tolist()
    return r


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int) -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache()
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    else:
        return "127.0.0.1"


def setup_env(env_args):
    env_vars = asdict(env_args)

    # When using Triton, it attempts to locate prebuilt kernels in a cache
    # located at ~/.triton/cache, but when that's backed by NFS this can fail
    # with a "OSError: [Errno 116] Stale file handle" error. If we were to set
    # it to a local directory it would belong to the first user who created it
    # and it would fail for the job of any other successive user assigned to
    # that machine. To avoid all this mess we use a temporary per-process cache.
    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    env_vars["TRITON_CACHE_DIR"] = triton_cache_dir

    # We change the tmp dir to /scratch in case it's slurm job
    # This avoids filling up the host's usually limited tmpfs
    # A full tmpfs leads to very slow creation of processes and weird bugs
    if get_is_slurm_job():
        new_tmp = f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}"
        if os.path.exists(new_tmp):
            env_vars["TMP_DIR"] = new_tmp

    for name, value in env_vars.items():
        if os.environ.get(name) != str(value):
            os.environ[name] = str(value)
            logger.warning(f"WARNING: Setting {name} to {value}")


def setup_torch_distributed(dist_args):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    mp.set_start_method(dist_args.spawn_method)
    with mp.Manager():
        pass

    local_rank = get_local_rank()

    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(
        get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
    )

    if get_is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_slurm_job():
        logger.info(f"Run launched with slurm, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    logger.info(f"ENV: {os.environ}")

    # set GPU device
    assert 0 <= local_rank < 8
    if dist_args.matmul_allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.warning(
            f"WARNING: Setting torch.backends.matmul.allow_tf32 to True. This is faster but less accurate."
        )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        dist_args.allow_bf16_reduced_precision_reduction
    )
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(init_method="env://", backend="nccl")
    torch.autograd.set_detect_anomaly(dist_args.detect_anomaly)


def get_module(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module(module, access_string, value):
    names = access_string.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], value)


def default_fsdp_grouping_plan(n_layers: int) -> List[Tuple[str, bool]]:
    '''
    >>> n_layers=4
    >>> [(f"layers.{i}", i < n_layers - 1) for i in range(n_layers)]
    [('layers.0', True), ('layers.1', True), ('layers.2', True), ('layers.3', False)]
    '''
    return [(f"layers.{i}", i < n_layers - 1) for i in range(n_layers)]


def get_default_policy(no_recompute_ops=None):
    no_recompute_ops = no_recompute_ops or default_no_recompute_ops

    def default_policy(ctx, func, *args, **kwargs):
        return (
            CheckpointPolicy.MUST_SAVE
            if func in no_recompute_ops
            else CheckpointPolicy.PREFER_RECOMPUTE
        )

    return default_policy


@torch.no_grad()
def check_model_value_range(
    model: torch.nn.Module, range: float = 1e3, std: float = 1e3
):
    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(param, DTensor):
            param = param.to_local()

        if torch.isnan(param).any() or torch.isinf(param).any():
            logger.warning(f"Model parameter {name} contains NaN or Inf")

        # try:
        #     param_range = param.max() - param.min()
        # except:
        #     print(f'name: {name}, param: {param}')
        param_range = param.max() - param.min()
        param_std = param.std()

        if param_range > range:
            logger.warning(
                f"Model parameter {name} has a suspiciously large range ({param_range}): please check initialization and init_weights is defined and called"
            )
        if param_std > std:
            logger.warning(
                f"Model parameter {name} has a suspiciously large standard deviation ({param_std}): please check initialization and init_weights is defined and called"
            )
        if (param == 0).all():
            logger.warning(
                f"Model parameter {name} is all zeros: it might be because of a missing initialization"
            )


def init_signal_handler(callable):
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR2, callable)
    logger.warning("Signal handler installed.")


def requeue_slurm_job():
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0 and os.environ.get("LAUNCH_WITH", "") != "DORA":
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(0)


@contextlib.contextmanager
def clean_env():
    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
        "DORA_FORCE_DISTRIB",
    )
    cluster_env = {
        x: os.environ.pop(x)
        for x in os.environ
        if x.startswith(
            ("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "WANDB_")
        )
        or x in distrib_names
    }
    try:
        yield
    finally:
        os.environ.update(cluster_env)


def parallelize_model(
    model,
    device_mesh,
    model_args,
    distributed_args: DistributedArgs,
    fsdp_grouping_plan: Optional[List[Tuple[str, bool]]] = None,
    tp_parallelize=None,
    no_recompute_ops=None,
    use_shampoo=False,
):
    if use_shampoo:
        ## idk why fsdp2 does not converge..?
        assert distributed_args.tp_size == 1, "currently shampoo does not support TP"
        
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # from torch.distributed.fsdp import ShardingStrategy
        # model = FSDP(
        #     model, 
        #     device_mesh=device_mesh, 
        #     use_orig_params=True,
        #     sharding_strategy=ShardingStrategy.HYBRID_SHARD
        # )

        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[distributed_args.model_dtype]
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
        mesh=(
            device_mesh["dp_replicate", "dp_shard"]
            if distributed_args.dp_shard > 1
            or distributed_args.fsdp_type == "no_shard"
            else device_mesh["dp_replicate"]
        )
        fsdp_config = {"mesh": mesh, "mp_policy": mp_policy}

        ## FSDP2 ver 1
        ## there was a parameter not flattend bug, because model.init_wegihts() is done before fsdp maybe
        if fsdp_grouping_plan is None:
            # Assume that the model has list of layers and group around it
            fsdp_grouping_plan = default_fsdp_grouping_plan(len(model.layers))

        for path, reshard_after_forward in fsdp_grouping_plan:
            module = get_module(model, path)
            set_module(
                model,
                path,
                fully_shard(
                    module, **fsdp_config, reshard_after_forward=reshard_after_forward
                ),
            )
        model = fully_shard(model, **fsdp_config, reshard_after_forward=True)

    else:
        ## FSDP 2, https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
        if distributed_args.tp_size > 1:
            assert (
                distributed_args.fsdp_type == "full_shard"
            ), "Only full shard is supported for TP parallelism"
            assert tp_parallelize is not None, "TP plan is required for TP parallelism"
            assert (
                distributed_args.compile == False
            ), "Compile is not supported for TP parallelism"
            
            tp_parallelize(model, device_mesh["tp"], model_args, distributed_args)

        if distributed_args.float8_recipe is not None:
            if distributed_args.tp_size > 1:
                raise RuntimeError("float8 is incompatible with tensor-parallelism for now")
            model = convert_linears_to_fp8(
                model, distributed_args.float8_recipe, distributed_args.float8_filter
            )

        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
            distributed_args.model_dtype
        ]
        if (
            distributed_args.fsdp_type == "full_shard"
            or distributed_args.fsdp_type == "no_shard"
        ):
            if distributed_args.fsdp_type == "no_shard":
                assert (
                    distributed_args.dp_shard == 1
                ), "dp_shard must be 1 for no_shard fsdp_type"
                assert (
                    device_mesh["dp_shard"].size() == 1
                ), "dp_shard must be 1 for no_shard fsdp_type"

            fsdp_config = dict(
                # https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
                mp_policy=(
                    MixedPrecisionPolicy(
                        param_dtype=param_dtype,
                        reduce_dtype=torch.float32,
                    )
                ),
                # https://github.com/facebookresearch/lingua/blob/31ab231b2d386c4c735e82e8a2c40f10ce13fbbc/apps/main/train.py#L171
                mesh=(
                    device_mesh["dp_replicate", "dp_shard"]
                    if distributed_args.dp_shard > 1
                    or distributed_args.fsdp_type == "no_shard"
                    else device_mesh["dp_replicate"]
                ),
            )

            if fsdp_grouping_plan is None:
                # Assume that the model has list of layers and group around it
                fsdp_grouping_plan = default_fsdp_grouping_plan(len(model.layers))

            for path, reshard_after_forward in fsdp_grouping_plan:
                module = get_module(model, path)
                set_module(
                    model,
                    path,
                    fully_shard(
                        module, **fsdp_config, reshard_after_forward=reshard_after_forward
                    ),
                )

            model = fully_shard(model, **fsdp_config, reshard_after_forward=True)
        else:
            raise ValueError(f"Invalid fsdp_type: {distributed_args.fsdp_type}")

    if distributed_args.selective_activation_checkpointing:
        model = checkpoint_wrapper(
            model,
            context_fn=partial(
                create_selective_checkpoint_contexts,
                get_default_policy(no_recompute_ops),
            ),
        )

    if distributed_args.compile:
        '''
        https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L68-L74
        https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/torchtitan/parallelisms/parallelize_llama.py#L299
        https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/torchtitan/models/llama/model.py#L371-L373
        https://github.com/pytorch/torchtitan/issues/61
        '''
        torch._dynamo.config.cache_size_limit = (
            distributed_args.compile_cache_size_limit
        )

        ## TODO
        # if model_args.fused_rms_norm == "fused_rmsnorm":
        #     raise NotImplementedError(
        #         "fused_rmsnorm is not compatible with torch.compile yet. "
        #         "Please use rmsnorm or layernorm."
        #     )

        ## og
        model = torch.compile(model)

        ## TODO
        # for layer_id, layer in model.layers.named_children():
        #     # layer = torch.compile(layer, fullgraph=True)
        #     layer = torch.compile(layer)
        #     model.layers.register_module(layer_id, layer)
        # logger.info("Compiling each TransformerBlock with torch.compile")

    return model


# ################################################################################
# ################################################################################
# ################################################################################
# '''
# https://pytorch.org/docs/main/distributed.pipelining.html
# https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html

# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipelining_utils.py

# https://github.com/pytorch/torchtitan/blob/3247841423429faf37bdf6918204350db293e482/train.py#L153
# https://github.com/pytorch/torchtitan/blob/3247841423429faf37bdf6918204350db293e482/train.py#L287

# '''

# import copy
# from typing import Callable, Union, Tuple

# import torch
# import torch.nn as nn
# from torch.distributed import DeviceMesh
# from torch.distributed.pipelining import PipelineStage
# from torch.distributed.pipelining.schedules import (
#     get_schedule_class,
#     PipelineScheduleMulti,
#     PipelineScheduleSingle,
# )


# DeviceType = Union[int, str, torch.device]


# def pipeline_model(
#     model,
#     device_mesh,
#     model_args,
#     distributed_args: DistributedArgs,
#     device,
#     loss_fn,
# ):
#     pp_mesh = device_mesh['pp']
#     pipeline_parallel_microbatches = None
#     pipeline_parallel_split_points = None
#     pipeline_parallel_schedule = None
#     pipeline_parallel_degree = None

#     stages, models = pipeline_model_manual_split(
#         model, 
#         pp_mesh, 
#         parallel_dims, 
#         pipeline_parallel_microbatches,
#         pipeline_parallel_split_points,
#         pipeline_parallel_schedule, 
#         device, 
#         model_args,
#     )

#     pp_schedule = build_pipeline_schedule(
#         pipeline_parallel_schedule, 
#         pipeline_parallel_microbatches, 
#         pipeline_parallel_degree, 
#         stages, 
#         loss_fn
#     )

#     return pp_schedule, models


# def pipeline_model_manual_split(
#     whole_model: nn.Module,
#     pp_mesh,
#     parallel_dims,
    
#     pipeline_parallel_microbatches,
#     pipeline_parallel_split_points,
#     pipeline_parallel_schedule,

#     device: DeviceType,
#     model_config: ModelArgs,
# ):
#     """
#     This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

#     It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

#     The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
#     parallelism.
#     """
#     pp_rank = pp_mesh.get_local_rank()
#     pp_size = pp_mesh.size()
#     microbatches = (
#         pipeline_parallel_microbatches or parallel_dims.pp
#     )
#     splits = (
#         pipeline_parallel_split_points
#         or generate_split_points(
#             pipeline_parallel_schedule, 
#             parallel_dims.pp, 
#             model_config
#         )
#     )

#     def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
#         model = copy.deepcopy(whole_model)
#         if not is_first:
#             model.tok_embeddings = None

#         drop_layers = start_layer is not None
#         for name in list(model.layers.keys()):
#             # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
#             if f"layers.{name}" == start_layer:
#                 drop_layers = False
#             if f"layers.{name}" == stop_layer:
#                 drop_layers = True
#             if drop_layers:
#                 del model.layers[name]

#         if not is_last:
#             model.norm = None
#             model.output = None

#         stage = PipelineStage(
#             model,
#             stage_idx,
#             num_stages,
#             device,
#             group=pp_mesh.get_group("pp"),
#         )
#         return stage, model

#     num_stages = len(splits) + 1
#     stage_idx = pp_rank

#     stages = []
#     models = []
#     for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
#         start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
#         stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
#         stage, model_chunk = _build_stage(
#             stage_idx,
#             start_layer,
#             stop_layer,
#             is_first=stage_idx == 0,
#             is_last=stage_idx == num_stages - 1,
#         )
#         logger.info(
#             f"PP rank {pp_rank} is building stage_idx {stage_idx}"
#             f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
#         )
#         stages.append(stage)
#         models.append(model_chunk)
#     return stages, models

# def generate_split_points(pipeline_parallel_schedule, pp_dim, model_config):
#     schedule_class = get_schedule_class(
#         pipeline_parallel_schedule
#     )
#     if issubclass(schedule_class, PipelineScheduleSingle):
#         num_stages_per_rank = 1
#     elif issubclass(schedule_class, PipelineScheduleMulti):
#         # Multi-stage schedules support more than 2 stages per rank, but this is the default if
#         # no pipeline split is specified
#         num_stages_per_rank = 2
#     else:
#         raise ValueError(
#             f"Unsupported pipeline schedule: {pipeline_parallel_schedule}"
#         )
#     total_stages = pp_dim * num_stages_per_rank
#     num_layers = model_config.n_layers
#     if total_stages > num_layers:
#         raise ValueError("Total stages cannot be greater than the number of layers")

#     base_interval = num_layers // total_stages
#     extra_layers = num_layers % total_stages

#     splits = []
#     current_layer = 0
#     for i in range(total_stages - 1):
#         if i == 0:
#             current_layer += base_interval
#         else:
#             # Middle stages get an extra layer if there are any remaining
#             if extra_layers > 0:
#                 current_layer += base_interval + 1
#                 extra_layers -= 1
#             else:
#                 current_layer += base_interval
#         splits.append("layers." + str(current_layer))
#     logger.info(
#         f"No 'pipeline_parallel_split_points' so the generated splits are: {splits} \
# This may be sub-optimal as the number of layers per stage may be unbalanced."
#     )
#     return splits


# def build_pipeline_schedule(
#     pipeline_parallel_schedule, 
#     pipeline_parallel_microbatches, 
#     pipeline_parallel_degree, 
#     stages, 
#     loss_fn
# ):
#     schedule_class = get_schedule_class(
#         pipeline_parallel_schedule
#     )
#     if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
#         raise ValueError(
#             f"{schedule_class} is not supported as we do not support custom CSV schedules."
#         )

#     looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
#     logger.info(
#         f"Using pipeline schedule {pipeline_parallel_schedule}"
#     )
#     n_microbatches = pipeline_parallel_microbatches
#     if n_microbatches is None:
#         n_microbatches = pipeline_parallel_degree

#     return schedule_class(
#         stages if looped_schedule else stages[0],
#         n_microbatches=n_microbatches,
#         loss_fn=loss_fn,
#     )


# # TODO(whc) should this be a utility inside torch.pipelining?
# def stage_ids_this_rank(
#     pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
# ) -> Tuple[int]:
#     """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
#     assert (
#         num_stages % pp_size == 0
#     ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
#     stages_per_rank = num_stages // pp_size
#     if style == "loop":
#         return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
#     elif style == "v":
#         assert (
#             stages_per_rank == 2
#         ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
#         stage_v_pairs = list(
#             zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
#         )
#         return stage_v_pairs[pp_rank]