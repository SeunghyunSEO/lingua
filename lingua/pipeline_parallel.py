
################################################################################
################################################################################
################################################################################
'''
https://pytorch.org/docs/main/distributed.pipelining.html
https://pytorch.org/tutorials/intermediate/pipelining_tutorial.html

https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipeline_llama.py
https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/pipelining_utils.py

https://github.com/pytorch/torchtitan/blob/3247841423429faf37bdf6918204350db293e482/train.py#L153
https://github.com/pytorch/torchtitan/blob/3247841423429faf37bdf6918204350db293e482/train.py#L287

'''

import math
import copy
import logging
from typing import Callable, Union, Tuple, Iterable, Optional

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor

from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,

    ## https://pytorch.org/docs/stable/distributed.pipelining.html#module-torch.distributed.pipelining.schedules
    Schedule1F1B,
    ScheduleInterleaved1F1B,
)


logger = logging.getLogger()

DeviceType = Union[int, str, torch.device]


def pipeline_model(
    model,
    device_mesh,
    model_args,
    device,
    loss_fn,
):
    pp_mesh = device_mesh['pp']
    pipeline_parallel_microbatches = 1
    # pipeline_parallel_split_points = ["layers.16"]
    pipeline_parallel_split_points = None
    pipeline_parallel_schedule = "Interleaved1F1B" # https://github.com/pytorch/pytorch/blob/release/2.5/torch/distributed/pipelining/schedules.py#L2143
    pipeline_parallel_degree = device_mesh["pp"].size()

    stages, models = pipeline_model_manual_split(
        model, 
        pp_mesh,
        pipeline_parallel_microbatches,
        pipeline_parallel_split_points,
        pipeline_parallel_schedule, 
        device, 
        model_args,
    )

    pp_schedule = build_pipeline_schedule(
        pipeline_parallel_schedule, 
        pipeline_parallel_microbatches, 
        pipeline_parallel_degree, 
        stages, 
        loss_fn
    )

    return pp_schedule, models


def pipeline_model_manual_split(
    whole_model: nn.Module,
    pp_mesh,
    
    pipeline_parallel_microbatches,
    pipeline_parallel_split_points,
    pipeline_parallel_schedule,

    device,
    model_config,
):
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    microbatches = (
        pipeline_parallel_microbatches or pp_size
    )
    splits = (
        pipeline_parallel_split_points
        or generate_split_points(
            pipeline_parallel_schedule, 
            pp_size, 
            model_config
        )
    )

    def _build_stage(stage_idx, start_layer, stop_layer, is_first=False, is_last=False):
        model = copy.deepcopy(whole_model)
        if not is_first:
            model.tok_embeddings = None

        drop_layers = start_layer is not None
        for name in list(model.layers.keys()):
            # we keep layers in a contiguous region between start (inclusive) and stop (exclusive)
            if f"layers.{name}" == start_layer:
                drop_layers = False
            if f"layers.{name}" == stop_layer:
                drop_layers = True
            if drop_layers:
                del model.layers[name]

        if not is_last:
            model.norm = None
            model.output = None

        '''
        https://github.com/pytorch/pytorch/blob/release/2.5/torch/distributed/pipelining/stage.py#L1230
        https://pytorch.org/docs/stable/distributed.pipelining.html#torch.distributed.pipelining.stage.PipelineStage
        The PipelineStage requires an example argument input_args representing the runtime input to the stage, 
        which would be one microbatch worth of input data. 
        This argument is passed through the forward method of the stage module to determine the input and output shapes required for communication.

        When composing with other Data or Model parallelism techniques, output_args may also be required, 
        if the output shape/dtype of the model chunk will be affected.
        '''

        # with torch.device("meta"):
        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,

            input_args=None,
            # input_args=torch.randint(512, (1, 4096)),
            output_args=None, #TODO

            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []
    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop"):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}: model chunk \n{model_chunk}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models

def generate_split_points(
    pipeline_parallel_schedule, 
    pp_dim, 
    model_config
):
    schedule_class = get_schedule_class(pipeline_parallel_schedule)
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(
            f"Unsupported pipeline schedule: {pipeline_parallel_schedule}"
        )
    total_stages = pp_dim * num_stages_per_rank
    num_layers = model_config.n_layers
    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    logger.info(
        f"No 'pipeline_parallel_split_points' so the generated splits are: {splits} \
This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


def build_pipeline_schedule(
    pipeline_parallel_schedule, 
    pipeline_parallel_microbatches, 
    pipeline_parallel_degree, 
    stages, 
    loss_fn
):
    schedule_class = get_schedule_class(pipeline_parallel_schedule)
    if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
        raise ValueError(
            f"{schedule_class} is not supported as we do not support custom CSV schedules."
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    logger.info(
        f"Using pipeline schedule {pipeline_parallel_schedule}"
    )
    n_microbatches = pipeline_parallel_microbatches
    if n_microbatches is None:
        n_microbatches = pipeline_parallel_degree

    return schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, 
    pp_size: int, 
    num_stages: int, 
    style: str = "loop",
) -> Tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]
    
'''
https://github.com/pytorch/torchtitan/pull/649
https://github.com/pytorch/torchtitan/issues/596
'''
@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    print(f'grads: {grads}')
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    if pp_mesh is not None:
        if isinstance(total_norm, DTensor):
            # will reach here if PP + other parallelism is used. If only using PP, total_norm will be a local tensor

            # if total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`
            # we can simply reduce the DTensor to get the total norm in this tensor's process group
            # and then convert it to a local tensor
            total_norm = total_norm.full_tensor()

        # TODO: fck there is a bug that oneshot allreduce only support bf16?? and grads is 0 tensor wtf?
        # https://github.com/pytorch/pytorch/blob/b379a28a95aa0b0d167f08b8d70c8b85a248309d/torch/csrc/distributed/c10d/intra_node_comm.cu#L14
        print(f'total_norm: {total_norm}')
        # if total_norm.dtype == torch.float32:
        #     total_norm = total_norm.bfloat16()

        # TODO: cleanup maybe using DTensor
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm