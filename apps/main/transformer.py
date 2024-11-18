# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from xformers.ops import fmha, AttentionBias
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    cross_entropy,
    Scaler,
    l2_norm,
)
from fused_kernels.fused_rms_norm import FusedRMSNorm
from fused_kernels.fused_ce import fused_cross_entropy

import logging
logger = logging.getLogger()


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False

    sliding_window: Optional[int] = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window

        assert args.vocab_size > 0
        d_model = args.dim or int(args.n_heads * args.head_dim)
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, d_model)

        self.output = nn.Linear(
            d_model,
            args.vocab_size,
            bias=False,
        )

        if args.weight_tying:
            self.output.weight = self.embeddings.tok_embeddings.weight

        if args.ngpt:
            self.logit_scaler = Scaler(
                args.vocab_size, 
                scale=d_model**-0.5,
                scale_init=1.0,
            )
        else:
            norm = FusedRMSNorm if self.args.fused_rms_norm else RMSNorm
            self.norm = norm(d_model, eps=args.norm_eps)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):

        if self.tok_embeddings:
            embedding_scale = self.args.input_mult if self.args.mup else 1.0
            h = embedding_scale * self.tok_embeddings(token_values)
        else:
            # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
            h = token_values
        _, seqlen, _ = h.shape
        

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        if self.args.mup:
            dim = self.args.dim or int(self.args.head_dim * self.args.n_heads)
            base_dim = self.args.base_dim or int(self.args.base_head_dim * self.args.base_n_heads)
            logit_scale = self.args.output_mult*(dim/base_dim)**-1.0
        else:
            logit_scale = 1.0

        if self.args.fused_ce:
            # raise NotImplementedError("currently not supported because of Dtensor")
            assert target is not None, "fused ce need target because it does not materialize logit to save VRAM memory"
            assert not self.args.ngpt, "fused ce does not materialize logit" 
            assert h.size(1) == target.size(1)
            h = self.norm(h).float()
            h = logit_scale * h
            h = h.contiguous().view(-1, h.size(-1))
            target = target.contiguous().view(-1)
            return fused_cross_entropy(h, self.output.weight, target)

        else:
            if self.norm is not None:
                if not self.args.ngpt:
                    h = self.norm(h)
            if self.output is not None:
                logits = self.output(logit_scale * h).float()
                if self.args.ngpt:
                    logits = self.logit_scaler(logits)
            else:
                return h
            if target is not None:
                return cross_entropy(logits, target)
            else:
                return logits

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters() # rope init
        if self.args.ngpt:
            self.logit_scaler.reset_parameters()
        else:
            if self.norm is not None:
                self.norm.reset_parameters()

        if self.args.mup:
            assert init_std is not None
            assert not self.weight_tying
            if self.tok_embeddings is not None:
                nn.init.trunc_normal_(
                    self.tok_embeddings.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
            if self.output is not None:
                nn.init.trunc_normal_(
                    self.output.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
                if self.args.mup and self.args.readout_zero_init:
                    nn.init.zeros_(self.output.weight)
                    # self.output.weight.zero_()
        else:
            init_std = init_std or (self.dim ** (-0.5))
            if self.tok_embeddings is not None:
                nn.init.trunc_normal_(
                    self.tok_embeddings.weight,
                    mean=0.0,
                    std=init_std,
                    a=-3 * init_std,
                    b=3 * init_std,
                )
            if self.output is not None:
                if not self.weight_tying:
                    nn.init.trunc_normal_(
                        self.output.weight,
                        mean=0.0,
                        std=init_std,
                        a=-3 * init_std,
                        b=3 * init_std,
                    )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    if model_args.fused_ce:
        return group_plan

    group_plan.append(("output", True))
    
    return group_plan


'''
it should be same as torchtitan
https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
'''
# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args, enable_float8=False):
    d_model = model_args.dim or int(model_args.n_heads * model_args.head_dim)
    assert d_model % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if distributed_args.loss_parallel else Replicate(),
                use_local_output=not distributed_args.loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears
    if enable_float8:
        # TODO(vkuzo): once float8 configuration supports delayed scaling,
        # add a check here to enforce supported float8 all-gather configurations
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437

    # for layer in model.layers:
    for _, layer in model.layers.items():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            # "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }
        
        ## TODO: should be splitted across time dimension like other norms ??? but we should modify o_proj, ffn2 so much
        # if model_args.qk_norm:
        #     layer_plan["attention.q_norm"]: SequenceParallel()
        #     layer_plan["attention.k_norm"]: SequenceParallel()
        # if model_args.residual_post_norm:
        #     layer_plan["attention.o_norm"]: SequenceParallel()
        #     layer_plan["feed_forward.fc2_norm"]: SequenceParallel()

        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size

    if distributed_args.enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if distributed_args.enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )