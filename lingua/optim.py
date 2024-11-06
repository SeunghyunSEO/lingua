# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from functools import partial
import math

import logging
import torch
from torch import nn
from torch.optim import lr_scheduler

logger = logging.getLogger()


@dataclass
class OptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0

    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    annealing_step: int = 1000

    exp_factor: float = 0.5

    truly_decoupled_wd: bool = False

    opt_cls_name: str = "adamw"


def lr_linear(step: int, warmup: int, n_steps: int, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = s * min_ratio + (1 - s)
    else:
        lr = min_ratio
    return lr


def lr_inv_sqrt(step: int, warmup: int, exp_factor: float, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    else:
        lr = max((warmup**exp_factor) / (step**exp_factor), min_ratio)
    return lr


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    cycle_length: float,
    theta: float,
    min_ratio: float,
) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            math.cos(math.pi * s**theta / cycle_length) + 1
        )
    else:
        lr = min_ratio
    return lr


def build_lr_fn(args: OptimArgs, n_steps: int):
    if args.scheduler == "constant":
        lr_fn = lambda x: 1.0
    elif args.scheduler == "linear":
        lr_fn = partial(
            lr_linear, warmup=args.warmup, n_steps=n_steps, min_ratio=args.lr_min_ratio
        )
    elif args.scheduler == "inv_sqrt":
        lr_fn = partial(
            lr_inv_sqrt,
            warmup=args.warmup,
            exp_factor=args.exp_factor,
            min_ratio=args.lr_min_ratio,
        )
    elif args.scheduler == "cosine":
        lr_fn = partial(
            lr_cosine,
            warmup=args.warmup,
            n_steps=n_steps,
            cycle_length=args.cycle_length,
            theta=args.cosine_theta,
            min_ratio=args.lr_min_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}")
    return lr_fn

def adjust_hidden_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

from collections import defaultdict
def get_optimizer(model, args, model_args, dist_args, device_mesh):

    if args.opt_cls_name.lower() == 'adamw':
        from torch.optim import AdamW
        opt_cls = AdamW
    elif args.opt_cls_name.lower() == 'shampoo':
        from distributed_shampoo.distributed_shampoo import DistributedShampoo
        from distributed_shampoo.shampoo_types import AdamGraftingConfig, FullyShardShampooConfig, HSDPShampooConfig
        from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
        opt_cls = DistributedShampoo
    else:
        raise NotImplementedError
    
    truly_decoupled_wd = args.truly_decoupled_wd
    if truly_decoupled_wd:
        assert (args.weight_decay < 0.001), f"weight_decay value ({args.weight_decay}) is too large. set this as 1e-4 ~ 1e-5"

    def new_group():
        new_g = {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
        new_g['params'] = []
        return new_g
    def new_group_():
        new_g = {
            'lr': args.lr,
            'weight_decay': 0.,
        }
        new_g['params'] = []
        return new_g

    no_decay_name_list = ["bias", "norm", "scaler"]
    optimizer_grouped_parameters = []

    if args.opt_cls_name.lower() == 'adamw':
        opt_kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'fused': True,
        }
    elif args.opt_cls_name.lower() == 'shampoo':
        opt_kwargs = {
            'betas': (args.beta1, args.beta2),
            'epsilon': args.epsilon,
            'grafting_config': AdamGraftingConfig(
                beta2=args.beta2,
                epsilon=args.epsilon,
            ),
            'use_decoupled_weight_decay': truly_decoupled_wd,

            'max_preconditioner_dim': 1024,
            'precondition_frequency': 1,
            'start_preconditioning_step': -1,
        }

        # ## for hsdp
        # opt_kwargs['distributed_config'] = HSDPShampooConfig(
        #     param_to_metadata=compile_fsdp_parameter_metadata(model),
        #     device_mesh=device_mesh,
        # )

        ## for fully_shard (fsdp2)
        opt_kwargs['distributed_config'] = FullyShardShampooConfig()

    else:
        raise NotImplementedError

    if model_args.mup:

        ## proxy model's configs
        base_dim = model_args.base_dim or int(model_args.base_head_dim * model_args.base_n_heads)
        base_head_dim = model_args.base_head_dim or int(model_args.base_dim // model_args.base_n_heads)
        base_ffn_hidden_dim = adjust_hidden_dim(
            4*base_dim, 
            model_args.base_ffn_dim_multiplier, 
            model_args.base_multiple_of,
        )

        ## target model's configs
        dim = model_args.dim or int(model_args.head_dim * model_args.n_heads)
        head_dim = model_args.head_dim or int(model_args.dim // model_args.n_heads)
        ffn_hidden_dim = adjust_hidden_dim(
            4*dim, 
            model_args.ffn_dim_multiplier, 
            model_args.multiple_of
        )
        
        matrix_like_p = defaultdict(new_group) # key is width_mult
        vector_like_p = new_group()
        no_decay_group = new_group_() # don't decay bias an layernorm

        for n, p in model.named_parameters():
            if p.requires_grad:
                ## per layer learning rate
                if ('tok_embeddings' in n) or ('output' in n):
                    vector_like_p['params'].append(p)
                elif any(ndnl in n for ndnl in no_decay_name_list):
                    # print(f'{n} is in {no_decay_name_list}, so wd is set as 0')
                    no_decay_group['params'].append(p)
                else:
                    if 'wo' in n:
                        width_mult = dim/base_dim
                    elif 'w2' in n:
                        width_mult = ffn_hidden_dim/base_ffn_hidden_dim
                    else:
                        width_mult = dim/base_dim
                    matrix_like_p[width_mult]['params'].append(p)

        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] /= width_mult

            if truly_decoupled_wd:
                print(f"(matrix_like) using truly truly_decoupled_wd with lambda, {group['weight_decay']}")
                group['weight_decay'] /= group['lr']
                print(f"(matrix_like) after compensating lr, {group['weight_decay']}")

        if truly_decoupled_wd:
            vector_like_p['weight_decay'] /= vector_like_p['lr'] 
            print(f"(vector_like) after compensating lr, {vector_like_p['weight_decay']}")

        optimizer_grouped_parameters.extend(
            list(matrix_like_p.values()) 
            + [vector_like_p] 
            + [no_decay_group]
        )
    else:

        default_group = new_group()
        no_decay_group = new_group_() # don't decay bias an layernorm

        for n, p in model.named_parameters():
            if p.requires_grad:
                if any(ndnl in n for ndnl in no_decay_name_list):
                    # print(f'{n} is in {no_decay_name_list}, so wd is set as 0')
                    no_decay_group['params'].append(p)
                else:
                    default_group['params'].append(p)

        if truly_decoupled_wd:
            default_group['weight_decay'] /= default_group['lr'] 

        optimizer_grouped_parameters.extend(
            [default_group] 
            + [no_decay_group]
        )

    return opt_cls(
        optimizer_grouped_parameters, 
        **opt_kwargs,
    )
    

def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int, model_args, dist_args, device_mesh):
    logger.info("Starting build of optimizer...")
    # optimizer
    optimizer = get_optimizer(model, args, model_args, dist_args, device_mesh)

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
