# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from functools import partial
import math

import logging
from torch import nn
from torch.optim import AdamW, lr_scheduler

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
def get_optimizer(model, args, model_args):
    opt_cls = AdamW
    truly_decoupled_wd = args.truly_decoupled_wd

    if model_args.mup:
        no_decay_name_list = ["bias", "norm"]
        optimizer_grouped_parameters = []
        opt_kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon,
            'fused': True,
        }

        base_dim = model_args.base_dim or int(model_args.base_head_dim * model_args.base_n_heads)
        base_head_dim = model_args.base_head_dim or model_args.base_dim // model_args.base_n_heads
        base_ffn_hidden_dim = adjust_hidden_dim(
            4*base_dim, 
            model_args.base_ffn_dim_multiplier, 
            model_args.base_multiple_of
        )
        dim = model_args.dim or int(model_args.head_dim * model_args.n_heads)
        head_dim = model_args.head_dim or model_args.dim // model_args.n_heads
        ffn_hidden_dim = adjust_hidden_dim(
            4*dim, 
            model_args.ffn_dim_multiplier, 
            model_args.multiple_of
        )

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
        
        matrix_like_p = defaultdict(new_group) # key is width_mult
        vector_like_p = new_group()
        no_decay_vector_like_p = new_group_() # don't decay bias an layernorm

        for n, p in model.named_parameters():
            if p.requires_grad:
                ## per layer learning rate
                if ('tok_embeddings' in n) or ('output' in n):
                    vector_like_p['params'].append(p)
                elif any(ndnl in n for ndnl in no_decay_name_list):
                    print(n)
                    no_decay_vector_like_p['params'].append(p)
                else:
                    if 'wo' in n:
                        width_mult = head_dim/base_head_dim
                    elif 'w2' in n:
                        width_mult = ffn_hidden_dim/base_ffn_hidden_dim
                    else:
                        width_mult = head_dim/base_head_dim
                    matrix_like_p[width_mult]['params'].append(p)

        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group['lr'] /= width_mult

            if truly_decoupled_wd:
                assert (group['weight_decay'] < 0.001), f"weight_decay value ({weight_decay}) is too large. set this as 1e-4 ~ 1e-5"
                print(f"(matrix_like) using truly truly_decoupled_wd with lambda, {group['weight_decay']}")
                group['weight_decay'] /= group['lr']
                print(f"(matrix_like) after compensating lr, {group['weight_decay']}")

        if truly_decoupled_wd:
            vector_like_p['weight_decay'] /= vector_like_p['lr'] 
            print(f"(vector_like) after compensating lr, {vector_like_p['weight_decay']}")

        optimizer_grouped_parameters.extend(
            list(matrix_like_p.values()) 
            + [vector_like_p] 
            + [no_decay_vector_like_p]
        )
        return opt_cls(
            optimizer_grouped_parameters, 
            **opt_kwargs,
        )
    else:
        return opt_cls(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay if not truly_decoupled_wd else args.weight_decay/args.lr,
            eps=args.epsilon,
            fused=True,  # Faster optim.step but can throw errors
        )

def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int, model_args):
    logger.info("Starting build of optimizer...")
    # optimizer
    optimizer = get_optimizer(model, args, model_args)

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
