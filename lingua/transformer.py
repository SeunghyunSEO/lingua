# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from lingua import probe

flex_attention_comp = torch.compile(flex_attention)

from fused_kernels.fused_rms_norm import FusedRMSNorm


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    # dim: int = 512
    # n_layers: int = 8
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256

    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

    ########################################
    ## https://arxiv.org/abs/2203.03466
    mup: bool = False

    base_dim: Optional[int] = None
    base_head_dim: Optional[int] = None
    base_n_heads: Optional[int] = None
    base_n_kv_heads: Optional[int] = None
    base_ffn_dim_multiplier: Optional[float] = None
    base_multiple_of: Optional[int] = None

    input_mult: Optional[float] = None
    output_mult: Optional[float] = None

    readout_zero_init: Optional[bool] = None
    query_zero_init: Optional[bool] = None

    ########################################
    ## extra fused kernel (except sdpa)
    fused_rms_norm: bool = False
    fused_ce: bool = False

    ########################################
    qk_norm: bool = False
    residual_post_norm: bool = False
    tp_size: int = 1

    ########################################
    ngpt: bool = False

    ########################################
    use_moe: bool = False

########################################
'''
reproducing normalized GPT (nGPT)
https://arxiv.org/abs/2410.01131

https://github.com/lucidrains/nGPT-pytorch/blob/main/nGPT_pytorch/nTransformer.py
https://github.com/alxndrTL/modded-nanogpt
'''

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2_norm(
    x, 
    dim = -1, 
    norm_eps = 0.0, 
    eps = None
):
    if norm_eps == 0.0:
        x = F.normalize(x, dim = dim, p = 2)
    else:
        eps = default(eps, 1e-5 if t.dtype == torch.float16 else 1e-10)
        norm = x.norm(dim = dim, keepdim = True)
        target_norm = norm.detach().clamp(min = 1.0 - norm_eps, max = 1.0 + norm_eps)
        divisor = norm / target_norm
        x = x / divisor.clamp(min = eps)
    return x

class Scaler(nn.Module):
    def __init__(
        self, 
        dim: int, 
        scale: float, 
        scale_init: float = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.scale = scale
        self.scale_init = scale_init

    def forward(self, x: torch.Tensor, additional_scaler: float = 1.0):
        x = (
            x.float() \
            * self.weight.float() \
            * self.scale \
            * additional_scaler
        ).type_as(x)
        return x

    def reset_parameters(self):
        # because model is wrapped with fsdp. directly modify param is not possible?
        if self.scale_init:
            torch.nn.init.normal_(self.weight, mean=self.scale_init, std=0.0)
        else:
            torch.nn.init.ones_(self.weight)
        
########################################

def cross_entropy(pred, target, **kwargs):
    '''
    there is a bug to use loss parallel
    https://github.com/pytorch/torchtitan/blob/1060feacc1b51cb6b339a04e53a5243b8466552b/train.py#L133
    '''
    # return F.nll_loss(
    #     F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
    #     target.flatten(end_dim=-1),
    #     **kwargs,
    # )
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), 
        target.flatten(0, 1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class Attention(nn.Module):
    def __init__(
        self,
        args,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()
        self.args = args

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

        self.d_model = int(n_heads * head_dim) 
        self.kv_d_model = int(n_kv_heads * head_dim)
        self.tp_size = self.args.tp_size ## for qknorm, oh but nheads are already divide in parallelize_llama module! 
        
        if self.args.qk_norm:
            assert not self.args.ngpt
            # should apply headwise like 
            # https://github.com/huggingface/transformers/blob/f745e7d3f902601686b83c7cce2660c2a94509f0/src/transformers/models/cohere/modeling_cohere.py#L117-L131
            # https://github.com/NVIDIA/Megatron-LM/blob/2e2bdf62382f3a77f5bd020cde51ed833ee62ead/megatron/core/transformer/attention.py#L634-L641
            # TODO: for TP, expected shape of query and key tensor is [B, T, n_heads//tp_size, head_dim]

            norm = FusedRMSNorm if args.fused_rms_norm else RMSNorm
            self.q_norm = norm(self.n_heads//self.tp_size * self.head_dim, eps=args.norm_eps)
            self.k_norm = norm(self.n_kv_heads//self.tp_size * self.head_dim, eps=args.norm_eps)

            ## TODO: there is fsdp bug it returns 0, 128 size tensor in distributed.py's sanity check routine
            # self.q_norm = norm((self.n_heads//self.tp_size, self.head_dim), eps=args.norm_eps)
            # self.k_norm = norm((self.n_kv_heads//self.tp_size, self.head_dim), eps=args.norm_eps)

        if self.args.residual_post_norm:
            assert not self.args.ngpt
            norm = FusedRMSNorm if args.fused_rms_norm else RMSNorm
            self.o_norm = norm(self.d_model, eps=args.norm_eps)

        if self.args.ngpt:
            # IIUC, qk scaling should be done by headwise (per head)
            # but because it's not normalization operation, we can apply qk_scaler to hidden dim (3d input tensor)
            # oops, because original paper didnt use GQA, they dont need to separate qk_scaler but we should if we want to apply per head

            self.q_scaler = Scaler(dim=(self.n_heads//self.tp_size * self.head_dim), scale=self.d_model**-0.5)
            self.k_scaler = Scaler(dim=(self.n_kv_heads//self.tp_size * self.head_dim), scale=self.d_model**-0.5)

            ## TODO: there is fsdp bug. it returns 0, 128 size tensor in distributed.py's sanity check routine
            # self.q_scaler = Scaler(dim=(self.n_heads//self.tp_size, self.head_dim), scale=self.d_model**-0.5)
            # self.k_scaler = Scaler(dim=(self.n_kv_heads//self.tp_size, self.head_dim), scale=self.d_model**-0.5)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # ## for TP sanity check
        # print(f'''
        # self.tp_size: {self.tp_size}
        # self.q_norm.weight.size(): {self.q_norm.weight.size()}
        # self.k_norm.weight.size(): {self.k_norm.weight.size()}
        # xq.size(): {xq.size()}
        # xk.size(): {xk.size()}
        # self.n_heads: {self.n_heads}
        # self.n_kv_heads: {self.n_kv_heads}
        # ''')
        # '''
        # self.tp_size: 2                                                                                                                                
        # self.q_norm.weight.size(): torch.Size([256])                                                                                                   
        # self.k_norm.weight.size(): torch.Size([128])                                                                                                   
        # xq.size(): torch.Size([4, 4096, 2, 128])                                                                                                       
        # xk.size(): torch.Size([4, 4096, 1, 128])                                                                                                       
        # self.n_heads: 2                                                                                                                                
        # self.n_kv_heads: 1   
        # '''

        ## headwise norm
        if self.args.qk_norm:
            # xq = self.q_norm(xq)
            # xk = self.k_norm(xk)

            ## TODO: currently we should do reshape and apply norm then reshape again because of fsdp bug
            ## TODO: qk_norm should be applied for 4D input 
            xq = self.q_norm(
                xq.contiguous().view(bsz, seq_len, self.n_heads * self.head_dim) ## n_heads already discounted by tp_size in parallelize module
            ).contiguous().view(bsz, seq_len, self.n_heads, self.head_dim)
            xk = self.k_norm(
                xk.contiguous().view(bsz, seq_len, self.n_kv_heads * self.head_dim) ## n_kv_heads already discounted by tp_size in parallelize module
            ).contiguous().view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        ## headwise norm
        if self.args.ngpt:
            # i think we should apply scaler after RoPE ?
            # but RoPE is just rotate input... idk it's important or not

            # xq = self.q_scaler(l2_norm(xq))  # ngpt paper (15)
            # xk = self.k_scaler(l2_norm(xk))  # ngpt paper (16)

            ## TODO: currently we should do reshape and apply norm then reshape again because of fsdp bug
            xq = self.q_scaler(
                l2_norm(xq).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim) ## n_heads already discounted by tp_size in parallelize module
            ).contiguous().view(bsz, seq_len, self.n_heads, self.head_dim)  # ngpt paper (15)
            xk = self.k_scaler(
                l2_norm(xk).contiguous().view(bsz, seq_len, self.n_kv_heads * self.head_dim) ## n_kv_heads already discounted by tp_size in parallelize module
            ).contiguous().view(bsz, seq_len, self.n_kv_heads, self.head_dim)  # ngpt paper (16)

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        sdpa_scale = float(self.head_dim) ** -0.5
        if (self.args.mup) or (self.args.ngpt):
            assert attn_impl == "sdpa"
            if self.args.mup:
                sdpa_scale = float(self.head_dim) ** -1.0
            elif self.args.ngpt:
                sdpa_scale = float(self.head_dim) ** 0.5

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
                scale=sdpa_scale
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = self.wo(output.reshape(output_shape))
        if self.args.residual_post_norm:
            output = self.o_norm(output)

        return output

    def reset_parameters(self, init_std=None, factor=1.0):

        if self.args.mup:
            assert init_std is not None
            assert InitStdFactor(self.args.init_std_factor) in [InitStdFactor.GLOBAL_DEPTH, InitStdFactor.CURRENT_DEPTH]

            base_dim = self.args.base_dim or int(self.args.base_head_dim * self.args.base_n_heads)
            base_head_dim = self.args.base_head_dim or int(self.args.base_dim // self.args.base_n_heads)

            # in_proj_scale = float(self.head_dim/base_head_dim) ** -0.5
            # out_proj_scale = float(self.head_dim/base_head_dim) ** -0.5
            in_proj_scale = float(self.dim/base_dim) ** -0.5
            out_proj_scale = float(self.dim/base_dim) ** -0.5

            out_proj_scale /= factor
        else:
            in_proj_scale = 1.0
            out_proj_scale = 1.0

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std *in_proj_scale,
                a=-3 * init_std *in_proj_scale,
                b=3 * init_std *in_proj_scale,
            )
        if self.args.query_zero_init:
            nn.init.zeros_(self.wq.weight)
            # self.wq.weight.data.zero_()
        
        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std *out_proj_scale,
            a=-3 * init_std *out_proj_scale,
            b=3 * init_std *out_proj_scale,
        )

        if self.args.qk_norm:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()

        if self.args.residual_post_norm:
            self.o_norm.reset_parameters()

        if self.args.ngpt:
            self.q_scaler.reset_parameters()
            self.k_scaler.reset_parameters()

def adjust_hidden_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    '''
    >>> adjust_hidden_dim(4096*4, 1.3, 256)
    14336
    >>> adjust_hidden_dim(512*4, 1.3, 256)
    1792
    >>> adjust_hidden_dim(128*4, 1.3, 256)
    512
    >>> adjust_hidden_dim(128*4, 1.3, 64)
    448
    (4096/128) == (14336/448)
    '''
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

def get_moe_args(
    args, 
    hidden_size, 
    moe_top_k,
    moe_normalize_expert_weights,
    device,
):
    from functools import partial
    common_args = {
        'hidden_size': hidden_size,
        'ffn_hidden_size': hidden_size,
        'moe_top_k': moe_top_k,
        'activation_fn': partial(F.gelu, approximate='none'),
        'moe_jitter_eps': 0.0,  # Disable randomiztion
        'moe_normalize_expert_weights': moe_normalize_expert_weights,
        'uniform_expert_assignment': False,
        'bias': False,
        'device': device,
        'moe_num_experts': 32,
        'mlp_type': 'glu',
    }
    return common_args

class FeedForward(nn.Module):
    def __init__(
        self,
        args,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()
        self.args = args

        self.dim = dim
        hidden_dim = adjust_hidden_dim(
            hidden_dim, 
            ffn_dim_multiplier, 
            multiple_of
        )
        assert hidden_dim % mp_size == 0
        self.hidden_dim = hidden_dim

        if self.args.mup:
            self.base_dim = self.args.base_dim or int(self.args.base_head_dim * self.args.base_n_heads)
            base_hidden_dim = adjust_hidden_dim(
                4 * self.base_dim, 
                self.args.base_ffn_dim_multiplier, 
                self.args.base_multiple_of
            )
            assert base_hidden_dim % mp_size == 0
            self.base_hidden_dim = base_hidden_dim

        if self.args.use_moe:
            raise NotImplementedError
            try:
                from megablocks.layers.dmoe import dMoE
                from megablocks.layers.moe import MoE
            except ImportError:
                raise ImportError("pip install megablocks==0.6.1")
            self.moe_args = None
            self.moe = dMoE(self.moe_args) if self.config.moe_dropless else MoE(self.moe_args)
        else:
            self.w1 = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )
            self.w3 = nn.Linear(
                dim,
                hidden_dim,
                bias=False,
            )
            self.w2 = nn.Linear(
                hidden_dim,
                dim,
                bias=False,
            )

        self.d_model = args.dim or int(args.n_heads * args.head_dim)

        if self.args.residual_post_norm:
            assert not self.args.ngpt
            norm = FusedRMSNorm if args.fused_rms_norm else RMSNorm
            self.fc2_norm = norm(self.d_model, eps=args.norm_eps)

        if self.args.ngpt:
            self.u_scaler = Scaler(hidden_dim, scale=1.0)
            self.v_scaler = Scaler(hidden_dim, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.args.moe:
            output = self.moe(x)
            
        else:
            # B S D
            x1 = self.w1(x.view_as(x))
            x3 = self.w3(x.view_as(x))


            if self.args.ngpt:
                x3 = self.u_scaler(x3) # ngpt paper (20)
                x1 = self.v_scaler(x1, additional_scaler = self.d_model ** 0.5) # ngpt paper (21)

            output = self.w2(F.silu(x1) * x3)
            if self.args.residual_post_norm:
                output = self.fc2_norm(output)

        return output

    def reset_parameters(self, init_std=None, factor=1.0):

        if self.args.mup:
            assert init_std is not None
            assert InitStdFactor(self.args.init_std_factor) in [InitStdFactor.GLOBAL_DEPTH, InitStdFactor.CURRENT_DEPTH]
            in_init_std = init_std
            out_init_std = init_std * (self.base_hidden_dim/self.base_dim) ** -0.5 # e.g. intermediate dim is 4 times larger

            in_proj_scale = float(self.dim/self.base_dim) ** -0.5
            out_proj_scale = float(self.hidden_dim/self.base_hidden_dim) ** -0.5
            out_proj_scale /= factor # discount residual branch

        else:
            in_init_std = init_std or (self.dim ** (-0.5))
            out_init_std = init_std or (self.hidden_dim ** (-0.5))
            in_init_std = in_init_std
            out_init_std = out_init_std

            in_proj_scale = 1.0
            out_proj_scale = factor**-1
            
        if self.args.use_moe:

            ## ffn in, GLU or not
            for w in [self.moe.experts.mlp.w1, self.moe.experts.mlp.v1, self.moe.router.layer]:
                nn.init.trunc_normal_(
                    w, 
                    mean=0.0,
                    std=in_init_std *in_proj_scale, 
                    a=-3 * in_init_std *in_proj_scale,
                    b=3 * in_init_std *in_proj_scale,
                )

            ## out proj
            nn.init.trunc_normal_(
                self.moe.experts.mlp.w,
                mean=0.0,
                std=out_init_std *out_proj_scale,
                a=-3 * out_init_std *out_proj_scale,
                b=3 * out_init_std *out_proj_scale,
            )

            ## bias
            if self.moe.experts.bias is not None:
                torch.nn.init.zeros_(self.moe.experts.bias)
                
        else:
            for w in [self.w1, self.w3]:
                nn.init.trunc_normal_(
                    w.weight,
                    mean=0.0,
                    std=in_init_std *in_proj_scale,
                    a=-3 * in_init_std *in_proj_scale,
                    b=3 * in_init_std *in_proj_scale,
                )
            nn.init.trunc_normal_(
                self.w2.weight,
                mean=0.0,
                std=out_init_std *out_proj_scale,
                a=-3 * out_init_std *out_proj_scale,
                b=3 * out_init_std *out_proj_scale,
            )

        if self.args.residual_post_norm:
            self.fc2_norm.reset_parameters()

        if self.args.ngpt:
            self.u_scaler.reset_parameters()
            self.v_scaler.reset_parameters()

class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args

        assert (args.head_dim is not None) or (args.n_heads is not None), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads
        d_model = args.dim or int(args.n_heads * args.head_dim)

        assert args.n_heads % self.n_kv_heads == 0
        assert d_model % args.n_heads == 0

        self.attention = Attention(
            args=args,
            dim=d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            args=args,
            dim=d_model,
            hidden_dim=4 * d_model,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

        if self.args.ngpt:
            self.attention_scaler = Scaler(d_model, scale=d_model**-0.5, scale_init=0.05)
            self.ffn_scaler = Scaler(d_model, scale=d_model**-0.5)
        else:
            norm = FusedRMSNorm if self.args.fused_rms_norm else RMSNorm
            self.attention_norm = norm(d_model, eps=args.norm_eps)
            self.ffn_norm = norm(d_model, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        if self.args.ngpt:
            x = l2_norm(
                x + self.attention_scaler(
                    l2_norm(
                        self.attention(
                            x,
                            freq_cis,
                            tok_idx=tok_idx,
                            mask=mask,
                            attn_impl=attn_impl,
                        )
                    ) - x
                )  # ngpt paper (10), table 1
            )
            x = l2_norm(
                x + self.ffn_scaler(
                    l2_norm(
                        self.feed_forward(x)
                    ) - x
                )  # ngpt paper (11), table 1
            )
            return x
        else:
            h = x + self.attention(
                self.attention_norm(x),
                freq_cis,
                tok_idx=tok_idx,
                mask=mask,
                attn_impl=attn_impl,
            )
            out = h + self.feed_forward(self.ffn_norm(h))
            return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.feed_forward.reset_parameters(init_std, factor)

        if self.args.ngpt:
            self.attention_scaler.reset_parameters()
            self.ffn_scaler.reset_parameters()
        else:
            self.attention_norm.reset_parameters()
            self.ffn_norm.reset_parameters()


class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim or int(args.n_heads * args.head_dim)
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or int(args.dim // args.n_heads),
            max_seqlen=args.max_seqlen,
        )

        if self.args.ngpt:
            assert not self.args.qk_norm, "ngpt already normalized qk"
            assert not self.args.residual_post_norm, "ngpt already normalized residual output"

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters(init_std=self.init_base_std)
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5, # mitchell init
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,  # classic residual discount (gpt-2 like)
                InitStdFactor.DIM_RATIO: self.dim / 4096, # it's like mup
                InitStdFactor.DISABLED: 1.0, # dumb
            }[self.init_std_factor]
            layer.init_weights(self.init_base_std, factor)
