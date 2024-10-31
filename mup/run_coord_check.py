# Copyright 2022 Microsoft Corporation.

import os
from functools import partial
from itertools import cycle

import numpy as np
import torch

import seaborn as sns
from mup.coord_check import get_coord_data, plot_coord_data

from transformers import GPT2Tokenizer, LlamaTokenizer, AutoTokenizer
from apps.main.transformer import LMTransformerArgs, LMTransformer
from copy import deepcopy

sns.set()


def get_dataloader():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "I love tensor program and greg yang."
    encoded_input = tokenizer(text, return_tensors='pt')
    inputs = {
        'token_values': encoded_input['input_ids'],
        'target': encoded_input['input_ids'],
    }
    dataloader = cycle([inputs])
    return dataloader


def get_lazy_model(
    width, nhead, kv_nhead,
    base_width, base_nhead, base_kv_nhead,
    gqa=False,
    mup=True, readout_zero_init=True, query_zero_init=True, 
    vary_nhead=False,
    init_std=0.02, input_mult=1, output_mult=1,
):
    print(f'''
    width: {width}, nhead: {nhead}, kv_nhead: {kv_nhead} 
    base_width: {base_width}, base_nhead: {base_nhead}, base_kv_nhead: {base_kv_nhead}
    gqa: {gqa}
    ''')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    def f():
        base_args = {
            'vocab_size': len(tokenizer),
            'n_layers': 2,
            'head_dim': width,
            'n_heads': nhead,
            'n_kv_heads': kv_nhead,
            'ffn_dim_multiplier': 1.3,
            'multiple_of': 256,
            'init_base_std': init_std,
            'init_std_factor': 'global_depth',
        }
        if mup:
            mup_args = {
                'mup': True,

                'base_head_dim': base_width,
                'base_n_heads': base_nhead,
                'base_n_kv_heads': kv_nhead,
                'base_ffn_dim_multiplier': 1.3,
                'base_multiple_of': 256,

                'input_mult': input_mult,
                'output_mult': output_mult,

                'readout_zero_init': readout_zero_init,
                'query_zero_init': query_zero_init,
            }
            base_args.update(mup_args)

        return LMTransformer(args=LMTransformerArgs(**base_args))
    
    return f

def plot_coord_check(
    mup=True, vary_nhead=False, 
    y='l1', widths=None, nheads=[8], gqa=False,
    optimizer='adam',
    nseeds=1, nsteps=4, loglog=False, logbase=2, legend=None,
    init_std=0.02, input_mult=10.0, output_mult=1.0,
    readout_zero_init=True, query_zero_init=True,
    weight_decay=0.1, adam_beta1=0.9, adam_beta2=0.95,
    lr=None,
    **get_coord_data_kw
):
    if vary_nhead:
        widths = [64]
        if len(nheads) == 1:
            nheads = 2**np.arange(2,5) # 4, 8, 16
        if gqa:
            kv_nheads = [int(nhead//4) for nhead in nheads]
        else:
            kv_nheads = [nhead for nhead in nheads]
        hiddens = [ (widths[0], nhead, kv_nhead) for nhead, kv_nhead in zip(nheads, kv_nheads) ]
    else:
        if widths is None:
            widths = 2**np.arange(6, 11) # 64 ~ 1024 * (8 heads)
        if gqa:
            kv_nheads = int(nheads[0]//4)
        else:
            kv_nheads = nheads[0]
        hiddens = [ (width, nheads[0], kv_nheads[0]) for width in widths ]

    base_width = widths[0]
    base_nhead = nheads[0]
    base_kv_nhead = kv_nheads[0]

    models = {
        int(width * nhead): get_lazy_model(
            width, nhead, kv_nhead,
            base_width, base_nhead, base_kv_nhead, 
            gqa,
            mup=mup, vary_nhead=vary_nhead,
            readout_zero_init=readout_zero_init, query_zero_init=query_zero_init,
            init_std=init_std, input_mult=input_mult, output_mult=output_mult,
        ) for (width, nhead, kv_nhead) in hiddens
    }
    dataloader = get_dataloader()
    df = get_coord_data(
        models, dataloader, mup=mup, 
        optimizer=optimizer,
        nseeds=nseeds, dict_in_out=True,
        nsteps=nsteps, 
        weight_decay=weight_decay, adam_beta1=adam_beta1, adam_beta2=adam_beta2,
        lr=lr,
        **get_coord_data_kw
    )

    parameterization = 'MuP' if mup else 'SP'
    width = 'nhead' if vary_nhead else 'dhead'

    save_dir = 'exp_logs/assets/images/mup_native_coord_check'
    os.makedirs(save_dir, exist_ok=True)
    suffix = ''
    save_file_name = f'{parameterization}_varying_{width}_gqa_{gqa}_basestd_{init_std}_inputmult_{input_mult}_outputmult_{output_mult}_lr_{lr}'
    save_file_name += f'{optimizer}_wd_{weight_decay}_b1_{adam_beta1}_b2_{adam_beta2}'
    save_file_name += f'{suffix}'

    return plot_coord_data(
        df, legend=legend, loglog=loglog, logbase=logbase, x='width',y=y,
        save_to=os.path.join(save_dir, f'{save_file_name}.png'),
        suptitle=f'{save_file_name}',
        face_color='xkcd:light grey' if not mup else None
    )