# Copyright 2022 Microsoft Corporation.
'''
Helper functions for performing coord check.
'''
import os
from copy import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mup.optim import MuAdam

from collections import defaultdict
from pdb import set_trace as Tra

def adjust_hidden_dim(hidden_dim, ffn_dim_multiplier, multiple_of):
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

def get_optimizer(model, lr, weight_decay, adam_beta1, adam_beta2, optimizer_name, truly_decoupled_wd=False):
    opt_cls = torch.optim.AdamW

    def new_group():
        new_g = {
            'lr': lr,
            'weight_decay': weight_decay,
        }
        new_g['params'] = []
        return new_g
    def new_group_():
        new_g = {
            'lr': lr,
            'weight_decay': 0.,
        }
        new_g['params'] = []
        return new_g

    no_decay_name_list = ["bias", "norm", "scaler"]
    optimizer_grouped_parameters = []
    opt_kwargs = {
        'betas': (adam_beta1, adam_beta2),
        'eps': 1e-8,
        'fused': True,
    }

    if model.args.mup:

        ## proxy model's configs
        base_dim = model.args.base_dim or int(model.args.base_head_dim * model.args.base_n_heads)
        base_head_dim = model.args.base_head_dim or int(model.args.base_dim // model.args.base_n_heads)
        base_ffn_hidden_dim = adjust_hidden_dim(
            4*base_dim, 
            model.args.base_ffn_dim_multiplier, 
            model.args.base_multiple_of,
        )

        ## target model's configs
        dim = model.args.dim or int(model.args.head_dim * model.args.n_heads)
        head_dim = model.args.head_dim or int(model.args.dim // model.args.n_heads)
        ffn_hidden_dim = adjust_hidden_dim(
            4*dim, 
            model.args.ffn_dim_multiplier, 
            model.args.multiple_of
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
                    print(f'{n} is in {no_decay_name_list}, so wd is set as 0')
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
                    print(f'{n} is in {no_decay_name_list}, so wd is set as 0')
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

def cov(x):
    '''Treat `x` as a collection of vectors and its Gram matrix.
    Input:
        x: If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Output:
        cov: a matrix of size N x N where N is the product of
            the non-last dimensions of `x`.
    '''
    if x.nelement() == 1:
        width = 1
        xx = x.reshape(1, 1)
    else:
        width = x.shape[-1]
        xx = x.reshape(-1, x.shape[-1])
    return xx @ xx.T / width

def covoffdiag(x):
    '''Get off-diagonal entries of `cov(x)` in a vector.
    Input:
        x: If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Output:
        Off-diagonal entries of `cov(x)` in a vector.'''
    c = cov(x)
    return c[~torch.eye(c.shape[0], dtype=bool)]

#: dict of provided functions for use in coord check
FDICT = {
    'l1': lambda x: torch.abs(x).mean(dtype=torch.float32),
    'l2': lambda x: (x**2).mean(dtype=torch.float32)**0.5,
    'mean': lambda x: x.mean(dtype=torch.float32),
    'std': lambda x: x.std(dtype=torch.float32),
    'covl1': lambda x: torch.abs(cov(x)).mean(dtype=torch.float32),
    'covl2': lambda x: (cov(x)**2).mean(dtype=torch.float32)**0.5,
    'covoffdiagl1': lambda x: torch.abs(covoffdiag(x)).mean(dtype=torch.float32),
    'covoffdiagl2': lambda x: (covoffdiag(x)**2).mean(dtype=torch.float32)**0.5
}

def convert_fdict(d):
    '''convert a dict `d` with string values to function values.
    Input:
        d: a dict whose values are either strings or functions
    Output:
        a new dict, with the same keys as `d`, but the string values are
        converted to functions using `FDICT`.
    '''
    return dict(
        [
            (
                (k, FDICT[v]) if isinstance(v, str) else (k, v)
            ) for k, v in d.items()
        ]
    )
    
def _record_coords(records, width, modulename, t,
                output_fdict=None, input_fdict=None, param_fdict=None):
    '''Returns a forward hook that records coordinate statistics.

    Returns a forward hook that records statistics regarding the output, input,
    and/or parameters of a `nn.Module`. This hook is intended to run only once,
    on the timestep specified by `t`.

    On forward pass, the returned hook calculates statistics specified in
    `output_fdict`, `input_fdict`, and `param_fdict`, such as the normalized l1
    norm, of output, input, and/or parameters of the module. The statistics are
    recorded along with the `width`, `modulename`, and `t` (the time step) as a
    dict and inserted into `records` (which should be a list). More precisely,
    for each output, input, and/or parameter, the inserted dict is of the form

        {
            'width': width, 'module': modified_modulename, 't': t,
            # keys are keys in fdict
            'l1': 0.241, 'l2': 0.420, 'mean': 0.0, ...
        }
    
    where `modified_modulename` is a string that combines the `modulename` with
    an indicator of which output, input, or parameter tensor is the statistics
    computed over.

    The `*_fdict` inputs should be dictionaries with string keys and whose
    values can either be functions or strings. The string values are converted
    to functions via `convert_fdict`. The default values of `*_dict` inputs are
    converted to `output_fdict = dict(l1=FDICT['l1'])`, `input_fdict = {}`,
    `param_fdict = {}`, i.e., only the average coordinate size (`l1`) of the
    output activations are recorded.

    Inputs:
        records:
            list to append coordinate data to
        width:
            width of the model. This is used only for plotting coord check later
            on, so it can be any notion of width.
        modulename:
            string name of the module. This is used only for plotting coord check.
        t:
            timestep of training. This is used only for plotting coord check.
        output_fdict, input_fdict, param_fdict:
            dicts with string keys and whose values can either be functions or
            strings. The string values are converted to functions via
            `convert_fdict`
    Output:
        a forward hook that records statistics regarding the output, input,
        and/or parameters of a `nn.Module`, as discussed above.
    '''
    if output_fdict is None:
        output_fdict = dict(l1=FDICT['l1'])
    else:
        output_fdict = convert_fdict(output_fdict)
    if input_fdict is None:
        input_fdict = {}
    else:
        input_fdict = convert_fdict(input_fdict)
    if param_fdict is None:
        param_fdict = {}
    else:
        param_fdict = convert_fdict(param_fdict)

    def f(module, input, output):
        
        def get_stat(d, x, fdict):
            if isinstance(x, (tuple, list)):
                for i, _x in enumerate(x):
                    _d = copy(d)
                    _d['module'] += f'[{i}]'
                    get_stat(_d, _x, fdict)
            elif isinstance(x, dict):
                for name, _x in x.items():
                    _d = copy(d)
                    _d['module'] += f'[{name}]'
                    get_stat(_d, _x, fdict)
            elif isinstance(x, torch.Tensor):
                _d = copy(d)
                for fname, f in fdict.items():
                    _d[fname] = f(x).item()
                records.append(_d)
            elif x is None:
                pass
            else:
                raise NotImplementedError(f'Unexpected output type: {type(x)}')
            
        with torch.no_grad():
            ret = {
                'width': width,
                'module': modulename,
                't': t
            }

            # output stats
            if isinstance(output, (tuple, list)):
                for i, out in enumerate(output):
                    _ret = copy(ret)
                    _ret['module'] += f':out[{i}]'
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, dict):
                for name, out in output.items():
                    _ret = copy(ret)
                    _ret['module'] += f':out[{name}]'
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, torch.Tensor):
                _ret = copy(ret)
                for fname, f in output_fdict.items():
                    _ret[fname] = f(output).item()
                records.append(_ret)
            else:
                raise NotImplementedError(f'Unexpected output type: {type(output)}')

            # input stats
            if input_fdict:
                if isinstance(input, (tuple, list)):
                    for i, out in enumerate(input):
                        _ret = copy(ret)
                        _ret['module'] += f':in[{i}]'
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, dict):
                    for name, out in input.items():
                        _ret = copy(ret)
                        _ret['module'] += f':in[{name}]'
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, torch.Tensor):
                    _ret = copy(ret)
                    for fname, f in input_fdict.items():
                        _ret[fname] = f(input).item()
                    records.append(_ret)
                else:
                    raise NotImplementedError(f'Unexpected output type: {type(input)}')

            # param stats
            if param_fdict:
                for name, p in module.named_parameters():
                    _ret = copy(ret)
                    _ret['module'] += f':param[{name}]'
                    for fname, f in param_fdict.items():
                        _ret[fname] = f(p).item()
                    records.append(_ret)

    return f

def _get_coord_data(models, dataloader, optcls, nsteps=3,
                dict_in_out=False, flatten_input=False, flatten_output=False, 
                output_name='loss', lossfn='xent', filter_module_by_name=None,
                fix_data=True, cuda=True, nseeds=1, 
                output_fdict=None, input_fdict=None, param_fdict=None,
                show_progress=True, one_hot_target=False):
    '''Inner method for `get_coord_data`.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        models: 
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optcls: 
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps: 
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict: 
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
        
    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.
    
    '''
    df = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model in models.items():
            model = model()
            model.init_weights()
            for n, p in model.named_parameters():
                print(f'{n}: {p.mean()} {p.std()}')
            model = model.train()
            if cuda:
                model = model.cuda()

            optimizer = optcls(model)

            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(
                        module.register_forward_hook(
                        _record_coords(
                                df, width, name, batch_idx,
                                output_fdict=output_fdict,
                                input_fdict=input_fdict,
                                param_fdict=param_fdict
                            )
                        )
                    )
                if dict_in_out:
                    if cuda:                        
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                batch[k] = v.cuda()
                    outputs = model(**batch)
                    # Tra()
                    if isinstance(outputs, torch.Tensor):
                        loss = outputs
                    else:
                        loss = outputs[output_name] if isinstance(outputs, dict) else outputs[0]
                else:
                    (data, target) = batch
                    if cuda:
                        data, target = data.cuda(), target.cuda()
                    if flatten_input:
                        data = data.view(data.size(0), -1)
                    output = model(data)
                    if flatten_output:
                        output = output.view(-1, output.shape[-1])
                    if one_hot_target:
                        target = F.one_hot(target,
                                  num_classes=output.size(-1)).float()
                    if lossfn == 'xent':
                        loss = F.cross_entropy(output, target)
                    elif lossfn == 'mse':
                        loss = F.mse_loss(output, target)
                    elif lossfn == 'nll':
                        loss = F.nll_loss(output, target)
                    elif lossfn == 'l1':
                        loss = F.l1_loss(output, target)
                    elif callable(lossfn):
                        loss = lossfn(output, target)
                    else:
                        raise NotImplementedError(f'unknown `lossfn`: {lossfn}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps: break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)


def get_coord_data(models, dataloader, optimizer='adam', lr=None, mup=True,
                    filter_trainable_by_name=None,
                    weight_decay=0., adam_beta1=0.9, adam_beta2=0.999,
                    **kwargs):
    '''Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models: 
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        lr: 
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup: 
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name: 
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps: 
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict: 
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).
        
    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.
    '''
    if lr is None:
        lr = 0.1 if optimizer == 'sgd' else 1e-3

    if optimizer in ['customized_adam', 'customized_adamw']:
        optimizer_name = optimizer.split('customized_')[-1]
        optcls = lambda model: get_optimizer(model, lr, weight_decay, adam_beta1, adam_beta2, optimizer_name)
    else:
        if mup:
            from mup.optim import MuAdam as Adam
            from mup.optim import MuAdamW as AdamW
            from mup.optim import MuSGD as SGD
        else:
            from torch.optim import SGD, Adam, AdamW

        def get_trainable(model):
            params = model.parameters()
            if filter_trainable_by_name is not None:
                params = []
                for name, p in model.named_parameters():
                    if filter_trainable_by_name(name):
                        params.append(p)
            return params
        
        if optimizer == 'sgd':
            optcls = lambda model: SGD(get_trainable(model), lr=lr)
        elif optimizer == 'adam':
            optcls = lambda model: Adam(get_trainable(model), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optcls = lambda model: AdamW(get_trainable(model), lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
        else:
            raise ValueError('optimizer should be sgd|adam|adamw or a custom function')

    data = _get_coord_data(models, dataloader, optcls, **kwargs)
    data['optimizer'] = optimizer
    data['lr'] = lr
    return data


def plot_coord_data(df, y='l1', save_to=None, suptitle=None, x='width', hue='module',
                    legend='full', name_contains=None, name_not_contains=None, module_list=None,
                    loglog=True, logbase=2, face_color=None, subplot_width=5,
                    subplot_height=4):
    '''Plot coord check data `df` obtained from `get_coord_data`.

    Input:
        df:
            a pandas DataFrame obtained from `get_coord_data`
        y:
            the column of `df` to plot on the y-axis. Default: `'l1'`
        save_to:
            path to save the resulting figure, or None. Default: None.
        suptitle:
            The title of the entire figure.
        x:
            the column of `df` to plot on the x-axis. Default: `'width'`
        hue:
            the column of `df` to represent as color. Default: `'module'`
        legend:
            'auto', 'brief', 'full', or False. This is passed to `seaborn.lineplot`.
        name_contains, name_not_contains:
            only plot modules whose name contains `name_contains` and does not contain `name_not_contains`
        module_list:
            only plot modules that are given in the list, overrides `name_contains` and `name_not_contains`
        loglog:
            whether to use loglog scale. Default: True
        logbase:
            the log base, if using loglog scale. Default: 2
        face_color:
            background color of the plot. Default: None (which means white)
        subplot_width, subplot_height:
            The width and height for each timestep's subplot. More precisely,
            the figure size will be
                `(subplot_width*number_of_time_steps, subplot_height)`.
            Default: 5, 4

    Output:
        the `matplotlib` figure object
    '''
    ### preprocessing
    df = copy(df)
    # nn.Sequential has name '', which duplicates the output layer
    df = df[df.module != '']
    if module_list is not None:
        df = df[df['module'].isin(module_list)]
    else:
        if name_contains is not None:
            df = df[df['module'].str.contains(name_contains)]
        if name_not_contains is not None:
            df = df[~(df['module'].str.contains(name_not_contains))]
    # for nn.Sequential, module names are numerical
    try:
        df['module'] = pd.to_numeric(df['module'])
    except ValueError:
        pass

    ts = df.t.unique()

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    def tight_layout(plt):
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ### plot
    fig = plt.figure(figsize=(subplot_width * len(ts), subplot_height))
    hue_order = sorted(set(df['module']))
    if face_color is not None:
        fig.patch.set_facecolor(face_color)
    Tra()
    ymin, ymax = min(df[y]), max(df[y])
    for t in ts:
        t = int(t)
        plt.subplot(1, len(ts), t)
        sns.lineplot(x=x, y=y, data=df[df.t == t], hue=hue, hue_order=hue_order, legend=legend if t == 1 else None)
        plt.title(f't={t}')
        if t != 1:
            plt.ylabel('')
        if loglog:
            plt.loglog(base=logbase)
        ax = plt.gca()
        ax.set_ylim([ymin, ymax])
    if suptitle:
        plt.suptitle(suptitle)
    tight_layout(plt)
    if save_to is not None:
        plt.savefig(save_to)
        print(f'coord check plot saved to {save_to}')

    return fig