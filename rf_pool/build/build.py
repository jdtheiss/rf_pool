from collections import OrderedDict
from copy import deepcopy
import warnings

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from rf_pool import losses, metrics, models, modules, pool, solver
from rf_pool.data import dataloaders, datasets, transforms

def check_keys(d, endswith, warn=True):
    """convenience function to return a dict key with given ending"""
    key = [k for k in d.keys() if k.endswith(endswith)]
    if len(key) == 0:
        if warn:
            warnings.warn('No "%s" field found in config' % endswith)
        return
    return key[0]

def _none_fn(*args, **kwargs):
    return None

def get_class(options, name, throw_error=False):
    """convenience function to get a class from several modules"""
    for opt in options:
        if hasattr(opt, name):
            return getattr(opt, name)
    if throw_error:
        raise Exception('Module "%s" not found' % name)
    return None

def _eval_str_inputs(inputs, startswith='torch.'):
    eval_fn = lambda x: eval(x) if type(x) is str and x.startswith(startswith) else x
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            inputs.update({k: eval_fn(v)})
    elif isinstance(inputs, (list,set,tuple)):
        for i, v in enumerate(inputs):
            inputs[i] = eval_fn(v)
    else:
        inputs = eval_fn(inputs)
    return inputs

def _call_parsed_inputs(fn, inputs, parse_torch_str=True):
    # parse for 'torch.'
    if parse_torch_str:
        inputs = _eval_str_inputs(inputs, startswith='torch.')
    # call function based on type of inputs
    if isinstance(inputs, dict):
        return fn(**inputs)
    elif isinstance(inputs, (list,set,tuple)):
        return fn(*inputs)
    return fn(inputs)

def build_module(mod):
    """convenience function to build module from dict (or str)"""
    if isinstance(mod, str):
        return eval(mod)
    elif isinstance(mod, nn.Module) or not isinstance(mod, dict):
        return mod

    # init output Sequential
    out = nn.Sequential()
    for i, (k, v) in enumerate(mod.items()):
        # get mod name and function
        mod_name = '%s_%d' % (k.lower(), i)
        mod_fn = get_class([modules, modules.ops, pool, nn], k, throw_error=True)
        # add module based on v type
        out.add_module(mod_name, _call_parsed_inputs(mod_fn, v))

    # if only one module, return it
    if len(out._modules) == 1:
        return list(out._modules.values())[0]
    return out

def build_solver(cfg):
    """
    Build solver from configuration with field Solver

    Parameters
    ----------
    cfg : dict
        configuration containing necessary components to build model, loss, etc.

    Returns
    -------
    solver : Solver
        solver used with pytorch_lightning

    Notes
    -----
    Solver can be derived from solver.py.
    """
    # get solver key from cfg
    solver_key = check_keys(cfg, 'Solver')

    # if none, return default solver
    if solver_key is None:
        return solver.Solver(cfg)

    # return chosen solver
    return getattr(solver, cfg.get(solver_key) or solver_key)(cfg)

def build_model(cfg):
    """
    Build model from configuration with field Model

    Parameters
    ----------
    cfg : dict
        configuration containing modules and inputs to build modules

    Returns
    -------
    model : Model
        model to be trained or used during testing

    Notes
    -----
    `cfg.get('Model')` should be dict with modules as keys and __init__
    arguments as values (e.g., {'ConvBlock': {'in_channels', 3, 'out_channels': 64}}).
    To add multiple modules of the same class, the associated value should be
    a list instead (e.g., {'ConvBlock': [(3,64), (64,128), (128,256)]}). If only
    a single argument should be passed, the value should be a set instead
    (e.g., {'AdaptiveAvgPool2d: {(7,7)}'}).
    Notice that arguments can also be a tuple instead of dictionary.

    Modules can be derived from models.py, modules.py, pool.py, or torch.nn.
    """
    # get model key from cfg
    model_key = check_keys(cfg, 'Model')
    if model_key is None:
        return _none_fn
    model_class = getattr(models, model_key)
    # set kwargs from sub-cfg
    kwargs = deepcopy(cfg.get(model_key))
    # init model
    model = nn.Sequential()
    # for each module in Model field, add_module
    for k, v in cfg.get(model_key).items():
        # get module
        mod = get_class([models.backbones, models, modules, modules.ops,
                         pool, nn, torchvision.models,
                         torchvision.models.detection,
                         torchvision.models.inception,
                         torchvision.models.segmentation],
                        k, throw_error=False)
        if mod is None:
            continue
        # pop field from kwargs
        kwargs.pop(k)
        # create multiple modules
        if not isinstance(v, list):
            v = [v]
        for i, v_i in enumerate(v):
            if len(v) > 1:
                mod_name = '%s_%d' % (k, i)
            else:
                mod_name = k
            # add module
            model.add_module(mod_name, _call_parsed_inputs(mod, v_i))
    # if only one module (e.g., backbone/torchvision model), return module
    if len(model._modules) == 1:
        model = list(model._modules.values())[0]
    return model_class(model, **kwargs)

def build_loss(cfg):
    """
    Build loss from configuration with field Loss

    Parameters
    ----------
    cfg : dict
        configuration containing losses to use during training

    Returns
    -------
    loss : Loss
        loss function used during training

    Notes
    -----
    `cfg.get('Loss')` should be a dictionary with keys corresponding to
    the name of the loss to use and values corresponding to the arguments passed
    in order to initialize the loss. Alternatively, `cfg.get('Loss')` can be
    a list of dictionaries (i.e. to accomodate multiple losses with the same class).

    Additionally, loss weights can be set via `cfg.get('Loss').get('weights')`
    (if `type(cfg.get('Loss')) is dict`, otherwise weights should be set for each
    loss class as a kwarg `weights`). Default is `1.` for each weight in
    `cfg.get('Loss')`.

    Losses can be derived from either losses.py or torch.nn.
    """
    # get loss key from cfg
    loss_key = check_keys(cfg, 'Loss')
    if loss_key is None:
        return _none_fn
    loss_class = getattr(losses, loss_key)

    # set items for building class
    assert isinstance(cfg.get(loss_key), (dict, list)), (
        'type(cfg["%s"]) should be dict or list, found %s.' % (loss_key, type(kwargs))
    )
    kwargs = cfg.get(loss_key).copy()
    if isinstance(kwargs, dict):
        items = list(kwargs.items())
    else: # get from list
        items = []
        [items.extend(l.items()) for l in kwargs]
        # set kwargs as {}
        kwargs = {}

    # for each field, update cls_dict
    loss_dict = OrderedDict()
    for i, (name, inputs) in enumerate(items):
        # get loss
        fn = get_class([losses, nn], name, throw_error=False)
        if fn is None:
            continue
        # pop field from kwargs
        kwargs.pop(name, None)
        # get count of previous functions with same name
        cnt = sum(k == name for k, _ in items[:i]) or ''
        loss_dict.update({'%s%s' % (name.lower(), cnt): _call_parsed_inputs(fn, inputs)})

    return loss_class(loss_dict, **kwargs)

def build_metric(cfg):
    """
    Build metrics from configuration with field Metric

    Parameters
    ----------
    cfg : dict
        configuration containing metrics to use during training

    Returns
    -------
    metric : Metric
        metric class used for monitoring during training

    Notes
    -----
    `cfg.get('Metric')` should be a dictionary with keys corresponding to
    the name of the metric to use and values corresponding to the arguments passed
    in order to initialize the metric.

    Metrics can be derived from metrics.py, pytorch_lightning.metrics, or
    torch.nn.
    """
    # get metric key from cfg
    metric_key = check_keys(cfg, 'Metric')
    if metric_key is None:
        return metrics.Metric()
    metric_class = getattr(metrics, metric_key)
    # set kwargs from sub-cfg
    kwargs = deepcopy(cfg.get(metric_key))
    # init metrics
    metric_dict = OrderedDict()
    # set kwargs from sub_cfg
    for k, v in cfg.get(metric_key).items():
        # get metric
        metric_fn = get_class([metrics, pl.metrics, nn], k, throw_error=False)
        if metric_fn is None:
            continue
        # pop field from kwargs
        kwargs.pop(k)
        # update metric dict
        metric_dict.update({k.lower(): _call_parsed_inputs(metric_fn, v)})
    return metric_class(metric_dict, **kwargs)

def build_transforms(kwargs):
    """convenience function to build transforms for datasets"""
    # update each key ending with transform
    keys = [k for k in kwargs.keys() if k.find('transform') != -1]
    for k in keys:
        t_list = []
        for name, args in kwargs.get(k).items():
            # get function
            fn = get_class([transforms, torchvision.transforms], name,
                           throw_error=True)
            # create transform
            t_list.append(_call_parsed_inputs(fn, args))
        # update kwargs with transforms
        kwargs.update({k: torchvision.transforms.Compose(t_list)})
    return kwargs

def build_dataloader(cfg):
    """
    Build dataloader from configuration with field DataLoader

    Parameters
    ----------
    cfg : dict
        configuration containing dataloader/datasets to build

    Returns
    -------
    dataloader : DataLoader
        dataloader used to load datasets

    Notes
    -----
    `cfg.get('DataLoader')` should be a dictionary with keys corresponding to
    the name of the dataset to use and values corresponding to the arguments passed
    in order to initialize the dataset.

    Datasets can be derived from datasets.py, torchvision.datasets, or
    torch.utils.data.
    """
    # get dataloader key from cfg
    dataloader_key = check_keys(cfg, 'DataLoader', warn=False)
    if dataloader_key is None:
        return None
    dataloader_class = getattr(dataloaders, dataloader_key)
    # set kwargs from sub-cfg
    kwargs = deepcopy(cfg.get(dataloader_key))
    # init dataloaders
    dataloader_dict = OrderedDict()
    # set kwargs from sub_cfg
    for k, v in cfg.get(dataloader_key).items():
        # get metric
        dataset_fn = get_class([datasets, torchvision.datasets, torch.utils.data],
                               k, throw_error=False)
        if dataset_fn is None:
            continue
        # pop field from kwargs
        kwargs.pop(k)
        # update metric dict
        if isinstance(v, dict):
            # update kwargs with transforms
            v = build_transforms(v.copy())
            dataloader_dict.update({k.lower(): dataset_fn(**v)})
        else:
            dataloader_dict.update({k.lower(): _call_parsed_inputs(dataset_fn, v)})
    return dataloader_class(dataloader_dict, **kwargs)

def build_optimizer(cfg):
    """
    Build optimizer or learning schedulers from configuration with field
    Optimizer

    Parameters
    ----------
    cfg : dict
        configuration containing optimizers/schedulers to build

    Returns
    -------
    optims : list[dict]
        list of dictionaries that will be used to create optimizers
        or learning schedulers

    Notes
    -----
    `cfg.get('Optimizer')` should be a dictionary or list of dictionaries with
    keys corresponding to the name of the optimizer/scheduler to use and values
    corresponding to the arguments passed in order to initialize the
    optimizer/scheduler. This can also contain other (key, value) pairs used
    by pytorch_lightning (e.g., {'interval': 'epoch', 'frequency': 1}).

    Optimizers/schedulers can be derived from torch.optim or
    torch.optim.lr_scheduler.
    """
    # get dataloader key from cfg
    optim_key = check_keys(cfg, 'Optimizer', warn=True)
    if optim_key is None:
        return []
    # set optimizers or learning schedulers from sub-cfg
    optims = deepcopy(cfg.get(optim_key))
    if not isinstance(optims, list):
        optims = [optims]
    # update with optimizer/scheduler functions
    for optim_group in optims:
        items = list(optim_group.items())
        for k, v in items:
            # get optimizer/scheduler function
            optim_fn = get_class([torch.optim, torch.optim.lr_scheduler], k,
                                 throw_error=False)
            if optim_fn is None:
                continue
            # pop from optim_group
            optim_group.pop(k)
            # set key and update optim_group
            if optim_fn.__module__.endswith('lr_scheduler'):
                key = 'scheduler'
            else:
                key = 'optimizer'
            optim_group.update({key: (optim_fn, v)})
    return optims
