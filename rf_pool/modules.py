from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops
from .utils import functions, visualize

class Module(nn.Module):
    """
    Base class for modules

    Attributes
    ----------
    forward_layer : torch.nn.Sequential
        functions to apply in forward pass
    reconstruct_layer : torch.nn.Sequential or None
        functions to apply in reconstruct pass
    """
    def __init__(self, input_shape=None):
        super(Module, self).__init__()
        self.input_shape = input_shape
        self.reconstruct_shape = input_shape
        self.forward_layer = nn.Sequential()
        self.reconstruct_layer = nn.Sequential()

    def output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        return self.forward(torch.zeros(input_shape)).shape

    def link_parameters(self, layer, layer_name=None):
        if layer_name:
            layer_name = str(layer_name)
        else:
            layer_name = ''
        for name, param in layer.named_parameters():
            if layer_name:
                param_name = '_'.join([layer_name, name.replace('.','_')])
            else:
                param_name = name.replace('.','_')
            self.register_parameter(param_name, param)

    def init_weights(self, named_parameters=None, pattern='weight',
                     fn=lambda x: 0.01 * torch.randn_like(x)):
        if named_parameters is None:
            named_parameters = self.named_parameters()
        for name, param in named_parameters:
            with torch.no_grad():
                if name.find(pattern) >=0:
                    param.set_(fn(param))

    def make_layer(self, layer_name, transpose=False, **kwargs):
        # init layer to nn.Sequential()
        setattr(self, layer_name, nn.Sequential())
        # set None to nn.Sequential()
        for name, module in kwargs.items():
            if module is None:
                kwargs.update({name: nn.Sequential()})
        # update layers
        self.update_layer(layer_name, transpose=transpose, **kwargs)

    def update_layer(self, layer_name, transpose=False, **kwargs):
        # get layer
        if hasattr(self, layer_name):
            layer = getattr(self, layer_name)
        else:
            layer = nn.Sequential()
        # update layer
        for name, module in kwargs.items():
            # set as Op if module not Module instance
            if not isinstance(module, torch.nn.modules.Module):
                module = ops.Op(module)
            if transpose:
                name = name + '_transpose'
                module = self.transposed_fn(module)
            # add reshape op if linear
            if self.input_shape is None and isinstance(module, torch.nn.Linear):
                reshape_op = ops.Op(lambda x: x.flatten(1))
                layer.add_module('reshape_%s' % name, reshape_op)
            if module is not None:
                layer.add_module(name, module)
        # set layer
        setattr(self, layer_name, layer)

    def update_module(self, layer_name, module_name, module):
        self.udpate_layer(layer_name, **{module_name: module})

    def insert_module(self, layer_name, idx, transpose=False, **kwargs):
        # get layer
        layer = nn.Sequential()
        if hasattr(self, layer_name):
            orig_layer = getattr(self, layer_name)
        else:
            orig_layer = layer
        # set empty layer
        setattr(self, layer_name, layer)
        # set modules in order
        mods = list(orig_layer.named_children())
        new_mods = list(kwargs.items())
        new_mods.reverse()
        [mods.insert(idx, new_mod) for new_mod in new_mods]
        # update layer
        self.update_layer(layer_name, transpose=transpose, **OrderedDict(mods))

    def remove_module(self, layer_name, module_name):
        # get layer
        layer = nn.Sequential()
        if hasattr(self, layer_name):
            orig_layer = getattr(self, layer_name)
            mods = OrderedDict([(k, v) for k, v in orig_layer.named_children()
                                if k != module_name])
        else:
            mods = {}
        # set modules other than module_name
        for key, value in mods.items():
            layer.add_module(key, value)
        # set layer
        setattr(self, layer_name, layer)

    def transposed_fn(self, fn):
        # transposed conv
        if hasattr(fn, 'weight') and torch.typename(fn).find('conv') >= 0:
            conv_kwargs = functions.get_attributes(fn, ['stride','padding','dilation'])
            transposed_fn = nn.ConvTranspose2d(fn.out_channels, fn.in_channels,
                                               fn.kernel_size, **conv_kwargs)
            transposed_fn.weight = fn.weight
        # transposed linear
        elif hasattr(fn, 'weight') and torch.typename(fn).find('linear') >= 0:
            transposed_fn = nn.Linear(fn.out_features, fn.in_features)
            transposed_fn.weight = nn.Parameter(fn.weight.t())
        elif hasattr(fn, 'weight'):
            #TODO: how to use transposed version of fn implicitly
            raise Exception('fn type not understood')
        # unpool with indices
        elif hasattr(fn, 'return_indices') and fn.return_indices:
            pool_kwargs = functions.get_attributes(fn, ['stride', 'padding'])
            transposed_fn = nn.MaxUnpool2d(fn.kernel_size, **pool_kwargs)
        elif hasattr(fn, 'kernel_size'):
            transposed_fn = nn.Upsample(scale_factor=fn.kernel_size)
        else:
            transposed_fn = fn
        return transposed_fn

    def get_module_names(self, output_module_name=None,
                         layer_name='forward_layer'):
        module_names = []
        layer = getattr(self, layer_name)
        if type(output_module_name) is not list:
            output_module_name = [output_module_name]
        cnt = -1
        for i, (name, _) in enumerate(layer.named_children()):
            module_names.append(name)
            if name in output_module_name:
                cnt = i + 1
        if cnt > -1:
            return module_names[:cnt]
        return module_names

    def get_modules(self, layer_name, module_names):
        modules = []
        layer = getattr(self, layer_name)
        for name, module in layer.named_children():
            if name in module_names:
                modules.append(module)
        return modules

    def forward(self, input, module_names=[], output={}, **kwargs):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        for name, module in self.forward_layer.named_children():
            if len(module_names) > 0 and name not in module_names:
                continue
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            input = module(input, **mod_kwargs)
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(input)
        return input

    def reconstruct(self, input, module_names=[], output={}, **kwargs):
        for name, module in self.reconstruct_layer.named_children():
            if len(module_names) > 0 and name not in module_names:
                continue
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            input = module(input, **mod_kwargs)
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(input)
        if self.reconstruct_shape:
            input = torch.reshape(input, self.reconstruct_shape)
        return input

    def apply(self, input, layer_name='forward_layer', module_names=[],
              output={}, output_module=None, **kwargs):
        """
        Apply modules with module-specific kwargs and/or collect outputs in dict

        Parameters
        ----------
        input : torch.Tensor
            input passed through modules
        layer_name : str, optional
            name of Sequential to apply to input [default: 'forward_layer']
        module_names : list, optional
            names of modules in Sequential to apply (i.e. only these modules)
            [default: [], apply all modules in Sequential]
        output : dict, optional
            dictionary like {module_name: []} to be updated with specific results
            [default: {}, will not set outputs to dictionary]
        output_module : str, optional
            name of module to stop passing input through Sequential (i.e., get
            module_names for each module up to and including output_module)
            [default: None, apply all layers]
        **kwargs : dict
            module-specific keyword arguments like {module_name: kwargs} to be
            applied for a given module

        Results
        -------
        output : torch.Tensor
            output of passing input through modules

        Examples
        --------
        >>> # pass input through modules and obtain outputs of specific module
        >>> layer = FeedForward(hidden=torch.nn.Conv2d(1,16,5),
                                activation=torch.nn.ReLU(),
                                pool=torch.nn.MaxPool2d(2)))
        >>> saved_outputs = {'activation': []}
        >>> output = layer.apply(torch.rand(1,1,6,6),
                                 module_names=['hidden','activation','pool'],
                                 output=saved_outputs)
        >>> print(output.shape, saved_outputs.get('activation')[0].shape)
        torch.Size([1, 16, 1, 1]) torch.Size([1, 16, 2, 2])

        Notes
        -----
        The argument `layer_name` must have a corresponding function, which is
        obtained as getattr(self, layer_name.replace('_layer', '')). For example,
        'forward_layer' is the layer_name for the forward function and
        'reconstruct_layer' is the layer_name for the reconstruct function.
        """
        # get module_names
        if len(module_names) == 0:
            module_names = self.get_module_names(output_module, layer_name)
        # apply modules
        fn = getattr(self, layer_name.replace('_layer', ''))
        return fn(input, module_names, output, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def train(self, input, label, loss_fn, optimizer=None, monitor_loss=None,
              **kwargs):
        if optimizer:
            optimizer.zero_grad()
        # get output and loss
        output = self.forward(input)
        loss = loss_fn(output, label)
        # backprop
        loss.backward()
        if optimizer:
            optimizer.step()
        # monitor loss
        with torch.no_grad():
            if monitor_loss is not None:
                out = monitor_loss(input, output)
            else:
                out = loss
        return out.item()

    def show_weights(self, field='hidden_weight', img_shape=None,
                     figsize=(5, 5), cmap=None):
        """
        #TODO:WRITEME
        """
        # get field for weights
        if not hasattr(self, field):
            raise Exception('attribute ' + field + ' not found')
        w = getattr(self, field).clone().detach()
        # plot weights
        return visualize.plot_images(w, img_shape, figsize, cmap)

class FeedForward(Module):
    """
    Feed-forward layer

    Parameters
    ----------
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]
    **kwargs : dict
        modules for layer like {module_name: module}

    Returns
    -------
    None
    """
    def __init__(self, input_shape=None, **kwargs):
        super(FeedForward, self).__init__(input_shape)
        # build layer
        self.make_layer('forward_layer', **kwargs)
        # initialize biases to zeros
        self.init_weights(pattern='bias', fn=torch.zeros_like)
        # link parameters
        self.link_parameters(self.forward_layer)

class Branch(Module):
    """
    Branch layer applying multiple streams of modules to input data

    Parameters
    ----------
    branches : list of torch.nn.Module
        list of branches of modules to apply to input data
    branch_names : list, optional
        list of names to set for each branch
        (i.e., 'forward_layer.%s' % branch_names[0])
        [default: None, branch names are set as 'branch_%d' % (index)]
    branch_shapes : list, optional
        list of output shapes to which each branch output should be reshaped
        [default: None, no reshaping of branch outputs]
    cat_output : boolean, optional
        True/False of whether to concatenate branch outputs (along channel dim)
        Note branch outputs must match shape on all dims other than channel.
        [default: False, does not concatenate branches]
    output_names : list, optional
        list of output names to associate each branch output into a dictionary
        Note output from forward pass will be dictionary like
        {branch_name: branch_output}
        [default: None, branch outputs will not be dictionary]
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]
    **kwargs : dict
        modules for layer like {module_name: module}

    Returns
    -------
    None
    """
    def __init__(self, branches, branch_names=None, branch_shapes=None,
                 cat_output=False, output_names=None, input_shape=None):
        super(Branch, self).__init__(input_shape)
        self.branches = branches
        if branch_names is not None:
            assert len(branch_names) == len(branches)
        self.branch_names = branch_names
        self.branch_shapes = branch_shapes
        self.cat_output = cat_output
        self.output_names = output_names
        for i, branch in enumerate(self.branches):
            if self.branch_names is not None:
                branch_name = self.branch_names[i]
            else:
                branch_name = 'branch_%d' % i
            self.forward_layer.add_module(branch_name, branch)

    def output_shape(self, input_shape):
        outputs = self.forward(torch.zeros(input_shape))
        return [output.shape for output in outputs]

    def forward(self, input, module_names=[], output={}, **kwargs):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        outputs = []
        for i, (name, branch) in enumerate(self.forward_layer.named_children()):
            if len(module_names) > 0 and name not in module_names:
                continue
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            outputs.append(branch.forward(input, **mod_kwargs))
            if self.branch_shapes:
                outputs[-1] = torch.reshape(outputs[-1], self.branch_shapes[i])
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(outputs[-1])
        if self.cat_output:
            outputs = torch.cat(outputs, 1)
        if self.output_names is not None:
            outputs = OrderedDict([(k,v)
                                   for k, v in zip(self.output_names, outputs)])
        return outputs

    def reconstruct(self, input, module_names=[], output={}, **kwargs):
        outputs = []
        for name, branch in self.forward_layer.named_children():
            if len(module_names) > 0 and name not in module_names:
                continue
            #TODO: slice input channels for each branch
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            outputs.append(branch.reconstruct(input, **mod_kwargs))
            if self.reconstruct_shape:
                outputs[-1] = torch.reshape(outputs[-1], self.reconstruct_shape)
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(outputs[-1])
        if self.cat_output:
            outputs = torch.cat(outputs, 1)
        return outputs

class Control(Module):
    """
    Control network layer to update arguments/keyword arguments passed to other
    modules/layers

    Parameters
    ----------
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]
    **kwargs : dict
        modules for layer like {module_name: module}

    Returns
    -------
    None

    Notes
    -----
    This network must contain a `control` module whose output is passed to the
    following module as a second argument (e.g., module(input, control_output)).
    The `type` of control_output is used to discern how the output should be
    passed to the following module (i.e., dictionary types are passed as
    module(input, **control_output)).
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Control, self).__init__(input_shape)
        assert 'control' in kwargs.keys(), ('must contain "control" module')
        # build layer
        self.make_layer('forward_layer', **kwargs)
        # init biases
        self.init_weights(pattern='bias', fn=torch.zeros_like)
        # link parameters
        self.link_parameters(self.forward_layer)

    def forward(self, input, module_names=[], output={}, **kwargs):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        # apply module
        control_out = None
        for name, module in self.forward_layer.named_children():
            if len(module_names) > 0 and name not in module_names:
                continue
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            if name == 'control':
                control_out = module(input, **mod_kwargs)
            elif control_out is not None:
                if type(control_out) is list:
                    input = module(input, *control_out, **mod_kwargs)
                elif isinstance(control_out, (dict, OrderedDict)):
                    input = module(input, **control_out, **mod_kwargs)
                else:
                    input = module(input, control_out, **mod_kwargs)
            else:
                input = module(input, **mod_kwargs)
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(input)
        return input

class Lambda(Module):
    """
    Lambda function wrapped in a torch.nn.Module

    Parameters
    ----------
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]
    **kwargs : dict
        lambda functions for layer like {module_name: lambda_fn}

    Returns
    -------
    None
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Lambda, self).__init__(input_shape)
        # build layer
        for name, module in kwargs.items():
            if not isinstance(module, torch.nn.modules.Module):
                module = ops.Op(module)
            self.update_layer('forward_layer', **{name: module})

class RBM(Module):
    """
    Restricted Boltzmann Machine module (convolutional or fully-connected)

    Attributes
    ----------
    forward_layer : torch.nn.Sequential
        functions to apply in forward pass (see Notes)
    reconstruct_layer : torch.nn.Sequential
        functions to apply in reconstruct pass (see Notes)
    vis_activation : torch.nn.modules.activation
        activation function to apply in reconstruct pass to obtain visible unit
        mean field estimates
    vis_sample : torch.distributions or None
        function used for sampling from visible unit mean field estimates
    hid_sample_fn : torch.distributions or None
        function used for sampling from hidden unit mean field estimates
    log_part_fn : torch.nn.functional
        log-partition function for the hidden unit exponential family
        [default: torch.nn.functional.softplus, i.e. binary hidden units]
    v_bias : torch.Tensor
        bias for visible units
    h_bias : torch.Tensor
        bias for hidden units
    output_shape : tuple
        output shape for layer

    Methods
    -------
    make_layer(hidden, activation, pool)
        make forward_layer and reconstruct_layer with hidden, activation,
        pool parameters (see Notes)
    update_layer(hidden=None, activation=None, pool=None)
        update forward_layer and reconstruct_layer with new hidden, activation,
        or pool parameters
    sample_h_given_v(v)
        sample hidden units given visible units
    sample_v_given_h(h)
        sample visible units given hidden units
    sample_h_given_vt(v, t)
        sample hidden units given visible units and top-down input
    contrastive_divergence(v, h)
        compute contrastive divergence for visible and hidden samples
    energy(v, h)
        compute energy for visible and hidden samples
    free_energy(v)
        compute free energy for visible sample
    train(input, optimizer, k=1, monitor_fn=nn.MSELoss(), **kwargs)
        train with contrastive divergence with k gibbs steps

    Notes
    -----
    forward_layer = torch.nn.Sequential(
        (hidden): hidden,
        (activation): activation,
        (pool): pool
    )
    where hidden, activation, and pool are input parameters with torch.nn.module
    type or None at initialization (e.g., hidden = torch.nn.Conv2d(1, 24, 11)).
    If hidden, activation, or pool are None, default is torch.nn.Sequential()

    reconstruct_layer = torch.nn.Sequential(
        (unpool): unpool,
        (hidden_transpose): hidden_transpose,
        (activation): vis_activation
    )
    where unpool is nn.MaxUnpool2d if pool.return_indices = True or
    nn.UpSample(scale_factor=pool.kernel_size) if hasattr(pool, 'kernel_size')
    and nn.Sequential otherwise, hidden_transpose is the transposed operation
    of hidden, and vis_activation is an input parameter at initialization.
    """
    def __init__(self, hidden=None, activation=None, pool=None, sample=None,
                 dropout=None, vis_activation=None, vis_sample=None,
                 log_part_fn=F.softplus, input_shape=None):
        super(RBM, self).__init__(input_shape)
        self.log_part_fn = log_part_fn
        # wrap sample functions in ops.Op
        if torch.typename(vis_sample).startswith('torch.distributions'):
            vis_sample = ops.Op(ops.sample_fn, distr=vis_sample)
        if torch.typename(sample).startswith('torch.distributions'):
            sample = ops.Op(ops.sample_fn, distr=sample)
        # initialize persistent
        self.persistent = None
        # make forward layer
        self.make_layer('forward_layer', hidden=hidden, activation=activation,
                        pool=pool, sample=sample, dropout=dropout)
        # init weights
        self.init_weights(pattern='weight', fn=lambda x: 0.01 * torch.randn_like(x))
        # make reconstruct layer
        self.make_layer('reconstruct_layer', transpose=True, pool=pool,
                        hidden=hidden)
        self.update_layer('reconstruct_layer', activation=vis_activation,
                          sample=vis_sample)
        # init biases
        self.init_weights(pattern='bias', fn=torch.zeros_like)
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set v_bias and h_bias
        self.v_bias = self.hidden_transpose_bias
        self.h_bias = self.hidden_bias

    def hidden_shape(self, input_shape):
        if len(input_shape) == 4:
            v_shp = torch.tensor(input_shape[-2:]).numpy()
            W_shp = torch.tensor(self.hidden_weight.shape[-2:]).numpy()
            img_shape = tuple(v_shp - W_shp + 1)
            hidden_shape = (input_shape[0], self.hidden_weight.shape[0]) + img_shape
        else:
            hidden_shape = (input_shape[0], self.hidden_weight.shape[1])
        return hidden_shape

    def sample(self, x, layer_name, pooled_output=False):
        # get activation
        x_mean = self.apply(x, layer_name, ['activation'])
        # get pooling x_mean, x_sample if rf_pool
        pool_module = self.get_modules(layer_name, ['pool'])
        if len(pool_module) > 0 and hasattr(pool_module[0], 'rfs'):
            if not pooled_output:
                pool_fn = pool_module[0].pool_fn.replace('_pool', '')
            else:
                pool_fn = pool_module[0].pool_fn
            x_mean = pool_module[0](x_mean, pool_fn=pool_fn)
        # sample from x_mean
        x_sample = self.apply(x_mean, layer_name, ['sample'])
        return x_mean, x_sample

    def sample_h_given_v(self, v, pooled_output=False):
        # get hidden output
        pre_act_h = self.apply(v, 'forward_layer', output_module='hidden')
        return (pre_act_h,) + self.sample(pre_act_h, 'forward_layer', pooled_output)

    def sample_v_given_h(self, h):
        # get hidden_transpose output
        pre_act_v = self.apply(h, 'reconstruct_layer', ['hidden_transpose'])
        return (pre_act_v,) + self.sample(pre_act_v, 'reconstruct_layer')

    def sample_h_given_vt(self, v, t, pooled_output=False):
        # get hidden output
        pre_act_h = self.apply(v, 'forward_layer', output_module='hidden')
        # repeat t to add to pre_act_h
        shape = [v_shp//t_shp for (v_shp,t_shp) in zip(pre_act_h.shape,t.shape)]
        t = functions.repeat(t, shape)
        ht = torch.add(pre_act_h, t)
        return (pre_act_h,) + self.sample(ht, 'forward_layer', pooled_output)

    def gibbs_vhv(self, v_sample, k=1):
        for _ in range(k):
            pre_act_h, h_mean, h_sample = self.sample_h_given_v(v_sample)
            pre_act_v, v_mean, v_sample = self.sample_v_given_h(h_sample)
        return pre_act_h, h_mean, h_sample, pre_act_v, v_mean, v_sample

    def gibbs_hvh(self, h_sample, k=1):
        for _ in range(k):
            pre_act_v, v_mean, v_sample = self.sample_v_given_h(h_sample)
            pre_act_h, h_mean, h_sample = self.sample_h_given_v(v_sample)
        return pre_act_v, v_mean, v_sample, pre_act_h, h_mean, h_sample

    def energy(self, v, h):
        #E(v,h) = −hTWv−bTv−cTh
        # reshape v_bias, hid_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        h_dims = tuple([1 for _ in range(h.ndimension()-2)])
        h_bias = torch.reshape(self.h_bias, (1,-1) + h_dims)
        # detach h from graph
        h = h.detach()
        # get Wv
        Wv = self.apply(v, 'forward_layer', output_module='hidden')
        Wv = Wv - h_bias
        # get hWv, bv, ch
        hWv = torch.sum(torch.mul(h, Wv).flatten(1), 1)
        bv = torch.sum(torch.mul(v, v_bias).flatten(1), 1)
        ch = torch.sum(torch.mul(h, h_bias).flatten(1), 1)
        return -hWv - bv - ch

    def free_energy(self, v):
        # reshape v_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # get Wv_b
        Wv_b = self.apply(v, 'forward_layer', output_module='hidden')
        # get vbias, hidden terms
        vbias_term = torch.flatten(v * v_bias, 1)
        vbias_term = torch.sum(vbias_term, 1)
        hidden_term = torch.sum(self.log_part_fn(Wv_b).flatten(1), 1)
        return -hidden_term - vbias_term

    def free_energy_comparison(self, v, valid_v):
        if not torch.is_tensor(valid_v):
            valid_v = iter(valid_v).next()[0]
        return torch.div(torch.mean(self.free_energy(v)),
                         torch.mean(self.free_energy(valid_v)))

    def log_prob(self, v, log_Z=None, **kwargs):
        """
        Log probability of data

        Parameters
        ----------
        v : torch.Tensor or torch.utils.data.dataloader.DataLoader
            data to compute log probability
        log_Z : float or None
            log of the partition function over model (calculated used ais)
            [default: None, log_Z computed by passing kwargs to ais]

        Returns
        -------
        log_p_v : float
            log probability of data

        See also
        --------
        ais : compute log_Z of the model

        References
        ----------
        (Salakhutdinov & Murray 2008)
        """
        # compute log_Z
        if log_Z is None:
            log_Z = self.ais(**kwargs)
        # compute free energy data
        if type(v) is torch.utils.data.dataloader.DataLoader:
            n_batches = len(v)
            # get mean free energy for each batch
            fe = 0.
            for data, _ in v:
                fe += torch.mean(self.free_energy(data))
        else: # get free energy for tensor input
            n_batches = 1.
            fe = torch.mean(self.free_energy(v))
        # return log prob of data
        return -torch.div(fe, n_batches) - log_Z

    def ais(self, m, beta, base_rate, base_log_part_fn=F.softplus):
        """
        Annealed Importance Sampling (AIS) for estimating log(Z) of model

        Parameters
        ----------
        m : int
            number of AIS runs to compute
        beta : list or array-like
            beta values in [0,1] for weighting distributions (see notes)
        base_rate : torch.Tensor
            visible biases for base model (natural parameter of exponential family)
            with base_rate.shape == data[0,None].shape
        base_log_part_fn : torch.nn.functional
            log-partition function for visible units
            [default: torch.nn.functional.softplus, i.e. binary units]

        Returns
        -------
        log_Z_model : float
            estimate of the log of the partition function for the model
            (used in computing log probability of data)

        See also
        --------
        log_prob : estimate log probability of data
        base_rate : estimate base_rate for some binary data

        References
        ----------
        (Salakhutdinov & Murray 2008)
        """
        # repeat base_rate m times
        base_rate_m = functions.repeat(base_rate, (m,))
        # reshape v_bias
        v_dims = tuple([1 for _ in range(base_rate.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # init log_pk (estimated log(Z_model/Z_base))
        log_pk = torch.zeros(m)
        # get v_0 from base_rate_m
        v_k = self.sample(base_rate_m,'reconstruct_layer')[1]
        # get log(p_0(v_1))
        log_pk -= self._ais_free_energy(v_k, beta[0], base_rate)
        # get log(p_k(v_k) and log(p_k(v_k+1)) for each beta in (0, 1)
        for b in beta[1:-1]:
            # get log(p_k(v_k))
            log_pk += self._ais_free_energy(v_k, b, base_rate)
            # sample h
            Wv_b = self.apply(v_k, 'forward_layer', output_module='hidden')
            h = self.sample(Wv_b * b, 'forward_layer')[1]
            # sample v_k+1
            pre_act_v = self.apply(h, 'reconstruct_layer', ['hidden_transpose'])
            v_k = self.sample((1. - b) * base_rate_m + b * pre_act_v,
                              'reconstruct_layer')[1]
            # get log(p_k(v_k+1))
            log_pk -= self._ais_free_energy(v_k, b, base_rate)
        # get log(p_k(v_k))
        log_pk += self._ais_free_energy(v_k, beta[-1], base_rate)
        # get mean across m cases for log AIS ratio of Z_model/Z_base
        r_AIS = torch.logsumexp(log_pk, 0) - np.log(m)
        # get log_Z_base
        base_h = torch.zeros_like(h[0,None])
        log_Z_base = torch.add(torch.sum(base_log_part_fn(base_rate)),
                               torch.sum(self.log_part_fn(base_h)))
        # return estimated log_Z_model
        log_Z_model = r_AIS + log_Z_base
        return log_Z_model

    def _ais_free_energy(self, v, beta, base_rate):
        # reshape v_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # get Wv_b
        Wv_b = self.sample_h_given_v(v)[0]
        # get vbias, hidden terms
        base_term = (1. - beta) * torch.sum(torch.flatten(v * base_rate, 1), 1)
        vbias_term = beta * torch.sum(torch.flatten(v * v_bias, 1), 1)
        hidden_term = torch.sum(self.log_part_fn(beta * Wv_b).flatten(1), 1)
        return base_term + vbias_term + hidden_term

    def base_rate(self, dataloader, lp=5.):
        """
        Base-rate model (for RBMs)

        (Salakhutdinov & Murray 2008)

        NOTE: Currently only for binary data
        """
        b = torch.zeros_like(iter(dataloader).next()[0][0,None])
        n_batches = len(dataloader)
        for data, _ in dataloader:
            b += torch.mean(data, 0, keepdim=True)
        p_b = (b + lp * n_batches) / (n_batches + lp * n_batches)
        return torch.log(p_b) - torch.log(1. - p_b)

    def pseudo_likelihood(self, v):
        """
        Get pseudo-likelihood via randomly flipping bits and measuring free energy

        Parameters
        ----------
        input : torch.tensor
            binary input to obtain pseudo-likelihood

        Returns
        -------
        pl : float
            pseudo-likelihood given input

        Notes
        -----
        A random index in the input image is flipped on each call. Averaging
        over many different indices approximates the pseudo-likelihood.
        """
        n_visible = np.prod(v.shape[1:])
        # get free energy for input
        xi = torch.round(v)
        fe_xi = self.free_energy(xi)
        # flip bit and get free energy
        xi_flip = torch.flatten(xi, 1)
        bit_idx = torch.randint(xi_flip.shape[1], (xi.shape[0],))
        xi_idx = np.arange(xi.shape[0])
        xi_flip[xi_idx, bit_idx] = 1. - xi_flip[xi_idx, bit_idx]
        xi_flip = torch.reshape(xi_flip, v.shape)
        fe_xi_flip = self.free_energy(xi_flip)
        # return pseudo-likelihood
        return torch.mean(n_visible * torch.log(torch.sigmoid(fe_xi_flip - fe_xi)))

    def show_negative(self, v, k=1, n_images=-1, img_shape=None, figsize=(5,5),
                      cmap=None):
        """
        #TODO:WRITEME
        """
        with torch.no_grad():
            # if neg is None:
            if hasattr(self, 'persistent'):
                neg = self.persistent
            else:
                neg = self.gibbs_vhv(v, k=k)[4]
        # reshape, permute for plotting
        if img_shape:
            v = torch.reshape(v, (-1,1) + img_shape)
            neg = torch.reshape(neg, (-1,1) + img_shape)
        v = torch.squeeze(v.permute(0,2,3,1), -1).numpy()
        neg = torch.squeeze(neg.permute(0,2,3,1), -1).numpy()
        v = functions.normalize_range(v, dims=(1,2))
        neg = functions.normalize_range(neg, dims=(1,2))
        # plot negatives
        if n_images == -1:
            n_images = neg.shape[0]
        fig, ax = plt.subplots(n_images, 2, figsize=figsize)
        ax = np.reshape(ax, (n_images, 2))
        for r in range(n_images):
            ax[r,0].axis('off')
            ax[r,1].axis('off')
            ax[r,0].imshow(v[np.minimum(r, v.shape[0]-1)], cmap=cmap)
            ax[r,1].imshow(neg[r], cmap=cmap)
        plt.show()
        return fig

    def train(self, input, k=1, optimizer=None, monitor_loss=nn.MSELoss(),
              **kwargs):
        """
        #TODO:WRITEME
        """
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        if optimizer:
            optimizer.zero_grad()
        with torch.no_grad():
            # positive phase
            pre_act_ph, ph_mean, ph_sample = self.sample_h_given_v(input)
            # persistent
            if self.persistent is not None:
                ph_sample = self.sample_h_given_v(self.persistent)[2]
                self.hidden_weight.add_(self.persistent_weights)
            elif kwargs.get('persistent') is not None:
                self.persistent = kwargs.get('persistent')
                ph_sample = self.sample_h_given_v(self.persistent)[2]
                self.persistent_weights = torch.zeros_like(self.hidden_weight,
                                                           requires_grad=True)
                self.persistent_weights = nn.Parameter(self.persistent_weights)
                if optimizer and kwargs.get('persistent_lr'):
                    optimizer.add_param_group({'params': self.persistent_weights,
                                               'momentum': 0.,
                                               'lr': kwargs.get('persistent_lr')})
            # dropout
            ph_sample = self.apply(ph_sample, 'forward_layer', ['dropout'])
            # negative phase
            [
                pre_act_nv, nv_mean, nv_sample, pre_act_nh, nh_mean, nh_sample
            ] = self.gibbs_hvh(ph_sample, k=k)
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
        # compute loss, pass backward through gradients
        loss = torch.sub(torch.mean(self.free_energy(input)),
                         torch.mean(self.free_energy(nv_sample)))
        hidsize = torch.tensor(self.hidden_shape(input.shape)).unsqueeze(-1)
        hidsize = torch.prod(hidsize[2:])
        loss = torch.div(loss, hidsize)
        loss.backward()
        # update persistent weights
        with torch.no_grad():
            if self.persistent is not None:
                self.persistent = nv_sample
                self.persistent_weights.mul_(0.95)
                self.persistent_weights.grad = self.hidden_weight.grad
        # update parameters
        if optimizer:
            optimizer.step()
        # monitor loss
        with torch.no_grad():
            if monitor_loss is not None:
                out = monitor_loss(input, nv_mean)
            else:
                out = loss
        return out.item()

class CRBM(RBM):
    """
    #TODO:WRITEME
    """#TODO: Update to use apply for activation and sample functions
    def __init__(self, top_down=None, y_activation_fn=None, y_sample_fn=None,
                 **kwargs):
        super(CRBM, self).__init__(**kwargs)
        self.y_activation_fn = y_activation_fn
        self.y_sample_fn = y_sample_fn
        # update forward layer
        self.update_layer('forward_layer', transpose=True, top_down=top_down)
        # init weights
        self.init_weights(pattern='weight', fn=lambda x: 0.01 * torch.randn_like(x))
        # make reconstruct layer
        self.make_layer('reconstruct_layer', top_down=top_down)
        self.update_layer('reconstruct_layer', transpose=True,
                          pool=kwargs.get('pool'), hidden=kwargs.get('hidden'))
        self.update_layer('reconstruct_layer', activation=vis_activation)
        # init biases
        self.init_weights(pattern='bias', fn=torch.zeros_like)
        # remove top_down_bias
        self.reconstruct_layer.top_down.bias = None
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set v_bias, h_bias, y_bias
        self.v_bias = self.hidden_transpose_bias
        self.y_bias = self.top_down_transpose_bias

    def sample_h_given_vy(self, v, y):
        # get top down input from y
        Uy = self.apply(y, 'reconstruct_layer', ['top_down'])
        return self.sample_h_given_vt(v, Uy)

    def sample_y_given_h(self, h):
        pre_act_y = self.apply(h, 'forward_layer', ['top_down_transpose'])
        if self.y_activation_fn:
            y_mean = self.y_activation_fn(pre_act_y)
        else:
            y_mean = pre_act_y
        if self.y_sample_fn:
            y_sample = self.hid_sample_fn(y_mean).sample()
        else:
            y_sample = y_mean
        return pre_act_y, y_mean, y_sample

    def sample_y_given_v(self, v):
        h_sample = self.sample_h_given_v(v)[-1]
        return self.sample_y_given_h(h_sample)

    def gibbs_vhv(self, v_sample, y_sample=None, k=1):
        for _ in range(k):
            if y_sample is not None:
                h_outputs = self.sample_h_given_vy(v_sample, y_sample)
            else:
                h_outputs = self.sample_h_given_v(v_sample)
            h_sample = h_outputs[-1]
            v_outputs = self.sample_v_given_h(h_sample)
            y_outputs = self.sample_y_given_h(h_sample)
        return h_outputs + v_outputs + y_outputs

    def gibbs_hvh(self, h_sample, k=1):
        for _ in range(k):
            v_outputs = self.sample_v_given_h(h_sample)
            v_sample = v_outputs[-1]
            y_outputs = self.sample_y_given_h(h_sample)
            y_sample = y_outputs[-1]
            h_outputs = self.sample_h_given_vy(v_sample, y_sample)
        return v_outputs + y_outputs + h_outputs

    def energy(self, v, h, y):
        #E(v,y,h) = −hTWv−bTv−cTh−dTy−hTUy
        # reshape v_bias, hid_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        h_dims = tuple([1 for _ in range(h.ndimension()-2)])
        h_bias = torch.reshape(self.hidden_bias, (1,-1) + h_dims)
        y_dims = tuple([1 for _ in range(y.ndimension() - 2)])
        y_bias = torch.reshape(self.y_bias, (1,-1) + y_dims)
        # detach h from graph
        h = h.detach()
        # get Wv, Uy
        Wv = self.apply(v, 'forward_layer', output_module='hidden')
        Uy = self.apply(y, 'reconstruct_layer', ['top_down'])
        Wv = Wv - h_bias
        # get hWv, hUy, bv, ch, dy
        hWv = torch.sum(torch.mul(h, Wv).flatten(1), 1)
        hUy = torch.sum(torch.mul(h, Uy).flatten(1), 1)
        bv = torch.sum(torch.mul(v, v_bias).flatten(1))
        ch = torch.sum(torch.mul(h, h_bias).flatten(1), 1)
        dy = torch.sum(torch.mul(y, y_bias).flatten(1), 1)
        return -hWv - bv - ch - dy - hUy

    def free_energy(self, v, y):
        # reshape v_bias, hid_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # get Wv_Uy_b
        Wv_Uy_b = self.sample_h_given_vy(v, y)[0]
        # get vbias, hidden terms
        vbias_term = torch.flatten(v * v_bias, 1)
        vbias_term = torch.sum(vbias_term, 1)
        hidden_term = torch.sum(self.log_part_fn(Wv_Uy_b).flatten(1), 1)
        return -hidden_term - vbias_term

    def train(self, inputs, k=1, optimizer=None, monitor_fn=nn.MSELoss(),
              **kwargs):
        """
        #TODO:WRITEME
        """
        # get input, top_down
        input, top_down = inputs[:2]
        if optimizer:
            optimizer.zero_grad()
        with torch.no_grad():
            # positive phase
            pre_act_ph, ph_mean, ph_sample = self.sample_h_given_vy(input, top_down)
            # persistent
            if self.persistent is not None:
                ph_sample = self.persistent
                self.hidden_weight.add_(self.persistent_weights)
            elif kwargs.get('persistent') is not None:
                self.persistent = kwargs.get('persistent')
                ph_sample = self.persistent
                self.persistent_weights = torch.zeros_like(self.hidden_weight,
                                                           requires_grad=True)
                self.persistent_weights = nn.Parameter(self.persistent_weights)
                if optimizer and kwargs.get('persistent_lr'):
                    optimizer.add_param_group({'params': self.persistent_weights,
                                               'momentum': 0.,
                                               'lr': kwargs.get('persistent_lr')})
            # dropout
            ph_sample = self.apply(ph_sample, 'forward_layer', ['dropout'])
            # negative phase
            [
                pre_act_nv, nv_mean, nv_sample,
                pre_act_ny, ny_mean, ny_sample,
                pre_act_nh, nh_mean, nh_sample,
            ] = self.gibbs_hvh(ph_sample, k=k)
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
        # compute loss, pass backward through gradients
        loss = torch.sub(torch.mean(self.free_energy(input)),
                         torch.mean(self.free_energy(nv_sample)))
        hidsize = torch.tensor(self.hidden_shape(input.shape)).unsqueeze(-1)
        hidsize = torch.prod(hidsize[2:])
        loss = torch.div(loss, hidsize)
        loss.backward()
        # update persistent weights
        with torch.no_grad():
            if self.persistent is not None:
                self.persistent = nh_sample
                self.persistent_weights.mul_(0.95)
                self.persistent_weights.grad = self.hidden_weight.grad
        # update parameters
        if optimizer:
            optimizer.step()
        # monitor loss
        with torch.no_grad():
            if monitor_loss is not None:
                out = monitor_loss(input, nv_mean)
            else:
                out = loss
        return out.item()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
