from collections import OrderedDict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops, pool
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
            if name.find(pattern) >=0:
                with torch.no_grad():
                    new_param = fn(param)
                    param.mul_(0.).add_(new_param)

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
            # if None, set Sequential; if not module, set Op
            if module is None:
                module = nn.Sequential()
            elif not isinstance(module, torch.nn.modules.Module):
                module = ops.Op(module)
            if transpose:
                name = name + '_transpose'
                module = self.transposed_fn(module)
            # add reshape op if linear
            if self.input_shape is None and isinstance(module, torch.nn.Linear):
                reshape_op = ops.Op(ops.flatten_fn(1))
                layer.add_module('reshape_%s' % name, reshape_op)
            if module is not None:
                layer.add_module(name, module)
        # set layer
        setattr(self, layer_name, layer)

    def update_module(self, layer_name, module_name, module):
        self.update_layer(layer_name, **{module_name: module})

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
        if hasattr(fn, 'weight') and isinstance(fn, torch.nn.Conv2d):
            conv_kwargs = functions.get_attributes(fn, ['stride','padding',
                                                        'dilation'])
            transposed_fn = nn.ConvTranspose2d(fn.out_channels, fn.in_channels,
                                               fn.kernel_size, **conv_kwargs)
            transposed_fn.weight = fn.weight
        # transposed linear
        elif hasattr(fn, 'weight') and isinstance(fn, torch.nn.Linear):
            transposed_fn = nn.Linear(fn.out_features, fn.in_features)
            transposed_fn.weight = nn.Parameter(fn.weight.t())
        elif hasattr(fn, 'weight') and not isinstance(fn, pool.Pool):
            #TODO: how to use transposed version of fn implicitly
            raise Exception('%a type not understood' % (fn))
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

    def get_modules_by_type(self, layer_name, module_types=(),
                            module_str_types=[]):
        # ensure module_types is tuple and module_str_types is list
        if isinstance(module_types, type):
            module_types = (module_types,)
        module_types = tuple(module_types)
        if isinstance(module_str_types, str):
            module_str_types = [module_str_types]
        module_str_types = list(module_str_types)
        # get modules in module_types or module_str_types
        modules = []
        layer = getattr(self, layer_name)
        for name, module in layer.named_children():
            # if isinstance or endswith type name, append
            m_type = torch.typename(module)
            if isinstance(module, module_types) or \
               any([m_type.endswith(s) for s in module_str_types]):
                modules.append(module)
        return modules

    def get_modules_by_attr(self, layer_name, attributes):
        # ensure attributes is list
        if isinstance(attributes, str):
            attributes = [attributes]
        attributes = list(attributes)
        # get modules with attributes
        modules = []
        layer = getattr(self, layer_name)
        for name, module in layer.named_children():
            # if any hasattr, append
            if any([hasattr(module, attr) for attr in attributes]):
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

    def show_weights(self, field='hidden_weight', img_shape=None, transpose=False,
                     **kwargs):
        """
        Show weights as image

        Parameters
        ----------
        field : str
            name of weights (e.g., getattr(module, field)) to show
        img_shape : tuple or None
            image shape to reshape to [default: None]
        transpose : boolean
            True/False transpose weights after reshaping [default: False]
        **kwargs : **dict
            keyword arguments passed to `rf_pool.utils.visualize.show_images`

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure with imgs.shape[0] (or imgs.shape[0] + imgs.shape[1]) images
            with axes shape (n_rows, n_cols)

        See Also
        --------
        rf_pool.utils.visualize.show_images
        """
        # get field for weights
        if not hasattr(self, field):
            raise Exception('attribute %a not found' % (field))
        w = getattr(self, field).clone().detach()
        # reshape
        if img_shape is not None:
            w = w.reshape(-1, *img_shape)
        # transpose
        if transpose:
            w = w.transpose(-2,-1)
        # plot weights
        return visualize.show_images(w, img_shape=img_shape, **kwargs)

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
    cat_dim : int, optional
        dimension to concatenate branch outputs along
        Note branch outputs must match shape on all dims other than `cat_dim`.
        [default: None, does not concatenate branches]
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
                 cat_dim=None, output_names=None, input_shape=None):
        super(Branch, self).__init__(input_shape)
        self.branches = branches
        self.n_branches = len(branches)
        if branch_names is not None:
            assert len(branch_names) == len(branches)
        self.branch_names = branch_names
        self.branch_shapes = branch_shapes
        self.cat_dim = cat_dim
        self.n_cat = [None,] * self.n_branches
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
        # if not list, copy n_branches times
        if isinstance(input, torch.Tensor):
            input = [input] * self.n_branches
        # for each branch, pass input
        outputs = []
        self.reconstruct_shape = []
        for i, (name, branch) in enumerate(self.forward_layer.named_children()):
            if self.input_shape:
                self.reconstruct_shape.append(input[i].shape)
                input[i] = torch.reshape(input[i], self.input_shape)
            if len(module_names) > 0 and name not in module_names:
                continue
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            outputs.append(branch.forward(input[i], **mod_kwargs))
            if self.branch_shapes:
                outputs[-1] = torch.reshape(outputs[-1], self.branch_shapes[i])
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(outputs[-1])
        # concatenate along cat_dim, record n_cat per output
        if self.cat_dim is not None:
            self.n_cat = [output.shape[self.cat_dim] for output in outputs]
            outputs = torch.cat(outputs, self.cat_dim)
        if self.output_names is not None:
            outputs = OrderedDict([(k,v)
                                   for k, v in zip(self.output_names, outputs)])
        return outputs

    def reconstruct(self, input, module_names=[], output={}, **kwargs):
        # reconstruct by passing input through each branch
        outputs = []
        start_slice = 0
        for i, (name, branch) in enumerate(self.forward_layer.named_children()):
            if len(module_names) > 0 and name not in module_names:
                continue
            # slice cat_dim using n_cat
            if self.cat_dim is not None:
                assert self.n_cat[i] is not None
                end_slice = start_slice + self.n_cat[i]
                input_i = input.transpose(0, self.cat_dim)[start_slice:end_slice]
                input_i = input_i.transpose(0, self.cat_dim)
                start_slice = end_slice
            else:
                input_i = input
            # get module-specific kwargs
            mod_kwargs = kwargs.get(name)
            if mod_kwargs is None:
                mod_kwargs = {}
            # apply module
            outputs.append(branch.reconstruct(input_i, **mod_kwargs))
            if self.reconstruct_shape:
                outputs[-1] = torch.reshape(outputs[-1], self.reconstruct_shape[i])
            # set to output
            if type(output.get(name)) is list:
                output.get(name).append(outputs[-1])
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
        # initialize persistent
        self.persistent = None
        # make forward layer
        self.make_layer('forward_layer', hidden=hidden, activation=activation,
                        pool=pool, sample=sample, dropout=dropout)
        # init weights
        self.init_weights(pattern='weight', fn=lambda x: 0.01*torch.randn_like(x))
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

    def untie_weights(self, pattern=''):
        """
        Untie `reconstruct_layer` weights from `forward_layer` weights

        Parameters
        ----------
        pattern : str
            pattern of weights in `reconstruct_layer` to be untied
            [default: '', all weights in `reconstruct_layer` untied]

        Returns
        -------
        None
        """
        for name, param in self.reconstruct_layer.named_parameters():
            if name.find(pattern) >=0:
                self.register_parameter(name.replace('.','_'),
                                        torch.nn.Parameter(param.clone()))

    def hidden_shape(self, input_shape):
        if len(input_shape) == 4:
            v_shp = torch.as_tensor(input_shape[-2:]).numpy()
            W_shp = torch.as_tensor(self.hidden_weight.shape[-2:]).numpy()
            img_shape = tuple(v_shp - W_shp + 1)
            hidden_shape = (input_shape[0], self.hidden_weight.shape[0])+img_shape
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
        """
        Energy function for RBMs with binary hidden units and binary or
        Gaussian visible units (assumes sigma=1 for Gaussian units)

        Parameters
        ----------
        v : torch.Tensor
            visible unit configuration with shape (batch,ch) or (batch,ch,h,w)
        h : torch.Tensor
            hidden unit configuration with shape (batch,ch) or (batch,ch,h,w)

        Returns
        -------
        e : torch.Tensor
            energy for given visible/hidden unit configuration

        Notes
        -----
        Bernoulli RBM energy: `E(v, h) = -hWv - bv - ch`
        Gaussian-Bernoulli RBM energy: `E(v, h) = -hWv/s - (v - b)**2/2s**2 - ch`
        where `v` is visible units, `h` is hidden units, `W` is weight matrix,
        `b` is visible unit biases, `c` is hidden unit biases, and `s` is sigma
        for Gaussian visible units (assumed to be 1 here).
        """
        # reshape v_bias, hid_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # detach h from graph
        h = h.detach()
        # get Wv (include h_bias)
        Wv = self.apply(v, 'forward_layer', output_module='hidden')
        # get hWv, bv
        hWv = torch.sum(torch.mul(h, Wv).flatten(1), 1)
        if torch.all(torch.ge(v, 0.)):
            bv = torch.sum(torch.mul(v, v_bias).flatten(1), 1)
        else:
            bv = torch.sum((torch.pow(v - v_bias, 2) / 2.).flatten(1), 1)
        return -hWv - bv

    def free_energy(self, v):
        # reshape v_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        # get Wv_b
        Wv_b = self.apply(v, 'forward_layer', output_module='hidden')
        # get vbias, hidden terms
        vbias_term = torch.sum(torch.flatten(v * v_bias, 1), 1)
        hidden_term = torch.sum(self.log_part_fn(Wv_b).flatten(1), 1)
        return -hidden_term - vbias_term

    def free_energy_comparison(self, v, valid_v):
        if not torch.is_tensor(valid_v):
            valid_v = iter(valid_v).next()[0]
        return torch.div(torch.mean(self.free_energy(v)),
                         torch.mean(self.free_energy(valid_v)))

    def log_prob(self, v, n_data=-1, log_Z=None, **kwargs):
        """
        Log probability of data

        Parameters
        ----------
        v : torch.Tensor or torch.utils.data.dataloader.DataLoader
            data to compute log probability
        n_data : int
            number of data points (or batches if v is `DataLoader`) within v
            to compute log probability [default: -1, all data in v]
        log_Z : float or None
            log of the partition function over model (calculated using `ais`)
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
        # set n_data
        n_data = n_data if n_data > 0 else len(v)
        # compute free energy data
        if isinstance(v, torch.utils.data.dataloader.DataLoader):
            # set number of batches to n_data
            n_batches = n_data
            # get mean free energy for each batch
            fe = 0.
            for i, (data, _) in enumerate(v):
                if i > n_batches:
                    break
                fe += torch.mean(self.free_energy(data))
        else: # get free energy for tensor input
            n_batches = 1.
            fe = torch.mean(self.free_energy(v[:n_data]))
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
            beta values in [0,1] for weighting distributions (see References)
        base_rate : torch.Tensor
            visible biases for base model (natural parameter of exponential family)
            with `base_rate.shape == data[0,None].shape`
        base_log_part_fn : torch.nn.functional
            log-partition function for visible units
            [default: `torch.nn.functional.softplus`, i.e. binary units]

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
        # return estimated log_Z_model log(Z_B/Z_A * Z_A)
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
        This likelihood estimate is only appropriate for binary data.

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
        return torch.mean(n_visible * torch.log(torch.sigmoid(fe_xi_flip-fe_xi)))

    def gaussian_pseudo_likelihood(self, v):
        n_visible = np.prod(v.shape[1:])
        # get free energy for input
        fe_xi = self.free_energy(v)
        # flip random bit
        xi_flip = torch.flatten(v, 1)
        bit_idx = torch.randint(xi_flip.shape[1], (v.shape[0],))
        xi_idx = np.arange(v.shape[0])
        xi_flip[xi_idx, bit_idx] = -xi_flip[xi_idx, bit_idx]
        fe_xi_flip = self.free_energy(xi_flip.reshape(v.shape))
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

    def train(self, input, k=1, optimizer=None, monitor_loss=None,
              **kwargs):
        """
        Train RBM with given optimizer and k Gibbs sampling steps

        Parameters
        ----------
        input : torch.Tensor
            input data
        k : int
            number of Gibbs sampling steps to perform [default: 1]
        optimizer : torch.optim
            optimizer used to update parameres in RBM
            [default: None, parameters are not updated]
        monitor_loss : torch.nn.modules.loss or rf_pool.losses
            loss function used only for monitoring loss (i.e., not used to
            update parameters), called as `monitor_loss(input, nv_mean)` where
            nv_mean is the probability of a visible unit being turned on during
            reconstruction after Gibbs sampling in the negative phase.
            [default: None, difference in free energy of positive/negative phases]

        Optional kwargs
        persistent : torch.Tensor
            persistent chains with shape (n_chains, *input.shape[1:]) used as
            fantasy particles to maintain Gibbs sampling across data points.
            This parameter should only be set for the first input. See Notes.
            [default: None, no persistent chains used]
        persistent_lr : float
            learning rate used for persistent weights (if persistent is not None)
            [default: None, persistent weights updated with learning rate in
             optimizer]

        Returns
        -------
        loss : float
            value of loss returned from `monitor_loss` function call or from
            `contrastive_divergence` function call if monitor_loss is None.

        Notes
        -----
        When persistent chains are used, persistent weights are initialized as
        zeros with shape `hidden_weight.shape`. On each call, the persistent
        weights are added to `hidden_weight` prior to the negative phase and
        are decayed by 0.95 and updated with `hidden_weight.grad`. This is known
        as using ``Fast weights`` for persistent contrsative divergence learning.
        See references below for more information on RBMs and persistent
        contrastive divergence.

        References
        ----------
        Hinton, G. E. (2002). Training products of experts by minimizing
        contrastive divergence. Neural computation, 14(8), 1771-1800.

        Tieleman, T., & Hinton, G. (2009, June). Using fast weights to improve
        persistent contrastive divergence. In Proceedings of the 26th Annual
        International Conference on Machine Learning (pp. 1033-1040).
        """
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        if optimizer:
            optimizer.zero_grad()
        with torch.no_grad():
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
            else:
                # positive phase without persistent
                ph_sample = self.sample_h_given_v(input)[2]
            # dropout
            ph_sample = self.apply(ph_sample, 'forward_layer', ['dropout'])
            # negative phase
            nv_mean, nv_sample = self.gibbs_hvh(ph_sample, k=k)[1:3]
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
        # compute loss, pass backward through gradients
        loss = torch.sub(torch.mean(self.free_energy(input)),
                         torch.mean(self.free_energy(nv_sample)))
        hidsize = torch.as_tensor(self.hidden_shape(input.shape)).unsqueeze(-1)
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
        # if persistent reshape input and nv_mean
        if self.persistent is not None:
            warnings.filterwarnings("ignore", message="Using a target size")
            input = torch.unsqueeze(input, 1)
            nv_mean = torch.unsqueeze(nv_mean, 0)
        # monitor loss
        with torch.no_grad():
            if monitor_loss is not None:
                out = monitor_loss(input, nv_mean)
            else:
                out = loss
        # reset default warning
        warnings.filterwarnings("default", message="Using a target size")
        return out.item()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
