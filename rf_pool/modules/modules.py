from collections import OrderedDict
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rf_pool import pool
from rf_pool.modules import ops
from rf_pool.solver import build
from rf_pool.utils import functions, rbm_utils, visualize

def get_pad_fn(pad_type='zero'):
    pad_type = pad_type.lower() if pad_type else None
    if pad_type == 'reflect':
        pad_fn = nn.ReflectionPad2d
    elif pad_type == 'replicate':
        pad_fn = nn.ReplicationPad2d
    elif pad_type in ['zero',None]:
        pad_fn = nn.ZeroPad2d
    else:
        raise Exception('unknown pad_type "%s"' % pad_type)
    return pad_fn

class Module(nn.Module):
    """
    Base class for modules
    """
    def __init__(self, input_shape=None, **kwargs):
        super(Module, self).__init__()
        self.input_shape = input_shape
        self.reconstruct_shape = input_shape
        self.build(**kwargs)

    def init_weights(self, pattern='weight', fn='xavier_normal_', **kwargs):
        """
        Initialize weights using torch.nn.init functions applied to parameter
        with given pattern

        Parameters
        ----------
        pattern : str
            pattern of parameter to initialize [default: 'weight']
        fn : str
            name of `torch.nn.init` function to be applied
        **kwargs : **dict
            keyword arguments passed to init function

        Returns
        -------
        None
        """
        init_fn = getattr(nn.init, fn)
        for name, param in self.named_parameters():
            if name.find(pattern) != -1:
                init_fn(param, **kwargs)

    def output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        return self.forward(torch.zeros(input_shape)).shape

    def build(self, **kwargs):
        # set _modules
        new_mods = OrderedDict((k, build.build_module(v)) for k, v in kwargs.items())
        self._modules = new_mods

    def insert_module(self, idx, **kwargs):
        # set modules in order
        mods = list(self.named_children())
        new_mods = list(kwargs.items())
        new_mods.reverse()
        [mods.insert(idx, new_mod) for new_mod in new_mods]
        # update modules
        self._modules = OrderedDict(mods)

    def remove_module(self, module_name):
        # get modules other than module_name
        mods = [(k, v) for k, v in self.named_children() if k != module_name]
        # update modules
        self._modules = OrderedDict(mods)

    def apply_modules(self, input, module_names=[], output={},
                      output_module=None, **kwargs):
        """
        Apply modules with module-specific kwargs and/or collect outputs in dict

        Parameters
        ----------
        input : torch.Tensor
            input passed through modules
        module_names : list, optional
            names of modules in Sequential to apply (i.e. only these modules)
            [default: [], apply all modules in Sequential]
        output : dict, optional
            dictionary like {module_name: []} to be updated with specific results
            [default: {}, will not set outputs to dictionary]
        output_module : str, optional
            name of module to stop passing input through Sequential
            [default: None, apply all modules]
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
        >>> net = FeedForward(hidden=torch.nn.Conv2d(1,16,5),
                              activation=torch.nn.ReLU(),
                              pool=torch.nn.MaxPool2d(2)))
        >>> saved_outputs = {'activation': []}
        >>> output = net.apply_modules(torch.rand(1,1,6,6),
                                       module_names=['hidden','activation','pool'],
                                       output=saved_outputs)
        >>> print(output.shape, saved_outputs.get('activation')[0].shape)
        torch.Size([1, 16, 1, 1]) torch.Size([1, 16, 2, 2])
        """
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        for name, module in self.named_children():
            if len(module_names) > 0 and name not in module_names:
                continue
            mod_kwargs = kwargs.get(name, {})
            input = module(input, **mod_kwargs)
            if isinstance(output.get(name), list):
                output.get(name).append(input)
            if name == output_module:
                break
        return input

    def forward(self, *args, **kwargs):
        return self.apply_modules(*args, **kwargs)

    def train_module(self, input, label, loss_fn, optimizer=None, **kwargs):
        if optimizer:
            optimizer.zero_grad()
        # get output and loss
        output = self.forward(input)
        loss = loss_fn(output, label)
        # backprop
        loss.backward()
        if optimizer:
            optimizer.step()
        return loss.item()

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
        # update modules
        self.build(**kwargs)

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
        # set branch names
        if branch_names is not None:
            assert len(branch_names) == len(branches)
        else:
            branch_names = ['branch_%d' % i for i in range(self.n_branches)]
        self.branch_names = branch_names
        self.branch_shapes = branch_shapes
        self.cat_dim = cat_dim
        self.n_cat = [None,] * self.n_branches
        self.output_names = output_names
        # build modules
        self.build(**dict((name, branch) for name, branch in zip(branch_names, branches)))

    def output_shape(self, input_shape):
        outputs = self.forward(torch.zeros(input_shape))
        return [output.shape for output in outputs]

    def apply_modules(self, input, module_names=[], output={},
                      output_module=None, **kwargs):
        # if not list, copy n_branches times
        if isinstance(input, torch.Tensor):
            input = [input] * self.n_branches
        # for each branch, pass input
        outputs = []
        for i, (name, branch) in enumerate(self.named_children()):
            # skip if name not in module_names
            if len(module_names) > 0 and name not in module_names:
                continue
            if self.input_shape:
                input[i] = torch.reshape(input[i], self.input_shape)
            # append output
            mod_kwargs = kwargs.get(name)
            outputs.append(branch.forward(input[i], **mod_kwargs))
            if self.branch_shapes:
                outputs[-1] = torch.reshape(outputs[-1], self.branch_shapes[i])
            # append to output dict
            if isinstance(output.get(name), list):
                output.get(name).append(outputs[-1])
            # break if output_module
            if name == output_module:
                break
        # concatenate along cat_dim, record n_cat per output
        if self.cat_dim is not None:
            self.n_cat = [output.shape[self.cat_dim] for output in outputs]
            outputs = torch.cat(outputs, self.cat_dim)
        if self.output_names is not None:
            outputs = OrderedDict((k,v) for k, v in zip(self.output_names, outputs))
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
        # update modules
        self.build(**kwargs)

    def apply_modules(self, input, module_names=[], output={},
                      output_module=None, **kwargs):
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        # init control_out
        control_out = None
        for name, module in self.named_children():
            # skip if not in module_names
            if len(module_names) > 0 and name not in module_names:
                continue
            # apply module based on name/control_out type
            mod_kwargs = kwargs.get(name)
            if name == 'control':
                control_out = module(input, **mod_kwargs)
            elif control_out is not None:
                if isinstance(control_out, list):
                    input = module(input, *control_out, **mod_kwargs)
                elif isinstance(control_out, dict):
                    input = module(input, **control_out, **mod_kwargs)
                else:
                    input = module(input, control_out, **mod_kwargs)
            else:
                input = module(input, **mod_kwargs)
            # append to output dict
            if isinstance(output.get(name), list):
                output.get(name).append(outputs[-1])
            # break if output_module
            if name == output_module:
                break
        return input

class Autoencoder(Module):
    """
    Autoencoder layer with encoder and decoder networks

    Parameters
    ----------
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]
    encoder_modules : dict
        modules for layer like {module_name: module}
    decoder_modules : dict
        modules for layer like {module_name: module}

    Returns
    -------
    None
    """
    def __init__(self, encoder_modules, decoder_modules, input_shape=None):
        super(Autoencoder, self).__init__(input_shape)
        # set encoder
        self.encoder = Module(**encoder_modules)
        # set decoder
        self.decoder = Module(**decoder_modules)

    def encode(self, input):
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        for module in self.encoder.children():
            input = module(input)
        return input

    def decode(self, input):
        for module in self.decoder.children():
            input = module(input)
        return input

    def apply_modules(self, input, module_names=[], output={}, output_module=None, **kwargs):
        if self.input_shape:
            input = torch.reshape(input, self.input_shape)
        for name, module in [*self.encoder.named_children(),
                             *self.decoder.named_children()]:
            # skip if not in module_names
            if len(module_names) > 0 and name not in module_names:
                continue
            # apply module
            mod_kwargs = kwargs.get(name)
            input = module(input, **mod_kwargs)
            # append output
            if isinstance(output.get(name), list):
                output.get(name).append(input)
            # break if output_module
            if name == output_module:
                break
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
        # update modules
        kwargs = kwargs.copy()
        for name, module in kwargs.items():
            if isinstance(module, (str,dict)):
                module = build.build_module(module)
                kwargs.update({name: module})
            if not isinstance(module, nn.Module):
                kwargs.update({name: ops.Op(module)})
        # build
        self.build(**kwargs)

class ConvBlock(Module):
    """
    Convolution-Normalization-Activation block

    Parameters
    ----------
    in_channels : int
        number of channels for input
    out_channels : int
        number of filters in conv layer
    kernel_size : int, tuple
        size of kernel in conv layer [default: 3]
    stride : int, tuple
        stride used in conv layer [default: 1]
    padding : int, tuple
        padding used with `pad_type` (i.e., not used in conv layer)
        [default: 1]
    pad_type : str
        type of pad function to be used with provided `padding`.
        should be one of ['reflect','replicate','zero'] [default: 'zero']
    normalization : nn.modules.normalization or dict, optional
        normalization function to use after conv layer
        [default: nn.InstanceNorm2d]
    activation : nn.modules.activation or dict, optional
        activation function to use after normalization
        [default: nn.LeakyReLU(0.2)]
    pool : nn.modules.pooling or rf_pool.pool or dict, optional
        pooling function to use after activation [default: None]
    transpose : bool
        True/False to use nn.ConvTranspose2d as conv layer
        [default: False, uses nn.Conv2d]
    **kwargs : **dict
        keyword arguments passed to nn.Conv2d or nn.ConvTranspose2d

    Notes
    -----
    If `normalization`, `activation`, or `pool` are dict, the dictionary should
    have a (key, value) pair as (class, args/kwargs) to build the module
    (e.g., {'LeakyReLU': (0.2,)}).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 pad_type='zero', normalization=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.2), pool=None, transpose=False, **kwargs):
        super(ConvBlock, self).__init__(input_shape=kwargs.pop('input_shape', None))
        # set padding
        conv_block = {}
        conv_block.update({'pad': get_pad_fn(pad_type)(padding)})
        # set conv_fn based on transpose
        if transpose:
            conv_fn = nn.ConvTranspose2d
        else:
            conv_fn = nn.Conv2d
        # set conv without padding
        conv_block.update({'conv': conv_fn(in_channels, out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride, padding=0, **kwargs)})
        # set norm, activation, and pooling
        if isinstance(normalization, dict):
            conv_block.update({'norm': normalization})
        elif normalization:
            conv_block.update({'norm': normalization(out_channels)})
        if activation:
            conv_block.update({'act': copy.deepcopy(activation)})
        if pool:
            conv_block.update({'pool': pool})
        # build conv block
        self.build(**conv_block)

class Linear(Module):
    """
    Flatten + Linear layer

    Parameters
    ----------
    in_features : int
        number of input channels
    out_features : int
        number of output channels
    bias : bool
        True/False use bias

    Notes
    -----
    This is a convenience module that combines `nn.Flatten` and `nn.Linear` to
    ensure convolutional outputs are automatically flattened before passing to
    the `nn.Linear` layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # register weight and bias from nn.Linear
        self._linear = nn.Linear(in_features, out_features, bias)
        self.register_parameter('weight', self._linear.weight)
        if self._linear.bias is not None:
            self.register_parameter('bias', self._linear.bias)
        else:
            self.bias = None

    def __repr__(self):
        return self._linear.__repr__()

    def apply_modules(self, *args, **kwargs):
        return self.forward(args[0])

    def forward(self, input):
        input = input.flatten(1)
        output = torch.mul(input.unsqueeze(1), self.weight.unsqueeze(0)).sum(-1)
        if self.bias is not None:
            output = output + self.bias
        return output

class ResNetBlock(Module):
    """
    Residual Network Block
    Two ConvBlocks with output added to input

    Parameters
    ----------
    in_channels : int
        number of channels for input
    kernel_size : int, tuple
        size of kernel in conv layer [default: 3]
    stride : int, tuple
        stride used in conv layer [default: 1]
    padding : int, tuple
        padding used with `pad_type` (i.e., not used in conv layer)
        [default: 1]
    pad_type : str
        type of pad function to be used with provided `padding`.
        should be one of ['reflect','replicate','zero'] [default: 'zero']
    normalization : nn.modules.normalization
        normalization function to use in each conv block
        [default: nn.InstanceNorm2d]
    activation : nn.modules.activation
        activation function used in first conv block
        [default: nn.LeakyReLU(0.2)]
    **kwargs : **dict
        keyword arguments passed to ConvBlock

    Notes
    -----
    If `output_module` is passed during forward call, the input is not added
    residually to the output.
    """
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1,
                 pad_type='zero', normalization=nn.InstanceNorm2d,
                 activation=nn.LeakyReLU(0.2), **kwargs):
        super(ResNetBlock, self).__init__(input_shape=kwargs.pop('input_shape', None))
        # set conv blocks
        conv_0 = ConvBlock(in_channels, in_channels, kernel_size, stride,
                           padding, pad_type, normalization, activation, **kwargs)
        conv_1 = ConvBlock(in_channels, in_channels, kernel_size, stride,
                           padding, pad_type, normalization, None, **kwargs)
        self.build(conv_0=conv_0, conv_1=conv_1)

    def apply_modules(self, input, **kwargs):
        # if output_module, do not add input
        if kwargs.get('output_module'):
            return Module.apply_modules(self, input, **kwargs)
        return input + Module.apply_modules(self, input, **kwargs)

class RBM(Module):
    """
    Restricted Boltzmann Machine module (convolutional or fully-connected)

    Parameters
    ----------
    forward_modules : dict
        modules for forward pass like {module_name: module}
    reconstruct_modules : dict
        modules for reconstruct pass like {module_name: module}
    visible_units : str or nn.Module
        name of distribution for visible units or 'none'
        (i.e., `getattr(rf_pool.ops, visible_units.capitalize())`)
        [default: 'bernoulli']
        if nn.Module, this module is set as sampling function
    hidden_units : str or nn.Module
        name of distribution for hidden units or 'none'
        (i.e., `getattr(rf_pool.ops, visible_units.capitalize())`)
        [default: 'bernoulli']
        if nn.Module, this module is set as sampling function
    tie_weights_module : str or None
        name of module in `forward_modules` and `reconstruct_modules` to tie
        weights (see Notes) [default: 'hidden']
    log_part_fn : torch.nn.functional, optional
        log-partition function for the hidden unit exponential family
        [default: torch.nn.functional.softplus, i.e. for bernoulli hidden units]
    input_shape : list, optional
        shape to which input data should be reshaped
        [default: None, input data not reshaped]

    Methods
    -------
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
    train_layer(input, optimizer, k=1, monitor_fn=nn.MSELoss(), **kwargs)
        train with contrastive divergence with k gibbs steps

    Notes
    -----
    Typically, during training an RBM contains a set of weights which are
    used during inference and transposed during generation. By default (i.e.,
    `tie_weights_module='hidden'`), the `weight` attribute of the module named
    'hidden' in `forward_modules` will be transposed and replace the `weight`
    attribute in the 'hidden' module for `reconstruct_modules`. These weights
    can be "untied" after training using the function `untie_weights`.

    The `weight` attributes of modules in `forward_modules` and `reconstruct_modules`
    will be initialized using `nn.init.normal_(param, 0., 0.01)`, and `bias`
    attributes will be initialized with `nn.init.zeros_(param)`.

    If the distribution for `visible_units` or `hidden_units` requires additional
    keyword arguments, these should be passed during initialization. For example,
    if `visible_units = 'normal'`, a kwarg `visible_scale = 1.` should be provided
    to indicate that the visible units should be sampled using
    `torch.distributions.Normal(x, scale=1.).sample()`.

    See Also
    --------
    rf_pool.ops : RBM sampler operations

    References
    ----------
    Hinton, G. E. (2012). A practical guide to training restricted Boltzmann
    machines. In Neural networks: Tricks of the trade (pp. 599-619).
    Springer, Berlin, Heidelberg.
    """
    def __init__(self, forward_modules, reconstruct_modules,
                 visible_units='bernoulli', hidden_units='bernoulli',
                 tie_weights_module='hidden', log_part_fn=F.softplus,
                 input_shape=None, **kwargs):
        super(RBM, self).__init__(input_shape)
        assert tie_weights_module in forward_modules
        assert tie_weights_module in reconstruct_modules
        # init vars
        self.tie_weights_module = tie_weights_module
        self.log_part_fn = log_part_fn
        self.persistent = None
        # build layers/samplers
        self.build_net(visible_units, hidden_units, forward_modules,
                       reconstruct_modules, **kwargs)
        # set hidden_weight
        self.hidden_weight = getattr(self.forward_layer, tie_weights_module).weight
        # get vis/hid_bias
        self.vis_bias = getattr(self.reconstruct_layer, tie_weights_module).bias
        self.hid_bias = getattr(self.forward_layer, tie_weights_module).bias
        # get pool_defaults
        self._reset_pool_fn()
        # init weights
        self.init_weights(pattern='weight', fn='normal_', std=0.01)
        self.init_weights(pattern='bias', fn='zeros_')
        # tie weights
        self.tie_weights(tie_weights_module)

    def build_net(self, visible_units, hidden_units, forward_modules,
                  reconstruct_modules, **kwargs):
        # parse kwargs for vis/hid samplers
        vis_sample_kwargs = dict((k.replace('visible_', ''), v) for k, v
                                  in kwargs.items() if k.startswith('visible_'))
        hid_sample_kwargs = dict((k.replace('hidden_', ''), v) for k, v
                                  in kwargs.items() if k.startswith('hidden_'))
        # build vis/hid samplers
        if isinstance(visible_units, nn.Module):
            self.vis_sample_fn = visible_units
        elif visible_units.lower() == 'none':
            self.vis_sample_fn = nn.Sequential()
        else:
            self.vis_sample_fn = ops.Sampler(visible_units.capitalize(),
                                             **vis_sample_kwargs)
        if isinstance(hidden_units, nn.Module):
            self.hid_sample_fn = hidden_units
        elif hidden_units.lower() == 'none':
            self.hid_sample_fn = nn.Sequential()
        else:
            self.hid_sample_fn = ops.Sampler(hidden_units.capitalize(),
                                             **hid_sample_kwargs)
        # build layers
        self.forward_layer = Module(**forward_modules)
        self.reconstruct_layer = Module(**reconstruct_modules)

    def tie_weights(self, module_name, param_name='weight'): #TODO: update using pattern
        """
        Tie `reconstruct_layer` parameter to `forward_layer` parameter for
        given module

        Parameters
        ----------
        module_name : str
            name of module in `forward_layer` and `reconstruct_layer` to be tied
        param_name : str
            name of parameter within module to be tied [default: 'weight']
        """
        forward_mod = getattr(self.forward_layer, module_name)
        param = getattr(forward_mod, param_name)

        reconstruct_mod = getattr(self.reconstruct_layer, module_name)
        param = nn.Parameter(param.transpose(0, 1))

        reconstruct_mod.register_parameter(param_name, param)

    def untie_weights(self, module_name, param_name='weight'):
        """
        Untie `reconstruct_layer` parameter from `forward_layer` parameter for
        given module

        Parameters
        ----------
        module_name : str
            name of module in `forward_layer` and `reconstruct_layer` to be tied
        param_name : str
            name of parameter within module to be tied [default: 'weight']
        """
        forward_mod = getattr(self.forward_layer, module_name)
        param = getattr(forward_mod, param_name)

        forward_mod.register_parameter(param_name, nn.Parameter(param.clone()))

        reconstruct_mod = getattr(self.reconstruct_layer, module_name)
        param = getattr(reconstruct_mod, param_name)

        reconstruct_mod.register_parameter(param_name, nn.Parameter(param.clone()))

    def _reset_pool_fn(self):
        # get defaults
        pool_defaults = getattr(self, 'pool_defaults', {})
        # apply function to reset or get defaults
        def fn(mod):
            if hasattr(mod, 'pool_fn'):
                if pool_defaults.get(mod) is None:
                    pool_defaults.update({mod: mod.pool_fn})
                else:
                    mod.pool_fn = pool_defaults.get(mod)
        self.apply(fn)
        # set defaults
        setattr(self, 'pool_defaults', pool_defaults)

    def _update_pool_fn(self, pooled_output=False):
        def fn(mod):
            if hasattr(mod, 'pool_fn'):
                mod.pool_fn = mod.pool_fn.replace('_pool', '')
                if pooled_output and hassatr(pool, mod.pool_fn + '_pool'):
                    mod.pool_fn = mod.pool_fn + '_pool'
        self.apply(fn)

    def sample(self, x, layer_name='forward', pooled_output=False):
        # pass through layer
        self._update_pool_fn(pooled_output=pooled_output)
        x_mean = getattr(self, layer_name)(x)
        self._reset_pool_fn()
        # pass through sampler
        if layer_name == 'forward':
            x_sample = self.hid_sample_fn(x_mean)
        else:
            x_sample = self.vis_sample_fn(x_mean)
        return x_mean, x_sample

    def sample_h_given_v(self, v, pooled_output=False):
        return self.sample(v, 'forward', pooled_output=pooled_output)

    def sample_v_given_h(self, h):
        return self.sample(h, 'reconstruct')

    def gibbs_vhv(self, v_sample, k=1):
        for _ in range(k):
            h_mean, h_sample = self.sample_h_given_v(v_sample)
            v_mean, v_sample = self.sample_v_given_h(h_sample)
        return h_mean, h_sample, v_mean, v_sample

    def gibbs_hvh(self, h_sample, k=1):
        for _ in range(k):
            v_mean, v_sample = self.sample_v_given_h(h_sample)
            h_mean, h_sample = self.sample_h_given_v(v_sample)
        return v_mean, v_sample, h_mean, h_sample

    def energy(self, v, h): #TODO: see if way to generalize
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
        v_bias = torch.reshape(self.vis_bias, (1,-1) + v_dims)
        # detach h from graph
        h = h.detach()
        # get Wv (include h_bias)
        Wv = self.forward(v, output_module=self.tie_weights_module)
        # get hWv, bv
        hWv = torch.sum(torch.mul(h, Wv).flatten(1), 1)
        if torch.all(torch.ge(v, 0.)):
            bv = torch.sum(torch.mul(v, v_bias).flatten(1), 1)
        else:
            bv = torch.sum((torch.pow(v - v_bias, 2) / 2.).flatten(1), 1)
        return -hWv - bv

    def free_energy(self, v): #TODO: update for gaussian visible units
        # reshape v_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.vis_bias, (1,-1) + v_dims)
        # get Wv_b
        Wv_b = self.forward(v, output_module=self.tie_weights_module)
        # get vbias, hidden terms
        vbias_term = torch.sum(torch.flatten(v * v_bias, 1), 1)
        hidden_term = torch.sum(self.log_part_fn(Wv_b).flatten(1), 1)
        return -hidden_term - vbias_term

    def forward(self, input, modules=[], output_module=None):
        # get modules
        if len(modules) == 0:
            modules = self.forward_layer._modules.keys()
        # apply modules
        for name in modules:
            mod = self.forward_layer._modules.get(name)
            if mod is None:
                continue
            input = mod(input)
            if output_module and mod == output_module:
                break
        return input

    def reconstruct(self, input, modules=[], output_module=None):
        # get modules
        if len(modules) == 0:
            modules = self.reconstruct_layer._modules.keys()
        # apply modules
        for name in modules:
            mod = self.reconstruct_layer._modules.get(name)
            if mod is None:
                continue
            input = mod(input)
            if output_module and mod == output_module:
                break
        return input

    def train_module(self, input, label=None, k=1, optimizer=None, **kwargs):
        """
        Train RBM with given optimizer and k Gibbs sampling steps

        Parameters
        ----------
        input : torch.Tensor
            input data
        label : torch.Tensor
            optional label for given input data (i.e. conditional RBM)
            [default: None] #TODO: not currently implemented
        k : int
            number of Gibbs sampling steps to perform [default: 1]
        optimizer : torch.optim
            optimizer used to update parameres in RBM
            [default: None, parameters are not updated]

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
                ph_mean, ph_sample = self.sample_h_given_v(self.persistent)
                self.hidden_weight.add_(self.persistent_weights)
            elif kwargs.get('persistent') is not None:
                self.persistent = kwargs.get('persistent')
                ph_mean, ph_sample = self.sample_h_given_v(self.persistent)
                self.persistent_weights = torch.zeros_like(self.hidden_weight,
                                                           requires_grad=True)
                self.persistent_weights = nn.Parameter(self.persistent_weights)
                if optimizer and kwargs.get('persistent_lr'):
                    optimizer.add_param_group({'params': self.persistent_weights,
                                               'momentum': 0.,
                                               'lr': kwargs.get('persistent_lr')})
            else:
                # positive phase without persistent
                ph_mean, ph_sample = self.sample_h_given_v(input)
            # dropout
            ph_sample = self.forward(ph_sample, ['dropout'])
            # negative phase
            nv_mean, nv_sample, nh_mean = self.gibbs_hvh(ph_sample, k=k)[:3]
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
        # compute loss, pass backward through gradients
        loss = torch.sub(torch.mean(self.energy(input, ph_mean)),
                         torch.mean(self.energy(nv_sample, nh_mean)))
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
        return loss.item()

class Attention(Module):
    """
    Spatial and/or feature-based attention using Gaussian-Bernoulli RBMs to
    learn a prior over spatial locations and/or features

    Spatial attention:
    For spatial attention, each pixel location is viewed as a sum of Gaussians
    with `sigma=\sqrt{in_channels}`. The input to the spatial RBM is therefore a
    vector with height * width dimensions. The RBM learns a prior over the different
    spatial locations and uses this to attend to different regions of the image.

    Feature-based attention:
    For feature-based attention, each feature vector (per pixel location) is
    input to the feature RBM in order to learn a prior over the feature space.

    For both types of attention, since the inputs and predictions are viewed as
    Gaussians, attention is applied via Gaussian multiplication
    (with equal sigmas):

    \mu_{output} = \frac{\mu_{input} + \mu_{attn}}{2}

    If both spatial and feature-based attention are used, `\mu_{attn}` is itself
    the Gaussian multiplication (with equal sigmas) of the spatial and feature
    attention `\mu`. After Gaussian multiplying the attention field with the
    input, layer normalization is applied based on the normalization model of
    attention (Reynolds & Heeger, 2009).

    Parameters
    ----------
    in_channels : int
        number of channels in input tensor
    n_hidden : int
        number of hidden units in RBM for feature-based attention
    k_iter : int
        number of Gibbs samples for training rbm (see RBM class) [default: 1]
    learn_hyperprior : bool
        True/False learn weights for hidden units via SGD to provide task-based
        attention [default: True]
    spatial_attention : bool
        True/False use spatial attention [default: True]
    feature_attention : bool
        True/False use feature-based attention [default: True]
    img_shape : tuple, optional
        expected image shape used to initialize spatial attention RBM
        [default: None, RBM initialized during forward call]

    Notes
    -----
    If `learn_hyperprior is True`, the hidden units are weighted by a vector
    which is updated via gradient descent to attend to task-related features.
    The spatial RBM is built during the first forward pass in order to infer the
    image shape of the input tensor. If subsequent inputs have different image
    shapes, `torch.nn.functional.adaptive_avg_pool2d` is used to interpolate
    between the expected and actual image shape.

    To monitor RBM training losses, set the following in configuration:
    LOG:
        fields: ['spatial_rbm_loss', 'feature_rbm_loss']
        reduce: 'mean'
    """
    def __init__(self, in_channels, n_hidden, k_iter=1, learn_hyperprior=True,
                 spatial_attention=True, feature_attention=True, img_shape=None):
        super(Attention, self).__init__(None)
        self.in_channels = in_channels
        self.n_hidden = n_hidden
        self.k_iter = k_iter
        self.learn_hyperprior = learn_hyperprior
        self.spatial_attention = spatial_attention
        self.feature_attention = feature_attention

        # init feature_rbm
        if feature_attention:
            self.feature_rbm = RBM({'hidden': nn.Conv2d(in_channels, n_hidden, kernel_size=1),
                                    'activation': nn.Sigmoid()},
                                    {'hidden': nn.Conv2d(n_hidden, in_channels, kernel_size=1)},
                                    visible_units='none', hidden_units='bernoulli')

        # init weights
        self.reset_parameters()

        # init spatial_rbm if img_shape given (otherwise build during forward call)
        if spatial_attention and img_shape is not None:
            self.build_spatial_rbm(img_shape)

    def reset_parameters(self):
        # init rbms
        if self.spatial_attention and hasattr(self, 'spatial_rbm'):
            self.spatial_rbm.init_weights(pattern='hidden.weight', fn='normal_',
                                          std=0.001)
        if self.feature_attention:
            self.feature_rbm.init_weights(pattern='hidden.weight', fn='normal_',
                                          std=0.001)
        # init hyperprior
        if self.learn_hyperprior:
            self.spatial_weight = nn.Parameter(torch.Tensor(self.n_hidden).normal_(std=0.02))
            self.feature_weight = nn.Parameter(torch.Tensor(self.n_hidden).normal_(std=0.02))
        else:
            self.spatial_weight = torch.ones(self.n_hidden)
            self.feature_weight = torch.ones(self.n_hidden)

    def build_spatial_rbm(self, img_shape):
        # if already built, return
        if hasattr(self, 'spatial_rbm'):
            return
        # set img shape for future
        self.img_shape = img_shape
        # infer number of input units
        n_input = np.prod(img_shape)
        # build rbm
        self.spatial_rbm = RBM({'hidden': nn.Linear(n_input, self.n_hidden),
                                'activation': nn.Sigmoid()},
                                {'hidden': nn.Linear(self.n_hidden, n_input)},
                                visible_units='none', hidden_units='bernoulli')
        # init weights
        self.reset_parameters()

    def train_rbm(self, i_norm, rbm_name='spatial_rbm'):
        """train specified rbm once using i_norm as input"""
        # train rbm with input
        if not self.training:
            return
        # get rbm
        rbm = getattr(self, rbm_name)
        # train update loss
        loss = rbm.train_module(i_norm.detach(), k=self.k_iter,
                                persistent=torch.zeros_like(i_norm))
        setattr(self, rbm_name + '_loss', loss)

    def attend(self, i_norm, rbm_name='spatial_rbm'):
        """attend using prior from specified rbm"""
        # get rbm and weight
        rbm = getattr(self, rbm_name)
        weight = getattr(self, rbm_name.replace('rbm','weight'))
        # set shape based on i_norm
        shape = [1, -1] + [1,] * (i_norm.dim() - 2)
        # turn grads off
        rbm.requires_grad_(False)
        # get hidden unit sample
        h_sample = rbm.sample_h_given_v(i_norm)[-1]
        # weight hidden units
        weighted_h = torch.mul(h_sample, weight.view(shape))
        # compute weighted prior
        attn = rbm.sample_v_given_h(weighted_h)[-1]
        # turn grad on
        rbm.requires_grad_(True)
        return attn

    def apply_spatial_attention(self, i_norm):
        """train rbm/attend using spatial attention"""
        if not self.spatial_attention:
            return None
        # sum of gaussians with sigma=sqrt(i_norm.size(1))
        img_shape = i_norm.shape[-2:]
        sigma = i_norm.size(1) ** 0.5
        i_norm = i_norm.sum(1, keepdim=True) / sigma
        # build spatial_rbm
        self.build_spatial_rbm(img_shape)
        # if different img_shape than expected, interpolate
        if img_shape != self.img_shape:
            i_norm = F.adaptive_avg_pool2d(i_norm, self.img_shape)
        # flatten i_norm
        i_norm = i_norm.flatten(1)
        # train rbm
        self.train_rbm(i_norm, rbm_name='spatial_rbm')
        # attend
        attn = self.attend(i_norm, rbm_name='spatial_rbm').view(-1, 1, *self.img_shape)
        # interpolate if needed
        if img_shape != self.img_shape:
            attn = F.adaptive_avg_pool2d(attn, img_shape)
        return attn

    def apply_feature_attention(self, i_norm):
        """train rbm/attend using feature attention"""
        if not self.feature_attention:
            return None
        # train rbm
        self.train_rbm(i_norm, rbm_name='feature_rbm')
        # attend
        return self.attend(i_norm, rbm_name='feature_rbm')

    @torch.no_grad()
    def cosine_similarity(self, input, **kwargs):
        """compute cosine similarity between input and attention"""
        # pass through forward and return attention
        kwargs.update({'need_weights': True})
        _, attn = self.forward(input, **kwargs)
        # normalize input
        i_norm = F.layer_norm(input, input.shape[1:]).type(input.dtype)
        # if feature_attention is False, compute across img space
        if kwargs.get('feature_attention') is False or not self.feature_attention:
            i_norm = i_norm.sum(1, keepdim=True) / (i_norm.size(1) ** 0.5)
            return functions.pairwise_cosine_similarity(i_norm.flatten(1), attn.flatten(1))
        # compute cosine similarity
        return functions.pairwise_cosine_similarity(i_norm, attn, axis=1)

    def forward(self, input, **kwargs):
        """train rbms/attend using spatial/feature priors"""
        # normalize input
        i_norm = F.layer_norm(input, input.shape[1:]).type(input.dtype)
        # attend
        spatial_attn = feature_attn = None
        if kwargs.get('spatial_attention') in [None, True]:
            spatial_attn = self.apply_spatial_attention(i_norm)
        if kwargs.get('feature_attention') in [None, True]:
            feature_attn = self.apply_feature_attention(i_norm)
        # gaussian multiplication of attention fields (sigma=1)
        if spatial_attn is not None and feature_attn is not None:
            attn = (spatial_attn + feature_attn) / 2.
        else:
            attn = spatial_attn if spatial_attn is not None else feature_attn
        # gaussian multiplication of attention and input (sigma=1)
        out = torch.add(input, attn) / 2.
        # normalize
        out = F.layer_norm(out, out.shape[1:]).type(input.dtype)
        # return attn weights as well as output
        if kwargs.get('need_weights'):
            return out, attn
        return out

if __name__ == '__main__':
    import doctest
    doctest.testmod()
