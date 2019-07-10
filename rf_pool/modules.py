import numpy as np
import matplotlib.pyplot as plt
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

    Methods
    -------
    output_shape(input_shape)
        return output_shape based on given input_shape
    get_modules(names)
        return modules from forward_layer or reconstruct_layer with given names
    link_parameters(layer, layer_name)
        register parameters from layer in self with appended layer_name
    init_weights(pattern='weight', fn=torch.randn_like)
        initialze weights for parameter names that end with pattern using fn
    make_layer(**kwargs)
        initialize forward_layer from keyword arguments (e.g.,
        hidden=torch.nn.Conv2d(1, 24, 11), activation=torch.nn.ReLU)
        Note: None is set to torch.nn.Sequential() by default.
    update_layer(**kwargs)
        update layer modules initialized by make_layer (see make_layer)
    add_loss(inputs, loss_fn, module_name, **kwargs)
        return loss from loss_fn(outputs, **kwargs) where outputs is a list of
        outputs from passing each input in inputs through forward_layer until
        module_name
    apply_modules()
        #TODO:WRITEME
    forward()
        #TODO:WRITEME
    reconstruct()
        #TODO:WRITEME
    train()
        #TODO:WRITEME
    show_weights()
        #TODO:WRITEME
    """
    def __init__(self, input_shape=None):
        super(Module, self).__init__()
        self.input_shape = input_shape
        self.reconstruct_shape = input_shape
        self.forward_layer = nn.Sequential()
        self.reconstruct_layer = nn.Sequential()

    def __call__(self, *args):
        return self.forward(*args)

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
        for key, value in kwargs.items():
            if value is None:
                kwargs.update({key: nn.Sequential()})

        # update layers
        self.update_layer(layer_name, transpose=transpose, **kwargs)

    def update_layer(self, layer_name, transpose=False, **kwargs):
        # get layer
        if hasattr(self, layer_name):
            layer = getattr(self, layer_name)
        else:
            layer = nn.Sequential()
        # update layer
        for key, value in kwargs.items():
            if transpose:
                key = key + '_transpose'
                value = self.transposed_fn(value)
            if value is not None:
                layer.add_module(key, value)
        # set layer
        setattr(self, layer_name, layer)

    def insert_layer(self, layer_name, idx, transpose=False, **kwargs):
        # get layer
        layer = nn.Sequential()
        if hasattr(self, layer_name):
            orig_layer = getattr(self, layer_name)
            mods = dict([(k, v) for k, v in orig_layer.named_children()])
        else:
            mods = {}
        # set pre-index modules
        for i, (key, value) in enumerate(mods.items()):
            if i < idx:
                layer.add_module(key, value)
        # set index modules
        for key, value in kwargs.items():
            if transpose:
                key = key + '_transpose'
                value = self.transposed_fn(value)
            if value is not None:
                layer.add_module(key, value)
        # set post-index modules
        for i, (key, value) in enumerate(mods.items()):
            if i >= idx:
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

    def get_module_names(self, layer_name, module_name=None):
        module_names = []
        layer = getattr(self, layer_name)
        for name, _ in layer.named_children():
            module_names.append(name)
            if module_name and name == module_name:
                break
        return module_names

    def get_modules(self, layer_name, module_names):
        modules = []
        layer = getattr(self, layer_name)
        for name, module in layer.named_children():
            if name in module_names:
                modules.append(module)
        return modules

    def apply_modules(self, input, layer_name, module_names):
        layer = getattr(self, layer_name)
        for i, (name, module) in enumerate(layer.named_children()):
            if name in module_names:
                if i==0 and layer == self.forward_layer:
                    if self.input_shape:
                        input = torch.reshape(input, self.input_shape)
                input = module(input)
                if i==len(layer) and layer == self.reconstruct_layer:
                    if self.reconstuct_shape:
                        input = torch.reshape(input, self.reconstruct_shape)
        return input

    def apply_module(self, input, layer_name, module_name, **kwargs):
        # use apply function of module with kwargs
        module = self.get_modules(layer_name, [module_name])
        if len(module) > 0:
            output = module[0].apply(input, **kwargs)
        else:
            output = input
        return output

    def forward(self, input):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        return self.forward_layer(input)

    def reconstruct(self, input):
        output = self.reconstruct_layer(input)
        if self.reconstruct_shape:
            output = torch.reshape(output, self.reconstruct_shape)
        return output

    def train(self, input, label, loss_fn, optimizer=None, **kwargs):
        if optimizer:
            optimizer.zero_grad()
        # get output and loss
        output = self.forward(input)
        loss = loss_fn(output, label)
        # additional loss
        if 'add_loss' in kwargs.keys():
            added_loss = self.add_loss(input, **kwargs.get('add_loss'))
            loss = loss + added_loss
        # sparsity
        if 'sparsity' in kwargs.keys():
            self.sparsity(input, **kwargs.get('sparsity'))
        # backprop
        loss.backward()
        if optimizer:
            optimizer.step()
        return loss.item()

    def add_loss(self, inputs, loss_fn, module_name=None, cost=1., **kwargs):
        """
        #TODO:WRITEME
        """
        module_names = self.get_module_names('forward_layer', module_name)
        outputs = []
        for input in inputs:
            output = self.apply_modules(input, 'forward_layer', module_names)
            if type(output) is list:
                output = torch.cat([torch.flatten(o) for o in output])
            outputs.append(output)
        return torch.mul(loss_fn(*outputs, **kwargs), cost)

    def sparsity(self, input, target, cost=1., module_name=None, l2_reg=0.):
        # (SparseRBM; Lee et al., 2008)
        if module_name:
            module_names = self.get_module_names('forward_layer', module_name)
            activity = self.apply_modules(input, 'forward_layer', module_names)
        else:
            activity = input
        q = torch.mean(activity.transpose(1,0).flatten(1), -1)
        p = torch.as_tensor(target, dtype=activity.dtype)
        sparse_cost =  q - p
        sparse_cost.mul_(cost)
        self.h_bias.grad += sparse_cost
        self.hidden_weight.grad += (l2_reg * self.hidden_weight)

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
    #TODO:WRITEME
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
    #TODO:WRITEME
    """
    def __init__(self, branches, branch_shapes=None, cat_output=False,
                 output_names=None, input_shape=None):
        super(Branch, self).__init__(input_shape)
        self.branches = branches
        self.branch_shapes = branch_shapes
        self.cat_output = cat_output
        self.output_names = output_names
        for i, branch in enumerate(self.branches):
            self.forward_layer.add_module('branch_'+str(i), branch)

    def output_shape(self, input_shape):
        outputs = self.forward(torch.zeros(input_shape))
        return [output.shape for output in outputs]

    def forward(self, input, names=[]):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        outputs = []
        for i, (name, branch) in enumerate(self.forward_layer.named_children()):
            if len(names) == 0 or name in names:
                outputs.append(branch.forward(input))
                if self.branch_shapes:
                    outputs[-1] = torch.reshape(outputs[-1], self.branch_shapes[i])
        if self.cat_output:
            outputs = torch.cat(outputs, 1)
        if self.output_names is not None:
            outputs = dict([(k,v) for k, v in zip(self.output_names, outputs)])
        return outputs

    def reconstruct(self, input, names=[]):
        outputs = []
        for name, branch in self.forward_layer.named_children():
            if len(names) == 0 or name in names:
                #TODO: slice input channels for each branch
                output = branch.reconstruct(input)
                if self.reconstruct_shape:
                    output = torch.reshape(output, self.reconstruct_shape)
                outputs.append(output)
        if self.cat_output:
            outputs = torch.cat(outputs, 1)
        return outputs

class Control(Module):
    """
    #TODO:WRITEME
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

    def forward(self, input):
        if self.input_shape:
            self.reconstruct_shape = input.shape
            input = torch.reshape(input, self.input_shape)
        # apply module
        control_out = None
        for name, module in self.forward_layer.named_children():
            if name == 'control':
                control_out = module(input)
            elif control_out is not None:
                if type(control_out) is list:
                    input = module(input, *control_out)
                elif type(control_out) is dict:
                    input = module(input, **control_out)
                else:
                    input = module(input, control_out)
            else:
                input = module(input)
        return input

class RBM(Module):
    """
    Restricted Boltzmann Machine module (convolutional or fully-connected)

    Attributes
    ----------
    forward_layer : torch.nn.Sequential
        functions to apply in forward pass (see Notes)
    reconstruct_layer : torch.nn.Sequential
        functions to apply in reconstruct pass (see Notes)
    vis_activation_fn : torch.nn.modules.activation
        activation function to apply in reconstruct pass to obtain visible unit
        mean field estimates
    vis_sample_fn : torch.distributions or None
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
        (activation): vis_activation_fn
    )
    where unpool is nn.MaxUnpool2d if pool.return_indices = True or
    nn.UpSample(scale_factor=pool.kernel_size) if hasattr(pool, 'kernel_size')
    and nn.Sequential otherwise, hidden_transpose is the transposed operation
    of hidden, and vis_activation_fn is an input parameter at initialization.
    """
    def __init__(self, hidden=None, activation=None, pool=None, dropout=None,
                 vis_activation_fn=None, vis_sample_fn=None,
                 hid_sample_fn=None, log_part_fn=F.softplus,
                 input_shape=None):
        super(RBM, self).__init__(input_shape)
        self.vis_activation_fn = vis_activation_fn
        self.vis_sample_fn = vis_sample_fn
        self.hid_activation_fn = activation
        self.hid_sample_fn = hid_sample_fn
        self.log_part_fn = log_part_fn
        # wrap sample functions in ops.Op
        if torch.typename(self.vis_sample_fn).startswith('torch.distributions'):
            self.vis_sample_fn = ops.Op(ops.sample_fn, distr=self.vis_sample_fn)
        if torch.typename(self.hid_sample_fn).startswith('torch.distributions'):
            self.hid_sample_fn = ops.Op(ops.sample_fn, distr=self.hid_sample_fn)
        # initialize persistent
        self.persistent = None
        # make forward layer
        self.make_layer('forward_layer', hidden=hidden, activation=activation,
                        pool=pool, sample=self.hid_sample_fn, dropout=dropout)
        # init weights
        self.init_weights(pattern='weight', fn=lambda x: 0.01 * torch.randn_like(x))
        # make reconstruct layer
        self.make_layer('reconstruct_layer', transpose=True, pool=pool,
                        hidden=hidden)
        self.update_layer('reconstruct_layer', activation=self.vis_activation_fn,
                          sample=self.vis_sample_fn)
        # init biases
        self.init_weights(pattern='bias', fn=torch.zeros_like)
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set v_bias and h_bias
        self.v_bias = self.hidden_transpose_bias
        self.h_bias = self.hidden_bias

    def sample(self, x, layer_name):
        # get activation
        x_mean = self.apply_modules(x, layer_name, ['activation'])
        # sample from h_mean
        x_sample = self.apply_modules(x_mean, layer_name, ['sample'])
        # get pooling h_mean, h_sample if rf_pool
        pool_output = self.apply_module(x_mean, layer_name, 'pool')
        if len(pool_output) == 3:
            x_mean, x_sample = pool_output[1:]
        return x_mean, x_sample

    def sample_h_given_v(self, v):
        # get hidden output
        pre_act_h = self.apply_modules(v, 'forward_layer', ['hidden'])
        return (pre_act_h,) + self.sample(pre_act_h, 'forward_layer')

    def sample_v_given_h(self, h):
        # get hidden_transpose output
        pre_act_v = self.apply_modules(h, 'reconstruct_layer', ['hidden_transpose'])
        return (pre_act_v,) + self.sample(pre_act_v, 'reconstruct_layer')

    def sample_h_given_vt(self, v, t):
        # get hidden output
        pre_act_h = self.apply_modules(v, 'forward_layer', ['hidden'])
        # repeat t to add to pre_act_h
        shape = [v_shp//t_shp for (v_shp,t_shp) in zip(pre_act_h.shape,t.shape)]
        t = functions.repeat(t, shape)
        return (pre_act_h,) + self.sample(torch.add(pre_act_h, t), 'forward_layer')

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
        #E(v,y,h) = −hTWv−bTv−cTh
        # reshape v_bias, hid_bias
        v_dims = tuple([1 for _ in range(v.ndimension()-2)])
        v_bias = torch.reshape(self.v_bias, (1,-1) + v_dims)
        h_dims = tuple([1 for _ in range(h.ndimension()-2)])
        h_bias = torch.reshape(self.h_bias, (1,-1) + h_dims)
        # detach h from graph
        h = h.detach()
        # get Wv
        Wv = self.apply_modules(v, 'forward_layer', ['hidden'])
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
        Wv_b = self.sample_h_given_v(v)[0]
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
        return torch.div(fe, n_batches) - log_Z

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
            visible biases for base model (zeros for uniform distribution)
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
            Wv_b = self.apply_modules(v_k,'forward_layer',['hidden'])
            h = self.sample(Wv_b * b, 'forward_layer')[1]
            # sample v_k+1
            pre_act_v = self.apply_modules(h,'reconstruct_layer',['hidden_transpose'])
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
            b += torch.mean(data, 0)
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
        # return log(fe_xi_flip) - log(fe_xi)
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

    def train(self, input, k=1, monitor_fn=nn.MSELoss(), optimizer=None, **kwargs):
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
                if optimizer:
                    optimizer.add_param_group({'params': self.persistent_weights})
                    optimizer.param_groups[-1].update({'momentum': 0.})
                    persistent_lr = kwargs.get('persistent_lr')
                    if persistent_lr is not None:
                        optimizer.param_groups[-1].update({'lr': persistent_lr})
            # dropout
            ph_sample = self.apply_modules(ph_sample, 'forward_layer', ['dropout'])
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
        loss.backward()
        # update persistent weights
        with torch.no_grad():
            if self.persistent is not None:
                self.persistent = nv_sample
                self.persistent_weights.mul_(0.95)
                self.persistent_weights.grad = self.hidden_weight.grad
        # compute additional loss from add_loss
        if kwargs.get('add_loss'):
            if kwargs.get('add_loss').get('module_name'):
                added_loss = self.add_loss(input, **kwargs.get('add_loss'))
            else:
                added_loss = self.add_loss(ph_mean, **kwargs.get('add_loss'))
            added_loss.backward()
        # sparsity
        if kwargs.get('sparsity'):
            if kwargs.get('sparsity').get('module_name'):
                self.sparsity(input, **kwargs.get('sparsity'))
            else:
                self.sparsity(ph_mean, **kwargs.get('sparsity'))
        # update parameters
        if optimizer:
            optimizer.step()
        # monitor loss
        with torch.no_grad():
            if type(monitor_fn) is dict:
                monitor_kwargs = monitor_fn.copy()
                fn = monitor_kwargs.pop('fn')
                if 'v' not in monitor_kwargs:
                    monitor_kwargs.update({'v': input})
                out = torch.mean(fn(**monitor_kwargs))
            elif monitor_fn is not None:
                out = monitor_fn(input, nv_mean)
            else:
                out = loss
        return out.item()

class CRBM(RBM):
    """
    #TODO:WRITEME
    """#TODO: Update to use apply_modules for activation and sample functions
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
        self.update_layer('reconstruct_layer', activation=self.vis_activation_fn)
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
        Uy = self.apply_modules(y, 'reconstruct_layer', ['top_down'])
        return self.sample_h_given_vt(v, Uy)

    def sample_y_given_h(self, h):
        pre_act_y = self.apply_modules(h, 'forward_layer', ['top_down_transpose'])
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
        Wv = self.apply_modules(v, 'forward_layer', ['hidden'])
        Uy = self.apply_modules(y, 'reconstruct_layer', ['top_down'])
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

    def train(self, inputs, k=1, monitor_fn=nn.MSELoss(), optimizer=None,
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
                if optimizer:
                    optimizer.add_param_group({'params': self.persistent_weights})
                    optimizer.param_groups[-1].update({'momentum': 0.})
            # dropout
            ph_sample = self.apply_modules(ph_sample, 'forward_layer', ['dropout'])
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
        loss = torch.sub(torch.mean(self.free_energy(input, top_down)),
                         torch.mean(self.free_energy(nv_sample, ny_sample)))
        loss.backward()
        # update persistent weights
        with torch.no_grad():
            if self.persistent is not None:
                self.persistent = nh_sample
                self.persistent_weights.mul_(0.95)
                self.persistent_weights.grad = self.hidden_weight.grad
        # compute additional loss from add_loss
        if kwargs.get('add_loss'):
            if kwargs.get('add_loss').get('module_name'):
                added_loss = self.add_loss(input, **kwargs.get('add_loss'))
            else:
                added_loss = self.add_loss(ph_mean, **kwargs.get('add_loss'))
            added_loss.backward()
        # sparsity
        if kwargs.get('sparsity'):
            if kwargs.get('sparsity').get('module_name'):
                self.sparsity(input, **kwargs.get('sparsity'))
            else:
                self.sparsity(ph_mean, **kwargs.get('sparsity'))
        # update parameters
        if optimizer:
            optimizer.step()
        # monitor loss
        with torch.no_grad():
            if type(monitor_fn) is dict:
                monitor_kwargs = monitor_fn.copy()
                fn = monitor_kwargs.pop('fn')
                if 'v' not in monitor_kwargs:
                    monitor_kwargs.update({'v': input})
                out = torch.mean(fn(**monitor_kwargs))
            elif monitor_fn is not None:
                out = monitor_fn(input, nv_mean)
            else:
                out = loss
        return out.item()

class ssRBM(RBM):
    """
    #TODO:WRITEME
    """
    def __init__(self, **kwargs):
        super(ssRBM, self).__init__(**kwargs)
        raise NotImplementedError

if __name__ == '__main__':
    import doctest
    doctest.testmod()
