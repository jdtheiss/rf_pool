import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .utils import functions

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
    init_weights(suffix='weight', fn=torch.randn_like)
        initialze weights for parameter names that end with suffix using fn
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

    def init_weights(self, suffix='weight', fn=torch.randn_like):
        for name, param in self.named_parameters():
            with torch.no_grad():
                if name.endswith(suffix):
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
        elif hasattr(fn, 'kernel_size'): #TODO: how to determine if pool
            transposed_fn = nn.Upsample(scale_factor=fn.kernel_size)
        else:
            transposed_fn = fn
        return transposed_fn

    def get_module_names(self, layer, module_name=None):
        module_names = []
        for name, _ in layer.named_children():
            module_names.append(name)
            if module_name and name == module_name:
                break
        return module_names

    def get_modules(self, layer, module_names):
        modules = []
        for name, module in layer.named_children():
            if name in module_names:
                modules.append(module)
        return modules

    def apply_modules(self, layer, input, module_names):
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
        module_names = self.get_module_names(self.forward_layer, module_name)
        outputs = []
        for input in inputs:
            output = self.apply_modules(self.forward_layer, input, module_names)
            if type(output) is list:
                output = torch.cat([torch.flatten(o) for o in output])
            outputs.append(output)
        return torch.mul(loss_fn(*outputs, **kwargs), cost)

    def sparsity(self, input, target, cost=1., module_name=None):
        # (SparseRBM; Lee et al., 2008)
        if module_name:
            module_names = self.get_module_names(self.forward_layer, module_name)
            activity = self.apply_modules(self.forward_layer, input, module_names)
        else:
            activity = input
        q = torch.mean(activity.transpose(1,0).flatten(1), -1)
        p = torch.as_tensor(target, dtype=activity.dtype)
        sparse_cost =  q - p
        sparse_cost.mul_(cost)
        self.hidden_bias.grad += sparse_cost

    def show_weights(self, field='hidden_weight', img_shape=None,
                     figsize=(5, 5), cmap=None):
        """
        #TODO:WRITEME
        """
        # get field for weights
        if not hasattr(self, field):
            raise Exception('attribute ' + field + ' not found')
        w = getattr(self, field).clone().detach()
        if w.shape[1] > 3:
            w = torch.flatten(w, 0, 1).unsqueeze(1)
        # get columns and rows
        n_cols = np.ceil(np.sqrt(w.shape[0])).astype('int')
        n_rows = np.ceil(w.shape[0] / n_cols).astype('int')
        # init figure and axes
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        ax = np.reshape(ax, (n_rows, n_cols))
        # plot weights
        cnt = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if cnt >= w.shape[0]:
                    w_n = torch.zeros_like(w[0])
                else:
                    w_n = w[cnt].detach()
                if img_shape:
                    w_n = torch.reshape(w_n, (-1,) + img_shape)
                w_n = torch.squeeze(w_n.permute(1,2,0), -1).numpy()
                w_n = functions.normalize_range(w_n, dims=(0,1))
                ax[r,c].axis('off')
                ax[r,c].imshow(w_n, cmap=cmap)
                cnt += 1
        plt.show()
        return fig

class FeedForward(Module):
    """
    #TODO:WRITEME
    """
    def __init__(self, input_shape=None, **kwargs):
        super(FeedForward, self).__init__(input_shape)
        # build layer
        self.make_layer('forward_layer', **kwargs)
        # init weights, biases
        self.init_weights(suffix='weight', fn=torch.randn_like)
        self.init_weights(suffix='bias', fn=torch.zeros_like)
        # link parameters
        self.link_parameters(self.forward_layer)

class Branch(Module):
    """
    #TODO:WRITEME
    """
    def __init__(self, branches, branch_shapes=None, cat_output=False,
                 input_shape=None):
        super(Branch, self).__init__(input_shape)
        self.branches = branches
        self.branch_shapes = branch_shapes
        self.cat_output = cat_output
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
        return outputs

    def reconstruct(self, input, names=[]):
        outputs = []
        for name, branch in self.forward_layer.named_children():
            if len(names) == 0 or name in names:
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
        # init weights, biases
        self.init_weights(suffix='weight', fn=torch.randn_like)
        self.init_weights(suffix='bias', fn=torch.zeros_like)
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
                if type(control_out) is not list:
                    control_out = [control_out]
            elif control_out is not None:
                input = module(input, *control_out)
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
    vis_bias : torch.Tensor
        bias for visible units
    hid_bias : torch.Tensor
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
    energy(v)
        compute energy for visible sample
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
                 vis_activation_fn=None, vis_sample_fn=None, hid_sample_fn=None,
                 input_shape=None):
        super(RBM, self).__init__(input_shape)
        self.vis_activation_fn = vis_activation_fn
        self.vis_sample_fn = vis_sample_fn
        self.hid_activation_fn = activation
        self.hid_sample_fn = hid_sample_fn
        # initialize persistent
        self.persistent = None
        # make forward layer
        self.make_layer('forward_layer', hidden=hidden, activation=activation,
                        pool=pool, dropout=dropout)
        # init weights
        self.init_weights(suffix='weight', fn=torch.randn_like)
        # make reconstruct layer
        self.make_layer('reconstruct_layer', transpose=True, pool=pool,
                        hidden=hidden)
        self.update_layer('reconstruct_layer', activation=self.vis_activation_fn)
        # init biases
        self.init_weights(suffix='bias', fn=torch.zeros_like)
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set vis_bias and hid_bias
        self.vis_bias = self.hidden_transpose_bias
        self.hid_bias = self.hidden_bias

    def sample_h_given_v(self, v):
        # apply each non-pooling module (unless rf_pool)
        pre_act_h = self.apply_modules(self.forward_layer, v, ['hidden'])
        h_mean = self.apply_modules(self.forward_layer, pre_act_h, ['activation'])
        # apply pool module if rf_pool type
        pool_module = self.get_modules(self.forward_layer, ['pool'])[0]
        if torch.typename(pool_module).find('layers') >= 0:
            h_mean = pool_module.apply(h_mean)[0]
        # sample from h_mean
        if self.hid_sample_fn:
            h_sample = self.hid_sample_fn(probs=h_mean).sample()
        else:
            h_sample = h_mean
        return pre_act_h, h_mean, h_sample

    def sample_v_given_h(self, h):
        # apply each non-pooling module
        pre_act_v = self.apply_modules(self.reconstruct_layer, h, ['hidden_transpose'])
        v_mean = self.apply_modules(self.reconstruct_layer, pre_act_v, ['activation'])
        # sample from v_mean
        if self.vis_sample_fn:
            v_sample = self.vis_sample_fn(probs=v_mean).sample()
        else:
            v_sample = v_mean
        return pre_act_v, v_mean, v_sample

    def sample_h_given_vt(self, v, t):
        # apply each module, add t instead of pooling (unless rf_pool)
        pre_act_h = self.apply_modules(self.forward_layer, v, ['hidden'])
        shape = [v_shp//t_shp for (v_shp,t_shp) in zip(pre_act_h.shape,t.shape)]
        t = functions.repeat(t, shape)
        h_mean = self.apply_modules(self.forward_layer, torch.add(pre_act_h, t),
                                   ['activation'])
        # apply pool module if rf_pool type
        pool_module = self.get_modules(self.forward_layer, ['pool'])[0]
        if torch.typename(pool_module).find('layers') >= 0:
            h_mean = pool_module.apply(h_mean)[0]
        # sample from h_mean
        if self.hid_sample_fn:
            h_sample = self.hid_sample_fn(probs=h_mean).sample()
        else:
            h_sample = h_mean
        return pre_act_h, h_mean, h_sample

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

    def contrastive_divergence(self, pv, ph, nv, nh):
        # get sizes to normalize params
        batch_size = torch.as_tensor(pv.shape[0], dtype=pv.dtype)
        # compute contrastive_divergence for conv layer
        if pv.ndimension() == 4:
            v_shp = torch.as_tensor(pv.shape[-2:], dtype=pv.dtype)
            W_shp = torch.as_tensor(self.hidden_weight.shape[-2:], dtype=pv.dtype)
            hidsize = torch.prod(v_shp - W_shp + 1)
            # compute vishidprods, hidact, visact
            posprods = torch.conv2d(pv.transpose(1,0),
                                    ph.transpose(1,0)).transpose(1,0)
            negprods = torch.conv2d(nv.transpose(1,0),
                                    nh.transpose(1,0)).transpose(1,0)
            vishidprods = torch.div(posprods - negprods, (batch_size * hidsize))
            hidact = torch.mean(ph - nh, dim=(0,2,3))
            visact = torch.mean(pv - nv, dim=(0,2,3))
        # compute contrastive_divergence for fc layer
        else:
            posprods = torch.matmul(pv.t(), ph)
            negprods = torch.matmul(nv.t(), nh)
            vishidprods = torch.div(posprods - negprods, batch_size)
            hidact = torch.mean(ph - nh, dim=0)
            visact = torch.mean(pv - nv, dim=0)
        return {'hidden_weight':vishidprods, 'hid_bias':hidact, 'vis_bias':visact}

    def update_grads(self, grad):
        assert type(grad) is dict
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                if name in grad.keys():
                    param.grad.sub_(grad[name])

    def show_negative(self, v, k=1, img_shape=None, figsize=(5,5), cmap=None):
        """
        #TODO:WRITEME
        """
        # gibbs sample
        with torch.no_grad():
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
        fig, ax = plt.subplots(v.shape[0], 2, figsize=figsize)
        ax = np.reshape(ax, (v.shape[0], 2))
        for r in range(v.shape[0]):
            ax[r,0].axis('off')
            ax[r,1].axis('off')
            ax[r,0].imshow(v[r], cmap=cmap)
            ax[r,1].imshow(neg[r], cmap=cmap)
        plt.show()
        return fig

    def train(self, input, k=1, monitor_fn=nn.MSELoss(), optimizer=None, **kwargs):
        """
        #TODO:WRITEME
        """
        if optimizer:
            optimizer.zero_grad()
        with torch.no_grad():
            # positive phase
            pre_act_ph, ph_mean, ph_sample = self.sample_h_given_v(input)
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
                optimizer.add_param_group({'params': self.persistent_weights})
            # dropout
            ph_sample = self.apply_modules(self.forward_layer, ph_sample,
                                           ['dropout'])
            # negative phase
            [
                pre_act_nv, nv_mean, nv_sample, pre_act_nh, nh_mean, nh_sample
            ] = self.gibbs_hvh(ph_sample, k=k)
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
            # compute loss with contrastive_divergence
            grads = self.contrastive_divergence(input, ph_mean, nv_mean, nh_mean)
            self.update_grads(grads)
            sum_grads = [torch.sum(torch.abs(g)) for g in grads.values()]
            loss = torch.sum(torch.stack(sum_grads))
            # update persistent weights
            if self.persistent is not None:
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
        if monitor_fn:
            out = monitor_fn(input, nv_mean)
        else:
            out = loss
        return out.item()

class CRBM(RBM):
    """
    #TODO:WRITEME
    """
    def __init__(self, top_down=None, y_activation_fn=None, y_sample_fn=None,
                 **kwargs):
        super(CRBM, self).__init__(**kwargs)
        self.y_activation_fn = y_activation_fn
        self.y_sample_fn = y_sample_fn
        # update forward layer
        self.update_layer('forward_layer', transpose=True, top_down=top_down)
        # init weights
        self.init_weights(suffix='weight', fn=torch.randn_like)
        # make reconstruct layer
        self.make_layer('reconstruct_layer', top_down=top_down)
        self.update_layer('reconstruct_layer', transpose=True,
                          pool=kwargs.get('pool'), hidden=kwargs.get('hidden'))
        self.update_layer('reconstruct_layer', activation=self.vis_activation_fn)
        # init biases
        self.init_weights(suffix='bias', fn=torch.zeros_like)
        # remove top_down_bias
        self.reconstruct_layer.top_down.bias = None
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set vis_bias, hid_bias, y_bias
        self.vis_bias = self.hidden_transpose_bias
        self.hid_bias = self.hidden_bias
        self.y_bias = self.top_down_transpose_bias

    def sample_h_given_vy(self, v, y):
        # get top down input from y
        Uy = self.apply_modules(self.reconstruct_layer, y, ['top_down'])
        return self.sample_h_given_vt(v, Uy)

    def sample_y_given_h(self, h):
        pre_act_y = self.apply_modules(self.forward_layer, h, ['top_down_transpose'])
        if self.y_activation_fn:
            y_mean = self.y_activation_fn(pre_act_y)
        else:
            y_mean = pre_act_y
        if self.y_sample_fn:
            y_sample = self.hid_sample_fn(probs=y_mean).sample()
        else:
            y_sample = y_mean
        return pre_act_y, y_mean, y_sample

    def sample_y_given_v(self, v):
        h_sample = self.sample_h_given_v(v)[-1]
        return self.sample_y_given_h(h_sample)

    def gibbs_vhv(self, v_sample, y_sample, k=1):
        for _ in range(k):
            pre_act_h, h_mean, h_sample = self.sample_h_given_vy(v_sample, y_sample)
            v_outputs = self.sample_v_given_h(h_sample)
            y_outputs = self.sample_y_given_h(h_sample)
        return (pre_act_h, h_mean, h_sample) + v_outputs + y_outputs

    def gibbs_hvh(self, h_sample, k=1):
        for _ in range(k):
            pre_act_v, v_mean, v_sample = self.sample_v_given_h(h_sample)
            pre_act_y, y_mean, y_sample = self.sample_y_given_h(h_sample)
            h_outputs = self.sample_h_given_vy(v_sample, y_sample)
        return (pre_act_v, v_mean, v_sample, pre_act_y, y_mean, y_sample) + h_outputs

    def energy(self, v, y, h):
        #E(v,y,h) = −hTWv−bTv−cTh−dTy−hTUy
        # get dims for each input
        v_dims = tuple([1 for _ in range(v.ndimension() - 2)])
        h_dims = tuple([1 for _ in range(h.ndimension() - 2)])
        y_dims = tuple([1 for _ in range(y.ndimension() - 2)])
        # detach h from graph
        h = h.detach()
        # get Wv, Uy
        Wv = self.apply_modules(self.forward_layer, v, ['hidden'])
        with torch.no_grad():
            Wv = Wv - self.hid_bias.reshape((1,-1) + h_dims)
        Uy = self.apply_modules(self.reconstruct_layer, y, ['top_down'])
        # flatten if ndim > 2
        if len(h_dims) > 0:
            h = torch.flatten(h, 1)
            Wv = torch.flatten(Wv, 1)
            Uy = torch.flatten(Uy, 1)
        # get hWv, hUy
        hWv = torch.sum(torch.mul(h, Wv), 1)
        hUy = torch.sum(torch.mul(h, Uy), 1)
        # get bv, ch, dy
        bv = torch.sum(torch.mul(v, self.vis_bias.reshape((1,-1) + v_dims)).flatten(1))
        ch = torch.sum(torch.mul(h, self.hid_bias.reshape((1,-1) + h_dims)).flatten(1))
        dy = torch.sum(torch.mul(y, self.y_bias.reshape((1,-1) + y_dims)).flatten(1))
        return -hWv - bv - ch - dy - hUy

    def contrastive_divergence(self, pv, ph, nv, nh):
        # get sizes to normalize params
        batch_size = torch.as_tensor(pv.shape[0], dtype=pv.dtype)
        # compute contrastive_divergence for conv layer
        if pv.ndimension() == 4:
            v_shp = torch.as_tensor(pv.shape[-2:], dtype=pv.dtype)
            W_shp = torch.as_tensor(self.hidden_weight.shape[-2:], dtype=pv.dtype)
            hidsize = torch.prod(v_shp - W_shp + 1)
            # compute vishidprods, hidact, visact
            posprods = torch.conv2d(pv.transpose(1,0),
                                    ph.transpose(1,0)).transpose(1,0)
            negprods = torch.conv2d(nv.transpose(1,0),
                                    nh.transpose(1,0)).transpose(1,0)
            vishidprods = torch.div(posprods - negprods, (batch_size * hidsize))
            hidact = torch.mean(ph - nh, dim=(0,2,3))
            visact = torch.mean(pv - nv, dim=(0,2,3))
        # compute contrastive_divergence for fc layer
        else:
            posprods = torch.matmul(pv.t(), ph)
            negprods = torch.matmul(nv.t(), nh)
            vishidprods = torch.div(posprods - negprods, batch_size)
            hidact = torch.mean(ph - nh, dim=0)
            visact = torch.mean(pv - nv, dim=0)
        return vishidprods, hidact, visact

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
                optimizer.add_param_group({'params': self.persistent_weights})
            # dropout
            ph_sample = self.apply_modules(self.forward_layer, ph_sample,
                                           ['dropout'])
            # negative phase
            [
                pre_act_nv, nv_mean, nv_sample,
                pre_act_ny, ny_mean, ny_sample,
                pre_act_nh, nh_mean, nh_sample,
            ] = self.gibbs_hvh(ph_sample, k=k)
            # persistent
            if self.persistent is not None:
                self.hidden_weight.sub_(self.persistent_weights)
            # compute loss with contrastive_divergence
            grads = {}
            prods = self.contrastive_divergence(input, ph_mean, nv_mean, nh_mean)
            grads.update({'hidden_weight': prods[0], 'hid_bias': prods[1],
                          'vis_bias': prods[2]})
            y_prods = self.contrastive_divergence(top_down, ph_mean, ny_mean, nh_mean)
            grads.update({'top_down_transpose_weight': y_prods[0],
                          'y_bias': y_prods[2]})
            self.update_grads(grads)
            sum_grads = [torch.sum(torch.abs(g)) for g in grads.values()]
            loss = torch.sum(torch.stack(sum_grads))
        # # compute loss E(pv, py, ph) - E(nv, ny, nh)
        # loss = torch.sub(torch.mean(self.energy(input, top_down, ph_mean)),
        #                  torch.mean(self.energy(nv_sample, ny_sample, nh_mean)))
        # loss.backward()
        # update persistent weights
        if self.persistent is not None:
            with torch.no_grad():
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
        if monitor_fn:
            out = monitor_fn(input, nv_mean)
        else:
            out = loss
        return out.item()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
