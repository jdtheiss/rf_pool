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

    def make_layer(self, **kwargs):
        # init forward_layer and reconstruct_layer
        self.forward_layer = nn.Sequential()
        self.reconstruct_layer = nn.Sequential()

        # set None to nn.Sequential()
        for key, value in kwargs.items():
            if value is None:
                kwargs.update({key: nn.Sequential()})

        # update layers
        self.update_layer(**kwargs)

    def update_layer(self, **kwargs):
        # update forward_layer
        for key, value in kwargs.items():
            if value is not None:
                self.forward_layer.add_module(key, value)

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

    def add_loss(self, inputs, loss_fn, module_name=None, **kwargs):
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
        return loss_fn(*outputs, **kwargs)

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

    def show_weights(self, img_shape=None, figsize=(5, 5), cmap=None):
        """
        #TODO:WRITEME
        """
        if not hasattr(self, 'hidden_weight'):
            raise Exception('attribute "hidden_weight" not found')
        w = self.layers[layer_id].hidden_weight.clone().detach()
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
        self.make_layer(**kwargs)
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
        self.make_layer(**kwargs)
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
        # make layers
        self.make_layer(hidden=hidden, activation=activation,
                        pool=pool, dropout=dropout)
        # link parameters to self
        self.link_parameters(self.forward_layer)
        self.link_parameters(self.reconstruct_layer)
        # set vis_bias and hid_bias
        self.vis_bias = self.hidden_transpose_bias
        self.hid_bias = self.hidden_bias

    def update_layer(self, **kwargs):
        # update forward_layer
        for key, value in kwargs.items():
            if value is not None:
                self.forward_layer.add_module(key, value)

        # update reconstruct_layer unpool
        pool = kwargs.get('pool')
        if pool and hasattr(pool, 'return_indices') and pool.return_indices:
            pool_kwargs = functions.get_attributes(pool, ['stride', 'padding'])
            unpool = nn.MaxUnpool2d(pool.kernel_size, **pool_kwargs)
            self.reconstruct_layer.add_module('unpool', unpool)
        elif pool and hasattr(pool, 'kernel_size'):
            unpool = nn.Upsample(scale_factor=pool.kernel_size)
            self.reconstruct_layer.add_module('unpool', unpool)

        # update reconstruct_layer hidden transpose
        hidden = kwargs.get('hidden')
        if hidden:
            # initialize weights with randn
            self.init_weights('hidden_weight', torch.randn_like)
            # set transpose layer based on hidden type
            if torch.typename(hidden).find('conv') >= 0:
                conv_kwargs = functions.get_attributes(hidden, ['stride',
                                                                'padding',
                                                                'dilation'])
                hidden_transpose = nn.ConvTranspose2d(hidden.out_channels,
                                                      hidden.in_channels,
                                                      hidden.kernel_size,
                                                      **conv_kwargs)
                hidden_transpose.weight = hidden.weight
            elif torch.typename(hidden).find('linear') >= 0:
                hidden_transpose = nn.Linear(hidden.out_features,
                                             hidden.in_features)
                hidden_transpose.weight = nn.Parameter(hidden.weight.t())
            else:
                raise Exception('hidden type not understood')
            self.reconstruct_layer.add_module('hidden_transpose', hidden_transpose)
            # initialize biases with zeros
            self.init_weights('bias', torch.zeros_like)

        # update reconstruct_layer activation
        if self.vis_activation_fn:
            self.reconstruct_layer.add_module('activation', self.vis_activation_fn)
        else:
            self.reconstruct_layer.add_module('activation', nn.Sequential())

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
            neg = self.gibbs_vhv(v, k=k)[-2]
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
