import pickle

import IPython.display
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .modules import RBM
from .utils import functions, visualize

class Model(nn.Module):
    """
    Base class for initializing, training, saving, loading, visualizing models

    Attributes
    ----------
    loss_type : str or torch.nn.modules.loss
        cost function choice
    optimizer_type : str or torch.optim
        optimizer choice

    Methods
    -------
    n_layers()
        return number of layers
    output_shapes(input_shape)
        return output_shape for each layer
    append(layer)
        append a new layer to the model
    get_layers(layer_ids)
        return list of layers with given layer_ids
    apply_layers(input, layer_ids, forward=True)
        return result from applying specific layers
        Note: if forward=False, layer_ids will be reversed and reconstruct will
        be called for each layer
    forward(input)
        return result of forward pass through layers
    reconstruct(input)
        return result of backward pass through layers
        Note: requires that layers have reconstruct function
    train(n_epochs, trainloader, lr=0.001, monitor=100, **kwargs)
        trains the model with a given torch.utils.data.DataLoader and loss_fn
    pre_layer_ids(layer_id)
        return layer_ids before layer_id
    post_layer_ids(layer_id)
        return layer_ids after layer_id
    save_model(filename, extras=[])
        saves parameters from model and extras in pickle format
        Note: first saves model parameters as dictionary (see download_weights)
    load_model(filename)
        loads a previously saved model from filename in pickle format
        will load either a model instance or model dictionary
        (see download_weights)
    download_weights(pattern='')
        return parameters as dictionary (i.e. {name: parameter})
    load_weights(model_dict, param_dict={})
        load parameters from a dictionary (see download_weights)
        Note: use param_dict to associate keys in model_dict to parameter names
        in the current model (e.g., {'layers.0.hidden_weight': 'W_0'})
    init_weights(suffix='weight', fn=torch.randn_like)
        initialze weights for parameter names that end with suffix using fn
    get_trainable_params(pattern='')
        return trainable parameters (requires_grad=True) that contains pattern
    get_param_names():
        return name for each parameter
    set_requires_grad(pattern, requires_grad=True)
        set requires_grad attribute for parameters that contains pattern
    monitor_loss()
        #TODO:WRITEME
    get_prediction(input, crop=None)
        return prediction (max across output layer) for given input
        Note: if crop is given, max will be across cropped region of output
        layer (i.e. output[:,:,crop[0],crop[1]])
    get_accuracy(dataLoader)
        return model accuracy given a torch.utils.data.DataLoader with labels
    rf_index()
        #TODO:WRITEME
    rf_heatmap()
        #TODO:WRITEME
    """
    def __init__(self):
        super(Model, self).__init__()
        self.data_shape = None
        self.layers = nn.ModuleDict({})

    def n_layers(self):
        return len(self.layers)

    def output_shapes(self, input_shape=None):
        if input_shape is None:
            input_shape = self.data_shape
        # create dummy input
        input = torch.zeros(input_shape)
        # get each layer output shape
        output_shapes = []
        for layer_id, layer in self.layers.named_children():
            input = layer(input)
            if type(input) is list:
                output_shapes.append([i.shape for i in input])
            else:
                output_shapes.append(input.shape)
        return output_shapes

    def append(self, layer_id, layer):
        layer_id = str(layer_id)
        self.layers.add_module(layer_id, layer)

    def get_layers(self, layer_ids):
        layers = []
        for layer_id in layer_ids:
            layers.append(self.layers[layer_id])
        return layers

    def apply_layers(self, input, layer_ids=[], output_layer=None, forward=True,
                     **kwargs):
        if len(layer_ids) == 0 and output_layer is not None:
            layer_ids = self.get_layer_ids(output_layer, forward=forward)
        layers = self.get_layers(layer_ids)
        for layer in layers:
            if forward:
                input = layer.apply_modules(input, 'forward_layer', **kwargs)
            else:
                input = layer.apply_modules(input, 'reconstruct_layer', **kwargs)
        return input

    def forward(self, input):
        for name, layer in self.layers.named_children():
            input = layer.forward(input)
        return input

    def reconstruct(self, input):
        layer_ids = list(self.layers.keys())
        layer_ids.reverse()
        for layer_id in layer_ids:
            input = self.layers[layer_id].reconstruct(input)
        return input

    def get_layer_ids(self, layer_id=None, forward=True):
        layer_ids = [id for id, _ in self.layers.named_children()]
        if not forward:
            layer_ids.reverse()
        if layer_id is not None:
            layer_id = str(layer_id)
            cnt = [n+1 for n, id in enumerate(layer_ids) if id == layer_id][0]
            layer_ids = layer_ids[:cnt]
        return layer_ids

    def save_model(self, filename, extras={}):
        if type(extras) is not dict:
            extras = {'extras': extras}
        extras.update({'model_str': str(self)})
        extras.update({'model_weights': self.download_weights()})
        with open(filename, 'wb') as f:
            pickle.dump(extras, f)

    def load_model(self, filename, param_dict={}):
        extras = pickle.load(open(filename, 'rb'))
        model = []
        if type(extras) is list:
            model = extras.pop(0)
        elif type(extras) is dict and 'model_weights' in extras:
            self.load_weights(extras.get('model_weights'), param_dict)
            model = self
        if type(model) is dict:
            self.load_weights(model, param_dict)
            model = self
        return model, extras

    def download_weights(self, pattern=''):
        model_dict = {}
        for name, param in self.named_parameters():
            if name.find(pattern) >=0:
                model_dict.update({name: param.detach().numpy()})
        return model_dict

    def load_weights(self, model_dict, param_dict={}):
        # for each param, register new param from model_dict
        for name, param in self.named_parameters():
            # get layer to register parameter
            fields = name.split('.')
            layer = self
            for field in fields:
                layer = getattr(layer, field)
            # get param name in model_dict
            if param_dict.get(name):
                model_key = param_dict.get(name)
            elif model_dict.get(name) is not None:
                model_key = name
            else: # skip param
                continue
            # update parameter
            setattr(layer, 'data', torch.as_tensor(model_dict.get(model_key)))

    def init_weights(self, named_parameters=None, pattern='weight',
                     fn=torch.randn_like):
        if named_parameters is None:
            named_parameters = self.named_parameters()
        for name, param in named_parameters:
            with torch.no_grad():
                if name.find(pattern) >=0:
                    param.set_(fn(param))

    def get_param_names(self):
        param_names = []
        for (name, param) in self.named_parameters():
            param_names.append(name)
        return param_names

    def get_trainable_params(self, parameters=None, pattern=''):
        trainable_params = []
        if parameters:
            for param in parameters:
                if param.requires_grad:
                    trainable_params.append(param)
        else:
            for (name, param) in self.named_parameters():
                if param.requires_grad and name.find(pattern) >=0:
                    trainable_params.append(param)
        return trainable_params

    def set_requires_grad(self, parameters=None, pattern='', requires_grad=True):
        if parameters:
            for param in parameters:
                param.requires_grad = requires_grad
        else:
            for (name, param) in self.named_parameters():
                if name.find(pattern) >=0:
                    param.requires_grad = requires_grad

    def get_prediction(self, input, crop=None):
        with torch.no_grad():
            output = self.forward(input)
            if crop:
                output = output[:,:,crop[0],crop[1]]
            if output.ndimension() == 4:
                output = torch.max(output.flatten(-2), -1)[0]
            pred = torch.max(output, -1)[1]
        return pred

    def get_accuracy(self, dataloader, crop=None, monitor=100):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # get images, labels
                images, labels = data
                # get predicted labels, update accuracy
                pred = self.get_prediction(images, crop=crop)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                # monitor accuracy
                if (i+1) % monitor == 0:
                    clear_output(wait=True)
                    display('[%5d] accuracy: %.2f%%' % (i+1, 100*correct/total))

        return 100 * correct / total

    def train(self, n_epochs, trainloader, loss_fn, optimizer, monitor=100, **kwargs):
        """
        #TODO:WRITEME
        Note
        ----
        When using kwarg layer_params, batch_size should be equal to 1.
        """
        # get layer_id (layer-wise training) from kwargs
        options = functions.pop_attributes(kwargs, ['layer_id'], default=None)
        # get options from kwargs
        options.update(functions.pop_attributes(kwargs,
                                                ['add_loss','sparsity','scheduler',
                                                 'label_params','show_negative',
                                                 'show_lattice'],
                                                default={}))
        # if layer-wise training, ensure layer_id str and get pre_layer_ids
        if options.get('layer_id') is not None:
            layer_id = str(options.get('layer_id'))
            pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        # set label parameter gradients to false
        if options.get('label_params'):
            label_params = options.get('label_params')
            self.set_grad_by_label(label_params.keys(), label_params, False)
        # train for n_epochs
        loss_history = []
        running_loss = 0.
        i = 0
        n_batches = len(trainloader)
        for epoch in range(n_epochs):
            for data in trainloader:
                # get inputs, labels
                inputs = data[:-1]
                label = data[-1]
                # turn on label-based parameter gradients
                if options.get('label_params'):
                    self.set_grad_by_label([label], label_params, True)
                # layerwise training
                if options.get('layer_id') is not None:
                    # get inputs for layer_id
                    layer_input = self.apply_layers(inputs[0], pre_layer_ids)
                    if len(inputs[1:]) > 0 and options.get('add_loss') == {}:
                        layer_input = (layer_input,) + tuple(inputs[1:])
                    # train
                    loss = self.layers[layer_id].train(layer_input,
                                                       optimizer=optimizer,
                                                       add_loss=options.get('add_loss'),
                                                       sparsity=options.get('sparsity'),
                                                       **kwargs)
                else: # normal training
                    # zero gradients
                    optimizer.zero_grad()
                    # get outputs
                    output = self.forward(inputs[0])
                    # get loss
                    loss = loss_fn(output, label)
                    # additional loss
                    if options.get('add_loss'):
                        added_loss = self.add_loss(inputs, **options.get('add_loss'))
                        loss = loss + added_loss
                    # sparsity
                    if options.get('sparsity'):
                        self.sparsity(inputs[0], **options.get('sparsity'))
                    # backprop
                    loss.backward()
                    loss = loss.item()
                    # update parameters
                    optimizer.step()
                # update scheduler
                if options.get('scheduler'):
                    options.get('scheduler').step()
                # set label_parameters
                if options.get('label_params'):
                    self.set_grad_by_label([label], label_params, False)
                # monitor
                running_loss += loss
                i += 1
                if i % monitor == 0:
                    # display loss
                    clear_output(wait=True)
                    display('learning rate: %g' % optimizer.param_groups[0]['lr'])
                    display('[%g%%] loss: %.3f' % (i % n_batches / n_batches * 100.,
                            running_loss / monitor))
                    # append loss and show history
                    loss_history.append(running_loss / monitor)
                    plt.plot(loss_history)
                    plt.show()
                    running_loss = 0.
                    # show negative
                    if options.get('show_negative'):
                        self.show_negative(inputs[0], **options.get('show_negative'))
                    # show lattice
                    if options.get('show_lattice'):
                        if 'x' not in options.get('show_lattice'):
                            self.show_lattice(inputs[0], **options.get('show_lattice'))
                        else:
                            self.show_lattice(**options.get('show_lattice'))
                    # call other monitoring functions
                    functions.kwarg_fn([IPython.display, self], None, **kwargs)
        return loss_history

    def optimize_texture(self, n_steps, input, seed, loss_fn, optimizer,
                         layer_ids, module_name=None, transform=None,
                         monitor=100, show_texture=[], **kwargs):
        """
        #TODO:WRITEME
        """
        # turn off model gradients
        on_parameters = self.get_trainable_params()
        self.set_requires_grad(pattern='', requires_grad=False)
        loss_input = input
        # optimize texture
        loss_history = []
        running_loss = 0.
        for i in range(n_steps):
            optimizer.zero_grad()
            # transform input, get loss
            if transform:
                loss_input = transform(input.squeeze(1)).reshape(seed.shape)
            loss = self.add_loss([input, seed], loss_fn, layer_ids, module_name,
                                 **kwargs)
            loss.backward()
            # update seed
            optimizer.step()
            running_loss += loss.item()
            # monitor loss, show_texture
            if (i+1) % monitor == 0:
                # display loss
                clear_output(wait=True)
                display('learning rate: %g' % optimizer.param_groups[0]['lr'])
                display('[%5d] loss: %.3f' % (i+1, running_loss / monitor))
                # append loss and show history
                loss_history.append(running_loss / monitor)
                plt.plot(loss_history)
                plt.show()
                running_loss = 0.
                # monitor texture
                if show_texture:
                    self.show_texture(*show_texture)
                functions.kwarg_fn([IPython.display, self], None, **kwargs)
        # turn on model gradients
        self.set_requires_grad(on_parameters, requires_grad=True)
        return seed

    def set_grad_by_label(self, labels, label_params, requires_grad=True):
        # set parameter gradients based on label
        for label in labels:
            if label_params.get(label) is not None:
                if type(label_params.get(label)) is str:
                    self.set_requires_grad(pattern=label_params.get(label),
                                           requires_grad=requires_grad)
                else:
                    self.set_requires_grad(label_params.get(label),
                                           requires_grad=requires_grad)

    def add_loss(self, inputs, loss_fn, layer_ids, module_name=None,
                 cost=1., parameters=None, **kwargs):
        """
        #TODO:WRITEME
        """
        if parameters:
            on_parameters = self.get_trainable_params()
            self.set_requires_grad(pattern='', requires_grad=False)
            self.set_requires_grad(parameters, requires_grad=True)
        loss = torch.zeros(1, requires_grad=True)
        for i, (name, layer) in enumerate(self.layers.named_children()):
            if name in layer_ids:
                kwargs_i = {}
                for key, value in kwargs.items():
                    if type(value) is list:
                        kwargs_i.update({key: value[i]})
                    else:
                        kwargs_i.update({key: value})
                loss = loss + layer.add_loss(inputs, loss_fn, module_name, **kwargs_i)
            for ii, input in enumerate(inputs):
                inputs[ii] = layer.forward(input)
        loss = torch.mul(loss, cost)
        if parameters:
            self.set_requires_grad(parameters, requires_grad=False)
            self.set_requires_grad(on_parameters, requires_grad=True)
        return loss

    def sparsity(self, input, layer_ids, module_name, target, cost=1., l2_reg=0.):
        for i, (name, layer) in enumerate(self.layers.named_children()):
            if name in layer_ids:
                if type(target) is list:
                    target_i = target[i]
                else:
                    target_i = target
                if type(cost) is list:
                    cost_i = cost[i]
                else:
                    cost_i = cost
                if type(l2_reg) is list:
                    l2_reg_i = l2_reg[i]
                else:
                    l2_reg_i = l2_reg
                layer.sparsity(input, module_name, target_i, cost_i, l2_reg_i)
            input = layer.forward(input)

    def show_texture(self, input, seed):
        assert input.shape == seed.shape, (
            'input and seed shapes must match'
        )
        # get number of input images
        n_images = input.shape[0]

        # permute, squeeze input and seed
        input = torch.squeeze(input.permute(0,2,3,1), -1).detach()
        seed = torch.squeeze(seed.permute(0,2,3,1), -1).detach()

        # set cmap
        if input.shape[-1] == 3:
            cmap = None
            input = functions.normalize_range(input)
            seed = functions.normalize_range(seed)
        else:
            cmap = 'gray'

        # init figure, axes
        fig, ax = plt.subplots(n_images, 2)
        ax = np.reshape(ax, (n_images, 2))
        for n in range(n_images):
            ax[n,0].imshow(input[n], cmap=cmap)
            ax[n,1].imshow(seed[n], cmap=cmap)
        plt.show()
        return fig

    def show_lattice(self, x=None, figsize=(5,5), cmap=None):
        # get rf_layers
        rf_layers = []
        for layer_id, layer in self.layers.named_children():
            pool = layer.get_modules('forward_layer', ['pool'])
            if len(pool) == 1 and torch.typename(pool[0]).find('layers') >=0 and \
               pool[0].rfs is not None:
                rf_layers.append(pool[0])
        n_lattices =  len(rf_layers)
        if n_lattices == 0:
            raise Exception('No rf_pool layers found.')

        # show lattices
        with torch.no_grad():
            # get lattices
            lattices = []
            for pool in rf_layers:
                pool.show_lattice(x, figsize, cmap)

    def show_weights(self, layer_id, field='hidden_weight', img_shape=None,
                     figsize=(5, 5), cmap=None):
        """
        #TODO:WRITEME
        """
        # get field for weights
        layer_id = str(layer_id)
        if not hasattr(self.layers[layer_id], field):
            raise Exception('attribute ' + field + ' not found')
        w = getattr(self.layers[layer_id], field).clone().detach()
        # get weights reconstructed down if not first layer
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        pre_layer_ids.reverse()
        if len(pre_layer_ids) > 0:
            w = self.layers[layer_id].apply_modules(w,'reconstruct_layer',
                                                    ['activation'])
            w = self.apply_layers(w, pre_layer_ids, forward=False)
        return visualize.plot_images(w, img_shape, figsize, cmap)

    def show_negative(self, input, layer_id, n_images=-1, k=1, img_shape=None,
                      figsize=(5,5), cmap=None):
        """
        #TODO:WRITEME
        """
        layer_id = str(layer_id)
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        # get persistent if hasattr
        if hasattr(self.layers[layer_id], 'persistent'):
            neg = self.layers[layer_id].persistent
        else:
            neg = None
        # pass forward, then reconstruct down
        with torch.no_grad():
            if neg is None:
                neg = self.apply_layers(input, pre_layer_ids)
                if hasattr(self.layers[layer_id], 'gibbs_vhv'):
                    neg = self.layers[layer_id].gibbs_vhv(neg, k=k)[4]
                else:
                    neg = self.layers[layer_id].forward(neg)
                    neg = self.layers[layer_id].reconstruct(neg)
            pre_layer_ids.reverse()
            neg = self.apply_layers(neg, pre_layer_ids, forward=False)
        # reshape, permute for plotting
        if img_shape:
            input = torch.reshape(input, (-1,1) + img_shape)
            neg = torch.reshape(neg, (-1,1) + img_shape)
        # check that negative has <= 3 channels
        assert neg.shape[1] <= 3, ('negative image must have less than 3 channels')
        input = torch.squeeze(input.permute(0,2,3,1), -1).numpy()
        neg = torch.squeeze(neg.permute(0,2,3,1), -1).numpy()
        input = functions.normalize_range(input, dims=(1,2))
        neg = functions.normalize_range(neg, dims=(1,2))
        # plot negatives
        if n_images == -1:
            n_images = neg.shape[0]
        fig, ax = plt.subplots(n_images, 2, figsize=figsize)
        ax = np.reshape(ax, (n_images, 2))
        for r in range(n_images):
            ax[r,0].axis('off')
            ax[r,1].axis('off')
            ax[r,0].imshow(input[np.minimum(r, input.shape[0]-1)], cmap=cmap)
            ax[r,1].imshow(neg[r], cmap=cmap)
        plt.show()
        return fig

    def rf_output(self, input, layer_id, **kwargs):
        rf_layer = self.layers[layer_id].forward_layer.pool
        assert torch.typename(rf_layer).find('layers') > -1, (
            'No rf_pool layer found.'
        )
        # get layers before layer id
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        # apply forward up to layer id
        layer_input = self.apply_layers(input.detach(), pre_layer_ids)
        # apply modules including pool
        return self.layers[layer_id].apply_modules(layer_input, 'forward_layer',
                                                   output_module='pool', **kwargs)

    def rf_index(self, input, layer_id, thr=0.):
        pool_output = self.rf_output(input, layer_id, retain_shape=True)
        # sum across channels
        rf_outputs = torch.sum(pool_output, 1)
        # find rf_outputs with var > thr
        rf_var = torch.var(rf_outputs.flatten(-2), -1)
        return torch.gt(rf_var, thr)

    def rf_heatmap(self, layer_id):
        rf_layer = self.layers[layer_id].forward_layer.pool
        assert torch.typename(rf_layer).find('layers') > -1, (
            'No rf_pool layer found.'
        )
        # get layers before layer id
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        pre_layer_ids.reverse()
        # for each layer apply transpose convolution of ones and unpooling
        rfs = torch.unsqueeze(rf_layer.rfs, 1).detach()
        w_shape = self.layers[layer_id].forward_layer.hidden.kernel_size
        w = torch.ones((1, 1) + w_shape)
        heatmap = torch.conv_transpose2d(rfs, w)
        heatmap = torch.gt(heatmap, 0.).float()
        for id in pre_layer_ids:
            # upsample
            pool = self.layers[id].get_modules('forward_layer', ['pool'])
            if len(pool) == 1 and hasattr(pool[0], 'kernel_size'):
                pool_size = pool[0].kernel_size
                heatmap = torch.nn.functional.interpolate(heatmap,
                                                          scale_factor=pool_size)
            # conv_transpose2d
            hidden = self.layers[id].get_modules('forward_layer', ['hidden'])
            if len(hidden) == 1 and hasattr(hidden[0], 'weight'):
                w_shape = hidden[0].kernel_size
                w = torch.ones((1, 1) + w_shape)
                heatmap = torch.conv_transpose2d(heatmap, w)
            heatmap = torch.gt(heatmap, 0.).float()
        return heatmap.squeeze(1)

    def rf_to_image_space(self, layer_id, coords=None):
        # get mu, sigma
        if coords is None:
            coords = self.layers[layer_id].forward_layer.pool.get(['mu','sigma'])
        elif type(coords) is not torch.Tensor:
            coords = torch.tensor(coords)
        if type(coords) is not list and type(coords) is not tuple:
            coords = [coords]
        # reversed layers
        layers = self.get_layers(self.get_layer_ids(layer_id)[:-1])
        layers.reverse()
        # for each layer, add half weight kernel and multiply by pool kernel
        half_k = (self.layers[layer_id].forward_layer.hidden.kernel_size[0] - 1) // 2
        coords = [c + half_k for c in coords]
        for layer in layers:
            coords = [c * layer.forward_layer.pool.kernel_size for c in coords]
            half_k = (layer.forward_layer.hidden.kernel_size[0] - 1) // 2
            coords = [c + half_k for c in coords]
        return coords

class FeedForwardNetwork(Model):
    """
    #TODO:WRITEME
    """
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()

class DeepBeliefNetwork(Model):
    """
    #TODO:WRITEME

    Attributes
    ----------
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers in deep belief network (each layer is RBM class)

    Methods
    -------
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
        train deep belief network with greedy layer-wise contrastive divergence

    References
    ----------
    #TODO:WRITEME
    """
    def __init__(self):
        super(DeepBeliefNetwork, self).__init__()

    def posterior(self, layer_id, input, top_down_input=None, k=1):
        # get layer_ids
        layer_ids = self.get_layer_ids()
        # get output of n_layers-1
        top_layer_input = self.apply_layers(input, layer_ids[:-1])
        # gibbs sample top layer
        if top_down_input is not None:
            top_down = self.layers[layer_ids[-1]].gibbs_vhv(top_layer_input,
                                                            top_down_input, k=k)[3]
        else:
            top_down = self.layers[layer_ids[-1]].gibbs_vhv(top_layer_input, k=k)[3]
        # reconstruct down to layer_id
        post_layer_ids = self.get_layer_ids(layer_id, forward=False)[1:]
        layer_top_down = self.apply_layers(top_down, post_layer_ids, forward=False)
        # get layer_id input
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        layer_input = self.apply_layers(input, pre_layer_ids)
        # sample h given input, top_down
        return self.layers[layer_id].sample_h_given_vt(layer_input, layer_top_down)

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        return self.train(n_epochs, trainloader, None, optimizer, monitor=monitor,
                          layer_id=layer_id, k=k, **kwargs)

class DeepBoltzmannMachine(DeepBeliefNetwork):
    """
    #TODO:WRITEME

    Attributes
    ----------
    layers : torch.nn.ModuleDict
        RBM layers for forward/reconstruct pass (each layer is RBM class)
    output_shapes : list of tuples
        output shape for each layer

    Methods
    -------
    forward(input)
        perform forward pass with bottom-up input v through layer_ids
    reconstruct(input)
        perform reconstruction pass with top-down input t through layer_ids
    train(trainloader, optimizer, **kwargs)
        train deep boltzmann machine with contrastive divergence

    References
    ----------
    #TODO:WRITEME
    """
    def __init__(self):
        super(DeepBoltzmannMachine, self).__init__()
