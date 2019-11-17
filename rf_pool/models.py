from collections import OrderedDict
import pickle
import re

import IPython.display
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import losses, ops
from .modules import RBM
from .utils import functions, visualize

class Model(nn.Module):
    """
    Base class for initializing, training, saving, loading, visualizing models

    Attributes
    ----------
    data_shape : tuple
        shape of input data
    layers : torch.nn.ModuleDict
        layers containing computations to be performed
    """
    def __init__(self):
        super(Model, self).__init__()
        self.data_shape = None
        self.layers = nn.ModuleDict({})

    def n_layers(self):
        return len(self.layers)

    def output_shapes(self, input_shape=None, layer_ids=None):
        if input_shape is None:
            input_shape = self.data_shape
        if layer_ids is None:
            layer_ids = self.get_layer_ids()
        # create dummy input
        input = torch.zeros(input_shape)
        # get each layer output shape
        output_shapes = []
        for layer_id, layer in self.layers.named_children():
            input = layer(input)
            if type(input) is list:
                shape_i = [i.shape for i in input]
            else:
                shape_i = input.shape
            if layer_id in layer_ids:
                output_shapes.append(shape_i)
            if layer_id == layer_ids[-1]:
                break
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
        output = []
        if len(layer_ids) == 0 and output_layer is not None:
            if type(output_layer) is not list:
                output_layer = [output_layer]
            layer_ids = self.get_layer_ids(output_layer, forward=forward)
        elif output_layer is None:
            output = input
            output_layer = []
        layers = self.get_layers(layer_ids)
        if forward:
            layer_name = 'forward_layer'
        else:
            layer_name = 'reconstruct_layer'
        # apply each layer
        for layer_id, layer in zip(layer_ids, layers):
            # apply modules
            if kwargs:
                output_i = layer.apply_modules(input, layer_name, **kwargs)
                input = layer.apply_modules(input, layer_name)
            else:
                output_i = layer.apply_modules(input, layer_name)
                input = output_i
            # set to output
            if layer_id in output_layer:
                output.append(output_i)
            else:
                output = output_i
        return output

    def update_modules(self, layer_ids, layer_name, module_name, op, overwrite=True,
                       append=True):
        """
        Update self.layers[layer_ids].layer_name.module_name by appending,
        prepending, or overwriting with op.

        Parameters
        ----------
        layer_ids : list
            string ids of layers to update module_name
        layer_name : str
            name of layer to update (e.g., 'forward_layer')
        module_name : str
            name of module to update (e.g., 'activation')
        op : torch.nn.Module or list
            module(s) set for each layer_id
        overwrite : bool
            Boolean whether to overwrite current module (overrides append below)
            [default: True]
        append : bool
            If append=True, append as nn.Sequential(mod, op); otherwise,
            prepend as nn.Sequential(op, mod) [default: True]

        Returns
        -------
        mods : list
            list of current modules (one per layer_id)
        """
        # get current modules
        mods = [layer.get_modules(layer_name, [module_name])[0]
                for layer in self.get_layers(layer_ids)]
        # update modules
        for i, layer_id in enumerate(layer_ids):
            if type(op) is list and len(op) > i:
                op_i = op[i]
            elif type(op) is list:
                op_i = op[-1]
            else:
                op_i = op
            if overwrite:
                new_mod = op_i
            elif append:
                new_mod = torch.nn.Sequential(mods[i], op_i)
            else:
                new_mod = torch.nn.Sequential(op_i, mods[i])
            self.layers[layer_id].update_module(layer_name, module_name, new_mod)
        return mods

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
            if type(layer_id) is not list:
                layer_id = [layer_id]
            cnt = [n+1 for n, id in enumerate(layer_ids) if id in layer_id][-1]
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
        if type(model) is dict or type(model) is OrderedDict:
            self.load_weights(model, param_dict)
            model = self
        return model, extras

    def download_weights(self, pattern=''):
        model_dict = OrderedDict()
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

    def train(self, n_epochs, trainloader, loss_fn, optimizer, monitor=100,
              **kwargs):
        """
        #TODO:WRITEME
        Note
        ----
        When using kwarg layer_params, batch_size should be equal to 1.
        """
        # get layer_id (layer-wise training) from kwargs
        options = functions.pop_attributes(kwargs, ['layer_id'], default=None)
        options.update(functions.pop_attributes(kwargs, ['retain_graph'],
                                                default=False))
        # get options from kwargs
        options.update(functions.pop_attributes(kwargs,
                                                ['add_loss','sparse_loss',
                                                 'scheduler','label_params',
                                                 'show_negative','show_lattice',
                                                 'monitor_loss'],
                                                default={}))
        # added loss
        if options.get('add_loss'):
            if type(options.get('add_loss')) is dict:
                add_loss = losses.LayerLoss(self, **options.get('add_loss'))
            else:
                add_loss = options.get('add_loss')
        # sparsity loss
        if options.get('sparse_loss'):
            if type(options.get('sparse_loss')) is dict:
                sparse_loss = losses.SparseLoss(self, **options.get('sparse_loss'))
            else:
                sparse_loss = options.get('sparse_loss')
        # monitor loss
        if options.get('monitor_loss'):
            if type(options.get('monitor_loss')) is dict:
                monitor_loss = losses.KwargsLoss(**options.get('monitor_loss'))
            else:
                monitor_loss = options.get('monitor_loss')
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
                                                       **kwargs)
                    if options.get('add_loss'):
                        loss = loss + add_loss(*inputs)
                    if options.get('sparse_loss'):
                        loss = loss + sparse_loss(*inputs)
                else: # normal training
                    # zero gradients
                    optimizer.zero_grad()
                    # get loss
                    if torch.typename(loss_fn).find('losses') > -1:
                        loss = loss_fn(*data)
                    else: # get outputs then loss
                        output = self.forward(inputs[0])
                        loss = loss_fn(output, label)
                    # additional loss
                    if options.get('add_loss'):
                        loss = loss + add_loss(*inputs)
                    # sparse_loss
                    if options.get('sparse_loss'):
                        loss = loss + sparse_loss(*inputs)
                    # backprop
                    loss.backward(retain_graph=options.get('retain_graph'))
                    loss = loss.item()
                    # update parameters
                    optimizer.step()
                # update scheduler
                if options.get('scheduler'):
                    options.get('scheduler').step()
                # set label_parameters
                if options.get('label_params'):
                    self.set_grad_by_label([label], label_params, False)
                # monitor loss
                with torch.no_grad():
                    if options.get('monitor_loss'):
                        running_loss += monitor_loss(*inputs)
                    else:
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
                         transform=None, monitor=100, show_images=[], **kwargs):
        """
        #TODO:WRITEME
        """
        # turn off model gradients
        on_parameters = self.get_trainable_params()
        self.set_requires_grad(pattern='', requires_grad=False)
        # transform input
        loss_input = input
        if transform:
            if type(input) is list:
                loss_input = [transform(input_i.squeeze(1)).reshape(seed.shape)
                              for input_i in input]
            else:
                loss_input = transform(input.squeeze(1)).reshape(seed.shape)
        # set loss_fn to LayerLoss if not
        if torch.typename(loss_fn).find('losses') == -1:
            loss_fn = losses.LayerLoss(self, loss_fn, target=loss_input, **kwargs)
            loss_input = [] # set to empty since target set in loss_fn
        # optimize texture
        loss_history = []
        running_loss = 0.
        for i in range(n_steps):
            optimizer.zero_grad()
            # get loss
            if len(loss_input) > 0:
                loss = loss_fn(seed, loss_input)
            else:
                loss = loss_fn(seed)
            loss.backward(retain_graph=True)
            # update seed
            optimizer.step()
            running_loss += loss.item()
            # monitor loss, show_images
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
                if show_images:
                    visualize.show_images(*show_images, **kwargs)
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

    def show_lattice(self, x=None, figsize=(5,5), cmap=None):
        # get rf_layers
        rf_layers = []
        for layer_id, layer in self.layers.named_children():
            pool = layer.get_modules('forward_layer', ['pool'])
            if len(pool) == 1 and hasattr(pool[0], 'rfs') and \
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
            w = self.apply_layers(w, pre_layer_ids, forward=False)
        return visualize.show_images(w, img_shape=img_shape, figsize=figsize,
                                     cmap=cmap)

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

    def sparseness(self, input, layer_id, module_name=None, **kwargs):
        # equation from Hoyer (2004) used in Ji et al. (2014)
        with torch.no_grad():
            pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
            layer_input = self.apply_layers(input.detach(), pre_layer_ids)
            activity = self.layers[layer_id].apply_modules(layer_input,
                                                           'forward_layer',
                                                           output_module=module_name,
                                                           **kwargs)
            n = torch.tensor(torch.numel(activity), dtype=activity.dtype)
            l1_l2 = torch.div(torch.sum(torch.abs(activity)),
                              torch.sqrt(torch.sum(torch.pow(activity, 2))))
            sqrt_n = torch.sqrt(n)
        return (sqrt_n - l1_l2) / (sqrt_n - 1)

    def rf_output(self, input, layer_id, **kwargs):
        rf_layer = self.layers[layer_id].forward_layer.pool
        assert hasattr(rf_layer, 'rfs'), (
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
        assert hasattr(rf_layer, 'rfs'), (
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
        post_layer_ids = self.get_layer_ids(layer_id, forward=False)[1:-1]
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
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers for forward/reconstruct pass (each layer is RBM class)

    Methods
    -------
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
        train deep boltzmann machine with contrastive divergence
    train_dbm()

    References
    ----------
    #TODO:WRITEME
    """
    def __init__(self):
        super(DeepBoltzmannMachine, self).__init__()

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        # get layer_ids, layer_name if layer_id is first/last
        layer_ids = self.get_layer_ids()
        if layer_id == layer_ids[0]:
            layer_name = 'forward_layer'
        elif layer_id == layer_ids[-1]:
            layer_name = 'reconstruct_layer'
        else:
            layer_name = None
        # update activation
        if layer_name is not None:
            mul_op = ops.Op(lambda x: 2. * x)
            act_op = self.update_modules([layer_id], layer_name, 'activation',
                                         mul_op, append=False, overwrite=False)
        # train
        try:
            output = self.train(n_epochs, trainloader, None, optimizer,
                                monitor=monitor, layer_id=layer_id, k=k, **kwargs)
        finally:
            if layer_name is not None:
                self.update_modules([layer_id], layer_name, 'activation', act_op)
        return output

    def train_dbm(self, n_epochs, trainloader, optimizer, k=1, n_iter=10,
                  monitor=100, **kwargs):
        # set loss function
        loss_fn = losses.KwargsLoss(self.contrastive_divergence, n_args=1,
                                    k=k, n_iter=n_iter, **kwargs)
        # train
        return self.train(n_epochs, trainloader, loss_fn, optimizer,
                          monitor=monitor, **kwargs)

    def contrastive_divergence(self, input, k=1, n_iter=10, **kwargs):
        #TODO: update mean_field to condition on output
        # positive phase mean field
        layer_ids = self.get_layer_ids()
        layers = self.get_layers(layer_ids)
        hids = None
        with torch.no_grad():
            for _ in range(n_iter):
                v, hids = self.layer_gibbs(input, hids, sampled=False)
        # get positive energies
        pos_energy = []
        for i, layer in enumerate(layers):
            if i == 0:
                pos_energy.append(layer.energy(input, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                pos_energy.append(layer.energy(h_n, hids[i][1]))
        # negative phase for each layer
        if hasattr(self, 'persistent'):
            v = self.persistent
        elif kwargs.get('persistent') is not None:
            self.persistent = kwargs.get('persistent')
            v = self.persistent
        hids = None
        with torch.no_grad():
            for _ in range(k):
                v, hids = self.layer_gibbs(v, hids, sampled=True)
        if hasattr(self, 'persistent'):
            self.persistent = v
        # get negative energies
        neg_energy = []
        for i, layer in enumerate(layers):
            if i == 0:
                neg_energy.append(layer.energy(v, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                neg_energy.append(layer.energy(h_n, hids[i][1]))
        # return mean difference in energies
        return torch.mean(torch.cat(pos_energy) - torch.cat(neg_energy))

    def layer_gibbs(self, input, hids=None, sampled=False, pooled=False):
        # get layer_ids
        layer_ids = self.get_layer_ids()
        layers = self.get_layers(layer_ids)
        # get hids from forward pass with weights doubled
        if hids is None:
            mul_op = ops.Op(lambda x: 2. * x)
            hid_ops = self.update_modules(layer_ids[:-1], 'forward_layer',
                                          'hidden', mul_op, overwrite=False)
            try: # forward pass with doubled weights
                hids = self.apply_layers(input, layer_ids, output_layer=layer_ids)
            finally:
                self.update_modules(layer_ids[:-1], 'forward_layer', 'hidden',
                                    hid_ops)
            hids = [(h,) + layer.sample(h, 'forward_layer', pooled)
                    for h, layer in zip(hids, layers)]
        # get idx based on sampled bool
        idx = np.int(sampled)
        # update even layers
        for i, layer in list(enumerate(layers))[::2]:
            if i > 0:
                v = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[idx]
            else:
                v = input
            if i < (len(layers) - 1):
                t = layers[i+1].sample_v_given_h(hids[i+1][idx+1])[idx+1]
                hids[i] = layer.sample_h_given_vt(v, t, pooled)
            else:
                hids[i] = layer.sample_h_given_v(v, pooled)
        # get v out
        v_out = layers[0].sample_v_given_h(hids[0][idx+1])[idx+1]
        # update odd layers
        for i, layer in list(enumerate(layers))[1::2]:
            v = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[idx]
            if i < (len(layers) - 1):
                t = layers[i+1].sample_v_given_h(hids[i+1][idx+1])[idx+1]
                hids[i] = layer.sample_h_given_vt(v, t, pooled)
            else:
                hids[i] = layer.sample_h_given_v(v, pooled)
        return v_out, hids
