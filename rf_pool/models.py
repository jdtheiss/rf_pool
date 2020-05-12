from collections import OrderedDict
import copy
import pickle
import re

import IPython.display
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import losses, ops, pool
from .modules import Module, FeedForward, RBM
from .utils import functions, visualize

def _get_dict_shapes(d):
    """
    Helper function to replace array-like with its shape within a dictionary
    """
    if not isinstance(d, (dict, OrderedDict)):
        return None
    # update with shapes
    for k, v in d.items():
        if type(v) is list:
            v = [v_i.shape if hasattr(v_i, 'shape') else None for v_i in v]
        elif hasattr(v, 'shape'):
            v = v.shape
        elif isinstance(v, (dict, OrderedDict)):
            v = _get_dict_shapes(v)
        d.update({k: v})
    return d

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
    def __init__(self, model=None):
        super(Model, self).__init__()
        self.data_shape = None
        self.layers = nn.ModuleDict({})
        # import model if given
        if model is not None:
            self.load_architecture(model)

    def n_layers(self):
        return len(self.layers)

    def get_layers(self, layer_ids):
        layers = []
        for layer_id in layer_ids:
            if layer_id is None:
                layers.append(lambda x: x)
            else:
                layers.append(self.layers[layer_id])
        return layers

    def get_layer_ids(self, output_layer_id=None, forward=True):
        layer_ids = [id for id, _ in self.layers.named_children()]
        if not forward:
            layer_ids.reverse()
        if type(output_layer_id) is not list:
            output_layer_id = [output_layer_id]
        cnt = -1
        for i, name in enumerate(layer_ids):
            if name in output_layer_id:
                cnt = i + 1
        if cnt > -1:
            return layer_ids[:cnt]
        return layer_ids

    def get_layers_by_type(self, layer_types=(), layer_str_types=[]):
        # ensure layer_types is tuple and layer_str_types is list
        if isinstance(layer_types, type):
            layer_types = (layer_types,)
        layer_types = tuple(layer_types)
        if isinstance(layer_str_types, str):
            layer_str_types = [layer_str_types]
        layer_str_types = list(layer_str_types)
        # get layers in layer_types or layer_str_types
        layers = []
        for name, layer in self.layers:
            # if isinstance or endswith type name, append
            l_type = torch.typename(layer)
            if isinstance(layer, layer_types) or \
               any([l_type.endswith(s) for s in layer_str_types]):
                layers.append(layer)
        return layers

    def get_layers_by_attr(self, attributes):
        # ensure attributes is list
        if isinstance(attributes, str):
            attributes = [attributes]
        attributes = list(attributes)
        # get layers with attributes
        layers = []
        for name, layer in self.layers:
            # if hasattr, append
            if any([hasattr(layer, attr) for attr in attributes]):
                layers.append(layer)
        return layers

    def append(self, layer_id, layer):
        layer_id = str(layer_id)
        self.layers.add_module(layer_id, layer)

    def insert(self, idx, layer_id, layer):
        layer_id = str(layer_id)
        new_layers = nn.ModuleDict({})
        for i, (layer_id_i, layer_i) in enumerate(self.layers.named_children()):
            if i == idx:
                new_layers.add_module(layer_id, layer)
            new_layers.add_module(layer_id_i, layer_i)
        self.layers = new_layers

    def remove(self, layer_id):
        return self.layers.pop(layer_id)

    def apply(self, input, layer_ids=[], forward=True, output={},
              output_layer=None, **kwargs):
        """
        Apply layers with layer-specific kwargs and/or collect outputs in dict

        Parameters
        ----------
        input : torch.Tensor
            input passed through model layers
        layer_ids : list, optional
            names of layers to apply to input [default: [], apply all layers]
        forward : boolean, optional
            True/False to apply forward (vs. reverse) pass through layers
            [default: True]
        output : dict, optional
            dictionary like {layer_id: []} to be updated with specific results
            [default: {}, will not set outputs to dictionary]
        output_layer : str, optional
            name of layer to stop passing input through model (i.e., get layer_ids
            for each layer up to and including output_layer)
            [default: None, uses layer_ids]
        **kwargs : dict
            layer-specific keyword arguments like {layer_id: kwargs} to be applied
            for a given layer

        Results
        -------
        output : torch.Tensor
            output of passing input through layers

        Examples
        --------
        >>> # pass input through model and obtain outputs of specific layer
        >>> model = FeedForwardNetwork()
        >>> model.append('0', FeedForward(hidden=torch.nn.Conv2d(1,16,5),
                                          activation=torch.nn.ReLU(),
                                          pool=torch.nn.MaxPool2d(2)))
        >>> model.append('1', FeedForward(hidden=torch.nn.Linear(16, 10)))
        >>> saved_outputs = {'0': {'pool': []}}
        >>> output = model.apply(torch.rand(1,1,6,6), layer_ids=['0','1'],
                                 output=saved_outputs)
        >>> print(output.shape, saved_outputs.get('0').get('pool')[0].shape)
        torch.Size([1, 10]) torch.Size([1, 16, 1, 1])
        """
        # get layers for layer_ids
        if len(layer_ids) == 0:
            layer_ids = self.get_layer_ids(output_layer, forward=forward)
        layers = self.get_layers(layer_ids)
        # set layer_name
        layer_name = ['reconstruct_layer','forward_layer'][forward]
        # for each layer, apply modules
        for layer_id, layer in zip(layer_ids, layers):
            # get layer_kwargs and set default layer_name
            layer_kwargs = kwargs.get(layer_id)
            if layer_kwargs is None:
                layer_kwargs = {}
            layer_kwargs.setdefault('layer_name', layer_name)
            # get output dict for layer
            layer_output = output.get(layer_id)
            if not isinstance(layer_output, (dict, OrderedDict)):
                layer_output = {}
            # apply modules
            if layer_id is not None:
                input = layer.apply(input, output=layer_output, **layer_kwargs)
            # append to output if list
            if type(output.get(layer_id)) is list:
                output.get(layer_id).append(input)
        return input

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def output_shapes(self, input_shape=None, layer_ids=[], output_layer=None,
                      **kwargs):
        """
        Get output shapes for layers within model

        Parameters
        ----------
        input_shape : tuple, optional
            shape of input data for which to get output shapes
            [default: None, tries self.data_shape]
        layer_ids : list, optional
            layer ids for which to get output shapes
            [default: [], all layers]
        output_layer : str, optional
            layer id at which to stop getting output shapes (i.e., output shapes
            for all layers up to and including output_layer)
            [default: None, uses layer_ids]
        **kwargs : dict, optional
            keyword arguments used in apply call (see self.apply)

        Returns
        -------
        output_shapes : dict
            dictionary like {layer_id: [output_shape]} containing output shapes
            for given input_shape and layer_ids

        Examples
        --------
        >>> # obtain output shapes for modules within model
        >>> model = FeedForwardNetwork()
        >>> model.append('0', FeedForward(hidden=torch.nn.Conv2d(1,16,5),
                                          activation=torch.nn.ReLU(),
                                          pool=torch.nn.MaxPool2d(2)))
        >>> model.append('1', FeedForward(hidden=torch.nn.Linear(16, 10)))
        >>> output = {'0': {'pool': []}, '1': {'hidden': []}}
        >>> shapes = output_shapes((1,1,6,6), output=output)
        >>> print(shapes)
        {'0': {'pool': [torch.Size([1, 16, 1, 1])]}, '1': {'hidden': [torch.Size([1, 10])]}}
        """
        if input_shape is None:
            input_shape = self.data_shape
        # get layer_ids
        if len(layer_ids) == 0:
            layer_ids = self.get_layer_ids(output_layer)
        # set outputs based on layer_ids
        output = kwargs.get('output')
        if output is None:
            output = dict([(id, []) for id in layer_ids])
            kwargs.update({'output': output})
        # create dummy input
        input = torch.zeros(input_shape)
        # get output shapes
        self.apply(input, layer_ids, **kwargs)
        # update with shapes
        return _get_dict_shapes(output)

    def update_modules(self, layer_ids, layer_name, module_name, op,
                       overwrite=True, append=True):
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

    def save_model(self, filename, **kwargs):
        """
        Save current model as dictionary containing `str(model)` and model
        weights.

        Parameters
        ----------
        filename : str
            file name to save model
        **kwargs : **dict
            optional extras to save like `key=value`

        Returns
        -------
        None

        Notes
        -----
        Model saved as dictionary containing following keys 'model_str',
        'model_weights', and `kwargs.keys()`.
        """
        kwargs.update({'model_str': str(self)})
        kwargs.update({'model_weights': self.download_weights()})
        with open(filename, 'wb') as f:
            pickle.dump(kwargs, f)

    def load_architecture(self, model, pattern='^.+$', verbose=False):
        # function to get typename
        get_typename = lambda v: torch.typename(v).split('.')[-1].lower()
        # for each child, append to layers
        for layer_id, layer in model.named_children():
            layer_id = re.findall(pattern, layer_id)
            if len(layer_id) == 0:
                continue
            # get alphanumeric layer_id, and typename
            layer_id = re.sub('[^a-zA-Z0-9_]*', '', layer_id[0])
            if len(layer_id) == 0:
                continue
            # print
            if verbose:
                print(layer_id)
            # append to layers if is Module
            if isinstance(layer, Module):
                self.append(layer_id, layer)
            else: # get inputs for FeedForward
                inputs = dict([('%s%s' % (get_typename(v), k.replace('.','_')), v)
                               for k, v in layer.named_modules()
                               if type(v) is not torch.nn.Sequential])
                self.append(layer_id, FeedForward(**inputs))

    def load_model(self, filename, dtype=torch.float, verbose=False):
        extras = pickle.load(open(filename, 'rb'))
        model = []
        if type(extras) is list:
            model = extras.pop(0)
        elif isinstance(extras, (dict, OrderedDict)) and 'model_weights' in extras:
            self.load_weights(extras.get('model_weights'), dtype, verbose)
            model = self
        if isinstance(model, (dict, OrderedDict)):
            self.load_weights(model, dtype, verbose)
            model = self
        return model, extras

    def download_weights(self, pattern='', verbose=False):
        model_dict = OrderedDict()
        for name, param in self.named_parameters():
            if name.find(pattern) >=0:
                if verbose:
                    print(name)
                model_dict.update({name: param.detach().numpy()})
        return model_dict

    def load_weights(self, model_dict, dtype=torch.float, verbose=False):
        # for each param, register new param from model_dict
        for name, param in self.named_parameters():
            # get layer to register parameter
            fields = name.split('.')
            layer = self
            for field in fields:
                layer = getattr(layer, field)
            # get param name in model_dict
            if model_dict.get(name) is not None:
                key = name
            elif any([name.endswith(k) for k in model_dict.keys()]):
                key = [k for k in model_dict.keys()
                       if name.endswith(k)][0]
            else: # skip param
                continue
            if verbose:
                print(key)
            # update parameter
            param = model_dict.get(key)
            setattr(layer, 'data', torch.as_tensor(param).type(dtype))

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

    def get_prediction(self, input, crop=None, top_n=1):
        with torch.no_grad():
            output = self.forward(input)
            if crop:
                output = output[:,:,crop[0],crop[1]]
            if output.ndimension() == 4:
                output = torch.max(output.flatten(-2), -1)[0]
            pred = torch.sort(output, -1)[1]
        return pred[:,-top_n:]

    def get_accuracy(self, dataloader, crop=None, monitor=100, top_n=1):
        correct = 0.
        total = 0.
        for i, (data, labels) in enumerate(dataloader):
            with torch.no_grad():
                # get predicted labels, update accuracy
                pred = self.get_prediction(data, crop=crop, top_n=top_n)
                total += float(labels.shape[0])
                correct += float(torch.eq(pred, labels.reshape(-1,1)).sum())
                # monitor accuracy
                if (i+1) % monitor == 0:
                    clear_output(wait=True)
                    display('[%5d] accuracy: %.2f%%' % (i+1, 100.*correct/total))
        return 100. * correct / total

    def train(self, n_epochs, trainloader, loss_fn, optimizer, monitor=100,
              **kwargs):
        """
        Train model using loss function and optimizer for given dataloader

        Parameters
        ----------
        n_epochs : int
            number of epochs to train for (complete passes through dataloader)
        trainloader : torch.utils.data.DataLoader
            dataloader containing training (data, label) pairs
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function to opimize during training
        optimizer : torch.optim
            optimizer used to update parameters during training
        monitor : int
            number of batches between plotting loss, showing weights, etc.
            [default: 100]

        Optional kwargs
        layer_id : str
            id of layer to train (especially for training RBMs layer-wise)
            [default: None, all layers trained]
        retain_graph : boolean
            kwarg passed to `loss.backward()` to maintain graph if True
            [default: False]
        add_loss : rf_pool.losses or dict
            additional loss function added to loss_fn. if dict, kwargs passed to
            `rf_pool.losses.LayerLoss`
            [default: {}, no added loss]
        sparse_loss : rf_pool.losses or dict
            sparse loss function added to loss_fn. if dict, kwargs passed to
             `rf_pool.losses.SparseLoss`
            [default: {}, no added loss]
        monitor_loss : rf_pool.losses or dict
            loss function used only during monitoring step (i.e. not used to
            update parameters). if dict, kwargs passed to
            `rf_pool.losses.KwargsLoss`
            [default: {}, loss_fn used for monitoring loss]
        metrics : module
            module from which metric functions can be called, which should be
            passed as separate **kwargs (e.g., `metrics=numpy, add=[1,2]`).
            [default: None]
        tensorboard : torch.utils.tensorboard.SummaryWriter
            SummaryWriter to monitor loss, metrics, and figures plotted.
            See `torch.utils.tensorboard.SummaryWriter` for more information.
            [default: None]
        scheduler : torch.optim.lr_scheduler
            scheduler used to periodically update learning rate
        show_weights : dict
            kwargs passed to `visualize.show_weights` function during monitoring
            step. [default: {}, function not called]
        show_lattice : dict
            kwargs passed to `visualize.show_lattice` function during monitoring
            step. [default: {}, function not called]
        show_negative : dict
            kwargs passed to `visualize.show_negative` function during
            monitoring step. [default: {}, function not called]
        label_params : dict
            dictionary with (label, params) pairs of parameters that should be
            set to `requires_grad=True` when the given label is observed in the
            dataloader. See `set_grad_by_label`.
            [default: {}, function not called]

        Returns
        -------
        loss_history : list
            list of loss values at each monitoring step
            (i.e., len(loss_history) == (n_epochs * len(trainloader) / monitor))

        Note
        ----
        When using kwarg `label_params`, `batch_size` in dataloader should be
        equal to 1.
        """
        # get layer_id (layer-wise training) from kwargs
        options = functions.pop_attributes(kwargs, ['layer_id','tensorboard',
                                                    'metrics'],
                                           default=None)
        options.update(functions.pop_attributes(kwargs, ['retain_graph'],
                                                default=False))
        # get options from kwargs
        options.update(functions.pop_attributes(kwargs,
                                                ['add_loss','sparse_loss',
                                                 'scheduler','label_params',
                                                 'monitor_loss'],
                                                default={}))
        # added loss
        if options.get('add_loss'):
            if isinstance(options.get('add_loss'), (dict, OrderedDict)):
                add_loss = losses.LayerLoss(self, **options.get('add_loss'))
            else:
                add_loss = options.get('add_loss')
        # sparsity loss
        if options.get('sparse_loss'):
            if isinstance(options.get('sparse_loss'), (dict, OrderedDict)):
                sparse_loss = losses.SparseLoss(self, **options.get('sparse_loss'))
            else:
                sparse_loss = options.get('sparse_loss')
        # monitor loss
        if options.get('monitor_loss'):
            if isinstance(options.get('monitor_loss'), (dict, OrderedDict)):
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
        for epoch in range(int(np.ceil(n_epochs))):
            for data in trainloader:
                # check if more than requested epochs
                if (i+1) > (n_epochs * n_batches):
                    return loss_history
                # get inputs, labels
                inputs = data[:-1]
                label = data[-1]
                # turn on label-based parameter gradients
                if options.get('label_params'):
                    self.set_grad_by_label([label], label_params, True)
                # layerwise training
                if options.get('layer_id') is not None:
                    # get inputs for layer_id
                    if len(pre_layer_ids) > 0:
                        layer_input = self.apply(inputs[0], pre_layer_ids)
                    else:
                        layer_input = inputs[0].clone()
                    if len(inputs[1:]) > 0 and options.get('add_loss') == {}:
                        layer_input = (layer_input,) + tuple(inputs[1:])
                    # train
                    loss = self.layers[layer_id].train(layer_input,
                                                       optimizer=optimizer,
                                                       **kwargs)
                    # backprop extra losses if given
                    ext_loss = 0.
                    if options.get('add_loss'):
                        ext_loss = ext_loss + add_loss(*inputs)
                    if options.get('sparse_loss'):
                        ext_loss = ext_loss + sparse_loss(*inputs)
                    if isinstance(ext_loss, torch.Tensor):
                        ext_loss.backward(retain_graph=options.get('retain_graph'))
                        optimizer.step()
                else: # normal training
                    # zero gradients
                    layer_input = inputs[0]
                    optimizer.zero_grad()
                    # get loss
                    if isinstance(loss_fn, losses.LayerLoss):
                        loss = loss_fn(*data)
                    else:
                        # get outputs then loss
                        output = self.forward(layer_input)
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
                # add loss to running loss
                running_loss += loss
                i += 1
                if i % monitor == 0:
                    # monitor loss
                    with torch.no_grad():
                        if options.get('monitor_loss'):
                            loss_history.append(monitor_loss(layer_input))
                        else:
                            loss_history.append(running_loss / monitor)
                    # display loss
                    clear_output(wait=True)
                    display('learning rate: %g' % optimizer.param_groups[0]['lr'])
                    display('%d [%g%%] loss: %.3f' % (epoch,
                                                      i % n_batches/n_batches*100.,
                                                      loss_history[-1]))
                    # show loss history
                    plt.plot(loss_history)
                    plt.show()
                    running_loss = 0.
                    # show weights
                    if kwargs.get('show_weights'):
                        kwargs.get('show_weights').update({'model': self})
                    # show negative
                    if kwargs.get('show_negative'):
                        kwargs.get('show_negative').update({'model': self,
                                                            'input': inputs[0]})
                    # show lattice
                    if kwargs.get('show_lattice'):
                        kwargs.get('show_lattice').update({'model': self,
                                                           'input': inputs[0]})
                    # update global_step in any tensorboard calls
                    if options.get('tensorboard'):
                        for k, v in kwargs.items():
                            if hasattr(options.get('tensorboard'), k):
                                if isinstance(v, dict):
                                    v.update({'global_step': i})
                    # call other monitoring functions
                    with torch.no_grad():
                        outputs = functions.kwarg_fn([IPython.display, self,
                                                      visualize,
                                                      options.get('metrics'),
                                                      options.get('tensorboard')],
                                                      None, **kwargs)
                    # TensorBoard
                    if options.get('tensorboard'):
                        options.get('tensorboard').add_scalar('loss',
                                                              loss_history[-1],
                                                              i)
                        for k, v in outputs.items():
                            if isinstance(v, plt.Figure):
                                options.get('tensorboard').add_figure(k, v, i)
                            elif isinstance(v, (int, float)):
                                options.get('tensorboard').add_scalar(k, v, i)
        return loss_history

    def optimize_texture(self, n_steps, seed, loss_fn, optimizer, input=[],
                         transform=None, monitor=100, **kwargs):
        """
        Optimize texture seed image

        Parameters
        ----------
        n_steps : int
            number of steps to optimizer seed image
        seed : torch.Tensor
            seed image to be optimized
        loss_fn : function or rf_pool.losses or torch.nn.modules.loss
            loss function used to optimize seed image.
            See `rf_pool.losses.LayerLoss`
        input : torch.Tensor or list
            input image(s) to use as target for loss function [default: []]
        transform : rf_pool.utils.transforms
            transforms to be applied to seed/input images [default: None]
        monitor : int
            number of optimization steps between monitoring loss, images
        **kwargs : **dict
            keyword arguments passed to `rf_pool.utils.visualize.show_images`
            during monitoring step

        Returns
        -------
        seed : torch.Tensor
            updated seed image based on optimization

        See Also
        --------
        rf_pool.losses.LayerLoss
        rf_pool.utils.visualize.visualize_features
        """
        # turn off model gradients
        on_parameters = self.get_trainable_params()
        self.set_requires_grad(pattern='', requires_grad=False)
        if input is None:
            input = []
        if type(input) is not list:
            input = [input]
        # optimize texture
        loss_history = []
        running_loss = 0.
        for i in range(n_steps):
            # apply transformation to seed, input
            if transform:
                seed.data = transform(seed)
                loss_input = [transform(input_n) for input_n in input]
            else:
                loss_input = input
            # zero gradients
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
                visualize.show_images(seed, *loss_input, **kwargs)
        # turn on model gradients
        self.set_requires_grad(on_parameters, requires_grad=True)
        return seed

    def set_grad_by_label(self, labels, label_params, requires_grad=True):
        """
        Set `requires_grad` for parameters in `label_params.get(label)`

        Parameters
        ----------
        labels : list or torch.Tensor
            labels to set `requires_grad`
        label_params : dict
            dictionary with (label, parameters) pairs to set `requires_grad`
        requires_grad : boolean
            True/False to set `requires_grad` for given parameters

        Returns
        -------
        None
        """
        # set parameter gradients based on label
        for label in labels:
            if label_params.get(label) is not None:
                if type(label_params.get(label)) is str:
                    self.set_requires_grad(pattern=label_params.get(label),
                                           requires_grad=requires_grad)
                else:
                    self.set_requires_grad(label_params.get(label),
                                           requires_grad=requires_grad)

    def sparseness(self, input, layer_id, module_name=None, **kwargs):
        # equation from Hoyer (2004) used in Ji et al. (2014)
        with torch.no_grad():
            pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
            layer_input = self.apply(input.detach(), pre_layer_ids)
            activity = self.layers[layer_id].apply(layer_input, 'forward_layer',
                                                   output_module=module_name,
                                                   **kwargs)
            n = torch.tensor(torch.numel(activity), dtype=activity.dtype)
            l1_l2 = torch.div(torch.sum(torch.abs(activity)),
                              torch.sqrt(torch.sum(torch.pow(activity, 2))))
            sqrt_n = torch.sqrt(n)
        return (sqrt_n - l1_l2) / (sqrt_n - 1)

    def rf_output(self, input, layer_id, module_name='pool', **kwargs):
        if module_name is None:
            rf_layer = self.layers[layer_id]
        else:
            rf_layer = getattr(self.layers[layer_id].forward_layer, module_name)
        assert hasattr(rf_layer, 'rfs'), (
            'No rf_pool layer found.'
        )
        # get layer_id kwargs
        layer_kwargs = copy.deepcopy(kwargs)
        layer_kwargs.setdefault(layer_id, {})
        layer_kwargs.get(layer_id).update({'output_module': 'pool'})
        # apply layers with output_module as pool for layer_id
        return self.apply(input.detach(), output_layer=layer_id, **layer_kwargs)

    def rf_index(self, input, layer_id, module_name='pool', thr=0.):
        # get heatmap
        heatmap = self.rf_heatmap(layer_id, module_name=module_name)
        # get RFs where any pixel is greater than threshold
        return torch.max(torch.gt(torch.mul(input,heatmap),thr).flatten(-2),-1)[0]

    def rf_heatmap(self, layer_id, module_name='pool',
                   layer_type='forward_layer'):
        """
        Show heatmap of receptive fields in image space (requires
        `rf_pool.pool.Pool` module)

        Parameters
        ----------
        layer_id : str
            layer id of layer containing pooling module
        module_name : str
            name of pooling module within layer [default: 'pool']
        layer_type : str
            layer name containing module [default: 'forward_layer']

        Returns
        -------
        heatmap : torch.Tensor
            heatmap with shape (n_RF, h, w) as a binary map of RF area in image
            space
        """
        if module_name is None:
            rf_layer = self.layers[layer_id]
        elif layer_type is None:
            rf_layer = getattr(self.layers[layer_id], module_name)
        else:
            rf_layer = getattr(getattr(self.layers[layer_id], layer_type),
                               module_name)
        assert hasattr(rf_layer, 'rfs'), (
            'No rf_pool layer found.'
        )
        # get layers up to layer id
        pre_layer_ids = self.get_layer_ids(layer_id)
        # for each layer apply transpose convolution of ones and unpooling
        rf_layer._update_rfs(rf_layer.mu, rf_layer.sigma)
        rfs = torch.unsqueeze(rf_layer.rfs, 1).detach()
        # get modules with kernel_size attribute
        modules = []
        for id in pre_layer_ids:
            modules.extend(self.layers[id].get_modules_by_attr(layer_type,
                                                               'kernel_size'))
        modules.reverse()
        # find rf_layer for start
        idx = [i for i, m in enumerate(modules) if m is rf_layer][0]
        modules = modules[idx+1:]
        # for each module, upsample or transpose convolution
        heatmap = rfs.clone()
        for module in modules:
            # get kernel_size
            size = getattr(module, 'kernel_size')
            if size is None:
                continue
            if type(size) is not tuple:
                size = (size,)*2
            # if pool, upsample
            if isinstance(module, pool.Pool) or \
               torch.typename(module).lower().find('pool') > -1:
                heatmap = torch.nn.functional.interpolate(heatmap,
                                                          scale_factor=size)
            else:
                w = torch.ones((1,1) + size)
                heatmap = torch.conv_transpose2d(heatmap, w)
            heatmap = torch.gt(heatmap, 0.).float()
        return heatmap.squeeze(1)

    def rf_to_image_space(self, layer_id, *args, module_name='pool',
                          layer_type='forward_layer', output_space=False):
        """
        Get image-space coordinates/sizes for RF `mu`/`sigma` values (or other)

        Parameters
        ----------
        layer_id : str
            layer id of layer containing pooling module
        *args : float, int, array-like
            inputs to obtain image-space coordinates/sizes
            [default: `mu`/`sigma` from RF pooling module]
        module_name : str
            name of pooling module within layer [default: 'pool']
        layer_type : str
            layer name containing module [default: 'forward_layer']
        output_space : boolean
            True/False whether to return image-space coordiantes based on
            output of module [default: False]

        Returns
        -------
        coords : list
            torch.Tensor of image-space coordinate/size per arg in *args

        Notes
        -----
        If `output_space is True`, image-space coordinates are in reference to
        the output of the given module. If `output_space is False`, image-space
        coordinates are in reference to the input to the given module (i.e.,
        the module `kernel_size` is not included in the coordinate computation).
        This distinction is useful for `rf_pool.pool` classes which use the
        input image shape as reference for RF locations.
        """
        #TODO: need to update for stride, dilation, etc.
        # start module
        if module_name is None:
            start_module = self.layers[layer_id]
        elif layer_type is None:
            start_module = getattr(self.layers[layer_id], module_name)
        else:
            start_module = getattr(getattr(self.layers[layer_id], layer_type),
                                   module_name)
        # get mu, sigma
        if len(args) == 0:
            args = start_module.get('mu','sigma')
            args = list(args.values())
        # ensure args are tensor
        args = [torch.as_tensor(a) for a in args]
        arg_shapes = [a.shape for a in args]
        # get layers up to layer_id
        layers = self.get_layers(self.get_layer_ids(layer_id))
        # get modules with kernel_size
        modules = []
        for layer in layers:
            modules.extend(layer.get_modules_by_attr('forward_layer',
                                                     'kernel_size'))
        modules.reverse()
        # find start_module
        idx = [i for i, m in enumerate(modules) if m is start_module][0]
        # if using input space, add 1 to idx
        if output_space is False:
            idx += 1
        # get reversed modules from start_module
        modules = modules[idx:]
        # for each module, add half weight and multiply by pool size
        for module in modules:
            size = getattr(module, 'kernel_size')
            if size is None:
                continue
            size = torch.as_tensor(size)
            # if pool, upsample
            if isinstance(module, pool.Pool) or \
               torch.typename(module).lower().find('pool') > -1:
                args = [a * size.type(a.dtype) for a in args]
            else:
                half_kernel = (size - 1) // 2
                args = [a + half_kernel.type(a.dtype) for a in args]
        # ensure same shape by averaging over different dim
        args = [torch.mean(a, -1).reshape(a_shp) if a.shape != a_shp else a
                for (a, a_shp) in zip(args, arg_shapes)]
        return args

class FeedForwardNetwork(Model):
    """
    Feed Forward Network

    Attributes
    ----------
    data_shape : tuple
        shape of input data
    layers : torch.nn.ModuleDict
        layers containing computations to be performed

    Methods
    -------
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train(n_epochs, trainloader, loss_fn, optimizer, monitor=100, **kwargs)
    optimize_texture(n_steps, seed, loss_fn, optimizer, input=[],
                     transform=None, monitor=100, **kwargs)
    """
    def __init__(self, model=None):
        super(FeedForwardNetwork, self).__init__(model)

class DeepBeliefNetwork(Model):
    """
    Deep Belief Network

    Attributes
    ----------
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers in deep belief network (each layer is RBM class)

    Methods
    -------
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
    train_model(n_epochs, trainloader, optimizer, k=1, persistent=None,
                monitor=100, **kwargs)

    References
    ----------
    Hinton, Osindero & Teh (2006)
    """
    def __init__(self, model=None):
        super(DeepBeliefNetwork, self).__init__(model)

    def init_complementary_prior(self, layer_id, module_name='hidden'):
        """
        Initialize complementary priors using transpose of weights from previous
        layer.

        Parameters
        ----------
        layer_id : str
            layer id to initialize weights
        module_name : str
            name of module containing weights to initialize (must match module
            name of previous layer) [defualt: 'hidden']

        Returns
        -------
        None

        Notes
        -----
        This function initializes weights to the transpose of the weights from
        the previous layer mulitplied by an orthonormal transformation matrix
        to project to the number of hidden units in the current layer. This
        allows the current layer to initialize as a complementary prior over the
        previous layer in order to learn more efficiently. See References for
        more information.

        References
        ----------
        Hinton, Osindero & Teh (2006)
        """#TODO: update to allow for different kernel size
        # get weight, bias and transpose bias names
        weight_name = module_name + '_weight'
        bias_name = module_name + '_bias'
        transpose_bias_name = module_name + '_transpose_bias'
        # get previous and current layers
        layer_ids = self.get_layer_ids(layer_id)[-2:]
        prev_layer, layer = self.get_layers(layer_ids)
        # get pervious layer weights and transpose (and flip if 4d)
        w = getattr(prev_layer, weight_name).detach()
        wT = w.transpose(0,1)
        if wT.ndimension() == 4:
            wT = wT.flip((-2,-1))
        # create orthonormal transformation to layer_id to_ch
        w0_shape = getattr(layer, weight_name).shape
        to_ch = w0_shape[0]
        from_ch = wT.shape[0]
        u = functions.modified_gram_schmidt(torch.randn(from_ch, to_ch))
        # create new weights with orthonormal transformation
        w_prior = torch.matmul(wT.flatten(1).t(), u).t()
        w_prior = w_prior.reshape(w0_shape)
        # init layer weights
        layer.init_weights(pattern=weight_name, fn=lambda x: w_prior)
        # init visible bias
        layer.init_weights(pattern=transpose_bias_name,
                           fn=lambda x: getattr(prev_layer, bias_name).detach())

    def _get_free_energy_loss(self, layer_id):
        def free_energy_loss(x):
            fe = torch.mean(self.layers[layer_id].free_energy(x))
            hidsize = torch.as_tensor(self.layers[layer_id].hidden_shape(x.shape))
            hidsize = torch.prod(hidsize.unsqueeze(-1)[2:])
            return torch.div(fe, hidsize)
        return free_energy_loss

    def contrastive_divergence(self, input, k=1):
        #TODO: not technically the same as Hinton, Osindero, Teh (2006)
        #TODO: should unlink rec/gen weights for all but top layer
        #TODO: then update is based on s(d - p) for lower layers
        #TODO: and normal CD for top layer
        # get free_energy functions for each layer
        layer_ids = self.get_layer_ids()
        layer_ids = [None,] + layer_ids
        fe_losses = [self._get_free_energy_loss(layer_id)
                     for layer_id in layer_ids[1:]]
        # get positive phase statistics
        pos_output = OrderedDict([(layer_id, {'sample': []})
                                  if layer_id is not None else (None, [])
                                  for layer_id in layer_ids[:-1]])
        # gibbs sample output layer
        with torch.no_grad():
            output = self.apply(input, output=pos_output, layer_ids=layer_ids)
            # if persistent, use output from persistent pass as top-down input
            if hasattr(self, 'persistent') and self.persistent is not None:
                output = self.apply(self.persistent, layer_ids=layer_ids)
            # gibbs sample
            output = self.layers[layer_ids[-1]].gibbs_hvh(output, k=k)[-1]
        # get negative phase statistics
        neg_output = OrderedDict([(layer_id, {'sample': []})
                                  for layer_id in layer_ids[1:]])
        neg_layer_ids = layer_ids.copy()[1:]
        neg_layer_ids.reverse()
        with torch.no_grad():
            reconstruct = self.apply(output, output=neg_output,
                                     layer_ids=neg_layer_ids, forward=False)
        # update persistent
        if hasattr(self, 'persistent') and self.persistent is not None:
            self.persistent = reconstruct
        # get difference in free_energy
        loss = torch.zeros(1, requires_grad=True)
        for fe_loss_fn, pos, neg in zip(fe_losses, pos_output.values(),
                                        neg_output.values()):
            p_sample = pos.get('sample')[0] if type(pos) is dict else pos[0]
            n_sample = neg.get('sample')[0] if type(neg) is dict else neg[0]
            loss = loss + torch.sub(fe_loss_fn(p_sample), fe_loss_fn(n_sample))
        return loss

    def untie_weights(self, pattern=''):
        """
        Untie `reconstruct_layer` weights from `forward_layer` weights
        for learning directed graphical model

        Parameters
        ----------
        pattern : str
            pattern of weights in `reconstruct_layer` to be untied
            [default: '', all weights in `reconstruct_layer` untied]

        Returns
        -------
        None
        """
        layer_ids = self.get_layer_ids()
        for layer in self.get_layers(layer_ids):
            if hasattr(layer, 'untie_weights'):
                layer.untie_weights(pattern=pattern)

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        #TODO: implement BEAM training (-KL based on NN of data/fantasy particles)
        return self.train(n_epochs, trainloader, None, optimizer, monitor=monitor,
                          layer_id=layer_id, k=k, **kwargs)

    def train_model(self, n_epochs, trainloader, optimizer, k=1, persistent=None,
                    monitor=100, **kwargs):
        # set persistent attribute
        self.persistent = persistent
        # set contrastive_divergence loss
        cd_loss = losses.KwargsLoss(self.contrastive_divergence, n_args=1, k=k)
        loss_fn = losses.LayerLoss(self, {None: []}, cd_loss, layer_ids=[None])
        # train
        return self.train(n_epochs, trainloader, loss_fn, optimizer,
                          monitor=monitor, **kwargs)

class DeepBoltzmannMachine(DeepBeliefNetwork):
    """
    Deep Boltzmann Machine

    Attributes
    ----------
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers for forward/reconstruct pass (each layer is RBM class)

    Methods
    -------
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
        train deep boltzmann machine with contrastive divergence
    train_model(n_epochs, trainloader, optimizer, k=1, n_iter=10, persistent=None,
                monitor=100, **kwargs)

    References
    ----------
    Salakhutdinov & Hinton (2009)
    """
    def __init__(self, model=None):
        super(DeepBoltzmannMachine, self).__init__(model)

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        # update module to double input and double top-down
        layer_ids = self.get_layer_ids()
        mul_op = ops.Op(lambda x: 2. * x)
        act_op0 = self.update_modules([layer_ids[0]], 'forward_layer',
                                      'activation', mul_op, overwrite=False,
                                      append=False)
        if len(layer_ids) > 1 and layer_id == layer_ids[-1]:
            act_op1 = self.update_modules([layer_id], 'reconstruct_layer',
                                          'activation', mul_op, overwrite=False,
                                          append=False)
        # train
        try:
            output = self.train(n_epochs, trainloader, None, optimizer,
                                monitor=monitor, layer_id=layer_id, k=k, **kwargs)
        finally:
            # replace mul_op with original activation operation
            self.update_modules([layer_ids[0]], 'forward_layer', 'activation',
                                act_op0, overwrite=True)
            if len(layer_ids) > 1 and layer_id == layer_ids[-1]:
                self.update_modules([layer_ids[0]], 'reconstruct_layer',
                                    'activation', act_op1, overwrite=True)
        return output

    def train_model(self, n_epochs, trainloader, optimizer, k=1, n_iter=10,
                    persistent=None, monitor=100, **kwargs):
        # set persistent
        self.persistent = persistent
        # set loss function
        cd_loss = losses.KwargsLoss(self.contrastive_divergence, n_args=1,
                                    k=k, n_iter=n_iter)
        loss_fn = losses.LayerLoss(self, {None: []}, cd_loss, layer_ids=[None])
        # train
        return self.train(n_epochs, trainloader, loss_fn, optimizer,
                          monitor=monitor, **kwargs)

    def contrastive_divergence(self, input, k=1, n_iter=10):
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
                pos_energy_i = torch.mean(layer.energy(input, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                pos_energy_i = torch.mean(layer.energy(h_n, hids[i][1]))
            hidsize = torch.prod(torch.as_tensor(hids[i][1].shape[-2:]))
            pos_energy.append(torch.div(pos_energy_i, hidsize))
        # negative phase for each layer
        if hasattr(self, 'persistent') and self.persistent is not None:
            v = self.persistent
        hids = None
        with torch.no_grad():
            for _ in range(k+1):
                even = (hids is not None)
                v, hids = self.layer_gibbs(v, hids, sampled=True, even=even)
        if hasattr(self, 'persistent') and self.persistent is not None:
            self.persistent = v
        # get negative energies
        neg_energy = []
        for i, layer in enumerate(layers):
            if i == 0:
                neg_energy_i = torch.mean(layer.energy(v, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                neg_energy_i = torch.mean(layer.energy(h_n, hids[i][1]))
            hidsize = torch.prod(torch.as_tensor(hids[i][1].shape[-2:]))
            neg_energy.append(torch.div(neg_energy_i, hidsize))
        # return mean difference in energies
        loss = torch.zeros(1, requires_grad=True)
        for (p, n) in zip(pos_energy, neg_energy):
            loss = torch.add(loss, torch.sub(p, n))
        return loss

    def layer_gibbs(self, input, hids=None, sampled=False, pooled=False,
                    even=True, odd=True):
        # get layer_ids
        layer_ids = self.get_layer_ids()
        layers = self.get_layers(layer_ids)
        # get idx based on sampled bool
        idx = np.int(sampled)
        # get hids from forward pass with weights doubled
        if hids is None:
            mul_op = ops.Op(lambda x: 2. * x)
            hid_ops = self.update_modules(layer_ids[:-1], 'forward_layer',
                                          'hidden', mul_op, overwrite=False)
            try: # forward pass with doubled weights
                saved_outputs = dict([(id, {'hidden': []}) for id in layer_ids])
                self.apply(input, layer_ids, output=saved_outputs)
                hids = [v.get('hidden')[0] for v in saved_outputs.values()]
            finally:
                self.update_modules(layer_ids[:-1], 'forward_layer', 'hidden',
                                    hid_ops)
            hids = [(h,) + layer.sample(h, 'forward_layer', pooled)
                    for h, layer in zip(hids, layers)]
        if even:
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
        if odd:
            for i, layer in list(enumerate(layers))[1::2]:
                v = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[idx]
                if i < (len(layers) - 1):
                    t = layers[i+1].sample_v_given_h(hids[i+1][idx+1])[idx+1]
                    hids[i] = layer.sample_h_given_vt(v, t, pooled)
                else:
                    hids[i] = layer.sample_h_given_v(v, pooled)
        return v_out, hids

if __name__ == '__main__':
    import doctest
    doctest.testmod()
