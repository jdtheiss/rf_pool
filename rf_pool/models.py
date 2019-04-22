import pickle

import IPython.display
from IPython.display import clear_output, display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .modules import RBM
from .utils import functions

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

    def apply_layers(self, input, layer_ids, forward=True):
        if not forward:
            layer_ids.reverse()
        layers = self.get_layers(layer_ids)
        for layer in layers:
            if forward:
                input = layer.forward(input)
            else:
                input = layer.reconstruct(input)
        return input

    def forward(self, input):
        for name, layer in self.layers.named_children():
            input = layer.forward(input)
        return input

    def reconstruct(self, input):
        for name, layer in self.layers.named_children():
            input = layer.reconstruct(input)
        return input

    def pre_layer_ids(self, layer_id):
        # get layer_ids prior to layer_id
        layer_id = str(layer_id)
        pre_layer_ids = []
        for pre_layer_id, _ in self.layers.named_children():
            if pre_layer_id != layer_id:
                pre_layer_ids.append(pre_layer_id)
            else:
                break
        return pre_layer_ids

    def post_layer_ids(self, layer_id):
        # get layer_ids after layer_id
        layer_id = str(layer_id)
        # append layer ids starting at end, then reverse post_layer_ids
        layer_ids = [name for name, _ in self.layers.named_children()]
        layer_ids.reverse()
        post_layer_ids = []
        for post_layer_id in layer_ids:
            if post_layer_id != layer_id:
                post_layer_ids.append(post_layer_id)
            else:
                break
        post_layer_ids.reverse()
        return post_layer_ids

    def save_model(self, filename, extras=[]):
        if type(extras) is not list:
            extras = [extras]
        model_dict = self.download_weights()
        with open(filename, 'wb') as f:
            pickle.dump([model_dict,] + extras, f)

    def load_model(self, filename):
        model = pickle.load(open(filename, 'rb'))
        if type(model) is list:
            model = model[0]
        if type(model) is dict:
            self.load_weights(model)
            model = self
        return model

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
            for field in fields[:-1]:
                layer = getattr(layer, field)
            # get param name in model_dict
            if param_dict.get(name):
                model_key = param_dict.get(name)
            elif model_dict.get(name) is not None:
                model_key = name
            else: # skip param
                continue
            # register parameter
            new_param = torch.nn.Parameter(torch.as_tensor(model_dict.get(model_key)))
            layer.register_parameter(fields[-1], new_param)

    def init_weights(self, suffix='weight', fn=torch.randn_like):
        for name, param in self.named_parameters():
            with torch.no_grad():
                if name.endswith(suffix):
                    param.set_(fn(param))

    def get_trainable_params(self, pattern=''):
        trainable_params = []
        for (name, param) in self.named_parameters():
            if param.requires_grad == True and name.find(pattern) >=0:
                trainable_params.append(param)

        return trainable_params

    def get_param_names(self):
        param_names = []
        for (name, param) in self.named_parameters():
            param_names.append(name)
        return param_names

    def set_requires_grad(self, pattern, requires_grad=True):
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
        # additional loss
        if 'add_loss' in kwargs.keys():
            add_loss = kwargs.pop('add_loss')
        else:
            add_loss = {}
        # sparsity
        if 'sparsity' in kwargs.keys():
            sparsity = kwargs.pop('sparsity')
        else:
            sparsity = {}
        loss_history = []
        running_loss = 0.
        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader):
                # get inputs, zero gradients
                inputs = data[:-1]
                label = data[-1]
                optimizer.zero_grad()
                # get outputs
                output = self.forward(inputs[0])
                # get loss
                loss = loss_fn(output, label)
                # additional loss
                if add_loss:
                    added_loss = self.add_loss(inputs, **add_loss)
                    loss = loss + added_loss
                # sparsity
                if sparsity:
                    self.sparsity(inputs[0], **sparsity)
                loss.backward()
                # update parameters
                optimizer.step()
                running_loss += loss.item()
                if (i+1) % monitor == 0:
                    # display loss
                    clear_output(wait=True)
                    display('[%5d] loss: %.3f' % (i+1, running_loss / monitor))
                    # append loss and show history
                    loss_history.append(running_loss / monitor)
                    plt.plot(loss_history)
                    plt.show()
                    running_loss = 0.
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
        self.set_requires_grad('', requires_grad=False)
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
                display('[%5d] loss: %.3f' % (i+1, running_loss / monitor))
                # append loss and show history
                loss_history.append(running_loss / monitor)
                plt.plot(loss_history)
                plt.show()
                running_loss = 0.
                # monitor texture
                if show_texture:
                    self.show_texture(*show_texture)
                # functions.kwarg_fn([IPython.display, self], None, **kwargs)
        # turn on model gradients
        self.set_requires_grad('', requires_grad=True)
        return seed

    def add_loss(self, inputs, loss_fn, layer_ids, module_name=None, **kwargs):
        """
        #TODO:WRITEME
        """
        #TODO: give option for specific optimizer
        # get loss/update inputs for each layer
        loss = 0.
        for i, (name, layer) in enumerate(self.layers.named_children()):
            if name in layer_ids:
                kwargs_i = {}
                for key, value in kwargs.items():
                    if type(value) is list:
                        kwargs_i.update({key: value[i]})
                    else:
                        kwargs_i.update({key: value})
                loss += layer.add_loss(inputs, loss_fn, module_name, **kwargs_i)
            for ii, input in enumerate(inputs):
                inputs[ii] = layer.forward(input)
        return loss

    def sparsity(self, input, layer_ids, module_name, target, cost=1.):
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
                layer.sparsity(input, module_name, target_i, cost_i)
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
            pool = layer.get_modules(layer.forward_layer, ['pool'])
            if len(pool) ==1 and torch.typename(pool[0]).find('layers') >=0:
                rf_layers.append(pool[0])
        n_lattices =  len(rf_layers)
        if n_lattices == 0:
            raise Exception('No rf_pool layers found.')

        # pass x through network, show lattices
        with torch.no_grad():
            if type(x) is torch.Tensor:
                self.forward(x)
            # get lattices
            lattices = []
            for pool in rf_layers:
                pool.show_lattice(x, figsize, cmap)

    def show_weights(self, layer_id, img_shape=None, figsize=(5, 5), cmap=None):
        """
        #TODO:WRITEME
        """
        # get weights reconstructed down if not first layer
        layer_id = str(layer_id)
        pre_layer_ids = self.pre_layer_ids(layer_id)
        w = self.layers[layer_id].hidden_weight.clone().detach()
        if len(pre_layer_ids) > 0:
            w[w < 0.] = 0.
            w[w > 1.] = 1.
            w = self.apply_layers(w, pre_layer_ids, forward=False)
        # if channels > 3, reshape
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
    train(layer_id, n_epochs, trainloader, optimizer, monitor=100, **kwargs)
        train deep belief network with greedy layer-wise contrastive divergence

    References
    ----------
    #TODO:WRITEME
    """
    def __init__(self):
        super(DeepBeliefNetwork, self).__init__()

    def show_negative(self, v, layer_id, k=1, img_shape=None, figsize=(5,5),
                      cmap=None):
        """
        #TODO:WRITEME
        """
        layer_id = str(layer_id)
        pre_layer_ids = self.pre_layer_ids(layer_id)
        # prop up, gibbs sample, then reconstruct down
        with torch.no_grad():
            neg = self.apply_layers(v, pre_layer_ids)
            neg = self.layers[layer_id].gibbs_vhv(neg, k=k)[-2]
            neg = self.apply_layers(neg, pre_layer_ids, forward=False)
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

    def train(self, layer_id, n_epochs, trainloader, optimizer, monitor=100,
              show_negative={}, **kwargs):
        """
        #TODO:WRITEME
        """
        # get first layer ids before layer_id
        layer_id = str(layer_id)
        pre_layer_ids = self.pre_layer_ids(layer_id)
        # get number of training examples
        n_train = len(trainloader)
        # additional loss
        if 'add_loss' in kwargs.keys():
            add_loss = kwargs.pop('add_loss')
        else:
            add_loss = {}
        # sparsity
        if 'sparsity' in kwargs.keys():
            sparsity = kwargs.pop('sparsity')
        else:
            sparsity = {}
        # train for n_epochs
        loss_history = []
        running_loss = 0.
        for epoch in range(n_epochs):
            for i, data in enumerate(trainloader):
                inputs = data[:-1]
                optimizer.zero_grad()
                # get inputs for layer_id
                layer_input = self.apply_layers(inputs[0], pre_layer_ids)
                # train
                loss = self.layers[layer_id].train(layer_input, add_loss=add_loss,
                                                   sparsity=sparsity, **kwargs)
                # update parameters
                optimizer.step()
                running_loss += loss
                # monitor loss, show negative and weights
                if (epoch * n_train + i+1) % monitor == 0:
                    # display loss
                    clear_output(wait=True)
                    display('[%5d] loss: %.3f' % (i+1, running_loss / monitor))
                    # append loss and show history
                    loss_history.append(running_loss / monitor)
                    plt.plot(loss_history)
                    plt.show()
                    running_loss = 0.
                    # call other monitoring functions
                    if show_negative:
                        self.show_negative(inputs[0], layer_id, **show_negative)
                    functions.kwarg_fn([IPython.display, self], None, **kwargs)
        return loss_history

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
