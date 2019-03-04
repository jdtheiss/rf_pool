import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import pickle
from modules import FeedForwardNetwork, ControlNetwork
import utils.lattice as lattice

class Model(nn.Module):
    """
    Base class for initializing, training, saving, loading, visualizing, and running models

    Attributes
    ----------
    loss_type : str or torch.nn.modules.loss
        cost function choice
    optimizer_type : str or torch.optim
        optimizer choice

    Methods
    -------
    set_loss_fn()
        sets a torch.nn.modules.loss function
    set_optimizer(**kwargs)
        sets a torch.optim optimizer
        see torch.optim for **kwargs
    load_model(filename)
        loads a previously saved model from filename
    save_model(filename, extras = [])
        saves a model instance
    show_lattice(x, figsize=(10,10))
        shows the lattice for each layer given input x
    get_trainable_params()
        gets the trainable parameters from the network
    set_requires_grad(net_type, requires_grad)
        sets net_type params to require gradient or not
    get_accuracy(dataLoader)
        gets the model's accuracy given a torch.utils.data.DataLoader
    train_model(epochs, trainloader, monitor=2000, **kwargs)
        trains the model with a given torch.utils.data.DataLoader

    """
    def __init__(self, loss_type, optimizer_type):
        super(Model, self).__init__()
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.net = None

    def set_loss_fn(self, loss_type):
        if type(loss_type) is not str:
            loss_name = torch.typname(loss_type)
        else:
            loss_name = loss_type

        # choose loss function
        if loss_name.lower() == 'cross_entropy':
            loss_criterion = nn.CrossEntropyLoss()
        elif loss_name.lower() == 'squared_error':
            loss_criterion = nn.MSELoss()
        elif loss_name.startswith('torch.nn.modules.loss'):
            loss_criterion = loss_type()
        else:
            raise Exception('loss_type not understood')

        return loss_criterion

    def set_optimizer(self, optimizer_type, prefix=[''], **kwargs):
        # set params dict for main, control networks
        params = []
        for net_prefix in prefix:
            params.append({'params': self.get_trainable_params(net_prefix)})

        # remove kwargs and set to params if list
        removed_keys = []
        for (key, value) in kwargs.items():
            if type(value) is list:
                removed_keys.append(key)
                for i, v in enumerate(value):
                    params[i].update({key: v})
        [kwargs.pop(key) for key in removed_keys]

        # get typename for optimizer
        if type(optimizer_type) is not str:
            optimizer_name = torch.typename(optimizer_type)
        else:
            optimizer_name = optimizer_type

        # choose optimizer
        if optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(params, **kwargs)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(params, **kwargs)
        elif optimizer_name.startswith('torch.optim'):
            optimizer = optimizer_type(params, **kwargs)
        else:
            raise Exception("optimizer_type not understood")

        return optimizer

    def save_model(self, filename, extras=[]):
        if type(extras) is not list:
            extras = [extras]
        with open(filename, 'wb') as f:
            pickle.dump([self,] + extras, f)

    def load_model(self, filename):
        model = pickle.load(open(filename, 'rb'))
        return model

    def load_weights(self):
        raise NotImplementedError

    def monitor_loss(self, loss, iter, **kwargs):
        if not hasattr(self, 'loss_history') or 'reset' in kwargs:
            self.loss_history = []

        # display loss
        clear_output(wait=True)
        display('[%5d] loss: %.3f' % (iter, loss))

        # append loss and show history
        self.loss_history.append(loss)
        plt.plot(self.loss_history)
        plt.show()

        # pass kwargs to other functions
        for key, value in kwargs.items():
            if hasattr(self, key):
                fn = getattr(self, key)
                if type(value) is list:
                    fn(*value)
                elif type(value) is dict:
                    fn(**value)
                else:
                    raise Exception('type not understood')

    def show_texture(self, input_image, seed_image):
        assert input_image.shape == seed_image.shape, (
            'input_image and seed_image shapes must match'
        )
        # get number of input images
        n_images = input_image.shape[0]

        # set cmap
        if input_image.shape[0] == 3:
            cmap = None
        else:
            cmap = 'gray'

        # permute, squeeze input_image and seed_image
        input_image = torch.squeeze(input_image.permute(0,2,3,1), -1)
        seed_image = torch.squeeze(seed_image.permute(0,2,3,1), -1)

        # init figure, axes
        fig, ax = plt.subplots(n_images, 2)
        for n in range(n_images):
            ax[n,0].imshow(input_image[n], cmap=cmap)
            ax[n,1].imshow(seed_image[n], cmap=cmap)
        plt.show()

    def show_lattice(self, x=None, figsize=(5,5), cmap=None):
        # get lattice_ids
        lattice_ids = [layer_id for i, layer_id in enumerate(self.net.layer_ids)
                        if self.net.pool_names[i].find('layers') >= 0]
        n_lattices =  len(lattice_ids)
        if n_lattices == 0:
            raise Exception('No rf_pool layers found.')

        # pass x through network, show lattices
        with torch.no_grad():
            if type(x) is torch.Tensor:
                self.net(x)
            # get lattices
            lattices = []
            for i, layer_id in enumerate(lattice_ids):
                rfs = self.net.pool_layers[layer_id].inputs['rfs']
                lattices.append(lattice.make_kernel_lattice(rfs))
            # show lattices
            lattice.show_kernel_lattice(lattices, x, figsize, cmap)

    def get_trainable_params(self, prefix=''):
        # set prefix to 'hidden_layers' to grab only net params
        # or set prefix to 'control_nets' to grab only control params
        #grabs only parameters with 'requires_grad' set to True
        trainable_params = []
        for (name, param) in self.net.named_parameters():
            if param.requires_grad == True and name.startswith(prefix):
                trainable_params.append(param)

        return trainable_params

    def get_param_names(self):
        param_names = []
        for (name, param) in self.net.named_parameters():
            param_names.append(name)
        return param_names

    def set_requires_grad(self, net_type, requires_grad = True):
        # set a net's parameters to require gradient or not
        for (name, param) in self.net.named_parameters():
            if name.startswith(net_type):
                param.requires_grad = requires_grad

    def get_accuracy(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def optimize_image(self, input_image, n_steps, layer_ids, lr=0.001, monitor=2000,
                       monitor_texture=False, **kwargs):
        seed_image = torch.rand_like(input_image, requires_grad = True)
        self.set_requires_grad("hidden_layers", requires_grad = False)

        kwargs.update({'lr':lr})
        optimizer = self.set_optimizer(self.optimizer_type, **kwargs)
        loss_criterion = self.set_loss_fn(self.loss_type)

        with torch.no_grad():
            self.net(input_image)
            fm_input = [self.net.layer_out[layer_id] for layer_id in layer_ids]

        running_loss = 0.
        for i in range(n_steps):
            optimizer.zero_grad()

            self.net(seed_image)
            fm_seed = [self.net.layer_out[layer_id] for layer_id in layer_ids]

            loss = torch.zeros(1)
            for fm_s, fm_i in zip(fm_seed, fm_input):
                loss += loss_criterion(fm_s, fm_i)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % monitor == 0:
                if monitor_texture:
                    show_texture_args = [input_image, seed_image]
                else:
                    show_texture_args = []
                self.monitor_loss(running_loss, i+1, monitor,
                                show_texture=show_texture_args)
                running_loss = 0.

    def train_model(self, epochs, trainloader, lr=0.001, monitor=2000,
                    monitor_lattice=False, **kwargs):
        assert self.net is not None, (
            "network must be initialized before training")

        # get kwargs for show_lattice
        show_lattice_kwargs = {}
        if 'figsize' in kwargs:
            show_lattice_kwargs['figsize'] = kwargs.pop('figsize')
        if 'cmap' in kwargs:
            show_lattice_kwargs['cmap'] = kwargs.pop('cmap')
        #initialize optimizer for training
        kwargs.update({'lr':lr})
        optimizer = self.set_optimizer(self.optimizer_type, **kwargs)
        loss_criterion = self.set_loss_fn(self.loss_type)
        # train the model
        running_loss = 0.
        for epoch in range(epochs):
            for i, data in  enumerate(trainloader, 0):
                # get inputs , labels
                inputs, labels = data
                # zero grad, get outputs
                optimizer.zero_grad()
                outputs = self.net(inputs)
                # get loss
                loss = loss_criterion(outputs, labels)
                # add penalty to loss
                if hasattr(self, 'penalty_attr'):
                    loss += self.loss_penalty(self.penalty_attr,
                                              self.penalty_cost,
                                              self.penalty_type)
                loss.backward()
                # update parameters
                optimizer.step()
                running_loss += loss.item()
                # monitor loss, show lattice
                if (i+1) % monitor == 0:
                    if monitor_lattice:
                        show_lattice_kwargs['x'] = inputs
                    else:
                        show_lattice_kwargs = {}
                    self.monitor_loss(running_loss / monitor, i+1,
                        show_lattice=show_lattice_kwargs)
                    running_loss = 0.

class FeedForwardModel(Model):
    """
    Attributes
    ----------
    net : FeedForwardNetwork
        network for running the model

    Methods
    -------
    ff_network(*args)
        adds a FeedForwardNetwork to the graph
        see modules.FeedForwardNetwork for arguments
    control_network(*args)
        adds a ControlNetwork to the graph
        see modules.ControlNetwork for arguments
    """
    def __init__(self, loss_type, optimizer_type):
        super(FeedForwardModel, self).__init__(loss_type, optimizer_type)
        self.net = None

    def __call__(self, x):
        return self.net(x)

    def ff_network(self, *args, **kwargs):
        self.net = FeedForwardNetwork(*args, **kwargs)

    def control_network(self, layer_id, *args, **kwargs):
        assert self.net is not None,  (
            "Feed forward network must be initialized before the control network(s)")
        if not hasattr(self.net, 'control_nets') or not self.net.control_nets:
            self.net.control_nets = nn.ModuleDict()

        self.net.control_nets.add_module(str(layer_id), ControlNetwork(self.net, layer_id, *args, **kwargs))

    def loss_penalty(self, attr, cost, penalty_type):
        # get value from attribute
        attr = attr.split('.')
        value = self.net
        for a in attr:
            assert hasattr(value, a), (
                'network does not have attribute')
            value = getattr(value, a)

        # get value as list
        if hasattr(value, 'values'):
            dict_values = value.values()
            value = []
            for v in dict_values:
                if type(v) is list:
                    value.extend(v)
                else:
                    value.append(v)

        # get penalty function
        if penalty_type == 'L1':
            penalty_fn = torch.abs

        # add penalties
        penalty = 0.
        if type(value) is list:
            assert len(value) != 0, (
                'no value found for attribute'
            )
            for v in value:
                penalty = torch.add(penalty, torch.sum(penalty_fn(v)))
        else:
            penalty = penalty_fn(value)

        return torch.as_tensor(torch.mul(penalty, cost), dtype=torch.float32)

    def set_penalty_fn(self, attr, cost=1., penalty_type='L1'):
        self.penalty_attr = attr
        self.penalty_cost = cost
        self.penalty_type = penalty_type

if __name__ == "__main__":
    import doctest
    doctest.testmod()
