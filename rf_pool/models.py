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
        self.set_loss_fn(self.loss_type) # set loss function
        self.optimizer = None # optimizer must be set after the graph is initialized

    def set_loss_fn(self, loss_type):
        if type(loss_type) is not str: 
            loss_name = torch.typname(loss_type)
        else:
            loss_name = loss_type

        # choose loss function
        if loss_name.lower() == 'cross_entropy':
            self.loss_criterion = nn.CrossEntropyLoss()
        elif loss_name.lower() == 'squared_error':
            self.loss_criterion = nn.MSELoss()
        elif loss_name.startswith('torch.nn.modules.loss'):
            self.loss_criterion = loss_type()
        else:
            raise Exception('loss_type not understood')

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
            self.optimizer = optim.SGD(params, **kwargs)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(params, **kwargs) 
        elif optimizer_name.startswith('torch.optim'):
            self.optimizer = optimizer_type(params, **kwargs)
        else:
            raise Exception("optimizer_type not understood")

    def save_model(self, filename, extras=[]):
        if type(extras) is not list:
            extras = [extras]
        with open(filename, 'wb') as f:
            pickle.dump([self,] + extras, f)

    def load_model(self, filename):
        model = pickle.load(open(filename, 'rb'))
        return model

    def monitor_loss(self, loss):
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        self.loss_history.append(loss)
        plt.plot(self.loss_history)
        plt.show()
    
    def show_lattice(self, x, figsize=(10,10)):
        assert self.net.control_nets is not None, (
            "control network must be activated to show lattice")

        n_examples = x.shape[0]
        n_lattices =  len(self.net.control_nets.layer_ids)
        fig, ax = plt.subplots(n_examples, 1+n_lattices, figsize=figsize)

        # adjust the pooling layers
        with torch.no_grad():
            self.net(x)
            for batch_id in range(n_examples):
                img = x[batch_id]
                img =  img / 2 + 0.5 # unnormalize
                img = img.numpy()
                img = np.transpose(img, (1, 2, 0))
                ax[batch_id, 0].imshow(img)

                for i, layer_id in enumerate(self.net.control_nets.layer_ids):
                    rfs = self.net.pool_layers[layer_id].inputs['rfs']
                    lattice_layer = lattice.make_kernel_lattice(rfs)
                    ax[batch_id, i+1].imshow(lattice_layer[batch_id])                 
        plt.show()

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

    def train_model(self, epochs, trainloader, lr=0.001, monitor=2000,
                    monitor_loss=False, monitor_lattice=False, **kwargs):
        assert self.loss_criterion is not None, (
            "loss function must be initialized before training")
        assert self.net is not None, (
            "network must be initialized before training")

        #initialize optimizer for training
        kwargs.update({'lr':lr})
        self.set_optimizer(self.optimizer_type, **kwargs)
        # train the model
        running_loss = 0.
        for epoch in range(epochs):
            for i, data in  enumerate(trainloader, 0):
                # get inputs , labels
                inputs, labels = data
                # zero grad, get outputs
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                # get loss
                loss = self.loss_criterion(outputs, labels)
                # add penalty to loss
                if hasattr(self, 'penalty_attr'):
                    loss += self.loss_penalty(self.penalty_attr, 
                                              self.penalty_cost,
                                              self.penalty_type)
                loss.backward()
                # update parameters
                self.optimizer.step()
                running_loss += loss.item()
                # monitor loss, lattice
                if (i+1) % monitor == 0:
                    clear_output(wait=True)
                    display('[%d, %5d] loss: %.3f' % (epoch, i+1, running_loss / monitor))
                    if monitor_loss:
                        self.monitor_loss(running_loss / monitor)
                    if monitor_lattice:
                        self.show_lattice(inputs)
                    # reset running_loss
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

    def control_network(self, *args, **kwargs):
        assert self.net is not None,  (
            "Feed forward network must be initialized before the control network(s)")

        self.net.control_nets = ControlNetwork(self.net, *args, **kwargs)

    def loss_penalty(self, attr, cost, penalty_type):
        assert hasattr(self.net, attr), (
            'network does not have attribute')
        
        # get value for attribute
        value = getattr(self.net, attr)
        if penalty_type == 'L1':
            penalty_fn = torch.abs

        # add penalties
        penalty = 0.
        if type(value) is list:
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
