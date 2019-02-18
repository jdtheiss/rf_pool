import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
from IPython.display import clear_output, display
import matplotlib.pyplot as plt 
import pickle
from modules import FeedForwardNetwork, ControlNetwork

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
    show_lattice()
        notImplemented
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
        self.set_loss_fn() # set loss function
        self.optimizer = None # optimizer must be set after the graph is initialized

    def set_loss_fn(self):
        if type(self.loss_type) is not str: 
            self.loss_name = torch.typname(self.loss_type)
        else:
            self.loss_name = self.loss_type

        # choose loss function
        if self.loss_name.lower() == 'cross_entropy':
            self.loss_criterion = nn.CrossEntropyLoss()
        elif self.loss_name.lower() == 'squared_error':
            self.loss_criterion = nn.MSELoss()
        elif self.loss_name.startswith('torch.nn.modules.loss'):
            self.loss_criterion = self.loss_type()
        else:
            raise Exception('loss_type not understood')

    def set_optimizer(self, **kwargs):
        trainable_params = self.get_trainable_params()

        if type(self.optimizer_type) is not str:
            self.optimizer_name = torch.typename(self.optimizer_type)
        else:
            self.optimizer_name = self.optimizer_type

        # choose optimizer 
        if self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(trainable_params, **kwargs)
        elif self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, **kwargs) 
        elif self.optimizer_name.startswith('torch.optim'):
            self.optimizer = self.optimizer_type(trainable_params, **kwargs)
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

    def show_lattice(self):
        raise NotImplementedError

    def get_trainable_params(self):
        #grabs only parameters with 'requires_grad' set to True
        trainable_params = []
        for (name, param) in self.net.named_parameters():
            if param.requires_grad == True:
                trainable_params.append(param)

        return trainable_params

    def set_requires_grad(self, net_type, requires_grad = True):
        if net_type not in ["hidden", "control"]:
            raise Exception("net_type not understood")
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

    def train_model(self, epochs, trainloader, monitor=2000, **kwargs):
        assert self.loss_criterion is not None, (
            "loss function must be initialized before training")
        assert self.net is not None, (
            "network must be initialized before training")

        #initialize optimizer for training
        self.set_optimizer(**kwargs)
        # train the model
        self.running_loss = 0
        for epoch in range(epochs):
            for i, data in  enumerate(trainloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()
                if (i+1) % monitor == 0:
                    clear_output(wait=True)
                    display('[%d, %5d] loss: %.3f' % (epoch , i, self.running_loss / monitor))
                    self.running_loss = 0.0


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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
