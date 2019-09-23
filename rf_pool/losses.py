import torch
import torch.nn as nn

from .utils import functions

class Loss(nn.Module):
    """
    """
    def __init__(self):
        super(Loss, self).__init__()

class MultiLoss(Loss):
    """
    """
    def __init__(self, losses=[], weights=[]):
        super(MultiLoss, self).__init__()
        self.losses = losses
        self.weights = weights
        if len(losses) > len(weights):
            for i in range(len(losses) - len(weights)):
                self.weights.append(1.)
        assert len(self.losses) == len(self.weights)

    def add_loss(self, loss, weight=1.):
        self.losses.append(loss)
        self.weights.append(weight)

    def forward(self, *args):
        list_args = functions.parse_list_args(len(self.losses), *args)[0]
        loss = torch.zeros(1, requires_grad=True)
        for loss_fn, weight, args_i in zip(self.losses, self.weights, list_args):
            loss = loss + torch.mul(loss_fn(*args_i), weight)
        return loss

class LayerLoss(Loss):
    """
    """
    def __init__(self, model, loss_fn, layer_ids, module_name=None, cost=1.,
                 parameters=None, **kwargs):
        super(LayerLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.layer_ids = layer_ids
        self.module_name = module_name
        self.cost = cost
        self.parameters = parameters
        self.kwargs = functions.parse_list_args(len(self.layer_ids), **kwargs)[1]

    def forward(self, *args):
        args = list(args)
        if self.parameters:
            on_parameters = self.model.get_trainable_params()
            self.model.set_requires_grad(pattern='', requires_grad=False)
            self.model.set_requires_grad(parameters, requires_grad=True)
        loss = torch.zeros(1, requires_grad=True)
        i = 0
        for (name, layer) in self.model.layers.named_children():
            if name in self.layer_ids:
                loss = loss + layer.add_loss(args, self.loss_fn, self.module_name,
                                             **self.kwargs[i])
                i += 1
            for ii, input in enumerate(args):
                args[ii] = layer.forward(input)
        loss = torch.mul(loss, self.cost)
        if self.parameters:
            self.model.set_requires_grad(self.parameters, requires_grad=False)
            self.model.set_requires_grad(on_parameters, requires_grad=True)
        return loss
