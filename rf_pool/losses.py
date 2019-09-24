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
                 parameters=None, target=None, **kwargs):
        super(LayerLoss, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.layer_ids = layer_ids
        self.module_name = functions.parse_list_args(len(layer_ids), module_name)[0]
        self.module_name = [m[0] for m in self.module_name]
        self.cost = functions.parse_list_args(len(layer_ids), cost)[0]
        self.cost = [c[0] for c in self.cost]
        self.parameters = parameters
        self.target = target
        self.kwargs = functions.parse_list_args(len(layer_ids), **kwargs)[1]
        # set target to features if given
        if self.target is not None:
            self.target = self.get_features(self.target)

    def get_features(self, *args):
        args = list(args)
        # turn on parameters
        if self.parameters:
            on_parameters = self.model.get_trainable_params()
            self.model.set_requires_grad(pattern='', requires_grad=False)
            self.model.set_requires_grad(parameters, requires_grad=True)
        # get features
        feat = []
        i = 0
        for (name, layer) in self.model.layers.named_children():
            if name in self.layer_ids:
                feat.append([layer.apply_modules(arg, 'forward_layer',
                             output_module=self.module_name[i], **self.kwargs[i])
                             for arg in args])
                i += 1
            for ii, arg in enumerate(args):
                args[ii] = layer.forward(arg)
        # turn off parameters
        if self.parameters:
            self.model.set_requires_grad(self.parameters, requires_grad=False)
            self.model.set_requires_grad(on_parameters, requires_grad=True)
        return feat

    def forward(self, *args):
        feat = self.get_features(*args)
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat):
            if self.target is not None:
                loss_i = self.loss_fn(*feat_i, *self.target[i])
            else:
                loss_i = self.loss_fn(*feat_i)
            loss = loss + loss_i * self.cost[i]
        return loss / len(self.layer_ids)
