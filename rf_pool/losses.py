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

class ArgLoss(Loss):
    """
    """
    def __init__(self, loss_fn, input, target):
        super(ArgLoss, self).__init__()
        self.loss_fn = loss_fn
        self.input = input
        self.target = target

    def forward(self, *args):
        return self.loss_fn(self.input, self.target)

class KwargsLoss(Loss):
    """
    """
    def __init__(self, loss_fn, n_args=1, **kwargs):
        super(KwargsLoss, self).__init__()
        self.loss_fn = loss_fn
        self.n_args = n_args
        self.kwargs = kwargs

    def forward(self, *args):
        return self.loss_fn(*args[:self.n_args], **self.kwargs)

class VarKernelLoss(Loss):
    """
    """
    def __init__(self, kernel_size=2, stride=1, **kwargs):
        super(VarKernelLoss, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.kwargs = kwargs
        self.loss_fn = self.var_kernel_loss

    def var_kernel_loss(self, x):
        m = torch.nn.functional.avg_pool2d(x, self.kernel_size,
                                           stride=self.stride,
                                           **self.kwargs)
        m = torch.nn.functional.interpolate(m, size=x.shape[-2:])
        d = torch.sub(x, m)
        return torch.nn.functional.lp_pool2d(d, 2, self.kernel_size,
                                             stride=self.stride,
                                             **self.kwargs).mean()

    def forward(self, *args):
        return self.loss_fn(*args)

class FeatureLoss(Loss):
    """
    """
    def __init__(self, model, layer_id, feature_indices, loss_fn, cost=1.,
                 parameters=None, input_target=None, target=None, **kwargs):
        super(FeatureLoss, self).__init__()
        self.model = model
        self.layer_id = layer_id
        self.feature_indices = feature_indices
        self.loss_fn = loss_fn
        self.cost = cost
        self.parameters = parameters
        self.input_target = input_target
        self.target = target
        self.kwargs = kwargs
        if self.input_target is not None:
            self.input_target = self.get_features(self.input_target)

    def get_features(self, input):
        # turn on parameters
        if self.parameters:
            on_parameters = self.model.get_trainable_params()
            self.model.set_requires_grad(pattern='', requires_grad=False)
            self.model.set_requires_grad(self.parameters, requires_grad=True)
        # get features for layer_id
        layer_ids = self.model.get_layer_ids(self.layer_id)
        output = self.model.apply_layers(input, layer_ids,
                                         **self.kwargs)
        # sum across image space
        output = torch.mean(output, [-2,-1])
        # turn off parameters
        if self.parameters:
            self.model.set_requires_grad(self.parameters, requires_grad=False)
            self.model.set_requires_grad(on_parameters, requires_grad=True)
        return output

    def forward(self, *args):
        if self.target is None and self.input_target is None:
            feat = self.get_features(*args)
        else:
            feat = self.get_features(args[0])
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat.t()):
            if i not in self.feature_indices:
                continue
            if self.input_target is not None:
                loss_i = self.loss_fn(feat_i, self.input_target)
            elif self.target is not None:
                loss_i = self.loss_fn(feat_i, self.target)
            else:
                loss_i = self.loss_fn(*feat_i)
            loss = loss + loss_i * self.cost
        return loss

class LayerLoss(Loss):
    """
    """
    def __init__(self, model, layer_ids, loss_fn, module_name=None, cost=1.,
                 parameters=None, input_target=None, target=None, **kwargs):
        super(LayerLoss, self).__init__()
        self.model = model
        self.layer_ids = layer_ids
        self.loss_fn = loss_fn
        self.module_name = functions.parse_list_args(len(layer_ids), module_name)[0]
        self.module_name = [m[0] for m in self.module_name]
        self.cost = functions.parse_list_args(len(layer_ids), cost)[0]
        self.cost = [c[0] for c in self.cost]
        self.parameters = parameters
        self.input_target = input_target
        self.target = target
        self.kwargs = functions.parse_list_args(len(layer_ids), **kwargs)[1]
        # set input_target to features if given
        if self.input_target is not None:
            self.input_target = self.get_features(self.input_target)

    def get_features(self, *args):
        args = list(args)
        # turn on parameters
        if self.parameters:
            on_parameters = self.model.get_trainable_params()
            self.model.set_requires_grad(pattern='', requires_grad=False)
            self.model.set_requires_grad(self.parameters, requires_grad=True)
        # get features
        feat = []
        i = 0
        for (name, layer) in self.model.layers.named_children():
            if name in self.layer_ids:
                feat.append([])
                for arg in args:
                    output = layer.apply_modules(arg, 'forward_layer',
                                                 output_module=self.module_name[i],
                                                 **self.kwargs[i])
                    if type(output) is list:
                        output = torch.cat([torch.flatten(o) for o in output])
                    feat[-1].append(output)
                i += 1
            for ii, arg in enumerate(args):
                args[ii] = layer.forward(arg)
        # turn off parameters
        if self.parameters:
            self.model.set_requires_grad(self.parameters, requires_grad=False)
            self.model.set_requires_grad(on_parameters, requires_grad=True)
        return feat

    def forward(self, *args):
        if self.target is None and self.input_target is None:
            feat = self.get_features(*args)
        else:
            feat = self.get_features(args[0])
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat):
            if self.input_target is not None:
                loss_i = self.loss_fn(*feat_i, *self.input_target[i])
            elif self.target is not None:
                loss_i = self.loss_fn(*feat_i, self.target)
            else:
                loss_i = self.loss_fn(*feat_i)
            loss = loss + loss_i * self.cost[i]
        return loss / len(self.layer_ids)

class SparseLoss(LayerLoss):
    """
    Encourage sparsity by taking gradient of constraint function

    Parameters
    ----------
    model : rf_pool.models.Model
        model used to compute features at layer_ids
    layer_ids : list
        layer ids corresponding to layers within model for which features will be
        computed
    loss_fn : string or function
        type of sparsity constraint: 'cross_entropy' (Lee et al., 2008),
        'log_sum' (Ji et al., 2014), 'lasso', 'group_lasso' (Yuan & Lin, 2006)
        [default: 'cross_entropy']
    module_name : string or None
        output module for passing input through forward layer
        [default: None, no modules are applied to the input]
    cost : float
        sparsity-cost determining how much the constraint is applied to weights
        [default: 1.]
    decay : float
        decay of running average [default: 0., only current input is used]
    target : float or None, optional
        sparsity target for type='cross_entropy'
    epsilon : float or None, optional
        sparsity value for type='log_sum'
    kernel_size : tuple or None, optional
        kernel size for type='group_lasso'
    **kwargs
        keyword arguments for apply_modules method if module_name is not None

    Returns
    -------
    loss : float
        scalar value of sparsity constraint function

    Notes
    -----
    The sparsity constraint function is applied to input (after optionally
    applying modules) and multiplied by the sparsity-cost. This value is then
    used to obtain the gradients with respect to all applicable parameters.

    Functions used for different loss_fn (q is mean of input across batch dim):
    'cross_entropy': sum(-target * log(q) - (1. - target) * log(1. - q))
    'log_sum': sum(log(1. + abs(q) / epsilon))
    'lasso': sum(abs(q))
    'group_lasso': sum(sqrt(prod(kernel_size)) * sqrt(sum_kernel(q**2)))
    Note: for 'cross_entropy', q is also averaged across image dimensions.
    """
    def __init__(self, model, layer_ids, loss_fn='cross_entropy', module_name=None,
                 cost=1., decay=0., **kwargs):
        if type(loss_fn) is str:
            assert hasattr(self, loss_fn)
            loss_fn = getattr(self, loss_fn)
        self.decay = decay
        self.options = functions.pop_attributes(kwargs, ['target','epsilon',
                                                         'kernel_size'],
                                                ignore_keys=True)
        self.q = [None,] * len(layer_ids)
        super(SparseLoss, self).__init__(model, layer_ids, loss_fn, module_name,
                                         cost, **kwargs)

    def cross_entropy(self, q, target):
        q = torch.mean(q.transpose(0,1).flatten(1), -1)
        assert torch.all(torch.gt(q, 0.)), (
            'Type ''cross_entropy'': log(0.) is -inf'
            )
        target = torch.tensor(target, dtype=q.dtype)
        return torch.sub(-target * torch.log(q), (1. - target) * torch.log(1. - q))

    def log_sum(self, q, epsilon):
        epsilon = torch.tensor(epsilon, dtype=q.dtype)
        return torch.log(1. + torch.abs(q) / epsilon)

    def lasso(self, q):
        return torch.abs(q)

    def group_lasso(self, q, kernel_size):
        assert q.ndimension() != 4, (
            'Type ''group_lasso'' requires ndimension == 4'
            )
        p = torch.prod(torch.tensor(kernel_size, dtype=q.dtype))
        g = torch.sqrt(F.lp_pool2d(torch.pow(q, 2.), 1, kernel_size))
        return torch.mul(torch.sqrt(p), g)

    def get_mean_activity(self, activity, decay, i=0):
        # get mean activity
        q = torch.mean(activity, 0, keepdim=True)
        # decay running average
        if self.q[i] is None:
            self.q[i] = q
        else:
            self.q[i] = decay * self.q[i].detach() + (1. - decay) * q
        return self.q[i]

    def forward(self, *args):
        feat = self.get_features(args[0])
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat):
            q = self.get_mean_activity(*feat_i, self.decay, i)
            sparse_cost = self.loss_fn(q, **self.options)
            if sparse_cost.ndimension() > 2:
                sparse_cost = torch.mean(torch.flatten(sparse_cost, 2), -1)
            loss = loss + torch.mul(self.cost[i], torch.sum(sparse_cost))
        return loss / len(self.layer_ids)
