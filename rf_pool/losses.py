from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn as nn

from .utils import functions

def _count_items(d):
    if not isinstance(d, (dict, OrderedDict)):
        return 1
    cnt = 0
    for v in d.values():
        cnt += _count_items(v)
    return cnt

def _get_items(d):
    if not isinstance(d, (dict, OrderedDict)):
        return [d]
    items = []
    for v in d.values():
        items.extend(_get_items(v))
    return items

class Loss(nn.Module):
    """
    Base Class for losses
    """
    def __init__(self):
        super(Loss, self).__init__()

    def L2(self, x):
        return torch.sqrt(torch.sum(torch.pow(x, 2)))

    def set_params(self, on_parameters=None, set='on'):
        if self.parameters is None:
            return None
        assert set in ['on','off']
        # set requires grad True for self.parameters
        if set == 'on':
            on_parameters = self.model.get_trainable_params()
            self.model.set_requires_grad(pattern='', requires_grad=False)
            self.model.set_requires_grad(self.parameters, requires_grad=True)
            return on_parameters
        # set requires_grad True for on_parameters
        elif set == 'off':
            self.model.set_requires_grad(self.parameters, requires_grad=False)
            self.model.set_requires_grad(on_parameters, requires_grad=True)
        return None

class MultiLoss(Loss):
    """
    Combine multiple losses as weighted sum

    Parameters
    ----------
    losses : list
        list of loss functions to be called
    weights : list
        list of weights (per loss function in `losses`)
        [default: [1.] * len(losses)]

    Returns
    -------
    None

    Methods
    -------
    add_loss(loss, weight=1.) : add another loss to `losses` attribute
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
    Simple loss function applied with optional, constant input/target parameters

    Parameters
    ----------
    loss_fn : torch.nn.modules.loss or function
        loss function to use
    input : torch.Tensor or None
        input passed to `loss_fn` as first argument if not None, else replaced
        by first argument passed during `forward` call [default: None]
    target : torch.Tensor or None
        target passed to `loss_fn` as second argument if not None, else replaced
        by arguments passed during `forward` call [default: None] (see Notes)

    Returns
    -------
    None

    See Also
    --------
    KwargsLoss

    Notes
    -----
    If `input is not None and target is None`, the second argument passed to
    `loss_fn` is either the first argument passed to `forward` if only one
    argument is given or the second argument if more than one argument is given.
    """
    def __init__(self, loss_fn, input=None, target=None):
        super(ArgLoss, self).__init__()
        self.loss_fn = loss_fn
        self.input = input
        self.target = target

    def forward(self, *args):
        args *= 2
        inputs = [a if v is None else v
                  for a, v in zip(args[:2], [self.input,self.target])]
        return self.loss_fn(*inputs)

class KwargsLoss(Loss):
    """
    Loss function using keyword arguments

    Parameters
    ----------
    loss_fn : torch.nn.modules.loss or function
        loss function to use
    n_args : int
        number of arguments to be passed to `loss_fn` during `forward` call
        [default: 1]
    **kwargs : **dict
        keyword arguments passed to `loss_fn` during `forward` call

    Returns
    -------
    None

    See Also
    --------
    ArgLoss
    """
    def __init__(self, loss_fn, n_args=1, **kwargs):
        super(KwargsLoss, self).__init__()
        self.loss_fn = loss_fn
        self.n_args = n_args
        self.kwargs = kwargs

    def forward(self, *args):
        return self.loss_fn(*args[:self.n_args], **self.kwargs)

class KernelLoss(Loss):
    """
    Convolve input with given weight

    Parameters
    ----------
    weight : torch.Tensor
        weight to convolve with input during `forward` call
    reduce : str
        name of reduction function to use (i.e., function from `torch` or `Loss`)
        [default: 'mean']
    **kwargs : **dict
        keyword arguments passed to `torch.conv2d` during `forward` call

    Returns
    -------
    None

    See Also
    --------
    SpatialFreqLoss
    KernelVarLoss
    """
    def __init__(self, weight, reduce='mean', **kwargs):
        super(KernelLoss, self).__init__()
        self.weight = weight
        self.kwargs = kwargs
        if reduce is None:
            self.reduce_fn = lambda x: x
        elif hasattr(self, reduce):
            self.reduce_fn = getattr(self, reduce)
        elif hasattr(torch, reduce):
            self.reduce_fn = getattr(torch, reduce)
        else:
            raise Exception('Unknown reduce type')

    def forward(self, *args):
        output = torch.conv2d(args[0], self.weight, **self.kwargs)
        return self.reduce_fn(output)

class SpatialFreqLoss(KernelLoss):
    """
    Reduce spatial frequecies in image space with given gabors

    Parameters
    ----------
    n_gabors : int
        number of gabors to use
    theta : list
        list of gabor orientations with `len(theta) == n_gabors`
    sigma : list
        list of sigma sizes with `len(sigma) == n_gabors`
    wavelength : list
        list of spatial frequecie wavelengths with `len(wavelength) == n_gabors`
    filter_shape : list
        list of filter shapes with `len(filter_shape) == n_gabors`
    gamma : list or float
        gamma value(s) passed to `functions.gabor_filter` [default: 0.3]
    psi : list or float
        psi value(s) passed to `functions.gabor_filter` [default: 0.]
    reduce : str
        name of reduction function to use (i.e., function from `torch` or `Loss`)
        [default: 'mean']
    **kwargs : **dict
        keyword arguments passed to `torch.conv2d` during `forward` call

    Returns
    -------
    None

    See Also
    --------
    rf_pool.utils.functions.gabor_filter
    """
    def __init__(self, n_gabors, theta, sigma, wavelength, filter_shape,
                 gamma=0.3, psi=0., reduce='mean', **kwargs):
        # get gabors
        list_args = functions.parse_list_args(n_gabors, theta, sigma, wavelength,
                                              filter_shape, gamma, psi)[0]
        gabors = [functions.gabor_filter(t, s, w, f_s, g, p)
                  for (t, s, w, f_s, g, p) in list_args]
        weight = torch.stack(gabors).unsqueeze(1)
        super(SpatialFreqLoss, self).__init__(weight, reduce, **kwargs)

    def forward(self, *args):
        if args[0].shape[1] > self.weight.shape[1]:
            self.weight = functions.repeat(self.weight, [1, args[0].shape[1]])
        output = torch.conv2d(args[0], self.weight, **self.kwargs)
        return self.reduce_fn(output)

class KernelVarLoss(Loss):
    """
    Reduce variance across kernels in image

    Parameters
    ----------
    kernel_size : int or tuple
        size of kernel to reduce variance within [default: 2]
    stride : int or tuple
        stride of convolution [default: 1]
    reduce : str
        name of reduction function to use (i.e., function from `torch` or `Loss`)
        [default: 'mean']
    **kwargs : **dict
        keyword arguments passed to `torch.nn.functional.avg_pool2d` or
        `torch.nn.functional.lp_pool2d` (see Notes)

    Returns
    -------
    None

    Notes
    -----
    Variance is computed across kernels by first subtracting the mean computed as
    `mean = torch.nn.functional.avg_pool2d(input, kernel_size, stride, **kwargs)`
    followed by computing the sum of squares across each kernel with
    `torch.nn.functional.lp_pool2d(input-mean, 2, kernel_size, stride, **kwargs)`.
    Finally, the result is reduced via the function defined by `reduce`.

    See Also
    --------
    torch.nn.functional.avg_pool2d
    torch.nn.lp_pool2d
    """
    def __init__(self, kernel_size=2, stride=1, reduce='mean', **kwargs):
        super(KernelVarLoss, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.kwargs = kwargs
        self.loss_fn = self.kernel_var_loss
        if reduce is None:
            self.reduce_fn = lambda x: x
        elif hasattr(self, reduce):
            self.reduce_fn = getattr(self, reduce)
        elif hasattr(torch, reduce):
            self.reduce_fn = getattr(torch, reduce)
        else:
            raise Exception('Unknown reduce type')

    def kernel_var_loss(self, x):
        m = torch.nn.functional.avg_pool2d(x, self.kernel_size,
                                           stride=self.stride,
                                           **self.kwargs)
        m = torch.nn.functional.interpolate(m, size=x.shape[-2:])
        d = torch.sub(x, m)
        output = torch.nn.functional.lp_pool2d(d, 2, self.kernel_size,
                                               stride=self.stride,
                                               **self.kwargs)
        return self.reduce_fn(output)

    def forward(self, *args):
        return self.loss_fn(*args)

class LayerLoss(Loss):
    """
    Layer-based loss

    Parameters
    ----------
    model : rf_pool.models
        model used to obtain layer outputs
    layer_modules : dict
        dictionary containing {layer_id: []} for obtaining layer-specific
        outputs on which to compute the loss (see model.apply). use
        {layer_id: {module_name: []}} to set specific module within layer.
    loss_fn : torch.nn.modules.loss or function
        loss function to compute over features chosen from layers/modules. if
        `type(lost_fn) is list`, should be list of loss functions per
        layer/module pair.
    cost : float or list, optional
        cost per layer/module pair or float value applied to all losses
        [default: 1.]
    parameters : torch.nn.parameter.Parameter
        parameters to use in update from loss
        [default: None, uses parameters as they are]
    input_target : torch.Tensor
        image used to obtain features to match with loss function
        (assumed to stay the same)
        [default: None]
    target : torch.Tensor
        features to match with loss function [default: None]
    **kwargs : **dict
        keyword arguments passed to model.apply function

    Returns
    -------
    None
    """
    def __init__(self, model, layer_modules, loss_fn, cost=1.,
                 parameters=None, input_target=None, target=None, **kwargs):
        super(LayerLoss, self).__init__()
        self.model = model
        self.layer_modules = layer_modules
        self.n_features = _count_items(self.layer_modules)
        self.loss_fn = loss_fn
        self.cost = functions.parse_list_args(self.n_features, cost)[0]
        self.cost = [c[0] for c in self.cost]
        self.parameters = parameters
        self.input_target = input_target
        self.target = target
        self.kwargs = kwargs
        # set input_target to features if given
        if self.input_target is not None:
            self.input_target = self.get_features(self.input_target)

    def get_features(self, *args):
        args = list(args)
        # turn on parameters
        on_parameters = self.set_params(set='on')
        # get features
        feat = copy.deepcopy(self.layer_modules)
        [self.model.apply(arg, output=feat, **self.kwargs) for arg in args]
        # turn off parameters
        self.set_params(on_parameters, set='off')
        return _get_items(feat)

    def forward(self, *args):
        # get features
        if len(args) > 1 and args[0].ndimension() != args[1].ndimension():
            feat = self.get_features(args[0])
        elif self.target is None and self.input_target is None:
            feat = self.get_features(*args)
        else:
            feat = self.get_features(args[0])
        # get losses
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat):
            if isinstance(self.loss_fn, list):
                loss_fn_i = self.loss_fn[i]
            else:
                loss_fn_i = self.loss_fn
            if self.input_target is not None:
                loss_i = loss_fn_i(*feat_i, *self.input_target[i])
            elif self.target is not None:
                loss_i = loss_fn_i(*feat_i, self.target)
            elif len(args) > 1 and args[0].ndimension() != args[1].ndimension():
                loss_i = loss_fn_i(*feat_i, *args[1:])
            else:
                loss_i = loss_fn_i(*feat_i)
            loss = loss + loss_i * self.cost[i]
        return loss / self.n_features

class FeatureLoss(LayerLoss):
    """
    Feature-based loss

    Parameters
    ----------
    model : rf_pool.models
        model used to obtain layer outputs
    layer_modules : dict
        dictionary containing {layer_id: []} for obtaining layer-specific
        outputs on which to compute the loss (see model.apply). use
        {layer_id: {module_name: []}} to set specific module within layer.
    feature_index : int or list
        index of feature to compute loss
    loss_fn : torch.nn.modules.loss or function or list
        loss function to compute over features chosen from layers/modules.
        input is index features with shape (batch_size, len(feature_index) h, w)
        or (batch_size, len(feature_index)). if `type(lost_fn) is list`, should
        be list of loss functions per layer/module pair.
    cost : float or list, optional
        cost per layer/module pair or float value applied to all losses
        [default: 1.]
    parameters : torch.nn.parameter.Parameter
        parameters to use in update from loss
        [default: None, uses parameters as they are]
    input_target : torch.Tensor
        image used to obtain features to match with loss function
        (assumed to stay the same)
        [default: None]
    target : torch.Tensor
        features to match with loss function [default: None]
    **kwargs : **dict
        keyword arguments passed to model.apply function

    Returns
    -------
    None
    """
    def __init__(self, model, layer_modules, feature_index, loss_fn, cost=1.,
                 parameters=None, input_target=None, target=None, **kwargs):
        super(FeatureLoss, self).__init__(model, layer_modules, loss_fn, cost,
                                          parameters, input_target, target,
                                          **kwargs)
        # get feature_index as list
        assert isinstance(feature_index, (int, list))
        if not isinstance(feature_index, list):
            feature_index = [feature_index]
        self.feature_index = feature_index
        # set input_target to indexed features if given
        if self.input_target is not None:
            self.input_target = [[f[:,self.feature_index] for f in feat_i]
                                 for feat_i in self.input_target]

    def forward(self, *args):
        if self.target is None and self.input_target is None:
            feat = self.get_features(*args)
        else:
            feat = self.get_features(args[0])
        # get losses
        loss = torch.zeros(1, requires_grad=True)
        for i, feat_i in enumerate(feat):
            # index features
            feat_i = [f[:,self.feature_index] for f in feat_i]
            if isinstance(self.loss_fn, list):
                loss_fn_i = self.loss_fn[i]
            else:
                loss_fn_i = self.loss_fn
            if self.input_target is not None:
                loss_i = loss_fn_i(*feat_i, *self.input_target[i])
            elif self.target is not None:
                loss_i = loss_fn_i(*feat_i, self.target)
            elif len(args) > 1 and args[0].ndimension() != args[1].ndimension():
                loss_i = loss_fn_i(*feat_i, *args[1:])
            else:
                loss_i = loss_fn_i(*feat_i)
            loss = loss + loss_i * self.cost[i]
        return loss / self.n_features

class SparseLoss(LayerLoss):
    """
    Encourage sparsity by taking gradient of constraint function

    Parameters
    ----------
    model : rf_pool.models.Model
        model used to compute features at layer_ids
    layer_modules : dict
        dictionary containing {layer_id: []} for obtaining layer-specific
        outputs on which to compute the loss (see model.apply). use
        {layer_id: {module_name: []}} to set specific module within layer.
    loss_fn : string or function
        type of sparsity constraint: 'cross_entropy' (Lee et al., 2008),
        'log_sum' (Ji et al., 2014), 'lasso', 'group_lasso' (Yuan & Lin, 2006)
        [default: 'cross_entropy']
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
        keyword arguments for apply method if module_name is not None

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
    def __init__(self, model, layer_modules, loss_fn='cross_entropy',
                 cost=1., decay=0., **kwargs):
        if type(loss_fn) is str:
            assert hasattr(self, loss_fn)
            loss_fn = getattr(self, loss_fn)
        self.decay = decay
        self.options = functions.pop_attributes(kwargs, ['target','epsilon',
                                                         'kernel_size'],
                                                ignore_keys=True)
        super(SparseLoss, self).__init__(model, layer_modules, loss_fn, cost=cost,
                                         **kwargs)
        self.q = [None,] * self.n_features

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
        return loss / self.n_features
