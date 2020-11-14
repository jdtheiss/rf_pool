import inspect

import torch
from torch import nn

from rf_pool.utils import functions

#TODO: add samplers
class Op(nn.Module):
    """
    Operation wrapper class
    Wrap an operation as a module to be used in nn.Module or nn.Sequential

    Attributes
    ----------
    fn : function
        operation to be applied to input when Op is called
    **kwargs : keyword arguments used in fn call

    Examples
    --------
    # wrap Bernoulli sampler
    >>> sampler = Op(sample_fn('Bernoulli'))
    >>> samples = sampler(torch.rand(4))
    """
    def __init__(self, fn, **kwargs):
        super(Op, self).__init__()
        self.fn = fn
        self._kwargs = kwargs

    def __repr__(self):
        if self.fn is None:
            return ''
        name = self.fn.__qualname__.replace('<', '').replace('>', '')
        try:
            sig = str(inspect.signature(self.fn))
        except:
            sig = '()'
        return name + sig

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        return self.fn(*args, **_kwargs)

class Sampler(nn.Module):
    """
    Distribution sampler wrapper class
    Wrap a `torch.distributions` class as nn.Module and perform
    `fn(*args, **kwargs).sample()` during `forward` call

    Parameters
    ----------
    distribution : str
        name of distribution within `torch.distributions` to use
    **kwargs : **dict
        keyword arguments passed to `getattr(torch.distributions, distribution)`
        during `forward` call

    Notes
    -----
    This class assumes that the dimension of interest in the input is the
    ``channel'' dimension. For example, for the 'Multinomial' distribution,
    the input is permuted so that the channel dimension is last for sampling.
    Also, for distributions that use `probs` or `logits` arguments, the default
    is to assume that the input to the `forward` call is `probs`, but this can
    be overridden by passing the input as a kwarg (e.g., `logits=input`).
    """
    def __init__(self, distribution, **kwargs):
        super(Sampler, self).__init__()
        self.fn = getattr(torch.distributions, distribution)
        self.arg_names = inspect.getfullargspec(self.fn).args
        self._kwargs = kwargs

    def __repr__(self):
        if self.fn is None:
            return ''
        name = self.fn.__qualname__.replace('<', '').replace('>', '')
        try:
            sig = str(inspect.signature(self.fn))
        except:
            sig = '()'
        return name + sig

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # update kwargs
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        # get ch dim last and flatten potential img dims
        args = list(args)
        for i, arg in enumerate(args):
            shape = arg.shape
            args[i] = arg.transpose(0, 1).flatten(1).t()
        # set probs if in arg_names and logits not set in _kwargs
        if 'probs' in self.arg_names and not any(k in _kwargs for k in ['probs','logits']):
            _kwargs.update({'probs': args.pop(0)})
        elif 'probs' in _kwargs:
            shape = _kwargs['probs'].shape
        elif 'logits' in _kwargs:
            shape = _kwargs['logits'].shape
        # sample
        out = self.fn(*args, **_kwargs).sample()
        # reshape and return
        out = out.view(shape[0], -1, shape[1]).transpose(1, 2)
        return out.contiguous().view(shape)

def reshape_fn(shape):
    out_fn = lambda x: x.reshape(shape)
    out_fn.__qualname__ = 'reshape_fn(%a)' % (shape,)
    return out_fn

def flatten_fn(start_dim=0, end_dim=-1):
    out_fn = lambda x: x.flatten(start_dim, end_dim)
    out_fn.__qualname__ = 'flatten_fn(%a, %a)' % (start_dim, end_dim)
    return out_fn

def sample_fn(distribution, **kwargs):
    if not hasattr(torch.distributions, distribution):
        raise Exception('%a distribution not found.' % distribution)
    fn = getattr(torch.distributions, distribution)
    # get args
    args = inspect.getfullargspec(fn).args
    # return with probs=x or *args
    if 'probs' in args:
        out_fn = lambda x: fn(probs=x, **kwargs).sample()
    else:
        out_fn = lambda *args: fn(*args, **kwargs).sample()
    # set qualname
    if len(kwargs) == 0:
        out_fn.__qualname__ = 'sample_fn(%a)' % (fn.__qualname__)
    else:
        out_fn.__qualname__ = 'sample_fn(%a, **%a)' % (fn.__qualname__, kwargs)
    return out_fn

def multinomial_sample(input):
    x = input.flatten(0,-2)
    x = torch.cat([x, torch.zeros(x.shape[0], 1)], -1)
    y = torch.distributions.Multinomial(1, x).sample()[:,:-1]
    return y.reshape(input.shape)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
