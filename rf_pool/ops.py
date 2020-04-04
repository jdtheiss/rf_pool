import inspect

import torch

from .utils import functions

class Op(torch.nn.Module):
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
        self.input_keys = kwargs.keys()
        functions.set_attributes(self, **kwargs)

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
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        return self.fn(*args, **input_kwargs)

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
