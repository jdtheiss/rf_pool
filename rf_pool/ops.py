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
    >>> sampler = Op(sample_fn, distr=torch.distributions.Bernoulli)
    >>> samples = sampler(torch.rand(4))
    """
    def __init__(self, fn, **kwargs):
        super(Op, self).__init__()
        self.fn = fn
        self.input_keys = kwargs.keys()
        functions.set_attributes(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        return self.fn(*args, **input_kwargs)

def sample_fn(input, distr, **kwargs):
    return distr(input, **kwargs).sample()

def reshape_fn(input, shape):
    return input.reshape(shape)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
