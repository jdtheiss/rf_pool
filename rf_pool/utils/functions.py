import numpy as np
import torch

def repeat(x, repeats):
    """
    Perform numpy-like repeat function

    Parameters
    ----------
    x : torch.Tensor
        tensor to be repeated across dimensions
    repeats : tuple or list
        number of repeats for each dimension

    Returns
    -------
    y : torch.Tensor
        repeated tensor

    Examples
    --------
    >>> x = torch.as_tensor([[1.,2.],[3.,4.]])
    >>> y = repeat(x, (2, 3))
    >>> print(y)
    tensor([[1., 1., 1., 2., 2., 2.],
            [1., 1., 1., 2., 2., 2.],
            [3., 3., 3., 4., 4., 4.],
            [3., 3., 3., 4., 4., 4.]])
    """
    y = x.detach().numpy()
    for i, r in enumerate(repeats):
        y = np.repeat(y, r, i)
    return torch.as_tensor(y)

def getattr_dict(obj, keys):
    out = {}
    for key in keys:
        if hasattr(obj, 'get') and key in obj:
            out.update({key: obj.get(key)})
        elif hasattr(obj, key):
            out.update({key: getattr(obj, key)})
        else:
            out.setdefault(key, None)
    return out

def setattr_dict(obj, **kwargs):
    for key, value in kwargs.items():
        setattr(obj, key, value)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
