import numpy as np
import torch
from torch.autograd import Function

import pool
from .utils import functions, lattice

class Pool(torch.nn.Module):
    """
    Base class for receptive field pooling layers
    """
    def __init__(self, mu, sigma, img_shape, lattice_fn, **kwargs):
        super(Pool, self).__init__()
        # input parameters
        self.mu = mu
        self.sigma = sigma
        self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        # check for optional kwargs
        options = functions.pop_attributes(kwargs, ['delta_mu', 'delta_sigma',
                                                    'update_img_shape'])
        functions.set_attributes(self, **options)
        # set inputs for rf_pool
        self.rfs = None
        self.rf_indices = None
        self._update_rfs()
        self.pool_fn = None
        self.kernel_size = None
        self.input_keys = ['rf_indices', 'pool_fn', 'kernel_size']
        self.input_keys.extend(kwargs.keys())
        self.input_keys = np.unique(self.input_keys).tolist()
        functions.set_attributes(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def apply(self, input, **kwargs):
        # get inputs for rf_pool
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        # apply rf_pool
        return apply(input, **input_kwargs)

    def set(self, **kwargs):
        functions.set_attributes(self, **kwargs)
        if 'mu' in kwargs or 'sigma' in kwargs or 'lattice_fn' in kwargs:
            self._update_rfs()
        elif 'img_shape' in kwargs:
            self.mu, self.sigma = self.update_mu_sigma()
            self._update_rfs()

    def get(self, keys):
        output = []
        for key in keys:
            if hasattr(self, key):
                output.append(getattr(self, key))
            else:
                output.append(None)
        return output

    def update_mu_sigma(self, delta_mu=None, delta_sigma=None, priority_map=None):
        if self.mu is None and self.sigma is None:
            return (None, None)
        else:
            mu = self.mu
            sigma = self.sigma
        # add delta_mu to mu
        if delta_mu is not None:
            mu = torch.add(self.mu, delta_mu)
            self.delta_mu = delta_mu
        # add delta_sigma to log(sigma**2) then exponentiate and sqrt
        if delta_sigma is not None:
            log_sigma = torch.add(torch.log(torch.pow(self.sigma, 2)),
                                  delta_sigma)
            sigma = torch.sqrt(torch.exp(log_sigma))
            self.delta_sigma = delta_sigma
        # update mu if img_shape doesnt match rfs.shape[-2:]
        if self.update_img_shape and \
           self.rfs.shape[-2:] != torch.Size(self.img_shape):
            with torch.no_grad():
                img_diff = torch.sub(torch.tensor(self.img_shape, dtype=mu.dtype),
                                     torch.tensor(self.rfs.shape[-2:], dtype=mu.dtype))
                mu = torch.add(mu, img_diff / 2.)
        elif self.rfs.shape[-2:] != torch.Size(self.img_shape):
            raise Exception('rfs.shape[-2:] != self.img_shape')
        # update mu, sigma with priority map
        if priority_map is not None:
            mu, sigma = lattice.update_mu_sigma(mu, sigma, priority_map)
        return mu, sigma

    def _update_rfs(self, mu=None, sigma=None, lattice_fn=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if lattice_fn is None:
            lattice_fn = self.lattice_fn
        if (mu is None and sigma is None):
            return
        elif (torch.allclose(mu, self.mu) and torch.allclose(sigma, self.sigma) \
              and lattice_fn == self.lattice_fn and self.rfs is not None):
            return
        assert self.lattice_fn is not None
        assert self.img_shape is not None
        # get rfs using lattice_fn
        args = [mu, sigma, self.img_shape]
        self.rfs = lattice_fn(*args)
        self.rf_indices = rf_to_indices(self.rfs)

    def update_rfs(self, mu=None, sigma=None, lattice_fn=None):
        self._update_rfs(mu, sigma, lattice_fn)
        return self.rfs

    def get_squeezed_coords(self, mu, sigma):
        # find min, max mu
        min_mu, min_idx = torch.min(mu, dim=0)
        max_mu, max_idx = torch.max(mu, dim=0)
        min_sigma = sigma[min_idx].t()
        max_sigma = sigma[max_idx].t()
        return torch.cat([min_mu - min_sigma, max_mu + max_sigma]).int()

    def crop_img(self, input, coords):
        output = torch.flatten(input, 0, -3)
        output = output[:, coords[0,0]:coords[1,0], coords[0,1]:coords[1,1]]
        return torch.reshape(output, input.shape[:-2] + output.shape[-2:])

    def show_lattice(self, x=None, figsize=(5,5), cmap=None):
        assert self.rfs is not None
        if self.lattice_fn is lattice.mask_kernel_lattice:
            mu, sigma = self.update_mu_sigma(self.delta_mu, self.delta_sigma)
            rfs = lattice.exp_kernel_lattice(mu, sigma, self.img_shape)
        else:
            rfs = self.rfs
        rf_lattice = lattice.make_kernel_lattice(rfs)
        lattice.show_kernel_lattice(rf_lattice, x=x, figsize=figsize, cmap=cmap)

    def forward(self):
        raise NotImplementedError

class RF_Pool(Pool):
    """
    Receptive field pooling layer

    Parameters
    ----------
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Pool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                      **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # return pooling outputs
        return self.apply(u, **kwargs)

class RF_Uniform(Pool):
    """
    Receptive field pooling layer with RFs initated using
    lattice.init_uniform_lattice

    Parameters
    ----------
    n_kernels : tuple or int
        number of kernels along height/width of the lattice. if type(n_kernels)
        is int, n_kernels will be set to (n_kernels,)*2
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    spacing : float
        spacing between receptive field centers
    sigma_init : float
        standard deviation initialization [default: 1.]
    rotate : float
        rotation (in radians, counterclockwise) to apply to the entire array
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply
    lattice.init_uniform_lattice
    """
    def __init__(self, n_kernels, img_shape, spacing, sigma_init=1., rotate=0.,
                lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set mu, sigma
        centers = torch.as_tensor(img_shape)/2.
        mu, sigma = lattice.init_uniform_lattice(centers, n_kernels, spacing,
                                                 sigma_init, rotate)
        super(RF_Uniform, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
       self._update_rfs(mu, sigma)
       # return pooling outputs
       return self.apply(u, **kwargs)

class RF_Random(Pool):
    """
    Receptive field pooling layer with RF mu and sigmas initated to random

    Parameters
    ----------
    n_kernels : int
        number of kernels in lattice
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    Notes
    -----
    If 'mu' or 'sigma' are not provided, they are set to random locations within
    the image shape and random sizes less than half the image shape, respectively.

    See Also
    --------
    pool.apply
    """
    def __init__(self, n_kernels, img_shape,
                lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set mu, sigma
        if 'mu' not in kwargs:
            mu = torch.rand(n_kernels, 2) * torch.tensor(img_shape).float()
        else:
            mu = kwargs.pop('mu')
        if 'sigma' not in kwargs:
            sigma = torch.rand(n_kernels, 1) * np.minimum(*img_shape) / 2.
        else:
            sigma = kwargs.pop('sigma')
        super(RF_Random, self).__init__(mu, sigma, img_shape, lattice_fn,
                                        **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
       self._update_rfs(mu, sigma)
       # return pooling outputs
       return self.apply(u, **kwargs)

class RF_Squeeze(Pool):
    """
    Receptive field pooling layer with output image size squeezed to extent of
    receptive field lattice

    Parameters
    ----------
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Squeeze, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # apply pooling function
        output = self.apply(u, **kwargs)
        # get squeezed coordinates
        coords = self.get_squeezed_coords(mu, sigma)
        # crop
        return self.crop_img(output, coords)

class RF_CenterCrop(Pool):
    """
    Receptive field pooling layer with center cropped output image shape

    Parameters
    ----------
    crop_size : tuple
        output image crop size
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None]
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply
    """
    def __init__(self, crop_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        self.crop_size = torch.tensor(crop_size)
        super(RF_CenterCrop, self).__init__(mu, sigma, img_shape, lattice_fn,
                                            **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # apply pooling function
        output = self.apply(u, **kwargs)
        # get coordinates of center size
        center = torch.max(torch.tensor(self.img_shape) // 2 - 1,
                           torch.tensor([0,0]))
        half_crop = self.crop_size // 2
        coords = torch.stack([center - half_crop,
                              center + half_crop + np.mod(self.crop_size, 2)])
        # return center crop
        return self.crop_img(output, coords)

class MaxPool(Pool):
    """
    Receptive field pooling layer with max pooling

    Parameters
    ----------
    kernel_size : tuple or int
        size of kernel for kernel-based pooling
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None]
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply
    """
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(MaxPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                      kernel_size=kernel_size,
                                      pool_fn='max_pool', **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # apply pooling function
        return self.apply(u, **kwargs)

class ProbmaxPool(Pool):
    """
    Receptive field pooling layer with probabilistic max pooling

    Parameters
    ----------
    kernel_size : tuple or int
        size of kernel for kernel-based pooling
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None]
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply

    References
    ----------
    Lee, H., Grosse, R., Ranganath, R., & Ng, A. Y. (2009, June). Convolutional
    deep belief networks for scalable unsupervised learning of hierarchical
    representations. In Proceedings of the 26th annual international conference
    on machine learning (pp. 609-616). ACM.
    """
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(ProbmaxPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                          kernel_size=kernel_size,
                                          pool_fn='probmax_pool', **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # apply pooling function
        return self.apply(u, **kwargs)

class StochasticPool(Pool):
    """
    Receptive field pooling layer with stochastic max pooling

    Parameters
    ----------
    kernel_size : tuple or int
        size of kernel for kernel-based pooling
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2) [default: None]
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1) [default: None]
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to pool.apply (see pool.apply, other ops pooling
        functions)

    See Also
    --------
    pool.apply

    References
    ----------
    Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization
    of deep convolutional neural networks. arXiv preprint arXiv:1301.3557.
    """
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(StochasticPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                             kernel_size=kernel_size,
                                             pool_fn='stochastic_pool', **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self._update_rfs(mu, sigma)
        # apply pooling function
        return self.apply(u, **kwargs)

def rf_to_indices(rfs):
    """
    Convert rf kernels into rf_indices

    Parameters
    ----------
    rfs : torch.Tensor
        receptive field mask with shape (n_kernels, h, w)

    Returns
    -------
    rf_indices : torch.Tensor
        receptive field indices with shape (n_kernels, h * w)
    """
    if rfs is None:
        return None
    rf_indices = torch.zeros_like(rfs.flatten(1))
    for i, rf in enumerate(rfs):
        idx = torch.nonzero(rf.flatten())
        rf_indices[i,:idx.numel()] = idx.flatten()
    return rf_indices

# cpp pooling functions
def max_pool(input, **kwargs):
    if 'rfs' in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.pop('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='max_pool', **kwargs)

def probmax(input, **kwargs):
    if 'rfs' in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.pop('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='probmax', **kwargs)

def probmax_pool(input, **kwargs):
    if 'rfs' in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.pop('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='probmax_pool', **kwargs)

def stochastic_pool(input, **kwargs):
    if 'rfs' in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.pop('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='stochastic_pool', **kwargs)

def unpool(input, index_mask, **kwargs):
    return apply(input, pool_fn='unpool', mask=index_mask, **kwargs)

def apply(u, pool_fn=None, rf_indices=None, kernel_size=None,
          stride=None, retain_shape=False, return_indices=False, **kwargs):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        input to pooling layer with shape (batch_size, ch, h, w)
    pool_fn : str
        pooling function (e.g., 'max_pool', 'probmax_pool', 'stochastic_pool').
        [default: None]
    rf_indices : torch.Tensor or None
        tensor containing receptive field indices to apply pooling over with
        shape (n_kernels, h * w)
        [default: None]
    kernel_size : int or tuple, optional
        size of subsampling blocks in hidden layer connected to pooling units
        [default: None]
    stride : int or tuple, optional
        stride of subsampling operation
        [default: None]
    retain_shape : bool, optional
        boolean whether to retain the shape of output after multiplying u with
        rfs (i.e. shape=(batch_size, ch, n_rfs, u_h, u_w))
        [default: False]
    return_indices : bool, optional
        boolean whether to return indices of u contributing to the pooled output
    **kwargs : dict
        keyword arguments passed to pool_fn

    Returns
    -------
    output : torch.Tensor
        pool output with shape (batch_size, ch, h//kernel_size, w//kernel_size)

    Examples
    --------
    # Performs max-pooling across 16 receptive fields tiling image
    >>> from utils import lattice
    >>> u = torch.rand(1,10,8,8)
    >>> mu, sigma = lattice.init_uniform_lattice((4,4), 2, 3, 2.)
    >>> rfs = lattice.mask_kernel_lattice(mu, sigma, (8,8))
    >>> rf_indices = rf_to_indices(rfs)
    >>> pool_output = apply(u, 'sum_pool', rf_indices=rf_indices, kernel_size=(2,2))
    """

    # get input shape
    batch_size, ch = u.shape[:2]

    # if no pool_fn, return u
    if pool_fn is None:
        return u

    # assert pool_fn in pool and get pool_grad
    assert hasattr(pool, pool_fn)
    if 'grad_fn' in kwargs:
        grad_fn = kwargs.pop('grad_fn')
    elif hasattr(pool, pool_fn + '_grad'):
        grad_fn = pool_fn + '_grad'
    else:
        grad_fn = None
    pool_fn = getattr(pool, pool_fn)

    # set kwargs
    kwargs.setdefault('mask_indices', rf_indices)
    kwargs.setdefault('kernel_size', kernel_size)
    kwargs.setdefault('stride', stride)
    kwargs.setdefault('retain_shape', retain_shape)

    # apply cpp pooling function
    input = u.data.flatten(0,1)
    outputs = list(pool_fn(input, **kwargs))
    for i, output in enumerate(outputs):
        if output is not None:
            output_shape = (batch_size, ch,) + output.shape[1:]
            output = output.reshape(output_shape)
            outputs[i] = torch.as_tensor(output)

    # return without grad if less than 3 outputs
    if len(outputs) < 3:
        return outputs[0]

    # set grad args
    grad_args = [u, grad_fn, *outputs, kernel_size, stride]

    # return with indices
    if return_indices:
        return _PoolGrad_with_indices.apply(*grad_args)

    return _PoolGrad.apply(*grad_args)

def _apply_grad(grad_output, grad_input, grad_fn=None, index_mask=None,
                index_kernel=None, kernel_size=None, stride=None):
    # if no grad_fn, use indices
    if grad_fn is None:
        if index_kernel is not None:
            grad_input.flatten(2).scatter_(2, index_kernel.flatten(2),
                                           grad_output.flatten(2));
        else:
            grad_input = grad_output
        if index_mask is not None:
            index_mask = torch.where(index_mask == 0)
            grad_input[index_mask] = 0.
        return grad_input

    # assert pool_fn in pool and get pool_grad
    assert hasattr(pool, grad_fn)
    grad_fn = getattr(pool, grad_fn)

    # set input kwargs
    keys = ['mask_indices', 'kernel_size', 'stride']
    args = [indices, kernel_size, stride]
    kwargs = dict([(k,d) for (k, d) in zip(keys, args)])

    # apply grad function
    input = grad_output.data.flatten(0,1)
    grad_input = grad_fn(input, **kwargs)
    grad_input = torch.as_tensor(grad_input, dtype=grad_output.dtype)
    output_shape = grad_output.shape[:2] + grad_input.shape[2:]
    grad_input = grad_input.reshape(output_shape)

    return grad_input

class _PoolGrad(Function):
    @staticmethod
    def forward(ctx, input, grad_fn, output, *args):
        ctx.grad_fn = grad_fn
        ctx.grad_args = args
        ctx.save_for_backward(input, args[0])
        output.requires_grad_(input.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, index_mask = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input = _apply_grad(grad_output, grad_input, ctx.grad_fn,
                                 *ctx.grad_args)
        return grad_input, None, None, None, None, None, None

class _PoolGrad_with_indices(Function):
    @staticmethod
    def forward(ctx, input, grad_fn, output, *args):
        ctx.grad_fn = grad_fn
        ctx.grad_args = args
        ctx.save_for_backward(input)
        output.requires_grad_(input.requires_grad)
        if args[0] is None:
            return output, args[1]
        elif args[1] is None:
            return output, args[0]
        return output, args[0], args[1]

    @staticmethod
    def backward(ctx, grad_output, index_mask, index_kernel):
        input = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_input = _apply_grad(grad_output, grad_input, ctx.grad_fn, *args,
                                 *ctx.grad_args)
        return grad_input, None, None, None, None, None, None

if __name__ == '__main__':
    import doctest
    doctest.testmod()
