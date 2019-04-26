import numpy as np
import torch
import torch.nn.functional as F

from . import ops
from .utils import lattice, functions

class Layer(torch.nn.Module):
    """
    Base class for receptive field pooling layers
    """
    def __init__(self, mu, sigma, img_shape, lattice_fn, **kwargs):
        super(Layer, self).__init__()
        # input parameters
        self.mu = mu
        self.sigma = sigma
        self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        # extra parameters
        self.ratio = None
        self.delta_mu = None
        self.delta_sigma = None
        # set inputs for rf_pool
        self.rfs = None
        self.pool_type = 'max'
        self.kernel_size = 2
        self.input_keys = ['rfs', 'pool_type', 'kernel_size']
        self.input_keys.extend(kwargs.keys())
        self.input_keys = np.unique(self.input_keys).tolist()
        functions.set_attributes(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def apply(self, *args, **kwargs):
        # get inputs for rf_pool
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        # apply rf_pool
        return ops.rf_pool(*args, **input_kwargs)

    def update_mu_sigma(self, delta_mu, delta_sigma):
        mu = self.mu
        sigma = self.sigma
        # add delta_mu to mu
        if delta_mu is not None:
            mu = torch.add(mu, delta_mu)
            self.delta_mu = delta_mu
        # multiply sigma by delta_sigma
        if delta_sigma is not None:
            sigma = torch.mul(sigma, (1. + delta_sigma) + 1e-6)
            self.delta_sigma = delta_sigma
        return mu, sigma

    def init_rfs(self):
        if self.mu is None and self.sigma is None:
            return None
        assert self.lattice_fn is not None
        assert self.img_shape is not None
        if self.ratio is not None:
            rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape, self.ratio)
        else:
            rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape)
        return rfs

    def update_rfs(self, delta_mu, delta_sigma):
        if self.mu is None and self.sigma is None:
            return None
        assert self.rfs is not None
        assert self.img_shape is not None
        # update mu if img_shape doesnt match rfs.shape[-2:]
        if self.rfs.shape[-2:] != self.img_shape:
            with torch.no_grad():
                self.mu.add_(torch.sub(torch.as_tensor(self.img_shape,
                                                       dtype=self.mu.dtype),
                                       torch.as_tensor(self.rfs.shape[-2:],
                                                       dtype=self.mu.dtype)))
        # update mu and sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma)
        # update rfs
        if self.ratio is not None:
            rfs = self.lattice_fn(mu, sigma, self.img_shape, self.ratio)
        else:
            rfs = self.lattice_fn(mu, sigma, self.img_shape)
        return rfs

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

class RF_Pool(Layer):
    """
    Receptive field pooling layer (see ops.rf_pool for details)

    Parameters
    ----------
    mu

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
        kwargs passed to ops.rf_pool (see ops.rf_pool, other ops pooling
        functions)

    Attributes
    ----------
    #TODO:WRITEME

    Methods
    -------
    apply(u, t=None)
        Apply pooling operation over receptive fields in input, u with optional
        top-down input, t. output is p_mean (see ops.rf_pool)
    init_rfs()
        Initialize receptive fields kernels with given lattice_fn, mu, sigma,
        and img_shape
    update_rfs(delta_mu, delta_sigma,
               lattice_fn=utils.lattice.gaussian_kernel_lattice)
        Update receptive fields kernels by adding delta_mu, delta_sigma and
        creating a new rfs attribute using lattice_fn.
    update_mu_sigma(delta_mu, delta_sigma)
        Update mu and sigma by adding delta_mu, delta_sigma
    show_lattice()
        Display receptive field lattice from rfs kernel
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.gaussian_kernel_lattice, **kwargs):
        super(RF_Pool, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)
        # init receptive fields
        self.rfs = self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        self.rfs = self.update_rfs(delta_mu, delta_sigma)
        # update mu_mask if given
        if hasattr(self, 'mu_mask'):
            self.mu_mask = lattice.mu_mask(self.mu, self.img_shape)
        # return pooling outputs
        return self.apply(u, **kwargs)[1]

class RF_Uniform(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, n_kernels, img_shape, spacing, sigma_init=1.,
                lattice_fn=lattice.gaussian_kernel_lattice, **kwargs):
        assert np.mod(np.sqrt(n_kernels), 1) == 0, (
            'sqrt of n_kernels must be integer'
        )
        # set mu, sigma
        centers = torch.as_tensor(img_shape)/2.
        n_kernel_side = np.sqrt(n_kernels).astype('int')
        mu, sigma = lattice.init_uniform_lattice(centers, n_kernel_side, spacing,
                                                 sigma_init)
        super(RF_Uniform, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)
        # initialize rfs
        self.rfs = self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       self.rfs = self.update_rfs(delta_mu, delta_sigma)
       # update mu_mask if given
       if hasattr(self, 'mu_mask'):
           self.mu_mask = lattice.mu_mask(self.mu, self.img_shape)
       # return pooling outputs
       return self.apply(u, **kwargs)[1]

class RF_Window(Layer):
    """
    #TODO:WRTIEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Window, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)
        # initialize rfs
        self.rfs = self.init_rfs()

    def forward(self, u, delta_mu=None, **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        self.rfs = self.update_rfs(delta_mu, None)
        # update mu_mask if given
        if hasattr(self, 'mu_mask'):
            self.mu_mask = lattice.mu_mask(self.mu, self.img_shape)
        # return pooling outputs
        return self.apply(u, **kwargs)[1]

class RF_Same(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Same, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)
        # initialize rfs
        self.rfs = self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        self.rfs = self.update_rfs(delta_mu, delta_sigma)
        # update mu_mask if given
        if hasattr(self, 'mu_mask'):
            self.mu_mask = lattice.mu_mask(self.mu, self.img_shape)
        # return h_mean output and
        h_mean = self.apply(u, **kwargs)[0]
        # pool if kernel_size > 1
        if type(self.kernel_size) is int and self.kernel_size > 1:
            pool_kwargs = functions.get_attributes(self, ['return_indices'])
            h_mean = F.max_pool2d(h_mean, self.kernel_size, **pool_kwargs)
        return h_mean
