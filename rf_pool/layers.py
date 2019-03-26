import torch
import torch.nn.functional as F
from . import ops
from .utils import lattice

class Layer(torch.nn.Module):
    """
    Base class for receptive field pooling layers
    """
    def __init__(self):
        super(Layer, self).__init__()
        self.mu = None
        self.sigma = None
        self.ratio = None
        self.delta_mu = None
        self.delta_sigma = None
        self.img_shape = None
        self.lattice_fn = None
        self.t = None
        self.rfs = None
        self.mu_mask = None
        self.pool_type = 'max'
        self.kernel_size = 2
        self.input_keys = ['t', 'rfs', 'mu_mask', 'pool_type', 'kernel_size']
        self.inputs = self.get_inputs(self.input_keys)

    def __call__(self, *args):
        return self.forward(*args)

    def apply(self, *args, **kwargs):
        return ops.rf_pool(*args, **kwargs)

    def get_inputs(self, keys):
        inputs = {}
        for key in keys:
            if hasattr(self, key):
                inputs.update({key: getattr(self, key)})
            else:
                inputs.setdefault(key, None)
        return inputs

    def forward(self):
        pass

    def init_rfs(self):
        assert self.lattice_fn is not None
        assert self.img_shape is not None
        self.mu_mask = lattice.mu_mask(self.mu, self.img_shape)
        if self.ratio is not None:
            self.rfs = self.lattice_fn(self.mu, self.sigma,
                                                 self.img_shape, self.ratio)
        else:
            self.rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape)

    def update_rfs(self, delta_mu, delta_sigma):
        assert self.rfs is not None
        assert self.img_shape is not None
        rfs_shape = self.rfs.shape[-2:]
        # update mu if img_shape doesnt match rfs.shape[-2:]
        if rfs_shape != self.img_shape:
            self.mu.add_(torch.sub(torch.as_tensor(self.img_shape, dtype=self.mu.dtype),
                                   torch.as_tensor(rfs_shape, dtype=self.mu.dtype)))
        # update mu and sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma)
        self.mu_mask = lattice.mu_mask(mu, self.img_shape)
        # update rfs
        self.rfs = self.lattice_fn(mu, sigma, self.img_shape)

    def update_mu_sigma(self, delta_mu, delta_sigma):
        img_hw = torch.as_tensor(self.img_shape, dtype=self.mu.dtype)
        mu = self.mu
        sigma = self.sigma
        # add delta_mu to mu
        if delta_mu is not None:
            mu = torch.add(mu, delta_mu)
            mu = torch.min(mu, img_hw)
            mu = torch.max(mu, torch.zeros_like(mu))
            self.delta_mu = delta_mu
        # multiply sigma by delta_sigma
        if delta_sigma is not None:
            sigma = torch.mul(sigma, (1. + delta_sigma))
            sigma = torch.min(sigma, torch.min(img_hw))
            sigma = torch.max(sigma, torch.ones_like(sigma))
            self.delta_sigma = delta_sigma
        return mu, sigma

    def show_lattice(self, x=None, figsize=(5,5), cmap=None):
        assert self.rfs is not None
        if self.lattice_fn is lattice.mask_kernel_lattice:
            mu, sigma = self.update_mu_sigma(self.delta_mu, self.delta_sigma)
            rfs = lattice.exp_kernel_lattice(mu, sigma, self.img_shape)
        else:
            rfs = self.rfs
        rf_lattice = lattice.make_kernel_lattice(rfs)
        lattice.show_kernel_lattice(rf_lattice, x=x, figsize=figsize, cmap=cmap)

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
        function used to update receptive field kernels given delta_mu and delta_sigma
        [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to ops.rf_pool (see ops.rf_pool, other ops pooling functions)

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
        super(RF_Pool, self).__init__()
        self.mu = mu
        self.sigma = sigma
        if 'ratio' in kwargs:
            self.ratio = kwargs.pop('ratio')
        if self.rfs is not None and self.img_shape is None:
            self.img_shape = self.rfs.shape[-2:]
        else:
            self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        self.inputs = self.get_inputs(self.input_keys)
        self.inputs.update(kwargs)
        if mu is not None and sigma is not None:
            self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
        # update inputs with kwargs
        self.inputs.update(kwargs)
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None and delta_sigma is not None:
            self.update_rfs(delta_mu, delta_sigma)
        # for updating mu, sigma directly as part of the graph
        if (self.mu and self.mu.requires_grad) or \
           (self.sigma and self.sigma.requires_grad):
             self.init_rfs()
        # return pooling outputs
        return self.apply(u, **self.inputs)[1]

class RF_Uniform(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, n_kernels, img_shape, spacing, sigma_init=1.,
                lattice_fn=lattice.gaussian_kernel_lattice, **kwargs):
        super(RF_Uniform, self).__init__()
        raise NotImplementedError
        assert np.mod(np.sqrt(n_kernels), 1) == 0, (
            'sqrt of n_kernels must be integer'
        )
        self.n_kernels = n_kernels
        self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        if 'ratio' in kwargs:
            self.ratio = kwargs.pop('ratio')
        self.inputs = self.get_inputs(self.input_keys)
        self.inputs.update(kwargs)
        # set mu, sigma
        centers = torch.as_tensor(self.img_shape)/2.
        n_kernel_side = np.sqrt(n_kernels)
        self.mu, self.sigma = lattice.init_uniform_lattice(centers,
                                                           n_kernel_side,
                                                           spacing,
                                                           sigma_init)
        self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
       # update inputs with kwargs
       self.inputs.update(kwargs)
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       if delta_mu is not None and delta_sigma is not None:
           self.update_rfs(delta_mu, delta_sigma)
       # return pooling outputs
       return self.apply(u, **self.inputs)[1]

class RF_Window(Layer):
    """
    #TODO:WRTIEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Window, self).__init__()
        self.mu = mu
        self.sigma = sigma
        if 'ratio' in kwargs:
            self.ratio = kwargs.pop('ratio')
        if self.rfs is not None and self.img_shape is None:
            self.img_shape = self.rfs.shape[-2:]
        else:
            self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        self.inputs = self.get_inputs(self.input_keys)
        self.inputs.update(kwargs)
        self.init_rfs()

    def forward(self, u, delta_mu=None, **kwargs):
        # update inputs with kwargs
        self.inputs.update(kwargs)
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None:
            self.update_rfs(delta_mu, None)
        # remove mu from inputs
        if 'mu_mask' in self.inputs:
            self.inputs.pop('mu_mask')
        # return pooling outputs
        return self.apply(u, **self.inputs)[1]

class RF_Same(Layer):
    """
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Same, self).__init__()
        self.mu = mu
        self.sigma = sigma
        if 'ratio' in kwargs:
            self.ratio = kwargs.pop('ratio')
        if self.rfs is not None and self.img_shape is None:
            self.img_shape = self.rfs.shape[-2:]
        else:
            self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        self.inputs = self.get_inputs(self.input_keys)
        self.inputs.update(kwargs)
        self.init_rfs()

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
        # update inputs
        self.inputs.update(kwargs)
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None and delta_sigma is not None:
            self.update_rfs(delta_mu, delta_sigma)
        # remove mu from inputs
        if 'mu_mask' in self.inputs:
            self.inputs.pop('mu_mask')
        # return pooling outputs
        h_mean = self.apply(u, **self.inputs)[0]
        if self.kernel_size > 1:
            h_mean = F.max_pool2d_with_indices(h_mean, self.kernel_size)[0]
        return h_mean
