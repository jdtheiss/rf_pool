import torch
import ops
from utils import lattice

class Layer(torch.nn.Module):
    """
    Base class for receptive field pooling layers
    """
    def __init__(self):
        super(Layer, self).__init__()
        self.mu = None
        self.sigma = None
        self.img_shape = None
        self.lattice_fn = None
        self.updates = None
        self.inputs = {'t': None, 'rfs': None, 'mu': self.mu,
                       'pool_type': 'max', 'block_size': 2}

    def __call__(self, *args):
        return self.forward(*args)

    def apply(self, *args, **kwargs):
        return ops.rf_pool(*args, **kwargs)

    def forward(self):
        pass

    def init_rfs(self):
        assert self.lattice_fn is not None
        assert self.mu.shape[0] == self.sigma.shape[0]
        assert self.img_shape is not None
        self.inputs['mu'] = self.mu
        self.inputs['rfs'] = self.lattice_fn(self.mu, self.sigma, self.img_shape)

    def update_rfs(self, delta_mu, delta_sigma):
        assert self.inputs['rfs'] is not None
        assert self.img_shape is not None
        rfs_shape = self.inputs['rfs'].shape[-2:]
        # update mu if img_shape doesnt match rfs.shape[-2:]
        if rfs_shape != self.img_shape:
            self.mu.add_(torch.sub(torch.as_tensor(self.img_shape, dtype=self.mu.dtype),
                                   torch.as_tensor(rfs_shape, dtype=self.mu.dtype)))
        # update mu and sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, self.updates)
        self.inputs['mu'] = mu
        # update rfs
        self.inputs['rfs'] = self.lattice_fn(mu, sigma, self.img_shape)

    def update_mu_sigma(self, delta_mu, delta_sigma, updates=False):
        img_hw = torch.as_tensor(self.img_shape, dtype=delta_mu.dtype)
        mu = self.mu
        sigma = self.sigma
        # add delta_mu to mu
        if delta_mu is not None:
            mu = torch.add(mu, delta_mu)
            mu = torch.min(mu, img_hw)
            mu = torch.max(mu, torch.zeros_like(mu))
        # multiply sigma by delta_sigma
        if delta_sigma is not None:
            sigma = torch.mul(sigma, (1. + delta_sigma))
            sigma = torch.min(sigma, torch.min(img_hw))
            sigma = torch.max(sigma, torch.ones_like(sigma))
        return mu, sigma

    def show_rfs(self, figsize=(5,5), cmap=None):
        assert self.inputs['rfs'] is not None
        rfs = lattice.make_kernel_lattice(self.inputs['rfs'])
        lattice.show_kernel_lattice(rfs, figsize=figsize, cmap=cmap)

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
    updates : bool
        update mu and sigma on each call to update_rfs (True) or keep mu and
        sigma static (False [default])
    lattice_fn : utils.lattice function
        function used to update receptive field kernels given delta_mu and delta_sigma
        [default: lattice.gaussian_kernel_lattice]
    **kwargs : dict
        kwargs passed to ops.rf_pool (see ops.rf_pool, other ops pooling functions)

    Attributes
    ----------
    inputs : dict
        inputs passed to ops.rf_pool

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
    show_rfs()
        Display receptive field lattice from rfs kernel
    """
    def __init__(self, mu=None, sigma=None, img_shape=None, updates=False,
                 lattice_fn=lattice.gaussian_kernel_lattice, **kwargs):
        super(RF_Pool, self).__init__()
        self.mu = mu
        self.sigma = sigma
        if self.inputs['rfs'] is not None and self.img_shape is None:
            self.img_shape = self.inputs['rfs'].shape[-2:]
        else:
            self.img_shape = img_shape
        self.updates = updates
        self.lattice_fn = lattice_fn
        self.inputs.update(kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, **kwargs):
        # update inputs with kwargs
        self.inputs.update(kwargs)
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None and delta_sigma is not None:
            self.update_rfs(delta_mu, delta_sigma)
        # return pooling outputs
        return self.apply(u, **self.inputs)[2]

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
       return self.apply(u, **self.inputs)[2]

class RF_Window(Layer):
    """
    #TODO:WRTIEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Window, self).__init__()
        self.mu = mu
        self.sigma = sigma
        if self.inputs['rfs'] is not None and self.img_shape is None:
            self.img_shape = self.inputs['rfs'].shape[-2:]
        else:
            self.img_shape = img_shape
        self.lattice_fn = lattice_fn
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
        if 'mu' in self.inputs:
            self.inputs.pop('mu')
        # return pooling outputs
        return self.apply(u, **self.inputs)[2]
