import torch
import ops
from utils import lattice

class RF_Pool(torch.nn.Module):
    """
    Receptive field pooling layer (see ops.rf_pool for details)
    
    Attributes
    ----------
    rfs : torch.Tensor
        kernels containing receptive fields to apply pooling over with shape
        (n_kernels, h, w) (see lattice_utils)
        kernels are element-wise multiplied with (u+t) prior to pooling
        [default: None, applies pooling over blocks]
    pool_type : string
        type of pooling ('prob', 'stochastic', 'div_norm', 'average', 'sum')
        [default: 'prob']
    block_size : tuple
        size of blocks in hidden layer connected to pooling units 
        [default: (2,2)]
    pool_args : list
        extra arguments sent to pooling function indicated by pool_type
        (especially for div_norm_pool) [default: []]
    lattice_fn : utils.lattice function
        function used to update rfs kernels given delta_mu and delta_sigma
        [default: lattice.gaussian_kernel_lattice]
        
    Methods
    -------
    apply(u, t=None)
        Apply pooling operation over receptive fields in input, u with optional
        top-down input, t. output is p_mean (see ops.rf_pool)
    update_rfs(delta_mu, delta_sigma, 
        lattice_fn=gaussian_lattice_utils.gaussian_kernel_lattice)
        Update receptive fields kernels by adding delta_mu, delta_sigma and
        creating a new rfs attribute using lattice_fn.
    """
    def __init__(self, rfs=None, pool_type='prob', block_size=(2, 2), pool_args=[],
                 mu=None, sigma=None, img_shape=None, lattice_fn=lattice.gaussian_kernel_lattice):
        super(RF_Pool, self).__init__()
        self.rfs = rfs
        self.pool_type = pool_type
        self.block_size = block_size
        self.pool_args = pool_args
        self.mu = mu
        self.sigma = sigma
        if self.rfs is not None:
            self.img_shape = self.rfs.shape[1:]
        else:
            self.img_shape = img_shape
        self.lattice_fn = lattice_fn
    
    def __call__(self, u, delta_mu=None, delta_sigma=None):
        self.img_shape = u.shape[-2:]
        if delta_mu is not None and delta_sigma is not None:
            self.rfs = self.update_rfs(delta_mu, delta_sigma)
        return ops.rf_pool(u, rfs=self.rfs, pool_type=self.pool_type, 
                           block_size=self.block_size, pool_args=self.pool_args)[2]
    
    def apply(self, u, t=None, delta_mu=None, delta_sigma=None):
        self.img_shape = u.shape[-2:]
        if delta_mu is not None and delta_sigma is not None:
            self.rfs = self.update_rfs(delta_mu, delta_sigma)
        return ops.rf_pool(u, t, rfs=self.rfs, pool_type=self.pool_type,
                           block_size=self.block_size, pool_args=self.pool_args)
    
    def init_rfs(self):
        assert self.mu.shape[0] == self.sigma.shape[0]
        assert self.img_shape is not None
        self.rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape)
    
    def update_rfs(self, delta_mu, delta_sigma):
        assert self.mu.shape == delta_mu.shape
        assert self.sigma.shape == delta_sigma.shape
        assert self.img_shape is not None
        self.mu.add_(delta_mu)
        self.sigma.add_(delta_sigma)
        self.rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape)
    
    def show_rfs(self):
        assert self.rfs is not None
        lattice.show_kernel_lattice(self.rfs)