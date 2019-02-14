
# coding: utf-8

# In[130]:

import torch
import rf
from utils import gaussian_lattice_utils

class RF_Pool(torch.nn.Module):
    """
    Receptive field pooling layer (see rf.pool for details)
    
    Attributes
    ----------
    rfs : list or torch.Tensor
        receptive fields to apply pooling over
        if type is list, index for receptive fields (see square_lattice_utils) 
        if type is torch.Tensor, kernels for receptive fields with shape 
        (h, w, n_kernels) (see gaussian_lattice_utils) 
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
        
    Methods
    -------
    apply(u, t=None)
        Apply pooling operation over receptive fields in input, u with optional
        top-down input, t. output is p_mean (see rf.pool)
    update_rfs(delta_mu, delta_sigma, 
        lattice_fn=gaussian_lattice_utils.gaussian_kernel_lattice)
        Update receptive fields kernels by adding delta_mu, delta_sigma and
        creating a new rfs attribute using lattice_fn.
    """
    def __init__(self, rfs=None, pool_type='prob', block_size=(2, 2), pool_args=[],
                 mu=None, sigma=None):
        super(RF_Pool, self).__init__()
        self.rfs = rfs
        self.pool_type = pool_type
        self.block_size = block_size
        self.pool_args = pool_args
        self.mu = mu
        self.sigma = sigma
    
    def __call__(self, u):
        return rf.pool(u, rfs=self.rfs, pool_type=self.pool_type, 
                       block_size=self.block_size, pool_args=self.pool_args)[2]
    
    def apply(self, u, t=None):
        return rf.pool(u, t, rfs=self.rfs, pool_type=self.pool_type,
                       block_size=self.block_size, pool_args=self.pool_args)
    
    def update_rfs(self, delta_mu, delta_sigma,
                   lattice_fn=gaussian_lattice_utils.gaussian_kernel_lattice):
        assert self.mu.shape == delta_mu.shape
        assert self.sigma.shape == delta_sigma.shape
        self.mu.add_(delta_mu)
        self.sigma.add_(delta_sigma)
        self.rfs = lattice_fn(self.mu, self.sigma, self.rfs.shape[0])
    
    def show_rfs(self):
        assert self.rfs is not None
        gaussian_lattice_utils.show_kernel_lattice(self.rfs)

