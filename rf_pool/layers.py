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
        (n_kernels, h, w) (see utils.lattice)
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
    mu : torch.Tensor
        #TODO:WRITEME
    sigma : torch.Tensor
        #TODO:WRITEME
    img_shape : tuple
        #TODO:WRITEME
    lattice_fn : utils.lattice function
        function used to update rfs kernels given delta_mu and delta_sigma
        [default: lattice.gaussian_kernel_lattice]
    updates : bool
        update mu and sigma on each call to update_rfs (True) or keep mu and
        sigma static (False [default])
        
    Methods
    -------
    apply(u, t=None)
        Apply pooling operation over receptive fields in input, u with optional
        top-down input, t. output is p_mean (see ops.rf_pool)
    init_rfs()
        Initialize receptive fields kernels with given lattice_fn, mu, sigma, 
        and img_shape
    update_rfs(delta_mu, delta_sigma, 
        lattice_fn=gaussian_lattice_utils.gaussian_kernel_lattice)
        Update receptive fields kernels by adding delta_mu, delta_sigma and
        creating a new rfs attribute using lattice_fn.
    update_mu_sigma(delta_mu, delta_sigma)
        Update mu and sigma by adding delta_mu, delta_sigma
    show_rfs()
        Display receptive field lattice from rfs kernel
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
        self.updates = False
    
    def __call__(self, u, delta_mu=None, delta_sigma=None):
        return self.apply(u, None, delta_mu, delta_sigma)[2]
    
    def apply(self, u, t=None, delta_mu=None, delta_sigma=None):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None and delta_sigma is not None:
            self.rfs = self.update_rfs(delta_mu, delta_sigma)
            if self.updates:
                self.update_mu_sigma(delta_mu, delta_sigma)
        # return pooling outputs
        return ops.rf_pool(u, t, rfs=self.rfs, pool_type=self.pool_type,
                           block_size=self.block_size, pool_args=self.pool_args)
    
    def init_rfs(self):
        assert self.mu.shape[0] == self.sigma.shape[0]
        assert self.img_shape is not None
        self.rfs = self.lattice_fn(self.mu, self.sigma, self.img_shape)
    
    def update_rfs(self, delta_mu, delta_sigma):
        assert self.img_shape is not None
        if self.rfs.shape[1:] != self.img_shape:
            self.mu.add_(torch.as_tensor(self.img_shape) - torch.as_tensor(self.rfs.shape[1:]))
        self.rfs = self.lattice_fn(self.mu + delta_mu, self.sigma + delta_sigma, self.img_shape)
    
    def update_mu_sigma(self, delta_mu, delta_sigma):
        self.mu.add_(delta_mu)
        self.sigma.add_(delta_sigma)
    
    def show_rfs(self):
        assert self.rfs is not None
        lattice.show_kernel_lattice(self.rfs)