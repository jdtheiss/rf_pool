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
        self.inputs = {'u': None, 't': None, 
                       'rfs': None, 'pool_type': 'prob', 
                       'block_size': (2,2), 'pool_args': []}
        
    def __call__(self, *args):
        return self.forward(*args)

    def apply(self, **kwargs):
        return ops.rf_pool(**kwargs)
    
    def forward(self):
        pass
    
    def set(self, name, var):
        setattr(self, name, var)

    def get(self, name):
        return getattr(self, name)

    def init_rfs(self):
        assert self.lattice_fn is not None
        assert self.mu.shape[0] == self.sigma.shape[0]
        assert self.img_shape is not None
        self.inputs['rfs'] = self.lattice_fn(self.mu, self.sigma, self.img_shape)
    
    def update_rfs(self, delta_mu, delta_sigma):
        assert self.inputs['rfs'] is not None
        assert self.img_shape is not None
        rfs_shape = self.inputs['rfs'].shape[1:]
        # update mu if img_shape doesnt match rfs.shape[1:]
        if rfs_shape != self.img_shape:
            self.mu.add_(torch.sub(torch.as_tensor(self.img_shape, dtype=self.mu.dtype), 
                                   torch.as_tensor(rfs_shape, dtype=self.mu.dtype)))
        self.inputs['rfs'] = self.lattice_fn(self.mu + delta_mu, self.sigma + delta_sigma, self.img_shape)
    
    def update_mu_sigma(self, delta_mu, delta_sigma):
        self.mu.add_(delta_mu)
        self.sigma.add_(delta_sigma)
    
    def show_rfs(self):
        assert self.inputs['rfs'] is not None
        lattice.show_kernel_lattice(self.inputs['rfs'])

class RF_Pool(Layer):
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
            self.img_shape = self.inputs['rfs'].shape[1:]
        else:
            self.img_shape = img_shape
        self.updates = updates
        self.lattice_fn = lattice_fn
        self.inputs.update(kwargs)
                         
    def forward(self, u, t=None, delta_mu=None, delta_sigma=None):
        # set u, t
        self.inputs['u'] = u
        self.inputs['t'] = t
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if delta_mu is not None and delta_sigma is not None:
            self.update_rfs(delta_mu, delta_sigma)
            if self.updates:
                self.update_mu_sigma(delta_mu, delta_sigma)
        # return pooling outputs
        return self.apply(**self.inputs)[2]
    
class RF_Uniform(Layer):
    def __init__(self):
        super(RF_Uniform, self).__init__()
                         
    