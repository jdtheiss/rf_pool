import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import shift

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
        # check for optional kwargs
        options = functions.pop_attributes(kwargs, ['ratio','delta_mu',
                                           'delta_sigma','update_img_shape'])
        functions.set_attributes(self, **options)
        # set inputs for rf_pool
        self.rfs = self.update_rfs()
        self.pool_type = None
        self.kernel_size = None
        self.input_keys = ['rfs', 'pool_type', 'kernel_size']
        self.input_keys.extend(kwargs.keys())
        self.input_keys = np.unique(self.input_keys).tolist()
        functions.set_attributes(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def apply(self, *args, **kwargs):
        # get output type from kwargs
        output_type = functions.pop_attributes(kwargs, ['output']).get('output')
        # get inputs for rf_pool
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        # apply rf_pool
        output = ops.rf_pool(*args, **input_kwargs)
        # return output by type
        if output_type == 'all':
            return output
        elif output_type == 'h_mean':
            return output[1]
        elif output_type == 'h_sample':
            return output[2]
        else: # default p_mean
            return output[0]

    def set(self, **kwargs):
        functions.set_attributes(self, **kwargs)
        if 'mu' in kwargs or 'sigma' in kwargs or 'lattice_fn' in kwargs:
            self.rfs = self.update_rfs()
        elif 'img_shape' in kwargs:
            self.mu, self.sigma = self.update_mu_sigma()
            self.rfs = self.update_rfs()

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
            sigma = torch.sqrt(torch.exp(torch.add(torch.log(torch.pow(self.sigma, 2)),
                                                   delta_sigma)))
            self.delta_sigma = delta_sigma
        # update mu if img_shape doesnt match rfs.shape[-2:]
        if self.update_img_shape and self.rfs.shape[-2:] != self.img_shape:
            with torch.no_grad():
                img_diff = torch.sub(torch.tensor(self.img_shape, dtype=mu.dtype),
                                     torch.tensor(self.rfs.shape[-2:], dtype=mu.dtype))
                mu = torch.add(mu, img_diff / 2.)
        elif self.rfs.shape[-2:] != self.img_shape:
            raise Exception('rfs.shape[-2:] != self.img_shape')
        # update mu, sigma with priority map
        if priority_map is not None:
            mu, sigma = lattice.update_mu_sigma(mu, sigma, priority_map)
        return mu, sigma

    def update_rfs(self, mu=None, sigma=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if mu is None and sigma is None:
            return None
        assert self.lattice_fn is not None
        assert self.img_shape is not None
        # get rfs using lattice_fn
        args = [mu, sigma, self.img_shape]
        if self.ratio is not None:
            args.append(self.ratio)
        return self.lattice_fn(*args)

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
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Pool, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self.rfs = self.update_rfs(mu, sigma)
        # return pooling outputs
        return self.apply(u, **kwargs)

class RF_Uniform(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, n_kernels, img_shape, spacing, sigma_init=1.,
                lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        assert np.mod(np.sqrt(n_kernels), 1) == 0, (
            'sqrt of n_kernels must be integer'
        )
        # set mu, sigma
        centers = torch.as_tensor(img_shape)/2.
        n_kernel_side = np.sqrt(n_kernels).astype('int')
        mu, sigma = lattice.init_uniform_lattice(centers, n_kernel_side, spacing,
                                                 sigma_init)
        super(RF_Uniform, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
       self.rfs = self.update_rfs(mu, sigma)
       # return pooling outputs
       return self.apply(u, **kwargs)

class RF_Random(Layer):
    """
    #TODO:WRITEME
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
        super(RF_Random, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
       # set img_shape
       self.img_shape = u.shape[-2:]
       # update rfs, mu, sigma
       mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
       self.rfs = self.update_rfs(mu, sigma)
       # return pooling outputs
       return self.apply(u, **kwargs)

class RF_Window(Layer):
    """
    #TODO:WRTIEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Window, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, priority_map=None, **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, None, priority_map)
        self.rfs = self.update_rfs(mu, None)
        # return pooling outputs
        return self.apply(u, **kwargs)

class RF_Same(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Same, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                output='h_mean', **kwargs):
        assert kwargs.get('output') != 'all', (
            'keyword "output" cannot be "all" for RF_Same.'
        )
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self.rfs = self.update_rfs(mu, sigma)
        # return h_mean output
        pool_kwargs = functions.pop_attributes(kwargs, ['kernel_size',
                                                        'return_indices'])
        h_mean = self.apply(u, output=output, **kwargs)
        # update pool_kwargs if None
        if pool_kwargs.get('kernel_size') is None:
            pool_kwargs.update({'kernel_size': self.kernel_size})
        if pool_kwargs.get('return_indices') is None and hasattr(self,'return_indices'):
            pool_kwargs.update({'return_indices': self.return_indices})
        # pool if kernel_size in pool_kwargs
        if pool_kwargs.get('kernel_size') is not None:
            h_mean = F.max_pool2d(h_mean, **pool_kwargs)
        return h_mean

class RF_Squeeze(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Squeeze, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        assert kwargs.get('output') != 'all', (
            'keyword "output" cannot be "all" for RF_Squeeze.'
        )
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self.rfs = self.update_rfs(mu, sigma)
        # get p_mean without subsampling
        pool_kwargs = functions.pop_attributes(kwargs, ['kernel_size',
                                                        'return_indices'])
        p_mean = self.apply(u, **kwargs)
        # get squeezed coordinates
        coords = self.get_squeezed_coords(mu, sigma)
        # crop
        p_mean = self.crop_img(p_mean, coords)
        # update pool_kwargs if None
        if pool_kwargs.get('kernel_size') is None:
            pool_kwargs.update({'kernel_size': self.kernel_size})
        if pool_kwargs.get('return_indices') is None and hasattr(self,'return_indices'):
            pool_kwargs.update({'return_indices': self.return_indices})
        # pool if kernel_size in pool_kwargs
        if pool_kwargs.get('kernel_size') is not None:
            p_mean = F.max_pool2d(p_mean, **pool_kwargs)
        return p_mean

class RF_CenterCrop(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, crop_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        self.crop_size = torch.tensor(crop_size)
        super(RF_CenterCrop, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        assert kwargs.get('output') != 'all', (
            'keyword "output" cannot be "all" for RF_CenterCrop.'
        )
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self.rfs = self.update_rfs(mu, sigma)
        # get outputs
        p_mean = self.apply(u, **kwargs)
        # get coordinates of center size
        center = torch.max(torch.tensor(self.img_shape) // 2 - 1,
                           torch.tensor([0,0]))
        half_crop = self.crop_size // 2
        coords = torch.stack([center - half_crop,
                              center + half_crop + np.mod(self.crop_size, 2)])
        # return center crop
        return self.crop_img(p_mean, coords)

class RF_Compress(Layer):
    """
    #TODO:WRITEME
    """
    def __init__(self, center, scale, crop_size, mu, sigma, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        self.center = torch.tensor(center, dtype=torch.float)
        self.scale = torch.tensor(scale, dtype=torch.float)
        self.crop_size = torch.tensor(crop_size, dtype=torch.float)
        super(RF_Compress, self).__init__(mu, sigma, img_shape, lattice_fn, **kwargs)

    def forward(self, u, delta_mu=None, delta_sigma=None, priority_map=None,
                **kwargs):
        assert kwargs.get('output') != 'all', (
            'keyword "output" cannot be "all" for RF_Compress.'
        )
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        mu, sigma = self.update_mu_sigma(delta_mu, delta_sigma, priority_map)
        self.rfs = self.update_rfs(mu, sigma)
        # get shifts for output
        mu_diff = torch.sub(self.center, mu)
        shifts = torch.mul(torch.relu(self.scale), mu_diff)
        # create new mu
        new_mu = shifts + mu
        mu_mask = lattice.mask_kernel_lattice(new_mu, torch.tensor(0.5), self.img_shape)
        # get outputs
        retain_shape = functions.pop_attributes(kwargs, ['retain_shape']).get('retain_shape')
        _, h_mean, h_sample = self.apply(u, retain_shape=True, output='all', **kwargs)
        h_mean = torch.max(h_mean.flatten(-2), -1)[0].unsqueeze(-1)
        mu_mean = torch.mul(h_mean.unsqueeze(-1), mu_mask)
        mu_mean = torch.max(mu_mean, -3)[0]
        # get p_mean without subsampling
        pool_kwargs = functions.pop_attributes(kwargs, ['kernel_size',
                                                        'return_indices'])
        # update pool_kwargs if None
        if pool_kwargs.get('kernel_size') is None:
            pool_kwargs.update({'kernel_size': self.kernel_size})
        if pool_kwargs.get('return_indices') is None and hasattr(self,'return_indices'):
            pool_kwargs.update({'return_indices': self.return_indices})
        # pool if kernel_size in pool_kwargs
        if pool_kwargs.get('kernel_size') is not None:
            p_mean = F.max_pool2d(mu_mean, **pool_kwargs)

        # get coordinates for cropping image
        half_crop = self.crop_size // 2
        start_crop = torch.max(self.center - half_crop, torch.zeros(2))
        end_crop = torch.min(start_crop + self.crop_size,
                             torch.tensor(self.img_shape, dtype=torch.float))
        start_crop = end_crop - self.crop_size
        coords = torch.stack([start_crop, end_crop]).int()
        # return center crop
        return self.crop_img(p_mean, coords)
