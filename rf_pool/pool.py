import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function

import pool
from .utils import functions, lattice, visualize

class Pool(torch.nn.Module):
    """
    Base class for receptive field pooling layers

    Parameters
    ----------
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

    Methods
    -------
    apply(input, **kwargs) : apply pooling function only
    forward(*args, **kwargs) : apply forward pass through pool layer
    set(**kwargs) : set attributes for pool layer
    get(keys, default=None) : get attributes from pool layer
    show_lattice(x=None, figsize=(5,5), cmap=None, **kwargs) : show pool lattice
    update_mu_sigma(delta_mu, delta_sigma, fn) : update mu/sigma with shifts
        or function
    apply_attentional_field(attention_mu, attention_sigma) : update mu/sigma via
        gaussian multiplication with a gaussian attentional field
    crop_img(input, coords) : crop input with bounding box coordinates

    See Also
    --------
    pool.apply
    """
    __methodsdoc__ = functions.get_doc(__doc__, 'Methods', end_field='See Also')
    def __init__(self, mu, sigma, img_shape, lattice_fn, **kwargs):
        super(Pool, self).__init__()
        # input parameters
        self.mu = mu
        self.sigma = sigma
        self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        # check for optional kwargs
        options = functions.pop_attributes(kwargs, ['delta_mu', 'delta_sigma',
                                                    'attention_mu','attention_sigma',
                                                    'update_mu','update_sigma'])
        functions.set_attributes(self, **options)
        self.apply_attentional_field()
        # set inputs for rf_pool
        self.rfs = None
        self.rf_indices = None
        self._update_rfs()
        self.pool_fn = None
        self.kernel_size = None
        self.apply_mask = False
        self.input_keys = ['rfs', 'rf_indices', 'pool_fn', 'kernel_size',
                           'apply_mask']
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

    def get(self, keys, default=None):
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        default = functions.parse_list_args(len(keys), default)[0]
        output = []
        for i, key in enumerate(keys):
            if hasattr(self, key) and getattr(self, key) is not None:
                output.append(getattr(self, key))
            else:
                output.append(*default[i])
        return output

    def update_mu_sigma(self, delta_mu=None, delta_sigma=None, fn=None, **kwargs):
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
        # update mu, sigma with a lattice function
        if fn is not None:
            mu, sigma = fn(mu, sigma, **kwargs)
        self.tmp_mu, self.tmp_sigma = mu, sigma
        return mu, sigma

    def _update_rfs(self, mu=None, sigma=None, lattice_fn=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if lattice_fn is None:
            lattice_fn = self.lattice_fn
        if mu is None or sigma is None:
            return
        assert self.lattice_fn is not None
        assert self.img_shape is not None
        # get rfs using lattice_fn
        args = [mu, sigma, self.img_shape]
        self.rfs = lattice_fn(*args)
        # get rf_indices using mask rfs
        if lattice_fn is not lattice.mask_kernel_lattice:
            self.rf_indices = rf_to_indices(lattice.mask_kernel_lattice(*args))
            self.apply_mask = True
        else:
            self.rf_indices = rf_to_indices(self.rfs)

    def update_rfs(self, mu=None, sigma=None, lattice_fn=None):
        if mu is not None:
            self.set(mu=mu)
        if sigma is not None:
            self.set(sigma=sigma)
        if lattice_fn is not None:
            self.set(lattice_fn=lattice_fn)
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

    def show_lattice(self, x=None, figsize=(5,5), cmap=None, **kwargs):
        # get mu, sigma
        if self.get('tmp_mu')[0] is not None:
            mu = self.tmp_mu
        else:
            mu = self.mu
        if self.get('tmp_sigma')[0] is not None:
            sigma = self.tmp_sigma
        else:
            sigma = self.sigma
        if x is not None:
            # show input
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            x = torch.squeeze(x.permute(0,2,3,1), -1).numpy()
            x = x - np.min(x, axis=(1,2), keepdims=True)
            x = x / np.max(x, axis=(1,2), keepdims=True)
            axes[0].imshow(x[0], cmap=cmap)
            visualize.scatter_rfs(mu, sigma, self.img_shape, ax=axes[1],
                                  **kwargs)
        else: # visualize rfs
            visualize.scatter_rfs(mu, sigma, self.img_shape, figsize=figsize,
                                  **kwargs)

    def apply_attentional_field(self, attention_mu=None, attention_sigma=None,
                                **kwargs):
        if attention_mu is None:
            attention_mu = self.get('attention_mu')[0]
        if attention_sigma is None:
            attention_sigma = self.get('attention_sigma')[0]
        if attention_mu is None or attention_sigma is None:
            return self.mu, self.sigma
        mu, sigma = lattice.multiply_gaussians(self.mu, self.sigma,
                                               attention_mu, attention_sigma)
        if self.get('update_mu', kwargs.get('update_mu'))[0] is False:
            self.tmp_mu = self.mu
        else:
            self.tmp_mu = mu
        if self.get('update_sigma', kwargs.get('update_sigma'))[0] is False:
            self.tmp_sigma = self.sigma
        else:
            self.tmp_sigma = sigma
        return self.tmp_mu, self.tmp_sigma

    def forward(self, u, **kwargs):
        """
        #TODO:WRITEME
        """
        # set img_shape
        self.img_shape = u.shape[-2:]
        # update rfs, mu, sigma
        if any([self.get(k, kwargs.get(k))
                for k in ['delta_mu','delta_sigma','fn']]):
            mu, sigma = self.update_mu_sigma(**kwargs)
            self._update_rfs(mu, sigma)
        # apply attentional field
        if any([self.get(k, kwargs.get(k))
                for k in ['attention_mu','attention_sigma']]):
            mu, sigma = self.apply_attentional_field(**kwargs)
            self._update_rfs(mu, sigma)
        # return pooling outputs
        return self.apply(u, **kwargs)

class RF_Pool(Pool):
    __doc__ = functions.update_doc(Pool.__doc__, lines=[1],
                                   updates=['Receptive field pooling layer'])
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Pool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                      **kwargs)

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
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, img_shape, n_kernels, spacing, sigma_init=1.,
                offset=[0.,0.], rotate=0.,
                lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set mu, sigma
        mu, sigma = lattice.init_uniform_lattice(img_shape, n_kernels, spacing,
                                                 sigma_init, offset, rotate)
        super(RF_Uniform, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         **kwargs)
class RF_Hexagon(Pool):
    """
    Receptive field pooling layer with RFs initated using
    lattice.init_hexagon_lattice

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
    lattice.init_hexagon_lattice
    """
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, img_shape, n_kernels, spacing, sigma_init=1.,
                 offset=[0.,0.], rotate=0.,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set mu, sigma
        mu, sigma = lattice.init_hexagon_lattice(img_shape, n_kernels, spacing,
                                                 sigma_init, offset, rotate)
        super(RF_Hexagon, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         **kwargs)

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

    See Also
    --------
    pool.apply

    Notes
    -----
    If 'mu' or 'sigma' are not provided, they are set to random locations within
    the image shape and random sizes less than half the image shape, respectively.
    """
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
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

class RF_Squeeze(Pool):
    __doc__ = functions.update_doc(Pool.__doc__, lines=[1],
                                   updates=[['Receptive field pooling layer',
                                            'with output image size squeezed ',
                                            'to extent of RF lattice']])
    def __init__(self, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(RF_Squeeze, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         **kwargs)

    def forward(self, u, **kwargs):
        # apply forward function
        output = Pool.forward(self, u, **kwargs)
        # get squeezed coordinates
        coords = self.get_squeezed_coords(self.mu, self.sigma)
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
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, crop_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        self.crop_size = torch.tensor(crop_size)
        super(RF_CenterCrop, self).__init__(mu, sigma, img_shape, lattice_fn,
                                            **kwargs)

    def forward(self, u, **kwargs):
        # apply forward function
        output = Pool.forward(self, u, **kwargs)
        # get coordinates of center size
        center = torch.max(torch.tensor(self.img_shape) // 2 - 1,
                           torch.tensor([0,0]))
        half_crop = self.crop_size // 2
        coords = torch.stack([center - half_crop,
                              center + half_crop + np.mod(self.crop_size, 2)])
        # return center crop
        return self.crop_img(output, coords)

class RF_Foveated(Pool):
    """
    Receptive field pooling layer with foveated sampling and optional
    cortical weighting

    Parameters
    ----------
    img_shape : tuple
        receptive field/detection layer shape [default: None]
    scale : float
        scaling rate for foveated RF array (see lattice.init_foveated_lattice)
    n_rings : int
        number of rings for foveated RF array (see lattice.init_foveated_lattice)
    cortical_mu : torch.Tensor, optional
        center for gaussian in cortical space to weight pooled outputs with
        shape (1, 2) [default: None]
    cortical_sigma : torch.Tensor, optional
        standard deviation for gaussian in cortical space with shape (1, 1)
        [default: None]
    cortical_kernel_fn : utils.lattice function, optional
        function used to sample weights from cortical kernel
        [default: lattice.exp_kernel_2d]
    lattice_fn : utils.lattice function, optional
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.mask_kernel_lattice]
    **kwargs : dict
        kwargs passed to lattice.init_foveated_lattice or pool.apply
        (see pool.apply, other ops pooling functions)

    See Also
    --------
    pool.apply
    lattice.init_foveated_lattice
    lattice.cortical_xy
    """
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, img_shape=None, scale=None, n_rings=None,
                 cortical_mu=None, cortical_sigma=None,
                 cortical_kernel_fn=lattice.exp_kernel_2d,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set variables
        self.scale = torch.tensor(scale)
        self.n_rings = n_rings
        self.cortical_mu = cortical_mu
        self.cortical_sigma = cortical_sigma
        self.cortical_kernel_fn = cortical_kernel_fn
        # get mu, sigma from init_foveated_lattice
        kws = ['spacing','std','n_rf','offset','min_ecc',
               'rotate_rings','rotate','keep_all_RFs']
        defaults = lattice.init_foveated_lattice.__defaults__
        self.lattice_kwargs = dict([(k, kwargs.pop(k)) if kwargs.get(k) else (k, v)
                                    for k, v in zip(kws, defaults)])
        mu, sigma = lattice.init_foveated_lattice(img_shape, scale, n_rings,
                                                  **self.lattice_kwargs)
        super(RF_Foveated, self).__init__(mu, sigma, img_shape, lattice_fn,
                                          **kwargs)
        # set cortical weight vars
        self.get_cortical_vars(self.lattice_kwargs)

    def get_cortical_vars(self, lattice_kwargs):
        if lattice_kwargs.get('n_rf') is None:
            n_rf = np.pi / self.scale
        else:
            n_rf = lattice_kwargs.get('n_rf')
        self.rf_angle = 2. * np.pi * torch.linspace(0., 1., int(n_rf))[1]
        self.rotate = torch.tensor(lattice_kwargs.get('rotate'))
        self.offset = torch.tensor(lattice_kwargs.get('offset'))
        self.offset += (torch.tensor(self.img_shape, dtype=torch.float) // 2)

    def get_cortical_weights(self, cortical_mu=None, cortical_sigma=None, beta=0.):
        if cortical_mu is None:
            cortical_mu = self.cortical_mu
        if cortical_sigma is None:
            cortical_sigma = self.cortical_sigma
        if cortical_mu is None or cortical_sigma is None:
            return None
        cortical_RFs = lattice.cortical_xy(self.mu - self.offset, self.scale,
                                           self.rf_angle, beta, self.rotate)
        weights = self.cortical_kernel_fn(cortical_mu, cortical_sigma,
                                          cortical_RFs.t().reshape(1,2,-1,1))
        return weights.reshape(1, 1, -1, 1, 1)

    def show_cortical_weights(self, cortical_mu=None, cortical_sigma=None, beta=0.):
        # get weights
        weights = self.get_cortical_weights(cortical_mu, cortical_sigma, beta)
        # get cortical location of RFs
        cortical_RFs = lattice.cortical_xy(self.mu - self.offset, self.scale,
                                           self.rf_angle, beta, self.rotate)
        # make gray cmap
        cmap = visualize.create_cmap(r=(0.5,1.), g=(0.5,1.), b=(0.5,1.))
        # scatter plot of RFs in cortical space
        fig = plt.figure()
        plt.subplot(111, facecolor='gray')
        sc = plt.scatter(cortical_RFs[:,1], cortical_RFs[:,0],
                         c=weights.flatten(), cmap=cmap)
        sc.set_clim([0.,1.])
        cbar = plt.colorbar()
        cbar.set_label('RF Weight')
        xlim, ylim = sc.axes.get_xlim(), sc.axes.get_ylim()
        sc.axes.set_ylim(np.flip(ylim))
        return fig;

    def forward(self, u, **kwargs):
        # set img_shape
        self.img_shape = u.shape[-2:]
        # get cortical_kwargs from kwargs
        cortical_kwargs = functions.pop_attributes(kwargs,
                                                   ['cortical_mu',
                                                    'cortical_sigma',
                                                    'beta'],
                                                   ignore_keys=True)
        # weight output by cortical locations relative to cortical_mu
        self.get_cortical_vars(self.lattice_kwargs)
        weights = self.get_cortical_weights(**cortical_kwargs)
        # get retain_shape from kwargs
        if kwargs.get('retain_shape') is None:
            retain_shape = False
        else:
            retain_shape = kwargs.pop('retain_shape')
        # apply forward function
        output = Pool.forward(self, u, retain_shape=True, **kwargs)
        # apply weights
        if weights is not None:
            output = torch.mul(output, weights)
        if not retain_shape and not self.get('retain_shape')[0]:
            output = torch.max(output, -3)[0]
        return output

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
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(MaxPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                      kernel_size=kernel_size,
                                      pool_fn='max_pool', **kwargs)

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
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(ProbmaxPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                          kernel_size=kernel_size,
                                          pool_fn='probmax_pool', **kwargs)

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
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, kernel_size, mu=None, sigma=None, img_shape=None,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        super(StochasticPool, self).__init__(mu, sigma, img_shape, lattice_fn,
                                             kernel_size=kernel_size,
                                             pool_fn='stochastic_pool', **kwargs)

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
    with torch.no_grad():
        rf_indices = torch.zeros_like(rfs.flatten(1))
        for i, rf in enumerate(rfs):
            idx = torch.nonzero(rf.flatten())
            rf_indices[i,:idx.numel()] = idx.flatten()
    return rf_indices

# cpp pooling functions
def max_pool(input, **kwargs):
    if 'rfs' in kwargs and 'rf_indices' not in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.get('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='max_pool', **kwargs)

def probmax(input, **kwargs):
    if 'rfs' in kwargs and 'rf_indices' not in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.get('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='probmax', **kwargs)

def probmax_pool(input, **kwargs):
    if 'rfs' in kwargs and 'rf_indices' not in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.get('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='probmax_pool', **kwargs)

def stochastic_pool(input, **kwargs):
    if 'rfs' in kwargs and 'rf_indices' not in kwargs:
        kwargs.setdefault('rf_indices', rf_to_indices(kwargs.get('rfs')))
    kwargs.setdefault('grad_fn', None)
    return apply(input, pool_fn='stochastic_pool', **kwargs)

def unpool(input, index_mask, **kwargs):
    return apply(input, pool_fn='unpool', mask=index_mask, **kwargs)

def apply(u, pool_fn=None, rfs=None, rf_indices=None, kernel_size=None,
          stride=None, retain_shape=False, return_indices=False, apply_mask=False,
          **kwargs):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        input to pooling layer with shape (batch_size, ch, h, w)
    pool_fn : str
        pooling function (e.g., 'max_pool', 'probmax_pool', 'stochastic_pool').
        [default: None]
    rfs : torch.Tensor or None
        tensor containing receptive field kernels to apply pooling over with
        shape (n_kernels, h, w)
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
    apply_mask : bool, optional
        boolean whether to multiply rfs with input prior to applying pool_fn
        [default: False]
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
    if rfs is not None:
        kwargs.setdefault('mask', rfs.data)
    kwargs.setdefault('mask_indices', rf_indices)
    kwargs.setdefault('kernel_size', kernel_size)
    kwargs.setdefault('stride', stride)
    kwargs.setdefault('retain_shape', retain_shape)
    kwargs.setdefault('apply_mask', apply_mask)

    # apply cpp pooling function
    input = u.data.flatten(0,1)
    outputs = list(pool_fn(input, **kwargs))
    for i, output in enumerate(outputs):
        if output is not None:
            if retain_shape:
                output_shape = (batch_size, ch, -1) + output.shape[1:]
            else:
                output_shape = (batch_size, ch,) + output.shape[1:]
            output = output.reshape(output_shape)
            outputs[i] = torch.as_tensor(output)

    # return without grad if less than 3 outputs
    if len(outputs) < 3:
        return outputs[0]

    # set grad args
    grad_args = [u, rfs, grad_fn, *outputs, kernel_size, stride, retain_shape,
                 apply_mask]

    # return with indices
    if return_indices:
        return _PoolGrad_with_indices.apply(*grad_args)

    return _PoolGrad.apply(*grad_args)

def _default_grad(grad_output, input, rfs, index_mask=None, index_kernel=None,
                  kernel_size=None, stride=None, retain_shape=False,
                  apply_mask=False):
    #TODO: deal with retain_shape in gradient
    input_shape = list(input.shape)
    if retain_shape:
        img_dim = 3
        input_shape.insert(2, grad_output.shape[2])
        grad_input = torch.zeros(input_shape)
    else:
        img_dim = 2
        grad_input = torch.zeros(input_shape)
    # set output gradients to locations in input space
    if index_kernel is not None:
        grad_input.flatten(img_dim).scatter_(img_dim, index_kernel.flatten(img_dim),
                                             grad_output.flatten(img_dim));
        grad_input = torch.reshape(grad_input, input_shape)
    else:
        grad_input = grad_output
    # set gradients outside rf to 0
    if index_mask is not None:
        index_mask = torch.where(index_mask == 0)
        grad_input[index_mask] = 0.
    # if multiplied with rfs, multiply with grad_input
    if apply_mask:
        if not retain_shape:
            grad_input = torch.unsqueeze(grad_input, -3)
        grad_input = torch.sum(torch.mul(grad_input, rfs), -3)
    elif retain_shape:
        grad_input = torch.sum(grad_input, -3)
    # if rfs used, set grad_rfs
    if rfs is not None:
        grad_rfs = torch.zeros_like(rfs)
        idx = torch.where(rfs)
        g = torch.sum(torch.mul(grad_input, input), [0,1]).unsqueeze(0)
        grad_rfs[idx] = g.repeat(rfs.shape[0],1,1)[idx]
    else:
        grad_rfs = None
    return grad_input, grad_rfs

def _apply_grad(grad_output, input, rfs, grad_fn, index_mask=None,
                index_kernel=None, kernel_size=None, stride=None,
                retain_shape=False, apply_mask=False):
    # assert pool_fn in pool and get pool_grad
    assert hasattr(pool, grad_fn)
    grad_fn = getattr(pool, grad_fn)

    # set kwargs
    kwargs = {}
    if rfs is not None:
        kwargs.update({'mask': rfs.data})
    kwargs.update({'mask_indices': index_mask, 'kernel_size': kernel_size,
                   'stride': stride, 'retain_shape': retain_shape,
                   'apply_mask': apply_mask})

    # apply grad function #TODO: make flexible using output tuple
    grad_input = grad_fn(grad_output.data.flatten(0,1), **kwargs)
    grad_input = torch.as_tensor(grad_input, dtype=grad_output.dtype)
    output_shape = grad_output.shape[:2] + grad_input.shape[2:]
    grad_input = grad_input.reshape(output_shape)

    # set grad for rfs
    if rfs is not None:
        grad_rfs = torch.zeros_like(rfs)
        idx = torch.where(rfs)
        g = torch.sum(torch.mul(grad_input, input), [0,1]).unsqueeze(0)
        grad_rfs[idx] = g.repeat(rfs.shape)[idx]
    else:
        grad_rfs = None

    return grad_input, grad_rfs

class _PoolGrad(Function):
    @staticmethod
    def forward(ctx, input, rfs, grad_fn, output, *args):
        ctx.grad_fn = grad_fn
        ctx.grad_args = args
        ctx.save_for_backward(input, rfs)
        output.requires_grad_(input.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, rfs = ctx.saved_tensors
        if ctx.grad_fn is not None:
            grad_input, grad_rfs = _apply_grad(grad_output, input, rfs,
                                               ctx.grad_fn, *ctx.grad_args)
        else:
            grad_input, grad_rfs = _default_grad(grad_output, input, rfs,
                                                 *ctx.grad_args)
        return (grad_input, grad_rfs, *[None]*(len(ctx.grad_args) + 2))

class _PoolGrad_with_indices(Function):
    @staticmethod
    def forward(ctx, input, rfs, grad_fn, output, *args):
        ctx.grad_fn = grad_fn
        ctx.grad_args = args
        ctx.save_for_backward(input, rfs)
        output.requires_grad_(input.requires_grad)
        if args[0] is None:
            return output, args[1]
        elif args[1] is None:
            return output, args[0]
        return output, args[0], args[1]

    @staticmethod
    def backward(ctx, grad_output, index_mask, index_kernel):
        input, rfs = ctx.saved_tensors
        if ctx.grad_fn is not None:
            grad_input, grad_rfs = _apply_grad(grad_output, input, rfs,
                                               ctx.grad_fn, *ctx.grad_args)
        else:
            grad_input, grad_rfs = _default_grad(grad_output, input, rfs,
                                                 *ctx.grad_args)
        return (grad_input, grad_rfs, *[None]*(len(ctx.grad_args) + 2))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
