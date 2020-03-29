from collections import OrderedDict
import inspect

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function

import pool
from .modules import RBM
from .utils import functions, lattice, visualize

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
    >>> pool_output = apply(u, 'sum_pool', rf_indices=rf_indices,
        kernel_size=(2,2))

    See Also
    --------
    pool.max_pool
    pool.probmax
    pool.probmax_pool
    pool.stochastic_pool
    pool.unpool
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

# set param docstring based on apply docstring parameters
__paramsdoc__ = functions.get_doc(apply.__doc__, 'pool_fn', end_field='**kwargs')

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

# set docstr from pool.function
[setattr(f, '__doc__', getattr(pool, n).__doc__)
 for n, f in zip(['max_pool','probmax','probmax_pool','stochastic_pool','unpool'],
                 [max_pool,probmax,probmax_pool,stochastic_pool,unpool])]

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

    Optional kwargs (see methods below for additional kwargs)

    Methods
    -------
    pool(input, **kwargs) : apply pooling function only
    forward(*args, **kwargs) : apply forward pass through pool layer
    set(**kwargs) : set attributes for pool layer
    get(keys, default) : get attributes from pool layer
    show_lattice(figsize, cmap, x, **kwargs) : show pool lattice
    crop_img(input, coords) : crop input with bounding box coordinates
    adaptive_update(img_shape, adaptive) : update mu/sigma proportional to
        change in img_shape
    shift_mu_sigma(delta_mu, delta_sigma, fn) : update mu/sigma with shifts
        or function. These kwargs can also be set at initialization.
    apply_RBM(u, training, optimizer, train_kwargs **kwargs) : get attentional
        Gaussians to update mu/sigma
    apply_attentional_field(attention_mu, attention_sigma, **kwargs) : update
        mu/sigma via gaussian multiplication with a gaussian attentional field.
        These kwargs can also be set at initialization.
    vectorize_output(output, vectorize) : vectorize pooled outputs to shape
        (batch, ch, n_kernels). These kwargs can also be set at initialization.
    weight_output(output, RF_weights) : weight outputs using given tensor. These
        kwargs can also be set at initialization.

    See Also
    --------
    pool.apply
    """
    __doc__ = functions.update_doc(__doc__, 'Methods', [-1], [__paramsdoc__])
    __methodsdoc__ = functions.get_doc(__doc__, 'Optional kwargs',
                                       end_field='See Also')
    def __init__(self, mu, sigma, img_shape, lattice_fn, **kwargs):
        super(Pool, self).__init__()
        # input parameters
        self.mu = mu
        self.sigma = sigma
        self.img_shape = img_shape
        self.lattice_fn = lattice_fn
        # check for optional kwargs
        self.option_keys = ['adaptive','delta_mu','delta_sigma','fn',
                            'training','attention_mu','attention_sigma','weight',
                            'update_mu','update_sigma','vectorize','RF_weights']
        options = functions.pop_attributes(kwargs, self.option_keys)
        functions.set_attributes(self, **options)
        self.apply_attentional_field(**options)
        # set inputs for rf_pool
        self.rfs = None
        self.rf_indices = None
        self._update_rfs(self.mu, self.sigma, self.lattice_fn)
        self.pool_fn = None
        self.kernel_size = None
        self.apply_mask = False
        self.input_keys = ['rfs','rf_indices','pool_fn','kernel_size','apply_mask']
        self.input_keys.extend(kwargs.keys())
        self.input_keys = np.unique(self.input_keys).tolist()
        functions.set_attributes(self, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def pool(self, input, **kwargs):
        # get inputs for rf_pool
        input_kwargs = functions.get_attributes(self, self.input_keys)
        input_kwargs.update(kwargs)
        # apply rf_pool
        return apply(input, **input_kwargs)

    def set(self, **kwargs):
        functions.set_attributes(self, **kwargs)

    def get(self, *args, **kwargs):
        output = {}.fromkeys(args, None)
        output.update(kwargs)
        for key, value in output.items():
            if hasattr(self, key) and getattr(self, key) is not None:
                output.update({key: getattr(self, key)})
        if len(output) == 1:
            return list(output.values())[0]
        return output

    def _update_rfs(self, mu=None, sigma=None, lattice_fn=None, img_shape=None):
        if all([x is None for x in [mu, sigma, lattice_fn, img_shape]]):
            return
        # get mu, sigma, lattice_fn, img_shape
        if mu is None:
            mu = self.get('mu')
        if sigma is None:
            sigma = self.get('sigma')
        if lattice_fn is None:
            lattice_fn = self.get('lattice_fn')
        if img_shape is None:
            img_shape = self.get('img_shape')
        if mu is None or sigma is None:
            return
        assert lattice_fn is not None
        assert img_shape is not None
        # get rfs using lattice_fn
        args = [mu, sigma, img_shape]
        self.rfs = lattice_fn(*args)
        # get rf_indices using mask rfs
        if lattice_fn is not lattice.mask_kernel_lattice:
            self.rf_indices = rf_to_indices(lattice.mask_kernel_lattice(*args))
            self.apply_mask = True
        else:
            self.rf_indices = rf_to_indices(self.rfs)

    def update_rfs(self, mu=None, sigma=None, lattice_fn=None, img_shape=None):
        """
        Update RFs by changing mu, sigma, lattice_fn, and/or img_shape

        Parameters
        ----------
        mu : torch.Tensor, optional
            center positions for RFs [default: None, set to self.mu]
        sigma : torch.Tensor, optional
            sizes for RFs [defualt: None, set to self.sigma]
        lattice_fn : function, optional
            lattice function (e.g., lattice.mask_kernel_lattice) for RFs
            [defualt: None, set to self.lattice_fn]
        img_shape : tuple, optional
            size of input images [default: None, set to self.img_shape]

        Returns
        -------
        rfs : torch.Tensor
            tensor of RFs with shape (n_kernels, h, w) used to pool inputs in forward
            call

        Notes
        -----
        This function sets mu, sigma, lattice_fn, img_shape, and rfs attributes
        of the pool layer class.
        """
        if mu is not None:
            self.set(mu=mu)
        if sigma is not None:
            self.set(sigma=sigma)
        if lattice_fn is not None:
            self.set(lattice_fn=lattice_fn)
        if img_shape is not None:
            self.set(img_shape=img_shape)
        self._update_rfs(mu, sigma, lattice_fn, img_shape)
        return self.rfs

    def get_squeezed_coords(self, mu, sigma):
        """
        Get min, max image coordinates from mu, sigma

        Parameters
        ----------
        mu : torch.Tensor
            mu positions with shape (n_kernels, 2)
        sigma : torch.Tensor
            sigma sizes with shape (n_kernels, 1)

        Returns
        -------
        coords : list
            coordinates with indices like [min_h, min_w, max_h, max_w]
        """
        # find min, max mu
        min_mu, min_idx = torch.min(mu, dim=0)
        max_mu, max_idx = torch.max(mu, dim=0)
        min_sigma = sigma[min_idx].t()
        max_sigma = sigma[max_idx].t()
        return torch.cat([min_mu - min_sigma, max_mu + max_sigma], -1).tolist()

    def crop_img(self, input, coords):
        """
        Crop image dimensions (-2, -1) using coordinates

        Parameters
        ----------
        input : torch.Tensor
            input tensor to crop with shape[-2:] == (h, w)
        coords : list
            coordinates with indices like [min_h, min_w, max_h, max_w]

        Returns
        -------
        output : torch.Tensor
            cropped tensor
        """
        output = torch.flatten(input, 0, -3)
        output = output[:, coords[0]:coords[2], coords[1]:coords[3]]
        return torch.reshape(output, input.shape[:-2] + output.shape[-2:])

    def show_lattice(self, figsize=(5,5), cmap=None, x=None, **kwargs):
        """
        Show current RF lattice

        Parameters
        ----------
        figsize : tuple
            figure size [default: (5,5)]
        cmap : str or matplotlib.colors.LinearSegmentedColormap
            colormap used for plotting
        x : torch.Tensor, optional
            image to display on top of RF lattice (e.g., stimulus)
        **kwargs : **dict
            keyword arguments passed to visualize.scatter_rfs

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing plotted RF lattice

        See Also
        --------
        visualize.scatter_rfs
        """
        # get mu, sigma
        if self.get('tmp_mu') is not None:
            mu = self.tmp_mu
        else:
            mu = self.mu
        if self.get('tmp_sigma') is not None:
            sigma = self.tmp_sigma
        else:
            sigma = self.sigma
        if self.get('tmp_img_shape') is not None:
            img_shape = self.tmp_img_shape
        else:
            img_shape = self.img_shape
        if x is not None:
            # show input
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            x = torch.squeeze(x.permute(0,2,3,1), -1).numpy()
            x = x - np.min(x, axis=(1,2), keepdims=True)
            x = x / np.max(x, axis=(1,2), keepdims=True)
            axes[0].imshow(x[0], cmap=cmap)
            return visualize.scatter_rfs(mu, sigma, x.shape[-2:], ax=axes[1],
                                         **kwargs)
        # visualize rfs
        return visualize.scatter_rfs(mu, sigma, img_shape, figsize=figsize,
                                     **kwargs)

    def init_mu_sigma(self, init_fn, img_shape, **kwargs):
        self.init_fn = init_fn
        argspec = inspect.getfullargspec(self.init_fn)
        kws, defaults = argspec.args[-len(argspec.defaults):], argspec.defaults
        self.init_kwargs = OrderedDict(zip(kws, defaults))
        updates = dict([(k, kwargs.pop(k)) for k in argspec.args
                        if kwargs.get(k) is not None])
        self.init_kwargs.update(updates)
        return self.init_fn(img_shape, **self.init_kwargs)

    def shift_mu_sigma(self, delta_mu=None, delta_sigma=None, fn=None, **kwargs):
        """
        Shift mu, sigma by some delta value

        Parameters
        ----------
        delta_mu : torch.Tensor
            value by which to shift mu positions with shape (-1, 2)
        delta_sigma : torch.Tensor
            value by which to change sigma sizes with shape (-1, 1)
            updated as `new_sigma = sqrt(exp(log(sigma**2) + delta_sigma))`
        fn : function, optional
            function used to produce new mu, sigma values
            (e.g., lattice.mask_kernel_lattice)
        **kwargs : **dict, optional
            keyword arguments passed to fn

        Returns
        -------
        mu : torch.Tensor
            updated mu values
        sigma : torch.Tensor
            updated sigma values

        See Also
        --------
        adaptive_update
        """
        if self.mu is None and self.sigma is None:
            return None, None
        else:
            mu = self.mu
            sigma = self.sigma
        # add delta_mu to mu
        if delta_mu is not None:
            mu = torch.add(self.mu, delta_mu)
        # add delta_sigma to log(sigma**2) then exponentiate and sqrt
        if delta_sigma is not None:
            log_sigma = torch.add(torch.log(torch.pow(self.sigma, 2)),
                                  delta_sigma)
            sigma = torch.sqrt(torch.exp(log_sigma))
        # update mu, sigma with a lattice function
        if fn is not None:
            mu, sigma = fn(mu, sigma, **kwargs)
        self.tmp_mu, self.tmp_sigma = mu, sigma
        # if all None, return None
        if all([x is None for x in [delta_mu, delta_sigma, fn]]):
            return None, None
        return mu, sigma

    def adaptive_update(self, img_shape, adaptive=True):
        """
        Update mu and sigma proportional to changes in img_shape.

        Parameters
        ----------
        img_shape : tuple
            new image shape by which mu/sigma will be updated
        adaptive : boolean
            True/False update mu/sigma proportional to img_shape
            [default: True]

        Returns
        -------
        mu : torch.Tensor
            updated mu values
        sigma : torch.Tensor
            updated sigma values

        See Also
        --------
        shift_mu_sigma
        """
        # return None if no mu/sigma or adaptive is False
        if self.mu is None and self.sigma is None or adaptive is False:
            return None, None
        else:
            mu = self.mu
            sigma = self.sigma
        # if img_shape unchanged, return None
        self.tmp_img_shape = img_shape
        img_shape = torch.as_tensor(img_shape, dtype=mu.dtype)
        img_shape0 = torch.as_tensor(self.img_shape, dtype=mu.dtype)
        if torch.all(torch.eq(img_shape, img_shape0)):
            return None, None
        # get ratio
        ratio = torch.div(img_shape, img_shape0)
        # multiply by ratio, return tmp_mu, tmp_sigma
        self.tmp_mu = self.mu * ratio
        self.tmp_sigma = self.sigma * torch.mean(ratio)
        return self.tmp_mu, self.tmp_sigma

    def apply_attentional_field(self, attention_mu=None, attention_sigma=None,
                                weight=None, **kwargs):
        """
        Apply attentional field via Gaussian multiplication between an attentional
        Gaussian and each RF Gaussian

        Parameters
        ----------
        attention_mu : torch.Tensor
            x,y center location of attentional Gaussian with shape (n_Gaussians,2)
        attention_sigma : torch.Tensor
            size of attentional Gaussian with shape (n_Gaussians,1)
        weight : torch.Tensor
            weight associated with each attentional Gaussian with shape
            (n_Gaussians,1) [default: None, each Gaussian weighted equally]
        update_mu : boolean, optional
            True/False update mu with Gaussian multiplication [default: True]
        update_sigma : boolean, optional
            True/False update sigma with Gaussian multiplication [default: True]

        Returns
        -------
        mu : torch.Tensor
            updated mu values for each RF with shape (n_kernels,2)
        sigma : torch.Tensor
            updated sigma values for each RF with shape (n_kernels,1)

        See Also
        --------
        lattice.multiply_gaussians

        Notes
        -----
        If either attention_mu or attention_sigma is None, None is returned. This
        function returns mu/sigma values but does not update the class values.
        Use self.update_rfs(mu=mu, sigma=sigma) to update class mu/sigma values.
        """
        if attention_mu is None or attention_sigma is None:
            return None, None
        # multiply attentional gaussian with each RF gaussian
        mu, sigma = lattice.multiply_gaussians(self.mu, self.sigma,
                                               attention_mu, attention_sigma,
                                               weight)
        if kwargs.get('update_mu') is False:
            self.tmp_mu = self.mu
        else:
            self.tmp_mu = mu
        if kwargs.get('update_sigma') is False:
            self.tmp_sigma = self.sigma
        else:
            self.tmp_sigma = sigma
        return self.tmp_mu, self.tmp_sigma

    def weight_output(self, output, RF_weights=None):
        """
        Weight output

        Paramaters
        ----------
        output : torch.Tensor
            pooled outputs with shape (batch, ch, n_kernels, h, w)
        RF_weights : torch.Tensor
            weights with shape (1, 1, n_kernels, 1, 1) or other compatible shape

        Returns
        -------
        output : torch.Tensor
            weighted pooled outputs with shape (batch, ch, n_kernels, h, w) by
            pointwise multiplying output and weights

        Notes
        -----
        If RF_weights is None, output is returned as is.
        """
        if RF_weights is None:
            return output
        return torch.mul(output, RF_weights)

    def vectorize_output(self, output, vectorize=None):
        """
        Vectorize output to shape (batch, ch, n_kernels)

        Parameters
        ----------
        output : torch.Tensor
            pooled outputs with shape (batch, ch, n_kernels, h, w)
        vectorize : boolean
            True/False vectorize outputs

        Returns
        -------
        output : torch.Tensor
            vectorized pooled outputs with shape (batch, ch, n_kernels) by maxing
            across last two dimensions

        Notes
        -----
        If vectorize is not True, output is returned as is.
        """
        if not vectorize:
            return output
        return torch.max(torch.flatten(output, -2), -1)[0]

    def init_RBM(self, n_Gaussians, n_channels, training=False, lr=1e-5,
                 **kwargs):
        """
        Initialize RBM used to learn spatial associations (modeled as Gaussians)
        among RF outputs to update RF locations/sizes using Gaussian
        multiplication (see `apply_attentional_field`).

        Parameters
        ----------
        n_Gaussians : int
            number of attentional Gaussians to learn (i.e., number of hidden
            units in RBM)
        n_channels : int
            number of channels in input (i.e., input vector to RBM is modeled as
            Binomial distributions with n=n_channels)
        training : boolean
            True/False allow training during forward pass [default: False]
        lr : float
            learning rate used during training [default: 1e-5]
        **kwargs : **dict
            keyword arguments for `modules.RBM.train`

        Returns
        -------
        None

        See Also
        --------
        apply_attentional_field
        train_RBM
        modules.RBM.train

        Notes
        -----
        This function sets the following attributes: `rbm`, `training`,
        `optimizer`, and `train_kwargs`.
        """
        assert self.mu is not None
        n_kernels = self.mu.shape[0]
        self.rbm = RBM(hidden=torch.nn.Linear(n_kernels, n_Gaussians),
                       activation=torch.nn.Sigmoid(),
                       sample=lambda x: torch.distributions.Bernoulli(x).sample(),
                       vis_activation=lambda x: torch.sigmoid(x) * n_channels,
                       vis_sample=lambda x: torch.round(x + torch.randn_like(x)))
        # set training
        self.training = training
        # set optimizer
        self.optimizer = torch.optim.SGD(self.rbm.parameters(), lr=lr)
        # set train_kwargs
        self.train_kwargs = kwargs

    def train_RBM(self, model, pool_layer_id, pool_module_name, n_epochs,
                  trainloader, optimizer=None, monitor=100, **kwargs):
        """
        Train RBM within context of given model by passing data through to
        pooling module, obtaining vectorized RF outputs as a Binomial vector,
        and passing this vector with shape (batch, n_kernels) to `self.rbm`.

        Parameters
        ----------
        model : rf_pool.models
            model containing current pooling module
        pool_layer_id : str
            layer_id in which current pooling module resides
        pool_module_name : str
            name of pooling module within
            `model.layers[pool_layer_id].forward_layer`
        n_epochs : int
            number of epochs to train (full passes through trainloader)
        trainloader : torch.utils.data.DataLoader
            dataloader containing (data,label) pairs of data used to train RBM
        optimizer : torch.optim or None
            optimizer used to update `self.rbm.parameters` during training
            [default: None, uses `self.optimizer`]
        monitor : int
            number of batches between each monitoring step
            [default: 100]
        **kwargs : **dict
            keyword arguments passed to `self.rbm.train` (see `modules.RBM.train`)
            additionally, display functions (e.g., `show_lattice`) from pooling
            class or `modules.RBM` class (see `rf_pool.utils.functions.kwarg_fn`)
            [default: `self.train_kwargs`]

        Returns
        -------
        loss_history : list
            list of loss values at each monitoring step
            (i.e., len(loss_history) == (n_epochs * len(trainloader) / monitor))

        See Also
        --------
        init_RBM
        modules.RBM.train
        utils.functions.kwarg_fn
        """
        assert hasattr(self, 'rbm')
        # set optimizer
        if optimizer is None:
            optimizer = self.get('optimizer')
        # set train_kwargs, pool_kwargs
        train_kwargs = self.get(train_kwargs={}).copy()
        train_kwargs.update(kwargs)
        pool_kwargs = {'training': True, 'optimizer': optimizer,
                       'train_kwargs': train_kwargs}
        if pool_module_name is None:
            layer_kwargs = {pool_layer_id: pool_kwargs}
        else:
            layer_kwargs = {pool_layer_id: {'output_module': pool_module_name,
                                            pool_module_name: pool_kwargs}}
        # train for n_epochs
        loss_history = []
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
        n_batches = len(trainloader)
        running_loss = 0.
        i = 0
        for epoch in range(int(np.ceil(n_epochs))):
            for data in trainloader:
                # check if more than requested epochs
                if (i+1) > (n_epochs * n_batches):
                    return loss_history
                # pass inputs through model to pool module with training=True
                model.apply(data[0], output_layer=pool_layer_id, **layer_kwargs)
                # update running_loss
                running_loss += self.loss_history.pop(-1)
                # monitor
                i += 1
                if i % monitor == 0:
                    # display loss
                    clear_output(wait=True)
                    display('[%g%%] loss: %.3f' % (i % n_batches/n_batches*100.,
                                                   running_loss / monitor))
                    # append loss and show history
                    loss_history.append(running_loss / monitor)
                    plt.plot(loss_history)
                    running_loss = 0.
                    # show weights, etc.
                    functions.kwarg_fn([self, self.rbm], **kwargs)
                    plt.show()
        return loss_history

    def get_binomial_input(self, u, **kwargs):
        """
        Get Binomial vector input to RBM from RF outputs after pooling across `u`.

        Parameters
        ----------
        u : torch.Tensor
            input to pooling module
        **kwargs : **dict
            keyword arguments passed to `Pool.forward`

        Returns
        -------
        binomial_input : torch.Tensor
            Binomial input vector for RBM, in which each value is represented as
            a sampled Binomial distribution with `n=u.shape[1]` and
            `p=torch.round(torch.sum(norm_output, 1))`, where `norm_output` is
            the vectorized output from the pooling layer normalized by the sum
            across the RFs.

        See Also
        --------
        init_RBM
        train_RBM

        Notes
        -----
        The number of channels in `u` (i.e., `u.shape[1]`) should match the
        number of channels set in the RBM (i.e., `n_channels` used in `init_RBM`).
        """
        # get output as Binomial vector with shape (batch, n_kernels)
        options = kwargs.copy()
        options.update({'vectorize': True, 'get_binomial_input': True})
        with torch.no_grad():
            output = Pool.forward(self, u, **options)
            output = output - torch.min(output, -1, keepdim=True)[0]
            sum_output = torch.sum(output, -1, keepdim=True) + 1e-6
            return torch.round(torch.sum(torch.div(output, sum_output), 1))

    def get_attentional_field(self, input=None):
        """
        Get attentional field from `self.rbm.hidden_weight`

        Parameters
        ----------
        input : torch.Tensor
            input to `self.rbm` with shape (batch, n_kernels)
            [default: None, will return weight=None]

        Returns
        -------
        attention_mu : torch.Tensor
            x-y center locations of attentional Gaussians with shape
            (n_kernels, 2)
        attention_sigma : torch.Tensor
            size of attentional Gaussians with shape (n_kernels, 1)
        weight : torch.Tensor
            weight associated with each attentional Gaussian with shape
            (n_kernels, 1). kwarg to be passed to `lattice.multiply_gaussians`.

        See Also
        --------
        apply_attentional_field
        lattice.multiply_gaussians

        Notes
        -----
        If weight is None, each attentional Gaussian is weighted equally in the
        Gaussian multiplication.
        """
        assert hasattr(self, 'rbm')
        # estimate mu, sigma for each attentional Gaussian
        mu = self.mu.detach()
        rbm_weight = self.rbm.hidden_weight.detach()
        attention_mu, attention_sigma = lattice.estimate_mu_sigma(rbm_weight, mu)
        # set weight based on hidden unit activity
        if input is None:
            return attention_mu, attention_sigma, None
        weight = self.rbm.apply(input, output_module='activation').detach()
        # if batch > 1, return normalized sum across batch
        weight = torch.sum(weight, 0)
        weight = torch.div(weight, torch.sum(weight, -1, keepdim=True))
        return attention_mu, attention_sigma, weight

    def show_attentional_field(self, input=None):
        """
        Show Attentional Gaussians

        Parameters
        ----------
        input : torch.Tensor
            input used to obtain weights for each attentional Gaussian
            [default: None, each Gaussian shown with same weight]

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing plotted RF lattice

        See Also
        --------
        get_attentional_field
        visualize.scatter_rfs

        Notes
        -----
        If attentional Gaussians are large, they may not be visible in scatter
        plot. Use `get_attentional_field` to obtain mu and sigma values.
        """
        attention_mu, attention_sigma, weight = self.get_attentional_field(input)
        return visualize.scatter_rfs(attention_mu, attention_sigma, self.img_shape,
                                     linewidths=[w for w in weight or [1.]])

    def apply_RBM(self, u, **kwargs):
        """
        Apply/train RBM to get attention_mu, attention_sigma, and weight
        for `apply_attentional_field` during forward pass.

        Parameters
        ----------
        u : torch.Tensor
            input to pooling module
        training : boolean or None
            True/False train RBM on each forward pass
            [default: None, uses `self.training`]
        optimizer : torch.optim or None
            optimizer used to update `self.rbm.parameters` during training
            [default: None, uses `self.optimizer`]
        train_kwargs : dict
            dictionary of keyword arguments passed to `self.rbm.train`
            [default: None, uses `self.train_kwargs`]
        **kwargs : **dict
            keyword arguments passed to `forward`

        Returns
        -------
        attention_kwargs : OrderedDict
            dictionary of `apply_attentional_field` inputs (`attention_mu`,
            `attention_sigma`, `weight`) obtained via `get_attentional_field`.

        See Also
        --------
        init_RBM
        train_RBM
        get_attentional_field
        apply_attentional_field

        Notes
        -----
        If `attention_mu`, `attention_sigma` or `weight` are in kwargs,
        {} is returned and no training occurs.
        """
        if not hasattr(self, 'rbm') or kwargs.get('get_binomial_input') is True:
            return {}
        # parse kwargs for training, optimizer, train_kwargs
        training = kwargs.get('training')
        if training is None:
            training = self.get('training')
        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            optimizer = self.get('optimizer')
        train_kwargs = self.get(train_kwargs={}).copy()
        train_kwargs.update(kwargs.get('train_kwargs') or {})
        # get Binomial vector input to RBM and for weighting attentional field
        output = self.get_binomial_input(u, **kwargs)
        # train RBM
        if training:
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
            self.loss_history.append(self.rbm.train(output, optimizer=optimizer,
                                                    **train_kwargs))
        # get attentional field
        keys = ['attention_mu', 'attention_sigma', 'weight']
        values = self.get_attentional_field(output)
        return OrderedDict([(k, v) for k, v in zip(keys, values)])

    def forward(self, u, **kwargs):
        """
        Apply forward pass through pooling layer

        Parameters
        ----------
        u : torch.Tensor
            input to pooling layer with shape (batch, ch, h, w)

        Optional kwargs (default to class attributes)

        Returns
        -------
        output : torch.Tensor
            pooled outputs with shape (batch, ch, h, w), (batch, ch, n_kernels, h, w)
            if `retain_shape=True`, or (batch, ch, n_kernels) if `vectorize=True`.

        Notes
        -----
        The following class methods are called (in order) if associated kwargs
        are in given during forward call (or set at layer initialization):

        adaptive_update: `adaptive`;
        shift_mu_sigma: `delta_mu`, `delta_sigma`, `fn`;
        apply_RBM: `training`, `optimizer`, `train_kwargs`;
        apply_attentional_field: `attention_mu`, `attention_sigma`, `weight`,
            `update_mu`, `update_sigma`;
        weight_output: `RF_weights`;
        vectorize_output: `vectorize`.

        See Also
        --------
        pool.apply
        adaptive_update
        shift_mu_sigma
        apply_RBM
        apply_attentional_field
        weight_output
        vectorize_output
        """
        # get options, update from kwargs
        options = self.get(*self.option_keys)
        options.update(kwargs)
        # update mu, sigma based on image shape
        mu, sigma = self.adaptive_update(u.shape[-2:], options.get('adaptive'))
        self._update_rfs(mu, sigma)
        # update rfs, mu, sigma
        mu, sigma = self.shift_mu_sigma(**options)
        self._update_rfs(mu, sigma)
        # apply RBM
        options.update(self.apply_RBM(u, **options))
        # apply attentional field
        mu, sigma = self.apply_attentional_field(**options)
        self._update_rfs(mu, sigma)
        # update retain_shape as True
        kwargs.update({'retain_shape': True})
        # apply pooling layer
        output = self.pool(u, **kwargs)
        # weight outputs
        output = self.weight_output(output, options.get('RF_weights'))
        # vectorize outputs
        if options.get('vectorize'):
            return self.vectorize_output(output, options.get('vectorize'))
        # max across RFs if not retain_shape
        if not options.get('retain_shape'):
            output = torch.max(output, -3)[0]
        return output

    # set forward docstring using apply docstring parameters
    forward.__doc__ = functions.update_doc(forward.__doc__, 'Returns', [-1],
                                           [__paramsdoc__], sub=['\n    ','\n'])

class RF_Pool(Pool):
    __doc__ = functions.update_doc(Pool.__doc__,
                                   sub=['Base class.+\n',
                                        'Receptive field pooling layer\n'])
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
        # init mu, sigma
        mu, sigma = self.init_mu_sigma(lattice.init_uniform_lattice, img_shape,
                                       n_kernels=n_kernels, spacing=spacing,
                                       sigma_init=sigma_init, offset=offset,
                                       rotate=rotate)
        super(RF_Uniform, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         init_fn=self.init_fn,
                                         init_kwargs=self.init_kwargs, **kwargs)

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
        # init mu, sigma
        mu, sigma = self.init_mu_sigma(lattice.init_hexagon_lattice, img_shape,
                                       n_kernels=n_kernels, spacing=spacing,
                                       sigma_init=sigma_init, offset=offset,
                                       rotate=rotate)
        super(RF_Hexagon, self).__init__(mu, sigma, img_shape, lattice_fn,
                                         init_fn=self.init_fn,
                                         init_kwargs=self.init_kwargs, **kwargs)

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
        if kwargs.get('mu') is None:
            mu = torch.rand(n_kernels, 2) * torch.tensor(img_shape).float()
        else:
            mu = kwargs.pop('mu')
        if kwargs.get('sigma') is None:
            sigma = torch.rand(n_kernels, 1) * np.minimum(*img_shape) / 2.
        else:
            sigma = kwargs.pop('sigma')
        super(RF_Random, self).__init__(mu, sigma, img_shape, lattice_fn,
                                        **kwargs)

class RF_Squeeze(Pool):
    __doc__ = functions.update_doc(Pool.__doc__,
                                   sub=['Base class.+\n',
                                        'Receptive field pooling layer ' \
                                        'with output image size squeezed ' \
                                        'to extent of RF lattice\n'])
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
        center = torch.max(torch.tensor(u.shape[-2:]) // 2 - 1,
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
    ref_axis : float
        reference axis angle (clockwise, in radians) from which polar angle is
        calculated [default: 0.]
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

    See Also
    --------
    pool.apply
    lattice.init_foveated_lattice
    lattice.cortical_xy

    Notes
    -----
    Since image origin is top-left, positive reference axis angles are clockwise
    (which is opposite of rotation angle) and positive offsets are down/right.
    """
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, img_shape=None, scale=None, n_rings=None,
                 ref_axis=0., cortical_mu=None, cortical_sigma=None,
                 cortical_kernel_fn=lattice.exp_kernel_2d,
                 lattice_fn=lattice.mask_kernel_lattice, **kwargs):
        # set variables
        self.scale = torch.as_tensor(scale)
        self.n_rings = n_rings
        self.ref_axis = ref_axis
        self.cortical_mu = cortical_mu
        self.cortical_sigma = cortical_sigma
        self.cortical_kernel_fn = cortical_kernel_fn
        # init mu, sigma
        mu, sigma = self.init_mu_sigma(lattice.init_foveated_lattice, img_shape,
                                       scale=scale, n_rings=n_rings, **kwargs)
        super(RF_Foveated, self).__init__(mu, sigma, img_shape, lattice_fn,
                                          init_fn=self.init_fn,
                                          init_kwargs=self.init_kwargs, **kwargs)
        # set cortical weight vars
        self.get_cortical_vars(self.init_kwargs)

    def get_cortical_vars(self, lattice_kwargs, img_shape=None):
        if lattice_kwargs.get('n_rf') is None:
            n_rf = np.pi / self.scale
        else:
            n_rf = lattice_kwargs.get('n_rf')
        if img_shape is None:
            img_shape = self.img_shape
        self.rf_angle = 2. * np.pi * torch.linspace(0., 1., int(n_rf))[1]
        self.offset = torch.tensor(lattice_kwargs.get('offset'))
        self.offset += (torch.tensor(img_shape, dtype=torch.float) // 2)

    def get_cortical_mu(self, mu=None, beta=0.):
        if mu is None:
            mu = self.mu
        return lattice.cortical_xy(mu - self.offset, self.scale,
                                   self.rf_angle, beta, self.ref_axis)

    def get_cortical_weights(self, cortical_mu=None, cortical_sigma=None, beta=0.):
        if cortical_mu is None:
            cortical_mu = self.cortical_mu
        if cortical_sigma is None:
            cortical_sigma = self.cortical_sigma
        if cortical_mu is None or cortical_sigma is None:
            return None
        cortical_RFs = self.get_cortical_mu(beta=beta)
        weights = self.cortical_kernel_fn(cortical_mu, cortical_sigma,
                                          cortical_RFs.t().reshape(1,2,-1,1))
        return weights.reshape(1, 1, -1, 1, 1)

    def show_cortical_weights(self, cortical_mu=None, cortical_sigma=None, beta=0.):
        # get weights
        weights = self.get_cortical_weights(cortical_mu, cortical_sigma, beta)
        if weights is None:
            return None
        # get cortical location of RFs
        cortical_RFs = self.get_cortical_mu(beta=beta)
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
        # get cortical_kwargs from kwargs
        if kwargs.get('RF_weights') is None:
            cortical_kwargs = functions.pop_attributes(kwargs,
                                                       ['cortical_mu',
                                                        'cortical_sigma',
                                                        'beta'],
                                                       ignore_keys=True)
            # weight output by cortical locations relative to cortical_mu
            self.get_cortical_vars(self.init_kwargs, img_shape=u.shape[-2:])
            RF_weights = self.get_cortical_weights(**cortical_kwargs)
            kwargs.update({'RF_weights': RF_weights})
        # apply forward function
        return Pool.forward(self, u, **kwargs)

class RBM_Attention(Pool):
    """
    Receptive field pooling layer with Restricted-Boltzmann Machine
    attentional field

    Parameters
    ----------
    n_Gaussians : int
        number of attentional Gaussians to be learned (also number of hidden
        units in RBM)
    n_channels : int
        number of channels from input layer
    mu : torch.Tensor
        receptive field centers (in x-y coordinate space) with shape
        (n_kernels, 2)
    sigma : torch.Tensor
        receptive field standard deviations with shape
        (n_kernels, 1)
    img_shape : tuple
        receptive field/detection layer shape
    lattice_fn : utils.lattice function, optional
        function used to update receptive field kernels given delta_mu and
        delta_sigma [default: lattice.mask_kernel_lattice]
    training : boolean
        True/False continuously train RBM with data from each forward pass
        through pooling layer. [default: False]
    lr : float
        learning rate to be used during training of RBM [default: 1e-5]
    train_kwargs : dict
        keyword arguments passed to `modules.RBM.train` during training
        (if `self.training=True`). See `modules.RBM.train`.
        [default: {}]

    See Also
    --------
    pool.apply
    modules.RBM
    modules.RBM.train

    Notes
    -----
    The RBM should be trained using `train_RBM` with the same training data to
    be used for the rest of the model. Once trained, attentional Gaussians are
    obtained via the weights of RBM. The attention_mu is the weighted average
    of each mu, and attention_sigma is computed from the full width at half
    maximum of the weights. The attentional update is then performed via
    Gaussian multiplication between the attentional Gaussians and RF Gaussians.
    The update of each attentional Gaussian is then weighted by its hidden unit
    activity (i.e., the attentional Gaussian that best matches the spatial
    distribution of the features in the input is weighted most).
    """
    __doc__ = functions.update_doc(__doc__, 'See Also', [-1],
                                   [['',Pool.__methodsdoc__]])
    def __init__(self, n_Gaussians, n_channels, mu, sigma, img_shape,
                 lattice_fn=lattice.mask_kernel_lattice, training=False,
                 lr=1e-5, train_kwargs={}, **kwargs):
        super(RBM_Attention, self).__init__(mu, sigma, img_shape, lattice_fn,
                                            **kwargs)
        # initialize RBM and optimizer
        self.init_RBM(n_Gaussians, n_channels, training, lr, **train_kwargs)

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

if __name__ == '__main__':
    import doctest
    doctest.testmod()
