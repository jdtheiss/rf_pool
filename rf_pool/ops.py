import numpy as np
from six.moves import xrange
import torch
from torch.distributions import Multinomial
import torch.nn.functional as F

from .utils import lattice, functions

def max_index(u):
    u_s = int(np.prod(u.shape[:-1]))
    m = torch.zeros((u_s, u.shape[-1]), dtype=u.dtype)
    m[np.arange(u_s), np.argmax(u.detach(), -1).flatten()] = 1.
    return m.reshape(u.shape)

def local_softmax(u, dim=-1, mask=None):
    """
    Apply softmax across dim with mask

    Parameters
    ----------
    u : torch.Tensor
        input to softmax
    dim : int, optional
        dimension to across which to softmax [default: -1]
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        to include in softmax [default: None]

    Returns
    -------
    s : torch.Tensor
        softmaxed output

    Notes
    -----
    Mask is applied by setting 1s to 0 and 0s to -inf before adding to u. This
    allows for indices that are not included in mask to output as 0 and not be
    included in sum of exponentials.
    """
    if type(mask) is torch.Tensor:
        mask = torch.mul(torch.exp(np.inf * (1. - 2 * mask)), -1).type(u.dtype)
        u = torch.add(u, mask)
    return torch.softmax(u, dim)

def prob_max_pool(u, out_shape, mask=None):
    """
    Probabilistic max-pooling across units in a receptive field (dim=-1)

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Probabilistic max-pooling considers each receptive field to be a
    multinomial unit in which only one unit can be on or all units can be off.
        h_mean is a softmax across all units in the receptive field and a unit
        representing all units being "off" (not returned in h_mean)
        p_mean is set to 1 - probability of all units being "off" indexed at a
        unit sampled from a multinomial distribution across the units in the
        receptive field in which at most one unit can be "on"

    References
    ----------
    Lee, H., Grosse, R., Ranganath, R., & Ng, A. Y. (2009, June). Convolutional
    deep belief networks for scalable unsupervised learning of hierarchical
    representations. In Proceedings of the 26th annual international conference
    on machine learning (pp. 609-616). ACM.
    """

    # get probabilities for each unit being on or all off, sample
    off_p = torch.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    events = torch.cat([u, off_p], -1)
    if type(mask) is torch.Tensor:
        off_p_mask = torch.ones(mask.shape[:-1] + (1,), dtype=mask.dtype)
        mask = torch.cat([mask, off_p_mask], -1)
    probs = local_softmax(events, -1, mask)
    probs = torch.flatten(probs, 0, -2)
    samples = Multinomial(probs=probs).sample()
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs[:,:-1], out_shape)
    h_sample = torch.reshape(samples[:,:-1], out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.zeros_like(u)
    p_mean = torch.add(p_mean, torch.reshape(1. - probs[:,-1], off_p.shape))
    p_mean = torch.reshape(p_mean, out_shape)
    p_mean = torch.mul(p_mean, h_sample)
    return h_mean, p_mean, h_sample

def stochastic_max_pool(u, out_shape, mask=None):
    """
    Stochastic max-pooling across units in a receptive field (dim=-1)

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Stochastic max-pooling considers each receptive field to be a
    multinomial unit in which only one unit can be on.
        h_mean is set to the input, u
        p_mean is set to u indexed at a unit sampled from a multinomial
        distribution across the units in the receptive field in which exactly
        one unit can be "on"

    References
    ----------
    Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization
    of deep convolutional neural networks. arXiv preprint arXiv:1301.3557.
    """

    # get probabilities for each unit being on
    probs = local_softmax(u, -1, mask)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(u, out_shape)
    h_sample = torch.reshape(Multinomial(probs=probs).sample(), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean, h_sample

def div_norm_pool(u, out_shape, mask=None, n=2., s=0.5):
    """
    Divisive normalization across units in all receptive fields (dim=[-2,-1])

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]
    n : float, optional
        exponent used in divisive normalization (see Notes)
        [default: 2.]
    s : float, optional
        constant used in divisive normalization (see Notes)
        [default: 0.5]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Divisive normalization raises the input to the power of n, and normalizes
    each unit with a constant, s, added in the denominator:
        h_mean = torch.pow(u, n)/torch.add(torch.pow(s, n),
                 torch.sum(torch.pow(u, n), dim=[-2,-1], keepdim=True))
        p_mean is set to the value of h_mean indexed at the maximum unit in h_mean

    References
    ----------
    Heeger, D. J. (1992). Normalization of cell responses in cat striate
    cortex. Visual neuroscience, 9(2), 181-197.
    """

    # apply mask to u
    if type(mask) is torch.Tensor:
        u = torch.mul(u, mask.type(u.dtype))
    # raise rf_u, sigma to nth power
    if n != 1:
        u_n = torch.pow(u, n)
        s_n = torch.pow(torch.as_tensor(s, dtype=u.dtype), n)
    else:
        u_n = u
        s_n = s
    probs = torch.div(u_n, s_n + torch.sum(u_n, dim=[-2,-1], keepdim=True))
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(max_index(probs), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean, h_sample

def max_pool(u, out_shape, mask=None):
    """
    Max pooling across units in a receptive field (dim=-1)

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Max pooling returns the following mean-field estimates and samples
    for the hidden and pooling layers:
        h_mean is set to the input, u
        p_mean is set to the value of h_mean indexed at the maximum unit in u
    """
    # apply mask to u
    if type(mask) is torch.Tensor:
        u = torch.mul(u, mask.type(u.dtype))
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(u, out_shape)
    h_sample = torch.reshape(max_index(u), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean, h_sample

def average_pool(u, out_shape, mask=None):
    """
    Average pooling across units in a receptive field (dim=-1)

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Average pooling returns the following mean-field estimates and samples
    for the hidden and pooling layers:
        h_mean is set to the sum across the input, u, divided by the total number
        units in the receptive field (u.shape[-1])
        p_mean is set to the value of h_mean indexed at the maximum unit in u
    """

    # apply mask to u
    if type(mask) is torch.Tensor:
        u = torch.mul(u, mask.type(u.dtype))
    # get count of units in last dim
    if type(mask) is torch.Tensor:
        n_units = torch.sum(mask, -1, keepdim=True)
    else:
        n_units = torch.as_tensor(u.shape[-1], dtype=u.dtype)
    # divide activity by number of units
    probs = torch.zeros_like(u)
    probs = torch.add(probs, torch.div(torch.sum(u, -1, keepdim=True), n_units))
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(max_index(u), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean, h_sample

def sum_pool(u, out_shape, mask=None):
    """
    Sum pooling across units in a receptive field (dim=-1)

    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    mask : torch.Tensor or None, optional
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        included in the receptive field [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)

    Notes
    -----
    Sum pooling returns the following mean-field estimates and samples for
    the hidden and pooling layers:
        h_mean is set to the input, u
        p_mean is set to the sum across across units in the receptive field
        indexed at the maximum unit in u
    """

    # apply mask to u
    if type(mask) is torch.Tensor:
        u = torch.mul(u, mask.type(u.dtype))
    # get sum across last dim in u
    sum_val = torch.sum(u, -1, keepdim=True)
    sum_val = torch.add(torch.zeros_like(u), sum_val)
    sum_val = torch.reshape(sum_val, out_shape)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(u, out_shape)
    h_sample = torch.reshape(max_index(u), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(sum_val, h_sample)
    return h_mean, p_mean, h_sample

def rf_pool(u, t=None, rfs=None, pool_type=None, kernel_size=2, mask_thr=1e-6,
            return_indices=False, retain_shape=False, **kwargs):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        bottom-up input to pooling layer with shape (batch_size, ch, h, w)
    t : torch.Tensor or None
        top-down input to pooling layer with shape
        (batch_size, ch, h//kernel_size, w//kernel_size) [default: None]
    rfs : torch.Tensor or None
        kernels containing receptive fields to apply pooling over with shape
        (n_kernels, h, w) (see lattice_utils)
        kernels are element-wise multiplied with (u+t) prior to pooling
        [default: None, applies pooling over blocks]
    pool_type : string
        type of pooling ('prob','stochastic','div_norm','max','average','sum')
        [default: 'max']
    kernel_size : int or tuple
        size of subsampling blocks in hidden layer connected to pooling units
        [default: 2]
    mask_thr : float
        threshold used to binarize rfs as mask input for pooling operations
        (overridden if 'mask' in kwargs)
        [default: 1e-6]
    return_indices : bool
        boolean whether to return indices of max-pooling (for kernel_size > 1)
        [default: False]
    retain_shape : bool
        boolean whether to retain the shape of output after multiplying u with
        rfs (i.e. shape=(batch_size, ch, n_rfs, u_h, u_w))
        [default: False]
    **kwargs : dict
        extra arguments passed to pooling function indicated by pool_type

    Returns
    -------
    p_mean : torch.Tensor or tuple
        pooling layer mean-field estimates with shape
        (batch_size, ch, h//kernel_size, w//kernel_size)
        if return_indices is True and kernel_size > 1, p_mean is tuple:
        (p_mean, p_mean_indices)
    h_mean : torch.Tensor
        hidden layer mean-field estimates with shape (batch_size, ch, h, w)
    h_sample : torch.Tensor
        hidden layer samples with shape (batch_size, ch, h, w)

    Examples
    --------
    # Performs probabilistic max-pooling across 4x4 regions tiling hidden
    # layer with top-down input
    >>> from utils import lattice
    >>> u = torch.rand(1,10,8,8)
    >>> t = torch.rand(1,10,4,4)
    >>> mu, sigma = lattice.init_uniform_lattice((4,4), 2, 3, 2.)
    >>> rfs = lattice.gaussian_kernel_lattice(mu, sigma, (8,8))
    >>> p_mean, h_mean, h_sample = rf_pool(u, t, rfs, 'sum', (2,2))

    Notes
    -----
    pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization (Heeger, 1992),
    'max' performs max pooling over the units in the receptive field,
    'average' divides units by total number of units in the receptive field,
    'sum' returns sum over units in receptive field (useful for Gaussian RFs).

    When kernel_size != 1, p_mean results from a max operation across (n,n)
    blocks using torch.nn.functional.max_pool2d. If kernel_size == 1, p_mean
    results from the pointwise multiplication between h_mean and h_sample.

    See Also
    --------
    layers : layer implementations of rf_pool function
    """

    # get bottom-up shape
    batch_size, ch, u_h, u_w = u.shape
    if type(kernel_size) is tuple:
        b_h, b_w = kernel_size
    else:
        b_h = b_w = kernel_size

    top_down = (type(t) is torch.Tensor)
    receptive_fields = (type(rfs) is torch.Tensor)

    # check bottom-up, top-down shapes
    if top_down:
        assert u.shape[:2] == t.shape[:2]
        assert u//kernel_size == t.shape[-2]
        assert u_w//kernel_size == t.shape[-1]

    # add top-down, get blocks
    if top_down:
        u = torch.add(u, functions.repeat(t, (1,1,b_w,b_h)))

    if not receptive_fields:
        # add bottom-up and top-down
        b = []
        for r in xrange(b_w):
            for c in xrange(b_h):
                b.append(u[:,:,r::b_w,c::b_h].unsqueeze(-1))
        b = torch.cat(b, -1)

    # set pool_fn
    if pool_type == 'prob':
        pool_fn = prob_max_pool
    elif pool_type == 'stochastic':
        pool_fn = stochastic_max_pool
    elif pool_type == 'div_norm':
        pool_fn = div_norm_pool
    elif pool_type == 'max':
        pool_fn = max_pool
    elif pool_type == 'average':
        pool_fn = average_pool
    elif pool_type == 'sum':
        pool_fn = sum_pool
    elif pool_type is None:
        pool_fn = None
    else:
        raise Exception('pool_type not understood')

    # pooling across receptive fields
    if receptive_fields:
        # elemwise multiply u with rf_kernels (batch_size, ch, n_rfs, u_h, u_w)
        rf_u = torch.mul(torch.unsqueeze(u, 2), rfs)

        # apply pooling function
        if pool_fn:
            # get mask from kwargs
            rfs_mask = functions.pop_attributes(kwargs, ['mask'])['mask']
            if rfs_mask is None:
                rfs_mask = torch.reshape(rfs, (1,1) + rfs.shape)
                rfs_mask = torch.gt(rfs_mask, mask_thr).float()
                rfs_mask = torch.flatten(rfs_mask, -2)
            # apply pooling operation
            h_mean, p_mean, h_sample = pool_fn(torch.flatten(rf_u, -2), rf_u.shape,
                                               mask=rfs_mask, **kwargs)
        else: # get max index in each RF
            h_mean = rf_u
            h_sample = max_index(torch.flatten(h_mean, -2))
            h_sample = torch.reshape(h_sample, h_mean.shape)
            p_mean = torch.mul(h_sample, h_mean)

        # max across RFs
        if not retain_shape:
            h_mean = torch.max(h_mean, -3)[0]
            p_mean = torch.max(p_mean, -3)[0]
            h_sample = torch.max(h_sample, -3)[0]

    # pooling across blocks
    elif rfs is None:
        # init h_mean, p_mean
        h_mean = torch.zeros_like(u)
        p_mean = torch.zeros_like(u)
        h_sample = torch.zeros_like(u)
        # pool across blocks
        assert pool_fn is not None, ('pool_type cannot be None if rfs is None')
        h_mean_b, p_mean_b, h_sample_b = pool_fn(b, b.shape, **kwargs)
        for r in xrange(b_h):
            for c in xrange(b_w):
                h_mean[:, :, r::b_h, c::b_w] = h_mean_b[:,:,:,:,(r*b_h) + c]
                p_mean[:, :, r::b_h, c::b_w] = p_mean_b[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = h_sample_b[:,:,:,:,(r*b_h) + c]

    else:
        raise Exception('rfs type not understood')

    # max pool across blocks
    if (b_h > 1 or b_w > 1):
        if retain_shape:
            kernel_size = (1, b_h, b_w)
            p_mean = F.max_pool3d(p_mean, kernel_size,
                                  return_indices=return_indices)
        else:
            p_mean = F.max_pool2d(p_mean, kernel_size,
                                  return_indices=return_indices)

    return p_mean, h_mean, h_sample

if __name__ == '__main__':
    import doctest
    doctest.testmod()
