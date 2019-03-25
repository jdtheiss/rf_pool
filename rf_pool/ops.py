import numpy as np
from six.moves import xrange
import torch
import torch.nn.functional as F
from torch.distributions import Multinomial
from utils import lattice, functions

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
    off_pt = torch.zeros(u.shape[:-1] + (1,), dtype=u.dtype)
    events = torch.cat([u, off_pt], -1)
    if type(mask) is torch.Tensor:
        mask = torch.cat([mask, 1. + off_pt], -1)
        mask = torch.flatten(mask, 0, -2)
    probs = local_softmax(torch.flatten(events, 0, -2), -1, mask)
    samples = Multinomial(probs=probs).sample()
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs[:,:-1], out_shape)
    h_sample = torch.reshape(samples[:,:-1], out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.zeros_like(u)
    p_mean = torch.add(p_mean, torch.reshape(1. - probs[:,-1], off_pt.shape))
    p_mean = torch.reshape(p_mean, out_shape)
    p_mean = torch.mul(p_mean, h_sample)
    return h_mean, p_mean

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
    return h_mean, p_mean

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
                 torch.sum(torch.pow(u, n), dim=-1, keepdim=True))
        p_mean is set to the value of h_mean indexed at the stochastically
        selected max unit

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
    return h_mean, p_mean

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
        p_mean is set to the value of h_mean indexed at the maximum unit in h_mean
    """
    # apply mask to u
    if type(mask) is torch.Tensor:
        u = torch.mul(u, mask.type(u.dtype))
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(u, out_shape)
    h_sample = torch.reshape(max_index(u), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean

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
        h_mean is set to the input, rf_u, divided by the total number units
        in the receptive field (rf_u.shape[-1])
        p_mean is set to the value of h_mean indexed at the maximum unit in h_mean
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
    probs = torch.div(u, n_units)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(max_index(probs), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    return h_mean, p_mean

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
        indexed at the maximum unit
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
    return h_mean, p_mean

def rf_pool(u, t=None, rfs=None, mu_mask=None, pool_type='max', kernel_size=2, **kwargs):
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
    mu_mask : torch.Tensor or None
        mask with center locations for receptive fields with shape (n_kernels, h, w)
        [default: None, outputs are indexed at position of maximum unit in each
        receptive field]
    pool_type : string
        type of pooling ('prob', 'stochastic', 'div_norm', 'max', 'average', 'sum')
        [default: 'max']
    kernel_size : int or tuple
        size of blocks in hidden layer connected to pooling units
        [default: 2]
    return_indices : bool
        boolean whether to return indices of max-pooling (for kernel_size > 1)
        [default: False]
    **kwargs : dict
        extra arguments passed to pooling function indicated by pool_type

    Returns
    -------
    h_mean : torch.Tensor
        hidden layer mean-field estimates with shape (batch_size, ch, h, w)
    p_mean : torch.Tensor or tuple
        pooling layer mean-field estimates with shape
        (batch_size, ch, h//kernel_size, w//kernel_size)
        if return_indices is True and kernel_size > 1, p_mean is tuple:
        (p_mean, p_mean_indices)

    Examples
    --------
    # Performs probabilistic max-pooling across 4x4 regions tiling hidden
    # layer with top-down input
    >>> from utils import lattice
    >>> u = torch.rand(1,10,8,8)
    >>> t = torch.rand(1,10,4,4)
    >>> mu, sigma = lattice.init_uniform_lattice((4,4), 2, 3, 2.)
    >>> kernels = lattice.gaussian_kernel_lattice(mu, sigma, (8,8))
    >>> h_mean, h_sample, p_mean, p_sample = rf_pool(u, t, kernels, 'sum', (2,2))

    Notes
    -----
    pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization with sigma=0.5 (Heeger, 1992),
    'max_pool' performs max pooling over the units in the receptive field,
    'average' divides units by total number of units in the receptive field,
    'sum' returns sum over units in receptive field (especially for Gaussians).

    When kernel_size != 1, p_mean and p_sample result from a max operation
    across (n,n) blocks according to either probabilistic max-pooling or
    stochastic max-pooling.

    See Also
    --------
    layers.RF_Pool : layer implementation of rf_pool function
    """

    # get bottom-up shape
    batch_size, ch, u_h, u_w = u.shape
    if type(kernel_size) is tuple:
        b_w, b_h = kernel_size
    else:
        b_w = b_h = kernel_size

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
        # elemwise multiply u with rf_kernels (batch_size, ch, rf, u_h, u_w)
        rf_u = torch.mul(u.unsqueeze(2), rfs)

        # apply pooling function
        if pool_fn:
            h_mean, p_mean = pool_fn(rf_u, rf_u.shape, **kwargs)
        else:
            h_mean = rf_u
            p_mean = rf_u

        # apply mu_mask
        if mu_mask is not None:
            assert mu_mask.shape[-3] == p_mean.shape[-3]
            p_mean = torch.max(p_mean.flatten(-2), -1, keepdim=True)[0].unsqueeze(-1)
            p_mean = torch.mul(mu_mask, p_mean)
            p_mean = torch.max(p_mean, -3)[0]
        # get max index in each RF
        elif pool_fn is None:
            h_sample = max_index(h_mean.flatten(-2)).reshape(h_mean.shape)
            h_sample = torch.max(h_sample, -3)[0]
            h_mean = torch.max(h_mean, -3)[0]
            p_mean = torch.mul(h_sample, h_mean)
        else: # max across RFs
            h_mean = torch.max(h_mean, -3)[0]
            p_mean = torch.max(p_mean, -3)[0]

    # pooling across blocks
    elif rfs is None:
        # init h_mean
        h_mean = torch.zeros_like(u)
        # pool across blocks
        assert pool_fn is not None, ('pool_type cannot be None if rfs is None')
        h_mean_b, p_mean_b = pool_fn(b, b.shape, **kwargs)
        for r in xrange(b_w):
            for c in xrange(b_h):
                h_mean[:, :, r::b_w, c::b_h] = h_mean_b[:,:,:,:,(r*b_w) + c]
        p_mean = h_mean

    else:
        raise Exception('rfs type not understood')

    # max pool across blocks
    if (b_w > 1 or b_h > 1) and return_indices:
        p_mean = F.max_pool2d_with_indices(p_mean, kernel_size)
    elif (b_w > 1 or b_h > 1):
        p_mean = F.max_pool2d(p_mean, kernel_size)
    
    return h_mean, p_mean

if __name__ == '__main__':
    import doctest
    doctest.testmod()
