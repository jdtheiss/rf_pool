import numpy as np
import torch
from torch.distributions import Multinomial

def local_softmax(u, dim=-1, mask=None):
    """
    Apply softmax across dim with mask
    
    Parameters
    ----------
    u : torch.Tensor
        input to softmax
    dim : int
        dimension to across which to softmax
    mask : torch.Tensor or None
        mask with shape = u.shape and dtype = u.dtype with 1s indicating units
        to include in softmax

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
        mask = torch.as_tensor(-1. * torch.exp(np.inf * (1. - 2 * mask)),
                               dtype=u.dtype)
        u = torch.add(u, mask)
    return torch.softmax(u, dim)

def prob_max_pool(u, out_shape, mask=None):
    """
    Probabilistic max-pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    h_sample : torch.Tensor
        samples of hidden layer after pooling with shape out_shape (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)
    p_sample : torch.Tensor
        samples of pooling layer with shape out_shape (see Notes)
        
    Notes
    -----
    Probabilistic max-pooling considers each receptive field to be a 
    multinomial unit in which only one unit can be on or all units can be off. 
        h_mean is a softmax across all units in the receptive field and a unit 
        representing all units being "off" (not returned in h_mean)
        h_sample is sampled from a multinomial distribution with at most one 
        unit set to 1
        p_mean is set to the element-wise multiplication of h_mean and h_sample 
        p_sample is set to h_sample

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
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def stochastic_max_pool(u, out_shape, mask=None):
    """
    Stochastic max-pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    h_sample : torch.Tensor
        samples of hidden layer after pooling with shape out_shape (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)
    p_sample : torch.Tensor
        samples of pooling layer with shape out_shape (see Notes)
        
    Notes
    -----
    Stochastic max-pooling considers each receptive field to be a 
    multinomial unit in which only one unit can be on. 
        h_mean is a softmax across all units in the receptive field
        h_sample is sampled from a multinomial distribution with and exactly 
        one unit set to 1
        p_mean is set to the element-wise multiplication of h_mean and h_sample
        p_sample is set to h_sample

    References
    ----------
    Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization 
    of deep convolutional neural networks. arXiv preprint arXiv:1301.3557.
    """

    # get probabilities for each unit being on
    probs = local_softmax(u, -1, mask)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(Multinomial(probs=probs).sample(), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def div_norm_pool(u, out_shape, mask=None, n=2., sigma=0.5):
    """
    Divisive normalization across units in a receptive field (last dim)
    
    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output
    n : float
        exponent used in divisive normalization (see Notes)
        [default: 2.]
    sigma : float
        constant used in divisive normalization (see Notes)
        [default: 0.5]

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    h_sample : torch.Tensor
        samples of hidden layer after pooling with shape out_shape (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)
    p_sample : torch.Tensor
        samples of pooling layer with shape out_shape (see Notes)
        
    Notes
    -----
    Divisive normalization raises the input to the power of n, and normalizes 
    each unit with a constant, sigma, added in the denominator:
        h_mean = torch.pow(rf_u, n)/torch.add(torch.pow(sigma, n),
                 torch.sum(torch.pow(rf_u, n), dim=-1, keepdim=True))
        h_sample is set to 1 indexed at the stochastically selected max unit
        (see stochastic_max_pool)
        p_mean is set to the value of h_mean indexed at the stochastically
        selected max unit
        p_sample is set to h_sample
    
    With n=2 and sigma=0.5, div_norm_pool simulates the average cortical 
    normalization observed emprically (Heeger, 1992).

    References
    ----------
    Heeger, D. J. (1992). Normalization of cell responses in cat striate 
    cortex. Visual neuroscience, 9(2), 181-197.
    """

    # raise rf_u, sigma to nth power
    u_n = torch.pow(u, n)
    sigma_n = torch.pow(torch.as_tensor(sigma, dtype=u.dtype), n)
    probs = torch.div(u_n, sigma_n + torch.sum(u_n, dim=-1, keepdim=True))
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(Multinomial(probs=local_softmax(probs, -1, mask)).sample(), out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def average_pool(u, out_shape, mask=None):
    """
    Average pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    h_sample : torch.Tensor
        samples of hidden layer after pooling with shape out_shape (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)
    p_sample : torch.Tensor
        samples of pooling layer with shape out_shape (see Notes)
    
    Notes
    -----
    Average pooling returns the following mean-field estimates and samples
    for the hidden and pooling layers:
        h_mean is set to the input, rf_u, divided by the total number units
        in the receptive field (rf_u.shape[-1])
        h_sample is set to 1 indexed at the stochastically selected max unit
        (see stochastic_max_pool)
        p_mean is set to the value of h_mean indexed at the stochastically
        selected max unit
        p_sample is set to h_sample
    """

    # get count of units in last dim
    n_units = torch.as_tensor(u.shape[-1], dtype=u.dtype)
    if type(mask) is torch.Tensor:
        n_units = torch.sum(mask, -1, keepdim=True)
    # divide activity by number of units
    probs = torch.div(u, n_units)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(probs, out_shape)
    samples = Multinomial(probs=local_softmax(probs, -1, mask)).sample()
    h_sample = torch.reshape(samples, out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def sum_pool(u, out_shape, mask=None):
    """
    Sum pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    u : torch.Tensor
        receptive field to pool over with n_units = shape[-1]
        or torch.sum(mask, -1)
    out_shape : tuple
        shape of tensor to output

    Returns
    -------
    h_mean : torch.Tensor
        mean-field estimates of hidden layer after pooling with shape out_shape
        (see Notes)
    h_sample : torch.Tensor
        samples of hidden layer after pooling with shape out_shape (see Notes)
    p_mean : torch.Tensor
        mean-field estimate of pooling layer with shape out_shape (see Notes)
    p_sample : torch.Tensor
        samples of pooling layer with shape out_shape (see Notes)
    
    Notes
    -----
    Sum pooling returns the following mean-field estimates and samples for
    the hidden and pooling layers:
        h_mean is set to the input, rf_u
        h_sample is set to 1 indexed at the stochastically selected max unit
        (see stochastic_max_pool)
        p_mean is set to the sum across across units in the receptive field 
        indexed at the stochastically selected max unit
        p_sample is set to h_sample
    """

    # multiply u by mask
    if type(mask) is torch.Tensor:
        u = torch.mul(u, torch.as_tensor(mask, dtype=u.dtype))
    # get sum across last dim in u
    sum_val = torch.zeros_like(u)
    sum_val = torch.add(sum_val, torch.sum(u, -1, keepdim=True))
    sum_val = torch.reshape(sum_val, out_shape)
    # set detection mean-field estimates and samples
    h_mean = torch.reshape(u, out_shape)
    h_sample = torch.reshape(Multinomial(probs=local_softmax(u, -1, mask)).sample(), 
                             out_shape)
    # set pooling mean-field estimates and samples
    p_mean = torch.mul(sum_val, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def rf_pool(u, t=None, rfs=None, pool_type='prob', block_size=(2,2), mask_thr=1e-5, **kwargs):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        bottom-up input to pooling layer with shape (batch_size, ch, h, w)
    t : torch.Tensor or None
        top-down input to pooling layer with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1]) [default: None]
    rfs : torch.Tensor or None
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
    mask_thr : float
        threshold for creating a mask from rfs tensor (if input) [default: 1e-5]
    **kwargs : dict
        extra arguments passed to pooling function indicated by pool_type

    Returns
    -------
    h_mean : torch.Tensor
        hidden layer mean-field estimates with shape (batch_size, ch, h, w)
    h_sample : torch.Tensor
        hidden layer samples with shape (batch_size, ch, h, w)
    p_mean : torch.Tensor
        pooling layer mean-field estimates with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1])
    p_sample : torch.Tensor
        pooling layer samples with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1])

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
    'average' divides units by total number of units in the receptive field,
    'sum' returns sum over units in receptive field (especially for Gaussians).
    
    When block_size != (1,1), p_mean and p_sample result from a max operation
    across (n,n) blocks according to either probabilistic max-pooling or 
    stochastic max-pooling.
    
    See Also
    --------
    layers.RF_Pool : layer implementation of rf_pool function
    """

    # get bottom-up shape, block size
    batch_size, ch, u_h, u_w = u.shape
    b_h, b_w = block_size

    # get top-down
    u_t = u.clone()
    top_down = (type(t) is torch.Tensor)
    receptive_fields = (type(rfs) is torch.Tensor)
    
    # check bottom-up, top-down shapes
    if top_down:
        assert u.shape[:2] == t.shape[:2]
        assert u_h//b_h == t.shape[-2]
        assert u_w//b_w == t.shape[-1]
        t_shape = t.shape
    else:
        t_shape = (batch_size, ch, u_h//b_h, u_w//b_w)
    
    # add top-down, get blocks
    if top_down or not receptive_fields:
        # add bottom-up and top-down
        b = []
        for r in range(b_h):
            for c in range(b_w):
                if top_down:
                    u_t[:, :, r::b_h, c::b_w].add_(t)
                b.append(u_t[:, :, r::b_h, c::b_w].unsqueeze(-1))
        b = torch.cat(b, -1)

    # set pool_fn
    if pool_type == 'prob':
        pool_fn = prob_max_pool
    elif pool_type == 'stochastic':
        pool_fn = stochastic_max_pool
    elif pool_type == 'div_norm':
        pool_fn = div_norm_pool
    elif pool_type == 'average':
        pool_fn = average_pool
    elif pool_type == 'sum':
        pool_fn = sum_pool
    else: #TODO: allow pool_fn = pool_type if function
        raise Exception('pool_type not understood')

    # pooling across receptive fields
    if receptive_fields:
        # elemwise multiply u_t with rf_kernels
        u_t = u_t.unsqueeze(2)
        rf_kernels = torch.add(torch.zeros_like(u_t), rfs)
        rf_u = torch.mul(u_t, rf_kernels)
        # create rf_mask of receptive field kernels
        rf_mask = torch.as_tensor(torch.gt(rf_kernels, mask_thr), dtype=rf_u.dtype)
        # apply pool function across image dims
        h_mean, h_sample, p_mean, p_sample = pool_fn(rf_u.flatten(-2), rf_u.shape,
                                                     mask=rf_mask.flatten(-2), **kwargs)
        # max across receptive fields
        h_mean = torch.max(h_mean, -3)[0]
        h_sample = torch.max(h_sample, -3)[0]
        p_mean = torch.max(p_mean, -3)[0]
        p_sample = torch.max(p_sample, -3)[0]
    # pooling across blocks
    elif rfs is None:
        # init h_mean, h_sample, p_mean, p_sample
        h_mean = torch.zeros_like(u_t)
        h_sample = torch.zeros_like(u_t)
        p_mean = torch.zeros_like(u_t)
        p_sample = torch.zeros_like(u_t)
        # pool across blocks
        h_mean_b, h_sample_b, p_mean_b, p_sample_b = pool_fn(b, b.shape, **kwargs)
        for r in range(b_h):
            for c in range(b_w):
                h_mean[:, :, r::b_h, c::b_w] = h_mean_b[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = h_sample_b[:,:,:,:,(r*b_h) + c]
                p_mean[:, :, r::b_h, c::b_w] = p_mean_b[:,:,:,:,(r*b_h) + c]
                p_sample[:, :, r::b_h, c::b_w] = p_sample_b[:,:,:,:,(r*b_h) + c]
    else:
        raise Exception('rfs type not understood')
        
    # set p_mean, p_sample if blocks larger than (1,1)
    if block_size != (1,1):
        tmp_mean = p_mean.clone()
        tmp_sample = p_sample.clone()
        p_mean = torch.zeros(t_shape)
        p_sample = torch.zeros(t_shape)
        for r in range(b_h):
            for c in range(b_w):
                p_mean = torch.max(p_mean, tmp_mean[:, :, r::b_h, c::b_w])
                p_sample = torch.max(p_sample, tmp_sample[:, :, r::b_h, c::b_w])

    return h_mean, h_sample, p_mean, p_sample

if __name__ == '__main__':
    import doctest
    doctest.testmod()
