import numpy as np
import torch
from torch.distributions import Multinomial

def prob_max_pool(rf_u, out_shape):
    """
    Probabilistic max-pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    rf_u : torch.Tensor
        receptive field to pool over with shape[-1] == n_units
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
    off_pt = torch.zeros(rf_u.shape[:-1] + (1,), dtype=rf_u.dtype)
    events = torch.cat([rf_u, off_pt], -1)
    probs = torch.softmax(events, -1)
    samples = Multinomial(probs=probs).sample()
    # get mean-field estimates and samples
    index = torch.cat([torch.ones_like(rf_u, dtype=torch.uint8), 
                       torch.zeros_like(off_pt, dtype=torch.uint8)], -1)
    h_mean = torch.reshape(probs[index], out_shape)
    h_sample = torch.reshape(samples[index], out_shape)
    p_mean = torch.zeros_like(rf_u)
    p_mean = torch.add(p_mean, torch.reshape(1. - probs[index-1], off_pt.shape))
    p_mean = torch.reshape(p_mean, out_shape)
    p_mean = torch.mul(p_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def stochastic_max_pool(rf_u, out_shape):
    """
    Stochastic max-pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    rf_u : torch.Tensor
        receptive field to pool over with shape[-1] == n_units
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
    probs = torch.softmax(rf_u, -1)
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(Multinomial(probs=probs).sample(), out_shape)
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def div_norm_pool(rf_u, out_shape, n=2., sigma=0.5):
    """
    Divisive normalization across units in a receptive field (last dim)
    
    Parameters
    ----------
    rf_u : torch.Tensor
        receptive field to pool over with shape[-1] == n_units
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
        h_sample = Multinomial(probs=torch.softmax(h_mean, -1)).sample()
        p_mean = torch.mul(h_mean, h_sample)
        p_sample = h_sample.clone()
    
    With n=2 and sigma=0.5, div_norm_pool simulates the average cortical 
    normalization observed emprically (Heeger, 1992).

    References
    ----------
    Heeger, D. J. (1992). Normalization of cell responses in cat striate 
    cortex. Visual neuroscience, 9(2), 181-197.
    """

    # get inf indices, set to 0.
    inf_mask = torch.isinf(rf_u)
    rf_u[inf_mask] = 0.
    # raise rf_u, sigma to nth power
    rf_u_n = torch.pow(rf_u, n)
    sigma_n = torch.pow(torch.as_tensor(sigma, dtype=rf_u.dtype), n)
    probs = torch.div(rf_u_n, sigma_n + torch.sum(rf_u_n, dim=-1, keepdim=True))
    h_mean = torch.reshape(probs.clone(), out_shape)
    # set inf indices to -inf for softmax
    probs[inf_mask] = -np.inf
    h_sample = torch.reshape(Multinomial(probs=torch.softmax(probs, -1)).sample(), out_shape)
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def average_pool(rf_u, out_shape):
    """
    Average pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    rf_u : torch.Tensor
        receptive field to pool over with shape[-1] == n_units
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

    # get inf indices, set to 0.
    inf_mask = torch.isinf(rf_u)
    rf_u[inf_mask] = 0.
    # get count of units in last dim
    n_units = torch.as_tensor(rf_u.shape[-1] - torch.sum(inf_mask, -1, keepdim=True), 
                              dtype=rf_u.dtype)
    # divide activity by number of units
    probs = torch.div(rf_u, n_units)
    h_mean = torch.reshape(probs.clone(), out_shape)
    # set inf indices to -inf for softmax
    probs[inf_mask] = -np.inf
    samples = Multinomial(probs=torch.softmax(probs, -1)).sample()
    h_sample = torch.reshape(samples, out_shape)
    p_mean = torch.mul(h_mean, h_sample)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def sum_pool(rf_u, out_shape):
    """
    Sum pooling across units in a receptive field (last dim)
    
    Parameters
    ----------
    rf_u : torch.Tensor
        receptive field to pool over with shape[-1] == n_units
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
        h_sample is set to the sum across the receptive field indexed at
        the stochastically selected max unit in the receptive field (see 
        stochastic_max_pool)
        p_mean is set to the input value indexed at the stochastically
        selected max unit
        p_sample is set to h_sample
    """

    # get inf indices, set to 0.
    inf_mask = torch.isinf(rf_u)
    rf_u[inf_mask] = 0.
    # set h_mean to rf_u, h_sample to sum
    h_mean = torch.reshape(rf_u.clone(), out_shape)
    h_sample = torch.zeros_like(rf_u)
    h_sample.add_(torch.sum(rf_u, dim=-1, keepdim=True))
    h_sample = torch.reshape(h_sample, out_shape)
    # set inf indices to -inf for softmax
    rf_u[inf_mask] = -np.inf
    sampled_pos = torch.reshape(Multinomial(probs=torch.softmax(rf_u, -1)).sample(), out_shape)
    h_sample = torch.mul(h_sample, sampled_pos)
    p_mean = torch.mul(h_mean, sampled_pos)
    p_sample = h_sample.clone()
    return h_mean, h_sample, p_mean, p_sample

def rf_pool(u, t=None, rfs=None, pool_type='prob', block_size=(2,2), pool_args=[]):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        bottom-up input to pooling layer with shape (batch_size, ch, h, w)
    t : torch.Tensor
        top-down input to pooling layer with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1]) [default: None]
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
    layers.RF_Pool : layer implementation of rf.pool function
    """

    # get bottom-up shape, block size
    batch_size, ch, u_h, u_w = u.shape
    b_h, b_w = block_size

    # get top-down
    if t is None:
        t = torch.zeros((batch_size, ch, u_h//b_h, u_w//b_w), dtype=u.dtype)

    # check bottom-up, top-down shapes
    assert u.shape[:2] == t.shape[:2]
    assert u_h//b_h == t.shape[-2]
    assert u_w//b_w == t.shape[-1]

    # add bottom-up and top-down
    u_t = u.clone()
    b = []
    for r in range(b_h):
        for c in range(b_w):
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

    # check pool_args is list
    assert type(pool_args) is list

    # init h_mean, h_sample
    h_mean = torch.zeros_like(u_t)
    h_sample = torch.zeros_like(u_t)
    p_mean = torch.zeros_like(u_t)
    p_sample = torch.zeros_like(u_t)
    if type(rfs) is torch.Tensor:
        # elemwise multiply u_t with rf_kernels
        u_t = u_t.unsqueeze(2)
        rf_kernels = torch.add(torch.zeros_like(u_t), rfs)
        g_u = torch.mul(u_t, rf_kernels).permute(2,0,1,3,4)
        # create rf_mask of 0s at rf and -inf elsewhere
        thr = 1e-5
        rf_mask = torch.as_tensor(torch.le(rf_kernels, thr).permute(2,0,1,3,4),
                                  dtype=g_u.dtype)
        rf_mask = -1. * torch.exp(np.inf * (2. * rf_mask - 1.))
        # get rf_u with g_u at rf and -inf elsewhere
        rf_u = torch.add(g_u, rf_mask)
        # apply pool function across image dims
        h_mean, h_sample, p_mean, p_sample = pool_fn(rf_u.flatten(-2), rf_u.shape, *pool_args)
        # max across receptive fields
        h_mean = torch.max(h_mean, 0)[0]
        h_sample = torch.max(h_sample, 0)[0]
        p_mean = torch.max(p_mean, 0)[0]
        p_sample = torch.max(p_sample, 0)[0]
    elif rfs is None:
        # pool across blocks
        h_mean_b, h_sample_b, p_mean_b, p_sample_b = pool_fn(b, b.shape, *pool_args)
        for r in range(b_h):
            for c in range(b_w):
                h_mean[:, :, r::b_h, c::b_w] = h_mean_b[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = h_sample_b[:,:,:,:,(r*b_h) + c]
                p_mean[:, :, r::b_h, c::b_w] = p_mean_b[:,:,:,:,(r*b_h) + c]
                p_sample[:, :, r::b_h, c::b_w] = p_sample_b[:,:,:,:,(r*b_h) + c]
    else:
        raise Exception('rfs type not understood')
        
    # set p_mean, p_sample
    if block_size != (1,1):
        tmp_mean = p_mean.clone()
        tmp_sample = p_sample.clone()
        p_mean = torch.zeros_like(t)
        p_sample = torch.zeros_like(t)
        for r in range(b_h):
            for c in range(b_w):
                p_mean = torch.max(p_mean, tmp_mean[:, :, r::b_h, c::b_w])
                p_sample = torch.max(p_sample, tmp_sample[:, :, r::b_h, c::b_w])

    return h_mean, h_sample, p_mean, p_sample

if __name__ == '__main__':
    import doctest
    doctest.testmod()
