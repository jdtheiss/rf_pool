import numpy as np
import torch
from torch.distributions import Multinomial, Binomial

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
        mean-field estimates of detection layer after pooling (see Notes)
    h_sample : torch.Tensor
        samples of detection layer after pooling (see Notes)
        
    Notes
    -----
    Probabilistic max-pooling considers each receptive field to be a multinomial
    unit in which only one unit can be on or all units can be off. h_mean is a 
    softmax across all units in the receptive field and a unit representing all
    units being "off" (not returned in h_mean). h_sample is sampled from a 
    multinomial distribution with at most one unit set to 1.

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
    return h_mean, h_sample

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
        mean-field estimates of detection layer after pooling (see Notes)
    h_sample : torch.Tensor
        samples of detection layer after pooling (see Notes)
        
    Notes
    -----
    Stochastic max-pooling considers each receptive field to be a multinomial
    unit in which only one unit can be on. h_mean is a softmax across all
    units in the receptive field. h_sample is sampled from a multinomial
    distribution with probs=h_mean and exactly one unit set to 1.

    References
    ----------
    Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization 
    of deep convolutional neural networks. arXiv preprint arXiv:1301.3557.
    """

    # get probabilities for each unit being on
    probs = torch.softmax(rf_u, -1)
    h_mean = torch.reshape(rf_u, out_shape)
    h_sample = torch.reshape(Multinomial(probs=probs).sample(), out_shape)
    return h_mean, h_sample

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
        mean-field estimates of detection layer after pooling (see Notes)
    h_sample : torch.Tensor
        samples of detection layer after pooling (see Notes)
        
    Notes
    -----
    Divisive normalization raises the input to the power of n, and normalizes 
    each unit with a constant, sigma, added in the denominator:
    
    h_mean = torch.pow(rf_u, n)/torch.add(torch.pow(sigma, n),
             torch.sum(torch.pow(rf_u, n), dim=-1, keepdim=True))
    h_sample = Binomial(probs=h_mean).sample()
    
    With n=2 and sigma=0.5, div_norm_pool simulates the average cortical 
    normalization observed emprically (Heeger, 1992).

    References
    ----------
    Heeger, D. J. (1992). Normalization of cell responses in cat striate 
    cortex. Visual neuroscience, 9(2), 181-197.
    """

    # raise rf_u, sigma to nth power
    rf_u_n = torch.pow(rf_u, n)
    sigma_n = torch.pow(torch.as_tensor(sigma, dtype=rf_u.dtype), n)
    probs = torch.div(rf_u_n, sigma_n + torch.sum(rf_u_n, dim=-1, keepdim=True))
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(Binomial(probs=probs).sample(), out_shape)
    return h_mean, h_sample

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
        mean-field estimates of detection layer after pooling operation
    h_sample : torch.Tensor
        samples of detection layer after pooling operation
    
    Notes
    -----
    Average pooling divides each unit by the total number of units in the
    receptive field. h_sample is drawn from a Bernoulli distribution with
    probs=torch.sigmoid(h_mean).
    """

    # divide activity by number of units
    probs = torch.div(rf_u, rf_u.shape[-1])
    samples = Binomial(probs=torch.sigmoid(probs)).sample()
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(samples, out_shape)
    return h_mean, h_sample

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
        mean-field estimates of detection layer after pooling operation
    h_sample : torch.Tensor
        samples of detection layer after pooling operation
    
    Notes
    -----
    Sum pooling returns input for mean-field estimates and the sum across
    last dimension for the sampled output with all units in the receptive
    field taking the summed value:
    h_mean = rf_u
    h_sample = torch.zeros_like(rf_u) + torch.sum(rf_u, dim=-1, keepdim=True)
    """

    # set h_mean to rf_u, h_sample to sum
    h_mean = torch.reshape(rf_u, out_shape)
    h_sample = torch.zeros_like(rf_u)
    h_sample.add_(torch.sum(rf_u, dim=-1, keepdim=True))
    h_sample = torch.reshape(h_sample, out_shape)
    return h_mean, h_sample

def pool(u, t=None, rfs=None, pool_type='prob', block_size=(2,2), mu=None, pool_args=[]):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        bottom-up input to pooling layer with shape (batch_size, ch, h, w)
    t : torch.Tensor
        top-down input to pooling layer with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1]) [default: None]
    rfs : list or torch.Tensor
        if type is list, index for receptive fields (see square_lattice_utils) 
        if type is torch.Tensor, kernels for receptive fields with shape 
        (h, w, n_kernels) (see gaussian_lattice_utils) 
        [default: None, applies pooling over blocks]
    pool_type : string
        type of pooling ('prob', 'stochastic', 'div_norm', 'average', 'sum')
        [default: 'prob']
    block_size : tuple
        size of blocks in detection layer connected to pooling units 
        [default: (2,2)]
    mu : torch.Tensor
        xy-coordinates of receptive field centers with shape (2, n_kernels) 
        for use with pool_type='sum' [default: None]
    pool_args : list
        extra arguments sent to pooling function indicated by pool_type
        (especially for div_norm_pool) [default: []]

    Returns
    -------
    h_mean : torch.Tensor
        detection layer mean-field estimates with shape (batch_size, ch, h, w)
    h_sample : torch.Tensor
        detection layer samples with shape (batch_size, ch, h, w)
    p_mean : torch.Tensor
        pooling layer mean-field estimates with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1])
    p_sample : torch.Tensor
        pooling layer samples with shape 
        (batch_size, ch, h//block_size[0], w//block_size[1])

    Examples
    --------
    # Performs probabilistic max-pooling across 4x4 regions tiling detection 
    # layer with top-down input
    >>> u = torch.rand(1,10,8,8)
    >>> t = torch.rand(1,10,4,4)
    >>> rfs = [(slice(0,4),slice(0,4)),
               (slice(4,8),slice(0,4)),
               (slice(4,8),slice(4,8)),
               (slice(0,4),slice(4,8))]
    >>> h_mean, h_sample, p_mean, p_sample = pool(u, t, rfs, 'prob', (2,2))

    Notes
    -----
    pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization with sigma=0.5 (Heeger, 1992),
    'average' divides units by total number of units in the receptive field,
    'sum' returns sum over units in receptive field (especially for Gaussians).
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
    b = []
    for r in range(b_h):
        for c in range(b_w):
            u[:, :, r::b_h, c::b_w].add_(t)
            b.append(u[:, :, r::b_h, c::b_w].unsqueeze(-1))
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
    else:
        raise Exception('pool_type not understood')

    # init h_mean, h_sample
    h_mean = torch.zeros_like(u)
    h_sample = torch.zeros_like(u)
    if type(rfs) is torch.Tensor:
        #TODO: could be more efficient if rf_kernels were shape (n_kernels, h, w)
        # elemwise multiply u with rf_kernels
        rf_kernels = rfs.permute(2,0,1).reshape(-1, 1, 1, u_h, u_w)
        g_u = torch.mul(u.unsqueeze(0), rf_kernels)
        # get rf_index as thresholded rf_kernels
        #TODO: determine reasonable threshold
        thr = 1e-5
        rf_index = torch.gt(rf_kernels, thr)
        rf_index = rf_index.repeat(1, batch_size, ch, 1, 1)
        for i, rf in enumerate(rf_index):
            rf_u = g_u[i][rf]
            if mu is not None and pool_type == 'sum':
                h_mean[rf], samples = pool_fn(rf_u.reshape(batch_size, ch, -1), rf_u.shape, *pool_args)
                h_sample[:,:,mu[0,i],mu[1,i]] = torch.flatten(samples.reshape(batch_size, ch, -1)[:,:,0])
            else:
                h_mean[rf], h_sample[rf] = pool_fn(rf_u.reshape(batch_size, ch, -1), 
                                                   rf_u.shape, 
                                                   *pool_args)
    elif type(rfs) is list:
        # index u with each rf
        for rf in rfs:
            rf_u = u[:,:,rf[0],rf[1]]
            h_mean[:,:,rf[0],rf[1]], h_sample[:,:,rf[0],rf[1]] = pool_fn(torch.flatten(rf_u, -2), 
                                                                         rf_u.shape,
                                                                         *pool_args)
    elif rfs is None:
        # pool across blocks
        probs, samples = pool_fn(b, b.shape, *pool_args)
        for r in range(b_h):
            for c in range(b_w):
                h_mean[:, :, r::b_h, c::b_w] = probs[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = samples[:,:,:,:,(r*b_h) + c]
    else:
        raise Exception('rfs type not understood')

    #TODO: decide on best p_mean, p_sample for each pooling operation and how to implement
    # set p_mean, p_sample
    if block_size == (1,1):
        p_mean = h_mean.clone()
        p_sample = h_sample.clone()
    else:
        p_mean = torch.zeros_like(t)
        p_sample = torch.zeros_like(t)
        # stochastic index
        h_mean_stochastic = torch.mul(h_sample, h_mean)
        for r in range(b_h):
            for c in range(b_w):
                if pool_type == 'prob':
                    p_mean = torch.add(p_mean, h_mean[:, :, r::b_h, c::b_w])
                else:
                    p_mean = torch.max(p_mean, h_mean_stochastic[:, :, r::b_h, c::b_w])
                p_sample = torch.max(p_sample, h_sample[:, :, r::b_h, c::b_w])

    return h_mean, h_sample, p_mean, p_sample

if __name__ == '__main__':
    import doctest
    doctest.testmod()
