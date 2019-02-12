import numpy as np
import torch
from torch.distributions import Multinomial, Binomial

def prob_max_pool(rf_u, out_shape):
    """
    Probabilistic max-pooling along last dim
    #TODO:WRITEME
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
    Stochastic max-pooling along last dim
    #TODO:WRITEME
    """

    # get probabilities for each unit being on
    probs = torch.softmax(rf_u, -1)
    h_mean = torch.reshape(rf_u, out_shape)
    h_sample = torch.reshape(Multinomial(probs=probs).sample(), out_shape)
    return h_mean, h_sample

def div_norm_pool(rf_u, out_shape, sigma_sqr=0.25):
    """
    Divisive normalization along last dim
    #TODO:WRITEME
    """

    # o = u**2/(sigma**2 + sum(u**2))
    rf_u_sqr = torch.pow(rf_u, 2.)
    probs = torch.div(rf_u_sqr, sigma_sqr + torch.sum(rf_u_sqr, dim=-1, keepdim=True))
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(Binomial(probs=probs).sample(), out_shape)
    return h_mean, h_sample

def average_pool(rf_u, out_shape):
    """
    Average pooling along last dim
    #TODO:WRITEME
    """

    # divide activity by number of units
    probs = torch.div(rf_u, rf_u.shape[-1])
    samples = Binomial(probs=torch.sigmoid(probs)).sample()
    h_mean = torch.reshape(probs, out_shape)
    h_sample = torch.reshape(samples, out_shape)
    return h_mean, h_sample

#TODO: not sure where to put sample
def sum_pool(rf_u, out_shape):
    """
    Sum pooling along last dim
    #TODO:WRITEME
    """

    # set h_mean to rf_u, h_sample to sum
    h_mean = torch.reshape(rf_u, out_shape)
    h_sample = torch.zeros_like(rf_u)
    h_sample.add_(torch.sum(rf_u, dim=-1, keepdim=True))
    h_sample = torch.reshape(h_sample, out_shape)
    return h_mean, h_sample

def pool(u, t=None, rf_index=None, rf_kernels=None, pool_type='prob', block_size=(2,2), mu=None):
    """
    Receptive field pooling

    Parameters
    ----------
    u : torch.Tensor
        bottom-up input to pooling layer with shape (batch_size, ch, h, w)
    t : torch.Tensor
        top-down input to pooling layer with shape (batch_size, ch, h//block_size[0], w//block_size[1])
        [default: None]
    rf_index : list
        indices for each receptive field (see square_lattice_utils) [default: None, applies pooling over blocks]
    rf_kernels : torch.Tensor
        kernels for each receptive field with shape (h, w, n_kernels)
    pool_type : string
        type of pooling ('prob' [default], 'stochastic', 'div_norm', 'average', 'sum')
    block_size : tuple
        size of blocks in detection layer connected to pooling units [default: (2,2)]
    mu : torch.Tensor
        xy-coordinates of receptive field centers with shape (2, n_kernels) for use with pool_type='sum'
        [default: None]

    Returns
    -------
    h_mean : torch.Tensor
        detection layer mean-field estimates with shape (batch_size, ch, h, w)
    h_sample : torch.Tensor
        detection layer samples with shape (batch_size, ch, h, w)
    p_mean : torch.Tensor
        pooling layer mean-field estimates with shape (batch_size, ch, h//block_size[0], w//block_size[1])
    p_sample : torch.Tensor
        pooling layer samples with shape (batch_size, ch, h//block_size[0], w//block_size[1])

    Examples
    --------
    # Performs probabilistic max-pooling across 4x4 regions tiling detection layer with top-down input
    >>> u = torch.rand(1,10,8,8)
    >>> t = torch.rand(1,10,4,4)
    >>> rf_index = [(slice(0,4),slice(0,4)),
                    (slice(4,8),slice(0,4)),
                    (slice(4,8),slice(4,8)),
                    (slice(0,4),slice(4,8))]
    >>> pool_type = 'prob'
    >>> block_size = (2,2)
    >>> h_mean, h_sample, p_mean, p_sample = pool(u, t, rf_index, None, pool_type, block_size)

    Notes
    -----
    pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization with sigma=0.5 (Heeger, 1992),
    'average' divides units in receptive field by number of units in the receptive field,
    'sum' returns sum over units in receptive field (especially for Gaussian rf_kernels).
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
    if rf_kernels is not None:
        #TODO: could be more efficient if rf_kernels were shape (n_kernels, h, w)
        # pointwise multiply u with rf_kernels
        rf_kernels = rf_kernels.permute(2,0,1).reshape(-1, 1, 1, u_h, u_w)
        g_u = torch.mul(u.unsqueeze(0), rf_kernels)
        # get rf_index as thresholded rf_kernels
        #TODO: determine reasonable threshold
        thr = 1e-5
        rf_index = torch.gt(rf_kernels, thr)
        rf_index = rf_index.repeat(1, batch_size, ch, 1, 1)
        for i, rf in enumerate(rf_index):
            rf_u = g_u[i][rf]
            if mu is not None and pool_type == 'sum':
                h_mean[rf], samples = pool_fn(rf_u.reshape(batch_size, ch, -1), rf_u.shape)
                h_sample[:,:,mu[0,i],mu[1,i]] = torch.flatten(samples.reshape(batch_size, ch, -1)[:,:,0])
            else:
                h_mean[rf], h_sample[rf] = pool_fn(rf_u.reshape(batch_size, ch, -1), rf_u.shape)
    elif rf_index is not None:
        # use rf_index
        for rf in rf_index:
            rf_u = u[:,:,rf[0],rf[1]]
            h_mean[:,:,rf[0],rf[1]], h_sample[:,:,rf[0],rf[1]] = pool_fn(torch.flatten(rf_u, -2), rf_u.shape)
    else:
        # pool across blocks
        probs, samples = pool_fn(b, b.shape)
        for r in range(b_h):
            for c in range(b_w):
                h_mean[:, :, r::b_h, c::b_w] = probs[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = samples[:,:,:,:,(r*b_h) + c]

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
