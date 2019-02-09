import numpy as np
import torch
from torch.distributions import Multinomial, Binomial

#TODO: create separate functions for each pooling operation
def prob_max_pool():
    raise NotImplementedError

def stochastic_max_pool():
    raise NotImplementedError

def div_norm_pool():
    raise NotImplementedError
    
def average_pool():
    raise NotImplementedError

#TODO: implement gaussian receptive fields in rf_pool
def rf_pool(u, t=None, rf_index=None, pool_type='prob', block_size=(2,2)):
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
    pool_type : string
        type of pooling ('prob' [default], 'stochastic', 'div_norm', 'average')
    block_size : tuple
        size of blocks in detection layer connected to pooling units [default: (2,2)]

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
    >>> u = torch.rand(1,10,16,16)
    >>> t = torch.rand(1,10,8,8)
    >>> rf_index = [(slice(0,4),slice(0,4)), 
                    (slice(4,8),slice(0,4)), 
                    (slice(4,8),slice(4,8)), 
                    (slice(0,4),slice(4,8))]
    >>> pool_type = 'prob'
    >>> block_size = (2,2)
    >>> h_mean, h_sample, p_mean, p_sample = rf_pool(u, t, rf_index, pool_type, block_size)

    Notes
    -----
    pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization with sigma=0.5 (Heeger, 1992),
    'average' divides each unit in receptive field by the total number of units in the receptive field.
    """

    # get bottom-up shape, block size
    batch_size, ch, u_h, u_w = u.shape
    b_h, b_w = block_size

    # set sigma_sqr for div_norm
    if pool_type == 'div_norm':
        sigma_sqr = torch.as_tensor(np.square(0.5), dtype=u.dtype)

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
    b.append(torch.zeros_like(b[-1]))
    b = torch.cat(b, -1)

    # init h_mean, h_sample
    if pool_type == 'stochastic':
        h_mean = u.clone()
    else:
        h_mean = torch.zeros_like(u)
    h_sample = torch.zeros_like(u)

    # set h_mean, h_sample
    if rf_index is not None:
        # pool over each RF
        for rf in rf_index:
            # get RF activity
            rf_act = u[:,:,rf[0],rf[1]]
            # set h_mean, h_sample based on pool_type
            if pool_type == 'prob':
                probs = torch.softmax(torch.cat([torch.flatten(rf_act, 2),
                                                 torch.zeros(rf_act.shape[:2] + (1,))], -1), -1)
                h_mean[:,:,rf[0],rf[1]] = torch.reshape(probs[:,:,:-1], rf_act.shape)
                h_sample[:,:,rf[0],rf[1]] = torch.reshape(Multinomial(probs=probs).sample()[:,:,:-1], rf_act.shape)
            elif pool_type == 'stochastic':
                probs = torch.softmax(torch.flatten(rf_act, 2), -1)
                h_sample[:,:,rf[0],rf[1]] = torch.reshape(Multinomial(probs=probs).sample(), rf_act.shape)
            elif pool_type == 'div_norm':
                rf_act_sqr = torch.pow(rf_act, 2.)
                probs = torch.div(rf_act_sqr, sigma_sqr + torch.sum(rf_act_sqr, dim=(2,3), keepdim=True))
                h_mean[:,:,rf[0],rf[1]] = probs
                h_sample[:,:,rf[0],rf[1]] = Binomial(probs=probs).sample()
            elif pool_type == 'average':
                probs = torch.sigmoid(torch.div(rf_act, torch.prod(rf_act.shape[2:])))
                h_mean[:,:,rf[0],rf[1]] = probs
                h_sample[:,:,rf[0],rf[1]] = Binomial(probs=probs).sample()
    else: # if no rf_index, pool over blocks
        if pool_type == 'prob':
            probs = torch.softmax(b, -1)
            sample = Multinomial(probs=probs).sample()
        elif pool_type == 'stochastic':
            probs = torch.softmax(b[:,:,:,:,:-1], -1)
            sample = Multinomial(probs=probs).sample()
        elif pool_type == 'div_norm':
            b_sqr = torch.pow(b[:,:,:,:,:-1], 2.)
            probs = torch.div(b_sqr, sigma_sqr + torch.sum(b_sqr, dim=-1, keepdim=True))
            sample = Binomial(probs=probs).sample()
        elif pool_type == 'average':
            probs = torch.sigmoid(torch.div(b[:,:,:,:,:-1], torch.prod(block_size)))
            sample = Binomial(probs=probs).sample()
        for r in range(b_h):
            for c in range(b_w):
                h_mean[:, :, r::b_h, c::b_w] = probs[:,:,:,:,(r*b_h) + c]
                h_sample[:, :, r::b_h, c::b_w] = sample[:,:,:,:,(r*b_h) + c]

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
                    p_mean = torch.min(torch.add(p_mean, h_mean[:, :, r::b_h, c::b_w]), torch.ones_like(p_mean))
                else:
                    p_mean = torch.max(p_mean, h_mean_stochastic[:, :, r::b_h, c::b_w])
                p_sample = torch.max(p_sample, h_sample[:, :, r::b_h, c::b_w])

    return h_mean, h_sample, p_mean, p_sample
