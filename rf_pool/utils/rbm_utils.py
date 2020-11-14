import numpy as np
import torch
import torch.nn.functional as F

from . import functions

def log_prob(rbm, v, n_data=-1, log_Z=None, **kwargs):
    """
    Log probability of data

    Parameters
    ----------
    v : torch.Tensor or torch.utils.data.dataloader.DataLoader
        data to compute log probability
    n_data : int
        number of data points (or batches if v is `DataLoader`) within v
        to compute log probability [default: -1, all data in v]
    log_Z : float or None
        log of the partition function over model (calculated using `ais`)
        [default: None, log_Z computed by passing kwargs to ais]

    Returns
    -------
    log_p_v : float
        log probability of data

    See also
    --------
    ais : compute log_Z of the model

    References
    ----------
    (Salakhutdinov & Murray 2008)
    """
    # compute log_Z
    if log_Z is None:
        log_Z = ais(**kwargs)
    # set n_data
    n_data = n_data if n_data > 0 else len(v)
    # compute free energy data
    if isinstance(v, torch.utils.data.dataloader.DataLoader):
        # set number of batches to n_data
        n_batches = n_data
        # get mean free energy for each batch
        fe = 0.
        for i, (data, _) in enumerate(v):
            if i > n_batches:
                break
            fe += torch.mean(rbm.free_energy(data))
    else: # get free energy for tensor input
        n_batches = 1.
        fe = torch.mean(rbm.free_energy(v[:n_data]))
    # return log prob of data
    return -torch.div(fe, n_batches) - log_Z

def ais(rbm, m, beta, base_rate, base_log_part_fn=F.softplus):
    """
    Annealed Importance Sampling (AIS) for estimating log(Z) of model

    Parameters
    ----------
    m : int
        number of AIS runs to compute
    beta : list or array-like
        beta values in [0,1] for weighting distributions (see References)
    base_rate : torch.Tensor
        visible biases for base model (natural parameter of exponential family)
        with `base_rate.shape == data[0,None].shape`
    base_log_part_fn : torch.nn.functional
        log-partition function for visible units
        [default: `torch.nn.functional.softplus`, i.e. binary units]

    Returns
    -------
    log_Z_model : float
        estimate of the log of the partition function for the model
        (used in computing log probability of data)

    See also
    --------
    log_prob : estimate log probability of data
    base_rate : estimate base_rate for some binary data

    References
    ----------
    (Salakhutdinov & Murray 2008)
    """
    # repeat base_rate m times
    base_rate_m = functions.repeat(base_rate, (m,))
    # reshape v_bias
    v_dims = tuple([1 for _ in range(base_rate.ndimension()-2)])
    v_bias = torch.reshape(rbm.v_bias, (1,-1) + v_dims)
    # init log_pk (estimated log(Z_model/Z_base))
    log_pk = torch.zeros(m)
    # get v_0 from base_rate_m
    v_k = rbm.sample(base_rate_m,'reconstruct_layer')[1]
    # get log(p_0(v_1))
    log_pk -= _ais_free_energy(v_k, beta[0], base_rate)
    # get log(p_k(v_k) and log(p_k(v_k+1)) for each beta in (0, 1)
    for b in beta[1:-1]:
        # get log(p_k(v_k))
        log_pk += _ais_free_energy(v_k, b, base_rate)
        # sample h
        Wv_b = rbm.apply_modules(v_k, 'forward_layer',
                                 output_module=rbm.tie_weights_module)
        h = rbm.sample(Wv_b * b, 'forward_layer')[1]
        # sample v_k+1
        pre_act_v = rbm.apply_modules(h, 'reconstruct_layer',
                                      [rbm.tie_weights_module])
        v_k = rbm.sample((1. - b) * base_rate_m + b * pre_act_v,
                          'reconstruct_layer')[1]
        # get log(p_k(v_k+1))
        log_pk -= _ais_free_energy(v_k, b, base_rate)
    # get log(p_k(v_k))
    log_pk += _ais_free_energy(v_k, beta[-1], base_rate)
    # get mean across m cases for log AIS ratio of Z_model/Z_base
    r_AIS = torch.logsumexp(log_pk, 0) - np.log(m)
    # get log_Z_base
    base_h = torch.zeros_like(h[0,None])
    log_Z_base = torch.add(torch.sum(base_log_part_fn(base_rate)),
                           torch.sum(rbm.log_part_fn(base_h)))
    # return estimated log_Z_model log(Z_B/Z_A * Z_A)
    log_Z_model = r_AIS + log_Z_base
    return log_Z_model

def _ais_free_energy(rbm, v, beta, base_rate):
    # reshape v_bias
    v_dims = tuple([1 for _ in range(v.ndimension()-2)])
    v_bias = torch.reshape(rbm.vis_bias, (1,-1) + v_dims)
    # get Wv_b
    Wv_b = rbm.sample_h_given_v(v)[0]
    # get vbias, hidden terms
    base_term = (1. - beta) * torch.sum(torch.flatten(v * base_rate, 1), 1)
    vbias_term = beta * torch.sum(torch.flatten(v * v_bias, 1), 1)
    hidden_term = torch.sum(rbm.log_part_fn(beta * Wv_b).flatten(1), 1)
    return base_term + vbias_term + hidden_term

def base_rate(rbm, dataloader, lp=5.):
    """
    Base-rate model (for RBMs)

    (Salakhutdinov & Murray 2008)

    NOTE: Currently only for binary data
    """
    b = torch.zeros_like(iter(dataloader).next()[0][0,None])
    n_batches = len(dataloader)
    for data, _ in dataloader:
        b += torch.mean(data, 0, keepdim=True)
    p_b = (b + lp * n_batches) / (n_batches + lp * n_batches)
    return torch.log(p_b) - torch.log(1. - p_b)

def pseudo_likelihood(rbm, v):
    """
    Get pseudo-likelihood via randomly flipping bits and measuring free energy

    Parameters
    ----------
    input : torch.tensor
        binary input to obtain pseudo-likelihood

    Returns
    -------
    pl : float
        pseudo-likelihood given input

    Notes
    -----
    This likelihood estimate is only appropriate for binary data.

    A random index in the input image is flipped on each call. Averaging
    over many different indices approximates the pseudo-likelihood.
    """
    n_visible = np.prod(v.shape[1:])
    # get free energy for input
    xi = torch.round(v)
    fe_xi = rbm.free_energy(xi)
    # flip bit and get free energy
    xi_flip = torch.flatten(xi, 1)
    bit_idx = torch.randint(xi_flip.shape[1], (xi.shape[0],))
    xi_idx = np.arange(xi.shape[0])
    xi_flip[xi_idx, bit_idx] = 1. - xi_flip[xi_idx, bit_idx]
    xi_flip = torch.reshape(xi_flip, v.shape)
    fe_xi_flip = rbm.free_energy(xi_flip)
    # return pseudo-likelihood
    return torch.mean(n_visible * torch.log(torch.sigmoid(fe_xi_flip-fe_xi)))

def gaussian_pseudo_likelihood(rbm, v):
    n_visible = np.prod(v.shape[1:])
    # get free energy for input
    fe_xi = rbm.free_energy(v)
    # flip random bit
    xi_flip = torch.flatten(v, 1)
    bit_idx = torch.randint(xi_flip.shape[1], (v.shape[0],))
    xi_idx = np.arange(v.shape[0])
    xi_flip[xi_idx, bit_idx] = -xi_flip[xi_idx, bit_idx]
    fe_xi_flip = rbm.free_energy(xi_flip.reshape(v.shape))
    # return pseudo-likelihood
    return torch.mean(n_visible * torch.log(torch.sigmoid(fe_xi_flip - fe_xi)))
