import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Multinomial, Binomial

def make_RFs(img_shape, rf_sizes, stride=0):
    '''Make receptive fields for rf_pool
    parameters
    ----------
    img_shape : tuple, height and width of bottom-up input to pooling layer
    rf_sizes : list, receptive field sizes to tile detection layer (see example)
    stride : int, spacing (if positive) or overlap (if negative) between receptive fields [default: 0]

    returns
    -------
    rf_index : list, slice indices for each receptive field (i.e. [(slice(start, end), slice(start, end))])

    example
    -------
    # creates indices for 32 receptive fields of size 4x4, 3x3, 2x2 tiling image from outer-most
    # indices to inner-most indices
    img_shape = (18,18)
    rf_sizes = [4,3,2]
    rf_index = make_RFs(img_shape, rf_sizes, stride=0)

    Note: If receptive field sizes do not fully cover image, rf_sizes[-1] will be appended to rf_sizes until
    the image can be completely covered (unless (rf_sizes[-1] + stride) <= 0).
    '''

    # ensure rf_sizes given
    assert len(rf_sizes) > 0

    # get image size
    img_h, img_w = img_shape

    # append rf_size[-1] if img not fully covered
    while (img_h - 2*np.sum(rf_sizes) - 2*len(rf_sizes)*stride) > 0 and \
          (img_w - 2*np.sum(rf_sizes) - 2*len(rf_sizes)*stride) > 0:
        if rf_sizes[-1] + stride > 0:
            rf_sizes.extend([rf_sizes[-1]])
        else: # warn that image won't be covered
            warnings.warn("Image will not be fully covered by receptive fields.")
            break

    # for each RF size, get slice indices
    rf_index = []
    for n in range(len(rf_sizes)):
        # get sum of previous rf sizes
        sum_rfs = np.sum(rf_sizes[:n], dtype='uint8') + stride*n
        # get current image size for rf
        if (img_shape[0] - 2*sum_rfs) < rf_sizes[n] and (img_shape[1] - 2*sum_rfs) < rf_sizes[n]:
            if (img_shape[0] - 2*sum_rfs) > (0 - 2*stride) or (img_shape[1] - 2*sum_rfs) > (0 - 2*stride):
                warnings.warn("Image may not be fully covered by receptive fields.")
            break
        if (img_shape[0] - 2*sum_rfs) >= rf_sizes[n]:
            img_h = img_shape[0] - 2*sum_rfs
        if (img_shape[1] - 2*sum_rfs) >= rf_sizes[n]:
            img_w = img_shape[1] - 2*sum_rfs
        # init i, j slices
        i = []
        j = []
        # set left side slices
        i.extend([slice(ii, ii+rf_sizes[n]) for ii in np.arange(sum_rfs, img_h+sum_rfs, rf_sizes[n] + stride)
                  if ii+rf_sizes[n] <= img_h+sum_rfs])
        j.extend([slice(jj, jj+rf_sizes[n]) for jj in np.repeat(sum_rfs, np.ceil(img_h/(rf_sizes[n]+stride)))])
        # set bottom side slices
        i.extend([slice(ii-rf_sizes[n], ii) for ii in np.repeat(img_h+sum_rfs, np.ceil(img_w/(rf_sizes[n]+stride)))])
        j.extend([slice(jj, jj+rf_sizes[n]) for jj in np.arange(sum_rfs, img_w+sum_rfs, rf_sizes[n] + stride)
                  if jj+rf_sizes[n] <= img_w+sum_rfs])
        # set right side slices
        i.extend([slice(ii-rf_sizes[n], ii) for ii in np.arange(img_h+sum_rfs, sum_rfs, -rf_sizes[n] - stride)
                  if ii-rf_sizes[n] >= sum_rfs])
        j.extend([slice(jj-rf_sizes[n], jj) for jj in np.repeat(img_w+sum_rfs, np.ceil(img_h/(rf_sizes[n]+stride)))])
        # set top side slices
        i.extend([slice(ii, ii+rf_sizes[n]) for ii in np.repeat(sum_rfs, np.ceil(img_w/(rf_sizes[n]+stride)))])
        j.extend([slice(jj-rf_sizes[n], jj) for jj in np.arange(img_w+sum_rfs, sum_rfs, -rf_sizes[n] - stride)
                  if jj-rf_sizes[n] >= sum_rfs])
        # append if not already in coords
        for ii,jj in zip(i,j):
            if len(rf_index) == 0 or (ii,jj) not in rf_index:
                rf_index.append((ii,jj))
    return rf_index

def view_RFs(img_shape, rf_index):
    '''View receptive fields created by make_RFs
    parameters
    ----------
    img_shape : tuple, image shape used in make_RFs to create receptive fields
    rf_index : list, list of receptive field indices created by make_RFs

    returns
    -------
    None

    Note: Values in image shown are proportional to the receptive field number in rf_index.
    '''
    # init image
    img = np.zeros(img_shape)

    # set each RF to index number
    for i, rf in enumerate(rf_index):
        img[rf[0],rf[1]] += (i + 1)

    # show image
    plt.imshow(img, cmap='gray', vmin=-len(rf_index), vmax=len(rf_index))
    plt.show()

def rf_pool(u, t=None, rf_index=None, pool_type='prob', block_size=(2,2)):
    '''Receptive field pooling
    parameters
    ----------
    u : tensor, bottom-up input to pooling layer
    t : tensor, top-down input to pooling layer [default: None]
    rf_index : list, indices for each receptive field (see make_RFs) [default: None, applies pooling over blocks]
    pool_type : string, type of pooling ('prob' [default], 'stochastic', 'div_norm', 'average')
    block_size : tuple, size of blocks in detection layer connected to pooling units [default: (2,2)]

    returns
    -------
    h_mean : tensor, detection layer mean-field estimates
    h_sample : tensor, detection layer samples
    p_mean : tensor, pooling layer mean-field estimates
    p_sample : tensor, pooling layer samples

    example
    -------
    u = torch.rand(1,10,18,18)
    t = torch.rand(1,10,9,9)
    rf_index = make_RFs((18,18), [4,3,2])
    pool_type = 'prob'
    block_size = (2,2)
    h_mean, h_sample, p_mean, p_sample = rf_pool(u, t, rf_index, pool_type, block_size)

    Note: pool_type 'prob' refers to probabilistic max-pooling (Lee et al., 2009),
    'stochastic' refers to stochastic max-pooling (Zeiler & Fergus, 2013),
    'div_norm' performs divisive normalization with sigma=0.5 (Heeger, 1992),
    'average' divides each unit in receptive field by the total number of units in the receptive field.
    '''

    # get bottom-up shape, block size
    batch_size, ch, u_h, u_w = u.shape
    b_h, b_w = block_size

    # set sigma_sqr for div_norm
    sigma_sqr = torch.as_tensor(np.square(0.5), dtype=u.dtype)

    # get top-down
    if t is None:
        t = torch.zeros((batch_size, ch, u_h//b_h, u_w//b_w), dtype=u.dtype)

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
