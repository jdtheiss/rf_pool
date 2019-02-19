"""
Utilities for creating receptive field lattices
for use with rf.pool function.

Examples
--------
# initialize uniform lattice
>>> img_shape = (200,200)
>>> center = (100,100)
>>> n_kernel_side = 8
>>> spacing = 12
>>> sigma_init = 3.
>>> mu, sigma = init_uniform_lattice(center, n_kernel_side, spacing, sigma_init)

# create gaussian kernels from mu, sigma
>>> kernels = gaussian_kernel_lattice(mu, sigma, img_shape)

# show kernels
>>> show_kernel_lattice(kernels)

# initialize foveated lattice
>>> img_shape = (200,200)
>>> scale = 0.25
>>> spacing = 0.15
>>> min_ecc = 1.
>>> mu, sigma = init_foveated_lattic(img_shape, scale, spacing, min_ecc)

# create exponential kernels from mu, sigma
>>> kernels = exp_kernel_lattice(mu, sigma, img_shape)

# show kernels
>>> show_kernel_lattice(kernels)

# initialize tiled lattice
>>> img_shape = (200,200)
>>> rf_sizes = [3]
>>> spacing = 9
>>> mu, sigma = init_uniform_lattice(img_shape, rf_sizes, spacing)
    
# create masked kernels from mu, sigma
>>> kernels = mask_kernel_lattice(mu, sigma, img_shape)

# show kernels
>>> show_kernel_lattice(kernels)
"""

import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

def generalized_sigmoid(x, v=1., q=1., b=1.):
    return 1./torch.pow(1. + q * np.exp(-b * x), 1./v)

def exp_kernel_2d(mu, sigma, xy):
    mu = mu.reshape(mu.shape + (1,1)).float()
    sigma = sigma.unsqueeze(-1).float()
    return torch.exp(-torch.sum((xy - mu)**2., dim=-3)/ (2*sigma**2))

def gaussian_kernel_2d(mu, sigma, xy):
    mu = mu.reshape(mu.shape + (1,1)).float()
    sigma = sigma.unsqueeze(-1).float()
    return (1./(2.*np.pi*sigma**2)) * torch.exp(-torch.sum((xy - mu)**2., dim=-3)/ (2*sigma**2))

def mask_kernel_2d(mu, sigma, xy):
    kernels = exp_kernel_2d(mu, sigma, xy)
    thr = torch.exp(torch.as_tensor(-1, dtype=kernels.dtype))
    return torch.gt(kernels, thr).float()

def exp_kernel_lattice(mu, sigma, kernel_shape):
    """
    Returns a tensor of exponential kernels with max value = 1

    Parameters
    ----------
    mu : torch.Tensor 
        kernel centers with shape (n_kernels, 2)
    sigma : torch.Tensor 
        kernel standard deviations with shape (n_kernels, 1)
    kernel_shape : tuple
        shape of input feature map

    Returns
    -------
    kernels : torch.Tensor 
        output kernels with shape 
        mu.shape[:-1] + (kernel_shape[0], kernel_shape[1])

    Examples
    --------
    # Create tensor of 10 kernels with random centers and sigma=1.
    >>> mu = torch.rand(10, 2)
    >>> sigma = torch.ones(10, 1)
    >>> kernel_shape = (200,200)
    >>> mu = mu * torch.as_tensor(kernel_shape, dtype=mu.dtype)
    >>> kernels = exp_kernel_lattice(mu, sigma, kernel_shape)
    """
    
    assert mu.shape[-2] == sigma.shape[-2]
    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return exp_kernel_2d(mu, sigma, xy)

def gaussian_kernel_lattice(mu, sigma, kernel_shape):
    """
    Returns a tensor of gaussian kernels

    Parameters
    ----------
    mu : torch.Tensor 
        kernel centers with shape (n_kernels, 2)
    sigma : torch.Tensor 
        kernel standard deviations with shape (n_kernels, 1)
    kernel_shape : tuple
        shape of input feature map

    Returns
    -------
    kernels : torch.Tensor 
        output kernels with shape 
        mu.shape[:-1] + (kernel_shape[0], kernel_shape[1])

    Examples
    --------
    # Create tensor of 10 kernels with random centers and sigma=1.
    >>> mu = torch.rand(10, 2)
    >>> sigma = torch.ones(10, 1)
    >>> kernel_shape = (200,200)
    >>> mu = mu * torch.as_tensor(kernel_shape, dtype=mu.dtype)
    >>> kernels = gaussian_kernel_lattice(mu, sigma, kernel_shape)
    """
    
    assert mu.shape[-2] == sigma.shape[-2]
    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return gaussian_kernel_2d(mu, sigma, xy)

def mask_kernel_lattice(mu, sigma, kernel_shape):
    """
    Returns a tensor of masked kernels

    Parameters
    ----------
    mu : torch.Tensor 
        kernel centers with shape (n_kernels, 2)
    sigma : torch.Tensor 
        kernel standard deviations with shape (n_kernels, 1)
    kernel_shape : tuple
        shape of input feature map

    Returns
    -------
    kernels : torch.Tensor 
        output kernels with shape 
        mu.shape[:-1] + (kernel_shape[0], kernel_shape[1])

    Examples
    --------
    # Create tensor of 10 kernels with random centers and sigma=1.
    >>> mu = torch.rand(10, 2)
    >>> sigma = torch.ones(10, 1)
    >>> kernel_shape = (200, 200)
    >>> mu = mu * torch.as_tensor(kernel_shape, dtype=mu.dtype)
    >>> kernels = mask_kernel_lattice(mu, sigma, kernel_shape)
    """
    
    assert mu.shape[-2] == sigma.shape[-2]
    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return mask_kernel_2d(mu, sigma, xy)

def init_foveated_lattice(img_shape, scale, spacing, min_ecc=1.):
    """
    Creates a foveated lattice of kernel centers (mu) and
    stantard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    scale : float
        rate at which receptive field size scales with eccentricity
    spacing : float
        spacing between receptive field centers (as fraction of radius)
    min_ecc : float
        minimum eccentricity for gaussian rings [default: 1.]

    Returns
    -------
    mu : torch.Tensor
        kernel x-y coordinate centers with shape (n_kernels, 2) and dtype 
        torch.int32
    sigma : torch.Tensor
        kernel standard deviations with shape (n_kernels, 1)

    Examples
    --------
    # generate a "V3" lattice on a 200x200 image
    >>> img_shape = (200,200)
    >>> scale = 0.25
    >>> spacing = 0.15
    >>> min_ecc = 1.
    >>> mu, sigma = init_foveated_lattic(img_shape, scale, spacing, min_ecc)
    
    Notes
    -----
    sigma will always be >= 1. (pixel space)

    References
    ----------
    (Winawer & Horiguchi, 2015) https://archive.nyu.edu/handle/2451/33887
    """

    assert scale > 0.
    assert min_ecc > 0.

    # set max/minimum eccentricity
    max_ecc = np.max(tuple(img_shape))/2.

    assert min_ecc < max_ecc/(1. + scale)

    # get number of receptive fields in ring
    n_rf = np.floor(np.pi/scale)

    # get angular positions for each receptive field
    angles = 2. * np.pi * torch.linspace(0., 1., int(n_rf))
    x_mu = torch.cos(angles)
    y_mu = torch.sin(angles)

    # get base sigma
    base_sigma = torch.as_tensor((1. - spacing) * scale, dtype=torch.float32)

    # calculate distance between rings
    cf = np.cos((angles[0] - angles[1])/2.)
    cfPlusR2 = cf + np.square(scale)
    eFactor = (cfPlusR2 - np.sqrt(np.square(cfPlusR2) + np.square(cf) * (np.square(scale) - 1.)))/np.square(cf)

    # get rotation angle between each ring
    rot_angle = torch.as_tensor(np.pi * n_rf, dtype=torch.float32)
    x_mu_rot = torch.cos(rot_angle)*x_mu + torch.sin(rot_angle)*y_mu
    y_mu_rot = -torch.sin(rot_angle)*x_mu + torch.cos(rot_angle)*y_mu

    # append mu, sigma for each eccentricity
    ecc = max_ecc/(1. + scale)
    mu = []
    sigma = []
    cnt = 0
    while (ecc > min_ecc and cnt < 100):
        if np.mod(cnt, 2):
            mu.append(torch.stack([ecc*x_mu_rot, ecc*y_mu_rot], dim=-1))
        else:
            mu.append(torch.stack([ecc*x_mu, ecc*y_mu], dim=-1))
        sigma.append(torch.mul(ecc, base_sigma).repeat(mu[-1].shape[0]))
        ecc *= eFactor
        cnt += 1
    # set mu, sigma
    mu = torch.as_tensor(torch.cat(mu, dim=0), dtype=torch.float32)
    sigma = torch.cat(sigma, 0).unsqueeze(1)
    # add img_shape//2 to mu, set sigma to max(sigma, 1.)
    half_img_shape = torch.as_tensor(img_shape, dtype=mu.dtype).unsqueeze(0)/2
    return torch.add(mu, half_img_shape), torch.max(sigma, torch.ones_like(sigma))

def init_uniform_lattice(center, n_kernel_side, spacing, sigma_init=1.):
    """
    Creates a uniform lattice of kernel centers (mu) 
    and standard deviations (sigma)

    Parameters
    ----------
    center : tuple
        x-y coordinate for center of lattice
    n_kernel_side : int
        height/width of the lattice. n_kernels = n_kernel_side**2
    spacing : float
        spacing between receptive field centers
    sigma_init : float
        standard deviation initialization [default: 1.]

    Returns
    -------
    mu : torch.Tensor
        kernel x-y coordinate centers with shape (n_kernels, 2) and dtype 
        torch.int32
    sigma : torch.Tensor
        kernel standard deviations with shape (n_kernels, 1)
        
    Examples
    --------
    # Generate lattice of size 8x8 with 12 pixel spacing centered on a 
    # 200x200 image
    >>> center = (100,100)
    >>> n_kernel_side = 8
    >>> spacing = 12
    >>> sigma_init = 3.
    >>> mu, sigma = init_uniform_lattice(center, n_kernel_side, spacing, sigma_init)
    """
    
    if sigma_init < 1.:
        warnings.warn('sigma < 1 will result in sum(pdf) > 1.')
    cx, cy = center
    if n_kernel_side % 2 == 0:
        cx += np.floor(spacing/2)
        cy += np.floor(spacing/2)
    lattice_coord = torch.arange(n_kernel_side) - np.floor(n_kernel_side/2)
    # x-coodinates
    x = cx + spacing * lattice_coord
    x = x.repeat(n_kernel_side).reshape(n_kernel_side, n_kernel_side)
    # y-coordinates
    y = cy + spacing * lattice_coord
    y = y.repeat(n_kernel_side).reshape(n_kernel_side, n_kernel_side).t()
    # mu and sigma
    mu = torch.stack([x,y], dim=-1)
    mu = torch.as_tensor(mu.reshape(-1, 2), dtype=torch.float32)
    sigma = torch.ones((mu.shape[0],1), dtype=torch.float32) * sigma_init

    return mu, sigma

def init_tile_lattice(img_shape, rf_sizes, spacing=0):
    """
    Creates a tiled lattice of kernel centers (mu) 
    and standard deviations (sigma)
    
    Parameters
    ----------
    img_shape : tuple
        height and width of bottom-up input to pooling layer
    rf_sizes : list
        receptive field sizes to tile detection layer (see example)
    spacing : int
        spacing (if positive) or overlap (if negative) between receptive 
        field edges [default: 0]

    Returns
    -------
    mu : torch.Tensor
        kernel x-y coordinate centers with shape (n_kernels, 2) and dtype 
        torch.int32
    sigma : torch.Tensor
        kernel standard deviations with shape (n_kernels, 1)
        
    Examples
    --------
    # Generate lattice tiling a 200x200 image with 9 pixels between receptive
    # fields and standard deviations of 3 pixels
    >>> img_shape = (200,200)
    >>> rf_sizes = [3]
    >>> spacing = 9
    >>> mu, sigma = init_uniform_lattice(center, size, spacing)
    
    Notes
    -----
    If receptive field sizes do not fully cover image, rf_sizes[-1] will be 
    appended to rf_sizes until the image can be completely covered 
    (unless (rf_sizes[-1] + spacing) <= 0).
    """
    
    # ensure rf_sizes given
    assert len(rf_sizes) > 0

    # get image size
    img_h, img_w = img_shape
    
    # append rf_size[-1] if img not fully covered
    while (img_h - 2*np.sum(rf_sizes) - 2*len(rf_sizes)*spacing) > 0 and \
          (img_w - 2*np.sum(rf_sizes) - 2*len(rf_sizes)*spacing) > 0:
        if rf_sizes[-1] + spacing > 0:
            rf_sizes.extend([rf_sizes[-1]])
        else: # warn that image won't be covered
            warnings.warn("Image will not be fully covered by receptive fields.")
            break

    # for each RF size, get slice indices
    mu = []
    sigma = []
    for n in range(len(rf_sizes)):
        # get sum of previous rf sizes
        sum_rfs = np.sum(rf_sizes[:n], dtype='uint8') + spacing*n
        # get current image size for rf
        if (img_shape[0] - 2*sum_rfs) < rf_sizes[n] and (img_shape[1] - 2*sum_rfs) < rf_sizes[n]:
            break
        if (img_shape[0] - 2*sum_rfs) >= rf_sizes[n]:
            img_h = img_shape[0] - 2*sum_rfs
        if (img_shape[1] - 2*sum_rfs) >= rf_sizes[n]:
            img_w = img_shape[1] - 2*sum_rfs
        # init (i, j) mu indices
        i = []
        j = []
        # set left side slices
        i.extend([ii+rf_sizes[n]//2 for ii in np.arange(sum_rfs, img_h+sum_rfs, rf_sizes[n] + spacing)
                  if ii+rf_sizes[n] <= img_h+sum_rfs])
        j.extend([jj+rf_sizes[n]//2 for jj in np.repeat(sum_rfs, np.ceil(img_h/(rf_sizes[n]+spacing)))])
        # set bottom side slices
        i.extend([ii-rf_sizes[n]//2 for ii in np.repeat(img_h+sum_rfs, np.ceil(img_w/(rf_sizes[n]+spacing)))])
        j.extend([jj+rf_sizes[n]//2 for jj in np.arange(sum_rfs, img_w+sum_rfs, rf_sizes[n] + spacing)
                  if jj+rf_sizes[n] <= img_w+sum_rfs])
        # set right side slices
        i.extend([ii-rf_sizes[n]//2 for ii in np.arange(img_h+sum_rfs, sum_rfs, -rf_sizes[n] - spacing)
                  if ii-rf_sizes[n] >= sum_rfs])
        j.extend([jj-rf_sizes[n]//2 for jj in np.repeat(img_w+sum_rfs, np.ceil(img_h/(rf_sizes[n]+spacing)))])
        # set top side slices
        i.extend([ii+rf_sizes[n]//2 for ii in np.repeat(sum_rfs, np.ceil(img_w/(rf_sizes[n]+spacing)))])
        j.extend([jj-rf_sizes[n]//2 for jj in np.arange(img_w+sum_rfs, sum_rfs, -rf_sizes[n] - spacing)
                  if jj-rf_sizes[n] >= sum_rfs])
        # append if not already in mu
        for ii,jj in zip(i,j):
            if len(mu) == 0 or (ii,jj) not in mu:
                mu.append((ii,jj))
                sigma.append([rf_sizes[n]])
    return torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sigma, dtype=torch.float32)

def plot_size_ecc(mu, sigma, img_shape):
    ecc = torch.abs(mu - torch.as_tensor(img_shape, dtype=mu.dtype).unsqueeze(0)/2.)
    ecc = torch.max(ecc, dim=1)[0]
    plt.plot(ecc.numpy(), sigma.numpy())
    plt.show()

def make_kernel_lattice(kernels):
    if len(kernels.shape) == 3:
        #normalize each kernel to max 1, then max across kernels to show
        max_kerns = torch.as_tensor(np.max(kernels.numpy(), axis=(1,2), keepdims=True), 
                                dtype=kernels.dtype)
        norm_kerns = torch.div(kernels, max_kerns + 1e-6)
        out = torch.max(norm_kerns, dim=0)[0]

    elif len(kernels.shape) == 5:
        out = torch.squeeze(torch.max(kernels, dim=2)[0])

    return out.numpy()

def show_kernel_lattice(kernels):
    out = make_kernel_lattice(kernels)
    plt.imshow(out)
    plt.show()

if __name__ == '__main__':
    import doctest
    doctest.testmod()