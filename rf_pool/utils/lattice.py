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
import imageio
import matplotlib.pyplot as plt

def multiply_gaussians(mu0, mu1, sigma0, sigma1):
    # reshape to (mu0_batch, 2, 1) (1, 2, mu1_batch)
    mu0 = torch.unsqueeze(mu0, -1)
    mu1 = torch.unsqueeze(mu1.t(), 0)
    sigma0 = torch.unsqueeze(sigma0, -1)
    sigma1 = torch.unsqueeze(sigma1.t(), 0)
    # compute multiplication of gaussians
    sigma0_2 = torch.pow(sigma0, 2)
    sigma1_2 = torch.pow(sigma1, 2)
    mu = (sigma1_2 * mu0 + sigma0_2 * mu1) / (sigma0_2 + sigma1_2)
    sigma = torch.sqrt((sigma0_2 * sigma1_2) / (sigma0_2 + sigma1_2))
    # return weighted combination based on mu1_batch size
    w = torch.ones(mu1.shape[-1], 1) / torch.tensor(mu1.shape[-1])
    mu = torch.matmul(mu, w).squeeze(-1)
    sigma = torch.matmul(sigma, w).squeeze(-1)
    return mu, sigma

def exp_kernel_2d(mu, sigma, xy):
    mu = mu.reshape(mu.shape + (1,1)).float()
    sigma = sigma.unsqueeze(-1).float()
    return torch.exp(-torch.sum((xy - mu)**2., dim=-3)/ (2*sigma**2))

def gaussian_kernel_2d(mu, sigma, xy):
    mu = mu.reshape(mu.shape + (1,1)).float()
    sigma = sigma.unsqueeze(-1).float()
    return (1./(2.*np.pi*sigma**2)) * torch.exp(-torch.sum((xy - mu)**2., dim=-3)/ (2*sigma**2))

def dog_kernel_2d(mu, sigma, ratio, xy):
    kernel_0 = gaussian_kernel_2d(mu, sigma, xy)
    kernel_1 = gaussian_kernel_2d(mu, ratio * sigma, xy)
    return kernel_0 - kernel_1

def mask_kernel_2d(mu, sigma, xy):
    kernels = exp_kernel_2d(mu, sigma, xy)
    # threshold at 1 std
    mask = torch.as_tensor(torch.ge(kernels, np.exp(-0.5)), dtype=kernels.dtype)
    with torch.no_grad():
        kernels_no_grad = torch.add(kernels, 1e-6)
    return torch.div(torch.mul(kernels, mask), kernels_no_grad)

def gaussian_field(priority_map):
    """
    """
    # get mu and sigma for map
    x = torch.arange(priority_map.shape[0])
    y = torch.arange(priority_map.shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0)
    # get mu, sigma from priority map
    map_mu = xy.reshape(2, -1).t()
    map_sigma = priority_map.reshape(-1, 1)
    # remove nonzero values
    keep_indices = map_sigma.nonzero()[:,0]
    if keep_indices.numel() == 0:
        return mu, sigma
    map_mu = map_mu[keep_indices].type(priority_map.dtype)
    map_sigma = 1. / map_sigma[keep_indices].type(priority_map.dtype)
    field = exp_kernel_2d(map_mu, map_sigma, xy.unsqueeze(0).float())
    return field

def gaussian_gradient(kernels):
    """
    """
    kernels = torch.sum(kernels, 0, keepdim=True)
    kernels = torch.unsqueeze(kernels, 0)
    # convolve with [-1,1] [-1;1]
    w = torch.ones(2,1,2,2)
    w[0,0,:,0] = -1.
    w[1,0,0,:] = -1.
    return torch.conv2d(kernels, w)[0]

def update_mu_sigma(mu, sigma, priority_map):
    """
    Returns updated mu and sigma after multiplication with a map of gaussian
    precision values

    Parameters
    ----------
    mu : torch.Tensor
        kernel centers with shape (n_kernels, 2)
    sigma : torch.Tensor
        kernel standard deviations with shape (n_kernels, 1)
    priority_map : torch.Tensor
        map of precisions to update mu, sigma with shape kernel_shape

    Returns
    -------
    new_mu : torch.Tensor
        updated kernel centers with shape (n_kernels, 2)
    new_sigma : torch.Tensor
        updated standard deviations with shape (n_kernels, 1)

    Examples
    --------
    >>> kernel_shape = torch.as_tensor((24,24), dtype=torch.float32)
    >>> mu, sigma = init_uniform_lattice(kernel_shape//2, 3, 8, 2)
    >>> priority_map = torch.rand(kernel_shape)
    >>> new_mu, new_sigma = update_mu_sigma(mu, sigma, priority_map)

    Notes
    -----
    Each location within priority_map should contain the precision associated
    with the gaussian at that location. Zero-valued locations will be ignored.
    """
    # get mu and sigma for map
    x = torch.arange(priority_map.shape[0])
    y = torch.arange(priority_map.shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0)
    # get mu, sigma from priority map
    map_mu = xy.reshape(2, -1).t()
    map_sigma = priority_map.reshape(-1, 1)
    # remove nonzero values
    keep_indices = map_sigma.nonzero()[:,0]
    if keep_indices.numel() == 0:
        return mu, sigma
    map_mu = map_mu[keep_indices].type(priority_map.dtype)
    map_sigma = 1. / map_sigma[keep_indices].type(priority_map.dtype)
    return multiply_gaussians(mu, map_mu, sigma, map_sigma)

def mu_mask(mu, kernel_shape):
    """
    Returns a tensor of kernels with value = 1 at each mu location

    Parameters
    ----------
    mu : torch.Tensor
        kernel centers with shape (n_kernels, 2)
    kernel_shape : tuple
        shape of input feature map

    Returns
    -------
    kernels : torch.Tensor
        output kernels with shape
        mu.shape[:-1] + (kernel_shape[0], kernel_shape[1])

    Examples
    --------
    # Create tensor mask of 10 kernels with random centers.
    >>> mu = torch.rand(10, 2)*200
    >>> kernel_shape = (200,200)
    >>> kernels = mu_mask(mu, kernel_shape)
    """#TODO: use mask_kernel_lattice instead

    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0)
    # reshape mu to same dimension as xy
    mu = mu.reshape(mu.shape + (1,1))
    # return 1s at mu locations
    return torch.as_tensor(torch.prod(torch.eq(mu.int(), xy.int()), -3), dtype=mu.dtype)

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

    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return gaussian_kernel_2d(mu, sigma, xy)

def dog_kernel_lattice(mu, sigma, kernel_shape, ratio=4.):
    """
    Returns a tensor of difference of gaussian (DoG) kernels

    Parameters
    ----------
    mu : torch.Tensor
        kernel centers with shape (n_kernels, 2)
    sigma : torch.Tensor
        kernel standard deviations with shape (n_kernels, 1)
    kernel_shape : tuple
        shape of input feature map
    ratio : float or torch.Tensor
        ratio of second gaussian to first gaussian [default: 4]

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
    >>> ratio = 4.
    >>> mu = mu * torch.as_tensor(kernel_shape, dtype=mu.dtype)
    >>> kernels = dog_kernel_lattice(mu, sigma, kernel_shape, ratio)
    """

    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return dog_kernel_2d(mu, sigma, ratio, xy)

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

    # create the coordinates input to kernel function
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()

    return mask_kernel_2d(mu, sigma, xy)

def init_foveated_lattice(img_shape, scale, spacing, std=1., n_rf=None, n_rings=None,
                          min_ecc=1., offset=[0.,0.], rotate_rings=True, rotate=0.):
    """
    Creates a foveated lattice of kernel centers (mu) and
    stantard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    scale : float
        rate at which receptive field radius scales with eccentricity
    spacing : float
        spacing between receptive field centers (as fraction of radius)
    std : float
        standard deviation multiplier [default: 1.]
    min_ecc : float
        minimum eccentricity for gaussian rings [default: 1.]
    rotate_rings : bool
        rotate receptive fields between rings [default: False]
    rotate : float
        rotation (in radians, counterclockwise) to apply to the entire array

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

    # get number of receptive fields in ring
    if n_rf is None:
        n_rf = np.floor(np.pi/scale)
    else:
        n_rf = n_rf + 1

    # get angular positions for each receptive field
    angles = 2. * np.pi * torch.linspace(0., 1., int(n_rf))[:-1]
    x_mu = torch.cos(angles)
    y_mu = torch.sin(angles)

    # get base sigma
    base_sigma = torch.as_tensor((1. - spacing) * scale, dtype=torch.float32)

    # eccentricity factor
    eFactor = (1. + scale) / (1. - scale)

    # get rotation angle between each ring
    if rotate_rings:
        rot_angle = torch.as_tensor(np.pi / n_rf, dtype=torch.float32)
        x_mu_rot = torch.cos(rot_angle)*x_mu + torch.sin(rot_angle)*y_mu
        y_mu_rot = -torch.sin(rot_angle)*x_mu + torch.cos(rot_angle)*y_mu

    # append mu, sigma for each eccentricity
    ecc = min_ecc * eFactor
    mu = []
    sigma = []
    for n in range(n_rings):
        if rotate_rings and np.mod(n, 2):
            mu.append(torch.stack([ecc*x_mu_rot, ecc*y_mu_rot], dim=-1))
        else:
            mu.append(torch.stack([ecc*x_mu, ecc*y_mu], dim=-1))
        sigma.append(torch.mul(ecc, base_sigma).repeat(mu[-1].shape[0]))
        ecc *= eFactor
    # set mu, sigma
    mu = torch.as_tensor(torch.cat(mu, dim=0), dtype=torch.float32)
    sigma = torch.cat(sigma, 0).unsqueeze(1)
    # rotate mu
    rx = np.cos(rotate) * mu[:,0] - np.sin(rotate) * mu[:,1]
    ry = np.sin(rotate) * mu[:,0] + np.cos(rotate) * mu[:,1]
    mu = torch.stack([rx,ry], dim=-1)
    # set offset of mu
    mu = mu + torch.as_tensor(offset, dtype=mu.dtype)
    # add img_shape//2 to mu
    half_img_shape = torch.as_tensor(img_shape, dtype=mu.dtype).unsqueeze(0)/2
    mu = torch.add(mu, half_img_shape)
    # multiply by std and set sigma to max(sigma, 1.)
    sigma = torch.mul(sigma, std)
    sigma = torch.max(sigma, 0.5 * torch.ones_like(sigma))
    # remove mu, sigma outside image frame
    remove_idx = np.where(torch.sum(torch.lt(mu + 2 * sigma, 0.), -1))[0]
    mu = torch.as_tensor([m for i, m in enumerate(mu.tolist()) if i not in remove_idx])
    sigma = torch.as_tensor([s for i, s in enumerate(sigma.tolist()) if i not in remove_idx])
    return mu, sigma

def init_uniform_lattice(center, n_kernel_side, spacing, sigma_init=1.,
                         rotate=0.):
    """
    Creates a uniform lattice of kernel centers (mu)
    and standard deviations (sigma)

    Parameters
    ----------
    center : tuple
        x-y coordinate for center of lattice
    n_kernel_side : tuple or int
        height/width of the lattice. n_kernels = np.prod(n_kernel_side)
    spacing : float
        spacing between receptive field centers
    sigma_init : float
        standard deviation initialization [default: 1.]
    rotate : float
        rotation (in radians, counterclockwise) to apply to the entire array

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
    if type(n_kernel_side) is int:
        n_kernel_side = (n_kernel_side,)*2
    lattice_coord_x = torch.arange(n_kernel_side[0]) - np.floor(n_kernel_side[0]/2)
    lattice_coord_y = torch.arange(n_kernel_side[1]) - np.floor(n_kernel_side[1]/2)
    # x-coodinates, y-coordinates
    x = spacing * lattice_coord_x.float()
    y = spacing * lattice_coord_y.float()
    # update based on kernel side
    if n_kernel_side[0] % 2 == 0:
        x = x + np.floor(spacing/2)
    if n_kernel_side[1] % 2 == 0:
        y = y + np.floor(spacing/2)
    # repeat
    x = x.repeat(n_kernel_side[1]).reshape(n_kernel_side).flatten()
    y = y.repeat(n_kernel_side[0]).reshape(n_kernel_side).t().flatten()
    # rotate
    rx = np.cos(rotate) * x - np.sin(rotate) * y
    ry = np.sin(rotate) * x + np.cos(rotate) * y
    # center
    x = rx + cx
    y = ry + cy
    # mu and sigma
    mu = torch.stack([x,y], dim=-1)
    mu = torch.as_tensor(mu, dtype=torch.float32)
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

def make_lattice_gif(filename, kernels):
    # save kernels as gif
    assert kernels.ndimension() >= 3
    if not filename.endswith('.gif'):
        filename += '.gif'
    # flat kernels to show entire lattice for 5 frames
    flat_kernels = make_kernel_lattice(kernels).flatten(0, -3).numpy()
    flat_kernels = np.repeat(flat_kernels, 5, axis=0)
    # flatten if greater than 3 dims, then divide by max and set to uint8
    kernels = kernels.flatten(0, -3).detach().numpy()
    kernels = np.divide(kernels, np.max(kernels, axis=(-2,-1), keepdims=True))
    kernels = np.concatenate([flat_kernels, kernels], 0)
    kernels = np.multiply(kernels, 255).astype('uint8')
    # save gif
    imageio.mimsave(filename, kernels)

def make_kernel_lattice(kernels):
    #normalize each kernel to max 1, then max across kernels to show
    max_kerns = torch.as_tensor(np.max(kernels.detach().numpy(), axis=(-2,-1), keepdims=True),
                                dtype=kernels.dtype)
    norm_kerns = torch.div(kernels, max_kerns + 1e-6)
    out = torch.max(norm_kerns, dim=-3)[0]
    # ensure ndim == 4
    new_dims = 4 - out.ndimension()
    assert new_dims >= 0, ('number of dimensions must be <= 5')
    for n in range(new_dims):
        out = torch.unsqueeze(out, 0)

    return out.detach()

def show_kernel_lattice(lattices, x=None, figsize=(5, 5), cmap=None):
    """
    #TODO:WRITEME
    """
    # check that lattices is list or torch.Tensor
    assert type(lattices) in [list, torch.Tensor], (
        'lattices type must be list or torch.Tensor'
    )
    if type(lattices) is not list:
        lattices = [lattices]
    # get number of lattices
    n_lattices = 0
    for i in range(len(lattices)):
        assert lattices[i].ndimension() == 4, (
            'lattices must have 4 dimensions'
        )
        n_lattices += lattices[i].shape[1]
    # transpose x if not None
    if x is not None:
        x = torch.squeeze(x.permute(0,2,3,1), -1).numpy()
        x = x - np.min(x, axis=(1,2), keepdims=True)
        x = x / np.max(x, axis=(1,2), keepdims=True)
    # init figure, axes
    n_rows = np.max([l.shape[0] for l in lattices])
    n_cols = n_lattices
    if x is not None:
        n_rows = np.maximum(n_rows, x.shape[0])
        n_cols += 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = np.reshape(ax, (n_rows, n_cols))
    # plot data and lattices
    for r in range(n_rows):
        # show data
        if x is not None:
            ax[r,0].imshow(x[r], cmap)
            c = 1
        else:
            c = 0
        # show lattices
        for l in range(len(lattices)):
            for ll in range(lattices[l].shape[1]):
                batch_id = np.minimum(r, lattices[l].shape[0]-1)
                ax[r,c].imshow(lattices[l][batch_id,ll], cmap)
                c += 1
    plt.show()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
