import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Function

def fwhm2sigma(fwhm):
    return fwhm / (2. * np.sqrt(2. * np.log(2)))

def sigma2fwhm(sigma):
    return 2. * np.sqrt(2. * np.log(2)) * sigma

def multiply_gaussians(mu0, sigma0, mu1, sigma1, weight=None):
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
    if weight is None:
        weight = torch.ones(mu1.shape[-1], 1) / torch.tensor(mu1.shape[-1])
        weight = weight.to(mu1)
    mu = torch.matmul(mu, weight.reshape(-1, 1)).squeeze(-1)
    sigma = torch.matmul(sigma, weight.reshape(-1, 1)).squeeze(-1)
    return mu, sigma

def cortical_dist(mu, scale_rate, beta=0.):
    r = torch.sqrt(torch.sum(torch.pow(mu, 2), 1))
    alpha = 1./torch.log((1. + scale_rate) / (1. - scale_rate))
    return alpha * torch.log(r) + beta

def cortical_xy(mu, scale_rate, rot_angle, beta=0., ref_axis=0.):
    theta = torch.atan2(*mu.t()) - ref_axis
    r = cortical_dist(mu, scale_rate, beta=beta)
    y = r * torch.sin((theta / rot_angle) / r + ref_axis)
    x = r * torch.cos((theta / rot_angle) / r + ref_axis)
    return torch.stack([y, x], -1)

def estimate_mu_sigma(kernel, mu=None):
    # get mu for each pixel in kernel
    if mu is None:
        assert kernel.ndimension() >= 2
        x = torch.arange(kernel.shape[-1]).float()
        y = torch.arange(kernel.shape[-2]).float()
        mu = torch.stack([xy.flatten() for xy in torch.meshgrid(x, y)], dim=-1)
    # get weights
    w = kernel.reshape(-1, mu.shape[0]).clone()
    w = w - w.min(-1, keepdim=True)[0]
    w = w / w.sum(-1, keepdim=True)
    # get expectation of mu, cov for each Gaussian in kernel
    m = torch.matmul(w, mu)
    m_dev = m.unsqueeze(1) - mu
    cov = torch.matmul(m_dev.transpose(-1,-2), torch.mul(w.unsqueeze(-1), m_dev))
    return m, cov

def get_xy_coords(kernel_shape):
    x = torch.arange(kernel_shape[0])
    y = torch.arange(kernel_shape[1])
    xy = torch.stack(torch.meshgrid(x, y), dim=0).unsqueeze(0).float()
    # add 0.5 to coords to account for middle of pixel
    return xy + 0.5

def gaussian_kernel_2d(mu, sigma, xy):
    mu = mu.reshape(mu.shape + (1,1)).float()
    sigma = sigma.unsqueeze(-1).float()
    return torch.exp(-torch.sum((xy - mu)**2., dim=-3) / (2*sigma**2))

def normal_kernel_2d(mu, sigma, xy):
    a = 1./(2.*np.pi*sigma**2)
    return a.reshape(-1,1,1) * gaussian_kernel_2d(mu, sigma, xy)

def dog_kernel_2d(mu, sigma, ratio, xy):
    kernel_0 = gaussian_kernel_2d(mu, sigma, xy)
    kernel_1 = gaussian_kernel_2d(mu, ratio * sigma, xy)
    return kernel_0 - kernel_1

def mask_kernel_2d(mu, sigma, xy):
    kernels = gaussian_kernel_2d(mu, sigma, xy)
    output = torch.as_tensor(torch.ge(kernels, np.exp(-0.5))).to(mu)
    # ensure at least one pixel (if in image) is set to 1.
    mu = mu.reshape(mu.shape + (1,1))
    d = torch.eq(torch.sum(torch.pow(torch.floor(mu) - (xy - 0.5), 2), 1), 0.)
    output = torch.max(output, d.type(kernels.dtype))
    return _MaskGrad.apply(kernels, output)

class _MaskGrad(Function):
    @staticmethod
    def forward(ctx, kernels, output):
        ctx.save_for_backward(kernels)
        ctx.output = output
        output.requires_grad_(kernels.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernels = ctx.saved_tensors[0]
        with torch.no_grad():
            grad_input = torch.mul(grad_output, ctx.output)
        return grad_input, None

def gaussian_field(priority_map):
    """
    """
    # get mu and sigma for map
    xy = get_xy_coords(priority_map.shape[:2])
    # get mu, sigma from priority map
    map_mu = xy.reshape(2, -1).t()
    map_sigma = priority_map.reshape(-1, 1)
    # remove nonzero values
    keep_indices = map_sigma.nonzero()[:,0]
    if keep_indices.numel() == 0:
        return
    map_mu = map_mu[keep_indices].type(priority_map.dtype)
    map_sigma = 1. / map_sigma[keep_indices].type(priority_map.dtype)
    field = gaussian_kernel_2d(map_mu, map_sigma, xy.unsqueeze(0).float())
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

def apply_attentional_field(mu, sigma, priority_map):
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
    >>> new_mu, new_sigma = apply_attentional_field(mu, sigma, priority_map)

    Notes
    -----
    Each location within priority_map should contain the precision associated
    with the gaussian at that location. Zero-valued locations will be ignored.
    """
    # get mu and sigma for map
    xy = get_xy_coords(priority_map.shape[:2]).to(priority_map)
    # get mu, sigma from priority map
    map_mu = xy.reshape(2, -1).t()
    map_sigma = priority_map.reshape(-1, 1)
    # remove nonzero values
    keep_indices = map_sigma.nonzero()[:,0]
    if keep_indices.numel() == 0:
        return mu, sigma
    map_mu = map_mu[keep_indices]
    map_sigma = 1. / map_sigma[keep_indices]
    return multiply_gaussians(mu, sigma, map_mu, map_sigma)

def gaussian_kernel_lattice(mu, sigma, kernel_shape):
    """
    Returns a tensor of gaussian kernels with peak value = 1

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
    xy = get_xy_coords(kernel_shape).to(mu)

    return gaussian_kernel_2d(mu, sigma, xy)

def normal_kernel_lattice(mu, sigma, kernel_shape):
    """
    Returns a tensor of normal distribution kernels
    (peak = `1./(2.*np.pi*sigma**2)`)

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
    >>> kernels = normal_kernel_lattice(mu, sigma, kernel_shape)
    """
    # create the coordinates input to kernel function
    xy = get_xy_coords(kernel_shape).to(mu)

    return normal_kernel_2d(mu, sigma, xy)

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
    xy = get_xy_coords(kernel_shape).to(mu)

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
    xy = get_xy_coords(kernel_shape).to(mu)

    return mask_kernel_2d(mu, sigma, xy)

def init_foveated_lattice(img_shape, scale, n_rings, spacing=0., std=1.,
                          n_rf=None, offset=[0.,0.], min_ecc=0.5,
                          rotate_rings=True, rotate=0., keep_all_RFs=False):
    """
    Creates a foveated lattice of kernel centers (mu) and
    stantard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    scale : float
        rate at which receptive field radius scales with eccentricity
    n_rings : int
        number of concentric rings in foveated array
    spacing : float
        spacing between receptive field centers (as fraction of radius)
    std : float
        standard deviation multiplier [default: 1.]
    n_rf : int
        number of RFs per ring [default: None, set to np.pi / scale]
    offset : list of floats
        (x,y) offset from center of image [default: [0.,0.]]
    min_ecc : float
        minimum eccentricity for gaussian rings [default: 1.]
    rotate_rings : bool
        rotate receptive fields between rings [default: False]
    rotate : float
        rotation (in radians, counterclockwise) to apply to the entire array
    keep_all_RFs : boolean
        True/False keep all RFs regardless of whether they are fully contained
        within the image space
        [default: False, remove RFs 1 sigma outside image space]

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
    # multiply by std
    sigma = torch.mul(sigma, std)
    # check if mu + sigma (along radial direction from fovea) is in image
    r = torch.sqrt(torch.sum(torch.pow(mu, 2), 1))
    r_sigma = r - sigma.flatten()
    theta = torch.atan2(*mu.t())
    h_w = torch.mul(torch.stack([torch.sin(theta), torch.cos(theta)], 1),
                    r_sigma.reshape(-1,1))
    # remove mu, sigma outside image frame
    center = torch.as_tensor(img_shape, dtype=mu.dtype).unsqueeze(0) / 2.
    keep_idx = torch.prod(torch.lt(torch.abs(h_w), center), -1).bool()
    # add img_shape//2 to mu
    mu = torch.add(mu, center)
    if keep_all_RFs:
        return mu, sigma
    return mu[keep_idx], sigma[keep_idx]

def init_uniform_lattice(img_shape, n_kernels, spacing, sigma_init=1.,
                         offset=[0.,0.], rotate=0.):
    """
    Creates a uniform lattice of kernel centers (mu)
    and standard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    n_kernels : tuple or int
        number of kernels along height/width of the lattice
    spacing : float
        spacing between receptive field centers
    sigma_init : float
        standard deviation initialization [default: 1.]
    offset : list of floats
        (x,y) offset from center of image [default: [0.,0.]]
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
    >>> img_shape = (200,200)
    >>> n_kernels = 8
    >>> spacing = 12
    >>> sigma_init = 3.
    >>> mu, sigma = init_uniform_lattice(img_shape, n_kernels, spacing,
                                         sigma_init)
    """
    cx, cy = np.array(img_shape) / 2. + np.array(offset)
    if type(n_kernels) is int:
        n_kernels = (n_kernels,)*2
    lattice_coord_x = torch.arange(n_kernels[0]) - np.floor(n_kernels[0]/2)
    lattice_coord_y = torch.arange(n_kernels[1]) - np.floor(n_kernels[1]/2)
    # x-coodinates, y-coordinates
    x = spacing * lattice_coord_x.float()
    y = spacing * lattice_coord_y.float()
    # update based on kernel side
    if n_kernels[0] % 2 == 0:
        x = x + np.floor(spacing / 2.)
    if n_kernels[1] % 2 == 0:
        y = y + np.floor(spacing / 2.)
    # repeat
    x = x.repeat(n_kernels[1]).reshape(n_kernels).flatten()
    y = y.repeat(n_kernels[0]).reshape(n_kernels).t().flatten()
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

def init_hexagon_lattice(img_shape, n_kernels, spacing, sigma_init=1.,
                         offset=[0.,0.], rotate=0.):
    """
    Creates a uniform hexagon lattice of kernel centers (mu)
    and standard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    n_kernels : tuple or int
        number of kernels along height/width of the lattice
    spacing : float
        spacing between receptive field centers
    sigma_init : float
        standard deviation initialization [default: 1.]
    offset : list of floats
        (x,y) offset from center of image [default: [0.,0.]]
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
    >>> img_shape = (200,200)
    >>> n_kernels = 8
    >>> spacing = 12
    >>> sigma_init = 3.
    >>> mu, sigma = init_hexagon_lattice(img_shape, n_kernels, spacing,
                                         sigma_init)
    """
    cx, cy = np.array(img_shape) / 2. + np.array(offset)
    if type(n_kernels) is int:
        n_kernels = (n_kernels,)*2
    # get uniform lattice
    mu, sigma = init_uniform_lattice((0.,0.), n_kernels, 1., sigma_init)
    # update based on kernel side
    if n_kernels[0] % 2 == 1:
        mu[:,0] -= 0.5
    if n_kernels[1] % 2 == 0:
        mu[:,1] += 0.5
    # scale based on hexagon distances
    spacing = spacing / 2. # make edge-to-edge center-to-center spacing
    mu[:,1] *= np.sqrt(3.) * spacing
    mu[:,0] *= 2. * spacing
    for n in range(n_kernels[0]):
        mu[n::n_kernels[0]*2,0] += spacing
    # rotate
    rx = np.cos(rotate) * mu[:,0] - np.sin(rotate) * mu[:,1]
    ry = np.sin(rotate) * mu[:,0] + np.cos(rotate) * mu[:,1]
    # center
    mu[:,0] = rx + cx
    mu[:,1] = ry + cy

    return mu, sigma

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
