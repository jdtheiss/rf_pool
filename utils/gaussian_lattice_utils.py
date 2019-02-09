import torch
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel_2d(mu, sigma, xy):
    return (1./(2.*np.pi*sigma**2)) * torch.exp(-torch.sum((xy - mu)**2., dim=2)/ (2*sigma**2))

def gaussian_kernel_lattice(mu, sigma, kernel_size):
    """
    Outputs a tensor of gaussian kernels of size (kernel_size X kernel_size X num_kernels)

    Inputs:
    -------
    mu: Tensor (2,num_kernels)
        xy-coordinate of gaussian centers
    sigma: Tensor (1, num_kernels)
        standard deviation of gaussians
    kernel_size: int
        size of input feature map

    Outputs:
    -------
    gk_kernels: Tensor (batch_size, kernel_size, kernel_size, num_kernels))

    """
    assert mu.shape[-1] == sigma.shape[-1]
    num_kernels = mu.shape[-1]

    mu = mu.view([1,1,2,num_kernels]).float()
    sigma = sigma.view([1,1,1,num_kernels]).float()

    # create the input to the gaussian
    x_coord = torch.arange(kernel_size)
    x = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y = x.t()
    xy = torch.stack([x,y], dim=-1)
    xy = torch.stack([xy]*mu.shape[-1], dim=-1).float()

    gk_kernels = gaussian_kernel_2d(mu, sigma, xy)

    return torch.squeeze(gk_kernels)

def init_foveated_lattice(img_shape, scale, spacing, min_ecc=1.):
    """
    Creates a foveated lattice of guassian kernel centers (mu) and
    stantard deviations (sigma)

    Parameters
    ----------
    img_shape : tuple
        shape of image
    scale : float
        rate at which receptive field size scales with eccentricity
    spacing : float
        spacing between gaussian centers (as fraction of radius)
    min_ecc : float
        minimum eccentricity for gaussian rings [default: 1.]

    Returns
    -------
    mu : torch.Tensor
        kernel centers with shape (2, n_kernels)
    sigma : torch.Tensor
        kernel standard deviations with shape (1, n_kernels)

    Examples
    --------
    # creates "V3" lattice

    >>> img_shape = (80,80)
    >>> scale = 0.25
    >>> spacing = 0.15
    >>> min_ecc = 1.
    >>> mu, sigma = init_foveated_lattic(img_shape, scale, spacing, min_ecc)

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
            mu.append(torch.stack([ecc*x_mu_rot, ecc*y_mu_rot]))
        else:
            mu.append(torch.stack([ecc*x_mu, ecc*y_mu]))
        sigma.append(torch.mul(ecc, base_sigma).repeat(mu[-1].shape[1]))
        ecc *= eFactor
        cnt += 1
    return torch.cat(mu, 1), torch.cat(sigma, 0).unsqueeze(0)

def init_uniform_lattice(center, size, spacing, sigma_init):
    """
    Creates a uniform lattice of guassian kernel centers(mu) and stantard deviations(sigma)

    Inputs:
    -------
    center: tuple (x,y)
        x-y coordinate for center of lattice
    size: int
        size of the lattice. tota number of filters = size**2
    spacing: int
        spacing between gaussian centers
    sigma_init:
        standard deviation initialization

    Outputs:
    -------
    mu: Tensor (2, size**2)
        tensor containg the kernel centers
    sigma: Tensor (1, size**2)
        tensor containing the kernel standard deviations

    """
    cx, cy = center
    if size % 2 == 0:
        cx += np.floor(spacing/2)
        cy += np.floor(spacing/2)
    lattice_coord = torch.arange(size) - np.floor(size/2)
    # x-coodinates
    x = cx + spacing * lattice_coord
    x = x.repeat(size).view(size, size)
    # y-coordinates
    y = cy + spacing * lattice_coord
    y = y.repeat(size).view(size, size).t()
    # mu and sigma
    mu = torch.stack([x,y], dim=-1)
    mu = mu.view(-1, 2).t()
    sigma = torch.ones([1, mu.shape[-1]]) * sigma_init

    return mu, sigma

def show_kernel_lattice(kerns):
    # normalize each kernel to max 1, then max across kernels to show
    norm_kerns = torch.div(kerns, torch.as_tensor(np.max(kerns.numpy(), axis=(0,1)), dtype=kerns.dtype) + 1e-6)
    out = torch.max(norm_kerns, dim=-1)[0]
    plt.imshow(out.numpy())
    plt.show()

if __name__ == '__main__':
    #test by generating a lattice of size 8x8 with 12 pixel spacing centered on a 200x200 image
    kernel_size = 200
    center = [kernel_size/2.]*2
    size = 8
    spacing = 12.
    sigma_init = 3.

    mu, sigma = init_uniform_lattice(center, size, spacing, sigma_init)
    g_kern = gaussian_kernel_lattice(mu, sigma, kernel_size)
    show_kernel_lattice(g_kern.numpy())
