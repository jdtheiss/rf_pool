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
    fig, ax = plt.subplots(1, 1)
    out = np.sum(kerns, axis=-1)
    ax.imshow(out)
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