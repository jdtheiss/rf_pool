"""
Utilities for creating square receptive field indices for use with rf.pool 
function.

Examples
--------
# create square receptive fields of different sizes tiling detection layer
>>> img_shape = (18,18)
>>> rf_sizes = [4,3,2]
>>> rfs = square_kernel_lattice(img_shape, rf_sizes, stride=0)

# show lattice
>>> show_kernel_lattice(img_shape, rfs)

# plot size vs. eccentricity for receptive fields in lattice
>>> plot_size_ecc(img_shape, rfs)
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch

def square_kernel_lattice(img_shape, rf_sizes, stride=0):
    """
    Make square receptive field indices for rf_pool
    
    Parameters
    ----------
    img_shape : tuple
        height and width of bottom-up input to pooling layer
    rf_sizes : list
        receptive field sizes to tile detection layer (see example)
    stride : int
        spacing (if positive) or overlap (if negative) between receptive 
        fields [default: 0]

    Returns
    -------
    rfs : list of tuples
        slice indices for each receptive field 
        (i.e. [(slice(start, end), slice(start, end))])

    Examples
    --------
    # creates indices for 32 receptive fields of size 4x4, 3x3, 2x2 tiling 
    # image from outer-most indices to inner-most indices
    >>> img_shape = (18,18)
    >>> rf_sizes = [4,3,2]
    >>> rfs = make_RFs(img_shape, rf_sizes, stride=0)

    Notes
    -----
    If receptive field sizes do not fully cover image, rf_sizes[-1] will be 
    appended to rf_sizes until the image can be completely covered 
    (unless (rf_sizes[-1] + stride) <= 0).
    """

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
    rfs = []
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
            if len(rfs) == 0 or (ii,jj) not in rfs:
                rfs.append((ii,jj))
    return rfs

def plot_size_ecc(img_shape, rfs):
    """
    Plot receptive field size as function of eccentricity
    
    Parameters
    ----------
    img_shape : tuple
        image shape to plot receptive fields
    rfs : list of tuples
        list of receptive field slice indices 
        [(slice(start, end), slice(start, end))]

    Returns
    -------
    None
    """

    # init sizes, locs
    sizes = []
    locs = []
    for rf in rfs:
        # get size and location
        sz = np.abs(rf[0].start - rf[0].stop)
        l = np.abs(img_shape[0]/2 - (rf[0].start + sz/2))
        # if already in sizes, get previous location
        if sz in sizes:
            prev_loc = locs[np.where(np.asarray(sizes) == sz)[0][0]]
        else:
            prev_loc = -1
        # append size, location
        if l > prev_loc:
            sizes.append(sz)
            locs.append(l)

    # plot size vs. eccentricy
    plt.plot(locs, sizes)
    plt.show()
    
def show_kernel_lattice(img_shape, rfs):
    # init image
    img = np.zeros(img_shape)

    # set each RF to index number
    for i, rf in enumerate(rfs):
        img[rf[0],rf[1]] += (i + 1)

    # show image
    plt.imshow(img, cmap='gray', vmin=-len(rfs), vmax=len(rfs))
    plt.show()

if __name__ == '__main__':
    import doctest
    doctest.testmod()