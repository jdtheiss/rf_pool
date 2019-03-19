import numpy as np
import torch

def make_crowded_stimuli(target, flankers, spacing, background_size, axis=0., random=False):
    """
    makes a crowded stimulus: central target with evenly-spaced flankers

    Parameters
    ----------
    target: numpy.array
        the target image
    flankers: numpy.array_like
        list of flanker images
    spacing: int
        distance between target center and flanker
        center in pixels
    background_size: int or tuple
        size of the blank background in pixels
        if type is int, then assumes background is square
    axis: float
        initialization axis for flankers in radians
        [initialized as 0]
    random: bool
        flankers are placed in clockwise fashion if False
        [initialized as False].

    Returns
    -------
    stimuli: nump.array
        stimulus image with shape (background_size,)

    Examples
    --------
    >>>target = np.random.randn(28,28)
    >>>flankers = np.random(2,28,28)
    >>>make_crowded_stimuli(target, flankers, 2, 200, axis=0.) # 2-flanker horizontal layout
    >>>make_crowded_stimuli(target, flankers, 2, 200, axis=np.pi/2) # 2-flanker vertical layout
    """
    n_flank = len(flankers)
    target_size = int(target.shape[0])
    if type(background_size) is int:
        background_size = (background_size,)*2
    center = tuple(int(edge_size // 2 - target_size // 2) for edge_size in background_size)
    stimuli = np.zeros(background_size+(n_flank+1,))
    stimuli[center[0]:center[0] + target_size, center[1]:center[1] + target_size, 0] = target
    space_size = int(spacing)

    if n_flank != 0:
        theta = (2.*np.pi) / n_flank
        theta_shift = [axis + (theta * i) for i in range(n_flank)]
        x_shift = [int(space_size*np.cos(ang)) for ang in theta_shift]
        y_shift = [int(space_size*np.sin(ang)) for ang in theta_shift]
        if random:
            np.random.shuffle(flankers)

        flankers = np.array(flankers)
        f_h, f_w = flankers.shape[-2:]
        for i, flank in enumerate(flankers,1):
            stimuli[:f_h,:f_w,i] = flank
            stimuli[:,:,i] = np.roll(stimuli[:,:,i],
                                    (center[0]+y_shift[i-1], center[1]+x_shift[i-1]), (0,1))
    stimuli = np.max(stimuli, -1)

    return stimuli

def make_crowded_circles(n_flank, radius_range, **kwargs):
    """
    Makes a crowded stimulus with circle of random size

    Parameters
    ----------
    n_flank: int
        the number of flankers surrounding a single target
        decides the equally-spaced layout
    radius_range: tuple
        the range of radii to be randomly sampled [low, high)
    image_size: int
        the size of the circle image
    **kwargs: dict
        see make_crowded_stimuli

    Returns
    -------
    s: numpy.array
        the crowded stimulus
    target_radius: float
        the radius of the central circle
    mean_radius: float
        the average radius of all the circles
    """
    image_size = 2.*radius_range[1]

    radii = np.random.randint(low=radius_range[0], high=radius_range[1],
                              size=n_flank+1)
    target_radius = np.float32(radii[0])
    mean_radius = np.mean(radii, dtype='float32')

    circles = [make_circle(r, image_size) for r in radii]
    s = make_crowded_stimuli(circles[0], circles[1:n_flank+1], **kwargs)

    return s, target_radius, mean_radius

def make_circle(radius, image_size):
    c = int(image_size/2)
    xx, yy = np.mgrid[:image_size, :image_size]
    circle = (xx - c)**2 + (yy - c)**2
    return np.uint8(255*(circle < radius**2).astype(int))
