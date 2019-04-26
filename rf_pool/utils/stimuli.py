import numpy as np
import torch

def make_crowded_stimuli(target, flankers, spacing, background_size, axis=0., random=False):
    """
    makes a crowded stimulus: central target with evenly-spaced flankers

    Parameters
    ----------
    target : numpy.ndarray
        the target image
    flankers : numpy.array_like
        list of flanker images
    spacing : int
        distance between target center and flanker
        center in pixels
    background_size : int or tuple
        size of the blank background in pixels
        if type is int, then assumes background is square
    axis : float
        initialization axis for flankers in radians
        [initialized as 0]
    random : bool
        flankers are placed in clockwise fashion if False
        [initialized as False].

    Returns
    -------
    stimulus : numpy.ndarray
        stimulus image with shape (background_size,background_size)

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

def make_search_stimuli(target, distractors, background_size, target_loc=[0,0],
                        distractor_locs=[], scramble=False, background_image=None):
    """
    makes a visual search array with target and distractors

    Parameters
    ----------
    target : numpy.ndarray
        target stimulus in visual search task
    distractors : list or numpy.ndarray
        distractor stimuli in visual search task
    spacing : int or tuple
        minimum spacing between centers of targets and distractors
        if tuple, (minimum spacing, maximum spacing)
    background_size : int or tuple
        size of visual search array
    target_loc : list of ints, optional
        height, width locations for target center (in normalized units (-1,1))
        [default: [0,0]] (center of image)
    distractor_locs : lists of ints or numpy.ndarray, optional
        height, width locations for each distractor center (in normalized units)
        [default: []] (uniform randomly selected from (-1, 1))
    scramble : bool, optional
        if True, distractor images are scrambled [default: False]
    background_image : numpy.ndarray, optional
        stimulus background image into which targets/distractors are insert
        note: background_size will be set to background_image.shape

    Returns
    -------
    stimulus : numpy.ndarray
        stimulus image with shape (background_size,background_size)

    Examples
    --------
    #TODO
    """
    # set background_size, background_image
    if type(background_size) is int:
        background_size = (background_size,)*2
    if background_image:
        background_size = background_image.shape
    else:
        background_image = np.zeros(background_size)

    # set target in background_image at location
    stimulus = insert_image(background_image, target, target_loc)

    # set distractors in stimulus with spacing
    for i, distractor in enumerate(distractors):
        # select random location with given spacing
        if len(distractor_locs) > 0:
            loc_i = distractor_locs[i]
        else:
            loc_i = 2. * np.random.rand(2) - 1.
        # scramble image
        if scramble:
            distractor = scramble_image(distractor)
        # insert to stimulus
        stimulus = insert_image(stimulus, distractor, loc_i)

    return stimulus

def scramble_image(image): #TODO:needs to better scramble
    """
    #TODO:WRITEME
    """
    # get cropped image
    cropped_indices = image.nonzero()
    min_hw, max_hw = [np.min(cropped_indices, 1), np.max(cropped_indices, 1)]
    cropped_image = image[min_hw[0]:max_hw[0], min_hw[1]:max_hw[1]]
    # rotation and flip image randomly
    if np.random.randint(2):
        cropped_image = np.rot90(cropped_image)
    if np.random.randint(2):
        cropped_image = np.flipud(cropped_image)
    # get image height, width
    ih, iw = image.shape
    ch, cw = cropped_image.shape
    # set image patches randomly
    n_h, n_w = [np.random.randint(ch//4, ch), np.random.randint(cw//4, cw)]
    x, y = [ih // 2 - n_h // 2, iw // 2 - n_w // 2] #[np.random.randint(ih-n_h), np.random.randint(iw-n_w)]
    i, j = [np.random.randint(ch-n_h), np.random.randint(cw-n_w)]
    output = np.zeros_like(image)
    output[x:x+n_h, y:y+n_w] = cropped_image[i:i+n_h, j:j+n_w]
    return output

def insert_image(stimulus, image, loc):
    """
    insert image into stimulus at normalized center location

    Parameters
    ----------
    stimulus : numpy.ndarray
        array to insert image into
    image : numpy.ndarray
        array to be inserted into stimulus with shape <= stimulus.shape
    loc : list of ints
        normalized (-1, 1) center location for image to be inserted into stimulus

    Returns
    -------
    stimulus : numpy.ndarray
        stimulus with image inserted at location with overlapping values set to
        nonzero image values

    Examples
    --------
    >>> output = insert_image(np.zeros((4,4)), np.ones((2,2)), [-0.5,-0.5])
    >>> output
    array([[1., 1., 0., 0.],
           [1., 1., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

    Notes
    -----
    Creates new_image with shape = stimulus.shape, inserts image at normalized
    location, then calls add_image_to_stimulus(stimulus, new_image).
    """
    assert np.all(np.array(stimulus.shape) - np.array(image.shape) >= 0), (
        'image shape must be less than or equal to stimulus shape'
    )
    # get stimulus height and width
    stim_hw = np.array(stimulus.shape, dtype='int')
    img_hw = np.array(image.shape, dtype='int')
    # get shift from location in pixels
    shift = np.array((stim_hw * (np.array(loc) + 1.) / 2.) - (img_hw // 2), dtype='int')
    # create zero_like stimulus
    new_image = np.zeros_like(stimulus)
    # insert image to zero_stimulus and roll to location
    new_image[:img_hw[0], :img_hw[1]] = image
    new_image = np.roll(new_image, shift, (0,1))
    # return stimulus at location
    return add_image_to_stimulus(stimulus, new_image)

def add_image_to_stimulus(stimulus, image):
    """
    insert nonzero values into stimulus array at overlapping pixel locations

    Parameters
    ----------
    stimulus : numpy.ndarray
        array to set values into
    image : numpy.ndarray
        array to be inserted into stimulus with shape stimulus.shape

    Returns
    -------
    stimulus : numpy.ndarray
        stimulus with values replaced with nonzero image values at overlapping
        pixel locations

    Examples
    --------
    >>> image = np.zeros((4,4))
    >>> image[:,1] = 1.
    >>> image[0,:] = -1.
    >>> stimulus = 0.5 * np.ones((4,4))
    >>> output = add_image_to_stimulus(stimulus, image)
    >>> output
    array([[-1. , -1. , -1. , -1. ],
           [ 0.5,  1. ,  0.5,  0.5],
           [ 0.5,  1. ,  0.5,  0.5],
           [ 0.5,  1. ,  0.5,  0.5]])
    """
    # create nonzero image mask
    mask = np.zeros_like(image)
    mask[image.nonzero()] = 1.
    # multiply mask with stimulus to find overlap
    overlap = mask * stimulus
    # subtract overlap from image and add to stimulus
    return stimulus + (image - overlap)

def make_crowded_circles(n_flank, radius_range, dtype=np.float, **kwargs):
    """
    Makes a crowded stimulus with circle of random size

    Parameters
    ----------
    n_flank : int
        the number of flankers surrounding a single target
        decides the equally-spaced layout
    radius_range : tuple or list
        if tuple, the range of radii to be randomly sampled [low, high)
        if list, the radii randomly assigned for each circle
        (len(radius_range) >= n_flank+1)
    dtype : type
        dtype of output labels [default: np.float]
    **kwargs : dict
        see make_crowded_stimuli

    Returns
    -------
    s : numpy.ndarray
        the crowded stimulus
    target_radius : float or int
        the radius of the central circle with dtype given
    mean_radius : float or int
        the average radius of all the circles with dtype given
    """
    image_size = 2.*np.max(radius_range)
    if type(radius_range) is tuple:
        radii = np.random.randint(low=radius_range[0], high=radius_range[1],
                                  size=n_flank+1)
    elif type(radius_range) is list:
        radii = np.random.permutation(radius_range)[:n_flank+1]
    else:
        raise Exception('radius_range type not understood')
    # set target_radius and mean_radius
    target_radius = dtype(radii[0])
    mean_radius = np.mean(radii, dtype=dtype)
    # make circle stimuli
    circles = [make_circle(r, image_size) for r in radii]
    s = make_crowded_stimuli(circles[0], circles[1:n_flank+1], **kwargs)

    return s, target_radius, mean_radius

def make_circle(radius, image_size):
    c = int(image_size/2)
    xx, yy = np.mgrid[:image_size, :image_size]
    circle = (xx - c)**2 + (yy - c)**2
    return np.uint8(255*(circle < radius**2).astype(int))
