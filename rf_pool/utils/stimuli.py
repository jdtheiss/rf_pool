import numpy as np

def make_crowded_stimuli(target, flankers, spacing, background_size, axis=0., random=False):
    """
    makes a crowded stimulus: central target with evenly-spaced flankers
    
    Parameters
    ----------
    target: numpy.array
        the target image
    flankers: numpy.array_like
        list of flanker images
    spacing: float
        distance between target center and flanker
        center as proportion of target size
    background_size: int
        size of the square blank background in pixels
    axis: float 
        initialization axis for flankers in radians 
        [initialized as 0]
    random: bool 
        flankers are placed in clockwise fashion if False 
        [initialized as False]. 
        
    Returns
    -------
    stimuli: nump.array
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
    center = int(background_size // 2 - target_size // 2)
    stimuli = np.zeros((background_size, background_size, n_flank+1))
    stimuli[center:center + target_size, center:center + target_size, 0] = target
    space_size = int(spacing * target_size)
    
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
            stimuli[:,:,i] = np.roll(stimuli[:,:,i], (center, x_shift[i-1]), (0,1))
            stimuli[:,:,i] = np.roll(stimuli[:,:,i], (y_shift[i-1], center), (0,1))           
    stimuli = np.max(stimuli, -1)
    
    return stimuli
