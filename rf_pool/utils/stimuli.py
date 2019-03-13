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
    spacing: float
        distance between target center and flanker
        center as proportion of target size
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
            stimuli[:,:,i] = np.roll(stimuli[:,:,i],
                                    (center[0]+y_shift[i-1], center[1]+x_shift[i-1]), (0,1))        
    stimuli = np.max(stimuli, -1)
    
    return stimuli


def crowded_MNIST(dataset, n_flank, **kwargs):
    """
    Converts an MNIST dataset into crowded stimuli with flankers
    
    Paramaters
    ----------
    dataset: torchvision.datasets.mnist.MNIST
        pytorch MNIST trainset or testset object
    n_flank: int
        number of flankers for the crowded stimuli
    **kwargs: dict
        The crowded stimul arguments
        see make_crowded_stimuli for details
        
    Returns
    -------
    dataset: torchvision.datasets.mnist.MNIST
        pytorch MNIST dataset with crowded stimuli
    """
    
    # get dataset images and labels
    if dataset.train:
        inputs = dataset.train_data
        labels = dataset.train_labels
    else:
        inputs = dataset.test_data
        labels = dataset.test_labels  
    if type(inputs) is torch.Tensor:
        inputs = inputs.numpy()
    # get the targets
    n_set = inputs.shape[0]
    target_indices = np.arange(0, n_set, n_flank+1)
    # crowd the input images
    crowded_inputs = []
    for i in target_indices:
        target = inputs[i]
        flankers = inputs[i+1:i+n_flank+1]
        s = make_crowded_stimuli(target, flankers,  **kwargs)
        crowded_inputs.append(s)
    # get the labels
    crowded_labels = labels[target_indices]
    
    # reassign to the dataset
    if dataset.train:
        dataset.train_data = torch.tensor(crowded_inputs, dtype=torch.uint8)
        dataset.train_labels = crowded_labels
    else:
        dataset.test_data = torch.tensor(crowded_inputs, dtype=torch.uint8)
        dataset.test_labels = crowded_labels
    
    return dataset
