import numpy as np 

def make_crowded_stimuli(target, flankers, spacing, background_size, layout_type=None, random=False):
    """
    TODO
    """
    # make background
    target_size = int(target.shape[0])
    center = int(target_size * np.floor(background_size / 2))
    stimuli_base = np.zeros((target_size * background_size, )*2)
    stimuli_base[center:center + target_size, center: center + target_size] = target
    space_size = int(spacing * target_size)

    if random:
        np.random.shuffle(flankers)

    # target with horizontal flankers
    if layout_type == 'h':
        stimuli = h_layout(stimuli_base, flankers, center, target_size, space_size)
    # target with vertical flankers
    elif layout_type == 'v':
        stimuli = v_layout(stimuli_base, flankers, center, target_size, space_size)
    # target with horizontal and vertical flankers
    elif layout_type == 't':
        stimuli = t_layout(stimuli_base, flankers, center, target_size, space_size)
    # target with hexagonal flankers
    elif layout_type == 'o':
        stimuli = o_layout(stimuli_base, flankers, center, target_size, space_size)
    # target with no flankers
    elif layout_type is None:
        stimuli = stimuli_base
    else:
        raise Exception("layout_type must be in (h, v, t, o, None)")

    return stimuli

def h_layout(base, flankers, c, t, s):
    left_shift = c-(s+t)
    right_shift = c+(s+t)
    base[c:c+t, left_shift:left_shift+t] = flankers[0]      
    base[c:c+t, right_shift:right_shift+t] = flankers[1] 
    return base

def v_layout(base, flankers, c, t, s):
    up_shift = c-(s+t)
    down_shift = c+(s+t)
    base[up_shift:up_shift+t, c: c+t] = flankers[0] # top
    base[down_shift:down_shift+t, c: c+t] = flankers[1] # bottom
    return base
    
def t_layout(base, flankers, c, t, s):
    base = h_layout(base, flankers[:2], c, t, s) # left and right
    base = v_layout(base, flankers[2:], c, t, s) # top and bottom
    return base

def o_layout(base, flankers, c, t, s):
    left_shift = c-(s+t)
    right_shift = c+(s+t)
    up_shift = c-s
    down_shift = c+s
    base = v_layout(base, flankers[:2], c, t, s) # top and bottom
    base[up_shift:up_shift+t, left_shift:left_shift+t] = flankers[2] # top-left
    base[up_shift:up_shift+t, right_shift:right_shift+t] = flankers[3] # top-right
    base[down_shift:down_shift+t, right_shift:right_shift+t] = flankers[4] # bottom-right
    base[down_shift:down_shift+t, left_shift:left_shift+t] = flankers[5] # bottom-left
    return base

