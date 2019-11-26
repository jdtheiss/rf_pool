import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from context import rf_pool
from rf_pool import models, modules, pool, ops
from rf_pool.utils import lattice, functions, visualize, datasets, stimuli

def rotate_fn(max_angle, base_angle=0., seed=0):
    gen = np.random.RandomState(seed=seed)
    return lambda : (2. * gen.rand() - 1.) * max_angle + base_angle

def jitter_fn(max_w, max_h, seed=0):
    gen = np.random.RandomState(seed=seed)
    w_fn = lambda : np.int((2. * gen.rand() - 1.) * max_w)
    h_fn = lambda : np.int((2. * gen.rand() - 1.) * max_h)
    return lambda x: torch.roll(x, [w_fn(),h_fn()], dims=(-2,-1))

def get_crowd_params(crowd_type, ref_axis=0.):
    """
    Gets the number of flankers and axis direction for the crowded configuration type
    """
    # number of flankers
    if crowd_type in ['outer','inner']:
        n_flankers = 1
    else:
        n_flankers = 2
    
    # axis with respect to the reference axis
    if crowd_type == 'inner':
        axis = ref_axis + np.pi
    elif crowd_type == 'tangential':
        axis = ref_axis + (np.pi / 2.)
    else:
        axis = ref_axis 
        
    return n_flankers, axis

def create_crowd_set(dataset, n_images, img_size, n_flankers, axis, spacing, base_set=None,
                     label_map=None, no_target=False, transform=transforms.ToTensor()):
    """
    Creates a crowded digit dataset 
    """
    if base_set is None:
        crowd_set = datasets.CrowdedDataset(dataset, n_flankers, n_images, no_target=no_target,
                                            load_previous=False, label_map=label_map,
                                            transform=transform,
                                            spacing=20*spacing, background_size=img_size, axis=axis)
    else:
        crowd_set = datasets.CrowdedDataset(dataset, n_flankers, n_images,
                                            base_set.recorded_target_indices,
                                            base_set.recorded_flanker_indices,
                                            no_target=no_target,
                                            load_previous=True, label_map=label_map,
                                            transform=transform,
                                            spacing=20*spacing, background_size=img_size, axis=axis)
    return crowd_set

def apply_attention_field(model, layer_id, mu, sigma, loc, extent, update_mu=True, update_sigma=True):
    """
    Updates the receptive fields of the model by applying an attentional field with a given extent
    """
    # update rfs with spatial extent
    img_shape = model.layers[layer_id].forward_layer.pool.get(['img_shape'])[0]
    attn_field = torch.zeros(img_shape)
    if extent > 0:
        attn_field[loc[0],loc[1]] = 1./extent
        new_mu, new_sigma = lattice.apply_attentional_field(mu, sigma, attn_field)
    else:
        new_mu, new_sigma = mu, sigma
    if not update_mu:
        new_mu = mu
    if not update_sigma:
        new_sigma = sigma
    model.layers[layer_id].forward_layer.pool.update_rfs(mu=new_mu, sigma=new_sigma)
    return model

def get_accuracy(target_loader, crowd_loader, layer_id='1', batch_size=1, model=None, RF_mask=None,
                 extent=None, lattice_fn=None, lattice_kwargs=None, update_mu=True, update_sigma=True):
    """
    Gets the accuracy over the whole set of stimuli by masking out non-target RFs
    """
    # parse lattice_kwargs
    if lattice_kwargs is not None:
        lattice_kwargs.setdefault('rotate', 0.)
        rotate = lattice_kwargs.pop('rotate')
        if type(rotate) is not type(lambda : 0.):
            rotate_fn = lambda : rotate
        else:
            rotate_fn = rotate
    else:
        rotate_fn = None
    # init acc
    correct = 0.
    acc_i = []
    # get accuracy for each image
    for i, ((target, labels), (crowd, _)) in enumerate(zip(target_loader, crowd_loader)):
        # reset RFs
        if lattice_fn is not None and lattice_kwargs is not None:
            if rotate_fn is not None:
                mu, sigma = lattice_fn(**lattice_kwargs, rotate=rotate_fn())
            else:
                mu, sigma = lattice_fn(**lattice_kwargs) 
            model.layers[layer_id].forward_layer.pool.update_rfs(mu=mu, sigma=sigma)
        # get mask
        if RF_mask is not None:
            mask_i = RF_mask.clone()
        else:
            mask_i = model.rf_index(target, layer_id, thr=0.1).float()
        # attention
        if extent:
            model = apply_attention_field(model, layer_id, mu, sigma, [26,26], extent, update_mu, update_sigma)
        # get target signal and noise (crowd)
        with torch.no_grad():
            crowd_output = model.rf_output(crowd, layer_id, retain_shape=True)
            # mask crowd_output and pass forward to get accuracy
            masked_output = torch.max(torch.mul(crowd_output, mask_i.reshape(batch_size, 1, -1, 1, 1)), 2)[0]
            output = torch.max(model.apply_layers(masked_output, ['2']).flatten(-2), -1)[0]
            correct_i = torch.sum(torch.max(output, -1)[1] == labels).item()
            correct += correct_i
            acc_i.append(correct_i)
    # get pct correct, SNR
    pct_correct = (correct / (batch_size * len(target_loader)))
    return pct_correct, acc_i

def get_contribution(target_loader, crowd_loader, layer_id='1', model=None, RF_mask=None, acc=None,
                     extent=None, lattice_fn=None, lattice_kwargs=None):
    """
    Gets the contribution for each receptive field to target channel output
    """
    assert iter(target_loader).next()[0].shape[0] == 1, ('Batch size must be 1.')
    # parse lattice_kwargs
    if lattice_kwargs is not None:
        lattice_kwargs.setdefault('rotate', 0.)
        rotate = lattice_kwargs.pop('rotate')
        if type(rotate) is not type(lambda : 0.):
            rotate_fn = lambda : rotate
        else:
            rotate_fn = rotate
    else:
        rotate_fn = None
    # counter
    cnt = 0.
    n_kernels = model.layers[layer_id].forward_layer.pool.mu.shape[0]
    RF_acc = torch.zeros(n_kernels)
    # get SNR, accuracy for each image
    for i, ((target, label), (crowd, _)) in enumerate(zip(target_loader, crowd_loader)):
        # skip wrong trials
        if acc is not None and acc[i] == 0:
            continue
        # reset RFs
        if lattice_fn is not None and lattice_kwargs is not None:
            if rotate_fn is not None:
                mu, sigma = lattice_fn(**lattice_kwargs, rotate=rotate_fn())
            else:
                mu, sigma = lattice_fn(**lattice_kwargs) 
            model.layers[layer_id].forward_layer.pool.update_rfs(mu=mu, sigma=sigma)
        # get mask
        if RF_mask is not None:
            mask_i = RF_mask.clone()
        else:
            mask_i = model.rf_index(target, layer_id, thr=0.1).float()
        # attention
        if extent:
            model = apply_attention_field(model, layer_id, mu, sigma, [26,26], extent)
        # get output for each RF
        with torch.no_grad():
            crowd_output = model.rf_output(target + crowd, layer_id, retain_shape=True)
            # mask crowd_output and pass forward to get accuracy
            masked_output = torch.mul(crowd_output, mask_i.reshape(1, 1, -1, 1, 1))
            # permute dimensions to (n_RF, n_ch, h, w)
            masked_output = masked_output[0].permute(1,0,2,3)
            # get flattened output of third layer (n_RF, n_ch, h * w)
            output = model.apply_layers(masked_output, ['2'])
            # get control output to account for target-flanker feature coincidences
            crowd_control_output = model.rf_output(crowd, layer_id, retain_shape=True)
            # mask crowd_output and pass forward to get accuracy
            masked_control_output = torch.mul(crowd_control_output, mask_i.reshape(1, 1, -1, 1, 1))
            # permute dimensions to (n_RF, n_ch, h, w)
            masked_control_output = masked_control_output[0].permute(1,0,2,3)
            # get flattened output of third layer (n_RF, n_ch, h * w)
            control_output = model.apply_layers(masked_control_output, ['2'])
            # remove control_output from output (to account for accidental target features from flankers)
            output = output - control_output
            # get contribution to sum value across RFs
            max_output = torch.max(torch.flatten(output, -2), -1)[0] #[:, label.item()], -1)
            # softmax across channels
            max_output = torch.softmax(max_output, -1)[:, label.item()]
            # re-mask RFs
            max_output = torch.mul(max_output, mask_i.reshape(-1))
            RF_acc += max_output
            cnt += 1.
    # update RF_acc as proportion correct
    RF_acc /= cnt
    return RF_acc

def get_ablated(target_loader, crowd_loader, layer_id='1', model=None, RF_mask=None,
                extent=None, lattice_fn=None, lattice_kwargs=None):
    """
    Gets the change in accuracy due to each receptive field
    """
    assert iter(target_loader).next()[0].shape[0] == 1, ('Batch size must be 1.')
    # parse lattice_kwargs
    if lattice_kwargs is not None:
        lattice_kwargs.setdefault('rotate', 0.)
        rotate = lattice_kwargs.pop('rotate')
        if type(rotate) is not type(lambda : 0.):
            rotate_fn = lambda : rotate
        else:
            rotate_fn = rotate
    else:
        rotate_fn = None
    # counter
    cnt = 0.
    n_kernels = model.layers[layer_id].forward_layer.pool.mu.shape[0]
    RF_acc = torch.zeros(n_kernels)
    correct = 0.
    # get SNR, accuracy for each image
    for i, ((target, label), (crowd, _)) in enumerate(zip(target_loader, crowd_loader)):
        # reset RFs
        if lattice_fn is not None and lattice_kwargs is not None:
            if rotate_fn is not None:
                mu, sigma = lattice_fn(**lattice_kwargs, rotate=rotate_fn())
            else:
                mu, sigma = lattice_fn(**lattice_kwargs) 
            model.layers[layer_id].forward_layer.pool.update_rfs(mu=mu, sigma=sigma)
        # get mask
        if RF_mask is not None:
            mask_i = RF_mask.clone()
        else:
            mask_i = model.rf_index(target, layer_id, thr=0.1).float()
        # attention
        if extent:
            model = apply_attention_field(model, layer_id, mu, sigma, [26,26], extent)
        # get output for each RF
        with torch.no_grad():
            crowd_output = model.rf_output(target + crowd, layer_id, retain_shape=True)
            # mask crowd_output and pass forward to get accuracy
            masked_output = torch.mul(crowd_output, mask_i.reshape(1, 1, -1, 1, 1))
            # permute dimensions to (n_RF, n_ch, h, w)
            masked_output = masked_output[0].permute(1,0,2,3)
            # get flattened output of third layer (n_RF, n_ch, h * w)
            output = torch.flatten(model.apply_layers(masked_output, ['2']), -2)
            output = torch.mul(output, mask_i.reshape(-1, 1, 1))
            # get prediction and accuracy
            pred = torch.max(torch.max(torch.sum(output, 0), -1)[0], 0)[1]
            correct += (pred == label).item()
            cnt += 1.
            # for each RF, get predicted target without RF
            for idx in torch.where(mask_i)[1].tolist():
                tmp_mask = mask_i.clone()
                tmp_mask[0,idx] = 0.
                tmp_output = torch.mul(output, tmp_mask.reshape(-1, 1, 1))
                pred = torch.max(torch.max(torch.sum(tmp_output, 0), -1)[0], 0)[1]
                RF_acc[idx] += (pred == label).item()
    RF_acc = (correct - RF_acc) / cnt
    correct /= cnt
    return RF_acc, correct