import re

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from . import functions

def plot_with_kwargs(fn, args, fn_prefix=None, **kwargs):
    # .Collection kwargs
    coll_keys = ['edgecolors','facecolors','linewidths','linestyles','capstyle',
                'joinstyle','antialiaseds','offsets','transOffset','offset_position',
                'norm','cmap','hatch','zorder']
    # .Line2D kwargs
    line_keys = ['alpha','animated','aa','clip_box','clip_on','clip_path','color','c',
                 'contains','dash_capstyle','dash_joinstyle','dashes','drawstyle',
                 'figure','fillstyle','gid','label','ls','lw','marker','mec','mew',
                 'mfc','mfcalt','ms','markevery','path_effects','picker','pickradius',
                 'rasterized','sketch_params','length','randomness','snap',
                 'solid_capstyle','solid_joinstyle','transform','a','url','visible',
                 'zorder']
    # get keys for fn
    if re.search('[\.\s]colorbar', str(fn)):
        keys = ['mappable','pyplot','cax','ax','use_gridspec']
    elif re.search('[\.\s]imshow', str(fn)):
        keys = ['X','cmap','aspect','interpolation','norm','vmax','vmin','alpha',
                'origin','extent','shape','filternorm','filterrad']
    elif re.search('[\.\s]savefig', str(fn)):
        keys = ['fname','dpi','facecolor','edgecolor','orientation','papertype',
                'format','transparent','frameon','bbox_inches','pad_inches',
                'bbox_extra_artists']
    elif re.search('[\.\s]scatter', str(fn)):
        keys = ['x','y','s','c','marker','cmap','norm','vmax','vmin','alpha',
                'linewidths','verts','edgecolors']
        keys += coll_keys
    elif re.search('[\.\s]plot', str(fn)):
        keys = ['x','y','fmt','data','scalex','scaley']
        keys += line_keys
    else:
        raise Exception('Unsupported fn: %a' % fn)
    # get kwargs
    if type(fn_prefix) is str:
        fn_prefix += '_'
        d = functions.get_attributes(kwargs, [fn_prefix + k for k in keys],
                                     ignore_keys=True)
        fn_kwargs = dict([(k.replace(fn_prefix, ''), v) for k, v in d.items()])
    else:
        fn_kwargs = functions.get_attributes(kwargs, keys, ignore_keys=True)
    # call function
    return fn(*args, **fn_kwargs)

def plot_images(w, img_shape=None, figsize=(5, 5), cmap=None, **kwargs):
    """
    Plot images contained in tensor

    Paramaters
    ----------
    w : torch.tensor
        tensor of images to plot with ndimension == 3 or ndimension == 4
        image dimensions should be contained in w.shape[-2:] and image channels
        should be contained in w.shape[1] (optional)
    img_shape : tuple or None
        shape of image contained in last dimension of w [default: None]
    figsize : tuple
        figure size (passed to subplots) [default: (5, 5)]
    cmap : str or None
        color map (passed to imshow) [default: None]

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with w.shape[0] (or w.shape[0] + w.shape[1]) images with
        np.ceil(np.sqrt(w.shape[0])) number of rows and columns

    Notes
    -----
    If w.ndimension() == 4 and w.shape[1] > 3, dimensions 0 and 1 will be
    flattened together.
    """
    # if channels > 3, reshape
    if w.ndimension() == 4 and w.shape[1] > 3:
        w = torch.flatten(w, 0, 1).unsqueeze(1)
    # get columns and rows
    n_cols = np.ceil(np.sqrt(w.shape[0])).astype('int')
    n_rows = np.ceil(w.shape[0] / n_cols).astype('int')
    # init figure and axes
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = np.reshape(ax, (n_rows, n_cols))
    # plot images
    cnt = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if cnt >= w.shape[0]:
                w_n = torch.zeros_like(w[0])
            else:
                w_n = w[cnt].detach()
            if img_shape:
                w_n = torch.reshape(w_n, (-1,) + img_shape)
            w_n = torch.squeeze(w_n.permute(1,2,0), -1).numpy()
            w_n = functions.normalize_range(w_n, dims=(0,1))
            ax[r,c].axis('off')
            ax[r,c].imshow(w_n, cmap=cmap, **kwargs)
            cnt += 1
    plt.show()
    return fig

def show_confusion_matrix(data, labels, cmap=plt.cm.afmhot):
    """
    #TODO:WRITEME
    """
    fig, ax = plt.subplots(1,1)
    heatmap = ax.pcolor(data, cmap=cmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(labels.numpy(), minor=False)
    ax.set_yticklabels(labels.numpy(), minor=False)
    fig.colorbar(heatmap)

def heatmap(model, layer_id, scores=None, input=None, outline_rfs=True,
            filename=None, figsize=(5,5), colorbar=False, **kwargs):
    """
    #TODO:WRITEME
    """
    # init figure
    if 'ax' not in kwargs:
        ax = plt
        fig = plt.figure(figsize=figsize)
    else:
        ax = kwargs.pop('ax')
        fig = ax.get_figure()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        figsize = [bbox.width, bbox.height]
    # plot RF outlines in image space
    if outline_rfs:
        mu, sigma = model.rf_to_image_space(layer_id)
        mu = mu + 0.5
        scatter_kwargs = {'s': np.prod(figsize)*sigma**2, 'alpha': 0.25,
                          'edgecolors': 'black', 'facecolors': 'none'}
        [kwargs.setdefault('RF_' + k, v) for k, v in scatter_kwargs.items()]
        plot_with_kwargs(ax.scatter, [mu[:,1], mu[:,0]], fn_prefix='RF',
                         **kwargs)
    # get heatmap
    heatmap = model.rf_heatmap(layer_id)
    if scores is None:
        scores = torch.zeros(heatmap.shape[0])
    scores = scores.reshape(scores.shape[0],1,1)
    mask = torch.isnan(scores).bitwise_not().float()
    scores[torch.isnan(scores)] = 0.
    score_map = scores * heatmap
    score_map = torch.div(torch.sum(score_map, 0),
                          torch.sum(mask * heatmap, 0))
    score_map[torch.isnan(score_map)] = 0.
    # show score_map, update colorbar
    kwargs.setdefault('cmap', 'Greys')
    plot_with_kwargs(ax.imshow, [score_map], **kwargs)
    if colorbar:
        plot_with_kwargs(ax.colorbar, [], **kwargs)
    # add input to image using masked array
    if input is not None:
        if type(input) is np.ma.core.MaskedArray:
            ma_input = input
        else: # binary image masking out zeros
            ma_input = np.ma.masked_array(torch.gt(input, 0.).bitwise_not(),
                                          input==0.)
        kwargs.setdefault('input_cmap', 'gray')
        kwargs.setdefault('input_alpha', 1.)
        plot_with_kwargs(ax.imshow, [ma_input], fn_prefix='input', **kwargs)
    # remove xticks, yticks
    [kwargs.setdefault(k, v) for k, v in
    {'axis':'on', 'xticks':[], 'yticks':[]}.items()]
    plt.axis(kwargs.get('axis'))
    plt.xticks(kwargs.get('xticks'))
    plt.yticks(kwargs.get('yticks'))
    # save file if given
    if filename:
        kwargs.setdefault('dpi', 600.)
        plot_with_kwargs(ax.savefig, [filename], **kwargs)
    # show plot
    kwargs.setdefault('show', True)
    if kwargs.get('show'):
        plt.show()
    return fig

def visualize_embedding(embeddings, images, labels=None, cmap='tab10', figsize=(15,15)):
    """
    2D Visualization for a dimensionally-reduced set of embeddings.
    Plots the images at the embedding points

    Paramaters
    ----------
    embeddings: numpy.array
        2-d point coordinates of feature vector embeddings
    images: torch.tensor
        input images
    labels: torch.tensor
        corresponding class labels

    Returns
    -------
    None
    """
    images = images.permute(0,2,3,1)

    ax_max = np.max(embeddings, axis=0)
    ax_min = np.min(embeddings, axis=0)
    ax_dist = np.linalg.norm(ax_max - ax_min)
    min_dist = (1/(3*figsize[0])) * ax_dist

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    cmap = plt.get_cmap(cmap)

    shown_embeddings = np.ones((1,2))
    for i in range(embeddings.shape[0]):
        dist_to_embeddings = np.linalg.norm((shown_embeddings - embeddings[i]), axis=1)

        if np.min(dist_to_embeddings) >= min_dist:
            shown_embeddings = np.vstack((shown_embeddings, embeddings[i,None]))
            patch = images[i]

            # colorize images
            if labels is not None:
                if images.shape[-1] != 3:
                    patch = color_code_patch(patch.numpy(), labels[i].numpy(),cmap)
            else:
                patch = torch.flatten(patch, -2)

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(patch, zoom=1, cmap=plt.cm.gray_r),
                xy=embeddings[i], frameon=False
                )
            ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()

def color_code_patch(patch, label, cmap):
    patch = np.tile(patch, (1,1,3))
    patch = (1,1,1) * (1 - patch) + cmap(label)[:3] * patch
    return np.clip(patch, 0, 1)
