import re

import imageio
from IPython.display import clear_output, display
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib import patches
import numpy as np
import torch

from . import functions
from .. import losses

def plot_with_kwargs(fn, args, fn_prefix=None, **kwargs):
    # .Collection kwargs
    coll_keys = ['edgecolors','facecolors','linewidths','linestyles','capstyle',
                'joinstyle','antialiaseds','offsets','transOffset','offset_position',
                'norm','cmap','hatch','zorder']
    # .Line2D kwargs
    line_keys = ['alpha','animated','aa','clip_box','clip_on','clip_path','color','c',
                 'contains','dash_capstyle','dash_joinstyle','dashes','drawstyle',
                 'figure','fillstyle','gid','label','linestyle','ls','linewidth','lw',
                 'marker','markeredgecolor','mec','markeredgewidth','mew',
                 'markerfacecolor','mfc','markerfacecoloralt','mfcalt','markersize',
                 'ms','markevery','path_effects','picker','pickradius','rasterized',
                 'sketch_params','length','randomness','snap','solid_capstyle',
                 'solid_joinstyle','transform','a','url','visible','zorder']
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
                'linewidths','verts','edgecolors','sizes']
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

def create_cmap(r=(0.,1.), g=(0., 1.), b=(0., 1.), N=256):
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(*r, N)
    vals[:, 1] = np.linspace(*g, N)
    vals[:, 2] = np.linspace(*b, N)
    return matplotlib.colors.ListedColormap(vals)

def make_gif(images, n_frames=1, filename=None, **kwargs):
    """
    Make/save images as gif

    Parameters
    ----------
    images : torch.Tensor
        images to make into gif with shape (batch, ch, h, w)
        if `ch != 3`, images will be flattened along first two dims
    n_frames : int
        number of frames per image in gif [default: 1]
    filename : str
        filename to save gif (if None, gif is shown using `matplotlib.imshow`)
        [default: None]
    **kwargs : **dict
        keyword arguments passed to either `imagio.mimsave` or
        `matplotlib.imshow` depending on whether `filename` is None

    Returns
    -------
    None
    """
    assert images.ndimension() >= 3
    if not filename is None and not filename.endswith('.gif'):
        filename += '.gif'
    # divide by max and set to uint8
    images = images.detach()
    max_images = torch.max(images.flatten(-2), -1)[0]
    images = torch.div(images, max_images.reshape(*images.shape[:-2], 1, 1))
    images = torch.mul(images, 255).type(torch.uint8)
    # permute or flatten based on ndimension
    if images.ndimension() == 4 and images.shape[1] == 3:
        images = images.permute(0,2,3,1)
    else:
        images = images.flatten(0, -3)
    # repeat for n_frames
    images = images.numpy()
    images = np.repeat(images, n_frames, axis=0)
    # save gif
    if filename is not None:
        imageio.mimsave(filename, images, **kwargs)
    else:
        for image in images:
            clear_output(wait=True)
            plt.imshow(image, **kwargs)
            plt.show()

def show_images(*args, img_shape=None, figsize=(5, 5), **kwargs):
    """
    Show images contained in tensor

    Paramaters
    ----------
    args : torch.tensor
        tensor of images to plot with ndimension == 3 or ndimension == 4
        image dimensions should be contained in w.shape[-2:] and image channels
        should be contained in w.shape[1] (optional)
    img_shape : tuple or None
        shape of image contained in last dimension of w [default: None]
    figsize : tuple
        figure size (passed to subplots) [default: (5, 5)]
    n_cols : int, optional
        number of columns for images [default: np.sqrt(imgs.shape[0]) or len(imgs)]
    n_rows : int, optional
        number of rows for images [default: imgs.shape[0]/n_cols or imgs[0].shape[0]]
    **kwargs : keyword arguments
        kwargs passed to imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure with imgs.shape[0] (or imgs.shape[0] + imgs.shape[1]) images with
        axes shape (n_rows, n_cols)

    Notes
    -----
    If imgs.ndimension() == 4 and imgs.shape[1] > 3, dimensions 0 and 1 will be
    flattened together.
    """

    # if list, set n_cols, n_rows and stack
    if len(args) > 1:
        kwargs.setdefault('n_cols', len(args))
        kwargs.setdefault('n_rows', args[0].shape[0])
        imgs = torch.stack(args).transpose(0,1).flatten(0,1)
    else:
        imgs = args[0]
    imgs = torch.detach(imgs)
    # if channels > 3, reshape
    if imgs.ndimension() == 4 and imgs.shape[1] > 3:
        imgs = torch.flatten(imgs, 0, 1).unsqueeze(1)
    # get columns and rows
    kwargs.setdefault('n_cols', np.ceil(np.sqrt(imgs.shape[0])).astype('int'))
    n_cols = kwargs.pop('n_cols')
    kwargs.setdefault('n_rows', np.ceil(imgs.shape[0] / n_cols).astype('int'))
    n_rows = kwargs.pop('n_rows')
    # init figure and axes
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    ax = np.reshape(ax, (n_rows, n_cols))
    # plot images
    cnt = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if cnt >= imgs.shape[0]:
                img_i = torch.zeros_like(imgs[0])
            else:
                img_i = imgs[cnt]
            if img_shape:
                img_i = torch.reshape(img_i, (-1,) + img_shape)
            img_i = torch.squeeze(img_i.permute(1,2,0), -1).numpy()
            img_i = functions.normalize_range(img_i, dims=(0,1))
            ax[r,c].axis('off')
            ax[r,c].imshow(img_i, **kwargs)
            cnt += 1
    plt.show()
    return fig

def visualize_features(model, layer_id, module_name, feature_index,
                       loss_fn=lambda x: -x.mean(), lr=0.05, n_iter=1000,
                       optim_fn=torch.optim.Adam, optim_kwargs={}, seed=None,
                       img_shape=(3,128,128), **kwargs):
    """
    Visualize features via gradient ascent of feature outputs

    Parameters
    ----------
    model : rf_pool.models
        Model containing features to visualize
    layer_id : str
        layer containing features to visualize
    module_name : str
        module containing features to visualize
    feature_index : int or list
        index of outputs on channel dimension from
        `getattr(model.layers[layer_id].forward_layer, module_name)` to visualize
        via gradient ascent to maximally activate given feature
    loss_fn : function or rf_pool.losses
        lambda function passed to `rf_pool.losses.FeatureLoss` or `rf_pool.losses`
        instance [default: lambda x: -x.mean(), where `x` is indexed features
        with shape (batch, len(feature_index), h, w)]
    lr : float
        learning rate used during gradient ascent [default: 0.05]
    n_iter : int
        number of iterations to perform gradient ascent [default: 1000]
    optim_fn : torch.optim
        optimizer function to use to update visualization
        [default: torch.optim.Adam]
    optim_kwargs : dict
        keyword arguments passed to optim_fn [default: {}]
    seed : torch.Tensor or None
        seed input to be updated for visualization
        [default: None, set to
        `(torch.randn(1, *img_shape) * 0.01).requires_grad_(True)`]
    img_shape : tuple or None
        image shape for seed [default: (3,128,128)]
    **kwargs : **dict
        keyword arguments passed to `model.optimize_texture`

    Returns
    -------
    seed : torch.Tensor
        input updated via gradient ascent to visualize features

    See Also
    --------
    rf_pool.models.Model.optimize_texture
    rf_pool.losses.FeatureLoss
    rf_pool.losses

    Notes
    -----
    The main loss function used for gradient ascent of feature outputs is
    `rf_pool.losses.FeatureLoss`.
    """
    # create seed
    if seed is None:
        seed = (torch.randn(1, *img_shape) * 0.01).requires_grad_(True)
    # set feature loss
    if not isinstance(loss_fn, losses.Loss):
        loss_fn = losses.FeatureLoss(model, {layer_id: {module_name: []}},
                                     feature_index, loss_fn,
                                     output_layer=layer_id, 
                                     **{layer_id: {'output_module': module_name}})
    # set optimizer
    optim = optim_fn([seed], lr=lr, **optim_kwargs)
    # optimize texture
    model.optimize_texture(n_iter, seed, loss_fn, optim, **kwargs);
    return seed

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

def bounding_box(ax, n_box, center, width, height, fill=False, **kwargs):
    # get figure
    fig = ax.get_figure()
    # set to list if not already
    if type(center) is not list:
        center = [center] * n_box
    if type(width) is not list:
        width = [width] * n_box
    if type(height) is not list:
        height = [height] * n_box
    # set fill to kwargs
    kwargs.update({'fill': fill})
    # function to get corner from center, width, height
    corner_fn = lambda x, w, h: (x[0] - w / 2., x[1] - h / 2.)
    # set args
    args = []
    for n in range(n_box):
        corner_n = corner_fn(center[n], width[n], height[n])
        args.append([corner_n, width[n], height[n]])
    # get list kwargs
    list_keys = [k for k, v in kwargs.items()
                 if type(v) is list and len(v) == n_box]
    list_kwargs = []
    for n in range(n_box):
        list_kwargs.append(dict([(k, kwargs.get(k)[n]) for k in list_keys]))
    [kwargs.pop(k) for k in list_keys]
    # set rectangle
    for arg, list_kwarg in zip(args, list_kwargs):
        ax.add_patch(patches.Rectangle(*arg, **list_kwarg, **kwargs))
    return fig

def get_adjusted_sizes(ax, size):
    # set sizes
    ppd = 72./ax.figure.dpi
    trans = ax.transData.transform
    size_data = np.stack([size, size]).T
    s = ((trans(size_data)-trans((0,0)))*ppd)
    return np.maximum(s[:,0], s[:,1])

def plot_rfs(mu, sigma, img_shape, mu0=None, sigma0=None, figsize=(5,5), **kwargs):
    """
    Plot RF outlines as a plot

    Parameters
    ----------
    mu : torch.Tensor
        centers of RFs with shape (n_RF, 2)
    sigma : torch.Tensor
        sizes of RFs with shape (n_RF, 1)
    img_shape : tuple
        image shape used for RFs
    mu0 : torch.Tensor
        optional initial center positions for each RF with shape (n_RF, 2)
        [default: None]
    sigma0 : torch.Tensor
        optional initial sizes for each RF with shape (n_RF, 1) [default: None]
    figsize : tuple
        figure size [default: (5,5)]
    **kwargs : **dict
        keyword arguments passed to `plot_with_kwargs`

    Optional kwargs
    ax : matplotlib.pyplot.Axes
        axis to plot scatter plot onto
        [default: `matplotlib.pyplot.figure().gca()`]
    fn_prefix : str
        prefix used for keywords passed to `plot_with_kwargs` [default: None]
    markersize: sigma.tolist()
    color : str
        color for data points [default: 'black']
    marker : str
        marker style for data points [default: 'o']
    markevery : int, float, tuple, list, None
        choices of data points to show [defalt: [int(mu0 is not None)]]
    markeredgecolor : str
        edgecolor of data points [default: 'black']
    markerfacecolor : str
        facecolor of data points [default: 'none']

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        figure containing scatter plot

    See Also
    --------
    rf_pool.models.rf_to_image_space
    plot_with_kwargs
    """
    # init figure
    if 'ax' not in kwargs:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = kwargs.pop('ax')
        fig = ax.get_figure()
    # set aspect equal
    ax.figure.canvas.draw()
    ax.set_aspect('equal', adjustable='box')
    # set limits to heatmap size
    ax.set_xlim(0.5, img_shape[1]-0.5)
    ax.set_ylim(img_shape[0]-0.5, 0.5)
    # set mu offsets
    offsets = np.flip(mu.data.numpy(), 1)
    if mu0 is not None:
        assert mu0.shape == mu.shape
        offsets0 = np.flip(mu0.data.numpy(), 1)
        offsets = np.stack([offsets0, offsets], -1)
    if sigma0 is not None:
        assert sigma0.shape == sigma.shape
    # set sigma sizes
    if sigma.ndimension() == 2:
        sigma = sigma.squeeze(-1)
    sigma = sigma.data.numpy()
    sigma = 2. * sigma
    sigma = get_adjusted_sizes(ax, sigma)
    # set kwargs for plot
    plot_kwargs = {'markersize': sigma.tolist(), 'color': 'black',
                   'marker': 'o', 'markevery': [int(mu0 is not None)],
                   'markeredgecolor': 'black', 'markerfacecolor': 'none'}
    if kwargs.get('fn_prefix') is not None:
        fn_prefix = kwargs.get('fn_prefix') + '_'
    else:
        fn_prefix = ''
    [kwargs.setdefault(fn_prefix + k, v) for k, v in plot_kwargs.items()]
    # get list kwargs
    list_keys = [k for k, v in kwargs.items()
                 if type(v) is list and len(v) == mu.shape[0]]
    list_kwargs = []
    for n in range(mu.shape[0]):
        list_kwargs.append(dict([(k, kwargs.get(k)[n]) for k in list_keys]))
    [kwargs.pop(k) for k in list_keys]
    # plot RFs
    for i, list_kwargs in enumerate(list_kwargs):
        plot_with_kwargs(ax.plot, offsets[i], **list_kwargs, **kwargs)
    return fig

def scatter_rfs(mu, sigma, img_shape, remove=False, updates={}, figsize=(5,5),
                **kwargs):
    """
    Plot RF outlines as a scatter plot

    Parameters
    ----------
    mu : torch.Tensor
        centers of RFs with shape (n_RF, 2)
    sigma : torch.Tensor
        sizes of RFs with shape (n_RF, 1)
    img_shape : tuple
        image shape used for RFs
    remove : boolean
        True/False whether to remove previous scatter plot from axis
        [default: False]
    updates : dict
        updates to scatter plot to be make like (attribute, update)
        where `hasattr(scatter_plot, attribute) is True` [default: {}]
    figsize : tuple
        figure size [default: (5,5)]
    **kwargs : **dict
        keyword arguments passed to `plot_with_kwargs`

    Optional kwargs
    ax : matplotlib.pyplot.Axes
        axis to plot scatter plot onto
        [default: `matplotlib.pyplot.figure().gca()`]
    fn_prefix : str
        prefix used for keywords passed to `plot_with_kwargs` [default: None]
    alpha : float
        alpha value used for `plot_with_kwargs` [default: 0.25]
    edgecolors : str
        edgecolor of scatter plot points [default: 'black']
    facecolors : str
        facecolor of scatter plot points [default: 'none']
    offsets : array-like
        centers of scatter plot points [default: `np.flip(mu, 1)`]
    sizes : array-like
        sizes of scatter plot points [default: `(2. * sigma)**2`]

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        figure containing scatter plot

    See Also
    --------
    rf_pool.models.rf_to_image_space
    plot_with_kwargs
    """
    # init figure
    if 'ax' not in kwargs:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = kwargs.pop('ax')
        fig = ax.get_figure()
    # get scatter plots in ax
    sc = [c for c in ax.get_children()
          if type(c) is matplotlib.collections.PathCollection]
    # remove scatter plot if remove and no updates
    if remove and len(updates) == 0:
        [c.remove() for c in sc]
    # set aspect equal
    ax.figure.canvas.draw()
    ax.set_aspect('equal', adjustable='box')
    # set limits to heatmap size
    ax.set_xlim(0.5, img_shape[1]-0.5)
    ax.set_ylim(img_shape[0]-0.5, 0.5)
    # set sigma
    if sigma.ndimension() == 2:
        sigma = sigma.squeeze(-1)
    sigma = sigma.data.numpy()
    sigma = 2. * sigma
    # set mu offsets
    offsets = np.flip(mu.data.numpy(), 1)
    # set sizes
    sigma = get_adjusted_sizes(ax, sigma)
    # set kwargs for scatter
    scatter_kwargs = {'offsets': offsets, 'sizes': sigma**2, 'alpha': 0.25,
                      'edgecolors': 'black', 'facecolors': 'none'}
    if kwargs.get('fn_prefix') is not None:
        fn_prefix = kwargs.get('fn_prefix') + '_'
    else:
        fn_prefix = ''
    [kwargs.setdefault(fn_prefix + k, v) for k, v in scatter_kwargs.items()]
    # set scatter plot
    if len(updates) == 0:
        sc = plot_with_kwargs(ax.scatter, offsets.T, **kwargs)
    else: # udpate
        for sc_i in sc:
            for key, value in updates.items():
                fn = getattr(sc_i, 'set_' + key)
                if value is not None:
                    fn(value)
                elif key in scatter_kwargs:
                    fn(scatter_kwargs.get(key))
    return fig

def heatmap(model, layer_id, module_name='pool', score_map=None, scores=None,
            input=None, rf_fn=scatter_rfs, filename=None, figsize=(5,5),
            colorbar=False, **kwargs):
    """
    Show heatmap of RF values

    Parameters
    ----------
    model : rf_pool.models
        model containing RF pooling layer
    layer_id : str
        layer id of RF pooling layer
    module_name : str
        module name for RF pooling layer [default: 'pool']
    score_map : array-like
        array of RF values to show with shape `model.rf_heatmap(layer_id).shape`
        [default: None]
    scores : array-like
        RF values to show with shape (n_RF,)
    input : array-like
        input image to show with shape `model.rf_heatmap(layer_id).shape`
    rf_fn : function
        function to show RF outlines (i.e., `scatter_rfs` or `plot_rfs`)
        [default: `scatter_rfs`]
    filename : str
        filename to save image [default: None]
    figsize : tuple
        figure size [default: (5,5)]
    colorbar : boolean
        True/False whether to include colorbar based on `scores`/`score_map`
    **kwargs : **dict
        keyword arguments passed to `rf_fn` (with prefix 'RF_') and
        `plot_with_kwargs` (with no prefix for imshow of `scores`/`score_map`
        and prefix 'input_' for imshow of `input`)

    Optional kwargs
    ax : matplotlib.pyplot.Axes
        axis to plot heatmap onto [default: `matplotlib.pyplot.figure().gca()`]
    axis : str
        'on' or 'off' for `ax.axis` function [default: 'on']
    cmap : matplotlib.pyplot.cm
        colormap used for `scores`/`score_map` [default: 'Greys']
    dpi : float
        dots per inch used in saving image (if `filename` given) [defualt: 600.]
    input_cmap : matplotlib.pyplot.cm
        colormap used for `input` (if given) [default: 'gray']
    input_alpha : float
        alpha used for imshow of `input` (if given) [default: 1.]
    show : boolean
        True/False to call `matplotlib.pyplot.show` [default: True]
    xticks : array-like or list
        x-axis tick values [default: []]
    yticks : array-like or list
        y-axis tick values [default: []]

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        figure containing image of heatmap

    See Also
    --------
    scatter_rfs
    plot_rfs
    plot_with_kwargs
    """
    # init figure
    if 'ax' not in kwargs:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = kwargs.pop('ax')
        fig = ax.get_figure()
    # get heatmap
    heatmap = model.rf_heatmap(layer_id, module_name)
    # plot RF outlines in image space
    if rf_fn:
        # get mu, sigma in image space
        mu, sigma = model.rf_to_image_space(layer_id, module_name=module_name)
        rf_fn(mu, sigma, heatmap.shape[1:], remove=False, ax=ax, fn_prefix='RF',
              **kwargs)
    # set scores
    if scores is None:
        scores = torch.zeros(heatmap.shape[0])
    if score_map is None:
        scores = scores.clone()
        scores = scores.reshape(-1,1,1)
        mask = torch.isnan(scores).bitwise_not().float()
        scores[torch.isnan(scores)] = 0.
        score_map = scores * heatmap
        score_map = torch.div(torch.sum(score_map, 0),
                              torch.sum(mask * heatmap, 0))
        score_map[torch.isnan(score_map)] = 0.
    # show score_map, update colorbar
    kwargs.setdefault('cmap', 'Greys')
    plot_with_kwargs(ax.imshow, [score_map], **kwargs)
    # colorbar
    if colorbar:
        fig.colorbar(ax.images[0], ax=ax)
        # update rf scatter plot
        if rf_fn:
            rf_fn(mu, sigma, heatmap.shape[1:], remove=True, ax=ax, fn_prefix='RF',
                  **kwargs)
    # add input to image using masked array
    if input is not None:
        if type(input) is np.ma.core.MaskedArray:
            ma_input = input
        else: # binary image masking out zeros
            ma_input = np.ma.masked_array(torch.gt(input, 0.), input==0.)
        kwargs.setdefault('input_cmap', 'gray')
        kwargs.setdefault('input_alpha', 1.)
        plot_with_kwargs(ax.imshow, [ma_input], fn_prefix='input', **kwargs)
    # remove xticks, yticks
    [kwargs.setdefault(k, v) for k, v in
    {'axis':'on', 'xticks':[], 'yticks':[]}.items()]
    ax.axis(kwargs.get('axis'))
    ax.set_xticks(kwargs.get('xticks'))
    ax.set_yticks(kwargs.get('yticks'))
    # save file if given
    if filename:
        kwargs.setdefault('dpi', 600.)
        plot_with_kwargs(fig.savefig, [filename], **kwargs)
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
