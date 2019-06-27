import numpy as np
import scipy
from scipy.io import loadmat
from skimage.transform import resize
import torch

def gabor_filter(theta, sigma, wavelength, filter_shape, gamma=0.3):
    """
    Create gabor filter

    Parameters
    ----------
    theta : float
        orientation (in degrees) of gabor filter
    sigma : float
        width of gabor filter
    wavelength : float
        wavelength of gabor filter
    filter_shape : list or tuple
        image shape of filter
    gamma : float
        aspect ratio [default: 0.3]

    Returns
    -------
    weight : torch.tensor
        gabor filter with shape filter_shape

    Examples
    --------
    >>> # create 45 degree gabor filter
    >>> weight = gabor_filter(45., 2.8, 3.5, [7,7])

    Notes
    -----
    Default gamma value from Serre et al. (2007).
    """
    assert len(filter_shape) == 2
    # convert orientation to radians
    theta = torch.tensor((theta / 180.) * np.pi)
    # get x, y coordinates for filter (centered)
    x = torch.arange(filter_shape[0]) - filter_shape[0] // 2
    y = torch.arange(filter_shape[1]) - filter_shape[1] // 2
    x, y = torch.stack(torch.meshgrid(x, y), dim=0).float()
    # update based on orientation
    x_0 = x * torch.cos(theta) + y * torch.sin(theta)
    y_0 = -x * torch.sin(theta) + y * torch.cos(theta)
    # create weight for filter
    weight = torch.mul(torch.exp(-(x_0**2 + gamma**2 * y**2)/(2. * sigma**2)),
                       torch.cos((2. * np.pi / wavelength) * x_0))
    return weight


def param_search(fn, args, kwargs, param_name, bounds, Ns, multi=None):
    """
    Search parameter space for given function

    Parameters
    ----------
    fn : function
        function to return values during parameter search
    args : list
        arguments passed to fn (i.e. fn(*args, **kwargs))
    kwargs : dict
        keyword arguments passed to fn, which includes parameter of interest
    param_name : str or list
        parameter name to update (or list of path to update within kwargs)
        if list, set_deepattr will be used (see example)
    bounds : list or tuple
        bounds of parameter search space with len(bounds) == 2
    Ns : int
        number of points to test within search space
    multi : float, optional
        multiplyer to separate Ns points starting from bounds[0] (see example)

    Returns
    -------
    costs : list
        list of results from each fn call (len(costs) == Ns)

    Examples
    --------
    # get MNIST test data
    testset = torchvision.datasets.MNIST(root='../data', train=False,
                                         transform=torchvision.transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                             shuffle=True, num_workers=2)
    # create Deep Belief Network
    model = rf_pool.models.DeepBeliefNetwork()
    model.append('0', rf_pool.modules.RBM(hidden=torch.nn.Conv2d(1, 32, 5),
                                          vis_activation_fn=torch.nn.Sigmoid()))
    # initialize optimizer
    optimizer = torch.optim.SGD(model.layers['0'].parameters(), lr=1e-8)
    # create param_name path to update param 'lr' in optimizer.param_groups[0]
    param_name = ['optimizer', 'param_groups', 0, 'lr']
    # search 'lr' parameter space
    costs = param_search(model.train_layer, ['0', 1, testloader],
                         {'optimizer': optimizer, 'monitor': len(testloader)},
                         param_name, (1e-8, 1e-1), 10, multi=5.)
    """
    assert type(bounds) is list or type(bounds) is tuple
    assert len(bounds) == 2
    assert Ns > 0
    if type(param_name) is not list:
        param_name = [param_name]
    # set param_space
    if multi is not None:
        assert bounds[0] != 0.
        param_space = [bounds[0] * pow(multi, i) for i in range(Ns)]
    else:
        param_space = np.linspace(bounds[0], bounds[1], Ns)
    print('Parameter search space: ', param_space)
    # for each value, update parameter and get cost
    cost = []
    for param in param_space:
        print('Parameter value: ', param)
        kwargs = set_deepattr(kwargs, param_name, param)
        cost.append(fn(*args, **kwargs))
    return cost

def repeat(x, repeats):
    """
    Perform numpy-like repeat function

    Parameters
    ----------
    x : torch.Tensor
        tensor to be repeated across dimensions
    repeats : tuple or list
        number of repeats for each dimension

    Returns
    -------
    y : torch.Tensor
        repeated tensor

    Examples
    --------
    >>> x = torch.as_tensor([[1.,2.],[3.,4.]])
    >>> y = repeat(x, (2, 3))
    >>> print(y)
    tensor([[1., 1., 1., 2., 2., 2.],
            [1., 1., 1., 2., 2., 2.],
            [3., 3., 3., 4., 4., 4.],
            [3., 3., 3., 4., 4., 4.]])
    """
    y = x.detach().numpy()
    for i, r in enumerate(repeats):
        y = np.repeat(y, r, i)
    return torch.as_tensor(y)

def to_numpy(x):
    if type(x) is torch.Tensor:
        x = x.numpy()
        is_tensor = True
    else:
        is_tensor = False
    return x, is_tensor

def to_tensor(x, is_tensor, dtype=None):
    if dtype is None:
        dtype = x.dtype
    elif type(dtype) is str and is_tensor:
        dtype = getattr(torch, dtype)
    if is_tensor:
        x = torch.as_tensor(x, dtype=dtype)
    else:
        x = np.asarray(x, dtype=dtype)
    return x

def demean(x, dims=(1,2)):
    """
    #TODO:WRITEME
    """
    if type(x) is torch.Tensor:
        x = x - torch.mean(x, dim=dims, keepdim=True)
    else:
        x = x - np.mean(x, axis=dims, keepdims=True)
    return x

def normalize(x, dims=(1,2)):
    """
    #TODO:WRITEME
    """
    x, is_tensor = to_numpy(x)
    x = (x - np.mean(x, axis=dims, keepdims=True)) / np.std(x, axis=dims, keepdims=True)
    x = to_tensor(x, is_tensor)
    return x

def normalize_range(x, dims=(1,2)):
    """
    #TODO:WRITEME
    """
    x, is_tensor = to_numpy(x)
    x = x - np.min(x, axis=dims, keepdims=True)
    x = x / (np.max(x, axis=dims, keepdims=True) + 1e-6)
    x = to_tensor(x, is_tensor)
    return x

def fast_whiten(data, Qss_file='Qss_kyoto.mat', Qnn=0.5):
    """
    #TODO:WRITEME
    """
    data, is_tensor = to_numpy(data)
    # normalize mean and std
    data = normalize(data)
    # load Qss_freq
    Qss_freq = loadmat(Qss_file)['Qss_freq']
    filt = np.sqrt(Qss_freq)/(Qss_freq + Qnn)
    # whiten each image
    wdata = []
    for i in range(len(data)):
        # resize filter
        filt_i = resize(filt, data[i].shape[-2:], order=3, mode='constant')
        If = scipy.fftpack.fft2(data[i])
        wdata.append(scipy.real(scipy.fftpack.ifft2(If * scipy.fftpack.fftshift(filt_i))))
    wdata = to_tensor(wdata, is_tensor, dtype='float32')
    return wdata

def kwarg_fn(modules=[list, dict, __builtins__, np, torch], x=None, **kwargs):
    """
    #TODO:WRITEME
    """
    # for each kwarg item, get/apply function
    for key, value in kwargs.items():
        # get function
        fn = None
        for module in modules:
            if hasattr(module, key):
                fn = getattr(module, key)
                break
        no_input = (x is None)
        if fn is None and hasattr(x, key):
            fn = getattr(x, key)
            no_input = True
        elif fn is None:
            continue
        # apply function
        output = None
        if type(value) is list and no_input:
            output = fn(*value)
        elif type(value) is list:
            output = fn(x, *value)
        elif type(value) is dict and no_input:
            output = fn(**value)
        elif type(value) is dict:
            output = fn(x, **value)
        if output is not None and x is not None:
            x = output
    return x

def get_deepattr(obj, path):
    for key in path:
        obj = get_attributes(obj, [key]).get(key)
    return obj

def set_deepattr(obj, path, value):
    for i in range(len(path)):
        obj_i = get_deepattr(obj, path[:-(i+1)])
        key = path[-(i+1)]
        if type(key) is int:
            key = str(key)
        set_attributes(obj_i, **{key: value})
        value = obj_i
    return value

def get_attributes(obj, keys, default=None):
    output = {}
    for key in keys:
        if type(obj) is list and type(key) is int:
            output.update({key: obj[key]})
        elif hasattr(obj, key):
            output.update({key: getattr(obj, key)})
        elif hasattr(obj, 'get') and key in obj:
            output.update({key: obj.get(key)})
        else:
            output.setdefault(key, default)
    return output

def pop_attributes(obj, keys, default=None):
    output = {}
    for key in keys:
        if hasattr(obj, 'pop') and key in obj:
            output.update({key: obj.pop(key)})
        elif hasattr(obj, key):
            output.update({key: getattr(obj, key)})
            delattr(obj, key)
        else:
            output.setdefault(key, default)
    return output

def set_attributes(obj, **kwargs):
    for key, value in kwargs.items():
        if type(obj) is list:
            obj[int(key)] = value
        elif hasattr(obj, 'update'):
            obj.update({key: value})
        else:
            setattr(obj, key, value)

def get_max_location(input, out_shape=None, normalized_units=True):
    h,w = input.shape[-2:]
    if out_shape is None:
        out_shape = [h,w]
    elif type(out_shape) is int:
        out_shape = [out_shape]
    scale = [out_shape[0]/h, out_shape[-1]/w]
    if input.ndimension() == 4:
        input = torch.max(input, 1)[0]
    max_index = torch.max(torch.flatten(input, -2), -1)[1]
    row = (max_index // w).type(input.dtype)
    col = (max_index % w).type(input.dtype)
    if normalized_units:
        row = 2. * row / (h - 1) - 1.
        col = 2. * col / (w - 1) - 1.
    else:
        row = torch.round(row * scale[0]).type(torch.int)
        col = torch.round(col * scale[-1]).type(torch.int)
    return row, col

def one_hot(i, n_classes): #TODO: allow i to be tensor with shape[0] > 1
    output = torch.zeros(n_classes)
    output[i] = 1.
    return output

def vector_norm(x, axis=-1):
    return torch.sqrt(torch.sum(torch.pow(x, 2), axis))

def cosine_similarity(feat_i, feat_j):
    """
    Computes the mean cosine similarity across
    every permutation pair of the features
    in feat_i and feat_j

    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to class label i
    feat_j: torch.tensor
        features corresponding to class label j

    Returns
    -------
    cosine_sim: torch.tensor
        vector of cosine similarities
    """
    # compute inner products
    inner = torch.matmul(feat_i, feat_j.t())

    # calculate the norms
    norm_i = vector_norm(feat_i).reshape(1,-1)
    norm_j = vector_norm(feat_j).reshape(1,-1)
    norm = torch.matmul(norm_i.t(), norm_j)

    # avarage over the cosine similarity
    cosine_sim = torch.div(inner, norm)

    return cosine_sim

def pairwise_cosine_similarity(feat_i, feat_j, axis=-1):
    """
    Computes pair-wise cosine similarity
    between features in feat_i and feat_j

    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to class label i
    feat_j: torch.tensor
        features corresponding to class label j

    Returns
    -------
    cosine_sim: torch.tensor
        vector of cosine similarities
    """
    # inner product
    inner = torch.sum(torch.mul(feat_i, feat_j), axis)

    # norms
    norm_i = vector_norm(feat_i, axis)
    norm_j = vector_norm(feat_j, axis)
    norm = torch.mul(norm_i, norm_j)

    # avarage over the cosine similarity
    cosine_sim = torch.div(inner, norm)

    return cosine_sim

def pooled_cosine_similarity(feat_i, feat_j):
    """
    Computes a receptive field interference score:
    mean cosine similarity between feat_i and max(feat_i, feat_j)

    Parameters
    ----------
    feat_i: torch.tensor
        features corresponding to first class label
    feat_j: torch.tensor
        features corresponding to second class label

    Returns
    -------
    interference: torch.tensor
        vector of cosine_sim
    """
    n_i = feat_i.shape[0]
    n_j = feat_j.shape[0]
    feat_i_r = torch.unsqueeze(feat_i, -1).repeat(1,1,n_j)
    feat_j_r = torch.unsqueeze(feat_j, -1).repeat(1,1,n_i)

    # max across channels
    feat_p = torch.max(feat_i_r, feat_j_r.permute(2,1,0))

    # inner product
    inner = torch.sum(torch.mul(feat_i_r, feat_p), 1)

    # norms
    norm_i = vector_norm(feat_i)
    norm_i = torch.unsqueeze(norm_i, -1).repeat(1,n_j)
    norm_j = vector_norm(feat_p)
    norm = torch.mul(norm_i, norm_j)

    # average over cosine similarity
    cosine_sim = torch.div(inner, norm)

    return cosine_sim

def confusion_matrix(feature_vectors, labels, interference_fn=cosine_similarity):
    """
    Computes a mean cosine similarity matrix
    from a set of feature vectors and corresponding class labels

    Parameters
    ----------
    feature_vectors: torch.tensor
        set of all corresponding class feature vectors
    labels: torch.tensor
        set of all class labels
    interference_fn: visualize.fn
        computes an interference score between features
    Returns
    -------
    matrix: torch.tensor
        confusion matrix of mean rf-interference scores
    unique_labels: torch.tensor
       class labels corresponding to matrix entries
    """
    unique_labels = torch.unique(labels, sorted=True)
    mean_matrix = torch.zeros((len(unique_labels),)*2)
    std_matrix = torch.zeros((len(unique_labels),)*2)

    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            feat_i = feature_vectors[np.where(labels == label_i)[0]]
            feat_j = feature_vectors[np.where(labels == label_j)[0]]
            # compute the mean cosine similarity cosine
            mean_matrix[i, j], std_matrix[i,j]  = interference_fn(feat_i, feat_j)

    return mean_matrix, std_matrix, unique_labels

if __name__ == '__main__':
    import doctest
    doctest.testmod()
