from collections import OrderedDict

import IPython.display
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import loadmat
import torch

def get_doc(docstr, field='', lines=[], end_field=None):
    start = docstr[:docstr.find(field)].count('\n')
    if end_field:
        end = docstr[:docstr.find(end_field)].count('\n')
        return '\n'.join(docstr.splitlines()[start:end])
    if len(lines) == 0:
        return '\n'.join(docstr.splitlines()[start:])
    lines = [start + idx for idx in lines]
    docstr = docstr.splitlines()
    return '\n'.join([docstr[idx] for idx in lines])

def update_doc(docstr, field='', lines=[], updates=[], replace=['','']):
    docstr = docstr.replace(*replace)
    if len(lines) == 0:
        return docstr
    assert len(lines) == len(updates)
    start = docstr[:docstr.find(field)].count('\n')
    lines = [start + idx for idx in lines]
    docstr = docstr.splitlines()
    for n, idx in enumerate(lines):
        if type(updates[n]) is list:
            updates[n] = '\n'.join(updates[n])
        docstr[idx] = '    %s' % updates[n]
    return '\n'.join(docstr)

def parse_list_args(n_iter, *args, **kwargs):
    if n_iter == 1:
        return [args], [kwargs]
    list_args = []
    list_kwargs = []
    for n in range(n_iter):
        list_args.append([])
        list_kwargs.append(OrderedDict())
        for arg in args:
            if type(arg) is list and len(arg) == n_iter:
                list_args[-1].append(arg[n])
            else:
                list_args[-1].append(arg)
        for k, v in kwargs.items():
            if type(v) is list and len(v) == n_iter:
                list_kwargs[-1].update({k: v[n]})
            else:
                list_kwargs[-1].update({k: v})
    return list_args, list_kwargs

def gabor_filter(theta, sigma, wavelength, filter_shape, gamma=0.3, psi=0.):
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
    psi : float
        offset [default: 0.]

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
    weight = torch.mul(torch.exp(-(x_0**2 + gamma**2 * y_0**2)/(2. * sigma**2)),
                       torch.cos(2. * np.pi * x_0 / wavelength + psi))
    return weight

def param_search(fn, args, kwargs, param_space, verbose=True):
    """
    Search parameter space for given function

    Parameters
    ----------
    fn : function
        function to return values during parameter search
    args : list
        arguments passed to fn (i.e. fn(*args, **kwargs))
    kwargs : dict
        keyword arguments passed to fn (set to {} if no kwargs)
    param_space : dict
        dictionary of (key, value) where key is the parameter to update and
        value is the search space for that parameter. The key can be of the
        following types:
            int (index of args),
            str (key of kwargs),
            tuple (path of kwargs; see set_deepattr)
        Search space must be the same length for each parameter given.
    verbose : bool
        True/False whether to show parameter search space on each iteration
        [default: True]

    Returns
    -------
    costs : list
        list of results from each fn call
        len(costs) == len(list(param_space.values())[0])

    Examples
    --------
    >>> # find parameter x closest to random value a
    >>> a = np.random.rand()
    >>> fn = lambda x, a: (x - a)**2
    >>> param_space = {0: np.linspace(0., 1., 10)}
    >>> costs = param_search(fn, [0, a], {}, param_space)

    See Also
    --------
    set_deepattr
    """
    # get length of search space
    n_search = len(list(param_space.values())[0])
    # check that all parameters have same search space length
    assert all([len(v) == n_search for v in param_space.values()])
    # check that all keys are int, str, or tuple
    assert all([isinstance(k, (int, str, tuple)) for k in param_space.keys()])
    # for each value, update parameter and get cost
    cost = []
    for i in range(n_search):
        # update params
        for k, v in param_space.items():
            if type(k) is int:
                args[k] = v[i]
            elif type(k) is str:
                kwargs.update({k: v[i]})
            elif type(k) is tuple:
                kwargs = set_deepattr(kwargs, k, v[i])
        # get cost
        cost_i = fn(*args, **kwargs)
        cost.append(cost_i)
        # display progress
        clear_output(wait=True)
        display('Progress: %0.2f%%' % (100. * (i+1) / n_search))
        if verbose:
            display('Cost: %a' % cost)
            display('Parameter value(s):')
            display('\n'.join([str((k, v[i])) for k, v in param_space.items()]))
            plt.plot(np.arange(i+1), cost)
            plt.show()
    # plot final cost
    clear_output(wait=True)
    if verbose:
        display('Cost: %a' % cost)
        plt.plot(np.arange(n_search), cost)
        plt.show()
    return cost

def bootstrap(*args, n_samples=1000, fn=np.mean, fn_kwargs={}):
    """
    Get a distribution of statistics by randomly sampling from a given set of
    values (with replacement)

    Parameters
    ----------
    *args : array-like
        input(s) to be resampled and passed to the statistic function
    n_samples : int
        number of resamples to perform [default: 1000]
    fn : function
        statistic function to apply to resampled distribution [default: np.mean]
    fn_kwargs : dict
        keyword arguments passed to the statistic function

    Returns
    -------
    stats : list
        distribution of statistics

    Examples
    --------
    >>> # get bootstrapped p-value for y > x
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100) + 5.
    >>> n_samples = 1000
    >>> fn = lambda x, y: np.greater(np.mean(x), np.mean(y))
    >>> p = np.mean(bootstrap(x, y, n_samples=n_samples, fn=fn))
    >>> print(p)
    0.0
    """
    stats = []
    for n in range(n_samples):
        idx = [np.random.randint(len(x), size=len(x)) for x in args]
        sampled_args = [np.array(x)[i] for x, i in zip(args, idx)]
        stats.append(fn(*sampled_args, **fn_kwargs))
    return stats

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
    if type(dtype) is str or is_tensor:
        dtype = getattr(torch, str(dtype))
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

def kwarg_fn(modules=[list, OrderedDict, dict, __builtins__, np, torch], x=None,
             **kwargs):
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
        elif (type(value) is dict or type(value) is OrderedDict) and no_input:
            output = fn(**value)
        elif (type(value) is dict or type(value) is OrderedDict):
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

def get_attributes(obj, keys, default=None, ignore_keys=False):
    output = OrderedDict()
    for key in keys:
        if type(key) is int:
            output.update({key: obj[key]})
        elif hasattr(obj, key):
            output.update({key: getattr(obj, key)})
        elif hasattr(obj, 'get') and key in obj:
            output.update({key: obj.get(key)})
        elif not ignore_keys:
            output.setdefault(key, default)
    return output

def pop_attributes(obj, keys, default=None, ignore_keys=False):
    output = OrderedDict()
    for key in keys:
        if hasattr(obj, 'pop') and key in obj:
            output.update({key: obj.pop(key)})
        elif hasattr(obj, key):
            output.update({key: getattr(obj, key)})
            delattr(obj, key)
        elif not ignore_keys:
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
