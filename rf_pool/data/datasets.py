from collections import OrderedDict
import glob
import io
import os.path
import pickle
import re
import urllib.request
import warnings

import imageio
import IPython.display
from IPython.display import clear_output, display
import numpy as np
from PIL import Image
import torch
import torchvision

from rf_pool.data import stimuli
from rf_pool.utils import functions

class Dataset(torch.utils.data.Dataset):
    """
    Base class for datasets
    """
    def __init__(self, **kwargs):
        self.root = None
        self.download = False
        self.timeout = 5.
        self.find_img_url = False
        self.url_pattern = '"([^\s]+\.jpg)"'
        self.url_replace = ['','']
        self.load_fn = imageio.imread
        self.data_info = OrderedDict()
        self.data = None
        self.extras = None
        self.labels = None
        self.transform = None
        self.label_map = OrderedDict()
        functions.set_attributes(self, **kwargs)

    def set_data(self):
        raise NotImplementedError

    def set_labels(self):
        raise NotImplementedError

    def get_data_labels(self, dataset, data_keys, label_keys):
        # get data, labels
        data = None
        labels = None
        for data_key in data_keys:
            if hasattr(dataset, data_key):
                data = getattr(dataset, data_key)
                break
        for label_key in label_keys:
            if hasattr(dataset, label_key):
                labels = getattr(dataset, label_key)
                break
        return data, labels

    def set_data_info(self, keys, labels):
        for i, key in enumerate(keys):
            if hasattr(labels, '__len__') and i < len(labels):
                label = labels[i]
            else:
                label = labels
            self.data_info.update({key: label})

    def update_data_info(self, pattern, label):
        # set labels for keys with pattern
        for key in self.data_info.keys():
            if type(key) is str and key.find(pattern) >= 0:
                self.data_info.update({key: label})
            elif key == pattern:
                self.data_info.update({key: label})

    def get_img_url(self, url, pattern='', replace=['',''], timeout=5.):
        try:
            img_url = None
            with urllib.request.urlopen(url, timeout=timeout) as response:
                html = response.read()
                img_url = re.findall(pattern, html.decode('utf-8'))
            if len(img_url) > 0:
                img_url = img_url[0]
                img_url = re.sub(*replace, img_url)
            else:
                img_url = None
        except Exception as detail:
            if img_url is not None:
                print('Error %a: %s' % (img_url, detail))
            else:
                print('Error: %s' % detail)
            img_url = None
        return img_url

    def download_image(self, url, id, download=False, timeout=5.):
        try:
            # if already downloaded, load
            fname = os.path.join(self.root, str(id))
            if os.path.isfile(fname):
                return self.load_fn(fname)
            # otherwise load load from html
            req = urllib.request.Request(url, headers={'User-Agent': 'Magic Browser'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                html = response.read()
                img = Image.open(io.BytesIO(html))
            if download:
                img.save(fname)
        except Exception as detail:
            print('Error: %s' % detail)
            img = None
        return img

    def to_numpy(self, x):
        # convert to numpy
        if hasattr(x, 'numpy'):
            x = torch.squeeze(x)
            x = x.numpy()
        else:
            x = np.array(x)
        return x

    def to_Image(self, x):
        # convert to PIL Image
        if type(x) is not Image.Image and np.array(x).ndim >= 2:
            x = Image.fromarray(x)
        return x

    def apply_transform(self, data, transform=None, **kwargs):
        # apply a specific transform to data
        if transform is None and len(kwargs) == 0:
            return data
        output = []
        for i, d in enumerate(data):
            d = functions.kwarg_fn([self,functions,list,dict,__builtins__,np,torch],
                                   d, **kwargs)
            # convert to numpy, Image
            d = self.to_numpy(d)
            d = self.to_Image(d)
            # apply transform
            if transform:
                d = transform(d)
            output.append(d)
        return output

    def collate_fn(self, batch):
        batch = list(filter(lambda x : x is not None, batch))
        if len(batch) == 0:
            index = np.random.randint(self.__len__())
            return self.collate_fn([self.__getitem__(index)])
        return torch.utils.data.dataloader.default_collate(batch)

    def __getitem__(self, index):
        if isinstance(self.data, (dict, OrderedDict)) and type(index) is int:
            img = list(self.data.values())[index]
        else:
            img = self.data[index]
        if isinstance(self.labels, (dict, OrderedDict)) \
           and self.labels.get(img) is not None:
            label = self.labels.get(img)
        elif isinstance(self.labels, (dict, OrderedDict)) and type(index) is int:
            label = list(self.labels.values())[index]
        elif self.labels is not None:
            label = self.labels[index]
            if self.label_map.get(label) is not None:
                label = self.label_map.get(label)
        else:
            label = -1
        if label is None:
            label = -1
        try:
            # call if type is fn
            if type(img) is type(lambda x: None):
                img = img(0)
            # get url within html if necessary
            if self.find_img_url and type(img) is str:
                img = self.get_img_url(img, pattern=self.url_pattern,
                                       replace=self.url_replace,
                                       timeout=self.timeout)
            # download if necessary
            if not self.download and type(img) is str:
                img = self.download_image(img, index, timeout=self.timeout)
                if img is None:
                    return None
            # apply transform
            if self.transform:
                img = self.transform(img)
        except Exception as detail:
            print('Error: %s' % detail)
            return None
        if hasattr(self, 'extras') and self.extras is not None:
            output = (img, self.extras[index], label)
        else:
            output = (img, label)
        return output

    def __len__(self):
        return len(self.data)

class URLDataset(Dataset):
    def __init__(self, root, urls=[], base_url=None, ids=[''], labels=None,
                 download=False, find_img_url=False, **kwargs):
        super(URLDataset, self).__init__(root=root, labels=None, download=download,
                                         find_img_url=find_img_url, **kwargs)
        # set data_info using urls
        if len(urls) > 0:
            self.set_data_info(urls, labels)
            data_dict = OrderedDict([(k,v) for k, v in
                                     zip(np.arange(len(urls)), urls)])
        # set data_info using url_mapping
        elif base_url is not None:
            data_dict = OrderedDict()
            map_kwargs = OrderedDict()
            if 'key_pattern' in kwargs:
                map_kwargs.update({'key_pattern': kwargs.pop('key_pattern')})
            if 'value_pattern' in kwargs:
                map_kwargs.update({'value_pattern': kwargs.pop('value_pattern')})
            for i, id in enumerate(ids):
                clear_output(wait=True)
                display('getting urls')
                display('progress: %g%%' % (100. * (i + 1) / len(ids)))
                mapping = self.url_mapping(id, base_url, **map_kwargs)
                data_dict.update(mapping)
                if type(labels) is list:
                    self.set_data_info(mapping.keys(), labels[i])
                else:
                    self.set_data_info(mapping.keys(), labels)
        self.set_data(data_dict, download=download, **kwargs)
        self.set_labels(download=download, **kwargs)

    def url_mapping(self, id, base_url, key_pattern='n\d+_\d+',
                    value_pattern='http.+\.jpg'):
        with urllib.request.urlopen(base_url + id) as response:
            html = response.readlines()
        mapping = OrderedDict()
        for line in html:
            key = re.findall(key_pattern, line.decode('utf-8'))
            value = re.findall(value_pattern, line.decode('utf-8'))
            if len(key) > 0 and len(value) > 0:
                mapping.update({key[0]: value[0]})
        return mapping

    def set_data(self, data_dict, download=False, load_transform=None, **kwargs):
        # get data from urls in data_dict
        if download:
            data = []
            for i, (id, url) in enumerate(data_dict.items()):
                clear_output(wait=True)
                display('downloading data')
                display('progress: %g%%' % (100. * (i + 1) / len(data_dict)))
                d = self.download_image(url, id, download)
                if d is None:
                    continue
                d = functions.kwarg_fn([self,functions,list,dict,__builtins__,np,torch],
                                       d, **kwargs)
                if load_transform:
                    d = load_transform(d)
                data.append(d)
            if np.all([type(d) is torch.Tensor and d.ndimension()==4 for d in data]):
                self.data = torch.cat(data)
            else:
                self.data = data
        else:
            self.data = data_dict.copy()

    def set_labels(self, download=False, label_dtype=torch.uint8, **kwargs):
        # get labels from data_info
        if download:
            self.labels = list(self.data_info.values())
            if np.all([label is not None for label in self.labels]):
                self.labels = torch.as_tensor(self.labels, dtype=label_dtype)
            else:
                self.labels = None
        else:
            self.labels = self.data_info.copy()

class SubsetDataset(Dataset):
    """
    Subset of a Dataset class

    This class wraps a dataset using a subset of the data indices

    Parameters
    ----------
    dataset_class : str
        name of dataset class to use (see Notes)
    indices : list
        indices to include in subset [default: None, full dataset used]
    index_file : str
        pickle file containing indices to use [default: None]
    **kwargs : **dict
        keyword arguments passed to initialize dataset

    Notes
    -----
    If `dataset_class` starts with 'torch', then `eval(dataset_class)` is used,
    otherwise the dataset class will be assumed to be from `rf_pool.data.datasets`.
    For example, the dataset can either be `torchvision.datasets.CocoDetection`
    or `CocoDetection`, which will use `rf_pool.data.datasets.CocoDetection`.
    """
    def __init__(self, dataset_class, indices=None, index_file=None, **kwargs):
        # initialize with dataset_class from rf_pool.data.datasets or torch
        if dataset_class.startswith('torch'):
            cls = eval(dataset_class)
        else:
            cls = globals().get(dataset_class)
        self.__class__.__bases__ = (cls,)
        super(SubsetDataset, self).__init__(**kwargs)

        # set indices
        self._indices = indices

        # load from index_file
        if index_file:
            self._indices = pickle.load(open(index_file, 'rb'))

    def __len__(self):
        # return len of indices
        if self._indices:
            return len(self._indices)
        return super().__len__()

    def __getitem__(self, index):
        # get index relative to indices
        if self._indices:
            index = self._indices[index]
        return super().__getitem__(index)

class ConcatDataset(torch.utils.data.Dataset):
    """
    Concatenate Datasets as [d_i[index] for d_i in datasets]

    Parameters
    ----------
    datasets : *torch.utils.data.Dataset
        datasets to be concatenated
    """
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = list(datasets)

    def __len__(self):
        return max(len(d_i) for d_i in self.datasets)

    def __getitem__(self, index):
        data, label = zip(*[d_i[index % len(d_i)] for d_i in self.datasets])
        return torch.stack(data), torch.tensor(label)

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    CocoDetection dataset converted to pytorch format

    This class wraps the torchvision.datasets.CocoDetection class and converts
    `bbox` (xmin,ymin,w,h) to (xmin,ymin,xmax,ymax)

    Parameters
    ----------
    root : str
        Root directory where images are downloaded to.
    annFile : str
        Path to json annotation file.
    transform : function
        A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.ToTensor``
        [default: None]
    target_transform : function
        A function/transform that takes in the target and transforms it.
        [default: None]
    transforms : function
        A function/transform that takes input sample and its target as entry
        and returns a transformed version. [default: None]
    """
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, annFile, transform,
                                            target_transform, transforms)

    def _convert_targets(self, targets):
        # convert targets to pytorch format (mostly bbox->boxes)
        instances = dict((k,[]) for k in ['boxes','labels','area','iscrowd'])
        for tgt in targets:
            # convert bbox (xmin,ymin,w,h) to (xmin,ymin,xmax,ymax)
            boxes = torch.tensor(tgt['bbox']).view(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            instances.get('boxes').append(boxes)
            # append labels, area, iscrowd
            instances.get('labels').append(torch.tensor([tgt['category_id']]))
            instances.get('area').append(torch.tensor([tgt['area']]))
            instances.get('iscrowd').append(torch.tensor([tgt['iscrowd']]))
        # concatenate tensors
        instances = dict((k, torch.cat(v)) for k, v in instances.items())
        instances.update({'image_id': targets[0]['image_id']})
        return instances

    def __getitem__(self, index):
        # get image, targets
        img, tgt = super().__getitem__(index)
        # convert targets
        return img, self._convert_targets(tgt)

    def collate_fn(self, batch):
        # return list[Tensor], list[dict]
        images = list(map(lambda x: x[0], batch))
        targets = list(map(lambda x: x[1], batch))
        return images, targets

class CrowdedDataset(torch.utils.data.Dataset):
    """
    Converts a dataset from torch.torchvision.datasets into crowded stimuli
    with flankers

    Attributes
    ----------
    dataset : torchvision.datasets
        the dataset to be converted into a crowded dataset
    n_flankers : int
        number of flankers for the crowded stimuli
    axis : float
        initialization axis for flankers in radians
    n_images : int
        number of crowded stimuli images to create
    target_labels : list, optional
        labels of the dataset designated for targets
    flanker_labels : list, optional
        labels of the dataset designated for flankers
    repeat_flankers : bool, optional
        if True, allows flanker labels to be repeated
    same_flankers : bool, optional
        if True, all flankers will be the same label
    target_flankers : bool, optional
        if True, all flankers will be same label as target
    transform : torchvision.transform, optional
        transform applied to the data during __getitem__ call
    label_map : dict, optional
        custom mapping applied to the label during __getitem__ call
    no_target : bool
        if True, leaves target location empty
    **kwargs : dict
        crowded stimul arguments
        see stimuli.make_crowded_stimuli for details

    Methods
    -------
    get_data_labels(dataset, data_keys, label_keys)
        return *data, *labels attributes from dataset (e.g., dataset.train_data)
    set_data_info(keys, labels)
        set the data_info dictionary for CrowdedDataset
        keys should be target_labels+flanker_labels;
        labels should be obtained using get_data_labels()
    set_data_labels(dataset, n_images, n_flankers, target_labels, flanker_labels,
                    **kwargs)
        set data, labels attributes for CrowdedDataset
        Note: dataset.transform will be applied for each image in crowded stimuli
    """
    def __init__(self, dataset, n_flankers, axis, target_labels=[],
                 flanker_labels=[], transform=None, label_map={},
                 load_previous=False, no_target=False, **kwargs):
        super(CrowdedDataset, self).__init__()
        self.dataset = dataset
        self.n_flankers = n_flankers
        self.axis = axis
        self.target_labels = target_labels.copy()
        self.flanker_labels = flanker_labels.copy()
        self.transform = transform
        self.label_map = label_map
        self.load_previous = load_previous
        self.no_target = no_target
        self.kwargs = kwargs

        if not self.load_previous:
            n_images = len(dataset)
            self.target_labels = np.random.permutation(np.arange(n_images))
            self.flanker_labels = [np.random.randint(0, n_images, n_flankers) for _ in range(n_images)]

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        
        target_index = self.target_labels[index]
        if isinstance(target_index, list):
            target_index = target_index[0]
        flanker_index = self.flanker_labels[index]
        
        tgt_img, tgt_label = self.dataset[target_index]

        if self.n_flankers:
            flnk_img, flnk_label = zip(*[self.dataset[i] for i in flanker_index[:self.n_flankers]])
        else:
            flnk_img, flnk_label = [torch.zeros_like(tgt_img)], -1

        if self.no_target:
            tgt_img = [torch.zeros_like(tgt) for tgt in tgt_img]
            tgt_label = -1
            
        crowd_img = stimuli.make_crowded_stimuli(tgt_img[0], torch.cat(flnk_img),
                                                 axis=self.axis, **self.kwargs)
        
        if self.label_map:
            tgt_label = self.label_map.get(tgt_label) or -1

        # apply transform
        if self.transform:
            crowd_img = self.transform(crowd_img)

        return crowd_img, tgt_label