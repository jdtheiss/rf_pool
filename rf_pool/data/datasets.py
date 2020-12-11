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

class CrowdedDataset(Dataset):
    """
    Converts a dataset from torch.torchvision.datasets into crowded stimuli
    with flankers

    Attributes
    ----------
    dataset : torchvision.datasets
        the dataset to be converted into a crowded dataset
    n_flankers : int
        number of flankers for the crowded stimuli
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
    def __init__(self, dataset, n_flankers, n_images, target_labels=[],
                 flanker_labels=[], repeat_flankers=True, same_flankers=False,
                 target_flankers=False, transform=None, label_map={},
                 load_previous=False, no_target=False, **kwargs):
        super(CrowdedDataset, self).__init__()
        self.n_flankers = n_flankers
        self.n_images = n_images
        self.target_labels = target_labels.copy()
        self.flanker_labels = flanker_labels.copy()
        self.repeat_flankers = repeat_flankers
        self.same_flankers = same_flankers
        self.target_flankers = target_flankers
        self.transform = transform
        self.label_map = label_map
        self.load_previous = load_previous
        self.no_target = no_target

        # get labels from dataset
        _, labels = self.get_data_labels(dataset, [],
                                         ['labels','train_labels','test_labels'])

        if not self.load_previous:
            # set target_labels, flanker_labels
            if len(self.target_labels) == 0:
                self.target_labels = np.unique(labels).tolist()
            if len(self.flanker_labels) == 0:
                self.flanker_labels = np.unique(labels).tolist()
            assert self.repeat_flankers or self.n_flankers <= len(self.flanker_labels), (
                'if repeat_flankers = True, n_flankers must be <= len(flanker_labels)'
            )

            # set data_info
            self.set_data_info(self.target_labels + self.flanker_labels, labels)

        # set data
        self.set_data_labels(dataset, self.n_images, self.n_flankers,
                             self.target_labels, self.flanker_labels, **kwargs)

    def set_data_info(self, keys, labels):
        self.data_info = OrderedDict()
        for key in keys:
            self.data_info.update({key: np.where(key==labels)[0].tolist()})

    def set_data_labels(self, dataset, n_images, n_flankers, target_labels,
                        flanker_labels, **kwargs):
        self.data = []
        self.labels = []
        self.recorded_target_indices = []
        self.recorded_flanker_indices = []
        for n in range(n_images):
            # sample target/flanker labels
            target_label_n = self.sample_label(target_labels, 1)[0]
            flanker_labels_n = self.sample_label(flanker_labels, n_flankers,
                                                 target_label_n)
            # sample target/flanker data
            target, target_record = self.sample_data(dataset, [target_label_n])
            flankers, flanker_record = self.sample_data(dataset, flanker_labels_n)

            # create crowded stimuli
            if self.no_target:
                target_input = np.zeros_like(target[0])
            else:
                target_input=target[0]

            crowded_stimuli = stimuli.make_crowded_stimuli(target_input, flankers, **kwargs)
            self.data.append(crowded_stimuli)

            if self.load_previous:
                self.labels.append(self.label_map[int(dataset[target_label_n][1])])
            else:
                self.labels.append(target_label_n)
            # append recorded indices information
            self.recorded_target_indices.append(target_record)
            self.recorded_flanker_indices.append(flanker_record)

    def sample_data(self, dataset, labels):
        data = []
        recorded_indices = []
        for label in labels:
            if self.load_previous:
                index = label
            else:
                indices = self.data_info.get(label)
                index = indices.pop(0)
                self.data_info.update({label: indices + [index]})
            data.append(self.to_numpy(dataset[index][0]))
            recorded_indices.append(index)
        return data, recorded_indices

    def sample_label(self, labels, n, target_label=None):
        copy_labels = labels.copy()
        if target_label in copy_labels:
            copy_labels.remove(target_label)
        if self.target_flankers and target_label is not None:
            copy_labels = [target_label] * n
        elif self.same_flankers:
            copy_labels = [np.random.permutation(copy_labels)[0]] * n
        elif self.repeat_flankers:
            rand_indices = np.random.randint(len(copy_labels), size=n)
            copy_labels = [copy_labels[i] for i in rand_indices]
        else:
            copy_labels = np.random.permutation(copy_labels)[:n]
        if self.load_previous:
            copy_labels = labels.pop(0)[:n]
        return copy_labels
