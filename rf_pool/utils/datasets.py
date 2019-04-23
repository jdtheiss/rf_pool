import glob
import os.path

import imageio
import numpy as np
from PIL import Image
import torch
import torchvision
import warnings

from . import stimuli, functions

class Dataset(torch.utils.data.Dataset):
    """
    #TODO:WRITEME
    """
    def __init__(self):
        self.root = None
        self.data_info = {}
        self.data = None
        self.labels = None
        self.transform = None

    def set_data(self):
        pass

    def set_labels(self):
        pass

    def set_data_info(self, keys, labels):
        for i, key in enumerate(keys):
            if labels is not None and i < len(labels):
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

    def apply_transform(self, transform=None, **kwargs):
        # apply a specific transform to data
        data = []
        for i, d in enumerate(self.data):
            d = functions.kwarg_fn([self, list, dict, __builtins__, np, torch],
                                   d, **kwargs)
            if transform:
                d = transform(d)
            data.append(d)
        if np.all([type(d) is torch.Tensor and d.ndimension()==4 for d in data]):
            self.data = torch.cat(data)
        else:
            self.data = data

    def __getitem__(self, index):
        img = self.data[index]
        if self.labels is not None and len(self.labels) > index:
            label = self.labels[index]
        else:
            label = 0
        # ensure correct number of dimensions
        if type(self.data) is torch.Tensor:
#             img = img.unsqueeze(-1)
            img = img.numpy()
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def __len__(self):
        return len(self.data)

class FilesDataset(Dataset):
    def __init__(self, root, files=[], labels=None, transform=None, **kwargs):
        super(FilesDataset, self).__init__()
        self.root = root
        self.transform = transform
        # set data_info, data, labels using patterns and pattern_labels
        for i, file in enumerate(files):
            if type(labels) is list and len(labels) > i:
                label = labels[i]
            else:
                label = labels
            pattern = os.path.abspath(os.path.join(self.root, file))
            file_names = glob.glob(pattern)
            self.set_data_info(file_names, label)
        self.set_data(**kwargs)
        self.set_labels(**kwargs)

    def set_data(self, load_fn=imageio.imread, load_transform=None, **kwargs):
        # load data from file_names using load_fn and apply transform
        data = []
        for file_name in self.data_info.keys():
            d = load_fn(file_name)
            d = functions.kwarg_fn([self, list, dict, __builtins__, np, torch],
                                   d, **kwargs)
            if load_transform:
                d = load_transform(d)
            data.append(d)
        if np.all([type(d) is torch.Tensor and d.ndimension()==4 for d in data]):
            self.data = torch.cat(data)
        else:
            self.data = data

    def set_labels(self, label_dtype=torch.uint8, **kwargs):
        # convert data_info to labels
        self.labels = list(self.data_info.values())
        if np.all([label is not None for label in self.labels]):
            self.labels = torch.as_tensor(self.labels, dtype=label_dtype)
        else:
            self.labels = None

class TripletDataset(Dataset):
    def __init__(self, dataset, positive_labels={}, negative_labels={},
                 transform=None):
        super(TripletDataset, self).__init__()
        self.root = dataset.root
        self.transform = transform
        # set data, labels from dataset
        self.set_data(dataset)
        self.set_labels(dataset)
        assert self.data is not None, ('dataset must have attribute "data"')
        assert self.labels is not None, ('dataset must have attribute "labels"')
        # set data_info, update with postive/negative labels
        patterns = [key for key in positive_labels.keys()]
        patterns.extend([key for key in negative_labels.keys()])
        patterns = np.unique(patterns)
        self.set_data_info(patterns, None)
        for pattern in patterns:
            self.update_data_info(pattern, (positive_labels.get(pattern),
                                  negative_labels.get(pattern)))

    def set_data(self, dataset):
        # set data, labels from dataset
        data_keys = ['data', 'train_data', 'test_data']
        data = None
        for data_key in data_keys:
            if hasattr(dataset, data_key):
                data = getattr(dataset, data_key)
                break
        self.data = data

    def set_labels(self, dataset):
        label_keys = ['labels', 'train_labels', 'test_labels']
        labels = None
        for label_key in label_keys:
            if hasattr(dataset, label_key):
                labels = getattr(dataset, label_key)
                break
        self.labels = labels

    def __getitem__(self, index):
        # get data and label for index
        img = self.data[index]
        label = self.labels[index]
        if type(label) is torch.Tensor:
            label = label.item()
        # get positive, negative labels and select random data for each
        positive_labels, negative_labels = self.data_info.get(label)
        positive = negative = None
        for new_index in np.random.permutation(np.arange(len(self.data))):
            if new_index == index:
                continue
            elif positive is not None and negative is not None:
                break
            elif self.labels[new_index] in positive_labels:
                positive = self.data[new_index]
            elif self.labels[new_index] in negative_labels:
                negative = self.data[new_index]
        # ensure correct number of dimensions
        if type(self.data) is torch.Tensor:
            img = torch.unsqueeze(img, 0)
            positive = torch.unsqueeze(positive, 0)
            negative = torch.unsqueeze(negative, 0)
        # apply transforms
        if self.transform:
            img = self.transform(img)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return (img, positive, negative, label)

class CrowdedDataset(Dataset):
    """
    Converts a dataset from torch.torchvision.datasets into crowded stimuli with flankers

    Attributes
    ----------
    dataset : torchvision.datasets
        the dataset to be converted into a crowded dataset
    n_flank: int
        number of flankers for the crowded stimuli
    target_labels : list, optional
        labels of the dataset designated for targets
    flank_labels : list, optional
        labels of the dataset designated for flankers
    same_flank : bool, optional
        if True, all flankers will be the same image
    flank_repeats: bool, optional
        if True,
    transform : torchvision.transform, optional
        transforms the data
    **kwargs : dict
        The crowded stimul arguments
        see stimuli.make_crowded_stimuli for details

    Methods
    -------
    make_stimuli(self, **kwargs)
        makes crowded stimuli
    """

    def __init__(self, dataset, n_flank, target_labels=[], flank_labels=[],
                 same_flank=False, flank_repeats=True, transform=None,
                 transform_init=None, offset=0, **kwargs):
        super(CrowdedDataset, self).__init__()
        self.n_flank = n_flank
        self.same_flank = same_flank
        self.flank_repeats = flank_repeats
        self.target_labels = target_labels
        self.flank_labels = flank_labels
        self.data = []
        self.labels = []
        self.groups = {}
        self.trackers = {}
        self.transform = transform
        self.transform_init = transform_init
        self.offset = offset

        # get data, labels from dataset
        if dataset.train and hasattr(dataset, 'train_data'):
            data = dataset.train_data
            labels = dataset.train_labels
        elif not self.train and hasattr(dataset, 'test_data'):
            data = dataset.test_data
            labels = dataset.test_labels
        elif hasattr(dataset, 'data'):
            data = dataset.data
            labels = dataset.labels
        else:
            raise Exception(
                'dataset has no attribute data, train_data, or test_data'
            )
            
        if type(data) is torch.Tensor:
            data = data.numpy()

        # # intialize groupings
        if len(self.target_labels) > 0 and len(self.flank_labels) == 0:
            all_labels = np.unique(self.target_labels)
        elif len(self.target_labels) == 0 and len(self.flank_labels) > 0:
            self.target_labels = list(np.unique(self.start_labels))
            all_labels = np.unique(self.target_labels + self.flank_labels)
        elif len(self.target_labels + self.flank_labels) == 0:
            self.target_labels = self.flank_labels = all_labels = np.unique(self.start_labels)
        else:
            all_labels = np.unique(self.target_labels + self.flank_labels)
        
        self.init_group_by_labels(data, labels, self.target_labels + self.flank_labels)
        
        # crowd the input images
        while np.all([(self.trackers[i] + self.n_flank) < len(self.groups[i])  for i in all_labels]):
            crowded_img, target_label = self.make_stimuli(self.n_flank, self.target_labels,
                                                          self.flank_labels, **kwargs)
            self.data.append(crowded_img)
            self.labels.append(target_label[0])
            
        self.data = torch.tensor(self.data, dtype=torch.float32)#, dtype=torch.uint8)
        self.labels = torch.tensor(self.labels)
        #TODO: the above approach does not get all targets

        #TODO: this approach takes forever
        # set indices for targets and flankers
#         if len(self.target_labels) == 0:
#             self.target_label = list(np.unique(labels))
#         if len(self.flank_labels) == 0:
#             self.flank_labels = list(np.unique(labels))
#         self.set_data_info(self.target_labels, labels)
#         self.set_data_info(self.flank_labels, labels)

#         # check if n_flank > len(flank_labels) and flank_repeats = False
#         if self.n_flank > len(self.flank_labels) and not self.flank_repeats:
#             warnings.warn("No repeats: n_flank set to len(flank_labels)")
#             self.n_flank = len(self.flank_labels)

#         # set data and labels
#         self.set_data(data, self.target_labels, self.flank_labels, self.n_flank,
#                       **kwargs)

#     def set_data_info(self, keys, labels):
#             for key in keys:
#                 self.data_info.update({key: np.where(labels==key)[0].tolist()})

#     def set_data(self, data, target_labels, flank_labels, n_flank, **kwargs):
#         self.data = []
#         self.labels = []
#         for target_label in target_labels:
#             # set target
#             for target_index in self.data_info.get(target_label):
#                 target = data[target_index]
#                 # get flank_label
#                 if self.same_flank:
#                     flank_labels_i = [target_label] * n_flank
#                 elif self.flank_repeats:
#                     rand_indices = np.random.randint(len(flank_labels), size=n_flank)
#                     flank_labels_i = np.array(flank_labels)[rand_indices]
#                 else:
#                     flank_labels_i = np.random.permutation(flank_labels)[:n_flank]
#                 # set flankers
#                 flanks = []
#                 for flank_label in flank_labels_i:
#                     # get flanker indices, remove target index
#                     flank_indices = self.data_info.get(flank_label)
#                     flank_indices = list(np.random.permutation(flank_indices))
#                     if target_index in flank_indices:
#                         flank_indices.remove(target_index)
#                     flanks.append(data[flank_indices[0]])
#                 # create crowded stimuli, append target_label
#                 self.data.append(stimuli.make_crowded_stimuli(target, flanks, **kwargs))
#                 self.labels.append(target_label)
#         self.data = torch.as_tensor(self.data, dtype=torch.uint8)

    def make_stimuli(self, n_flank, target_labels, flank_labels, **kwargs):
        target, target_label = self.sample_by_label(1, target_labels)
        flankers, _ = self.sample_by_label(n_flank, flank_labels)
        s = stimuli.make_crowded_stimuli(target[0], flankers, **kwargs)
        return s, target_label

    def sample_by_label(self, n, group_labels):
        if len(group_labels) > 0:
            if self.same_flank:
                choices = group_labels[np.random.randint(0, len(group_labels))]
                samples = [self.groups[choices][self.trackers[choices]]] * n
                self.trackers[choices] += 1
            else:
                if self.flank_repeats:
                    choices = np.array(group_labels)[np.random.randint(0, len(group_labels), size=n)]
                else:
                    if len(group_labels) < self.n_flank:
                        warnings.warn("No repeats: n_flank set to len(flank_labels)")
                    choices = np.random.permutation(group_labels)[:n]
                samples = []
                for i in choices:
                    samples.append(self.groups[i][self.trackers[i]])
                    self.trackers[i] += 1
        else:
            samples = []
            choices = []     
        if self.transform_init:
            samples = self.transform_samples(samples)  
        return samples, choices

    def init_group_by_labels(self, data, labels, group_labels):
        for label in group_labels:
            self.groups[label] = data[np.where(labels == label)]
            self.trackers[label] = 0
    def transform_samples(self, samples):
        return [torch.squeeze(self.transform_init(Image.fromarray(img, mode='L'))).numpy() for img in samples]
        
    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index] - self.offset
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform:
            img = self.transform(img)
        return (img, label)

class CrowdedCircles(torch.utils.data.Dataset):
    """
    Class for creating a dataset of crowded circles stimuli

    Attributes
    ----------
    n : int
        number of total images to be made
    label_type : str
        decides the label for training ('center' or 'mean')
    offset : int
        offset to subtract from target labels to reduce number of classes for
        cross entropy [default: 0]
    **kwargs : dict
        see stimuli.make_crowded_circles()

    Methods
    -------
    make_stimuli(self, **kwargs)
        makes self.n random crowded circle stimuli
    """
    def __init__(self, root, n, label_type, offset=0, train=True, download=False,
                 transform=None, **kwargs):
        self.root = root
        self.n = n
        self.label_type = label_type
        self.offset = offset
        self.train = train
        self.download = download
        self.transform = transform
        self.data = []
        self.labels = []

        self.train_data_file = None
        self.test_data_file = None
        # load in previously saved dataset (TODO)
        if self.download:
            if self.train:
                self.data, self.labels = torch.load(os.path.join(self.root, self.train_data_file))
            else:
                self.data, self.labels = torch.load(os.path.join(self.root, self.test_data_file))

            self.data = torch.load(data_file)
            self.labels = torch.load(label_fle)
        # make new dataset of size self.n from keyword arguments
        else:
            self.make_stimuli(**kwargs)

    def make_stimuli(self, **kwargs):
        for i in range(self.n):
            s, target_r, mean_r = stimuli.make_crowded_circles(**kwargs)
            self.data.append(s)
            if self.label_type.lower() == "center":
                self.labels.append(target_r)
            elif self.label_type.lower() == "mean":
                self.labels.append(mean_r)
            else:
                raise Exception("label type not undetstood")
        self.data = torch.tensor(self.data, dtype=torch.uint8)
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index] - self.offset
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def __len__(self):
        return len(self.labels)
