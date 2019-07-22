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
    def __init__(self, **kwargs):
        self.root = None
        self.data_info = {}
        self.data = None
        self.extras = None
        self.labels = None
        self.transform = None
        self.label_map = {}
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

    def __getitem__(self, index):
        img = self.data[index]
        if self.labels is not None and len(self.labels) > index:
            label = self.labels[index]
            if self.label_map.get(label) is not None:
                label = self.label_map.get(label)
        else:
            label = -1
        # convert to numpy, Image
        img = self.to_numpy(img)
        img = self.to_Image(img)
        # apply transform
        if self.transform:
            img = self.transform(img)
        if hasattr(self, 'extras') and self.extras is not None:
            output = (img, self.extras[index], label)
        else:
            output = (img, label)
        return output

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
            d = functions.kwarg_fn([self,functions,list,dict,__builtins__,np,torch],
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

class SearchDataset(Dataset):
    """
    #TODO:WRITEME
    """
    def __init__(self, dataset, n_distractors, n_images, target_labels=[],
                 distractor_labels=[], target_loc=[], distractor_locs=[],
                 label_map={}, transform=None, **kwargs):
        super(SearchDataset, self).__init__()
        self.n_distractors = n_distractors
        self.n_images = n_images
        self.target_labels = target_labels
        self.distractor_labels = distractor_labels
        self.target_loc = target_loc
        self.distractor_locs = distractor_locs
        self.label_map = label_map
        self.transform = transform

        # get labels from dataset
        _, labels = self.get_data_labels(dataset, [],
                                         ['labels','train_labels','test_labels'])

        # set target_labels, flanker_labels
        if len(self.target_labels) == 0:
            self.target_labels = np.unique(labels).tolist()
        if len(self.distractor_labels) == 0:
            self.distractor_labels = np.unique(labels).tolist()

        # ensure target_loc, distractor_locs are list of lists
        if len(self.target_loc) == 0 or type(self.target_loc[0]) is not list:
            self.target_loc = [self.target_loc]
        if len(self.distractor_locs) == 0 or type(self.distractor_locs[0]) is not list:
            self.distractor_locs = [self.distractor_locs]

        # set data_info
        self.set_data_info(self.target_labels + self.distractor_labels, labels)

        # set data
        self.set_data_labels(dataset, self.n_images, self.n_distractors,
                             self.target_labels, self.distractor_labels,
                             self.target_loc, self.distractor_locs, **kwargs)

    def set_data_info(self, keys, labels):
        self.data_info = {}
        for key in keys:
            self.data_info.update({key: np.where(key==labels)[0].tolist()})

    def set_data_labels(self, dataset, n_images, n_distractors, target_labels,
                        distractor_labels, target_loc, distractor_locs, **kwargs):
        self.data = []
        self.locations = []
        self.labels = []
        for n in range(n_images):
            # sample target/flanker labels
            target_label_n = self.sample_label(target_labels, 1)[0]
            distractor_labels_n = self.sample_label(distractor_labels, n_distractors)
            # sample target/flanker data
            target = self.sample_data(dataset, [target_label_n])[0]
            distractors = self.sample_data(dataset, distractor_labels_n)
            # permute target_loc, distractor_locs
            target_loc_i = np.random.permutation(target_loc)[0]
            distractor_locs_i = np.random.permutation(distractor_locs)[:n_distractors]
            # create crowded stimuli
            stim, loc = stimuli.make_search_stimuli(target, distractors,
                                               target_loc=target_loc_i,
                                               distractor_locs=distractor_locs_i,
                                               **kwargs)
            self.data.append(stim)
            self.locations.append(loc)
            self.labels.append(target_label_n)

    def sample_data(self, dataset, labels):
        data = []
        for label in labels:
            indices = self.data_info.get(label)
            index = indices.pop(0)
            self.data_info.update({label: indices + [index]})
            data.append(self.to_numpy(dataset[index][0]))
        return data

    def sample_label(self, labels, n):
        return np.random.permutation(labels)[:n]

    def __getitem__(self, index):
        img = self.data[index]
        loc = self.locations[index]
        if self.labels is not None and len(self.labels) > index:
            label = self.labels[index]
            if self.label_map.get(label) is not None:
                label = self.label_map.get(label)
        else:
            label = -1
        # convert to numpy, Image
        img = self.to_numpy(img)
        img = self.to_Image(img)
        # apply transform
        if self.transform:
            img = self.transform(img)
        return (img, loc, label)

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
        self.data_info = {}
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
