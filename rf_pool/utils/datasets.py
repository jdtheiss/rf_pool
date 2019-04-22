import numpy as np
from PIL import Image
import torch
import torchvision
import warnings

from . import stimuli

class CrowdedDataset(torch.utils.data.Dataset):
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
                 same_flank=False, flank_repeats=True, transform=None, **kwargs):

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

        # get the data
        if dataset.train and hasattr(dataset, 'train_data'):
            self.start_data = dataset.train_data
            self.start_labels = dataset.train_labels
        elif not self.train and hasattr(dataset, 'test_data'):
            self.start_data = dataset.test_data
            self.start_labels = dataset.test_labels
        elif hasattr(dataset, 'data'):
            self.start_data = dataset.data
            self.start_labels = dataset.labels
        else:
            raise Exception(
                'dataset has no attribute data, train_data, or test_data'
            )
        if type(self.start_data) is torch.Tensor:
            self.start_data = self.start_data.numpy()

        # intialize groupings
        if len(self.target_labels) > 0 and len(self.flank_labels) == 0:
            all_labels = np.unique(self.target_labels)
        elif len(self.target_labels) == 0 and len(self.flank_labels) > 0:
            self.target_labels = list(np.unique(self.start_labels))
            all_labels = np.unique(self.target_labels + self.flank_labels)
        elif len(self.target_labels + self.flank_labels) == 0:
            self.target_labels = self.flank_labels = all_labels = np.unique(self.start_labels)   
        else:
            all_labels = np.unique(self.target_labels + self.flank_labels)

        self.init_group_by_labels(all_labels)

        # crowd the input images
        while np.all([(self.trackers[i] + self.n_flank) < len(self.groups[i])  for i in all_labels]):
            crowded_img, target_label = self.make_stimuli(**kwargs)
            self.data.append(crowded_img)
            self.labels.append(target_label)
        self.data = torch.tensor(self.data, dtype=torch.uint8)
        self.labels = torch.tensor(self.labels)

    def make_stimuli(self, **kwargs):
        target, target_label = self.sample_by_label(1, self.target_labels)
        flankers, _ = self.sample_by_label(self.n_flank, self.flank_labels)
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
        return samples, choices

    def init_group_by_labels(self, group_labels):
        for label in group_labels:
            self.groups[label] = self.start_data[np.where(self.start_labels == label)]
            self.trackers[label] = 0

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def __len__(self):
        return len(self.labels)
            
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
