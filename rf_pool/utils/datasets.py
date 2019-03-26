import numpy as np
from PIL import Image
import torch
import torchvision

from . import stimuli

class CrowdedMNIST(torchvision.datasets.MNIST):
    """
    Converts an MNIST dataset into crowded stimuli with flankers

    Attributes
    ----------
    n_flank: int
        number of flankers for the crowded stimuli
    **kwargs : dict
        The crowded stimul arguments
        see stimuli.make_crowded_stimuli for details
    """

    def __init__(self, root, n_flank, download=True,
                 train=True, transform=None, **kwargs):

        super(CrowdedMNIST, self).__init__(root=root, train=train,
                                           download=download, transform=transform)
        self.n_flank = n_flank
        if self.train:
            inputs = self.train_data
            labels = self.train_labels
        else:
            inputs = self.test_data
            labels = self.test_labels

        if type(inputs) is torch.Tensor:
            inputs = inputs.numpy()
        # get the targets
        n_set = inputs.shape[0]
        target_indices = np.arange(0, n_set, n_flank+1)
        # crowd the input images
        crowded_inputs = []
        for i in target_indices:
            target = inputs[i]
            flankers = inputs[i+1:i+n_flank+1]
            s = stimuli.make_crowded_stimuli(target, flankers, **kwargs)
            crowded_inputs.append(s)
        # get the labels
        crowded_labels = labels[target_indices]
        # reassign to the dataset
        if self.train:
            self.train_data = torch.tensor(crowded_inputs, dtype=torch.uint8)
            self.train_labels = crowded_labels
        else:
            self.test_data = torch.tensor(crowded_inputs, dtype=torch.uint8)
            self.test_labels = crowded_labels


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
