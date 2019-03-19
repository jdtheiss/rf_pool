import numpy as np
import torch 
import torchvision
import utils.stimuli as stimuli

class CrowdedMNIST(torchvision.datasets.MNIST):
    """
    Converts an MNIST dataset into crowded stimuli with flankers

    Attributes
    ----------
    dataset: torchvision.datasets.mnist.MNIST
        pytorch MNIST trainset or testset object
    n_flank: int
        number of flankers for the crowded stimuli
    **kwargs: dict
        The crowded stimul arguments
        see make_crowded_stimuli for details

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
    TODO
    """
    def __init__(self, root, n, label_type, train=True, download=False,
                 transform=None, **kwargs):
        self.root = root
        self.n = n
        self.label_type = label_type
        self.train = train
        self.download = download
        self.transform = transform
        
        if self.download:
            if self.train:
                data_file = self.root + "/train_data.npz"
                label_file = self.root + "/train_labels_"+label_type+".npz"
            else:
                data_file = self.root + "/test_data.npz"
                label_file = self.root + "/test_labels_"+label_type+".npz"

            self.data = np.load(data_file)
            self.labels = np.load(label_fle)
        else:
            self.make_stimuli(**kwargs)

    def make_stimuli(self, **kwargs):
        self.data = []
        self.labels = []
        for i in range(self.n):
            s, target_r, mean_r = stimuli.make_crowded_circles(**kwargs)
            self.data.append(s)
            if self.label_type.lower() == "center":
                self.labels.append(target_r)
            elif self.label_type.lower() == "mean":
                self.labels.append(mean_r)
            else:
                raise Exception("label type not undetstood")
        self.data = np.transpose(self.data, (1,2,0))
        self.labels = np.array(self.labels)
       
    def __getitem__(self, index):
        img = self.data[:, :, index, None]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return (img, label)
    
    def __len__(self):
        return len(self.labels)



