from torch.utils.data import DataLoader as PTDataLoader
from torch.utils.data import ConcatDataset

class DataLoader(PTDataLoader):
    """
    Base class for dataloaders

    Parameters
    ----------
    datasets : dict
        dictionary of datasets to use as (name, dataset) key/value pair
    **kwargs : **dict
        keyword arguments passed to `torch.utils.data.DataLoader`

    Notes
    -----
    The base class allows only one dataset in `datasets`, and is passed as
    `torch.utils.data.DataLoader(*datasets.values(), **kwargs)`.
    """
    def __init__(self, datasets, **kwargs):
        assert len(datasets) == 1, (
            'DataLoader class requires one dataset (%d found).' % (len(datasets))
        )
        super(DataLoader, self).__init__(*datasets.values(), **kwargs)

class ConcatDataLoader(PTDataLoader):
    """
    Dataloader with concatenated datasets

    Parameters
    ----------
    datasets : dict
        dictionary of datasets to use as (name, dataset) key/value pair passed
        as `torch.utils.data.ConcatDataset(datasets.values())`
    **kwargs : **dict
        keyword arguments passed to `torch.utils.data.DataLoader`
    """
    def __init__(self, datasets, **kwargs):
        concat_dataset = ConcatDataset(datasets.values())
        super(ConcatDataLoader, self).__init__(concat_dataset, **kwargs)
