import torch
import torch.nn as nn
import torch.nn.functional as F

from rf_pool.models import Model
from rf_pool.modules import Discretize, FlowBlock

class FlowModel(Model):
    """
    Flow Model (basic GLOW implementation)

    Parameters
    ----------
    in_channels : int
        number of input channels
    n_blocks : int
        number of FlowBlocks
    n_flows : int
        number of Flows per block
    n_channels : int
        number of channels per flow [default: 512]
    affine : bool
        True/False use affine coupling [default: True]
    **kwargs : **dict
        keyword arguments passed to Discretize (if any given, otherwise no
        Discretize module)
    """
    def __init__(self, in_channels, n_blocks, n_flows, n_channels=512, affine=True,
                 learn_prior=True, **kwargs):
        super(Model, self).__init__()

        # build flow blocks
        b_channels = in_channels
        blocks = nn.ModuleList()
        for i in range(n_blocks - 1):
            blocks.append(FlowBlock(b_channels, n_flows, split=True,
                                    n_channels=n_channels, affine=affine,
                                    learn_prior=learn_prior))
            b_channels *= 2
        blocks.append(FlowBlock(b_channels, n_flows, split=False,
                                n_channels=n_channels, affine=affine,
                                learn_prior=learn_prior))
        super().__init__(blocks)

        # if kwargs, set Discretize
        if kwargs:
            self.discretize = Discretize(**kwargs)
        else:
            self.discretize = None

        # set parameters
        self.in_channels = in_channels
        self.n_blocks = n_blocks
        self.n_flows = n_flows

    def _get_z_shapes(self, image_size):
        image_size = F._pair(image_size)
        in_channels = self.in_channels

        z_shapes = []
        for i in range(self.n_blocks - 1):
            image_size = tuple([s // 2 for s in image_size])
            in_channels *= 2

            z_shapes.append((in_channels, *image_size))

        image_size = tuple([s // 2 for s in image_size])
        z_shapes.append((in_channels * 4, *image_size))

        return z_shapes

    def forward(self, input, *args, **kwargs):
        log_p_sum = 0
        logdet = 0
        out = self.discretize(input) if self.discretize else input
        z_outs = []

        for name, block in self._modules.items():
            if name == 'discretize':
                continue
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        idx = -1
        input = None
        for name, block in list(self._modules.items())[::-1]:
            if name == 'discretize':
                continue
            input = block.reverse(input if input is not None else z_list[idx],
                                  z_list[idx], reconstruct=reconstruct)
            idx -= 1

        return input

    def sample(self, n_samples, image_size, temp=0.7, device='cpu'):
        z_sample = []
        for z_shape in self._get_z_shapes(image_size):
            z_new = torch.randn(n_samples, *z_shape) * temp
            z_sample.append(z_new.to(device))
        return self.reverse(z_sample)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
