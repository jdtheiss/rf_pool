import torch
import torch.nn as nn
import torch.nn.functional as F


class LSH(nn.Module):
    """
    Locality-Sensitive Hashing

    Project a feature vector or patch of feature vectors to a normalized
    space [-1,1] using a random weight matrix R containing unit-normalized rows

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels (dimensionality of normalized space)
    kernel_size : int or tuple[int]
        kernel size for convolution (input is unit normalized wrt kernel size)
    **kwargs : **dict
        keyword arguments passed to `torch.conv2d`

    Notes
    -----
    Input features are normalized with respect to the convolution kernel such that
    the output vectors are the cosine similarity with each row in the random matrix R.
    The weight matrix R is not learned, but this approach is useful for projecting
    features into a normalized space for vector symbolic architectures (VSAs).
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = F._pair(kernel_size)

        w = torch.randn(out_channels, in_channels, *self.kernel_size)
        self.register_buffer('weight', F.normalize(w, dim=[1,2,3]))
        self.conv_kwargs = kwargs

    def forward(self, x):
        # normalize x
        h, w = x.shape[2:]
        x = F.unfold(x, self.kernel_size, **self.conv_kwargs)
        x = F.normalize(x, dim=1)
        x = F.fold(x, (h, w), self.kernel_size, **self.conv_kwargs)
        # convolve with normalized weight
        return torch.conv2d(x, self.weight, **self.conv_kwargs)
