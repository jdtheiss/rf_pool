from math import log, pi

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discretize(nn.Module):
    """
    """
    def __init__(self, n_bits=8, max_n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.n_bins = 2.0 ** self.n_bits
        self.max_n_bits = max_n_bits
        self.max_n_bins = 2.0 ** self.max_n_bits - 1

    @torch.no_grad()
    def _normalize(self, x):
        x = x - x.min()
        return x / (x.max() + 1e-5)

    def forward(self, x):
        x = self._normalize(x) * self.max_n_bins

        x = torch.floor(x / 2 ** (self.max_n_bits - self.n_bits))

        x = x / self.n_bins - 0.5

        x = x + torch.rand_like(x) / self.n_bins

        return x

class ActNorm(nn.Module):
    """
    Activation Normalization Layer from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels
    eps : float
        epsilon value to add to std at initialization of scale parameter
        [default: 1e-6]

    Notes
    -----
    Forward pass multiplies by scale parameter and subtracts location parameter
    and also returns log determinant: `h * w * sum(log(abs(scale)))`.
    Reverse pass subtracts location parameter and divides by scale parameter.
    """
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()

        self.register_parameter('loc', nn.Parameter(torch.zeros(1, in_channels, 1, 1)))
        self.register_parameter('scale', nn.Parameter(torch.ones(1, in_channels, 1, 1)))
        self.initialized = False
        self.eps = eps

    @torch.no_grad()
    def initialize(self, x):
        self.loc.data.copy_(-x.mean([0,2,3], keepdim=True))
        self.scale.data.copy_((x.std([0,2,3], keepdim=True) + self.eps).inverse())
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        h, w = x.shape[2:]

        logdet = h * w * torch.sum(torch.log(torch.abs(self.scale)))

        return (x + self.loc) * self.scale, logdet

    def reverse(self, x):
        return x / self.scale - self.loc

class InvConv2d(nn.Module):
    """
    Invertible 1x1 Convolution Layer from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels

    Notes
    -----
    Forward pass is 1x1 convolution, and reverse pass is 1x1
    convolution using inverse of forward weight matrix.
    """
    def __init__(self, in_channels):
        super().__init__()

        w = torch.qr(torch.randn(in_channels, in_channels))[0]
        self.register_parameter('weight', nn.Parameter(w.view(in_channels, in_channels, 1, 1)))

    def forward(self, x):
        h, w = x.shape[2:]

        logdet = h * w * torch.slogdet(self.weight.squeeze())[1]

        return F.conv2d(x, self.weight), logdet

    def reverse(self, x):
        return F.conv2d(x, self.weight.squeeze().inverse().view_as(self.weight))

class ZeroConv2d(nn.Module):
    """
    Zero-initialized Conv2d Layer from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    kernel_size : int
        kernel size of convolution [default: 3]
    padding : int
        padding size of convolution [default: 1]

    Notes
    -----
    Convolution weight and bias parameters initialized as zeros.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.padding = padding
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

        self.net.weight.data.zero_()
        self.net.bias.data.zero_()
        self.register_parameter('scale', nn.Parameter(torch.zeros(1, out_channels, 1, 1)))

    def forward(self, x):
        out = self.net(F.pad(x, [self.padding]*4, value=1))

        return out * torch.exp(self.scale * 3)

class AffineCouple(nn.Module):
    """
    Affine Coupling Layer from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels
    n_channels : int
        number of intermediate channels for affine net
        [default: 512]
    affine : bool
        True/False whether to use affine (vs. additive coupling)
        [default: True, affine coupling]

    Notes
    -----
    Forward pass splits input along channel dim, computes affine parameters
    from first chunk of input to transform second chunk. First chunk is passed
    as identity transform. Reverse pass performs the same operation but with
    reversed affine transform.
    """
    def __init__(self, in_channels, n_channels=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, n_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(n_channels, in_channels if self.affine else in_channels // 2)
        )

        self.register_parameter('scale', nn.Parameter(torch.zeros(1, in_channels // 2, 1, 1)))

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(x_a).chunk(2, 1)
            log_s = torch.tanh(log_s / self.scale.exp()) * self.scale.exp()
            out_b = (x_b + t) * torch.exp(log_s)

            logdet = torch.sum(log_s.view(x.size(0), -1), -1)
        else:
            out_b = x_b + self.net(x_a)

            logdet = None

        return torch.cat([x_a, out_b], 1), logdet

    def reverse(self, x):
        x_a, x_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(x_a).chunk(2, 1)
            log_s = torch.tanh(log_s / self.scale.exp()) * self.scale.exp()

            out_b = x_b * torch.exp(-log_s) - t
        else:
            out_b = x_b - self.net(x_a)

        return torch.cat([x_a, out_b], 1)

class Flow(nn.Module):
    """
    Normalizing Flow Layer from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels
    n_channels : int
        number of intermediate channels in AffineCouple layer
        [default: 512]
    affine : bool
        True/False affine coupling layer (vs. additive coupling)
        [default: True]

    Notes
    -----
    Forward pass returns output from ActNorm, InvConv2d, AffineCouple
    as well as log determinant. Reverse pass reverses process of forward pass.
    """
    def __init__(self, in_channels, n_channels=512, affine=True):
        super().__init__()

        self.actnorm = ActNorm(in_channels)
        self.invconv = InvConv2d(in_channels)
        self.couple = AffineCouple(in_channels, n_channels=n_channels, affine=affine)

    def forward(self, x):
        out, logdet = self.actnorm(x)
        out, det1 = self.invconv(out)
        out, det2 = self.couple(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, x):
        out = self.couple.reverse(x)
        out = self.invconv.reverse(out)
        return self.actnorm.reverse(out)

class FlowBlock(nn.Module):
    """
    Block of Flow Layers from GLOW architecture

    Parameters
    ----------
    in_channels : int
        number of input channels
    n_flows : int
        number of normalizing flow layers
    split : bool
        True/False split output to compute log(p)
        [default: True]
    n_channels : int
        number of intermediate channels in AffineCouple layer of each Flow
        [default: 512]
    affine : bool
        True/False use affine coupling (vs. additive coupling)
        [default: True, affine coupling]

    Notes
    -----
    During forward pass, input is passed through each flow layer, and
    the log determinant as well as log prob is returned along with z.
    During reverse pass, z is sampled and passed in reverse order through
    each flow layer.
    """
    def __init__(self, in_channels, n_flows, split=True, n_channels=512, affine=True,
                 learn_prior=True):
        super().__init__()

        self.in_channels = in_channels
        squeeze_dim = in_channels * 4
        self.split = split

        self.net = nn.ModuleList()
        for n in range(n_flows):
            self.net.append(Flow(squeeze_dim, n_channels=n_channels, affine=affine))

        self.learn_prior = learn_prior
        if self.learn_prior is False:
            self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        elif self.split:
            self.prior = ZeroConv2d(in_channels * 2, in_channels * 4)
        else:
            self.prior = ZeroConv2d(in_channels * 4, in_channels * 8)

    def gaussian_log_p(self, x, mu, log_sigma):
        if self.learn_prior is False:
            return self.prior.log_prob(x)
        return -0.5 * log(2 * pi) - log_sigma - 0.5 * (x - mu) ** 2 / torch.exp(2 * log_sigma)

    def gaussian_sample(self, eps, mu, log_sigma):
        if self.learn_prior is False:
            return self.prior.sample(eps.shape)
        return mu + torch.exp(log_sigma) * eps

    def compute_log_p(self, x):
        if self.split:
            x, z = x.chunk(2, 1)
            if self.learn_prior:
                mu, log_sigma = self.prior(x).chunk(2, 1)
            else:
                mu = log_sigma = None
            log_p = self.gaussian_log_p(z, mu, log_sigma)
            log_p = log_p.view(x.size(0), -1).sum(-1)
        else:
            if self.learn_prior:
                mu, log_sigma = self.prior(torch.zeros_like(x)).chunk(2, 1)
            else:
                mu = log_sigma = None
            log_p = self.gaussian_log_p(x, mu, log_sigma)
            log_p = log_p.view(x.size(0), -1).sum(-1)
            z = x

        return log_p, z

    def sample_z(self, x, eps):
        if self.split:
            if self.learn_prior:
                mu, log_sigma = self.prior(x).chunk(2, 1)
            else:
                mu = log_sigma = None
            z = self.gaussian_sample(eps, mu, log_sigma)
            z = torch.cat([x, z], 1)
        else:
            if self.learn_prior:
                mu, log_sigma = self.prior(torch.zeros_like(x)).chunk(2, 1)
            else:
                mu = log_sigma = None
            z = self.gaussian_sample(eps, mu, log_sigma)

        return z

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.contiguous().view(b, c * 4, h // 2, w // 2)

        logdet = 0
        for flow in self.net:
            x, det = flow(x)
            logdet = logdet + det

        log_p, z = self.compute_log_p(x)
        if self.split:
            x = x.chunk(2, 1)[0]

        return x, logdet, log_p, z

    def reverse(self, x, eps, reconstruct=False):
        if reconstruct:
            out = torch.cat([x, eps], 1) if self.split else eps
        else:
            out = self.sample_z(x, eps)

        for flow in self.net[::-1]:
            out = flow.reverse(out)

        b, c, h, w = out.size()
        out = out.view(b, c // 4, 2, 2, h, w)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.contiguous().view(b, c // 4, h * 2, w * 2)

        return out

if __name__ == '__main__':
    import doctest
    doctest.testmod()
