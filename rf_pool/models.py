import numpy as np 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from modules import FeedForwardModule

class FeedForwardReceptiveFieldModel(nn.Module): 
    """
    Attributes
    ----------
    TODO: WRITEME

    Methods
    -------
    TODO: WRITEME

    Examples
    --------
    Do one forward pass (ff_net + control_net) for a random input tensor
    
    >>> net = FeedForwardReceptiveFieldModel()
    >>> data_shape = (10,3,28,28)
    >>> layer_types = ['conv', 'fc', 'fc']
    >>> output_channels = [25, 256, 10]
    >>> patch_sizes = [3, None, None]
    >>> conv_strides = [1, None, None]
    >>> dropout = [None, None, None]
    >>> pool_types = ["sum", None, None]
    >>> pool_ksizes = [2, None, None]
    >>> net.ff_network(data_shape, layer_types, output_channels, patch_sizes, conv_strides, dropout, \
                       pool_types, pool_ksizes)
    >>> control_layer_ids = [0]
    >>> trunk_layer_types = ['conv','fc']
    >>> trunk_channels = [64, 256]
    >>> trunk_filter_sizes = [3, None]
    >>> net.control_network(control_layer_ids, trunk_layer_types, trunk_channels, trunk_filter_sizes)
    >>> inputs = torch.rand(data_shape)
    >>> outputs = net(inputs)
    """


    def __init__(self):
        super(FeedForwardReceptiveFieldModel, self).__init__()

        self.ff_net_initialized = False
        self.control_net_initialized = False

    def ff_network(self, *args):
        """
        TODO: write
        """
        # build the feed forward classifier network with custom pooling
        self.ff_net = FeedForwardModule(*args)
        self.ff_net_initialized = True

    def control_network(self, control_layer_ids, trunk_layer_types, trunk_channels, trunk_filter_sizes):
        """
        TODO: write
        """

        assert self.ff_net_initialized, (
            "The Feed Forward network must be initialized first")
        assert trunk_layer_types[-1] == 'fc', (
            "The last trunk layer must be fully-connected")
        for layer_id in control_layer_ids:
            assert self.ff_net.layer_types[layer_id] == 'conv', (
                "Control networks must follow conv layers")

        # trunk params
        self.control_layer_ids = control_layer_ids
        self.trunk_num_layers = len(trunk_layer_types)
        self.trunk_layer_types = trunk_layer_types
        self.trunk_channels = trunk_channels
        self.trunk_filter_sizes = trunk_filter_sizes
        self.layer_shapes = self.get_detection_map_shapes()

        # branch params
        self.fv_shape = (self.ff_net.data_shape[0], self.trunk_channels[-1])

        # build control networks
        self.control_nets = {}
        for layer_id in self.control_layer_ids:
            # make the trunk
            fm_shape = (self.ff_net.data_shape[0],
                        self.ff_net.output_channels[layer_id],
                        self.layer_shapes[layer_id],
                        self.layer_shapes[layer_id])

            trunk = self.make_trunk(fm_shape)
            trunk.act_choices[str(self.trunk_num_layers-1)] = nn.ReLU()
            # make the mu and sigma branches 
            mu_branch = self.make_branch(2)
            sigma_branch = self.make_branch(1) 

            self.control_nets[str(layer_id)] = trunk, mu_branch, sigma_branch

        self.control_net_initialized = True

    def make_trunk(self, shape):
        # builds a trunk of the control network
        none_args = ([None]*self.trunk_num_layers,)*3
        trunk_net = FeedForwardModule(shape, self.trunk_layer_types, self.trunk_channels, self.trunk_filter_sizes,
            [1]*self.trunk_num_layers, *none_args)

        return trunk_net

    def make_branch(self, out_shape):
        # builds a branch of the control network with output size (batch, out_shape) 
        none_args = ([None],)*5
        branch_net = FeedForwardModule(self.fv_shape, ['fc'], [out_shape], *none_args)
        
        return branch_net

    def forward_layer_with_control_net(self, layer_id, x):
        # preforms computations for one layer of the feedforward + control layer
        layer_id = str(layer_id)
        # get detection map
        x = self.ff_net.apply_forward_pass(self.ff_net.layer_choices[layer_id], x)
        x = self.ff_net.apply_forward_pass(self.ff_net.act_choices[layer_id], x)
        # control network trunk
        feature_vector = self.control_nets[layer_id][0](x)
        # control network mu and sigma branches
        delta_mu = self.control_nets[layer_id][1](feature_vector)
        delta_sigma = self.control_nets[layer_id][2](feature_vector)
        # update sigma and mu and pool detection map
        x = self.ff_net.apply_forward_pass(self.ff_net.pool_choices[layer_id], x, delta_mu, delta_sigma)
        x = self.ff_net.apply_forward_pass(self.ff_net.dropout_choices[layer_id], x)

        return x

    def forward(self, x):
        assert self.ff_net_initialized, (
            "The Feed Forward network must be initialized first")

        # feedforward only
        if not self.control_net_initialized:
            x = self.ff_net(x)
        # feedforward + control
        else:
            for layer_id in range(self.ff_net.num_layers):
                #reshape if switching form conv to fc
                if self.ff_net.layer_types[layer_id] != self.ff_net.layer_types[layer_id - 1] and layer_id != 0:
                    x = x.view(self.ff_net.data_shape[0], -1)
                # perform layer computations with rf control    
                if layer_id in self.control_layer_ids:
                    x = self.forward_layer_with_control_net(layer_id, x)
                # perform layer computations without rf control
                else:
                    x = self.ff.net.forward_layer(layer_id, x)
        return x

    def get_detection_map_shapes(self):
        size = []
        size_pool = self.ff_net.data_shape[-1]
        for i in range(self.ff_net.num_conv_layers):
            # convolution
            size_conv = self.ff_net.conv_rules(size_pool, self.ff_net.patch_sizes[i], self.ff_net.conv_strides[i])
            size.append(size_conv)
            # pooling
            if self.ff_net.pool_types[i]:
                size_pool = self.ff_net.conv_rules(size_conv, self.ff_net.pool_ksizes[i], self.ff_net.pool_ksizes[i])
            else:
                size_pool = size_conv

        return size

    def get_variable_names(self):
        raise NotImplementedError
    def set_requires_grad(self, variables):
        raise NotImplementedError
    def load_saved_checkpoint(self):
        raise NotImplementedError
    def save_checkpoint(self):
        raise NotImplementedError
    def plot_updates(self):
        raise NotImplementedError

if __name__ == "__main__":
    import doctest
    doctest.testmod()
