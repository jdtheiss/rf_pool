import torch
import torch.nn as nn 
import numpy as np 


class FeedForwardModule(nn.Module):
    def __init__(self, data_shape,
        layer_types, output_channels, patch_sizes, conv_strides, dropout, 
        pool_types, pool_ksizes):

        """
        Module for doing Feed Forward Convolutional or Fully-Connected (or combo) Neural networks with 
        custom pooling layers.

        Paramters
        ---------
        data_shape: tuple
            shape of the inpute data
        layer_types: list
            layer types, 'conv' or 'fc', at each layer
        output_channels: list
            number of output channels at each layer
        patch_size: list
            size of patch at each convolutional layer
        conv_strides: list
            size of stride at each convolutional layer
        dropout: list
            dropout keep probability at each layer
        pool_types: list
            pooling type at each convolutional layer
        pool_ksizes
            pooling kernel size at each convolutional layer

        Returns
        -------
        None

        Examples
        --------
        # Does one forward pass of a random dataset
        >>> data_shape = (10,3,28,28)
        >>> layer_types = ['conv', 'conv', 'fc']
        >>> output_channels = [25, 25, 10]
        >>> patch_sizes = [5, 5, None]
        >>> conv_strides = [1, 1, None]
        >>> dropout = [None, None, .5]
        >>> pool_types = ["max_pool", "max_pool", None]
        >>> pool_ksizes = [2,2,None]
        >>> net = FeedForwardModule(data_shape, layer_types, output_channels, patch_sizes, conv_strides, dropout, \
                                    pool_types, pool_ksizes)
        >>> inputs = torch.rand(data_shape)
        >>> outputs = net(inputs)

        """

        super(FeedForwardModule, self).__init__()
        # training data
        self.data_shape = data_shape
        data_dim = len(self.data_shape)
        assert (data_dim == 2 or data_dim == 4), (
            "Data must have shape [batch_size, num_features] or [batch_size, y, x, num_features]")
        if data_dim == 2:
            assert layer_types[0] == "fc", (
                "Data must be 2 dimensional for fc layers")
        else:
            assert layer_types[0] == 'conv', (
                "Data must be 4 dimensional for conv layers")

        #layer params
        self.layer_types = layer_types
        self.num_fc_layers = layer_types.count('fc')
        self.num_conv_layers = layer_types.count('conv')
        self.num_layers = self.num_fc_layers + self.num_conv_layers

        assert self.num_layers == len(self.layer_types), (
            "Layer types must be 'fc' or 'conv'")

        self.output_channels = output_channels
        self.patch_sizes = patch_sizes
        self.conv_strides = conv_strides

        # pooling layer params
        self.pool_types = pool_types
        self.pool_ksizes = pool_ksizes

        # misc. params
        self.dropout = dropout

        list_attr_names = ["output_channels", "patch_sizes", "conv_strides", "dropout", "pool_types", "pool_ksizes"]
        for name in list_attr_names: 
            assert len(getattr(self, name)) == self.num_layers, (
                name + " must a be a list of size "+ str(self.num_layers))

        self.make_layers()

    def conv_rules(self, i, k, s):
        return (i - k)/s + 1

    def num_flat_features(self):
        size = self.data_shape[-1]
        for i in range(self.num_conv_layers):
            # convolution
            size = self.conv_rules(size, self.patch_sizes[i], self.conv_strides[i])
            # pooling
            if self.pool_types[i]:
                size = self.conv_rules(size, self.pool_ksizes[i], self.pool_ksizes[i])

        return int(size**2)

    def pool_layer(self, layer_id):
        # choose pooling operation
        # TODO: add custom pooling operations
        if self.pool_types[layer_id] == "max_pool":
            return nn.MaxPool2d(self.pool_ksizes[layer_id], self.pool_ksizes[layer_id])
        else:
            return None

    def make_layers(self):
        """
        Flow style: --> layer operation: {'fc', 'conv'} 
        --> activation func: {'ReLU', 'Identity (last)'} 
        --> pooling: {'max_pool', TODO:others } 
        --> dropout
        """
        self.layer_choices = nn.ModuleDict({})
        self.act_choices = nn.ModuleDict({})
        self.pool_choices = nn.ModuleDict({})
        self.dropout_choices = nn.ModuleDict({})

        for layer_id in range(self.num_layers):
            # layer types
            if self.layer_types[layer_id] == 'fc':
                # fc layers
                if layer_id == 0:
                    in_features = self.data_shape[1]
                elif self.layer_types[layer_id-1] == 'conv':
                    in_features = self.num_flat_features() * self.output_channels[layer_id - 1]
                else:
                    in_features = self.output_channels[layer_id - 1]

                self.layer_choices[str(layer_id)] = nn.Linear(in_features, self.output_channels[layer_id])
            else:
                # conv layers
                if layer_id == 0:
                    in_channels = self.data_shape[1]
                else:
                    assert self.layer_types[layer_id-1] == 'conv', (
                    "conv layers cannot follow fc layers")
                    in_channels = self.output_channels[layer_id-1]

                self.layer_choices[str(layer_id)] = nn.Conv2d(in_channels, self.output_channels[layer_id], 
                    self.patch_sizes[layer_id], self.conv_strides[layer_id])

            # activation types
            if layer_id == (self.num_layers-1):
                self.act_choices[str(layer_id)] = None
            else:
                self.act_choices[str(layer_id)] = nn.ReLU()

            # pooling types
            self.pool_choices[str(layer_id)] = self.pool_layer(layer_id)

            # dropout 
            if self.dropout[layer_id]:
                self.dropout_choices[str(layer_id)] = nn.Dropout(self.dropout[layer_id]) 
            else:
                self.dropout_choices[str(layer_id)] = None

    def apply_forward_pass(self, func, x):
        if func:
            return func(x)
        else:
            return x
        
    def forward(self, x):
        # forward pass of network 
        for layer_id in range(self.num_layers):
            # flatten layer input if switching from 'conv' to 'fc'
            if self.layer_types[layer_id] != self.layer_types[layer_id - 1] and layer_id != 0:
                x = x.view(self.data_shape[0], -1)
            layer_id = str(layer_id)
            x = self.apply_forward_pass(self.layer_choices[layer_id], x)
            x = self.apply_forward_pass(self.act_choices[layer_id], x)
            x = self.apply_forward_pass(self.pool_choices[layer_id], x)
            x = self.apply_forward_pass(self.dropout_choices[layer_id], x)

        return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()