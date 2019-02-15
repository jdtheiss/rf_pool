import torch
import torch.nn as nn
import numpy as np
from layers import RF_Pool

class FeedForwardNetwork(nn.Module):
    def __init__(self, data_shape, layer_types, output_channels, kernel_sizes,
                 conv_strides, act_types, pool_types, pool_ksizes, dropout_types):

        """
        Module for doing Feed Forward Convolutional or Fully-Connected (or combo) Neural networks with
        custom pooling layers.

        Attributes
        ----------
        data_shape : tuple
            shape of the inpute data
        layer_types : list of strings or torch.nn.Module
            layer types, 'conv' or 'fc', at each layer
        hidden_layers : list of torch.nn.modules
            hidden layer objects chosen for each layer
        output_channels : list of ints
            number of output channels at each layer
        output_shapes : list of tuples
            output shape for each layer
        patch_size : list of ints
            size of patch at each convolutional layer
        conv_strides : list of ints
            size of stride at each convolutional layer
        act_types : list of strings or torch.nn.modules.activation or None
            activation function at each layer ('ReLU', torch.nn.modules.activation,
            or None) last layer has no activation function
        activations : list of torch.nn.modules.activation or None
            activation objects chosen for each layer
        pool_types : list of strings or torch.nn.modules.pooling or rf_pool.layers or None
            pooling type at each convolutional layer
            ['max_pool', torch.nn.modules.pooling, 'prob', 'stochastic', 'div_norm',
            'average', 'sum', rf_pool.layers, None]
        pool_layers : list of torch.nn.modules.pooling or rf_pool.layers or None
            pooling layer objects chosen for each layer
        pool_ksizes : list of ints or None
            pooling kernel size at each convolutional layer
        dropout_types : list of floats or None
            dropout probability at each layer (0. or None indicates no dropout)
        dropouts : list of torch.nn.modules.dropout or None
            dropout objects chosen for each layer

        Methods
        -------
        set_hidden_layer(layer_id)
            set hidden layer for layer_id based on layer_types
        set_activation_fn(layer_id)
            set activation for layer_id based on act_types
        set_pool_layer(layer_id)
            set pooling layer for layer_id based on pool_types
        make_layers()
            initialize network with layer_types, pool_types, etc.
        apply_forward_pass(func, x, delta_mu=None, delta_sigma=None)
            perform forward pass through function with input x and optional arguments
            delta_mu and delta_sigma which are used to update receptive field kernels
            when using RF_Pool (see RF_Pool, rf.pool)
        forward_layer(layer_id, x)
            perform forward pass through layer_id with input x
        forward(x)
            perform forward pass through network with input x

        Examples
        --------
        # Does one forward pass of a random dataset
        >>> data_shape = (10,3,28,28)
        >>> layer_types = ['conv', 'conv', 'fc']
        >>> output_channels = [25, 25, 10]
        >>> kernel_sizes = [5, 5, None]
        >>> conv_strides = [1, 1, None]
        >>> pool_types = ['max_pool', 'prob', None]
        >>> pool_ksizes = [2,2,None]
        >>> dropout_types = [None, None, .5]
        >>> net = FeedForwardNetwork(data_shape, layer_types, output_channels,
                                    kernel_sizes, conv_strides, act_types,
                                    pool_types, pool_ksizes, dropout_types)
        >>> inputs = torch.rand(data_shape)
        >>> outputs = net(inputs)

        See Also
        --------
        RF_Pool : layer implementation for receptive field pooling
        rf.pool : receptive field pooling operation
        """

        super(FeedForwardNetwork, self).__init__()
        # check data shape
        self.data_shape = data_shape
        data_dim = len(self.data_shape)
        assert (data_dim == 2 or data_dim == 4), (
               'Data must have shape [batch_size, n_features] or [batch_size, n_features, h, w]')

        # hidden layer params
        self.layer_types = layer_types
        self.layer_names = [torch.typename(l) if type(l) is not str else l for l in layer_types]
        self.n_layers = len(self.layer_types)
        self.output_channels = output_channels
        self.output_shapes = [()]*self.n_layers
        self.kernel_sizes = kernel_sizes
        self.conv_strides = conv_strides

        # activation functions
        self.act_types = act_types
        self.act_names = [torch.typename(l) if type(l) is not str else l for l in act_types]

        # pooling layer params
        self.pool_types = pool_types
        self.pool_names = [torch.typename(p) if type(p) is not str else p for p in pool_types]
        self.pool_ksizes = pool_ksizes

        # misc. params
        self.dropout_types = dropout_types

        list_attr_names = ['output_channels', 'kernel_sizes', 'conv_strides',
                           'dropout_types', 'pool_types', 'pool_ksizes']
        for name in list_attr_names:
            assert len(getattr(self, name)) == self.n_layers, (
                   name + ' must a be a list of size ' + str(self.n_layers))

        self.make_layers()

    def conv_rules(self, i, k, s):
        return (i - k)/s + 1

    def set_hidden_layer(self, layer_id):
        # get input_shape, layer_input to compute output_shape
        if layer_id == 0:
            input_shape = self.data_shape
        else:
            input_shape = self.output_shapes[layer_id - 1]
        layer_input = torch.zeros(input_shape)
        # choose hidden layer type
        if self.layer_names[layer_id].lower() == 'conv':
            # conv layers
            hidden_layer = nn.Conv2d(input_shape[1], self.output_channels[layer_id],
                                     self.kernel_sizes[layer_id], self.conv_strides[layer_id])
        elif self.layer_names[layer_id].lower() == 'fc':
            # fc layers
            layer_input = torch.flatten(layer_input, 1)
            input_shape = layer_input.shape
            hidden_layer = nn.Linear(input_shape[1], self.output_channels[layer_id])
        elif self.layer_names[layer_id].startswith('torch.nn.modules'):
            hidden_layer = self.layer_types[layer_id]
        else:
            raise Exception('layer_type not understood')
        self.output_shapes[layer_id] = hidden_layer(layer_input).shape
        return hidden_layer

    def set_activation_fn(self, layer_id):
        # choose activation function
        if layer_id == (self.n_layers-1):
            return None
        if self.act_names[layer_id].lower() == 'relu':
            return nn.ReLU()
        elif self.act_names[layer_id].startswith('torch.nn.modules.activation'):
            return self.act_types[layer_id]
        elif self.act_names[layer_id].lower() == 'nonetype':
            return None
        else:
            raise Exception('act_type not understood')

    def set_pool_layer(self, layer_id):
        # get input_shape, pool_input to compute output_shape
        input_shape = self.output_shapes[layer_id]
        pool_input = torch.zeros(input_shape)
        # choose pooling operation
        if self.pool_names[layer_id].lower() == 'max_pool':
            pool_layer = nn.MaxPool2d(self.pool_ksizes[layer_id], self.pool_ksizes[layer_id])
        elif self.pool_names[layer_id] in ['prob', 'stochastic', 'div_norm', 'average', 'sum']:
            pool_layer = RF_Pool(pool_type=self.pool_types[layer_id],
                           block_size=(self.pool_ksizes[layer_id],)*2)
        elif self.pool_names[layer_id].startswith('torch.nn.modules.pooling'):
            pool_layer = self.pool_types[layer_id]
        elif self.pool_names[layer_id].find('layers') >= 0:
            pool_layer = self.pool_types[layer_id]
        elif self.pool_names[layer_id].lower() == 'nonetype':
            pool_layer = None
        else:
            raise Exception('pool_type not understood')
        if pool_layer:
            self.output_shapes[layer_id] = pool_layer(pool_input).shape
        return pool_layer

    def make_layers(self):
        """
        Flow style: --> layer operation: {'fc', 'conv', torch.nn.modules}
        --> activation func: {'ReLU', torch.nn.modules.activation, None}
        --> pooling: {'max_pool', torch.nn.modules.pooling, 'prob', 'stochastic'
                     'div_norm', 'average', 'sum', rf_pool.layers, None}
        --> dropout
        """
        self.hidden_layers = nn.ModuleDict({})
        self.activations = nn.ModuleDict({})
        self.pool_layers = nn.ModuleDict({})
        self.dropouts = nn.ModuleDict({})

        for layer_id in range(self.n_layers):
            # layer types
            self.hidden_layers[str(layer_id)] = self.set_hidden_layer(layer_id)

            # activation types
            self.activations[str(layer_id)] = self.set_activation_fn(layer_id)

            # pooling types
            self.pool_layers[str(layer_id)] = self.set_pool_layer(layer_id)

            # dropout
            if self.dropout_types[layer_id]:
                self.dropouts[str(layer_id)] = nn.Dropout(self.dropout_types[layer_id])
            else:
                self.dropouts[str(layer_id)] = None

    def apply_forward_pass(self, fn, x, delta_mu=None, delta_sigma=None):
        if fn:
            if delta_mu and delta_sigma:
                return fn(x, delta_mu, delta_sigma)
            else:
                return fn(x)
        else:
            return x

    def forward_layer(self, layer_id, x):
        # preform computations for one layer
        layer_id = str(layer_id)
        x = self.apply_forward_pass(self.hidden_layers[layer_id], x)
        x = self.apply_forward_pass(self.activations[layer_id], x)
        x = self.apply_forward_pass(self.pool_layers[layer_id], x)
        x = self.apply_forward_pass(self.dropouts[layer_id], x)
        return x

    def forward(self, x, delta_mu=None, delta_sigma=None):
        # forward pass of network
        for layer_id in range(self.n_layers):
            # flatten layer input if switching from 'conv' to 'fc'
            if len(self.output_shapes[np.maximum(layer_id - 1, 0)]) == 2 or \
               len(self.output_shapes[layer_id]) == 2:
                x = x.flatten(1)
            # perform computations for given layer
            x = self.forward_layer(layer_id, x)
        return x

if __name__ == '__main__':
    import doctest
    doctest.testmod()
