import torch
import torch.nn as nn
import numpy as np
from layers import RF_Pool

class FeedForwardNetwork(nn.Module):
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
    control_nets : ControlNetwork
        control networks used to update receptive fields for RF_Pool
        must be set directly (see ControlNetwork)

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
    ControlNetwork : module to create control_nets to update receptive fields
        for use with RF_Pool
    """
    def __init__(self, data_shape, layer_types, output_channels, kernel_sizes=[None],
                 conv_strides=[1], act_types=[None], pool_types=[None], 
                 pool_ksizes=[None], dropout_types=[None]):
        super(FeedForwardNetwork, self).__init__()
        # check data shape
        self.data_shape = data_shape
        data_dim = len(self.data_shape)
        assert (data_dim == 2 or data_dim == 4), (
               'Data must have shape [batch_size, n_features] or [batch_size, n_features, h, w]')
        # hidden layer params
        self.layer_types = layer_types
        self.layer_names = [torch.typename(l) if type(l) is not str else l for l in self.layer_types]
        self.n_layers = len(self.layer_types)
        self.output_channels = self.set_list_vars(output_channels)
        self.output_shapes = [()]*self.n_layers
        self.kernel_sizes = self.set_list_vars(kernel_sizes)
        self.conv_strides = self.set_list_vars(conv_strides)
        
        # activation functions
        self.act_types = self.set_list_vars(act_types)
        self.act_names = [torch.typename(l) if type(l) is not str else l for l in self.act_types]

        # pooling layer params
        self.pool_types = self.set_list_vars(pool_types)
        self.pool_names = [torch.typename(p) if type(p) is not str else p for p in self.pool_types]
        self.pool_ksizes = self.set_list_vars(pool_ksizes)

        # initialize control_nets
        self.control_nets = None
        
        # misc. params
        self.dropout_types = self.set_list_vars(dropout_types)
        
        # check each var has len == n_layers
        list_attr_names = ['output_channels', 'kernel_sizes', 'conv_strides',
                           'act_types', 'pool_types', 'pool_ksizes', 'dropout_types']
        for name in list_attr_names:
            assert len(getattr(self, name)) == self.n_layers, (
                   name + ' must a be a list of size ' + str(self.n_layers))
        # initialize network
        self.make_layers()

    def __call__(self, x):
        return self.forward(x)

    def set_list_vars(self, var):
        if type(var) is not list:
            var = [var]
        if len(var) < self.n_layers:
            var = var + var[-1:] * (self.n_layers - len(var))
        return var

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
            if delta_mu is not None and delta_sigma is not None:
                return fn(x, delta_mu, delta_sigma)
            else:
                return fn(x)
        else:
            return x

    def forward_layer(self, layer_id, x):
        # preform computations for one layer
        delta_mu = None
        delta_sigma = None
        layer_id = str(layer_id)
        x = self.apply_forward_pass(self.hidden_layers[layer_id], x)
        x = self.apply_forward_pass(self.activations[layer_id], x)
        if self.control_nets and layer_id in self.control_nets.layer_ids:
            delta_mu, delta_sigma = self.control_nets(layer_id, x)
        x = self.apply_forward_pass(self.pool_layers[layer_id], x, delta_mu, delta_sigma)
        x = self.apply_forward_pass(self.dropouts[layer_id], x)
        return x

    def forward(self, x):
        # forward pass of network
        for layer_id in range(self.n_layers):
            # flatten layer input if switching from 'conv' to 'fc'
            if len(self.output_shapes[np.maximum(layer_id - 1, 0)]) == 2 or \
               len(self.output_shapes[layer_id]) == 2:
                x = x.flatten(1)
            # perform computations for given layer
            x = self.forward_layer(layer_id, x)
        return x
    
class ControlNetwork(nn.Module):
    """
    TODO: WRITEME
    """
    def __init__(self, net, layer_ids, layer_types, output_channels, 
                 kernel_sizes=[None], act_types=['relu']):
        super(ControlNetwork, self).__init__()
        assert type(layer_types) is list, (
            "layer_types must be list")
        assert layer_types[-1] == 'fc', (
            "The last trunk layer must be fully-connected")
        for layer_id in layer_ids:
            assert len(net.output_shapes[layer_id]) == 4, (
                "Control networks must follow conv layers")

        # trunk params
        self.layer_ids = [str(i) for i in layer_ids]
        self.n_layers = len(layer_types)
        self.layer_types = layer_types
        self.output_channels = self.set_list_vars(output_channels)
        self.kernel_sizes = self.set_list_vars(kernel_sizes)
        self.act_types = self.set_list_vars(act_types)

        # branch params
        self.branch_input_shape = (net.data_shape[0], self.output_channels[-1])

        # build control networks
        self.layer_shapes = []
        self.control_nets = {}
        for i, layer_id in enumerate(self.layer_ids):
            # get layer shape before pooling
            self.layer_shapes.append(self.get_conv_shape(net.output_shapes[int(layer_id)], 
                                                         net.pool_ksizes[int(layer_id)]))
            # make the trunk
            trunk = self.make_trunk(self.layer_shapes[-1])
            # make the mu and sigma branches 
            mu_branch = self.make_branch(trunk.output_shapes[-1], 2)
            sigma_branch = self.make_branch(trunk.output_shapes[-1], 1) 
            # set control net
            self.control_nets[layer_id] = (trunk, mu_branch, sigma_branch)
            
    def __call__(self, layer_id, x):
        return self.apply(layer_id, x)

    def apply(self, layer_id, x):
        layer_id = str(layer_id)
        trunk, mu_branch, sigma_branch = self.control_nets[layer_id]
        trunk_out = trunk.forward(x)
        return mu_branch.forward(trunk_out), sigma_branch.forward(trunk_out)
    
    def set_list_vars(self, var):
        if type(var) is not list:
            var = [var]
        if len(var) < self.n_layers:
            var = var + var[-1:] * (self.n_layers - len(var))
        return var

    def append_list_vars(self, var, new_var):
        if type(new_var) is not list:
            new_var = [new_var]
        var = var + new_var
        return self.set_list_vars(var)
    
    def get_conv_shape(self, output_shape, pool_ksize):
        return output_shape[:2] + (output_shape[-2]*pool_ksize, output_shape[-1]*pool_ksize)

    def append(self, layer_id, control_net):
        raise NotImplementedError
        
    def link_parameters(self):
        raise NotImplementedError
        
    def make_trunk(self, input_shape):
        # build trunk for control network
        trunk_net = FeedForwardNetwork(input_shape, self.layer_types, 
                                       self.output_channels, self.kernel_sizes)
        return trunk_net

    def make_branch(self, input_shape, output_channels):
        # build branch for control network with shape (batch, output_channels) 
        branch_net = FeedForwardNetwork(input_shape, ['fc'], [output_channels])
        return branch_net

class GenerativeNetwork(nn.Module):
    """
    #TODO:WRITEME
    """
    def __init__(self):
        super(GenerativeNetwork, self).__init__()
        raise NotImplementedError
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
