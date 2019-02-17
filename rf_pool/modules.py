import torch
import torch.nn as nn
import numpy as np
from layers import RF_Pool

class Module(nn.Module):
    """
    Base class for modules
    
    Attributes
    ----------
    attrs : dict
        dictionary tracking attributes in network
    data_shape : tuple
        shape of data input to network [initialized as None]
    n_layers : int
        number of layers in network [initialized as None]
    layer_ids : list
        list of layer ids for each layer [initialized as None]

    Methods
    -------
    init_attrs(keys)
        initialize attributes in self.attrs as None (if key not
        in self.attrs)
    update_attrs(new_attrs=None)
        update attributes in self.attrs using new_attrs
        dictionary. if new_attrs is None, update self.attrs
        values with getattr(self, key) for key in self.attrs.keys()
    set_from_attrs()
        set attributes to self using self.attrs dictionary
    """
    def __init__(self):
        super(Module, self).__init__()
        attr_names = ['data_shape', 'n_layers', 'layer_ids']
        self.init_attrs(attr_names)
        self.update_attrs(self.attrs)
        self.set_from_attrs()

    def __call__(self, *args):
        return self.forward(*args)

    def init_attrs(self, keys):
        # initialize attributes in self.attrs
        if not hasattr(self, 'attrs'):
            self.attrs = {}
        if type(keys) is not list:
            keys = [keys]
        for key in keys:
            self.attrs.setdefault(key, None)
    
    def update_attrs(self, new_attrs=None):
        # update attrs in self.attrs with new_attrs dict
        if not hasattr(self, 'attrs'):
            self.attrs = {}
        if new_attrs is None:
            new_attrs = {}
            for key in self.attrs.keys():
                new_attrs[key] = getattr(self, key)
        self.attrs.update(new_attrs)

    def set_from_attrs(self):
        # setattr(self, key, value) for (key,value) in self.attrs.items()
        if not hasattr(self, 'attrs'):
            self.attrs = {}
        for (key, value) in self.attrs.items():
            setattr(self, key, value)
    
    def set_list_vars(self, var):
        assert self.n_layers is not None, (
            'network must be initialized')
        if type(var) is not list:
            var = [var]
        if len(var) < self.n_layers:
            var = var + var[-1:] * (self.n_layers - len(var))
        return var
        
    def append_list_vars(self, var, new_var):
        assert type(var) is list, (
            'type of initial variable must be list')
        if type(new_var) is not list:
            new_var = [new_var]
        return var + new_var

    def get_typenames(self, var):
        if type(var) is not list:
            var = [var]
        return [torch.typename(v) if type(v) is not str else v for v in var]
    
    def append(self, net):
        """
        Append a new network to end of current network
        
        Parameters
        ----------
        net : Module
            network to append to current network
        """
        assert type(net) == type(self), (
            'type of appended network must match current')
        assert self.n_layers is not None and net.n_layers is not None, (
            'networks must be initialized')
        # update n_layers
        self.n_layers += net.n_layers
        if not hasattr(net, 'attrs') or type(net.attrs) is not dict:
            raise Exception('appended network must contain "attrs" attribute')
        # append each list var, update each dict/nn.ModuleDict var
        for (key, value) in net.attrs.items():
            if type(self.attrs[key]) is list:
                self.attrs[key] = self.append_list_vars(self.attrs[key], value)
            elif hasattr(self.attrs[key], 'update'):
                self.attrs[key].update({key: value})
        self.update_attrs(self.attrs)
        # link new parameters
        for layer_id in net.layer_ids:
            self.link_control_parameters(layer_id)
    
    def link_control_parameters(self, layer_id):
        # if no control nets, return
        if not hasattr(self, 'control_nets') or not self.control_nets:
            return
        # set layer_id str, register each layer_id_net_name_param to self
        layer_id = str(layer_id)
        net_names = ['trunk','mu_branch','sigma_branch']
        for net_name, net in zip(net_names, self.control_nets[layer_id]):
            for i, (name, param) in enumerate(net.named_parameters()):
                reg_name = '_'.join([layer_id, net_name, name.replace('.','_')])
                self.register_parameter(reg_name, param)
    
    def set_hidden_layer(self, layer_id):
        # get input_shape, layer_input to compute output_shape
        layer_id = int(layer_id)
        if layer_id == 0:
            input_shape = self.data_shape
        else:
            input_shape = self.output_shapes[layer_id - 1]
        layer_input = torch.zeros(input_shape)
        # choose hidden layer type
        if self.layer_names[layer_id].lower() == 'conv':
            # conv layers
            hidden_layer = nn.Conv2d(input_shape[1], self.output_channels[layer_id],
                                     self.kernel_sizes[layer_id])
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
        layer_id = int(layer_id)
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
        layer_id = int(layer_id)
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
    
    def set_dropout(self, layer_id):
        # set dropout based on dropout_names 
        layer_id = int(layer_id)
        if self.dropout_names[layer_id] == 'float':
            dropout = nn.Dropout(self.dropout_types[layer_id])
        elif self.dropout_names[layer_id].startswith('torch.nn.modules.dropout'):
            dropout = self.dropout_types[layer_id]
        elif self.dropout_names[layer_id].lower() == 'nonetype':
            dropout = None
        else:
            raise Exception('dropout_type not understood')
        return dropout
    
    def make_layers(self):
        pass

    def link_parameters(self):
        pass
    
    def apply_forward_pass(self):
        pass

    def forward_layer(self):
        pass
        
    def forward(self):
        pass
        
class FeedForwardNetwork(Module):
    """
    Module for doing Feed Forward Convolutional or Fully-Connected (or combo) Neural networks with
    custom pooling layers.
    
    Parameters
    ----------
    data_shape : tuple
        shape of the input data
    layer_types : list of strings or torch.nn.Module
        layer types, 'conv' or 'fc', at each layer
    act_types : list of strings or torch.nn.modules.activation or None
        activation function at each layer 
        [e.g., 'ReLU', torch.nn.modules.activation, or None]
    pool_types : list of strings or torch.nn.modules.pooling or rf_pool.layers or None
        pooling type at each convolutional layer
        [e.g., 'max_pool', torch.nn.modules.pooling, 'prob', 'stochastic', 'div_norm',
        'average', 'sum', rf_pool.layers, or None (default)]
    dropout_types : list of floats or None
        dropout probability at each layer [e.g., 0. or None (default)]
    output_channels : list of ints, optional
        number of output channels at each layer
    kernel_size : list of ints, optional
        size of patch at each convolutional layer
    pool_ksizes : list of ints or None, optional
        pooling kernel size at each convolutional layer
    control_nets : ControlNetwork, optional
        control networks used to update receptive fields for RF_Pool
        (see ControlNetwork)
    
    Attributes
    ----------
    hidden_layers : list of torch.nn.modules
        hidden layer objects chosen for each layer
    activations : list of torch.nn.modules.activation or None
        activation objects chosen for each layer
    pool_layers : list of torch.nn.modules.pooling or rf_pool.layers or None
        pooling layer objects chosen for each layer
    dropouts : list of torch.nn.modules.dropout or None
        dropout objects chosen for each layer
    output_shapes : list of tuples
        output shape for each layer
        
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
    >>> pool_types = ['max_pool', 'prob', None]
    >>> pool_ksizes = [2,2,None]
    >>> dropout_types = [None, None, .5]
    >>> net = FeedForwardNetwork(data_shape, layer_types, act_types, pool_types,
                                dropout_types, output_channels=output_channels,
                                kernel_sizes=kernel_sizes, pool_ksizes=pool_ksizes)
    >>> inputs = torch.rand(data_shape)
    >>> outputs = net(inputs)

    See Also
    --------
    layers.RF_Pool : layer implementation for receptive field pooling
    ops.rf_pool : receptive field pooling operation
    ControlNetwork : module to create control_nets to update receptive fields
        for use with RF_Pool
    """
    def __init__(self, data_shape, layer_types, act_types, pool_types=[None], dropout_types=[None], **kwargs):
        super(FeedForwardNetwork, self).__init__()
        # set inputs to attributes
        self.update_attrs({'data_shape':data_shape, 'layer_types':layer_types, 
                           'act_types':act_types, 'pool_types':pool_types,
                           'dropout_types':dropout_types})
        # initialize additional attributes
        attr_names = ['output_channels', 'kernel_sizes', 'pool_ksizes',
                      'control_nets', 'layer_names', 'act_names', 'pool_names', 
                      'dropout_names', 'hidden_layers', 'activations', 
                      'pool_layers', 'dropouts']
        self.init_attrs(attr_names)
        # update self.attrs with kwargs, set_from_attrs
        self.update_attrs(kwargs)
        self.set_from_attrs()
        
        # check data_shape is ndim 2 or 4
        data_dim = len(self.data_shape)
        assert (data_dim == 2 or data_dim == 4), (
               'Data must have shape [batch_size, n_features] or [batch_size, n_features, h, w]')
        
        # hidden layer params
        self.layer_names = self.get_typenames(self.layer_types)
        self.n_layers = len(self.layer_types)
        self.layer_ids = [str(i) for i in range(self.n_layers)]
        self.output_channels = self.set_list_vars(self.output_channels)
        self.output_shapes = [()]*self.n_layers
        self.kernel_sizes = self.set_list_vars(self.kernel_sizes)
        
        # activation functions
        self.act_types = self.set_list_vars(self.act_types)
        self.act_names = self.get_typenames(self.act_types)

        # pooling layer params
        self.pool_types = self.set_list_vars(self.pool_types)
        self.pool_names = self.get_typenames(self.pool_types)
        self.pool_ksizes = self.set_list_vars(self.pool_ksizes)
        
        # misc. params
        self.dropout_types = self.set_list_vars(self.dropout_types)
        self.dropout_names = self.get_typenames(self.dropout_types)
        
        # check each list var has len == n_layers
        for key in self.attrs.keys():
            value = getattr(self, key)
            if type(value) is list:
                assert len(value) == self.n_layers, (
                   ' '.join(key,'must a be a list of size',str(self.n_layers)))
            
        # initialize network
        self.make_layers()
        
        # update attrs
        self.update_attrs()

    def make_layers(self):
        """
        Create layers for network with following flow:
            1. layer operation: {'fc', 'conv', torch.nn.modules}
            2. activation func: {'ReLU', torch.nn.modules.activation, None}
            3. pooling: {'max_pool', torch.nn.modules.pooling, 'prob', 'stochastic'
                     'div_norm', 'average', 'sum', rf_pool.layers, None}
            4. dropout: {float, torch.nn.modules.dropout, None}
        """
        self.hidden_layers = nn.ModuleDict({})
        self.activations = nn.ModuleDict({})
        self.pool_layers = nn.ModuleDict({})
        self.dropouts = nn.ModuleDict({})

        for i, layer_id in enumerate(self.layer_ids):
            layer_id = str(layer_id)
            # layer types
            self.hidden_layers[layer_id] = self.set_hidden_layer(i)

            # activation types
            self.activations[layer_id] = self.set_activation_fn(i)

            # pooling types
            self.pool_layers[layer_id] = self.set_pool_layer(i)

            # dropout
            self.dropouts[layer_id] = self.set_dropout(i)

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
        for i, layer_id in enumerate(self.layer_ids):
            # flatten layer input if switching from 'conv' to 'fc'
            if len(self.output_shapes[np.maximum(i - 1, 0)]) == 2 or len(self.output_shapes[i]) == 2:
                x = x.flatten(1)
            # perform computations for given layer
            x = self.forward_layer(layer_id, x)
        return x
    
class ControlNetwork(Module):
    """
    TODO: WRITEME
    """
    def __init__(self, net, layer_ids, layer_types, act_types=['relu'], **kwargs):
        super(ControlNetwork, self).__init__()
        # set inputs to attributes
        self.update_attrs({'layer_ids':layer_ids, 'layer_types':layer_types,
                           'act_types':act_types})
        # initialize additional attributes
        attr_names = [key for key in net.attrs.keys()]
        self.init_attrs(attr_names)
        # update self.attrs with kwargs, set_from_attrs
        self.update_attrs(kwargs)
        self.set_from_attrs()
        
        assert type(self.layer_types) is list, (
            "layer_types must be list")
        assert self.layer_types[-1] == 'fc', (
            "The last trunk layer must be fully-connected")
        for layer_id in self.layer_ids:
            layer_id = int(layer_id)
            assert len(net.output_shapes[layer_id]) == 4, (
                "Control networks must follow conv layers")

        # trunk params
        self.layer_ids = [str(i) for i in self.layer_ids]
        self.n_layers = len(self.layer_types)
        self.output_channels = self.set_list_vars(self.output_channels)
        self.kernel_sizes = self.set_list_vars(self.kernel_sizes)
        self.act_types = self.set_list_vars(self.act_types)
        
        # initialize control nets
        self.make_layers(net)
        
        # link parameters
        for layer_id in self.layer_ids:
            self.link_control_parameters(layer_id)

        # update attrs
        self.update_attrs()
    
    def get_conv_shape(self, output_shape, pool_ksize):
        if pool_ksize is None:
            return output_shape
        return output_shape[:2] + (output_shape[-2]*pool_ksize, output_shape[-1]*pool_ksize)
        
    def make_layers(self, net):
        # build control networks
        self.control_nets = {}
        for i, layer_id in enumerate(self.layer_ids):
            # get layer shape prior to pooling
            layer_shape = self.get_conv_shape(net.output_shapes[int(layer_id)], 
                                              net.pool_ksizes[int(layer_id)])
            # make the trunk
            trunk = self.make_trunk(layer_shape)
            # make the mu and sigma branches 
            mu_branch = self.make_branch(trunk.output_shapes[-1], 2)
            sigma_branch = self.make_branch(trunk.output_shapes[-1], 1) 
            # set control net
            self.control_nets[str(layer_id)] = (trunk, mu_branch, sigma_branch)
        
    def make_trunk(self, input_shape):
        # build trunk for control network
        trunk_net = FeedForwardNetwork(input_shape, self.layer_types, self.act_types,
                                       output_channels=self.output_channels, 
                                       kernel_sizes=self.kernel_sizes)
        return trunk_net

    def make_branch(self, input_shape, output_channels):
        # build branch for control network with shape (batch, output_channels) 
        branch_net = FeedForwardNetwork(input_shape, ['fc'], [None], output_channels=[output_channels])
        return branch_net

    def forward(self, layer_id, x):
        layer_id = str(layer_id)
        trunk, mu_branch, sigma_branch = self.control_nets[layer_id]
        trunk_out = trunk.forward(x)
        return mu_branch.forward(trunk_out), sigma_branch.forward(trunk_out)

class GenerativeNetwork(Module):
    """
    #TODO:WRITEME
    """
    def __init__(self):
        super(GenerativeNetwork, self).__init__()
        raise NotImplementedError
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
