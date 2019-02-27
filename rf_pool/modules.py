#TODO: perhaps use nn.Sequential?
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

    def init_attrs(self, keys, defaults=[None]):
        # initialize attributes in self.attrs
        if not hasattr(self, 'attrs'):
            self.attrs = {}
        if type(keys) is not list:
            keys = [keys]
        defaults = self.set_list_vars(defaults, len(keys))
        for key, default in zip(keys, defaults):
            self.attrs.setdefault(key, default)
    
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
    
    def set_list_vars(self, var, n_layers):
        if type(var) is not list:
            var = [var]
        if len(var) < n_layers:
            var = var + var[-1:] * (n_layers - len(var))
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
            network to append to current network (must not be trained)
        """
        assert type(net) == type(self), (
            'type of appended network must match current')
        assert self.n_layers is not None and net.n_layers is not None, (
            'networks must be initialized')
        # update layer_ids
        new_layer_ids = [str(i) for i in range(self.n_layers, self.n_layers+net.n_layers)]
        net.update_layers(new_layer_ids)
        
        # update n_layers
        self.n_layers += net.n_layers
        if not hasattr(net, 'attrs') or type(net.attrs) is not dict:
            raise Exception('appended network must contain "attrs" attribute')
            
        # append each list var, update each dict/nn.ModuleDict var
        for (key, value) in net.attrs.items():
            if type(self.attrs[key]) is list:
                self.attrs[key] = self.append_list_vars(self.attrs[key], value)
            elif hasattr(self.attrs[key], 'update'):
                self.attrs[key].update(value.items())

        # update attributes
        self.update_attrs(self.attrs)
        self.set_from_attrs()
        
        # link new parameters
        for layer_id in net.layer_ids:
            self.link_parameters(layer_id)
    
    def link_parameters(self, layer_id, net_name='branch'):
        # set layer_id str, register each layer_id_net_name_param to self
        if not hasattr(self, 'nets'):
            return
        layer_id = str(layer_id)
        for i, net in enumerate(self.nets):
            for name, param in net.named_parameters():
                reg_name = '_'.join([layer_id, net_name, str(i), name.replace('.','_')])
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
    apply_forward_pass(func, x, *control_out)
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
    def __init__(self, data_shape, layer_types, act_types=[None], 
                 pool_types=[None], dropout_types=[None], **kwargs):
        super(FeedForwardNetwork, self).__init__()
        # set inputs to attributes
        self.update_attrs({'data_shape':data_shape, 'layer_types':layer_types, 
                           'act_types':act_types, 'pool_types':pool_types,
                           'dropout_types':dropout_types})
        # initialize additional attributes
        attr_names = ['output_channels', 'kernel_sizes', 'pool_ksizes',
                      'control_nets', 'layer_names', 'act_names', 'pool_names', 
                      'dropout_names', 'hidden_layers', 'activations', 
                      'pool_layers', 'dropouts', 'output_shapes']
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
        self.output_channels = self.set_list_vars(self.output_channels, self.n_layers)
        self.output_shapes = [()]*self.n_layers
        self.kernel_sizes = self.set_list_vars(self.kernel_sizes, self.n_layers)
        
        # activation functions
        self.act_types = self.set_list_vars(self.act_types, self.n_layers)
        self.act_names = self.get_typenames(self.act_types)
        
        # pooling layer params
        self.pool_types = self.set_list_vars(self.pool_types, self.n_layers)
        self.pool_names = self.get_typenames(self.pool_types)
        self.pool_ksizes = self.set_list_vars(self.pool_ksizes, self.n_layers)
        
        # misc. params
        self.dropout_types = self.set_list_vars(self.dropout_types, self.n_layers)
        self.dropout_names = self.get_typenames(self.dropout_types)
        self.control_out = []
        
        # check each list var has len == n_layers
        for key in self.attrs.keys():
            value = getattr(self, key)
            if type(value) is list:
                assert len(value) == self.n_layers, (
                   ' '.join([key,'must a be a list of size',str(self.n_layers)]))
            
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
            self.hidden_layers.add_module(layer_id, self.set_hidden_layer(i))

            # activation types
            self.activations.add_module(layer_id, self.set_activation_fn(i))

            # pooling types
            self.pool_layers.add_module(layer_id, self.set_pool_layer(i))

            # dropout
            self.dropouts.add_module(layer_id, self.set_dropout(i))

    def update_layers(self, new_layer_ids):
        # ensure new_layer_ids are string
        new_layer_ids = [str(i) for i in new_layer_ids]
        for layer_id, new_layer_id in zip(self.layer_ids, new_layer_ids):
            # update hidden layers
            if self.hidden_layers and layer_id in self.hidden_layers:
                hidden_layer = self.hidden_layers.pop(layer_id)
                self.hidden_layers.update({new_layer_id: hidden_layer})
            
            # update activations
            if self.activations and layer_id in self.activations:
                activation = self.activations.pop(layer_id)
                self.activations.update({new_layer_id: activation})
            
            if self.control_nets and layer_id in self.control_nets:
                control_net = self.control_nets.pop(layer_id)
                self.control_nets.update({new_layer_id: control_net})
            
            # update pooling layers
            if self.pool_layers and layer_id in self.pool_layers:
                pool_layer = self.pool_layers.pop(layer_id)
                self.pool_layers.update({new_layer_id: pool_layer})
            
            # update dropouts
            if self.dropouts and layer_id in self.dropouts:
                dropout = self.dropouts.pop(layer_id)
                self.dropouts.update({new_layer_id: dropout})
            
        # set new layer_ids
        self.layer_ids = new_layer_ids
        
        # update attributes
        self.update_attrs()

    def apply_forward_pass(self, fn, x, *args):
        if fn:
            return fn(x, *args)
        else:
            return x

    def forward_layer(self, layer_id, x):
        # preform computations for one layer
        control_out = []
        layer_id = str(layer_id)
        x = self.apply_forward_pass(self.hidden_layers[layer_id], x)
        x = self.apply_forward_pass(self.activations[layer_id], x)
        if self.control_nets and layer_id in self.control_nets:
            control_out = self.control_nets[layer_id](x)
            self.control_out = control_out
        x = self.apply_forward_pass(self.pool_layers[layer_id], x, *control_out)
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
    Module for building control network that can have a trunk network with multiple
    branch networks. 
    
    Parameters
    ----------
    net : Module
        network that control network will be connected to
    layer_id : string
        layer id within net that indicates the layer the control network connects to
    output_shapes : list of tuples
        expected output shape for each branch network (see example)
    trunk : dict, optional
        arguments to be passed to FeedForwardNetwork to create the trunk network
        [default: [], no trunk network]
    branches : list of dicts, optional
        arguments to be passed to FeedForwardNetwork to create each branch network.
        additionally, a fully connected layer will be appended to each branch network
        to obtain the number of units indicated by np.prod(output_shapes[b][1:])
        [default: [], single fully connected layer]
    branch_act_types : list, optional kwarg
        activation types for each branch to be applied to final output (see FeedForwardNetwork)
        [default: [None]*n_branches]
        
    Attributes
    ----------
    nets : list of Module
        trunk and branch networks that are called during forward() method
    layer_id : str
        layer id to which control network connects
    n_branches : int
        number of branches in control network (n_branches = len(output_shapes))
    
    Methods
    -------
    get_conv_shape(output_shape, pool_ksize)
        return output shape of convolutional layer prior to pooling
    make_layers(net, layer_id, output_shapes, trunk, branches, branch_act_types)
        create trunk and branch networks (attribute nets)
    forward(x)
        apply forward pass through control network, which passes the output of the 
        trunk network to each branch network followed by reshaping each branch output
        to shape of each output_shapes

    Examples
    --------
    # creates control network to output mu and sigma deltas for a receptive field pool layer
    >>> rf_layer = RF_Pool(torch.rand(36,2)*24, torch.ones(36,1), img_shape=(24,24), pool_type='sum')
    >>> net = FeedForwardNetwork((1,1,28,28), ['conv'], ['relu'], [rf_layer], output_channels=[20],
                                kernel_sizes=[5], pool_ksizes=[2])
    >>> trunk = {'layer_types': ['fc'], 'act_types': ['relu'], 'output_channels': [128]}
    >>> branches = []
    >>> net.control_nets = nn.ModuleDict()
    >>> net.control_nets['0'] = ControlNetwork(net, '0', [(-1,1,36,2), (-1,1,36,1)], 
                                               trunk=trunk, branches=branches)
    """
    def __init__(self, net, layer_id, output_shapes, trunk=[], branches=[], **kwargs):
        super(ControlNetwork, self).__init__()
        # set inputs to attributes
        self.update_attrs({'layer_id':layer_id, 'output_shapes':output_shapes})
        # initialize additional attributes
        attr_names = ['branch_act_types']
        self.init_attrs(attr_names, [None])
        
        # update self.attrs with kwargs, set_from_attrs
        self.update_attrs(kwargs)
        self.set_from_attrs()
        
        # set layer_ids, check net_params
        self.layer_id = str(self.layer_id)
        if type(self.output_shapes) is not list:
            self.output_shapes = [self.output_shapes]
        self.n_branches = len(self.output_shapes)
        branches = self.set_list_vars(branches, self.n_branches)
        self.branch_act_types = self.set_list_vars(self.branch_act_types, self.n_branches)
        
        # make layers
        self.make_layers(net, self.layer_id, self.output_shapes, trunk, branches,
                         self.branch_act_types)
        
        # link parameters
        self.link_parameters(self.layer_id)
        
        # update attrs
        self.update_attrs()
    
    def get_conv_shape(self, output_shape, pool_ksize): 
        #TODO: if pool_layer is set directly, pool_ksize may not be accurate
        if pool_ksize is None:
            return output_shape
        return output_shape[:2] + (output_shape[-2]*pool_ksize, output_shape[-1]*pool_ksize)
        
    def make_layers(self, net, layer_id, output_shapes, trunk, branches, branch_act_types):
        # get layer shape prior to pooling
        input_shape = self.get_conv_shape(net.output_shapes[int(layer_id)], 
                                          net.pool_ksizes[int(layer_id)])
        
        # init nets
        self.nets = []
        
        # make trunk
        if trunk:
            trunk_net = FeedForwardNetwork(input_shape, **trunk)
            self.nets.append(trunk_net)
            branch_input_shape = trunk_net.output_shapes[-1]
        else:
            branch_input_shape = input_shape
            
        # make branches
        for b in range(len(output_shapes)):
            # append fc with output_channels = np.prod(output_shapes[b][1:])
            output_channels = np.prod(output_shapes[b][1:])
            
            # set trunk, append branch
            if branches:
                self.nets.append(FeedForwardNetwork(branch_input_shape, **branches[b]))
                self.nets[-1].append(FeedForwardNetwork(self.nets[-1].output_shapes[-1], 
                                                        ['fc'], [branch_act_types[b]], 
                                                        output_channels=output_channels))
            else: # set branch with single layer
                self.nets.append(FeedForwardNetwork(branch_input_shape, ['fc'], [branch_act_types[b]], 
                                                    output_channels=output_channels))
            
    def forward(self, x):
        # get nets
        if self.n_branches < len(self.nets):
            trunk_net = self.nets[0]
            branch_nets = self.nets[1:]
            # get trunk output
            x = trunk_net.forward(x)
        else:
            branch_nets = self.nets
        # set branch outputs
        branch_out = []
        for branch_net, output_shape in zip(branch_nets, self.output_shapes):
            branch_out.append(torch.reshape(branch_net.forward(x), output_shape))
        return branch_out

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
