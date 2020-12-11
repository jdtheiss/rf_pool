from collections import OrderedDict
import inspect
import warnings

import pytorch_lightning as pl
import torch
from torch import nn

from rf_pool import build, modules, pool

def _append_modules(mod, **kwargs):
    """
    Convenience function to append modules to current module without changing
    module (subsequent modules are added with `add_module`)
    """
    # add modules
    [mod.add_module(k, v) for k, v in kwargs.items()]
    # set new forward function to call appended modules
    def forward(x):
        x = mod.__class__.forward(mod, x)
        for k in kwargs.keys():
            x = mod._modules[k](x)
        return x
    mod.forward = forward
    return mod

def _sequential(mods):
    """
    Convenience function to create Sequential model with Flatten added before
    Linear modules
    """
    flat = False
    net = nn.Sequential()
    for k, v in mods.items():
        if isinstance(v, nn.Linear) and not flat:
            net.add_module('%s_flatten' % k, nn.Flatten(1))
            flat = True
        net.add_module(k, v)
    return net

@torch.no_grad()
def _get_out_channels(model, img_shape=[3,12,12]):
    """Convenience function to get output channels from intermediate layers"""
    # set training to False
    _training = model.training
    model.eval()
    # ensure img_shape is list
    img_shape = list(img_shape)
    channels = []
    for mod in model.children():
        ch = None
        try:
            # get out channels from weight or out_channels
            if hasattr(mod, 'weight') and mod.weight.ndim >= 4:
                channels.append(mod.weight.size(0))
                continue
            elif hasattr(mod, 'out_channels'):
                channels.append(mod.out_channels)
                continue
            # otherwise set img_shape and pass a dummy input
            elif hasattr(mod, 'in_channels') and hasattr(mod, 'kernel_size'):
                img_shape = [mod.in_channels]
                if isinstance(mod.kernel_size, int):
                    img_shape.extend([mod.kernel_size]*2)
                else:
                    img_shape.extend(mod.kernel_size)
            elif len(channels) > 0:
                img_shape[0] = channels[-1]
            # pass dummy input to get output shape
            x = torch.zeros(1, *img_shape)
            img_shape = list(mod(x).shape[1:])
            ch = img_shape[0]
        except Exception as msg:
            warnings.warn('Warning: %s' % msg)
        channels.append(ch)
    # reset training
    model.train(_training)
    return channels

class Model(nn.Module):
    """
    Base model class

    Parameters
    ----------
    model : dict or nn.Module
        model to be set or built (if `isinstance(model, dict)`)
    **kwargs : **dict
        (method, kwargs) pairs for calling additional methods (see Methods)

    Methods
    -------
    replace_modules(**layers : **dict) : replace modules in model
    insert_modules(**layers : **dict) : insert modules into model
    set_parameters(**params : **dict) : set parameters for training
    print_model(verbose : bool) : print model and other attributes

    See Also
    --------
    rf_pool.solver.build.build_model
    """
    def __init__(self, model, **kwargs):
        super(Model, self).__init__()
        # build model
        if isinstance(model, dict):
            self._model = build.build_model({'MODEL': model})
        elif isinstance(model, nn.Module):
            self._model = model
        # apply methods
        for k, v in kwargs.items():
            assert hasattr(self, k)
            getattr(self, k)(**v)

    def __repr__(self):
        # set _model repr as self repr
        return self._model.__repr__()

    def get_modules(self, model):
        # replacement for named_modules method when module is repeated
        output = []
        for name, mod in model.named_children():
            output.append((name, mod))
            if hasattr(mod, 'named_children') and len(mod._modules) > 0:
                output.extend([('%s.%s' % (name, n), m) for n, m in self.get_modules(mod)])
        return output

    def _set_modules(self, layers, append=False):
        # get named modules as dict
        named_modules = dict(self.get_modules(self))
        # for each layer, update with new module
        for layer, module in layers.copy().items():
            # get layer by index
            if isinstance(layer, int):
                layer = list(named_modules.keys())[layer]
            elif named_modules.get(layer) is None:
                # recursively update modules that endswith layer
                updates = dict((k, module) for k in named_modules.keys()
                               if k.endswith(layer))
                assert len(updates) > 0, ('No modules ending with "%s" found.' % layer)
                layers.pop(layer)
                layers.update(updates)
                return self._set_modules(layers, append=append)
            # build new module
            new_mod = build.build_module(module)
            # get parent module
            split_name = layer.split('.')
            parent_name, name = split_name[:-1], split_name[-1]
            if parent_name:
                parent = named_modules.get('.'.join(split_name[:-1]))
            else:
                parent = self
            # append module to current module
            if append:
                current_mod = named_modules[layer]
                sub_name = new_mod.__class__.__name__.lower()
                new_mod = _append_modules(current_mod, **{sub_name: new_mod})
            # add_module to append or overwrite
            parent.add_module(name, new_mod)

    def replace_modules(self, **layers):
        """
        Replace modules in model with new module

        Parameters
        ----------
        layers : dict
            dictionary of key/value pairs like {module_name: module}
            where `module_name` can be either the index or name of the module
            (found using the model's `.named_modules()` method).

        Returns
        -------
        None, overwrites module in model

        Notes
        -----
        A specific module can be replaced by using its fullpath module name
        (e.g., 'model.layer1.1.activation'). Alternatively, multiple modules
        can be replaced by using the ending module name (e.g., 'activation').

        See also
        --------
        insert_modules
        """
        self._set_modules(layers, append=False)

    def insert_modules(self, **layers):
        """
        Insert modules in model after a given module

        Parameters
        ----------
        layers : dict
            dictionary of key/value pairs like {module_name: module}
            where `module_name` can be either the index or name of the current
            module (found using the model's `.named_modules()` method) after
            which `module` will be appended.

        Returns
        -------
        None, overwrites module in model

        Notes
        -----
        A specific module can be inserted by using the fullpath name of the
        preceding module (e.g., 'model.layer1.1.activation'). Alternatively,
        multiple modules can be inserted by using the ending module name
        (e.g., 'activation').

        See also
        --------
        replace_modules
        """
        self._set_modules(layers, append=True)

    def _find_parameters(self, patterns):
        """"convenience function to return params that match given patterns"""
        if not hasattr(self, '_parameter_names'):
            self._parameter_names = {}
        if not isinstance(patterns, list):
            patterns = [patterns]
        params = []
        for name, param in self.named_parameters():
            if any(name.find(pattern) != -1 for pattern in patterns):
                if self._parameter_names.get(name) is not None:
                    warnings.warn('Parameter "%s" already found. Skipping.' % name)
                    continue
                params.append(param)
                self._parameter_names.update({name: True})
        if len(params) == 0:
            warnings.warn('No parameters found for patterns: %a' % patterns)
        return params

    def set_parameters(self, **kwargs):
        """
        Set parameters from model components for training

        Parameters
        ----------
        params : dict
            (name, patterns) key/value pairs for setting parameters like
            `setattr(self, name, self._find_parameters(patterns))`

        Returns
        -------
        None, parameters set to `self.model`

        Examples
        --------
        >>> model = Model(nn.ModuleDict({'backbone':
                                         nn.Sequential(
                                            nn.Conv2d(3, 64, 3),
                                            nn.Conv2d(64, 128, 3)
                                         )}))
        >>> params = {'weight_parameters': 'weight', 'bias_parameters': 'bias'}
        >>> model.set_parameters(params)
        >>> print(['number of %s: %d' % (p, len(getattr(model, p))) for p in
                   ['weight_parameters','bias_parameters']])
        ['number of weight_parameters: 2', 'number of bias_parameters: 2']
        """
        # set parameters for each set of patterns
        for name, patterns in kwargs.items():
            # set parameters to model
            setattr(self, name, self._find_parameters(patterns))

    def print_model(self, verbose=False):
        """
        Print model and other attributes

        Parameters
        ----------
        verbose : bool
            True/False print model as well as all attributes in `self.__dict__`
            [default: False, print model only]

        Returns
        -------
        None, prints to sys.stdout
        """
        print('Model:\n%a\n' % self._model)
        if verbose:
            for k, v in self.__dict__.items():
                print('%s:\n%a\n' % (k, v))

    def forward(self, inputs, targets=None, **kwargs):
        """
        Pass inputs through model and return outputs and targets

        Parameters
        ----------
        inputs : torch.Tensor
            input to model passed to return outputs
        targets : torch.Tensor
            targets to be passed to loss function with model outputs
            [default: None]
        **kwargs : **dict
            keyword arguments passed to model with `inputs`

        Returns
        -------
        outputs : any
            model outputs passed to loss function
        targets : any
            targets passed to loss function
        """
        # return model outputs and targets
        return self._model(inputs, **kwargs), targets

if __name__ == '__main__':
    import doctest
    doctest.testmod()
