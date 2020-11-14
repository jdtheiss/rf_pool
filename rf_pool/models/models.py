from collections import OrderedDict
import inspect
import warnings

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from rf_pool import modules, pool
from rf_pool.solver import build

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
    insert_modules(**modules:**dict) : insert modules into model at specified layers
    set_parameters(**params:**dict) : set parameters for training
    print_model(verbose:bool) : print model and other attributes

    See Also
    --------
    rf_pool.solver.build.build_model
    """
    def __init__(self, model, **kwargs):
        super(Model, self).__init__()
        # build model
        if isinstance(model, dict):
            self._model = build.build_model({'MODEL': backbone})
        elif isinstance(model, nn.Module):
            self._model = model
        # apply methods
        for k, v in kwargs.items():
            assert hasattr(self, k)
            getattr(self, k)(**v)

    def __repr__(self):
        # set _model repr as self repr
        return self._model.__repr__()

    def insert_modules(self, **kwargs):
        """
        Insert modules into network at given layer indices

        Parameters
        ----------
        **kwargs : **dict
            key/value pairs with following structure:
            LAYERS : list
                indices of network `_modules` to insert module
            NETWORK : str, optional
                network to insert modules [default: None, inserts in model]
            IMAGE_SHAPE : list
                shape of input image used to infer output channels of intermediate
                layers within the network (i.e., `[in_channels, h, w]`)
                [default: [3, 12, 12]]
            {module name} : dict
                kwargs to initialize {module name} (see Notes)

        Returns
        -------
        None, updates model

        Notes
        -----
        The {module name} should be a dictionary where {module name} is a class
        of `rf_pool.modules`, `rf_pool.pool`, or `torch.nn`. The dictionary will
        be passed e.g. `getattr(rf_pool.modules, {module name})(**kwargs)`,
        where `kwargs` is the dictionary of keyword arguments used to initialize
        {module name}.

        A separate module is inserted at each index in `LAYERS`, such that the
        output of previous layer is passed to the inserted module and its output
        is passed to the following layer in the network. If {module name} has
        the parameter `in_channels` or `out_channels`, these will be set by the
        `out_channels` of the previous layer (unless either parameter is set in
        `kwargs`). See Examples.

        Examples
        --------
        >>> model = Model(nn.ModuleDict({'backbone':
                                         nn.Sequential(
                                            nn.Conv2d(3, 64, 3),
                                            nn.Conv2d(64, 128, 3)
                                         )}))
        >>> insert = {'NETWORK': 'backbone',
                      'LAYERS': [1,2],
                      'LeakyReLU': {'negative_slope': 0.2}}
        >>> model.insert_modules(insert)
        >>> print(model)
        ModuleDict(
          (backbone): Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
            (leakyrelu_1): LeakyReLU(negative_slope=0.2)
            (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
            (leakyrelu_3): LeakyReLU(negative_slope=0.2)
          )
        )
        """
        # get network and layers
        net_name = kwargs.pop('NETWORK', '_model')
        if net_name == '_model':
            net = getattr(self, net_name)
        else:
            net = getattr(self._model, net_name, None)
        layers = kwargs.pop('LAYERS', [])
        n_layers = len(layers)
        if net is None or n_layers == 0:
            return
        # get network modules
        net_mods = list(net._modules.items())
        # get out channels of network modules
        img_shape = kwargs.pop('IMAGE_SHAPE', [3,12,12])
        net_out_channels = _get_out_channels(net, img_shape)
        # get modules to insert
        insert_mods = list(kwargs.items())
        insert_mods = (insert_mods + insert_mods[-1:] * n_layers)[:n_layers]
        # insert modules to network
        cnt = 0
        for idx, (name, kwargs) in zip(layers, insert_mods):
            kwargs = kwargs.copy()
            # set in_channels, out_channels if needed
            if idx > 0:
                out_channels = net_out_channels[idx-1]
            else:
                out_channels = img_shape[0]
            # update idx based on number of previous inserted modules
            idx = idx + cnt
            # get module to insert
            mod_fn = build.get_class([modules, pool, nn], name)
            args = inspect.getfullargspec(mod_fn).args
            if 'in_channels' in args:
                kwargs.setdefault('in_channels', out_channels)
            if 'out_channels' in args:
                kwargs.setdefault('out_channels', out_channels)
            # insert to net_mods
            net_mods.insert(idx, ('%s_%d' % (name.lower(), idx), mod_fn(**kwargs)))
            cnt += 1
        # update modules
        updated_net = _sequential(OrderedDict(net_mods))
        if net_name == '_model':
            setattr(self, net_name, updated_net)
        else:
            setattr(self._model, net_name, updated_net)

    def _find_parameters(self, patterns):
        """"convenience function to return params that match given patterns"""
        if not isinstance(patterns, list):
            patterns = [patterns]
        params = []
        for name, param in self.named_parameters():
            if any(name.find(pattern) != -1 for pattern in patterns):
                params.append(param)
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

    def apply_modules(self, *args, **kwargs):
        pass

    def forward(self, x):
        for name, mod in self._model.named_children():
            x = mod(x)
        return x

#TODO: update classes to work as Model class does
class VAE(Model):
    """
    Variational Autoencoder

    Model should contain `forward_layer` and `reconstruct_layer` (e.g.,
    `modules.Autoencoder`), with output factored into branches `mu` and `logvar`:
        `Branch(
           (forward_layer): Sequential(
             (mu): Linear(in_features=1024, out_features=1024, bias=True)
             (logvar): Linear(in_features=1024, out_features=1024, bias=True)
           )
           (reconstruct_layer): Sequential()
         )`

    Build the model by appending `Autoencoder` layer(s), and then call
    `add_output_branch` to add the branching layer.

    Attributes
    ----------
    data_shape : tuple
        shape of input data
    layers : torch.nn.ModuleDict
        layers containing computations to be performed

    Methods
    -------
    append(layer_id, layer)
    add_output_branch(layer_id, input_dim, output_dim)
    vae_train_fn(input, label=None, loss_fn=torch.nn.functional.mse_loss,
                 optimizer=None, layer_id=None, branch_name='branch',
                 forward_fn=None, **kwargs)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_layer(layer_id, n_epochs, trainloader, loss_fn, optimizer,
                monitor=100, **kwargs)
    train_model(n_epochs, trainloader, loss_fn, optimizer, monitor=100, **kwargs)
    optimize_texture(n_steps, seed, loss_fn, optimizer, input=[],
                     transform=None, monitor=100, **kwargs)

    References
    ----------
    Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.
    arXiv preprint arXiv:1312.6114.
    """
    def __init__(self, model=None):
        super(VAE, self).__init__(model)

    def gaussian_sample(self, args):
        """
        Sample from `mu`, `logvar` as:

            `mu + exp(logvar/2) * randn_like(logvar)`

        if `self.training==True` otherwise, return `mu`.

        Parameters
        ----------
        args : tuple or list
            mu, logvar passed as tuple from branching module
            (see `add_output_branch`)

        Returns
        -------
        sample : torch.Tensor
            sample from `mu`, `logvar`
        """
        mu, logvar = args
        if self.training:
            return mu + torch.exp(logvar/2.) * torch.randn_like(logvar)
        else:
            return mu

    def add_output_branch(self, output_dim, layer_id=None, branch_name='branch'):
        """
        Add branching module which factorizes latent units into `mu`, `logvar`
        and outputs samples as `mu + exp(logvar/2.) * randn_like(logvar)`.

        Parameters
        ----------
        output_dim : int
            number of dimensions in latent space (i.e., output dimension from
            `Autoencoder` layer)
        layer_id : str
            layer id to add branching module (should be an `Autoencoder` layer)
            [default: None, set to `self.get_layer_ids()[-1]`]
        branch_name : str
            name for branching module
            [default: 'branch']

        Returns
        -------
        None

        Notes
        -----
        When training with `vae_train_fn`, the given `layer_id` and `branch_name`
        should be used.

        See Also
        --------
        vae_train_fn
        """
        if layer_id is None:
            layer_id = self.get_layer_ids()[-1]
        assert layer_id in self.layers
        # set branching module
        branch = Branch([torch.nn.Linear(output_dim, output_dim),
                         torch.nn.Linear(output_dim, output_dim)],
                        ['mu','logvar'])
        # update modules in layer_id
        self.update_modules([layer_id], 'forward_layer', branch_name, branch)
        self.update_modules([layer_id], 'forward_layer', 'sample',
                            self.gaussian_sample)

    def kl_loss(self, mu, logvar):
        """
        KL divergence for VAE:

            `-0.5 * sum(1 + logvar - mu**2 - exp(logvar), -1)`

        Parameters
        ----------
        mu : torch.Tensor
            mu of latent space with shape (batch_size, latent_dim)
        logvar : torch.Tensor
            log of variance of latent space with shape (batch_size, latent_dim)

        Returns
        -------
        kl_div : torch.Tensor
            KL divergence between q(z|x) and p(z)

        See Also
        --------
        vae_train_fn
        add_output_branch
        """
        mu = mu.flatten(1)
        logvar = logvar.flatten(1)
        return -0.5 * torch.sum(1. + logvar - mu**2 - torch.exp(logvar), -1)

    def vae_train_fn(self, inputs, label=None, loss_fn=nn.MSELoss(reduction='sum'),
                     optimizer=None, layer_id=None, branch_name='branch',
                     pre_layer_ids=None, forward_fn=None, **kwargs):
        """
        VAE training function combining reconstruction and KL loss:

            `recon(x_hat, input) - 0.5 * sum(exp(logvar) * mu**2 - 1 - logvar)`

        where `x_hat` is the reconstructed input.

        Parameters
        ----------
        inputs : torch.Tensor
            input to model for training
        label : torch.Tensor or None
            label passed from trainloader (currently unused)
            [default: None]
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function for reconstruction error
            [default: torch.nn.MSELoss(reduction='sum')]
        optimizer : torch.optim
            optional, optimizer used for updating parameters (to show
            learning rate) [default: None]
        layer_id : str
            layer id for the layer containing the `mu` and `var` branch
            [default: None, set to `self.get_layer_ids()[-1]`]
        branch_name : str
            name of branch module in `self.layers[layer_id]`
            [default: `branch`]
        pre_layer_ids : list
            list of layer_ids between `input` and `layer_id` to be called. If
            training "layer-wise", `pre_layer_ids` should likely be [] (i.e.,
            only `layer_id` will be used as `input` is considered to be output
            of previous layer).
            [default: None, set to `self.get_layer_ids(layer_id)[:-1]`]
        forward_fn : function or None
            forward function used to pass inputs through model e.g. to obtain
            inputs to a given layer if training layer-wise.
            [default: None, set to `lambda x: x[0]`]

        Returns
        -------
        loss : torch.Tensor
            combined reconstruction and KL loss for training VAE

        Notes
        -----
        The layer `self.layers[layer_id]` should contain the branching module
        that models `mu` and `logvar`. This can be set using `add_output_branch`.

        See Also
        --------
        add_output_branch
        """
        if layer_id is None:
            layer_id = self.get_layer_ids()[-1]
        # get pre_layer_ids is None
        if pre_layer_ids is None:
            pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        # get x from potentially passing through model
        if forward_fn is None:
            x = inputs[0]
        else:
            x = forward_fn(inputs)
        # zero gradients
        if optimizer is not None:
            optimizer.zero_grad()
        # get forward pass, collecting mu, logvar
        output = {layer_id: {branch_name: []}}
        h = self.apply(x, pre_layer_ids + [layer_id], output=output)
        pre_layer_ids.reverse()
        # reconstruct from h for MSE loss
        recon = self.apply(h, [layer_id] + pre_layer_ids, forward=False)
        # compute KL loss
        mu, logvar = output.get(layer_id).get(branch_name)[0]
        loss = torch.add(loss_fn(recon, x),
                         torch.sum(self.kl_loss(mu, logvar)))
        loss.backward()
        # step for generator
        if optimizer is not None:
            optimizer.step()
        return loss.item()

    def train_model(self, n_epochs, trainloader,
                    loss_fn=nn.MSELoss(reduction='sum'), optimizer=None,
                    monitor=100, **kwargs):
        """
        Train VAE using `vae_train_fn` and factorized `mu` and `logvar` in last
        layer of model (i.e., `self.get_layer_ids()[-1]`).

        Parameters
        ----------
        n_epochs : int
            number of epochs to train for (complete passes through dataloader)
        trainloader : torch.utils.data.DataLoader
            dataloader containing training (data, label) pairs
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function to opimize during training
            [default: torch.nn.MSELoss(reduction='sum')]
        optimizer : torch.optim
            optimizer used to update parameters during training
            [default: None]
        monitor : int
            number of batches between plotting loss, showing weights, etc.
            [default: 100]
        **kwargs : **dict
            #TODO: use kwargs from Model.train_model

        Returns
        -------
        loss_history : list
            list of loss values at each monitoring step
            (i.e., len(loss_history) == (n_epochs * len(trainloader) / monitor))

        See Also
        --------
        vae_train_fn
        """
        # set forward function to just pass inputs[0]
        kwargs.setdefault('forward_fn', lambda x: x[0])
        kwargs.setdefault('train_fn', self.vae_train_fn)
        # train for n_epochs
        return self.train_n_epochs(n_epochs, trainloader, loss_fn, optimizer,
                                   monitor=100, **kwargs)

    def train_layer(self, layer_id, n_epochs, trainloader,
                    loss_fn=nn.MSELoss(reduction='sum'), optimizer=None,
                    monitor=100, **kwargs):
        """
        Train VAE layer-wise using `vae_train_fn`, for which `layer_id` should
        be the layer containing the branching module modeling `mu` and `logvar`.

        Parameters
        ----------
        layer_id : str
            layer id of `Autoencoder` layer to train (should also contain
            branching module if using `vae_train_fn`)
        n_epochs : int
            number of epochs to train for (complete passes through dataloader)
        trainloader : torch.utils.data.DataLoader
            dataloader containing training (data, label) pairs
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function to opimize during training
            [default: torch.nn.MSELoss(reduction='sum')]
        optimizer : torch.optim
            optimizer used to update parameters during training
            [default: None]
        monitor : int
            number of batches between plotting loss, showing weights, etc.
            [default: 100]
        **kwargs : **dict
            see `train_model` and `vae_train_fn` for optional keyword arguments

        Returns
        -------
        loss_history : list
            list of loss values at each monitoring step
            (i.e., len(loss_history) == (n_epochs * len(trainloader) / monitor))

        Notes
        -----
        When using `vae_train_fn`, `forward_fn` is set to either
        `lambda x: x[0]` if `len(self.get_layer_ids(layer_id)[:-1]) == 0` or
        `lambda x: self.apply(x[0], self.get_layer_ids(layer_id)[:-1])` and
        `pre_layer_ids` is set to []. See `vae_train_fn` for more details.

        See Also
        --------
        train_model
        vae_train_fn
        """
        # set forward function
        pre_layer_ids = self.get_layer_ids(layer_id)[:-1]
        if len(pre_layer_ids) > 0:
            forward_fn = lambda x: self.apply(x[0], pre_layer_ids)
        else:
            forward_fn = lambda x: x[0]
        kwargs.setdefault('forward_fn', forward_fn)
        # set default layer_id, pre_layer_ids for vae_train_fn layer-wise
        kwargs.setdefault('layer_id', layer_id)
        kwargs.setdefault('pre_layer_ids', [])
        kwargs.setdefault('train_fn', self.vae_train_fn)
        # train for n_epochs
        return self.train_n_epochs(n_epochs, trainloader, loss_fn, optimizer,
                                   monitor=monitor, **kwargs)

class GAN(Model):
    """
    Generative Adversarial Network
    Model contains `generator` and `discriminator` networks as follows:

        `FeedForwardNetwork(
          (layers): ModuleDict(
            (generator): Branch(
              (forward_layer): Sequential(
                (real): Sequential()
                (fake): generator
              )
              (reconstruct_layer): Sequential()
            )
            (discriminator): discriminator
          )
         )`

    which outputs a tensor with shape (m*2,) with first m items corresponding
    to probability of real data and last m items corresponding to fake data.

    Attributes
    ----------
    data_shape : tuple
        shape of input data
    layers : torch.nn.ModuleDict
        layers containing computations to be performed

    Methods
    -------
    add_discriminator(discriminator)
    add_generator(generator)
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_model(n_epochs, trainloader, loss_fn=F.binary_cross_entropy,
                optimizer=None, k=1, monitor=100, **kwargs)

    Notes
    -----
    The generator network should take an input of any shape where `input.shape[0]`
    indicates the batch size. The first module in the generator should then
    sample from a random normal distribution (e.g.,
    `torch.randn(input_shape[0], n_features)`). The Branch module used in the
    model concatenates real data samples with the fake samples generated by the
    generator network.

    References
    ----------
    Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
    Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances
    in neural information processing systems (pp. 2672-2680).
    """
    def __init__(self, model=None, discriminator=None, generator=None):
        super(GAN, self).__init__(model)
        # add generator and discriminator to model
        self.add_generator(generator)
        self.add_discriminator(discriminator)

    def add_generator(self, generator):
        """
        Add generator network to model

        Parameters
        ----------
        generator : rf_pool.model or torch.nn.Module
            generator used to generate fake data to fool discriminator

        Returns
        -------
        None
        """
        self.append('generator', Branch([torch.nn.Sequential(), generator],
                                        ['real', 'fake'], cat_dim=0))

    def add_discriminator(self, discriminator):
        """
        Add discriminator network to model

        Parameters
        ----------
        discriminator : rf_pool.model or torch.nn.Module
            discriminator used to discriminate between real and fake data

        Returns
        -------
        None
        """
        self.append('discriminator', discriminator)

    def minimax_train_fn(self, inputs, label=None, loss_fn=F.binary_cross_entropy,
                         optimizer=None, k=1, forward_fn=None, **kwargs):
        """
        Minimax training function

        Parameters
        ----------
        inputs : torch.Tensor
            input to model for training
        label : torch.Tensor or None
            label passed from trainloader (currently unused)
            [default: None]
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function taking output of discriminator and ones for real data
            and zeros for fake data [default: F.binary_cross_entropy]
        optimizer : torch.optim
            optional, optimizer used for updating parameters (to show
            learning rate) [default: None]
        k : int
            number of minibatches for optimizing discriminator only
            [default : 1]
        forward_fn : function or None
            forward function used to pass inputs through model to obtain
            probability of real and fake data to be passed to `loss_fn`
            [default: None, set to `self.forward`]

        Returns
        -------
        error_d : float
            error for discriminator (i.e., `log(p(D(x))) + log(1 - D(G(x)))`)
        """
        # get y from model
        if forward_fn is None:
            y = self.forward(inputs[0])
        else:
            y = forward_fn(inputs)
        # zero gradients
        if optimizer is not None:
            optimizer.zero_grad()
        # get batch size
        m = y.shape[0] // 2
        # update discriminator
        self.set_requires_grad(pattern='generator', requires_grad=False)
        error_d = loss_fn(y, torch.cat([torch.ones(m,1), torch.zeros(m,1)]),
                          reduction='sum')
        error_d = torch.div(error_d, m)
        error_d.backward()
        error_d = error_d.item()
        # step for discriminator, zero grad for generator
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        # if k > 1 and _k_updates < k, return error_d
        if not hasattr(self, 'discriminator_updates'):
            self._k_updates = 1
        else:
            self._k_updates += 1
        if k > 1 and self._k_updates < k:
            return error_d
        elif self._k_updates >= k:
            self._k_updates = 0
        # generate fake data and prediction
        self.set_requires_grad(pattern='generator', requires_grad=True)
        self.set_requires_grad(pattern='discriminator', requires_grad=False)
        y_g = self.apply(torch.zeros(m), generator={'module_names': ['fake']})
        # update generator
        error_g = loss_fn(y_g, torch.ones(m, 1))
        error_g.backward()
        self.set_requires_grad(pattern='discriminator', requires_grad=True)
        # step for generator
        if optimizer is not None:
            optimizer.step()
        return error_d

    def train_model(self, n_epochs, trainloader, loss_fn=F.binary_cross_entropy,
                    optimizer=None, k=1, monitor=100, **kwargs):
        """
        Train model using `minimax_train_fn` in which the discriminator is
        trained for `k` minibatches followed by the generator being trained once.

        Parameters
        ----------
        n_epochs : int
            number of epochs to train for (complete passes through dataloader)
        trainloader : torch.utils.data.DataLoader
            dataloader containing training (data, label) pairs
        loss_fn : torch.nn.modules.loss or rf_pool.losses
            loss function to opimize during training
            [default: F.binary_cross_entropy]
        optimizer : torch.optim
            optimizer used to update parameters during training
            [default: None]
        k : int
            number of minibatches for optimizing discriminator only
            [default : 1]
        monitor : int
            number of batches between plotting loss, showing weights, etc.
            [default: 100]
        **kwargs : **dict
            see `train_model` for optional keyword arguments

        Returns
        -------
        loss_history : list
            list of loss values at each monitoring step
            (i.e., len(loss_history) == (n_epochs * len(trainloader) / monitor))
        """
        # init _k_updates to 0
        self._k_updates = 0
        # train with minimax_train_fn and k discriminator updates
        kwargs.setdefault('train_fn', self.minimax_train_fn)
        return self.train_n_epochs(n_epochs, trainloader, loss_fn, optimizer,
                                   k=k, monitor=monitor, **kwargs)

class DeepBeliefNetwork(Model):
    """
    Deep Belief Network

    Attributes
    ----------
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers in deep belief network (each layer is RBM class)

    Methods
    -------
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
    train_model(n_epochs, trainloader, optimizer, k=1, persistent=None,
                monitor=100, **kwargs)

    References
    ----------
    Hinton, Osindero & Teh (2006)
    """
    def __init__(self, model=None):
        super(DeepBeliefNetwork, self).__init__(model)

    def init_complementary_prior(self, layer_id, module_name='hidden'):
        """
        Initialize complementary priors using transpose of weights from previous
        layer.

        Parameters
        ----------
        layer_id : str
            layer id to initialize weights
        module_name : str
            name of module containing weights to initialize (must match module
            name of previous layer) [defualt: 'hidden']

        Returns
        -------
        None

        Notes
        -----
        This function initializes weights to the transpose of the weights from
        the previous layer mulitplied by an orthonormal transformation matrix
        to project to the number of hidden units in the current layer. This
        allows the current layer to initialize as a complementary prior over the
        previous layer in order to learn more efficiently. See References for
        more information.

        References
        ----------
        Hinton, Osindero & Teh (2006)
        """#TODO: update to allow for different kernel size
        # get weight, bias and transpose bias names
        weight_name = module_name + '_weight'
        bias_name = module_name + '_bias'
        transpose_bias_name = module_name + '_transpose_bias'
        # get previous and current layers
        layer_ids = self.get_layer_ids(layer_id)[-2:]
        prev_layer, layer = self.get_layers(layer_ids)
        # get pervious layer weights and transpose (and flip if 4d)
        w = getattr(prev_layer, weight_name).detach()
        wT = w.transpose(0,1)
        if wT.ndimension() == 4:
            wT = wT.flip((-2,-1))
        # create orthonormal transformation to layer_id to_ch
        w0_shape = getattr(layer, weight_name).shape
        to_ch = w0_shape[0]
        from_ch = wT.shape[0]
        u = functions.modified_gram_schmidt(torch.randn(from_ch, to_ch))
        # create new weights with orthonormal transformation
        w_prior = torch.matmul(wT.flatten(1).t(), u).t()
        w_prior = w_prior.reshape(w0_shape)
        # init layer weights
        layer.init_weights(pattern=weight_name, fn=lambda x: w_prior)
        # init visible bias
        layer.init_weights(pattern=transpose_bias_name,
                           fn=lambda x: getattr(prev_layer, bias_name).detach())

    def _get_free_energy_loss(self, layer_id):
        def free_energy_loss(x):
            fe = torch.mean(self.layers[layer_id].free_energy(x))
            hidsize = torch.as_tensor(self.layers[layer_id].hidden_shape(x.shape))
            hidsize = torch.prod(hidsize.unsqueeze(-1)[2:])
            return torch.div(fe, hidsize)
        return free_energy_loss

    def contrastive_divergence(self, input, k=1):
        #TODO: not technically the same as Hinton, Osindero, Teh (2006)
        #TODO: should unlink rec/gen weights for all but top layer
        #TODO: then update is based on s(d - p) for lower layers
        #TODO: and normal CD for top layer
        # get free_energy functions for each layer
        layer_ids = self.get_layer_ids()
        layer_ids = [None,] + layer_ids
        fe_losses = [self._get_free_energy_loss(layer_id)
                     for layer_id in layer_ids[1:]]
        # get positive phase statistics
        pos_output = OrderedDict([(layer_id, {'sample': []})
                                  if layer_id is not None else (None, [])
                                  for layer_id in layer_ids[:-1]])
        # gibbs sample output layer
        with torch.no_grad():
            output = self.apply(input, output=pos_output, layer_ids=layer_ids)
            # if persistent, use output from persistent pass as top-down input
            if hasattr(self, 'persistent') and self.persistent is not None:
                output = self.apply(self.persistent, layer_ids=layer_ids)
            # gibbs sample
            output = self.layers[layer_ids[-1]].gibbs_hvh(output, k=k)[-1]
        # get negative phase statistics
        neg_output = OrderedDict([(layer_id, {'sample': []})
                                  for layer_id in layer_ids[1:]])
        neg_layer_ids = layer_ids.copy()[1:]
        neg_layer_ids.reverse()
        with torch.no_grad():
            reconstruct = self.apply(output, output=neg_output,
                                     layer_ids=neg_layer_ids, forward=False)
        # update persistent
        if hasattr(self, 'persistent') and self.persistent is not None:
            self.persistent = reconstruct
        # get difference in free_energy
        loss = torch.zeros(1, requires_grad=True)
        for fe_loss_fn, pos, neg in zip(fe_losses, pos_output.values(),
                                        neg_output.values()):
            p_sample = pos.get('sample')[0] if type(pos) is dict else pos[0]
            n_sample = neg.get('sample')[0] if type(neg) is dict else neg[0]
            loss = loss + torch.sub(fe_loss_fn(p_sample), fe_loss_fn(n_sample))
        return loss

    def untie_weights(self, pattern=''):
        """
        Untie `reconstruct_layer` weights from `forward_layer` weights
        for learning directed graphical model

        Parameters
        ----------
        pattern : str
            pattern of weights in `reconstruct_layer` to be untied
            [default: '', all weights in `reconstruct_layer` untied]

        Returns
        -------
        None
        """
        layer_ids = self.get_layer_ids()
        for layer in self.get_layers(layer_ids):
            if hasattr(layer, 'untie_weights'):
                layer.untie_weights(pattern=pattern)

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        #TODO: implement BEAM training (-KL based on NN of data/fantasy particles)
        return Model.train_layer(self, layer_id, n_epochs, trainloader, None,
                                 optimizer, monitor=monitor, k=k, **kwargs)

    def train_model(self, n_epochs, trainloader, optimizer, k=1, persistent=None,
                    monitor=100, **kwargs):
        # set persistent attribute
        self.persistent = persistent
        # set contrastive_divergence loss
        cd_loss = losses.KwargsLoss(self.contrastive_divergence, n_args=1, k=k)
        loss_fn = losses.LayerLoss(self, {None: []}, cd_loss, layer_ids=[None])
        # train
        return Model.train_model(self, n_epochs, trainloader, loss_fn, optimizer,
                                 monitor=monitor, **kwargs)

class DeepBoltzmannMachine(DeepBeliefNetwork):
    """
    Deep Boltzmann Machine

    Attributes
    ----------
    data_shape : shape of input data (optional, default: None)
    layers : torch.nn.ModuleDict
        RBM layers for forward/reconstruct pass (each layer is RBM class)

    Methods
    -------
    append(layer_id, layer)
    insert(idx, layer_id, layer)
    remove(layer_id)
    apply(input, layer_ids=[], forward=True, output={}, output_layer=None,
          **kwargs)
    train_layer(layer_id, n_epochs, trainloader, optimizer, k=1, monitor=100,
                **kwargs)
        train deep boltzmann machine with contrastive divergence
    train_model(n_epochs, trainloader, optimizer, k=1, n_iter=10, persistent=None,
                monitor=100, **kwargs)

    References
    ----------
    Salakhutdinov & Hinton (2009)
    """
    def __init__(self, model=None):
        super(DeepBoltzmannMachine, self).__init__(model)

    def train_layer(self, layer_id, n_epochs, trainloader, optimizer, k=1,
                    monitor=100, **kwargs):
        # update module to double input and double top-down
        layer_ids = self.get_layer_ids()
        mul_op = ops.Op(lambda x: 2. * x)
        act_op0 = self.update_modules([layer_ids[0]], 'forward_layer',
                                      'activation', mul_op, overwrite=False,
                                      append=False)
        if len(layer_ids) > 1 and layer_id == layer_ids[-1]:
            act_op1 = self.update_modules([layer_id], 'reconstruct_layer',
                                          'activation', mul_op, overwrite=False,
                                          append=False)
        # train
        try:
            output = Model.train_layer(self, layer_id, n_epochs, trainloader,
                                       None, optimizer, monitor=monitor, k=k,
                                       **kwargs)
        finally:
            # replace mul_op with original activation operation
            self.update_modules([layer_ids[0]], 'forward_layer', 'activation',
                                act_op0, overwrite=True)
            if len(layer_ids) > 1 and layer_id == layer_ids[-1]:
                self.update_modules([layer_ids[0]], 'reconstruct_layer',
                                    'activation', act_op1, overwrite=True)
        return output

    def train_model(self, n_epochs, trainloader, optimizer, k=1, n_iter=10,
                    persistent=None, monitor=100, **kwargs):
        # set persistent
        self.persistent = persistent
        # set loss function
        cd_loss = losses.KwargsLoss(self.contrastive_divergence, n_args=1,
                                    k=k, n_iter=n_iter)
        loss_fn = losses.LayerLoss(self, {None: []}, cd_loss, layer_ids=[None])
        # train
        return Model.train_model(self, n_epochs, trainloader, loss_fn, optimizer,
                                 monitor=monitor, **kwargs)

    def contrastive_divergence(self, input, k=1, n_iter=10):
        #TODO: update mean_field to condition on output
        # positive phase mean field
        layer_ids = self.get_layer_ids()
        layers = self.get_layers(layer_ids)
        hids = None
        with torch.no_grad():
            for _ in range(n_iter):
                v, hids = self.layer_gibbs(input, hids, sampled=False)
        # get positive energies
        pos_energy = []
        for i, layer in enumerate(layers):
            if i == 0:
                pos_energy_i = torch.mean(layer.energy(input, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                pos_energy_i = torch.mean(layer.energy(h_n, hids[i][1]))
            hidsize = torch.prod(torch.as_tensor(hids[i][1].shape[-2:]))
            pos_energy.append(torch.div(pos_energy_i, hidsize))
        # negative phase for each layer
        if hasattr(self, 'persistent') and self.persistent is not None:
            v = self.persistent
        hids = None
        with torch.no_grad():
            for _ in range(k+1):
                even = (hids is not None)
                v, hids = self.layer_gibbs(v, hids, sampled=True, even=even)
        if hasattr(self, 'persistent') and self.persistent is not None:
            self.persistent = v
        # get negative energies
        neg_energy = []
        for i, layer in enumerate(layers):
            if i == 0:
                neg_energy_i = torch.mean(layer.energy(v, hids[i][1]))
            else:
                h_n = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[0]
                neg_energy_i = torch.mean(layer.energy(h_n, hids[i][1]))
            hidsize = torch.prod(torch.as_tensor(hids[i][1].shape[-2:]))
            neg_energy.append(torch.div(neg_energy_i, hidsize))
        # return mean difference in energies
        loss = torch.zeros(1, requires_grad=True)
        for (p, n) in zip(pos_energy, neg_energy):
            loss = torch.add(loss, torch.sub(p, n))
        return loss

    def layer_gibbs(self, input, hids=None, sampled=False, pooled=False,
                    even=True, odd=True):
        # get layer_ids
        layer_ids = self.get_layer_ids()
        layers = self.get_layers(layer_ids)
        # get idx based on sampled bool
        idx = np.int(sampled)
        # get hids from forward pass with weights doubled
        if hids is None:
            mul_op = ops.Op(lambda x: 2. * x)
            hid_ops = self.update_modules(layer_ids[:-1], 'forward_layer',
                                          'hidden', mul_op, overwrite=False)
            try: # forward pass with doubled weights
                saved_outputs = dict([(id, {'hidden': []}) for id in layer_ids])
                self.apply(input, layer_ids, output=saved_outputs)
                hids = [v.get('hidden')[0] for v in saved_outputs.values()]
            finally:
                self.update_modules(layer_ids[:-1], 'forward_layer', 'hidden',
                                    hid_ops)
            hids = [(h,) + layer.sample(h, 'forward_layer', pooled)
                    for h, layer in zip(hids, layers)]
        if even:
            # update even layers
            for i, layer in list(enumerate(layers))[::2]:
                if i > 0:
                    v = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[idx]
                else:
                    v = input
                if i < (len(layers) - 1):
                    t = layers[i+1].sample_v_given_h(hids[i+1][idx+1])[idx+1]
                    hids[i] = layer.sample_h_given_vt(v, t, pooled)
                else:
                    hids[i] = layer.sample_h_given_v(v, pooled)
        # get v out
        v_out = layers[0].sample_v_given_h(hids[0][idx+1])[idx+1]
        # update odd layers
        if odd:
            for i, layer in list(enumerate(layers))[1::2]:
                v = layers[i-1].sample(hids[i-1][0], 'forward_layer', True)[idx]
                if i < (len(layers) - 1):
                    t = layers[i+1].sample_v_given_h(hids[i+1][idx+1])[idx+1]
                    hids[i] = layer.sample_h_given_vt(v, t, pooled)
                else:
                    hids[i] = layer.sample_h_given_v(v, pooled)
        return v_out, hids

if __name__ == '__main__':
    import doctest
    doctest.testmod()
