from math import log

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from pytorch_lightning.metrics import Metric as PLMetric
import wandb

class Metric(PLMetric):
    """
    Base class for metrics

    Parameters
    ----------
    metrics : dict
        dictionary of metric functions to use as (name, metric_fn) key/value pair
        [default: {}]
    n_steps : int
        number of steps between logging [default: 1]
    """
    def __init__(self, metrics={}, n_steps=1, model=None):
        super(Metric, self).__init__()
        self.metrics = metrics
        self.n_steps = n_steps
        self.cnt = -1

        # build metrics requiring model
        self.build_from_model(model)

    def build_from_model(self, model=None):
        for name, metric_fn in self.metrics.items():
            if hasattr(metric_fn, 'build_from_model') and model is not None:
                metric_fn.build_from_model(model)

    def update(self, *args, **kwargs):
        self.cnt += 1
        if self.cnt % self.n_steps != 0:
            return

        output = {}
        for name, metric_fn in self.metrics.items():
            output.update({name: metric_fn(*args, **kwargs)})
        self.output = output

    def compute(self):
        output = {}
        for name, metric_fn in self.metrics.items():
            if hasattr(metric_fn, 'compute'):
                result = metric_fn.compute()
            else:
                result = self.output.get(metric_fn)
            output.update({name: result})
        return output

class HookMetric(Metric):
    """"""
    def __init__(self, module_name, module_fn):
        super(HookMetric, self).__init__()

        self.outputs = {}
        self.hooks = []

        self.reset_hooks(model, module_name, module_fn)

    def named_hook_fn(self, mod_name, fn_name, **kwargs):
        # hook function to pass input, **kwargs to fn
        def fn(mod, input):
            # append function output
            fn = getattr(mod, fn_name)
            if self.outputs.get(mod_name) is None:
                self.outputs.update({mod_name: []})
            self.outputs.get(mod_name).append(fn(*input, **kwargs))
        return fn

    def reset_hooks(self, model, module_name, module_fn):
        # get named modules as dict
        named_modules = dict(model.named_modules())

        # get module directly
        mod = named_modules.get(module_name)
        # otherwise, find modules ending with module_name
        if mod is None:
            mods = [(n, m) for n, m in named_modules.items() if n.endswith(module_name)]
        else:
            mods = [(mod, module_name)]

        assert len(mods) > 0, 'No modules found for "%s".' % module_name

        # set hooks
        self.hooks = []
        for name, mod in mods:
            hook_fn = named_hook_fn(name, module_fn, **kwargs)
            self.hooks.append(mod.register_forward_pre_hook(hook_fn))

    def close_hooks(self):
        if not hasattr(self, 'hooks') or not isinstance(self.hooks, list):
            return
        [h.remove() for h in self.hooks]
        self.hooks = []

    def compute(self):
        return


class FlowMetric(PLMetric):
    """
    """
    def __init__(self, in_channels, image_size, name='log_p', **kwargs):
        super(FlowMetric, self).__init__()
        self.n_pixel = in_channels * np.prod(F._pair(image_size))
        self.idx = ['log_p','logdet'].index(name.lower())
        self.n_steps = kwargs.get('n_steps', 1)
        self.cnt = -1

    @torch.no_grad()
    def update(self, *args, **kwargs):
        self.cnt += 1
        if self.cnt % self.n_steps != 0:
            return

        log_var = args[0][self.idx]

        self.output = torch.mean(log_var / (log(2) * self.n_pixel))

    def compute(self):
        return self.output

class FlowSample(PLMetric):
    """
    """
    def __init__(self, image_size, n_sample=20, temp=0.7, model=None, **kwargs):
        super(FlowSample, self).__init__()
        self.image_size = image_size
        self.n_sample = n_sample
        self.temp = temp
        self.n_steps = kwargs.pop('n_steps', 1)
        self.cnt = -1

        self.make_grid_kwargs = kwargs

        # build from model unless None
        if model is not None:
            self.build_from_model(model)

    def _log_image(self, image, normalize=True, range=(-0.5, 0.5), **kwargs):
        images = []
        for image_i in image:
            image_i = make_grid(image_i.detach(), normalize=normalize,
                                range=range, **kwargs)
            images.append(wandb.Image(image_i))
        return images

    def build_from_model(self, model):
        self.model = model

    @torch.no_grad()
    def update(self, *args, **kwargs):
        self.cnt += 1
        if self.cnt % self.n_steps != 0:
            return

        device = args[0][0].device
        x = self.model.sample(self.n_sample, self.image_size, temp=self.temp,
                              device=device)

        self.output = self._log_image(x, **self.make_grid_kwargs)

    def compute(self):
        return self.output
