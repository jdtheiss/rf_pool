from pytorch_lightning.metrics import Metric as PLMetric

class Metric(PLMetric):
    """
    Base class for metrics

    Parameters
    ----------
    metrics : dict
        dictionary of metric functions to use as (name, metric_fn) key/value pair
        [default: {}]
    """
    def __init__(self, metrics={}):
        super(Metric, self).__init__()
        self.metrics = metrics

    def update(self, *args, **kwargs):
        output = {}
        for name, metric_fn in self.metrics.items():
            output.update({name: metric_fn(*args, **kwargs)})
        return output

    def compute(self):
        output = {}
        for name, metric_fn in self.metrics.items():
            if hasattr(metric_fn, 'compute'):
                result = metric_fn.compute()
            else:
                result = None
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
