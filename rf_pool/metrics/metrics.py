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
