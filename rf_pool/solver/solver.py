import pprint as pp

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from rf_pool import build
from rf_pool.utils import functions

class Solver(pl.LightningModule):
    """
    Base pytorch_lightning class for training models

    Parameters
    ----------
    cfg : dict
        configuration used to build model, loss, metric, etc.
        should contain the following fields:
        Model : dict
            configuration to build model
        Loss : dict
            configuration to build loss (see loss.py)
        Metric : dict, optional
            configuration to build metrics (see metrics.py)
        MODEL_WEIGHTS : str, optional
            path to model checkpoint to load into model
        INIT_WEIGHTS : dict, optional
            configuration to initialize weights (see functions.init_weights)
        DEBUG : bool, optional
            True/False print each component (e.g., model, loss, etc.)
            [default: False]
        LOG : list
            list of model attributes to log (e.g., 'model.')
    """
    def __init__(self, cfg):
        super(Solver, self).__init__()
        self.cfg = cfg
        self.model = build.build_model(cfg)
        # # get loss/metric, update with model
        self.loss = build.build_loss(cfg)
        if hasattr(self.loss, 'build_from_model'):
            self.loss.build_from_model(self.model)
        self.metric = build.build_metric(cfg)
        if hasattr(self.metric, 'build_from_model'):
            self.metric.build_from_model(self.model)
        self._init_weights(cfg)
        self._load_model(cfg)
        self._debug(cfg=cfg, model=self.model, loss=self.loss, metric=self.metric)

    def _debug(self, **kwargs):
        if self.cfg.get('DEBUG') is not True:
            return
        pp.pprint(kwargs)

    def _load_model(self, cfg, strict=False):
        if cfg.get('MODEL_WEIGHTS') is None:
            return
        ckpt = torch.load(cfg.get('MODEL_WEIGHTS'))
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict)
        # debug
        self._debug(state_dict=state_dict)

    def _init_weights(self, cfg):
        functions.init_weights(self.model, **cfg.get('INIT_WEIGHTS', {}))

    def _log_attrs(self, cfg):
        # get fields, reduce from cfg
        if cfg.get('LOG') is None:
            return {}
        fields = cfg.get('LOG').get('fields', [])
        reduce = cfg.get('LOG').get('reduce', 'sum')
        # get attributes
        logs = functions.get_model_attrs(self.model, fields)
        # get reduce function (try __builtins__ first, then numpy)
        reduce_fn = __builtins__.get(reduce, getattr(np, reduce, None))
        # reduce_fn = getattr(np, reduce, None)
        if reduce_fn is None:
            # debug
            self._debug(log_attrs=logs)
            return logs
        # reduce
        for k, v in logs.items():
            logs.update({k: reduce_fn(v)})
        # debug
        self._debug(log_attrs=logs)
        return logs

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            return output
        elif len(args) == 2:
            target = args[1]
        else:
            target = None
        return output, target

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # pass forward and compute loss
        output, target = self.forward(*batch)
        loss = self.loss(output, target)
        # log losses
        logs = getattr(self.loss, 'logs', {})
        logs.update({'train_loss': loss})
        self.log_dict(logs)
        # log metrics
        metrics = self.metric(output, target)
        self.log_dict(metrics)
        # log other attributes
        attrs = self._log_attrs(self.cfg)
        self.log_dict(attrs)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        metrics = self.metric.compute()
        self.log_dict(metrics)
        return {'val_loss': sum(outputs) / len(outputs)}

    def test_step(self, batch):
        return self.training_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        metrics = self.metric.compute()
        self.log_dict(metrics)
        return {'test_loss': sum(outputs) / len(outputs)}

    def configure_optimizers(self):
        """
        #TODO:WRITEME
        """
        # build optimizers
        optims = build.build_optimizer(self.cfg)
        # if already optimizer, return
        if isinstance(optims, torch.optim.Optimizer):
            optims = [{'optimizer': optims}]
            self._debug(optimizer=optims)
            return optims
        # create optimizers from (fn, kwargs) pairs
        for optim_group in optims:
            # update optimizer
            fn, kwargs = optim_group.get('optimizer')
            # set params if attribute(s) given
            if kwargs.get('params') is not None:
                params = kwargs.get('params')
                if not isinstance(params, list):
                    params = [{'params': params}]
                for param in params:
                    param.update({'params': getattr(self.model, param['params']).parameters(),
                                  'name': param['params']})
                kwargs.update({'params': params})
            else: # otherwise, set model.parameters()
                kwargs.update({'params': self.model.parameters()})
            optim_group.update({'optimizer': fn(**kwargs)})
            # update with scheduler
            if optim_group.get('scheduler') is not None:
                fn, kwargs = optim_group.get('scheduler')
                lr_scheduler = fn(optim_group['optimizer'], **kwargs)
                optim_group.update({'scheduler': lr_scheduler})
        # debug
        self._debug(optimizer=optims)
        return optims

    def train_dataloader(self):
        dataloader = build.build_dataloader(self.cfg.get('Train', {}))
        # debug
        self._debug(train_dataloader=dataloader)
        return dataloader

    def val_dataloader(self):
        dataloader = build.build_dataloader(self.cfg.get('Val', {}))
        # debug
        self._debug(val_dataloader=dataloader)
        return dataloader

    def test_dataloader(self):
        dataloader = build.build_dataloader(self.cfg.get('Test', {}))
        # debug
        self._debug(test_dataloader=dataloader)
        return dataloader
