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
        # get loss key, update cfg with model
        loss_key = build.check_keys(cfg, 'Loss', warn=False)
        if loss_key:
            cfg.get(loss_key).update({'model': self.model})
        self.loss = build.build_loss(cfg)
        self.metric = build.build_metric(cfg)
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
        # get reduce function
        reduce_fn = getattr(np, reduce, None)
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
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # pass forward and compute loss
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        # log losses
        logs = getattr(self.loss, 'logs', {})
        logs.update({'train_loss': loss})
        self.log_dict(logs)
        # log metrics
        metrics = self.metric(output, y) or {}
        self.log_dict(metrics)
        # log other attributes
        attrs = self._log_attrs(self.cfg)
        self.log_dict(attrs)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch):
        pass

    def configure_optimizers(self):
        # build optimizers
        optims = build.build_optimizer(self.cfg)
        for optim_group in optims:
            # update optimizer
            fn, kwargs = optim_group.get('optimizer')
            # set params if attribute(s) given
            if kwargs.get('params') is not None:
                params = kwargs.get('params')
                if isinstance(params, list):
                    for param in params:
                        param.update({'params': getattr(self.model, param['params'])})
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
