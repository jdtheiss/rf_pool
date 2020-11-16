from argparse import ArgumentParser
import inspect

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from rf_pool.solver.build import build_solver
from rf_pool.utils.functions import load_from_yaml

def parse_args():
    # set argument parser with debug, config_file
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False,
                        help='show config, model, loss, etc. during training')
    parser.add_argument('--config_file', type=str, required=False,
                        help='config file used to build model, loss, etc.')
    parser.add_argument('--entity', type=str, required=False,
                        help='team name for logging to wandb')
    # add WandbLogger args
    default_kwargs = dict(inspect.signature(WandbLogger).parameters)
    default_kwargs.pop('kwargs', None)

    # add pl.Trainer args
    default_kwargs.update(inspect.signature(pl.Trainer).parameters)
    for k, v in default_kwargs.items():
        # set type (unless typing class)
        if type(v.annotation) is type:
            arg_type = v.annotation
        else:
            arg_type = None
        # add argument
        parser.add_argument('--%s' % k, type=arg_type, default=v.default,
                            help='See pytorch_lightning for information.')


    return parser.parse_args()

def main(args):
    # load config
    cfg = {}
    if args.config_file:
        cfg = load_from_yaml(args.config_file)

    # debug
    cfg.setdefault('DEBUG', args.debug)

    # build solver
    solver = build_solver(cfg)

    # get trainer kwargs
    kwargs = vars(args)
    kwargs.pop('debug')
    kwargs.pop('config_file')

    # get wandb kwargs
    wandb_keys = inspect.signature(WandbLogger).parameters.keys()
    wandb_kwargs = dict((k, kwargs.pop(k)) for k in wandb_keys if k in kwargs)
    wandb_kwargs.update({'entity': kwargs.pop('entity')})

    # init wandb and logger
    if any(v for v in wandb_kwargs.values()):
        # init run
        run = wandb.init(name=wandb_kwargs.get('name'),
                         project=wandb_kwargs.get('project'),
                         entity=wandb_kwargs.get('entity'),
                         id=wandb_kwargs.get('id'), anonymous='allow',
                         resume='allow', reinit=True,
                         mode='offline' if wandb_kwargs.get('offline') else 'online')
        wandb_kwargs.update({'experiment': run})
        # init logger
        logger = WandbLogger(**wandb_kwargs)
        kwargs.update({'logger': logger})

    # build trainer
    trainer = pl.Trainer(**kwargs)

    # fit
    trainer.fit(solver)

if __name__ == '__main__':
    args = parse_args()
    main(args)
