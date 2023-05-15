
import torch
import hydra
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from barkify.datas import StageDataloader
from barkify.pl_model import NanoGPT
from barkify.utils import Bestckpt

@hydra.main(config_path='configs', config_name='barkify')
def main(cfg=None):
    
    exp_name = f'stage_{cfg.stage}'
    exp_cfg = cfg[f'stage{cfg.stage}']
    exp_root_dir = f'{cfg.start_path}/{cfg.name}/{exp_name}'

    # define datas
    train_loader = StageDataloader(exp_cfg, cfg.stage, 'train')
    val_loader = StageDataloader(exp_cfg, cfg.stage, 'eval')

    # define model
    model = NanoGPT(model_config = exp_cfg.model, **exp_cfg.optim)

    # define trainer
    trainer = pl.Trainer(
        default_root_dir=exp_root_dir,
        callbacks = ModelCheckpoint(**cfg.common.ckpt),
        max_steps = exp_cfg.optim.max_iters,
        gradient_clip_val = exp_cfg.optim.gradient_clip,
        **cfg.common.trainer
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # load best ckpt
    ckpt = Bestckpt(exp_root_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()













