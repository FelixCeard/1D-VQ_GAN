import librosa
import numpy as np
import sys

# sys.path.insert(1, './1D-VQ_GAN')

import logging
import os
import warnings
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import soundfile as sf
import torch
import wandb
from VQ_train_utils import instantiate_from_config
import librosa.display
from tqdm import tqdm
import seaborn as sns

if __name__ == '__main__':
    path_drive = '../Logs'

    # wandb
    wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')
    wandb_logger = WandbLogger(project="[NNTI]lstm")

    # load configs
    print('loading configs')
    configs = [OmegaConf.load('configs/[LSTM].yaml')]
    config = OmegaConf.merge(*configs)

    # model
    print('loading model')
    model = instantiate_from_config(config.model)
    # model.eval()

    # data
    print('loading data')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print('init callbacks')
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + 'Unsup'
    logdir = os.path.join("logs", nowname)
    logdir = os.path.join(path_drive, 'NNTI', logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    iterator = data.val_dataloader()._get_iterator()
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True, save_top_k=5, monitor='test/F1_epoch', every_n_train_steps=2),
        # AudioLoggingCallback([next(iterator) for _ in range(10)])
    ]

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        accelerator="gpu",
        devices=-1
    )

    trainer.fit(model, data)


    