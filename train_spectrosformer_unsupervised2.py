import librosa
import numpy as np
import sys

# sys.path.insert(1, './1D-VQ_GAN')

import logging
import os
import warnings
from datetime import datetime

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

if __name__ == '__main__':
    path_drive = '../Logs'

    # wandb
    wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')
    wandb_logger = WandbLogger(project="[NNTI]SpectroformerUsupervised2")

    # load configs
    print('loading configs')
    configs = [OmegaConf.load('configs/[TRAIN]SpectogramTransformerUnsupervised2.yaml')]
    config = OmegaConf.merge(*configs)

    # model
    print('loading model')
    model = instantiate_from_config(config.model)

    # data
    print('loading data')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # dirs
    print('init callbacks')
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + 'Unsup2'
    logdir = os.path.join("logs", nowname)
    logdir = os.path.join(path_drive, 'NNTI', logdir)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)


    class AudioLoggingCallback(Callback):
        def __init__(self, batch):
            self.batch = batch
            # self.sample = sample.reshape(26, -1)
            self.index = 0
            os.makedirs('./logging_audio', exist_ok=True)

        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            model = pl_module.first_stage_model
            fig, ax = plt.subplots(2, 10, figsize=(10, 2), sharex=True)

            for i in range(10):
                sample = model.get_input(self.batch[i], model.image_key)
                with torch.no_grad():
                    sample = sample.to(model.device)
                    rec = model.forward(sample.clone())[0]

                    sample = sample.detach().cpu().numpy()
                    rec = torch.squeeze(rec, 0).detach().cpu().numpy()

                ax[0][i].set_xticks([])
                ax[0][i].set_yticks([])
                ax[1][i].set_xticks([])
                ax[1][i].set_yticks([])

                ax[0][i].xaxis.set_visible(False)
                ax[0][i].yaxis.set_visible(False)
                ax[1][i].xaxis.set_visible(False)
                ax[1][i].yaxis.set_visible(False)

                librosa.display.specshow(
                    sample,
                    sr=8_000,
                    x_axis='time',
                    y_axis='mel',
                    cmap='viridis',
                    fmax=4000,
                    hop_length=80,
                    ax=ax[0][i]
                )

                librosa.display.specshow(
                    rec,
                    sr=8_000,
                    x_axis='time',
                    y_axis='mel',
                    cmap='viridis',
                    fmax=4000,
                    hop_length=80,
                    ax=ax[1][i]
                )

            path = os.path.join('logging_audio', f'{self.index}.png')
            fig.savefig(path)
            fig.clear()
            plt.close(fig)

            pl_module.logger.log_metrics({
                'reconstruction': wandb.Image(path)
            })

            self.index += 1

        def get_audio(self, sample, sample_rate, caption):
            path = os.path.join('./logging_audio', f'{self.index}{caption}.wav')
            sf.write(path, np.ravel(sample), sample_rate, 'PCM_24')
            return wandb.Audio(data_or_path=path, caption=caption, sample_rate=sample_rate)


    # callbacks
    iterator = data.val_dataloader()._get_iterator()
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True, save_top_k=5, monitor='test/F1_epoch', every_n_train_steps=2),
        # AudioLoggingCallback([next(iterator) for _ in range(10)])
    ]

    # trainer
    accumulate_grad_batches = 12
    batch_size = config.data.params.batch_size
    model.learning_rate = accumulate_grad_batches * batch_size * config.model.base_learning_rate

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        accelerator="gpu",
        devices=-1
    )

    trainer.fit(model, data)

