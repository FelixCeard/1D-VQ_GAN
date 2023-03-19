import numpy as np
import sys

# sys.path.insert(1, './1D-VQ_GAN')

import logging
import os
import warnings
from datetime import datetime

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import soundfile as sf
import torch
import wandb
from VQ_train_utils import instantiate_from_config

if __name__ == '__main__':
	# fuck warnings, me and my homies hate on warnings
	warnings.filterwarnings("ignore")
	path_drive = '../Logs'

	# load configs
	logging.debug('loading configs')
	configs = [OmegaConf.load('configs/small_transformer.yaml')]
	config = OmegaConf.merge(*configs)

	# # wandb
	wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')
	wandb_logger = WandbLogger(project="[NNTI]TransformerHome", config=config)

	# model
	logging.debug('loading model')
	model = instantiate_from_config(config.model)

	# data
	logging.debug('loading data')
	data = instantiate_from_config(config.data)
	data.prepare_data()
	data.setup()

	# dirs
	logging.debug('init callbacks')
	now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
	nowname = now + 'custom_transformer'
	logdir = os.path.join("logs", nowname)
	logdir = os.path.join(path_drive, 'NNTI', logdir)
	ckptdir = os.path.join(logdir, "checkpoints")
	cfgdir = os.path.join(logdir, "configs")

	os.makedirs(logdir, exist_ok=True)
	os.makedirs(ckptdir, exist_ok=True)
	os.makedirs(cfgdir, exist_ok=True)


	class AudioLoggingCallback(Callback):
		def __init__(self, sample):
			self.sample = sample
			self.index = 0
			os.makedirs('./logging_audio', exist_ok=True)

		def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
			model = pl_module.first_stage_model
			with torch.no_grad():
				self.sample = self.sample.to(model.device)
				rec = model.forward(self.sample.clone())[0]

			pl_module.logger.log_metrics({
				'OG': self.get_audio(self.sample.detach().cpu().numpy(), sample_rate=8_000, caption='OG'),
				'Reconstructed': self.get_audio(rec.detach().cpu().numpy(), sample_rate=8_000, caption='REC'),
			})
			self.index += 1

		def get_audio(self, sample, sample_rate, caption):
			path = os.path.join('./logging_audio', f'{self.index}{caption}.wav')
			# print(path, sample.shape)
			sf.write(path, np.ravel(sample), sample_rate, 'PCM_24')
			return wandb.Audio(data_or_path=path, caption=caption, sample_rate=sample_rate)


	# callbacks
	callbacks = [
		LearningRateMonitor(logging_interval='step'),
		ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True),
		AudioLoggingCallback(next(data.val_dataloader()._get_iterator())['wav'])
	]

	# trainer
	accumulate_grad_batches = 12
	batch_size = config.data.params.batch_size
	model.learning_rate = accumulate_grad_batches * batch_size * config.model.base_learning_rate

	trainer = Trainer(
		logger=wandb_logger,
		enable_checkpointing=True,
		callbacks=callbacks,
		# accumulate_grad_batches=accumulate_grad_batches,
		accelerator="gpu",
		devices=-1
	)

	trainer.fit(model, data)