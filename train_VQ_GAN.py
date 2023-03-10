import logging
import os
import warnings
from datetime import datetime

import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from VQ_train_utils import instantiate_from_config

if __name__ == "__main__":
	# fuck warnings, me and my homies hate on warnings
	warnings.filterwarnings("ignore")

	# # wandb
	wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')
	wandb_logger = WandbLogger(project="[NNTI]VQ-GAN1D")

	# load configs
	logging.debug('loading configs')
	configs = [OmegaConf.load('./configs/1d-VQ_GAN.yaml')]
	config = OmegaConf.merge(*configs)

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
	ckptdir = os.path.join(logdir, "checkpoints")
	cfgdir = os.path.join(logdir, "configs")

	# callbacks
	callbacks = [
		LearningRateMonitor(logging_interval='step'),
		ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True)
	]

	# trainer
	accumulate_grad_batches = 12
	batch_size = config.data.params.batch_size
	model.learning_rate = accumulate_grad_batches * batch_size * config.model.base_learning_rate

	trainer = Trainer(
		logger=wandb_logger,
		enable_checkpointing=True,
		callbacks=callbacks,
		accumulate_grad_batches=accumulate_grad_batches
	)

	trainer.fit(model, data)
