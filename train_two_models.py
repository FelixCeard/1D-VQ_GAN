import logging
import os
import warnings
from datetime import datetime

import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
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
	configs = [OmegaConf.load('./configs/CustomPipeline.yaml')]
	config = OmegaConf.merge(*configs)

	# model
	logging.debug('loading model')
	model = instantiate_from_config(config.model)

	# dataset
	logging.debug('loading dataset')
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


	class AudioLoggingCallback(Callback):
		def __init__(self, sample):
			self.sample = sample

		def on_train_end(self, trainer, pl_module):
			print("Training is ending")

		def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
			model = pl_module.first_stage_model
			rec = model.forward(self.sample.clone())

			model.log_dict({
				'OG': wandb.Audio(self.sample, sample_rate=8_000, caption='Original Audio'),
				'Reconstructed': wandb.Audio(rec, sample_rate=8_000, caption='Original Audio'),
			})


	# callbacks
	callbacks = [
		LearningRateMonitor(logging_interval='step'),
		ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True),
		AudioLoggingCallback(next(data.val_dataloader())['wav'].numpy())
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
