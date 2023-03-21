"""
Pipeline:
	1. Input
	2. encode with VQ-GAN
	3. predict with a single forward pass
"""
from typing import Any

import pytorch_lightning as pl
import torch

from VQ_train_utils import instantiate_from_config
from taming.models.vqgan_1D import VQModel1D
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class DaddyTransformer(pl.LightningModule):
	def __init__(self,
	             transformer_config,
	             first_stage_config,
	             first_stage_key="image",
	             response_key='label',
	             ckpt_path=None,
	             sos_token=0,
	             freeze_vq_vae:bool=True,
	             *args: Any,
	             **kwargs: Any):
		super().__init__(*args, **kwargs)

		self.be_unconditional = True
		self.sos_token = sos_token
		self.first_stage_key = first_stage_key
		self.response_key = response_key

		# big brain optimization is NOW
		self.automatic_optimization = False

		self.init_first_stage_from_ckpt(first_stage_config)

		# freeze the VQ-VAE for faster training
		if freeze_vq_vae:
			self.first_stage_model.freeze()

		self.freeze_vq_vae = not freeze_vq_vae
		# self.freeze_transformer = True
		self.switch_after_n_epochs = 25

		self.transformer = instantiate_from_config(config=transformer_config)

		if ckpt_path is not None:
			self.init_from_ckpt(ckpt_path, ignore_keys=['transformer.head.weight'])


	def init_from_ckpt(self, path, ignore_keys=list()):
		sd = torch.load(path, map_location="cpu")["state_dict"]
		keys = list(sd.keys())
		for k in keys:
			for ik in ignore_keys:
				if k.startswith(ik):
					print("Deleting key {} from state_dict.".format(k))
					del sd[k]
		# try:
		self.load_state_dict(sd, strict=False)
		# except RuntimeError:
		# 	print('Upsi, you did sth. wrong UwU')
		print(f"Restored from {path}")

	def init_first_stage_from_ckpt(self, config):
		model = instantiate_from_config(config)
		self.first_stage_model: VQModel1D = model

	def forward(self, x):
		quant_z, z_indices = self.encode_to_z(x)

		# print(quant_z.shape, z_indices.shape)


		# make the prediction
		logits, _ = self.transformer(z_indices.long())

		return logits

	def validation_step(self, batch, batch_idx):
		# #### VQ-VAE
		x = self.first_stage_model.get_input(batch, self.first_stage_key)
		xrec, qloss = self.first_stage_model(x)

		if self.freeze_vq_vae:
			aeloss, log_dict_ae = self.first_stage_model.loss(qloss, x, xrec, 0, self.first_stage_model.global_step,
			                                                  last_layer=self.first_stage_model.get_last_layer(),
			                                                  split="val")

			discloss, log_dict_disc = self.first_stage_model.loss(qloss, x, xrec, 1, self.first_stage_model.global_step,
			                                                      last_layer=self.first_stage_model.get_last_layer(),
			                                                      split="val")

			self.log_dict(log_dict_ae | log_dict_disc)

		#### transformer
		x = self.first_stage_model.get_input(batch, self.first_stage_key)
		y = self.first_stage_model.get_input(batch, self.response_key)

		with torch.no_grad():
			logits = self(x)#[:, -1, :]
			# print(logits.shape, x.shape, y.shape)
			loss = F.cross_entropy(logits.reshape(1, -1), y.long())
			# loss = self.transformer.shared_step(batch, batch_idx)
			accuracy = Accuracy(task='multiclass', num_classes=10)
			acc = accuracy(logits.reshape(1, -1).detach().cpu(), y.long().cpu())
			self.log('val/Accuracy', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
			self.log("val/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			F1 = F1Score(task='multilabel')
			f1 = F1(logits.reshape(1, -1).detach().cpu(), y.long().cpu())
			self.log("test/F1", f1, prog_bar=True, logger=True, on_step=True, on_epoch=True)


	def training_step(self, batch, batch_idx):
		x = self.first_stage_model.get_input(batch, self.first_stage_key)
		y = self.first_stage_model.get_input(batch, self.response_key)

		if self.freeze_vq_vae:
			# no freeze
			xrec, qloss = self.first_stage_model(x)

			# autoencode
			aeloss, log_dict_ae = self.first_stage_model.loss(qloss, x, xrec, 0,
			                                                  self.first_stage_model.global_step,
			                                                  last_layer=self.first_stage_model.get_last_layer(),
			                                                  split="train")

			# discriminator
			discloss, log_dict_disc = self.first_stage_model.loss(qloss, x, xrec, 1,
			                                                      self.first_stage_model.global_step,
			                                                      last_layer=self.first_stage_model.get_last_layer(),
			                                                      split="train")

			self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
			self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)


		logits = self(x)
		loss = F.cross_entropy(logits.reshape(1, -1), y.long())

		accuracy = Accuracy(task='multiclass', num_classes=10)
		F1 = F1Score(task='multilabel')

		acc = accuracy(logits.reshape(1, -1).detach().cpu(), y.long().cpu())
		f1 = F1(logits.reshape(1, -1).detach().cpu(), y.long().cpu())

		self.log("train/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		self.log("train/F1", f1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		self.log('train/Accuracy', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)

		# backwards
		if self.current_epoch < self.switch_after_n_epochs and self.freeze_vq_vae:
			# backwards
			self.manual_backward(aeloss)
			self.manual_backward(discloss)
		else:
			self.manual_backward(loss)

		# schedulers
		schs = self.lr_schedulers()
		for sch in schs:
			sch.step()

		# accumulate gradients of N batches
		if (batch_idx + 1) % 16 == 0:
			opt1, opt2, opt3 = self.optimizers()

			# clip gradients
			if self.current_epoch < self.switch_after_n_epochs and self.freeze_vq_vae:
				self.clip_gradients(opt1, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
				self.clip_gradients(opt2, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

				opt1.step()
				opt2.step()

				opt1.zero_grad()
				opt2.zero_grad()
			else:
				self.clip_gradients(opt3, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

				opt3.step()

				opt3.zero_grad()


	def configure_optimizers(self):
		#### VQ-VAE
		lr = self.learning_rate
		opt_ae = torch.optim.Adam(list(self.first_stage_model.encoder.parameters()) +
		                          list(self.first_stage_model.decoder.parameters()) +
		                          list(self.first_stage_model.quantize.parameters()) +
		                          list(self.first_stage_model.quant_conv.parameters()) +
		                          list(self.first_stage_model.post_quant_conv.parameters()),
		                          lr=lr, betas=(0.5, 0.9))
		opt_disc = torch.optim.Adam(self.first_stage_model.loss.discriminator.parameters(),
		                            lr=lr, betas=(0.5, 0.9))

		### Transformer
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear,)
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
		for mn, m in self.transformer.named_modules():
			for pn, p in m.named_parameters():
				fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

				if pn.endswith('bias'):
					# all biases will not be decayed
					no_decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					# weights of whitelist modules will be weight decayed
					decay.add(fpn)
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					# weights of blacklist modules will NOT be weight decayed
					no_decay.add(fpn)

		# special case the position embedding parameter in the root GPT module as not decayed
		no_decay.add('pos_emb')

		# validate that we considered every parameter
		param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
		inter_params = decay & no_decay
		union_params = decay | no_decay
		assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
		assert len(
			param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
		                                            % (str(param_dict.keys() - union_params),)

		# create the pytorch optimizer object
		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))


		return [opt_ae, opt_disc, optimizer], [
			# transformer
			torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4_000 * 100, 4_000 * 120], gamma=0.1), # reduce after a few epochs
			torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=25_000), # low to high during the first epochs

			# generator
			torch.optim.lr_scheduler.MultiStepLR(opt_ae, milestones=[4_000 * 30, 4_000 * 50], gamma=0.1), # reduce after a few epochs
			torch.optim.lr_scheduler.LinearLR(opt_ae, start_factor=0.1, total_iters=25_000), # low to high during the first epochs

			# discriminator
			torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[4_000 * 30, 4_000 * 50], gamma=0.1), # reduce after a few epochs
			torch.optim.lr_scheduler.LinearLR(opt_disc, start_factor=0.1, total_iters=25_000), # low to high during the first epochs

		]

	def top_k_logits(self, logits, k):
		v, ix = torch.topk(logits, k)
		out = logits.clone()
		out[out < v[..., [-1]]] = -float('Inf')
		return out

	@torch.no_grad()
	def encode_to_z(self, x):
		quant_z, _, info = self.first_stage_model.encode(x)
		indices = info[2].view(quant_z.shape[0], -1)
		# indices = self.permuter(indices)
		return quant_z, indices
