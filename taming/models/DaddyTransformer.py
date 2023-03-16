"""
Pipeline:
	1. Input
	2. encode with VQ-GAN
	3. predict with a single forward pass
"""
from typing import Any

import pytorch_lightning as pl
import torch
import wandb

from VQ_train_utils import instantiate_from_config
from taming.models.vqgan_1D import VQModel1D
import torch.nn.functional as F


class DaddyTransformer(pl.LightningModule):
	def __init__(self,
	             transformer_config,
	             first_stage_config,
	             first_stage_key="image",
	             response_key='label',
	             sos_token=0,
	             *args: Any,
	             **kwargs: Any):
		super().__init__(*args, **kwargs)

		self.be_unconditional = True
		self.sos_token = sos_token
		self.first_stage_key = first_stage_key
		self.response_key = response_key

		self.init_first_stage_from_ckpt(first_stage_config)

		self.transformer = instantiate_from_config(config=transformer_config)

	def init_first_stage_from_ckpt(self, config):
		model = instantiate_from_config(config)
		self.first_stage_model: VQModel1D = model

	def forward(self, x):
		quant_z, z_indices = self.encode_to_z(x)

		print('quant_z:', quant_z.shape, type(quant_z))
		print('z_indices:', z_indices.shape, type(z_indices))
		print(z_indices)

		# make the prediction
		logits, _ = self.transformer(z_indices)

		return logits

	def validation_step(self, batch, batch_idx):
		# #### VQ-VAE
		# x = self.first_stage_model.get_input(batch, self.image_key)
		# xrec, qloss = self(x)
		#
		# aeloss, log_dict_ae = self.first_stage_model.loss(qloss, x, xrec, 0, self.first_stage_model.global_step,
		#                                                   last_layer=self.first_stage_model.get_last_layer(),
		#                                                   split="val")
		#
		# discloss, log_dict_disc = self.first_stage_model.loss(qloss, x, xrec, 1, self.first_stage_model.global_step,
		#                                                       last_layer=self.first_stage_model.get_last_layer(),
		#                                                       split="val")
		#
		# self.log_dict(log_dict_ae | log_dict_disc)
		# wandb.log(log_dict_ae | log_dict_disc)

		#### transformer
		x = self.first_stage_model.get_input(batch, self.first_stage_key)
		y = self.first_stage_model.get_input(batch, self.response_key)

		with torch.no_grad():
			logits= self(x)
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
			# loss = self.transformer.shared_step(batch, batch_idx)
			self.log("val/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			return loss

		# loss = self.transformer.shared_step(batch, batch_idx)
		# self.log("val/TransLoss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)

		# return loss

	def training_step(self, batch, batch_idx):
		x = self.first_stage_model.get_input(batch, self.first_stage_key)
		y = self.first_stage_model.get_input(batch, self.response_key)
		# xrec, qloss = self.first_stage_model(x)

		# if optimizer_idx == 0:
		# 	# autoencode
		# 	aeloss, log_dict_ae = self.first_stage_model.loss(qloss, x, xrec, optimizer_idx,
		# 	                                                  self.first_stage_model.global_step,
		# 	                                                  last_layer=self.first_stage_model.get_last_layer(),
		# 	                                                  split="train")
		#
		# 	# wandb.log({"train/loss": loss})
		#
		# 	self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		# 	self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		# 	wandb.log(log_dict_ae)
		# 	return aeloss
		#
		# if optimizer_idx == 1:
		# 	# discriminator
		# 	discloss, log_dict_disc = self.first_stage_model.loss(qloss, x, xrec, optimizer_idx,
		# 	                                                      self.first_stage_model.global_step,
		# 	                                                      last_layer=self.first_stage_model.get_last_layer(),
		# 	                                                      split="train")
		# 	self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
		# 	self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		# 	wandb.log(log_dict_disc)
		# 	return discloss
		#
		# if optimizer_idx == 2:

		# self.transformer.forward()
		logits = self(x)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
		# loss = self.transformer.shared_step(batch, batch_idx)
		self.log("train/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		return loss

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
		# return optimizer

		# only change the transformer weights
		return [optimizer], []
		# return [opt_ae, opt_disc, optimizer], []

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
