import torch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import instantiate_from_config
from taming.modules.vqvae.quantize import EMAVectorQuantizer, GumbelQuantize, VectorQuantizer2 as VectorQuantizer
import wandb

def Normalize(in_channels):
	return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
	# swish
	return x * torch.sigmoid(x)


class ResnetBlock1D(nn.Module):
	def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
	             dropout, temb_channels=512):
		super().__init__()
		self.in_channels = in_channels
		out_channels = in_channels if out_channels is None else out_channels
		self.out_channels = out_channels
		self.use_conv_shortcut = conv_shortcut

		self.norm1 = Normalize(in_channels)
		self.conv1 = torch.nn.Conv1d(in_channels,
		                             out_channels,
		                             kernel_size=3,
		                             stride=1,
		                             padding=1)
		if temb_channels > 0:
			self.temb_proj = torch.nn.Linear(temb_channels,
			                                 out_channels)
		self.norm2 = Normalize(out_channels)
		self.dropout = torch.nn.Dropout(dropout)
		self.conv2 = torch.nn.Conv1d(out_channels,
		                             out_channels,
		                             kernel_size=3,
		                             stride=1,
		                             padding=1)
		if self.in_channels != self.out_channels:
			if self.use_conv_shortcut:
				self.conv_shortcut = torch.nn.Conv1d(in_channels,
				                                     out_channels,
				                                     kernel_size=3,
				                                     stride=1,
				                                     padding=1)
			else:
				self.nin_shortcut = torch.nn.Conv1d(in_channels,
				                                    out_channels,
				                                    kernel_size=1,
				                                    stride=1,
				                                    padding=0)

	def forward(self, x, temb):
		h = x
		h = self.norm1(h)
		h = nonlinearity(h)
		h = self.conv1(h)

		if temb is not None:
			h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

		h = self.norm2(h)
		h = nonlinearity(h)
		h = self.dropout(h)
		h = self.conv2(h)

		if self.in_channels != self.out_channels:
			if self.use_conv_shortcut:
				x = self.conv_shortcut(x)
			else:
				x = self.nin_shortcut(x)

		return x + h


class AttnBlock1D(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.in_channels = in_channels

		self.norm = Normalize(in_channels)
		self.q = torch.nn.Conv1d(in_channels,
		                         in_channels,
		                         kernel_size=1,
		                         stride=1,
		                         padding=0)
		self.k = torch.nn.Conv1d(in_channels,
		                         in_channels,
		                         kernel_size=1,
		                         stride=1,
		                         padding=0)
		self.v = torch.nn.Conv1d(in_channels,
		                         in_channels,
		                         kernel_size=1,
		                         stride=1,
		                         padding=0)
		self.proj_out = torch.nn.Conv1d(in_channels,
		                                in_channels,
		                                kernel_size=1,
		                                stride=1,
		                                padding=0)

	def forward(self, x):
		h_ = x
		h_ = self.norm(h_)
		q = self.q(h_)
		k = self.k(h_)
		v = self.v(h_)

		# compute attention
		b, c, s = q.shape
		q = q.reshape(b, c, s)
		q = q.permute(0, 2, 1)  # b,hw,c
		k = k.reshape(b, c, s)  # b,c,hw
		w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
		w_ = w_ * (int(c) ** (-0.5))
		w_ = torch.nn.functional.softmax(w_, dim=2)

		# attend to values
		v = v.reshape(b, c, s)
		w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
		h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
		h_ = h_.reshape(b, c, s)

		h_ = self.proj_out(h_)

		return x + h_


class Downsample1D(nn.Module):
	def __init__(self, in_channels, with_conv):
		super().__init__()
		self.with_conv = with_conv
		if self.with_conv:
			# no asymmetric padding in torch conv, must do it ourselves
			self.conv = torch.nn.Conv1d(in_channels,
			                            in_channels,
			                            kernel_size=3,
			                            stride=2,
			                            padding=0)

	def forward(self, x):
		if self.with_conv:
			pad = (1, 1)
			x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
			x = self.conv(x)
		else:
			x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
		return x


class Encoder1D(nn.Module):
	"""
	#! bug @ encoder: if input is not divisible by 16 => error while decoding
	"""
	def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
	             attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
	             resolution, z_channels, double_z=True, **ignore_kwargs):
		super().__init__()
		self.ch = ch
		self.temb_ch = 0
		self.num_resolutions = len(ch_mult)
		self.num_res_blocks = num_res_blocks
		self.resolution = resolution
		self.in_channels = in_channels

		# downsampling
		self.conv_in = torch.nn.Conv1d(in_channels,
		                               self.ch,
		                               kernel_size=3,
		                               stride=1,
		                               padding=1)

		curr_res = resolution
		in_ch_mult = (1,) + tuple(ch_mult)
		self.down = nn.ModuleList()
		for i_level in range(self.num_resolutions):
			block = nn.ModuleList()
			attn = nn.ModuleList()
			block_in = ch * in_ch_mult[i_level]
			block_out = ch * ch_mult[i_level]
			for i_block in range(self.num_res_blocks):
				block.append(ResnetBlock1D(in_channels=block_in,
				                           out_channels=block_out,
				                           temb_channels=self.temb_ch,
				                           dropout=dropout))
				block_in = block_out
				if curr_res in attn_resolutions:
					attn.append(AttnBlock1D(block_in))
			down = nn.Module()
			down.block = block
			down.attn = attn
			if i_level != self.num_resolutions - 1:
				down.downsample = Downsample1D(block_in, resamp_with_conv)
				curr_res = curr_res // 2
			self.down.append(down)

		# middle
		self.mid = nn.Module()
		self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
		                                 out_channels=block_in,
		                                 temb_channels=self.temb_ch,
		                                 dropout=dropout)
		self.mid.attn_1 = AttnBlock1D(block_in)
		self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
		                                 out_channels=block_in,
		                                 temb_channels=self.temb_ch,
		                                 dropout=dropout)

		# end
		self.norm_out = Normalize(block_in)
		self.conv_out = torch.nn.Conv1d(block_in,
		                                2 * z_channels if double_z else z_channels,
		                                kernel_size=3,
		                                stride=1,
		                                padding=1)

	def forward(self, x):
		# assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

		# timestep embedding
		temb = None
		# print('entering Encoder')
		# print(x.shape)

		# downsampling
		hs = [self.conv_in(x)]
		for i_level in range(self.num_resolutions):
			# print(hs[-1].shape)
			for i_block in range(self.num_res_blocks):
				h = self.down[i_level].block[i_block](hs[-1], temb)
				if len(self.down[i_level].attn) > 0:
					h = self.down[i_level].attn[i_block](h)
				hs.append(h)
			if i_level != self.num_resolutions - 1:
				hs.append(self.down[i_level].downsample(hs[-1]))

		# middle
		h = hs[-1]
		h = self.mid.block_1(h, temb)
		h = self.mid.attn_1(h)
		h = self.mid.block_2(h, temb)

		# end
		h = self.norm_out(h)
		h = nonlinearity(h)
		h = self.conv_out(h)
		return h


class Upsample1D(nn.Module):
	def __init__(self, in_channels, with_conv):
		super().__init__()
		self.with_conv = with_conv
		if self.with_conv:
			self.conv = torch.nn.Conv1d(in_channels,
			                            in_channels,
			                            kernel_size=3,
			                            stride=1,
			                            padding=1)

	def forward(self, x):
		x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
		if self.with_conv:
			x = self.conv(x)
		return x


class Decoder1D(nn.Module):
	def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
	             attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
	             resolution, z_channels, give_pre_end=False, **ignorekwargs):
		super().__init__()
		self.ch = ch
		self.temb_ch = 0
		self.num_resolutions = len(ch_mult)
		self.num_res_blocks = num_res_blocks
		self.resolution = resolution
		self.in_channels = in_channels
		self.give_pre_end = give_pre_end

		# compute in_ch_mult, block_in and curr_res at lowest res
		in_ch_mult = (1,) + tuple(ch_mult)
		block_in = ch * ch_mult[self.num_resolutions - 1]
		curr_res = resolution // 2 ** (self.num_resolutions - 1)
		self.z_shape = (1, z_channels, curr_res, curr_res)

		# z to block_in
		self.conv_in = torch.nn.Conv1d(z_channels,
		                               block_in,
		                               kernel_size=3,
		                               stride=1,
		                               padding=1)

		# middle
		self.mid = nn.Module()
		self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
		                                 out_channels=block_in,
		                                 temb_channels=self.temb_ch,
		                                 dropout=dropout)
		self.mid.attn_1 = AttnBlock1D(block_in)
		self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
		                                 out_channels=block_in,
		                                 temb_channels=self.temb_ch,
		                                 dropout=dropout)

		# upsampling
		self.up = nn.ModuleList()
		for i_level in reversed(range(self.num_resolutions)):
			block = nn.ModuleList()
			attn = nn.ModuleList()
			block_out = ch * ch_mult[i_level]
			for i_block in range(self.num_res_blocks + 1):
				block.append(ResnetBlock1D(in_channels=block_in,
				                           out_channels=block_out,
				                           temb_channels=self.temb_ch,
				                           dropout=dropout))
				block_in = block_out
				if curr_res in attn_resolutions:
					attn.append(AttnBlock1D(block_in))
			up = nn.Module()
			up.block = block
			up.attn = attn
			if i_level != 0:
				up.upsample = Upsample1D(block_in, resamp_with_conv)
				curr_res = curr_res * 2
			self.up.insert(0, up)  # prepend to get consistent order

		# end
		self.norm_out = Normalize(block_in)
		self.conv_out = torch.nn.Conv1d(block_in,
		                                out_ch,
		                                kernel_size=3,
		                                stride=1,
		                                padding=1)

	def forward(self, z):
		# assert z.shape[1:] == self.z_shape[1:]
		self.last_z_shape = z.shape

		# timestep embedding
		temb = None

		# z to block_in
		h = self.conv_in(z)

		# middle
		h = self.mid.block_1(h, temb)
		h = self.mid.attn_1(h)
		h = self.mid.block_2(h, temb)

		# upsampling
		for i_level in reversed(range(self.num_resolutions)):
			for i_block in range(self.num_res_blocks + 1):
				h = self.up[i_level].block[i_block](h, temb)
				if len(self.up[i_level].attn) > 0:
					h = self.up[i_level].attn[i_block](h)
			if i_level != 0:
				h = self.up[i_level].upsample(h)

		# end
		if self.give_pre_end:
			return h

		h = self.norm_out(h)
		h = nonlinearity(h)
		h = self.conv_out(h)
		return h


class VQModel1D(pl.LightningModule):
	def __init__(self,
	             ddconfig,
	             lossconfig,
	             n_embed,
	             embed_dim,
	             ckpt_path=None,
	             ignore_keys=[],
	             image_key="image",
	             colorize_nlabels=None,
	             monitor=None,
	             remap=None,
	             sane_index_shape=False,  # tell vector quantizer to return indices as bhw
	             ):
		super().__init__()
		self.image_key = image_key
		self.encoder = Encoder1D(**ddconfig)
		self.decoder = Decoder1D(**ddconfig)
		self.loss = instantiate_from_config(lossconfig)
		self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
		                                remap=remap, sane_index_shape=sane_index_shape)
		self.quant_conv = torch.nn.Conv1d(ddconfig["z_channels"], embed_dim, 1)
		self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)
		if ckpt_path is not None:
			self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
		self.image_key = image_key
		if colorize_nlabels is not None:
			assert type(colorize_nlabels) == int
			self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
		if monitor is not None:
			self.monitor = monitor

		self.best_loss = float('inf')

	def init_from_ckpt(self, path, ignore_keys=list()):
		sd = torch.load(path, map_location="cpu")["state_dict"]
		keys = list(sd.keys())
		for k in keys:
			for ik in ignore_keys:
				if k.startswith(ik):
					print("Deleting key {} from state_dict.".format(k))
					del sd[k]
		self.load_state_dict(sd, strict=False)
		print(f"Restored from {path}")

	def encode(self, x):
		x = torch.unsqueeze(x, 0)
		h = self.encoder(x)
		h = self.quant_conv(h)
		quant, emb_loss, info = self.quantize(h)
		return quant, emb_loss, info

	def decode(self, quant):
		quant = self.post_quant_conv(quant)
		dec = self.decoder(quant)
		return dec

	def decode_code(self, code_b):
		quant_b = self.quantize.embed_code(code_b)
		dec = self.decode(quant_b)
		return dec

	def forward(self, input):
		quant, diff, _ = self.encode(input)
		dec = self.decode(quant)
		return dec, diff

	def get_input(self, batch, k):
		x = batch[k]
		if len(x.shape) == 3:
			x = x[..., None]
		# x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
		return x.float()

	def training_step(self, batch, batch_idx, optimizer_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)

		if optimizer_idx == 0:
			# autoencode
			aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
			                                last_layer=self.get_last_layer(), split="train")

			# wandb.log({"train/loss": loss})

			self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
			self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			wandb.log(log_dict_ae)
			return aeloss

		if optimizer_idx == 1:
			# discriminator
			discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
			                                    last_layer=self.get_last_layer(), split="train")
			self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
			self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			wandb.log(log_dict_disc)
			return discloss

	def validation_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)

		aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
		                                last_layer=self.get_last_layer(), split="val")

		discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
		                                    last_layer=self.get_last_layer(), split="val")

		self.log_dict(log_dict_ae | log_dict_disc)
		wandb.log(log_dict_ae | log_dict_disc)

		return self.log_dict

	def configure_optimizers(self):
		lr = self.learning_rate
		opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
		                          list(self.decoder.parameters()) +
		                          list(self.quantize.parameters()) +
		                          list(self.quant_conv.parameters()) +
		                          list(self.post_quant_conv.parameters()),
		                          lr=lr, betas=(0.5, 0.9))
		opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
		                            lr=lr, betas=(0.5, 0.9))

		# reduce_on_plateau_ae = {
		# 	'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, 'min'),
		# 	'name': 'Reduce_on_plateau_ae',
		# 	'monitor': 'train/aeloss',
		# 	'frequency': 1,
		# 	"interval": "epoch",
		# }
		# reduce_on_plateau_disc = {
		# 	'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_disc, 'min'),
		# 	'name': 'Reduce_on_plateau_disc',
		# 	'monitor': 'train/discloss',
		# 	"interval": "epoch",
		# 	'frequency': 1
		# }

		return [opt_ae, opt_disc], []
		# return [opt_ae, opt_disc], [reduce_on_plateau_ae, reduce_on_plateau_disc]
		# return [opt_ae], []  # skip LR scheduler

	def get_last_layer(self):
		return self.decoder.conv_out.weight

	def log_images(self, batch, **kwargs):
		log = dict()
		x = self.get_input(batch, self.image_key)
		x = x.to(self.device)
		xrec, _ = self(x)
		if x.shape[1] > 3:
			# colorize with random projection
			assert xrec.shape[1] > 3
			x = self.to_rgb(x)
			xrec = self.to_rgb(xrec)
		log["inputs"] = x
		log["reconstructions"] = xrec
		return log

	def to_rgb(self, x):
		assert self.image_key == "segmentation"
		if not hasattr(self, "colorize"):
			self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
		x = F.conv1d(x, weight=self.colorize)
		x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
		return x


class VQSegmentationModel1D(VQModel1D):
	def __init__(self, n_labels, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

	def configure_optimizers(self):
		lr = self.learning_rate
		opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
		                          list(self.decoder.parameters()) +
		                          list(self.quantize.parameters()) +
		                          list(self.quant_conv.parameters()) +
		                          list(self.post_quant_conv.parameters()),
		                          lr=lr, betas=(0.5, 0.9))
		return opt_ae

	def training_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)
		aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
		self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		return aeloss

	def validation_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)
		aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
		self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		total_loss = log_dict_ae["val/total_loss"]
		self.log("val/total_loss", total_loss,
		         prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
		return aeloss

	@torch.no_grad()
	def log_images(self, batch, **kwargs):
		log = dict()
		x = self.get_input(batch, self.image_key)
		x = x.to(self.device)
		xrec, _ = self(x)
		if x.shape[1] > 3:
			# colorize with random projection
			assert xrec.shape[1] > 3
			# convert logits to indices
			xrec = torch.argmax(xrec, dim=1, keepdim=True)
			xrec = F.one_hot(xrec, num_classes=x.shape[1])
			xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
			x = self.to_rgb(x)
			xrec = self.to_rgb(xrec)
		log["inputs"] = x
		log["reconstructions"] = xrec
		return log


class VQNoDiscModel1D(VQModel1D):
	def __init__(self,
	             ddconfig,
	             lossconfig,
	             n_embed,
	             embed_dim,
	             ckpt_path=None,
	             ignore_keys=[],
	             image_key="image",
	             colorize_nlabels=None
	             ):
		super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
		                 ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
		                 colorize_nlabels=colorize_nlabels)

	def training_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)
		# autoencode
		aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
		output = pl.TrainResult(minimize=aeloss)
		output.log("train/aeloss", aeloss,
		           prog_bar=True, logger=True, on_step=True, on_epoch=True)
		output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
		return output

	def validation_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)
		aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
		rec_loss = log_dict_ae["val/rec_loss"]
		output = pl.EvalResult(checkpoint_on=rec_loss)
		output.log("val/rec_loss", rec_loss,
		           prog_bar=True, logger=True, on_step=True, on_epoch=True)
		output.log("val/aeloss", aeloss,
		           prog_bar=True, logger=True, on_step=True, on_epoch=True)
		output.log_dict(log_dict_ae)

		return output

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
		                             list(self.decoder.parameters()) +
		                             list(self.quantize.parameters()) +
		                             list(self.quant_conv.parameters()) +
		                             list(self.post_quant_conv.parameters()),
		                             lr=self.learning_rate, betas=(0.5, 0.9))
		return optimizer


class GumbelVQ1D(VQModel1D):
	def __init__(self,
	             ddconfig,
	             lossconfig,
	             n_embed,
	             embed_dim,
	             temperature_scheduler_config,
	             ckpt_path=None,
	             ignore_keys=[],
	             image_key="image",
	             colorize_nlabels=None,
	             monitor=None,
	             kl_weight=1e-8,
	             remap=None,
	             ):

		z_channels = ddconfig["z_channels"]
		super().__init__(ddconfig,
		                 lossconfig,
		                 n_embed,
		                 embed_dim,
		                 ckpt_path=None,
		                 ignore_keys=ignore_keys,
		                 image_key=image_key,
		                 colorize_nlabels=colorize_nlabels,
		                 monitor=monitor,
		                 )

		self.loss.n_classes = n_embed
		self.vocab_size = n_embed

		self.quantize = GumbelQuantize(z_channels, embed_dim,
		                               n_embed=n_embed,
		                               kl_weight=kl_weight, temp_init=1.0,
		                               remap=remap)

		self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # annealing of temp

		if ckpt_path is not None:
			self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

	def temperature_scheduling(self):
		self.quantize.temperature = self.temperature_scheduler(self.global_step)

	def encode_to_prequant(self, x):
		h = self.encoder(x)
		h = self.quant_conv(h)
		return h

	def decode_code(self, code_b):
		raise NotImplementedError

	def training_step(self, batch, batch_idx, optimizer_idx):
		self.temperature_scheduling()
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x)

		if optimizer_idx == 0:
			# autoencode
			aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
			                                last_layer=self.get_last_layer(), split="train")

			self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			return aeloss

		if optimizer_idx == 1:
			# discriminator
			discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
			                                    last_layer=self.get_last_layer(), split="train")
			self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
			return discloss

	def validation_step(self, batch, batch_idx):
		x = self.get_input(batch, self.image_key)
		xrec, qloss = self(x, return_pred_indices=True)
		aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
		                                last_layer=self.get_last_layer(), split="val")

		discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
		                                    last_layer=self.get_last_layer(), split="val")
		rec_loss = log_dict_ae["val/rec_loss"]
		self.log("val/rec_loss", rec_loss,
		         prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
		self.log("val/aeloss", aeloss,
		         prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
		self.log_dict(log_dict_ae)
		self.log_dict(log_dict_disc)
		return self.log_dict

	def log_images(self, batch, **kwargs):
		log = dict()
		x = self.get_input(batch, self.image_key)
		x = x.to(self.device)
		# encode
		h = self.encoder(x)
		h = self.quant_conv(h)
		quant, _, _ = self.quantize(h)
		# decode
		x_rec = self.decode(quant)
		log["inputs"] = x
		log["reconstructions"] = x_rec
		return log


class EMAVQ1D(VQModel1D):
	def __init__(self,
	             ddconfig,
	             lossconfig,
	             n_embed,
	             embed_dim,
	             ckpt_path=None,
	             ignore_keys=[],
	             image_key="image",
	             colorize_nlabels=None,
	             monitor=None,
	             remap=None,
	             sane_index_shape=False,  # tell vector quantizer to return indices as bhw
	             ):
		super().__init__(ddconfig,
		                 lossconfig,
		                 n_embed,
		                 embed_dim,
		                 ckpt_path=None,
		                 ignore_keys=ignore_keys,
		                 image_key=image_key,
		                 colorize_nlabels=colorize_nlabels,
		                 monitor=monitor,
		                 )
		self.quantize = EMAVectorQuantizer(n_embed=n_embed,
		                                   embedding_dim=embed_dim,
		                                   beta=0.25,
		                                   remap=remap)

	def configure_optimizers(self):
		lr = self.learning_rate
		# Remove self.quantize from parameter list since it is updated via EMA
		opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
		                          list(self.decoder.parameters()) +
		                          list(self.quant_conv.parameters()) +
		                          list(self.post_quant_conv.parameters()),
		                          lr=lr, betas=(0.5, 0.9))
		opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
		                            lr=lr, betas=(0.5, 0.9))
		return [opt_ae, opt_disc], []
