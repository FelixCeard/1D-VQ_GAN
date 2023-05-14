from typing import Any

import torch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from main import instantiate_from_config
from taming.modules.vqvae.quantize import EMAVectorQuantizer, GumbelQuantize, VectorQuantizer2 as VectorQuantizer
import wandb
from torchmetrics import Accuracy, F1Score


class LSTM(pl.LightningModule):
    def __init__(self,
                 transformer_and_co_config,
                 input_dim,
                 hidden_dim,
                 layer_dim,
                 output_dim,
                 first_stage_key="spec",
                 response_key='label',
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        # transformer
        self.transformer_and_co = instantiate_from_config(transformer_and_co_config)
        self.transformer_and_co.freeze()

        self.first_stage_key = first_stage_key
        self.response_key = response_key

        self.automatic_optimization = False

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x, *args, **kwargs) -> Any:
        logits = self.transformer_and_co(torch.squeeze(x), for_tsne=True)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, logits.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, logits.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(logits, (h0.detach().to(self.device), c0.detach().to(self.device)))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

    def training_step(self, batch, batch_idx):
        x = self.transformer_and_co.first_stage_model.get_input(batch, self.first_stage_key)
        y = self.transformer_and_co.first_stage_model.get_input(batch, self.response_key)

        # print('X:', x.shape)
        # logits = self.transformer_and_co(x, for_tsne=True)
        # print('logits1:', logits.shape)

        logits = self.forward(x)
        # print('logits2:', logits.shape)

        accuracy = Accuracy(task='multiclass', num_classes=10).to(self.device)
        F1 = F1Score(task='multilabel').to(self.device)
        loss = F.cross_entropy(logits.reshape(1, -1), y.long())
        acc = accuracy(logits.reshape(1, -1), y.long())
        f1 = F1(logits.reshape(1, -1), y.long())

        self.log("train/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/F1", f1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train/Accuracy', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log('train/skips', self.skips, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # backwards
        self.manual_backward(loss)

        # accumulate gradients of N batches
        if (batch_idx + 1) % 8 == 0:
            opt3 = self.optimizers()

            # clip gradients
            self.clip_gradients(opt3, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            opt3.step()
            opt3.zero_grad()

    def validation_step(self, batch, batch_idx):
        x = self.transformer_and_co.first_stage_model.get_input(batch, self.first_stage_key)
        y = self.transformer_and_co.first_stage_model.get_input(batch, self.response_key)

        # print('X:', x.shape)
        # logits = self.transformer_and_co(x, for_tsne=True)
        # print('logits1:', logits.shape)
        with torch.no_grad():
            logits = self.forward(x)
        # print('logits2:', logits.shape)

        accuracy = Accuracy(task='multiclass', num_classes=10).to(self.device)
        F1 = F1Score(task='multilabel').to(self.device)
        loss = F.cross_entropy(logits.reshape(1, -1), y.long())
        acc = accuracy(logits.reshape(1, -1), y.long())
        f1 = F1(logits.reshape(1, -1), y.long())

        self.log("test/Transloss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("test/F1", f1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('test/Accuracy', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log('train/skips', self.skips, prog_bar=False, logger=True, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        #### VQ-VAE
        # lr = self.learning_rate
        opt = torch.optim.Adam(list(self.parameters()), lr=1e-3)

        return [opt], []