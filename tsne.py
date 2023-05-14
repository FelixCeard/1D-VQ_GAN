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
    # wandb.login(key='e5ef4f3a1142de13823dd7b320a9e133b3f5bdfc')
    # wandb_logger = WandbLogger(project="[NNTI]SpectroformerUsupervised2")

    # load configs
    print('loading configs')
    configs = [OmegaConf.load('configs/[TRAIN]SpectogramTransformerUnsupervised2.yaml')]
    config = OmegaConf.merge(*configs)

    # model
    print('loading model')
    model = instantiate_from_config(config.model)
    model.eval()

    # data
    print('loading data')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # make predictions
    train_embeddings = torch.empty(len(data.train_dataloader()), 128)
    true_label_train = []

    # train
    for i, batch in tqdm(enumerate(data.train_dataloader()._get_iterator()), total=len(data.train_dataloader())):
        spec = batch['spec']
        with torch.no_grad():
            # print(spec.shape)
            logits = model(torch.squeeze(spec), for_tsne=True)[:, 0, :]
            logits = torch.squeeze(logits)
            # print(logits.shape)
        # print(logits.shape)
        train_embeddings[i] = logits#.tolist()
        true_label_train.append(batch['label'].detach().cpu().tolist()[0])

    # # make predictions
    # test_embeddings = torch.zeros(len(data.test_dataloader()), 128)
    # true_label_test = []
    #
    # # train
    # for i, batch in tqdm(enumerate(data.test_dataloader()._get_iterator())):
    #     spec = batch['spec']
    #     with torch.no_grad():
    #         # print(spec.shape)
    #         logits = model(torch.squeeze(spec), for_tsne=True)[:, 0, :]
    #         logits = torch.squeeze(logits)
    #         # print(logits.shape)
    #     test_embeddings[i] = logits
    #     true_label_test.append(batch['label'].detach().cpu().tolist()[0])
    #
    #
    # val_embeddings = torch.zeros(len(data.val_dataloader()), 128)
    # true_label_val = []
    #
    # # train
    # for i, batch in tqdm(enumerate(data.val_dataloader()._get_iterator())):
    #     spec = batch['spec']
    #     with torch.no_grad():
    #         # print(spec.shape)
    #         logits = model(torch.squeeze(spec), for_tsne=True)[:, 0, :]
    #         logits = torch.squeeze(logits)
    #         # print(logits.shape)
    #     val_embeddings[i] = logits
    #     true_label_val.append(batch['label'].detach().cpu().tolist()[0])

    # TSNE
    from sklearn.manifold import TSNE
    import colorcet as cc

    # sns.set_theme(style="whitegrid", palette="pastel")

    print(train_embeddings.shape)
    print('TSNEEE')
    X_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(train_embeddings.to(torch.float))#.detach().cpu().numpy())
    print('TSNEEE!')

    tsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:, 0], 'tsne_2': X_embedded[:, 1], 'label': true_label_train})
    fig, ax = plt.subplots(1)
    palette = sns.color_palette(cc.glasbey, n_colors=10)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=20, palette=palette)
    lim = (X_embedded.min() - 5, X_embedded.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    fig.savefig('tsne_new_new.png', dpi=300)
    plt.show()