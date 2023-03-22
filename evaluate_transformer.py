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
from tqdm import tqdm

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

    # data
    print('loading data')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()


    # make predictions
    true_label_train = []
    predicted_label_train = []

    # train
    for batch in tqdm(data.train_dataloader()._get_iterator()):
        spec = batch['spec']
        with torch.no_grad():
            # print(spec.shape)
            logits = model(torch.squeeze(spec))
        _, label = torch.max(logits, 1)
        true_label_train.append(batch['label'].detach().cpu().tolist()[0])
        predicted_label_train.append(label.detach().cpu().tolist()[0])

    # make predictions
    true_label_test = []
    predicted_label_test = []

    # train
    for batch in tqdm(data.test_dataloader()._get_iterator()):
        spec = batch['spec']
        with torch.no_grad():
            logits = model(torch.squeeze(spec))
        _, label = torch.max(logits, 1)
        true_label_test.append(batch['label'].detach().cpu().tolist()[0])
        predicted_label_test.append(label.detach().cpu().tolist()[0])

    # make predictions
    true_label_validation = []
    predicted_label_validation = []

    # train
    for batch in tqdm(data.val_dataloader()._get_iterator()):
        spec = batch['spec']
        with torch.no_grad():
            # print(spec.shape)
            logits = model(torch.squeeze(spec))
            # quant_z, _, info = model.first_stage_model.encode(torch.squeeze(spec))
            # logits, _ = model.transformer(z_indices.long())

        _, label = torch.max(logits, 1)
        true_label_validation.append(batch['label'].detach().cpu().tolist()[0])
        predicted_label_validation.append(label.detach().cpu().tolist()[0])


    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

    # acc
    train_acc = accuracy_score(torch.tensor(true_label_train), torch.tensor(predicted_label_train))
    test_acc = accuracy_score(torch.tensor(true_label_test), torch.tensor(predicted_label_test))
    val_acc = accuracy_score(torch.tensor(true_label_validation), torch.tensor(predicted_label_validation))

    # f1
    train_f1 = f1_score(torch.tensor(true_label_train), torch.tensor(predicted_label_train), average='micro')
    test_f1 = f1_score(torch.tensor(true_label_test), torch.tensor(predicted_label_test), average='micro')
    val_f1 = f1_score(torch.tensor(true_label_validation), torch.tensor(predicted_label_validation), average='micro')

    # acc
    train_precision = precision_score(torch.tensor(true_label_train), torch.tensor(predicted_label_train), average='micro')
    test_precision = precision_score(torch.tensor(true_label_test), torch.tensor(predicted_label_test), average='micro')
    val_precision = precision_score(torch.tensor(true_label_validation), torch.tensor(predicted_label_validation), average='micro')

    # recall
    train_recall = recall_score(torch.tensor(true_label_train), torch.tensor(predicted_label_train), average='micro')
    test_recall = recall_score(torch.tensor(true_label_test), torch.tensor(predicted_label_test), average='micro')
    val_recall = recall_score(torch.tensor(true_label_validation), torch.tensor(predicted_label_validation), average='micro')

    # confusion matrix
    matrix_train = confusion_matrix(y_true = true_label_train, y_pred = predicted_label_train)
    matrix_test = confusion_matrix(y_true = true_label_test, y_pred = predicted_label_test)
    matrix_validation = confusion_matrix(y_true = true_label_validation, y_pred = predicted_label_validation)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].set_title('train')
    ax[1].set_title('test')
    ax[2].set_title('val')

    ConfusionMatrixDisplay(matrix_train).plot(ax=ax[0])
    ConfusionMatrixDisplay(matrix_test).plot(ax=ax[1])
    ConfusionMatrixDisplay(matrix_validation).plot(ax=ax[2])

    fig.savefig('confusion_matrix.png', dpi=300)

    print('Accuracy')
    print(train_acc)
    print(test_acc)
    print(val_acc)

    print('F1')
    print(train_f1)
    print(test_f1)
    print(val_f1)

    print('PRECISION')
    print(train_precision)
    print(test_precision)
    print(val_precision)

    print('RECALL')
    print(train_recall)
    print(test_recall)
    print(val_recall)