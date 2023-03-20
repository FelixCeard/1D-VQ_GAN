from transformers import DistilBertConfig, AutoConfig, DistilBertForSequenceClassification
import pytorch_lightning as pl
from typing import Any


class CustomTransformer(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        config = DistilBertConfig()

        model = DistilBertForSequenceClassification()