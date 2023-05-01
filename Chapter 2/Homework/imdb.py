# contains code from:
# Composer - Apache-2.0 license - Copyright 2022 MosaicML Composer authors

from __future__ import annotations
from typing import Optional

import typer
try:
    from rich import print
except ImportError:
    pass

import warnings
import pickle
import time
import yaml
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import transformers
import datasets
from multiprocessing import cpu_count

from torchmetrics.classification import MulticlassAccuracy
from composer.algorithms import FusedLayerNorm
from composer.callbacks import EarlyStopper, LRMonitor
from composer.loggers import WandBLogger
from composer.models.huggingface import HuggingFaceModel
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer import Trainer, Evaluator

from timm.optim.optim_factory import param_groups_weight_decay

from fastxtend.utils import less_random

try:
    from adan import Adan
    ADAN = True
except ImportError:
    ADAN = False

try:
    import apex
    APEX = True
except ImportError:
    APEX = False

from transformers.utils import logging as hf_logging

warnings.simplefilter('ignore')
hf_logging.set_verbosity_error()

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

class OptimizerChoice(str, Enum):
    adam   = 'adam'
    adan   = 'adan'

# from maxb2: https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
def conf_callback(ctx: typer.Context, param: typer.CallbackParam, config: Optional[str] = None):
    if config is not None:
        typer.echo(f"Loading config file: {config}\n")
        try:
            with open(config, 'r') as f:    # Load config file
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}  # Initialize the default map
            ctx.default_map.update(conf)   # Merge the config dict into default_map
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return config

def dataset(model='bert-base-uncased', max_length=256):
    # Create a BERT sequence classification model using Hugging Face transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    # Create tokenizer
    def tokenize_function(sample):
        return tokenizer(
            text=sample['text'],
            padding='max_length',
            max_length=max_length,
            truncation=True
        )

    # Tokenize IMDB
    imdb_dataset = datasets.load_dataset('imdb')
    tokenized_imbd_dataset = imdb_dataset.map(tokenize_function,
                                              batched=True,
                                              num_proc=cpu_count(),
                                              batch_size=100)

    # Split dataset into train and validation sets
    train_dataset = tokenized_imbd_dataset['train']
    eval_dataset = tokenized_imbd_dataset['test']

    return tokenizer, train_dataset, eval_dataset


def create_model(model='bert-base-uncased', bs=40, lr=1e-4, wd=1e-3, epochs=4,
                 opt='adam', compiler=False, fused_ln=False, subset=750,
                 max_length=256, logger=None, progress_bar=False, seed=42,
                 early_stopping=True):
    tokenizer, train_dataset, eval_dataset = dataset(model, max_length)

    data_collator = transformers.data.data_collator.default_data_collator
    with less_random(seed):
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True,
                                      collate_fn=data_collator, num_workers=cpu_count())

    eval_dataloader = DataLoader(eval_dataset, batch_size=bs*2, shuffle=False, drop_last=False,
                                 collate_fn=data_collator, num_workers=cpu_count())

    with less_random(seed):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)

    # Package as a trainer-friendly Composer model
    metrics = [MulticlassAccuracy(num_classes=2, average='micro')]
    composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

    # Setup optimizer and scheduler
    if wd != 0:
        params = param_groups_weight_decay(composer_model, weight_decay=wd)
    else:
        params = composer_model.parameters()

    if opt=='adam' and APEX:
        optimizer = apex.optimizers.FusedAdam(
            params=params,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-6
        )
    elif opt=='adam':
        optimizer = AdamW(
            params=params,
            lr=lr,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=0,
            foreach=True
        )
    elif opt=='adan':
        optimizer = Adan(
            params=params,
            lr=lr,
            eps=1e-6,
            fused=True
        )

    cos_sched = CosineAnnealingWithWarmupScheduler('0.25dur', '1dur')

    early_stopper = EarlyStopper('MulticlassAccuracy', 'val', patience=1)
    evaluator = Evaluator(
        dataloader = eval_dataloader,
        label = 'val',
        metric_names = ['MulticlassAccuracy']
    )

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=evaluator,
        max_duration=f'{epochs}ep',
        optimizers=optimizer,
        schedulers=[cos_sched],
        device='gpu' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        eval_subset_num_batches=subset,
        compile_config = {} if compiler else None,
        algorithms=[FusedLayerNorm()] if fused_ln else [],
        callbacks=[early_stopper, LRMonitor()] if early_stopping else [LRMonitor()],
        loggers=[logger] if logger is not None else [],
        progress_bar=progress_bar,
    )

    return trainer

@app.command()
def train(ctx:typer.Context, # Typer Context to grab config for --verbose and passing to WandB
    # Config file
    config:Optional[Path]=typer.Option(None, callback=conf_callback, is_eager=True),
    model:str=typer.Option('bert-base-uncased'),
    batch_size:int=typer.Option(64),
    learning_rate:float=typer.Option(1e-4),
    weight_decay:float=typer.Option(1e-2),
    epochs:int=typer.Option(4),
    opt:str=typer.Option('adam'),
    compiler:bool=typer.Option(True),
    fused_ln:bool=typer.Option(False),
    train_subset:int=typer.Option(100),
    eval_subset:int=typer.Option(100),
    max_length:int=typer.Option(256),
    progress_bar:bool=typer.Option(True),
    seed:int=typer.Option(42),
    early_stopping:bool=typer.Option(True),
    # Weights and Biases
    log_wandb:bool=typer.Option(False, "--wandb"),
    name:Optional[str]=typer.Option(None),
    project:str=typer.Option('nlp_study_group_imdb'),
    group:Optional[str]=typer.Option(None),
    tags:Optional[str]=typer.Option(None),
    entity:Optional[str]=typer.Option(None),

):
    ignore_params = ['config', 'verbose', 'log_wandb', 'name', 'project',
                     'group', 'tags', 'entity', 'save_code']
    config = {k:v for k,v in ctx.params.items() if k not in ignore_params}

    if log_wandb:
        logger = WandBLogger(project, f'{model}_sweep' if group is None else group,
                             name, entity, tags, init_kwargs={'config':config})
    else:
        logger = None

    train_subset = (train_subset * 64 ) // batch_size
    eval_subset = (eval_subset * 64) // (batch_size * 2)

    trainer = create_model(model, batch_size, learning_rate, weight_decay, epochs,
                           opt, compiler, fused_ln, eval_subset, max_length,
                           logger, progress_bar, seed, early_stopping)

    trainer.fit(
        precision='amp_fp16',
        duration=f'{epochs}ep',
        train_subset_num_batches=train_subset,
    )

if __name__=="__main__":
    app()