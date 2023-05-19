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
import yaml
from pathlib import Path
from enum import Enum
from functools import partial
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

import transformers
import datasets
from multiprocessing import cpu_count

from torchmetrics import Metric

from composer.algorithms import LowPrecisionLayerNorm
from composer.callbacks import EarlyStopper, LRMonitor
from composer.loggers import WandBLogger
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import LanguagePerplexity
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer import Trainer

from timm.optim.optim_factory import param_groups_weight_decay

from fastxtend.utils import less_random

from flashgptneox import *

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


class LossHuggingFaceModel(HuggingFaceModel):
    "Composer Hugging Face Adapter which allows setting a loss function instead of using model's"
    def __init__(self,
        model: transformers.PreTrainedModel,
        tokenizer: Optional[Union[transformers.PreTrainedTokenizer,
                                transformers.PreTrainedTokenizerFast]] = None,
        use_logits: Optional[bool] = False,
        metrics: Optional[List[Metric]] = None,
        eval_metrics: Optional[List[Metric]] = None,
        shift_labels: Optional[bool] = None,
        allow_embedding_resizing: bool = False,
        loss: Optional[nn.Module] = None
    ) -> None:
        super().__init__(model, tokenizer, use_logits, metrics, eval_metrics, shift_labels, allow_embedding_resizing)
        self.loss_fn = loss

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            if self.loss_fn is not None:
                batch = {k: v for k, v in batch.items() if k in self.model_forward_args and k != 'labels'}
                self.output = self.model(**batch)  # type: ignore (thirdparty)
            else:
                batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
                self.output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )

    def loss(self, outputs, batch):
        if self.loss_fn is not None:
            return self.loss_fn(outputs, batch['labels'])
        elif self.config.use_return_dict:
            return outputs['loss']
        else:
            # loss is at index 0 in the output tuple
            return outputs[0]


def dataset(model='EleutherAI/pythia-160m-deduped', max_length=256):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenized_dataset = datasets.load_from_disk(f"data/tales_pythia_{max_length}.hf")

    return tokenizer, tokenized_dataset


def create_model(model='EleutherAI/pythia-160m-deduped', batch_size=40, learning_rate=1e-4,
                 weight_decay=1e-3, epochs=4, warmup='0.25dur', opt='adam', compiler=False,
                 lowprecision_ln=False, max_length=256, logger=None, progress_bar=False, seed=42,
                 grad_accum_bs=None, label_smoothing=0, flash_attn=True, **kwargs):
    tokenizer, train_dataset = dataset(model, max_length)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    with less_random(seed):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                      collate_fn=data_collator, num_workers=cpu_count())

    with less_random(seed):
        if 'pythia' in model or 'neo' in model:
            config = transformers.GPTNeoXConfig.from_pretrained(model)
            if flash_attn:
                config.use_flash_attention = True
            model = transformers.GPTNeoXForCausalLM.from_pretrained(model, config=config)
        else:
            raise ValueError(f"Only GPT-NeoX (and derived) models supported in this file: {model=}")

    # Package as a trainer-friendly Composer model
    metrics = [LanguagePerplexity()]
    if label_smoothing > 0:
        composer_model = LossHuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True,
                                              loss=nn.CrossEntropyLoss(label_smoothing=label_smoothing))
    else:
        composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

    # Setup optimizer and scheduler
    if weight_decay != 0:
        params = param_groups_weight_decay(composer_model, weight_decay=weight_decay)
    else:
        params = composer_model.parameters()

    if opt=='adam' and APEX:
        optimizer = apex.optimizers.FusedAdam(
            params=params,
            lr=learning_rate,
            betas=(0.9, 0.99),
            eps=1e-6
        )
    elif opt=='adam':
        optimizer = AdamW(
            params=params,
            lr=learning_rate,
            betas=(0.9, 0.99),
            eps=1e-6,
            foreach=True
        )
    elif opt=='adan':
        optimizer = Adan(
            params=params,
            lr=learning_rate,
            eps=1e-6,
            fused=True
        )

    if warmup.isdigit():
        warmup = f'{int(warmup)/epochs}dur'
    else:
        try:
            float(warmup)
            warmup = f'{warmup}dur'
        except ValueError:
            pass
    cos_sched = CosineAnnealingWithWarmupScheduler(warmup, '1dur')

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        max_duration=f'{epochs}ep',
        optimizers=optimizer,
        schedulers=[cos_sched],
        device='gpu' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        compile_config = {} if compiler else None,
        algorithms=[LowPrecisionLayerNorm()] if lowprecision_ln else [],
        callbacks=[LRMonitor()],
        loggers=[logger] if logger is not None else [],
        progress_bar=progress_bar,
        device_train_microbatch_size=grad_accum_bs
    )

    return trainer

@app.command()
def train(ctx:typer.Context, # Typer Context to grab config for passing to WandB
    # Optional config file
    config:Optional[Path]=typer.Option(None, callback=conf_callback, is_eager=True),
    model:str=typer.Option('EleutherAI/pythia-160m-deduped'),
    batch_size:int=typer.Option(64),
    learning_rate:float=typer.Option(1e-4),
    weight_decay:float=typer.Option(1e-2),
    epochs:int=typer.Option(4),
    warmup:str=typer.Option('0.25dur'),
    opt:str=typer.Option('adam'),
    compiler:bool=typer.Option(True),
    lowprecision_ln:bool=typer.Option(True),
    train_subset:Optional[int]=typer.Option(None),
    max_length:int=typer.Option(256),
    progress_bar:bool=typer.Option(True),
    seed:int=typer.Option(42),
    grad_accum_bs:Optional[int]=typer.Option(None),
    label_smoothing:float=typer.Option(0),
    flash_attn:bool=typer.Option(True),
    save_name:str=typer.Option('EleutherAI/pythia-160m-deduped'),
    # Weights and Biases
    log_wandb:bool=typer.Option(False, "--wandb"),
    name:Optional[str]=typer.Option(None),
    project:str=typer.Option('nlp_study_group'),
    group:Optional[str]=typer.Option(None),
    tags:Optional[str]=typer.Option(None),
    entity:Optional[str]=typer.Option(None),

):
    ignore_params = ['config', 'verbose', 'log_wandb', 'name', 'project',
                     'group', 'tags', 'entity', 'save_code']
    config = {k:v for k,v in ctx.params.items() if k not in ignore_params}

    if log_wandb:
        logger = WandBLogger(project, f'{model}_sweep' if group is None else group,
                             name, entity, tags.split(','), init_kwargs={'config':config})
    else:
        logger = None

    trainer = create_model(**config, logger=logger)

    trainer.fit(
        precision='amp_fp16',
        duration=f'{epochs}ep',
        train_subset_num_batches=train_subset,
    )

    trainer.save_checkpoint(f'models/{save_name}.bin', weights_only=True)

if __name__=="__main__":
    app()