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

from torchmetrics import Metric, BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from composer.algorithms import LowPrecisionLayerNorm
from composer.callbacks import LRMonitor
from composer.core import Evaluator, Event
from composer.loggers import WandBLogger
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import LanguagePerplexity
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.utils import dist
from composer import Algorithm, Trainer

from timm.optim.optim_factory import param_groups_weight_decay

from fastxtend.utils import less_random

try:
    from adan import Adan
    ADAN = True
except ImportError:
    ADAN = False

from transformers.utils import logging as hf_logging

from transformers.models.t5 import T5ForConditionalGeneration

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


class T5HuggingFaceModel(HuggingFaceModel):
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
        loss: Optional[nn.Module] = None,
        max_gen_length: int = 128,
        generation_kwargs : dict = {},
    ) -> None:
        super().__init__(model, tokenizer, use_logits, metrics, eval_metrics, shift_labels, allow_embedding_resizing)
        self.loss_fn = loss
        self.max_gen_length = max_gen_length
        self.generation_kwargs = generation_kwargs
        # Accelerate might be breaking the inspect method with its hook, so manually set
        self.model_forward_args = ["input_ids","attention_mask","decoder_input_ids","decoder_attention_mask",
                                   "head_mask","decoder_head_mask","cross_attn_head_mask","encoder_outputs",
                                   "past_key_values","inputs_embeds","decoder_inputs_embeds","labels",
                                   "use_cache","output_attentions","output_hidden_states","return_dict"]

    def forward(self, batch):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            if self.loss_fn is not None:
                batch = {k: v for k, v in batch.items() if k in self.model_forward_args and k != 'labels'}
                output = self.model(**batch, return_dict=True)  # type: ignore (thirdparty)
            else:
                batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
                output = self.model(**batch, return_dict=True)  # type: ignore (thirdparty)
            return output
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

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        "Hardcodes eval to generate mode"
        if self.tokenizer is None:
            raise ValueError(
                'Generation eval cannot be used without providing a tokenizer to the model constructor.')

        self.labels = self.tokenizer.batch_decode(batch.pop('labels'), skip_special_tokens=True)
        generation = self.generate(batch['input_ids'],
                                   attention_mask=batch['attention_mask'],
                                   max_length=self.max_gen_length,
                                   synced_gpus=dist.get_world_size() > 1,
                                   **self.generation_kwargs)
        return self.tokenizer.batch_decode(generation, skip_special_tokens=True)

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else None


def dataset(model='google/flan-t5-small', max_length=768, instruction=False):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    def prepare_dialogue(sample, instruction=instruction):
        dialogues = []
        for dialogue in sample["dialogue"]:
            if instruction:
                dialogues.append("Summarize the following text:\n" + dialogue.replace('\r', ''))
            else:
                dialogues.append(dialogue.replace('\r', ''))
        sample["dialogue"] = dialogues
        return sample

    def tokenize_function(sample):
        input_encodings = tokenizer(sample["dialogue"],
                                    max_length=max_length,
                                    padding='max_length',
                                    truncation=True)

        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(sample["summary"],
                                         max_length=128,
                                         padding='max_length',
                                         truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}

    dataset = datasets.load_dataset("samsum")

    prepared_dataset = dataset.map(prepare_dialogue,
                                    batched=True,
                                    num_proc=cpu_count(),
                                    batch_size=100)

    tokenized_dataset = prepared_dataset.map(tokenize_function,
                                             batched=True,
                                             num_proc=cpu_count(),
                                             batch_size=100)

    return tokenizer, tokenized_dataset.remove_columns(dataset['train'].column_names)


def create_model(model='google/flan-t5-small', batch_size=40, learning_rate=1e-4,
                 weight_decay=1e-3, epochs=4, warmup='0.25dur', opt='adam', compiler=False,
                 lowprecision_ln=False, max_length=768, logger=None, progress_bar=False, seed=42,
                 grad_accum_bs=None, instruction=False, **kwargs):

    tokenizer, tokenized_datasets = dataset(model, max_length)

    with less_random(seed):
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model, device_map='auto')

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    with less_random(seed):
        train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True,
                                      drop_last=True, collate_fn=data_collator, num_workers=cpu_count())
        eval_dataloader  = DataLoader(tokenized_datasets['validation'], batch_size=batch_size, shuffle=False,
                                      drop_last=False, collate_fn=data_collator, num_workers=cpu_count())
        test_dataloader  = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False,
                                      drop_last=False, collate_fn=data_collator, num_workers=cpu_count())

    # Package as a trainer-friendly Composer model
    metrics = [BLEUScore(), ROUGEScore()]
    composer_model = T5HuggingFaceModel(model, tokenizer=tokenizer, eval_metrics=metrics, metrics=None,
                                        loss=None, generation_kwargs={'num_beams': 8, 'length_penalty':0.8})
    composer_model.train_metrics = None

    # Setup optimizer and scheduler
    if weight_decay != 0:
        params = param_groups_weight_decay(composer_model, weight_decay=weight_decay)
    else:
        params = composer_model.parameters()

    if opt=='adam':
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

    eval_evaluator = Evaluator(label='eval', dataloader=eval_dataloader,
                               device_eval_microbatch_size=grad_accum_bs*4)

    # Create Trainer Object
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_evaluator,
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
        device_train_microbatch_size=grad_accum_bs,
        eval_interval=epochs
    )

    return trainer, test_dataloader

@app.command()
def train(ctx:typer.Context, # Typer Context to grab config for passing to WandB
    # Optional config file
    config:Optional[Path]=typer.Option(None, callback=conf_callback, is_eager=True),
    model:str=typer.Option('google/flan-t5-small'),
    batch_size:int=typer.Option(64),
    learning_rate:float=typer.Option(1e-4),
    weight_decay:float=typer.Option(1e-2),
    epochs:int=typer.Option(4),
    warmup:str=typer.Option('0.25dur'),
    opt:str=typer.Option('adam'),
    compiler:bool=typer.Option(False),
    train_subset:Optional[int]=typer.Option(None),
    max_length:int=typer.Option(768),
    progress_bar:bool=typer.Option(True),
    seed:int=typer.Option(42),
    grad_accum_bs:Optional[int]=typer.Option(None),
    instruction:bool=typer.Option(False),
    save_name:Optional[str]=typer.Option(None),
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
                             name, entity, tags.split(',') if tags is not None else tags,
                             init_kwargs={'config':config})
    else:
        logger = None

    trainer, test_dataloader = create_model(**config, logger=logger)

    trainer.fit(
        precision='amp_bf16',
        duration=f'{epochs}ep',
        train_subset_num_batches=train_subset,
    )

    if save_name is not None:
        trainer.save_checkpoint(f'models/{save_name}.bin', weights_only=True)

if __name__=="__main__":
    app()