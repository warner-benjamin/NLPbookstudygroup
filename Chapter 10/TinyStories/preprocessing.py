
import gc, os
from enum import Enum
from functools import partial
from pathlib import Path
from hashlib import md5
from typing import Optional

import ftfy.bad_codecs

import tqdm
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

import datasets
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers, Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer, SentencePieceUnigramTokenizer
from huggingface_hub import hf_hub_download, dataset_info

import typer
try:
    from rich import print
except ImportError:
    pass

class DataSlice(str, Enum):
    combined = 'combined'
    original = 'original'
    version2 = 'version2'

class TokenizerType(str, Enum):
    bytebpe = 'bytebpe'
    wordpiece = 'wordpiece'
    sentencepiece = 'sentencepiece'

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})

def download_data(repo):
    files = []
    datainfo = dataset_info(repo)
    for info in datainfo.siblings:
        if info.rfilename.endswith('txt'):
            files.append(hf_hub_download(repo_id=datainfo.id, filename=info.rfilename, repo_type='dataset',cache_dir='./data'))
    return files

def preprocess_dataset(files, dedup=True, valid=False):
    "Process file in chunks, adding to a buffer until we hit <|endoftext|> then yield to Dataset.from_generator"
    hashes = set()
    count = 0
    for file in files:
        with open(file, 'r', encoding='sloppy-windows-1252') as f:
            text_buffer = []
            for line in f:
                text_buffer.append(line)
                if '<|endoftext|>' in text_buffer[-1]:
                    parts = ''.join(text_buffer).split('<|endoftext|>')
                    chunk = parts[0].strip(' \n')
                    text_buffer = [parts[1]]
                    # Simple dedup step since we know that TinyStories has GPT-4 stories repeated accross
                    # version 1 and version 2. If different or larger dataset, use something like MiniHash
                    if dedup:
                        chunk_hash = md5(chunk.encode(), usedforsecurity=False).hexdigest()
                        if chunk_hash not in hashes:
                            hashes.add(chunk_hash)
                            yield {'text': chunk }
                        else:
                            count+=1
                    else:
                        yield {'text': chunk }
    hashes = None
    if dedup:
        print(f'\nRemoved {count} duplicates from {"valid" if valid else "train"} dataset')
    gc.collect()


def get_training_corpus(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]['text']


def tokenize_data(batch, tokenizer:Tokenizer):
    input_ids, num_tokens = [], []
    out = tokenizer.encode_batch(batch['text'])
    for o in out:
        input_ids.append(o.ids)
        num_tokens.append(len(o.ids))
    return {'input_ids': input_ids, 'num_tokens': num_tokens}


def group_data(batch, sequence_len:int, eos_token:int):
    concat = []
    for b in batch['input_ids']:
        concat.extend(b)
        concat.append(eos_token)
    total_tokens = len(concat)
    if total_tokens >= sequence_len:
        total_tokens = (total_tokens // sequence_len) * sequence_len
    return {'input_ids': [concat[i : i + sequence_len] for i in range(0, total_tokens, sequence_len)]}


@app.command(help="Download & preprocess dataset. Then Train tokenizer & tokenize dataset.")
def process(ctx:typer.Context, # Typer Context to grab config for passing to WandB
    data_slice:DataSlice=typer.Option(DataSlice.combined, help='Which slice of the TinyStories dataset to use'),
    tokenizer_type:TokenizerType=typer.Option(TokenizerType.bytebpe, help='Which tokenizer to train'),
    batch_size:int=typer.Option(4096, help='Batch size for processing data'),
    lower_case:bool=typer.Option(True, help='Add lower case normalizer to WordPiece'),
    vocab_size:int=typer.Option(8000, help='Tokenizer vocabulary size'),
    sequence_len:Optional[int]=typer.Option(None, help='Optionally group tokenized text into chunks of this size'),
):
    repo = "roneneldan/TinyStories"
    if data_slice == DataSlice.combined:
        name = 'TinyStories-Combined'
    elif data_slice == DataSlice.version2:
        name = 'TinyStories-V2'
    else:
        name = 'TinyStories-Org'
    fname = f'./data/{name}'

    if not Path(fname).exists():
        print('\nDownloading dataset: TinyStories\n')
        files = download_data(repo)

        print('\nProcessing dataset: TinyStories\n')
        train_files, valid_files = [], []
        for file in files:
            valid = 'valid' in file.lower()
            if 'tinystoriesv2' in file.lower() and data_slice in [DataSlice.combined, DataSlice.version2]:
                valid_files.append(file) if valid else train_files.append(file)
            elif 'tinystoriesv2' not in file.lower() and data_slice in [DataSlice.combined, DataSlice.original]:
                valid_files.append(file) if valid else train_files.append(file)

        dataset = DatasetDict({
            'train': Dataset.from_generator(partial(preprocess_dataset,
                                                    files=train_files,
                                                    dedup=data_slice==DataSlice.combined,
                                                    valid=False),
                                            cache_dir='./data'),
            'validation': Dataset.from_generator(partial(preprocess_dataset,
                                                         files=valid_files,
                                                         dedup=data_slice==DataSlice.combined,
                                                         valid=True),
                                                 cache_dir='./data'),
        })
        dataset.save_to_disk(fname)
    else:
        dataset = load_from_disk(fname)

    name = f'{name}_{tokenizer_type.value}_{vocab_size}'
    if data_slice==TokenizerType.wordpiece:
        name = f'{name}_{lower_case}'

    try:
        tokenizer = Tokenizer.from_file(f"./data/{name}.json")
    except:
        tokenizer = None

    # This section is loosely based on the Hugging Face course: https://huggingface.co/learn/nlp-course/chapter6/8
    if tokenizer is None:
        print(f'\nTraining {tokenizer_type.value} tokenizer: {vocab_size=}\n')
        special_tokens = ['<|unknown|>', '<|pad|>', '<|cls|>', '<|endoftext|>', '<|mask|>']
        if tokenizer_type == TokenizerType.wordpiece:
            tokenizer = Tokenizer(models.WordPiece(unk_token="<|unknown|>"))
            if lower_case:
                tokenizer.normalizer = normalizers.Sequence(
                    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
            else:
                tokenizer.normalizer = normalizers.Sequence(
                        [normalizers.NFD(), normalizers.StripAccents()])

            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.decoder = decoders.WordPiece(prefix="##")
            trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
            tokenizer.train_from_iterator(get_training_corpus(dataset['train'], batch_size=batch_size),
                                          trainer=trainer,
                                          length=len(dataset['train']))
        elif tokenizer_type == TokenizerType.bytebpe:
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator(get_training_corpus(dataset['train'], batch_size=batch_size),
                                          vocab_size=vocab_size,
                                          special_tokens=special_tokens[1:],
                                          length=len(dataset['train']))
        elif tokenizer_type == TokenizerType.sentencepiece:
            tokenizer = SentencePieceUnigramTokenizer()
            tokenizer.train_from_iterator(get_training_corpus(dataset['train'], batch_size=batch_size),
                                          vocab_size=vocab_size,
                                          special_tokens=special_tokens,
                                          length=len(dataset['train']),
                                          unk_token='<|unknown|>')
        tokenizer.save(f"./data/{name}.json")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    fname = f"./data/{name}_{sequence_len}" if sequence_len is not None else f"./data/{name}"
    if not Path(fname).exists():
        print(f"\nTokenizing dataset with {tokenizer_type.value} tokenizer: {vocab_size=}\n")
        dataset['train'] = dataset['train'].map(
            partial(tokenize_data, tokenizer=tokenizer),
            remove_columns=dataset['train'].column_names,
            batched=True, batch_size=batch_size, num_proc=cpu_count(),
        )
        dataset['validation'] = dataset['validation'].map(
            partial(tokenize_data, tokenizer=tokenizer),
            remove_columns=dataset['validation'].column_names,
            batched=True, batch_size=batch_size, num_proc=cpu_count(),
        )

        if sequence_len is not None:
            print(f"\nGrouping tokenized text into chunks of {sequence_len=}\n")
            eos_token=tokenizer.encode('<|endoftext|>').ids[0]
            # add 1 to the sequence length so when the data colaltor shifts and removes
            # one token for inputs and labels, the inputs are still of length sequence_len
            dataset['train'] = dataset['train'].map(
                partial(group_data, sequence_len=sequence_len+1, eos_token=eos_token),
                remove_columns=dataset['train'].column_names,
                batched=True, batch_size=batch_size, num_proc=cpu_count(),
            )
            dataset['validation'] = dataset['validation'].map(
                partial(group_data, sequence_len=sequence_len+1, eos_token=eos_token),
                remove_columns=dataset['validation'].column_names,
                batched=True, batch_size=batch_size, num_proc=cpu_count(),
            )

            dataset.save_to_disk(f'{fname}')
        else:
            print(f"\nStoring Number of Tokens as 'homepage'\n")
            for name in ['train', 'validation']:
                count = 0
                # definately a hack. There should be a way to return the total number of tokens
                # when tokenizing a dataset but I haven't figured it out
                for batch in tqdm.tqdm(DataLoader(dataset[name].select_columns('num_tokens').to_iterable_dataset(cpu_count()),
                                                  batch_size=1024, num_workers=cpu_count()), total=len(dataset[name])//1024 + 1):
                    count += batch['num_tokens'].sum().item()
                dataset[name].info.homepage = count

            dataset.save_to_disk(fname)

if __name__=="__main__":
    app()