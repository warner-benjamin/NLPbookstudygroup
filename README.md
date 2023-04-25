# *NLP With Transformers* Study Group Resources
[NLP with Transformers](https://transformersbook.com) study group materials and resources

## Setup with PyTorch 2.0

The following Conda + pip commands will install all the required packages for the
book's notebooks. Or run `setup-lite.sh` from bash. You may need to change
`source activate nlpbook` if conda isn't setup for bash.

```bash
conda create -n nlpbook python=3.10 "pytorch>=2.0.0" torchvision torchaudio torchtext \
pytorch-cuda=11.8 "transformers>=4.28.1" "datasets>=2.11.0" sentencepiece optuna \
scikit-learn onnxruntime matplotlib ipywidgets jupyterlab umap-learn seqeval nltk \
sacrebleu py7zr nlpaug scikit-multilearn psutil accelerate \
-c pytorch -c nvidia/label/cuda-11.8.0 -c huggingface -c conda-forge

conda activate nlpbook

pip install rouge-score bertviz
```

These commands will additionally install Cuda, fastai, blurr, fastxtend, Composer,
& Weights and Biases. Or run `setup-full.sh` from bash. You may need to change
`source activate nlpbook` if conda isn't setup for bash.

```bash
conda create -n nlpbook python=3.10 "pytorch>=2.0.0" torchvision torchaudio torchtext \
pytorch-cuda=11.8 cuda "transformers>=4.28.1" "datasets>=2.11.0" fastai sentencepiece \
optuna scikit-learn onnxruntime matplotlib ipywidgets jupyterlab umap-learn seqeval \
nltk sacrebleu py7zr nlpaug scikit-multilearn psutil accelerate wandb openpyxl xlrd \
torchmetrics requests coolname tabulate py-cpuinfo importlib-metadata \
-c pytorch -c nvidia/label/cuda-11.8.0 -c huggingface -c fastai -c conda-forge

conda activate nlpbook

pip install rouge-score bertviz fastxtend torch_optimizer
pip install ohmeow-blurr mosaicml --no-deps
```

`--no-deps` is so blurr and Composer won't uninstall PyTorch 2.0 for 1.13.