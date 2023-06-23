conda create -y -n nlpbook python=3.10 "pytorch>=2.0.1" torchvision torchaudio torchtext pytorch-cuda=11.8 cuda "transformers>=4.28.1" "datasets>=2.11.0" "tokenizers>=0.13.3" timm fastai sentencepiece optuna scikit-learn onnxruntime matplotlib ipywidgets jupyterlab umap-learn seqeval nltk sacrebleu py7zr nlpaug scikit-multilearn psutil accelerate wandb openpyxl xlrd torchmetrics requests coolname tabulate py-cpuinfo importlib-metadata -c pytorch -c nvidia/label/cuda-11.8.0 -c huggingface -c fastai -c conda-forge

source activate nlpbook

pip install rouge-score bertviz fastxtend torch_optimizer composer
pip install ohmeow-blurr --no-deps