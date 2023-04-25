# Chapter 2 of NLP with Transformers

This chapter covers tokenizers and text classification models.

## Requirements

Read Chapter 2 of NLP with Transformers and run the companion [notebook](https://github.com/nlp-with-transformers/notebooks/blob/main/02_classification.ipynb)

## Suggested Homework

Train a model to classify movie reviews using the [IMDB dataset](https://huggingface.co/datasets/imdb). You may want to experiment on a subset before training on the entire dataset. (If using fastai, there is the [IMDB sample dataset](https://docs.fast.ai/data.external.html#main-datasets)).

Suggested models to try: DistilBERT, BERT, DistilRoBERTa, or DeBERTa. Or pick your own.

You can train using the Hugging Face trainer per book/notebook example, fastai via [blurr](https://ohmeow.github.io/blurr), [Composer](https://docs.mosaicml.com/projects/composer/en/latest/examples/finetune_huggingface.html), or your preferred framework.

You'll probably want to use Mixed Precision, TensorFloat32 MatMuls (if using an Ampere or newer GPU), fused optimizers, and PyTorch 2.0's Compile for fast training. If using fastai & blurr, you can find fused optimizers and a PyTorch Compile callback in [fastxtend](https://fastxtend.benjaminwarner.dev).

## Optional Additional Reading

Hugging Face Course:
 - [Fine-Tuning a Pretrained Model](https://huggingface.co/learn/nlp-course/chapter3/1)
 - [Tokenizers and the Tokenizers Library](https://huggingface.co/learn/nlp-course/chapter6/1)

If you only have time to read one, look at the Tokenizers chapter and videos.