# Chapter 8 of NLP with Transformers

Chapter 8 of *NLP with Transformers* is about efficient model inference, and covers model distillation, quantization, and sparsity.

## Optional Homework
Train (or fine tune) a distilled model

## Additional Resources

### Quantization

- [LLM.int8() and Emergent Features — Tim Dettmers](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features) & [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078)

### Model Changes

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) - used in StarCoder
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - used in Falcon

### Algorithm Changes

- [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) & [BetterTransformer](https://pytorch.org/tutorials/beginner/bettertransformer_tutorial.html)
- [Blockwise Parallel Transformer for Long Context Large Models](https://arxiv.org/abs/2305.19370)

### Libraries

- [ggml: Tensor library for machine learning](https://github.com/ggerganov/ggml) - base for llama.cpp & whisper.cpp
- [TorchServe: Increasing inference speed while improving efficiency](https://pytorch.org/serve/)
- [Accelerating Inference Up to 6x Faster in PyTorch with Torch-TensorRT](https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/)
- [NVIDIA/FasterTransformer: Transformer related optimization, including BERT, GPT](https://github.com/NVIDIA/FasterTransformer)
- [🤗 Optimum](https://huggingface.co/docs/optimum/v1.8.6/index) - easy way to use BetterTransformer and ONNX with Hugging Face models
- [text-generation-inference: Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [Optimize ONNX Model — Hidet Documentation](https://docs.hidet.org/stable/gallery/tutorials/run-onnx-model.html)
- [Lightning-AI/lit-parrot: Implementation of the StableLM/Pythia/INCITE language models](https://github.com/Lightning-AI/lit-parrot)
- [FasterAI](https://nathanhubens.github.io/fasterai/) - distillation & sparsity for fastai