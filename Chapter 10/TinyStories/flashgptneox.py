# GPTNeoXAttention implementation from:
# Transformers - Apache-2.0 license - Copyright Hugging Face

import torch
from fastcore.foundation import patch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXLayer, GPTNeoXMLP, GPTNeoXAttention, RotaryEmbedding, apply_rotary_pos_emb, CausalLMOutputWithPast, CrossEntropyLoss

from typing import Optional, Tuple

try:
    # Prevents torch.compile from graph breaking on einops functions. Requires einops>=0.6.1
    # https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    # Was using for testing flash_attn, don't think HF GPTNeoX uses `rearrange`
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()
except ImportError:
    pass

class FlashGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config):
        super(GPTNeoXAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # New code
        self.flash = getattr(config, 'use_flash_attention', False)
        self.fast_inference = getattr(config, 'use_fast_inference', False)
        # PyTorch 2.0 documentation states using the `torch.backends.cuda.sdp_kernel()` context manager is
        # prefered over this method, but context managers cause graph breaks with `compile` and this doesn't
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        if (self.flash or self.fast_inference) and not torch.backends.cuda.flash_sdp_enabled():
            torch.backends.cuda.enable_flash_sdp(True)
        # If in inference mode, we want all kernels enabled so PyTorch can pick the best one for the inputs
        if self.fast_inference and not torch.backends.cuda.mem_efficient_sdp_enabled():
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if self.fast_inference and not torch.backends.cuda.math_sdp_enabled():
            torch.backends.cuda.enable_math_sdp(True)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention - new code
        if self.flash:
            attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            attn_weights = None
        elif self.fast_inference:
            # attn_mask doesn't work with the flash attention kernel, hence the need for a seperate inference call
            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=attention_mask is None)
            attn_weights = None
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

@patch
def __init__(self:GPTNeoXLayer, config):
    super(GPTNeoXLayer, self).__init__()
    self.use_parallel_residual = config.use_parallel_residual
    self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention = FlashGPTNeoXAttention(config)
    self.mlp = GPTNeoXMLP(config)

@patch
def forward(
    self:GPTNeoXForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
        only required when the model is used as a decoder in a Sequence to Sequence model.

        Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
        `past_key_values` input) to speed up sequential decoding.

        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
        `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config.is_decoder = True
    >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.gpt_neox(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    lm_logits = self.embed_out(hidden_states)

    lm_loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # dataloader shifts this for us, so don't need to do it in the hugging face model
        # shift_logits = lm_logits[:, :-1, :].contiguous()
        # labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(lm_logits.view(-1, labels.size(-1)), labels.view(-1))

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    return CausalLMOutputWithPast(
        loss=lm_loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )