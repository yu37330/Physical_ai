from __future__ import annotations

from typing import Optional

SUPPORTED_TRANSFORMERS_VERSION = "4.40.1"
_PATCH_MARKER = "_parc_openvla_oft_bidirectional_patch"


def assert_supported_transformers_version() -> None:
    import transformers

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Expected transformers=={SUPPORTED_TRANSFORMERS_VERSION}, got {transformers.__version__}"
        )


def _padding_only_bidirectional_mask(causal_mask):
    if causal_mask is None:
        return None
    query_length = causal_mask.shape[-2]
    last_row = causal_mask[:, :, -1, :].clone()
    return last_row.unsqueeze(2).expand(-1, -1, query_length, -1)


def apply_bidirectional_attention_patch() -> None:
    """Patch Transformers 4.40.1 to reproduce the OpenVLA-OFT Llama SDPA change."""
    assert_supported_transformers_version()

    import torch
    from transformers.models.llama import modeling_llama

    attention_class = modeling_llama.LlamaSdpaAttention
    if getattr(attention_class, _PATCH_MARKER, False):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if output_attentions:
            return modeling_llama.LlamaAttention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
        value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

        mask = attention_mask
        if mask is not None:
            mask = mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda" and mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        mask = _padding_only_bidirectional_mask(mask)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    attention_class.forward = forward
    setattr(attention_class, _PATCH_MARKER, True)
