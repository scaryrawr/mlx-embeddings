from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings
from .pooling import pool_by_config


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen2"
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 32768
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = None
    architectures: List[str] = field(default_factory=lambda: ["Qwen2Model"])
    pooling_config: dict = field(default_factory=lambda: {"pooling_mode": "lasttoken"})

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.use_sliding_window:
            raise NotImplementedError(
                "Qwen2 sliding-window attention is not supported."
            )


class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rotary_emb = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(self, hidden_states: mx.array, attention_mask=None) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        query_states = self.rotary_emb(query_states)
        key_states = self.rotary_emb(key_states)

        if self.num_key_value_groups > 1:
            key_states = mx.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = mx.repeat(value_states, self.num_key_value_groups, axis=1)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scale,
            mask=attention_mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states: mx.array, attention_mask=None) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _create_attention_mask(
        self, attention_mask: mx.array, dtype: mx.Dtype
    ) -> mx.array:
        batch_size, seq_length = attention_mask.shape
        causal_mask = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
        causal_mask = mx.where(causal_mask, 0.0, -mx.inf).astype(dtype)
        causal_mask = causal_mask[None, None, :, :]

        padding_mask = attention_mask[:, None, None, :]
        padding_mask = mx.where(padding_mask == 1, 0.0, -mx.inf).astype(dtype)
        return mx.broadcast_to(
            causal_mask + padding_mask, (batch_size, 1, seq_length, seq_length)
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        hidden_states = self.embed_tokens(input_ids)
        input_mask = attention_mask[:, :, None].astype(hidden_states.dtype)
        hidden_states = mx.where(input_mask == 1, hidden_states, 0.0)
        mask = self._create_attention_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=mask)
            hidden_states = mx.where(input_mask == 1, hidden_states, 0.0)

        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length), dtype=mx.int32)
        elif attention_mask.shape != (batch_size, seq_length):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} does not match "
                f"input_ids shape {input_ids.shape}"
            )

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        text_embeds = pool_by_config(
            last_hidden_state, attention_mask, self.config.pooling_config
        )
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            text_embeds=text_embeds,
        )

    def sanitize(self, weights: dict) -> dict:
        sanitized_weights = {}
        for key, value in weights.items():
            if "lm_head.weight" in key or "rotary_emb.inv_freq" in key:
                continue
            new_key = key if key.startswith("model.") else f"model.{key}"
            sanitized_weights[new_key] = value
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
