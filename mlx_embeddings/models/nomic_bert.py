from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings
from .pooling import pool_by_config


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "nomic_bert"
    n_embd: Optional[int] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_inner: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    vocab_size: int = 30528
    type_vocab_size: int = 2
    max_position_embeddings: Optional[int] = None
    n_positions: Optional[int] = None
    activation_function: str = "swiglu"
    attn_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    layer_norm_epsilon: Optional[float] = None
    layer_norm_eps: Optional[float] = None
    initializer_range: float = 0.02
    rotary_emb_base: float = 1000.0
    rotary_emb_fraction: float = 1.0
    rotary_emb_interleaved: bool = False
    rotary_scaling_factor: Optional[float] = None
    max_trained_positions: int = 2048
    qkv_proj_bias: bool = False
    mlp_fc1_bias: bool = False
    mlp_fc2_bias: bool = False
    prenorm: bool = False
    causal: bool = False
    add_pooling_layer: bool = False
    pad_token_id: int = 0
    architectures: List[str] = field(default_factory=lambda: ["NomicBertModel"])
    pooling_config: dict = field(default_factory=lambda: {"pooling_mode": "mean"})

    def __post_init__(self):
        self.n_embd = self.n_embd if self.n_embd is not None else self.hidden_size
        self.n_layer = (
            self.n_layer if self.n_layer is not None else self.num_hidden_layers
        )
        self.n_head = self.n_head if self.n_head is not None else self.num_attention_heads
        self.n_inner = (
            self.n_inner if self.n_inner is not None else self.intermediate_size
        )
        if None in (self.n_embd, self.n_layer, self.n_head, self.n_inner):
            raise ValueError(
                "NomicBERT config must include hidden/layer/head/intermediate sizes."
            )
        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.intermediate_size = self.n_inner
        if self.layer_norm_epsilon is None:
            self.layer_norm_epsilon = (
                self.layer_norm_eps if self.layer_norm_eps is not None else 1e-12
            )
        self.layer_norm_eps = self.layer_norm_epsilon
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.n_positions or self.max_trained_positions
        if self.causal:
            raise NotImplementedError("Causal NomicBERT variants are not supported.")


class NomicBertEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.type_vocab_size = config.type_vocab_size
        self.max_position_embeddings = (
            config.max_position_embeddings if config.rotary_emb_fraction <= 0 else 0
        )
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                self.max_position_embeddings, config.hidden_size
            )

    def __call__(self, input_ids, token_type_ids=None, position_ids=None):
        embeddings = self.word_embeddings(input_ids)
        batch_size, seq_length, _ = embeddings.shape

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = mx.zeros((batch_size, seq_length), dtype=mx.int32)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = mx.arange(seq_length, dtype=mx.int32)[None, :]
            embeddings = embeddings + self.position_embeddings(position_ids)

        return embeddings


class NomicBertGatedMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.activation_function = config.activation_function
        self.fc11 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc12 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_fc2_bias
        )

    def _activation(self, x: mx.array) -> mx.array:
        if self.activation_function == "swiglu":
            return nn.silu(x)
        if self.activation_function == "geglu":
            return nn.gelu(x)
        if self.activation_function == "glu":
            return mx.sigmoid(x)
        return nn.gelu(x)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        y = self.fc11(hidden_states)
        gate = self.fc12(hidden_states)
        return self.fc2(y * self._activation(gate))


class NomicBertMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.activation_function = config.activation_function
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_fc1_bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_fc2_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return self.fc2(hidden_states)


class NomicBertAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.rotary_emb_dim = int(self.head_dim * config.rotary_emb_fraction)

        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.qkv_proj_bias
        )
        self.drop = nn.Dropout(p=config.attn_pdrop)
        self.rotary_emb = (
            nn.RoPE(
                self.rotary_emb_dim,
                traditional=config.rotary_emb_interleaved,
                base=config.rotary_emb_base,
            )
            if self.rotary_emb_dim > 0
            else None
        )

    def __call__(self, hidden_states: mx.array, attention_mask=None) -> mx.array:
        batch_size, seq_length, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)

        query = qkv[:, :, 0].transpose(0, 2, 1, 3)
        key = qkv[:, :, 1].transpose(0, 2, 1, 3)
        value = qkv[:, :, 2].transpose(0, 2, 1, 3)

        if self.rotary_emb is not None:
            query = self.rotary_emb(query)
            key = self.rotary_emb(key)

        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=attention_mask
        )
        attn_output = self.drop(attn_output)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )
        return self.out_proj(attn_output)


class NomicBertBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.prenorm = config.prenorm
        self.attn = NomicBertAttention(config)
        self.mlp = (
            NomicBertGatedMLP(config)
            if config.activation_function in ("glu", "swiglu", "geglu")
            else NomicBertMLP(config)
        )
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )
        self.dropout2 = nn.Dropout(config.resid_pdrop)
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )

    def __call__(self, hidden_states: mx.array, attention_mask=None) -> mx.array:
        if self.prenorm:
            residual = hidden_states
            hidden_states = self.norm1(hidden_states)
            hidden_states = residual + self.dropout1(
                self.attn(hidden_states, attention_mask=attention_mask)
            )
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            return residual + self.dropout2(self.mlp(hidden_states))

        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm1(self.dropout1(attn_outputs) + hidden_states)
        mlp_outputs = self.mlp(hidden_states)
        return self.norm2(self.dropout2(mlp_outputs) + hidden_states)


class NomicBertEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [NomicBertBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, attention_mask=None) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return hidden_states


class NomicBertPooler(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.activation(self.dense(hidden_states[:, 0]))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = NomicBertEmbeddings(config)
        self.emb_drop = nn.Dropout(config.embd_pdrop)
        self.emb_ln = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon, bias=True
        )
        self.encoder = NomicBertEncoder(config)
        self.pooler = NomicBertPooler(config) if config.add_pooling_layer else None

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return (1.0 - extended_attention_mask) * -10000.0

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        token_type_ids: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_length), dtype=mx.int32)

        hidden_states = self.embeddings(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )
        hidden_states = self.emb_ln(hidden_states)
        hidden_states = self.emb_drop(hidden_states)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        sequence_output = self.encoder(
            hidden_states, attention_mask=extended_attention_mask
        )
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        text_embeds = pool_by_config(
            sequence_output, attention_mask, self.config.pooling_config
        )
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            text_embeds=text_embeds,
            pooler_output=pooled_output,
        )

    def sanitize(self, weights: dict) -> dict:
        sanitized_weights = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key or "position_ids" in key:
                continue
            sanitized_weights[key] = value
        return sanitized_weights
