import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.lfm2 import Lfm2DecoderLayer
from mlx_lm.models.lfm2 import ModelArgs as Lfm2ModelArgs
from mlx_lm.models.lfm2 import ShortConv

from .base import BaseModelOutput, normalize_embeddings
from .pooling import mean_pooling, pool_by_config


@dataclass
class ModelArgs(Lfm2ModelArgs):
    out_features: Optional[int] = None
    architectures: List[str] = field(default_factory=list)
    auto_map: Optional[Dict[str, str]] = None
    dense_config: Optional[Dict[str, Any]] = None
    pooling_config: Optional[Dict[str, Any]] = None
    prompts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.dense_config is not None:
            self.out_features = self.dense_config.get("out_features", self.out_features)


class BidirectionalShortConv(ShortConv):
    """Short convolution with centered padding for bidirectional LFM2 encoders."""

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ):
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x

        target_length = Bx.shape[1]
        pad = self.L_cache // 2
        Bx = mx.pad(Bx, [(0, 0), (pad, pad), (0, 0)])
        conv_out = self.conv(Bx)

        if conv_out.shape[1] > target_length:
            conv_out = conv_out[:, :target_length, :]
        elif conv_out.shape[1] < target_length:
            conv_out = mx.pad(
                conv_out,
                [(0, 0), (0, target_length - conv_out.shape[1]), (0, 0)],
            )

        y = C * conv_out
        return self.out_proj(y)


class BidirectionalLfm2DecoderLayer(Lfm2DecoderLayer):
    """LFM2 layer variant that uses non-causal short convolution."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__(args, layer_idx=layer_idx)
        if not self.is_attention_layer:
            self.conv = BidirectionalShortConv(args, layer_idx)


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.is_bidirectional = _uses_bidirectional_backbone(args)
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        layer_class = (
            BidirectionalLfm2DecoderLayer if self.is_bidirectional else Lfm2DecoderLayer
        )
        self.layers = [
            layer_class(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if self.is_bidirectional:
            attn_mask = self._bidirectional_attention_mask(h, attention_mask)
            for layer in self.layers:
                mask = attn_mask if layer.is_attention_layer else None
                h = layer(h, mask, cache=None)
            return self.embedding_norm(h)

        if cache is None:
            cache = [None] * len(self.layers)

        attn_mask = create_attention_mask(h, cache[self.fa_idx])
        conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)

    def _bidirectional_attention_mask(
        self, h: mx.array, attention_mask: Optional[mx.array]
    ) -> Optional[mx.array]:
        if attention_mask is None:
            return None
        padding_mask = attention_mask[:, None, None, :]
        return mx.where(padding_mask.astype(mx.bool_), 0.0, -mx.inf).astype(h.dtype)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm2Model(config)
        self.dense = []
        if config.out_features is not None:
            dense_config = config.dense_config or {}
            self.dense = [
                nn.Linear(
                    dense_config.get("in_features", config.block_dim),
                    config.out_features,
                    bias=dense_config.get("bias", False),
                ),
            ]

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = mx.repeat(
                extended_attention_mask, attention_mask.shape[-1], -2
            )

        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return extended_attention_mask

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        input_ids: Optional[mx.array] = None,
    ):
        if inputs is None:
            inputs = input_ids
        elif input_ids is not None:
            raise ValueError("Pass either inputs or input_ids, not both.")
        if inputs is None:
            raise ValueError("inputs or input_ids must be provided.")

        if attention_mask is None:
            attention_mask = mx.ones(inputs.shape)

        if self.model.is_bidirectional:
            h = self.model(inputs, attention_mask=attention_mask)
        else:
            h = self.model(inputs, cache=self.make_cache)
        out = h
        for dense in self.dense:
            out = dense(out)

        token_embeds = normalize_embeddings(out) * attention_mask[:, :, None]

        if self.config.pooling_config is not None:
            text_embeds = pool_by_config(
                out, attention_mask, self.config.pooling_config
            )
            text_embeds = normalize_embeddings(text_embeds)
            pooled = text_embeds
        else:
            text_embeds = token_embeds
            pooled = mean_pooling(token_embeds, attention_mask)

        return BaseModelOutput(
            last_hidden_state=h,
            text_embeds=text_embeds,
            pooler_output=pooled,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():

            if "linear" not in k and "dense" not in k:
                new_key = f"model.{k}" if not k.startswith("model") else k
                if "conv.weight" in new_key:
                    if v.shape[-1] > v.shape[1]:
                        v = v.transpose(0, 2, 1)

                sanitized_weights[new_key] = v
            elif "1_Dense.linear" in k:
                new_key = k.replace("1_Dense.linear", "dense.0")
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.model.layers
        ]


def _uses_bidirectional_backbone(config: ModelArgs) -> bool:
    architectures = config.architectures or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if "Lfm2BidirectionalModel" in architectures:
        return True

    auto_map = config.auto_map or {}
    auto_model = auto_map.get("AutoModel", "")
    return "Lfm2BidirectionalModel" in auto_model
