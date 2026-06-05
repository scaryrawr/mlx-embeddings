from typing import Dict, Iterable, Optional, Union

import mlx.core as mx

from .models.base import normalize_embeddings

PromptInput = Union[str, Iterable[str]]

PROMPT_TEMPLATES: Dict[str, str] = {
    "search_query": "search_query: {text}",
    "search_document": "search_document: {text}",
    "code_query": "Represent this query for searching relevant code: {text}",
    "code_document": "{text}",
}


def _as_list(texts: PromptInput) -> tuple[list[str], bool]:
    if isinstance(texts, str):
        return [texts], True
    return list(texts), False


def _is_code_model(model) -> bool:
    if model is None or not hasattr(model, "config"):
        return False
    config = model.config
    if getattr(config, "model_type", None) == "qwen2":
        return True
    pooling_config = getattr(config, "pooling_config", {})
    return pooling_config.get("pooling_mode") == "cls" or pooling_config.get(
        "pooling_mode_cls_token", False
    )


def _resolve_prompt_name(prompt_name: Optional[str], model=None) -> Optional[str]:
    if prompt_name is None:
        return None
    normalized = prompt_name.lower()
    if normalized in ("none", "raw"):
        return None
    if normalized in PROMPT_TEMPLATES:
        return normalized
    if normalized == "query":
        return "code_query" if _is_code_model(model) else "search_query"
    if normalized in ("document", "doc"):
        return "code_document" if _is_code_model(model) else "search_document"
    raise ValueError(
        f"Unknown prompt_name {prompt_name!r}. "
        f"Supported names: {sorted(PROMPT_TEMPLATES)} plus 'query' and 'document'."
    )


def apply_prompt_template(
    texts: PromptInput,
    *,
    prompt_name: Optional[str] = None,
    prompt_template: Optional[str] = None,
    model=None,
) -> PromptInput:
    """Apply an opt-in prompt template to a string or list of strings."""

    if prompt_template is None:
        resolved_name = _resolve_prompt_name(prompt_name, model=model)
        if resolved_name is None:
            return texts
        prompt_template = PROMPT_TEMPLATES[resolved_name]
    elif "{text}" not in prompt_template:
        raise ValueError("prompt_template must include a '{text}' placeholder.")

    values, was_string = _as_list(texts)
    prefix = prompt_template.split("{text}", 1)[0]
    formatted = [
        (
            text
            if prefix and text.startswith(prefix)
            else prompt_template.format(text=text)
        )
        for text in values
    ]
    return formatted[0] if was_string else formatted


def truncate_embeddings(embeddings: mx.array, dimensions: int) -> mx.array:
    """Slice embeddings to ``dimensions`` and renormalize the result."""

    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0.")
    if dimensions > embeddings.shape[-1]:
        raise ValueError(
            f"dimensions ({dimensions}) cannot exceed embedding dimension "
            f"({embeddings.shape[-1]})."
        )
    return normalize_embeddings(embeddings[..., :dimensions])
