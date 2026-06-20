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


def _template_from_prefix(prefix: str) -> str:
    return prefix if "{text}" in prefix else f"{prefix}{{text}}"


def _model_prompts(model) -> Dict[str, str]:
    if model is None or not hasattr(model, "config"):
        return {}
    prompts = getattr(model.config, "prompts", {})
    return prompts if isinstance(prompts, dict) else {}


def _resolve_prompt_template(prompt_name: Optional[str], model=None) -> Optional[str]:
    if prompt_name is None:
        return None
    normalized = prompt_name.lower()
    if normalized in ("none", "raw"):
        return None
    model_prompts = _model_prompts(model)
    if normalized in model_prompts:
        return _template_from_prefix(model_prompts[normalized])
    if normalized in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[normalized]
    if normalized == "query":
        resolved = "code_query" if _is_code_model(model) else "search_query"
        return PROMPT_TEMPLATES[resolved]
    if normalized in ("document", "doc"):
        resolved = "code_document" if _is_code_model(model) else "search_document"
        return PROMPT_TEMPLATES[resolved]
    raise ValueError(
        f"Unknown prompt_name {prompt_name!r}. "
        f"Supported names: {sorted(PROMPT_TEMPLATES)} plus 'query', 'document', "
        "and names from the loaded model prompts."
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
        prompt_template = _resolve_prompt_template(prompt_name, model=model)
        if prompt_template is None:
            return texts
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
