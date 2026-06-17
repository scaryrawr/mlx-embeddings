# Repository Guidelines

## Project Structure & Module Organization
`mlx_embeddings/` is the Python package. Public imports come from `mlx_embeddings/__init__.py`, with `utils.py` owning `load()`/`generate()` and Hugging Face download/loading helpers. Model dispatch is driven by `config["model_type"]` (hyphens remapped to underscores) into `mlx_embeddings/models/<model_type>.py`; new architectures should expose `Model` and `ModelArgs` like the existing modules. Multimodal-specific code lives under `mlx_embeddings/models/qwen3_vl/` and `mlx_embeddings/models/llama_nemotron_vl/`. Conversion and quantization CLI logic is in `mlx_embeddings/convert.py`. Tests live in `mlx_embeddings/tests/`.

## Build, Test, and Development Commands
- `uv run pytest mlx_embeddings/tests -q` — preferred when `uv` is available; it uses the checked-in `uv.lock` environment and avoids relying on a system `python` executable.
- `python -m pip install -e ".[dev]"` — editable install with pytest helpers; this project requires Python >=3.10 and MLX/Metal, so develop on macOS/Apple Silicon.
- `python -m pip install pre-commit && pre-commit run --all` — matches CI style checks (Black and isort only).
- `python -m pytest mlx_embeddings/tests -q` — run the normal unit suite from the repo root; `pyproject.toml` excludes the smoke script.
- `python -m pytest mlx_embeddings/tests/test_pooling.py -q` — fast narrow validation for pooling changes.
- `python -m build` — build release artifacts, matching the publish workflow.

## Coding Style & Naming Conventions
Use 4-space indentation, LF endings, Black formatting, and isort with `--profile=black`; flake8 is configured for 88 columns. Keep model filenames aligned with Hugging Face `model_type` after replacing `-` with `_`. Prefer small dataclass-style config objects via `BaseModelArgs.from_dict()` and return the output dataclasses in `models/base.py` where possible.

## Testing Guidelines
CI runs on `macos-14` with Python 3.10, installs MLX, then runs pre-commit before tests. Avoid adding unit tests that require network downloads or large real checkpoints; mock Hugging Face/model loading as in `test_convert.py` and `test_models.py`. `mlx_embeddings/tests/test_smoke.py` is a manual integration script for real model paths/images: run it explicitly with `python mlx_embeddings/tests/test_smoke.py --models <model-or-list> [--images <paths...>]`.

## Commit & Pull Request Guidelines
No commit convention is enforced; existing history uses concise imperative subjects, sometimes with PR numbers. For PRs to `main`, include the command output for pre-commit and relevant pytest runs, and call out any model downloads, large files, or Apple-Silicon-only behavior.

## Agent-Specific Instructions
Do not commit generated artifacts (`dist/`, `build/`, `*.egg-info/`, caches, downloaded/private models). Be careful with conversion or smoke-test commands: they can download large Hugging Face checkpoints and may need authentication for gated repositories.
