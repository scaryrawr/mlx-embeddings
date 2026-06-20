import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_embeddings.helpers import apply_prompt_template, truncate_embeddings
from mlx_embeddings.models.base import (
    BaseModelArgs,
    BaseModelOutput,
    ViTModelOutput,
    normalize_embeddings,
)
from mlx_embeddings.tokenizer_utils import TokenizerWrapper, load_tokenizer
from mlx_embeddings.utils import _enrich_config_from_model_path


class TestBaseModelArgs:
    def test_from_dict(self):
        # Create a sample class that inherits from BaseModelArgs
        class TestArgs(BaseModelArgs):
            def __init__(self, a=1, b=2, c=3):
                self.a = a
                self.b = b
                self.c = c

        # Test with exact params
        params = {"a": 10, "b": 20, "c": 30}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 20
        assert args.c == 30

        # Test with extra params (should be ignored)
        params = {"a": 10, "b": 20, "c": 30, "d": 40}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 20
        assert args.c == 30
        assert not hasattr(args, "d")

        # Test with missing params (should use defaults)
        params = {"a": 10}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 2
        assert args.c == 3


class TestBaseModelOutput:
    def test_initialization(self):
        # Test default initialization
        output = BaseModelOutput()
        assert output.last_hidden_state is None
        assert output.pooler_output is None
        assert output.text_embeds is None
        assert output.hidden_states is None

        # Test with values
        mock_array = mx.array([1, 2, 3])
        mock_list = [mx.array([1, 2]), mx.array([3, 4])]
        output = BaseModelOutput(
            last_hidden_state=mock_array,
            pooler_output=mock_array,
            text_embeds=mock_array,
            hidden_states=mock_list,
        )
        assert output.last_hidden_state is mock_array
        assert output.pooler_output is mock_array
        assert output.text_embeds is mock_array
        assert output.hidden_states is mock_list


class TestViTModelOutput:
    def test_initialization(self):
        # Test default initialization
        output = ViTModelOutput()
        assert output.logits is None
        assert output.text_embeds is None
        assert output.image_embeds is None
        assert output.logits_per_text is None
        assert output.logits_per_image is None
        assert output.text_model_output is None
        assert output.vision_model_output is None

        # Test with values
        mock_array = mx.array([1, 2, 3])
        output = ViTModelOutput(
            logits=mock_array,
            text_embeds=mock_array,
            image_embeds=mock_array,
            logits_per_text=mock_array,
            logits_per_image=mock_array,
            text_model_output=mock_array,
            vision_model_output=mock_array,
        )
        assert output.logits is mock_array
        assert output.text_embeds is mock_array
        assert output.image_embeds is mock_array
        assert output.logits_per_text is mock_array
        assert output.logits_per_image is mock_array
        assert output.text_model_output is mock_array
        assert output.vision_model_output is mock_array


class TestNormalizeEmbeddings:
    def test_normalize_embeddings(self):
        # Test case 1: 2D array
        embeddings = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = normalize_embeddings(embeddings)

        # Check that each row has unit norm
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        np.testing.assert_allclose(norms.tolist(), [1.0, 1.0], rtol=1e-5)

        # Test case 2: 3D array
        embeddings = mx.random.normal((2, 3, 4))
        normalized = normalize_embeddings(embeddings)

        # Check shape is preserved
        assert normalized.shape == embeddings.shape

        # Check that each vector in the last dimension has unit norm
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        expected_norms = mx.ones((2, 3))
        np.testing.assert_allclose(norms.tolist(), expected_norms.tolist(), rtol=1e-5)

        # Test case 3: Small values (testing the epsilon)
        embeddings = mx.zeros((2, 3))
        normalized = normalize_embeddings(embeddings, eps=1.0)
        expected = mx.zeros((2, 3))
        np.testing.assert_allclose(normalized.tolist(), expected.tolist(), rtol=1e-5)


class TestTokenizerWrapper:
    def test_call_forwards_to_underlying_tokenizer(self):
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

            def decode(self, tokens):
                return str(tokens)

        wrapper = TokenizerWrapper(DummyTokenizer())
        output = wrapper(["hello"], return_tensors="mlx", padding=True)

        assert output["args"] == (["hello"],)
        assert output["kwargs"] == {"return_tensors": "mlx", "padding": True}

    def test_batch_encode_plus_falls_back_to_call(self):
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

            def decode(self, tokens):
                return str(tokens)

        wrapper = TokenizerWrapper(DummyTokenizer())
        output = wrapper.batch_encode_plus(["hello", "world"], return_tensors="mlx")

        assert output["args"] == (["hello", "world"],)
        assert output["kwargs"] == {"return_tensors": "mlx"}

    def test_load_tokenizer_recovers_from_list_extra_special_tokens(
        self, tmp_path, monkeypatch
    ):
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {}

            def decode(self, tokens):
                return ""

        calls = []

        def fake_from_pretrained(path, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise AttributeError("'list' object has no attribute 'keys'")
            return DummyTokenizer()

        monkeypatch.setattr(
            "mlx_embeddings.tokenizer_utils.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )

        wrapper = load_tokenizer(tmp_path)

        assert isinstance(wrapper, TokenizerWrapper)
        assert calls == [{}, {"extra_special_tokens": {}}]

    def test_load_tokenizer_reraises_unrelated_attribute_errors(
        self, tmp_path, monkeypatch
    ):
        def fake_from_pretrained(path, **kwargs):
            raise AttributeError("different tokenizer failure")

        monkeypatch.setattr(
            "mlx_embeddings.tokenizer_utils.AutoTokenizer.from_pretrained",
            fake_from_pretrained,
        )

        with pytest.raises(AttributeError, match="different tokenizer failure"):
            load_tokenizer(tmp_path)


class TestPromptHelpers:
    def test_apply_named_prompt_template(self):
        assert (
            apply_prompt_template("Who is Laurens?", prompt_name="search_query")
            == "search_query: Who is Laurens?"
        )

    def test_apply_prompt_template_does_not_duplicate_prefix(self):
        text = "search_query: Who is Laurens?"
        assert apply_prompt_template(text, prompt_name="search_query") == text

    def test_query_prompt_infers_code_model(self):
        class Config:
            model_type = "qwen2"
            pooling_config = {"pooling_mode": "lasttoken"}

        class Model:
            config = Config()

        assert (
            apply_prompt_template(
                "Calculate factorial", prompt_name="query", model=Model()
            )
            == "Represent this query for searching relevant code: Calculate factorial"
        )

    def test_query_prompt_infers_legacy_cls_pooling_as_code_model(self):
        class Config:
            model_type = "nomic_bert"
            pooling_config = {
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken": False,
            }

        class Model:
            config = Config()

        assert (
            apply_prompt_template("Find factorial", prompt_name="query", model=Model())
            == "Represent this query for searching relevant code: Find factorial"
        )

    def test_document_prompt_infers_text_model(self):
        class Config:
            model_type = "nomic_bert"
            pooling_config = {"pooling_mode": "mean"}

        class Model:
            config = Config()

        assert apply_prompt_template(
            ["A document"], prompt_name="document", model=Model()
        ) == ["search_document: A document"]

    def test_prompt_name_uses_model_prompts_before_heuristics(self):
        class Config:
            model_type = "lfm2"
            pooling_config = {"pooling_mode_cls_token": True}
            prompts = {"query": "query: ", "document": "document: "}

        class Model:
            config = Config()

        assert (
            apply_prompt_template("Where is Paris?", prompt_name="query", model=Model())
            == "query: Where is Paris?"
        )
        assert (
            apply_prompt_template(
                "Paris is in France.", prompt_name="document", model=Model()
            )
            == "document: Paris is in France."
        )

    def test_empty_model_prompt_leaves_text_unchanged(self):
        class Config:
            prompts = {"query": ""}

        class Model:
            config = Config()

        assert (
            apply_prompt_template("hello", prompt_name="query", model=Model())
            == "hello"
        )

    def test_truncate_embeddings_renormalizes(self):
        embeddings = mx.array([[3.0, 4.0, 12.0], [1.0, 2.0, 2.0]])
        truncated = truncate_embeddings(embeddings, 2)

        assert truncated.shape == (2, 2)
        norms = mx.linalg.norm(truncated, ord=2, axis=-1)
        np.testing.assert_allclose(norms.tolist(), [1.0, 1.0], rtol=1e-5)


class TestConfigEnrichment:
    def test_reads_sentence_transformers_sidecars(self, tmp_path):
        dense_dir = tmp_path / "1_Dense"
        dense_dir.mkdir()
        (dense_dir / "config.json").write_text(
            json.dumps({"in_features": 16, "out_features": 8, "bias": False})
        )

        pooling_dir = tmp_path / "1_Pooling"
        pooling_dir.mkdir()
        (pooling_dir / "config.json").write_text(
            json.dumps({"pooling_mode_cls_token": True})
        )

        (tmp_path / "config_sentence_transformers.json").write_text(
            json.dumps(
                {
                    "prompts": {"query": "query: "},
                    "query_prefix": "[Q] ",
                    "document_prefix": "[D] ",
                }
            )
        )

        config = _enrich_config_from_model_path({"model_type": "lfm2"}, tmp_path)

        assert config["dense_config"]["out_features"] == 8
        assert config["out_features"] == 8
        assert config["pooling_config"]["pooling_mode_cls_token"] is True
        assert config["prompts"] == {"query": "query: ", "document": "[D] "}

    def test_pylate_prefixes_fill_empty_prompts(self, tmp_path):
        (tmp_path / "config_sentence_transformers.json").write_text(
            json.dumps(
                {
                    "prompts": {"query": "", "document": ""},
                    "query_prefix": "[Q] ",
                    "document_prefix": "[D] ",
                }
            )
        )

        config = _enrich_config_from_model_path({"model_type": "lfm2"}, tmp_path)

        assert config["prompts"] == {"query": "[Q] ", "document": "[D] "}
