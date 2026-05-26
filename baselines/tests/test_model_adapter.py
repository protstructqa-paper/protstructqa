"""TDD tests for baselines/model_adapter.py.

We mock urllib so the tests don't need a real vLLM server.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0,
    "."
)

from baselines import model_adapter as ma


# --------------------------- vLLM HTTP --------------------------- #


def _fake_urlopen_response(payload: dict):
    """Return a fake context manager that yields a fake response body."""
    body = json.dumps(payload).encode("utf-8")
    fake_resp = MagicMock()
    fake_resp.read.return_value = body
    cm = MagicMock()
    cm.__enter__.return_value = fake_resp
    cm.__exit__.return_value = False
    return cm


def test_vllm_generate_happy_path():
    adapter = ma.VLLMHTTPAdapter(url="http://x:8000", model="test-model")
    fake_payload = {
        "choices": [{"text": "mean_plddt(range(1, 50))",
                       "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
    }
    with patch("baselines.model_adapter.urllib.request.urlopen",
                 return_value=_fake_urlopen_response(fake_payload)):
        out = adapter.generate("test prompt", max_tokens=50)
    assert out["text"] == "mean_plddt(range(1, 50))"
    assert out["n_input_tokens"] == 100
    assert out["n_output_tokens"] == 12
    assert out["stop_reason"] == "stop"
    assert out["elapsed_s"] >= 0


def test_vllm_generate_handles_url_error():
    import urllib.error
    adapter = ma.VLLMHTTPAdapter(url="http://x:8000", model="m")
    with patch("baselines.model_adapter.urllib.request.urlopen",
                 side_effect=urllib.error.URLError("dns")):
        out = adapter.generate("test")
    assert out["text"] == ""
    assert out["stop_reason"] == "error"
    assert "error" in out


def test_vllm_generate_handles_bad_json():
    adapter = ma.VLLMHTTPAdapter(url="http://x:8000", model="m")
    fake_resp = MagicMock()
    fake_resp.read.return_value = b"not json"
    cm = MagicMock(); cm.__enter__.return_value = fake_resp
    cm.__exit__.return_value = False
    with patch("baselines.model_adapter.urllib.request.urlopen",
                 return_value=cm):
        out = adapter.generate("test")
    assert out["text"] == ""
    assert "error" in out


def test_vllm_passes_stop_list():
    """Verify stop strings are forwarded to the request payload."""
    adapter = ma.VLLMHTTPAdapter(url="http://x:8000", model="m")
    captured = {}
    def fake_open(req, timeout):
        captured["data"] = json.loads(req.data.decode("utf-8"))
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"choices": [{"text": "ok", "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        ).encode("utf-8")
        cm = MagicMock(); cm.__enter__.return_value = fake_resp
        cm.__exit__.return_value = False
        return cm
    with patch("baselines.model_adapter.urllib.request.urlopen",
                 side_effect=fake_open):
        adapter.generate("p", stop=["</s>", "\n\n"])
    assert captured["data"]["stop"] == ["</s>", "\n\n"]


# --------------------------- factory --------------------------- #


def test_make_adapter_vllm():
    a = ma.make_adapter("vllm-http", url="http://x:8000", model="m")
    assert isinstance(a, ma.VLLMHTTPAdapter)


def test_make_adapter_hf():
    a = ma.make_adapter("hf-transformers", model_path="/x")
    assert isinstance(a, ma.HFTransformersAdapter)


def test_make_adapter_unknown():
    with pytest.raises(ValueError):
        ma.make_adapter("magic-llm")
