"""Tests for saliency model selection - config logic only, no model download."""

import pytest

from recompose.perception.saliency import MODEL_URLS, saliency_model_name


def test_default_is_the_lightweight_model(monkeypatch):
    monkeypatch.delenv("RECOMPOSE_SALIENCY_MODEL", raising=False)
    assert saliency_model_name() == "u2netp"


def test_env_var_selects_full_model(monkeypatch):
    monkeypatch.setenv("RECOMPOSE_SALIENCY_MODEL", "u2net")
    assert saliency_model_name() == "u2net"


def test_unknown_model_name_rejected_loudly(monkeypatch):
    monkeypatch.setenv("RECOMPOSE_SALIENCY_MODEL", "u3net-ultra")
    with pytest.raises(ValueError, match="u3net-ultra"):
        saliency_model_name()


def test_every_selectable_model_has_a_pinned_url():
    assert set(MODEL_URLS) == {"u2net", "u2netp"}
    assert all(url.startswith("https://") for url in MODEL_URLS.values())
