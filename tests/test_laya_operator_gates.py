"""Operator gates are tested without network or model requests."""

import asyncio

import pytest

from scripts import check_laya_remote, evaluate_laya_value


@pytest.mark.parametrize("module", [check_laya_remote, evaluate_laya_value])
def test_remote_cli_requires_explicit_test_approval(monkeypatch, module):
    monkeypatch.setenv("LAYA_BASE_URL", "http://100.64.0.10:8000")
    monkeypatch.setenv("LAYA_APPROVED_ORIGIN", "http://100.64.0.10:8000")
    monkeypatch.delenv("LAYA_REMOTE_TEST_APPROVED", raising=False)
    calls = []

    def unexpected_client(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("remote client created before approval")

    monkeypatch.setattr(module, "LayaClient", unexpected_client)
    assert asyncio.run(module.main()) == 2
    assert calls == []
