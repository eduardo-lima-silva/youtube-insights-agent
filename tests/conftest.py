"""Configuração comum dos testes."""
import pytest

from insights_agent import config


@pytest.fixture(autouse=True)
def _default_provider_anthropic(monkeypatch):
    """O app usa o Groq por padrão; os testes do loop principal exercitam o caminho Anthropic.

    Os testes do Groq (test_groq_agent.py) trocam o provedor explicitamente.
    """
    monkeypatch.setattr(config, "PROVIDER", "anthropic")
