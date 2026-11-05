"""Testes offline do provedor gratuito (Groq), com um cliente falso no formato OpenAI."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from insights_agent import agent, config, groq_agent
from insights_agent.trace import TraceStep


def _call(id_, name, args):
    return NS(id=id_, type="function", function=NS(name=name, arguments=json.dumps(args)))


def _resp(content=None, calls=None, finish="stop", pt=1000, ct=100):
    msg = NS(content=content, tool_calls=calls, reasoning=None)
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=pt, completion_tokens=ct))


class RateLimitError(Exception):
    """Imita groq.RateLimitError (mesmo nome de classe, status 429, header retry-after)."""
    status_code = 429

    def __init__(self, wait="3"):
        super().__init__("rate limited")
        self.response = NS(headers={"retry-after": wait})


class FakeGroq:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def offline(monkeypatch):
    def fake_dispatch(name, inp, state, *, max_chars=None):
        state.setdefault("n", 0)
        state["n"] += 1
        ok = name != "make_chart_fail"
        text = "| n | mediana |\n|---|---|\n| 1520 | 63412 |"
        step = TraceStep(kind="tool", name=name, input=inp or {}, output_preview=text, ok=ok,
                         extra={"result_text": text, "max_chars": max_chars})
        return f'<resultado_ferramenta fonte="{name}" confiavel="nao">\n{text}\n</resultado_ferramenta>', step, None

    monkeypatch.setattr(groq_agent.tools, "dispatch", fake_dispatch)
    monkeypatch.setattr(groq_agent, "_sleep", lambda s: None)
    monkeypatch.setattr(config, "PROVIDER", "groq")


def test_tool_call_then_answer_openai_format(offline):
    client = FakeGroq([
        _resp(calls=[_call("c1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], finish="tool_calls"),
        _resp(content="A mediana foi 63 mil (n = 1520)."),
    ])
    r = agent.run_agent("Mediana no Brasil?", client=client)   # despacha para o Groq via config.PROVIDER
    assert r.status == "ok" and "63 mil" in r.answer
    assert r.model == f"groq:{config.GROQ_MODEL}"
    assert [s.kind for s in r.steps] == ["llm", "tool", "llm", "guard"]
    assert r.cost_usd == 0.0 and r.usage.input_tokens == 2000

    first = client.calls[0]
    assert first["model"] == config.GROQ_MODEL and first["tool_choice"] == "auto"
    assert first["messages"][0]["role"] == "system"
    assert {t["type"] for t in first["tools"]} == {"function"}
    second = client.calls[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert 'confiavel="nao"' in second[-1]["content"]


def test_compact_prompt_is_smaller_and_deterministic():
    from insights_agent import prompts
    a, b = groq_agent.compact_system_prompt(), groq_agent.compact_system_prompt.__wrapped__()
    assert a == b
    assert len(a) < len(prompts.get_system_prompt()) * 0.7
    tools_json = json.dumps(groq_agent.openai_tools(), ensure_ascii=False)
    assert '"Brazil", "Other"' not in tools_json          # enum longo de países removido
    for t in groq_agent.openai_tools():
        assert t["type"] == "function" and {"name", "parameters"} <= set(t["function"])


def test_rate_limit_waits_and_retries(offline):
    client = FakeGroq([RateLimitError("3"), _resp(content="Sem números aqui.")])
    r = groq_agent.run_agent("Oi?", client=client)
    assert r.status == "ok"
    waits = [s for s in r.steps if s.name == "rate_limit_wait"]
    assert len(waits) == 1 and waits[0].extra["wait_s"] == 3.0


def test_rate_limit_gives_up_after_retries(offline, monkeypatch):
    monkeypatch.setattr(config, "GROQ_MAX_RETRIES_429", 1)
    client = FakeGroq([RateLimitError("2"), RateLimitError("2")])
    r = groq_agent.run_agent("Oi?", client=client)
    assert r.status == "error" and "Groq" in r.answer


def test_tool_results_use_smaller_limit(offline):
    client = FakeGroq([
        _resp(calls=[_call("c1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], finish="tool_calls"),
        _resp(content="Pronto: n = 1520."),
    ])
    r = groq_agent.run_agent("Teste", client=client)
    tool_step = next(s for s in r.steps if s.kind == "tool")
    assert tool_step.extra["max_chars"] == config.GROQ_TOOL_RESULT_MAX_CHARS


def test_invalid_json_arguments_become_tool_error(offline, monkeypatch):
    seen = {}

    def dispatch(name, inp, state, *, max_chars=None):
        seen["inp"] = inp
        return "erro", TraceStep(kind="tool", name=name, ok=False, extra={"result_text": "erro"}), None

    monkeypatch.setattr(groq_agent.tools, "dispatch", dispatch)
    bad = NS(id="c1", type="function", function=NS(name="run_sql", arguments="{nao é json"))
    client = FakeGroq([_resp(calls=[bad], finish="tool_calls"), _resp(content="Não consegui consultar.")])
    r = groq_agent.run_agent("Teste", client=client)
    assert seen["inp"] is None and r.status == "ok"


def test_max_tool_calls_forces_finish(offline, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 1)
    client = FakeGroq([
        _resp(calls=[_call("c1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], finish="tool_calls"),
        _resp(content="Resposta final com n = 1520."),
    ])
    r = groq_agent.run_agent("Teste", client=client)
    assert r.status == "ok"
    assert client.calls[1]["tool_choice"] == "none"


def test_missing_key_is_friendly(offline, monkeypatch):
    def boom():
        raise TypeError("The api_key client option must be set")
    monkeypatch.setattr(groq_agent, "_make_client", boom)
    r = groq_agent.run_agent("Teste")
    assert r.status == "error" and "GROQ_API_KEY" in r.answer


class TooLarge(Exception):
    """Imita o 413 do Groq (requisição grande demais para o plano gratuito)."""
    status_code = 413


def test_fit_context_shrinks_oldest_tool_results_first():
    big = '<resultado_ferramenta fonte="run_sql" confiavel="nao">\n' + "x" * 9000 + "\n</resultado_ferramenta>"
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "a", "content": big},
            {"role": "tool", "tool_call_id": "b", "content": big}]
    n = groq_agent.fit_context(msgs, tools_chars=0, budget=3500)
    assert n == 1                                   # o mais antigo basta para caber
    assert groq_agent.TRIMMED_NOTE in msgs[2]["content"] and msgs[3]["content"] == big
    assert msgs[2]["content"].endswith("</resultado_ferramenta>")
    assert groq_agent.estimate_tokens(msgs) <= 3500


def test_context_is_trimmed_before_request(offline, monkeypatch):
    monkeypatch.setattr(config, "GROQ_CONTEXT_BUDGET", 2500)

    def big_dispatch(name, inp, state, *, max_chars=None):
        text = "x" * 6000
        return (f'<resultado_ferramenta fonte="{name}" confiavel="nao">\n{text}\n</resultado_ferramenta>',
                TraceStep(kind="tool", name=name, ok=True, extra={"result_text": text}), None)

    monkeypatch.setattr(groq_agent.tools, "dispatch", big_dispatch)
    client = FakeGroq([
        _resp(calls=[_call("c1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], finish="tool_calls"),
        _resp(calls=[_call("c2", "run_sql", {"sql": "SELECT 2", "purpose": "y"})], finish="tool_calls"),
        _resp(content="Pronto."),
    ])
    r = groq_agent.run_agent("Palavras mais frequentes?", client=client)
    assert r.status == "ok"
    assert any(s.name == "context_budget" for s in r.steps)
    last_msgs = client.calls[-1]["messages"]
    assert groq_agent.TRIMMED_NOTE.strip()[:20] in json.dumps(last_msgs, ensure_ascii=False)


def test_413_compacts_and_retries(offline, monkeypatch):
    def big_dispatch(name, inp, state, *, max_chars=None):
        text = "y" * 4000
        return (f'<resultado_ferramenta fonte="{name}" confiavel="nao">\n{text}\n</resultado_ferramenta>',
                TraceStep(kind="tool", name=name, ok=True, extra={"result_text": text}), None)

    monkeypatch.setattr(groq_agent.tools, "dispatch", big_dispatch)
    client = FakeGroq([
        _resp(calls=[_call("c1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], finish="tool_calls"),
        TooLarge("Request too large"),
        _resp(content="Resposta final."),
    ])
    r = groq_agent.run_agent("Teste 413", client=client)
    assert r.status == "ok" and r.answer == "Resposta final."
    assert any(s.name == "context_budget" and s.extra.get("after_413") for s in r.steps)
