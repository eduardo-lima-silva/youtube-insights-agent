"""Loop do agente com cliente FALSO (sem rede, sem warehouse, sem modelos)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insights_agent import agent, config, tools  # noqa: E402
from insights_agent.trace import AgentResult, TraceStep  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def usage(i=100, o=50, cr=0, cw=0):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=cr, cache_creation_input_tokens=cw)


def text(t):
    return NS(type="text", text=t)


def thinking():
    return NS(type="thinking", thinking="", signature="sig")


def tool_use(id_, name, inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def resp(content, stop, u=None, model="claude-opus-5-5", stop_details=None):
    return NS(content=content, stop_reason=stop, usage=u or usage(), model=model, stop_details=stop_details)


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        # guarda uma cópia rasa das mensagens (a lista é mutada pelo loop depois)
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.beta = NS(messages=FakeMessages(script))

    @property
    def calls(self):
        return self.beta.messages.calls


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Prompt e tools estáveis sem catálogo/predictor reais; dispatch falso."""
    monkeypatch.setattr(agent.prompts, "get_system_prompt", lambda: "SYSTEM PROMPT DE TESTE")
    monkeypatch.setattr(agent.tools, "get_tools", lambda: [{"name": "run_sql"}])
    calls = []

    def fake_dispatch(name, inp, state):
        calls.append((name, inp))
        if inp.get("fail"):
            step = TraceStep(kind="tool", name=name, input=inp, ok=False,
                             extra={"result_text": "Erro: coluna inexistente"})
            return tools.wrap_untrusted(name, "Erro: coluna inexistente"), step, None
        rid = f"r{state['next_id']}"
        state["next_id"] += 1
        state["results"][rid] = object()
        out = f"result_id: {rid}\nlinhas: 2\n| grupo | mediana | n |\n|---|---|---|\n| 8-12 | 63412 | 1520 |\n| 4-8 | 41000 | 2210 |"
        step = TraceStep(kind="tool", name=name, input=inp, ok=True,
                         extra={"result_text": out, "row_count": 2, "sql_validated": inp.get("sql")})
        spec = {"mark": "bar"} if name == "make_chart" else None
        return tools.wrap_untrusted(name, out), step, spec

    monkeypatch.setattr(agent.tools, "dispatch", fake_dispatch)
    return calls


# --------------------------------------------------------------------------- #
# Casos
# --------------------------------------------------------------------------- #
def test_single_tool_call_then_answer(offline):
    client = FakeClient([
        resp([thinking(), tool_use("t1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], "tool_use"),
        resp([text("A mediana de 8-12 min foi 63 mil (n = 1520) contra 41 mil (n = 2210).")], "end_turn"),
    ])
    seen = []
    r = agent.run_agent("Compare durações no Brasil", client=client, on_step=seen.append)
    assert r.status == "ok"
    assert "63 mil" in r.answer
    assert [s.kind for s in r.steps] == ["llm", "tool", "llm", "guard"]
    assert seen == r.steps                      # on_step recebeu cada passo
    assert r.usage.llm_calls == 2

    first = client.calls[0]
    assert first["model"] == config.MODEL
    assert first["output_config"] == {"effort": config.EFFORT}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["fallbacks"] == "default"
    assert "thinking" not in first and "tool_choice" not in first

    msgs = client.calls[1]["messages"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0].type == "thinking"     # conteúdo completo devolvido, thinking incluso
    tr = msgs[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "t1" and "is_error" not in tr
    assert tr["content"].startswith('<resultado_ferramenta fonte="run_sql" confiavel="nao">')

    g = r.steps[-1]
    assert g.name == "grounding" and g.extra["unverified_numbers"] == []


def test_parallel_tool_uses_one_user_message(offline):
    client = FakeClient([
        resp([tool_use("a", "run_sql", {"sql": "SELECT 1", "purpose": "a"}),
              tool_use("b", "run_sql", {"sql": "SELECT 2", "purpose": "b"})], "tool_use"),
        resp([text("Pronto.")], "end_turn"),
    ])
    r = agent.run_agent("duas consultas", client=client)
    assert r.status == "ok"
    msgs = client.calls[1]["messages"]
    assert len(msgs) == 3                                  # user, assistant, user(tool_results)
    results = msgs[2]["content"]
    assert [b["tool_use_id"] for b in results] == ["a", "b"]
    assert all(b["type"] == "tool_result" for b in results)


def test_tool_error_is_flagged_and_loop_continues(offline):
    client = FakeClient([
        resp([tool_use("e1", "run_sql", {"sql": "SELECT x", "purpose": "a", "fail": True})], "tool_use"),
        resp([tool_use("ok1", "run_sql", {"sql": "SELECT 1", "purpose": "b"})], "tool_use"),
        resp([text("Feito.")], "end_turn"),
    ])
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "ok"
    err_block = client.calls[1]["messages"][2]["content"][0]
    assert err_block["is_error"] is True
    assert len(client.calls) == 3
    assert [s.ok for s in r.steps if s.kind == "tool"] == [False, True]


def test_max_tool_calls_limit(offline, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 2)
    loop = lambda i: resp([tool_use(f"t{i}", "run_sql", {"sql": "SELECT 1", "purpose": "p"})], "tool_use")  # noqa: E731
    client = FakeClient([loop(1), loop(2), loop(3)])
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "max_tool_calls"
    assert len(offline) == 2                               # só 2 ferramentas executadas
    last_user = client.calls[2]["messages"][-1]["content"]
    assert last_user[-1]["type"] == "text"                 # nota pedindo para finalizar
    assert last_user[0]["type"] == "tool_result"           # tool_results vêm antes do texto


def test_max_tool_calls_then_model_finishes(offline, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 1)
    client = FakeClient([
        resp([tool_use("t1", "run_sql", {"sql": "SELECT 1", "purpose": "p"}),
              tool_use("t2", "run_sql", {"sql": "SELECT 2", "purpose": "p"})], "tool_use"),
        resp([text("Resposta com o que deu.")], "end_turn"),
    ])
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "ok"
    blocks = client.calls[1]["messages"][-1]["content"]
    assert [b.get("is_error", False) for b in blocks if b["type"] == "tool_result"] == [False, True]


def test_consecutive_tool_errors_force_finish(offline):
    bad = lambda i: tool_use(f"e{i}", "run_sql", {"sql": "x", "purpose": "p", "fail": True})  # noqa: E731
    client = FakeClient([
        resp([bad(1), bad(2), bad(3)], "tool_use"),
        resp([text("Não consegui consultar a base.")], "end_turn"),
    ])
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "ok"
    note = client.calls[1]["messages"][-1]["content"][-1]
    assert note["type"] == "text" and "erros seguidos" in note["text"]


def test_refusal(offline):
    client = FakeClient([resp([], "refusal", stop_details=NS(category="cyber", explanation=None))])
    r = agent.run_agent("algo", client=client)
    assert r.status == "refused_by_model"
    assert r.answer
    assert r.steps[0].extra["refusal_category"] == "cyber"


def test_max_tokens_is_friendly_error(offline):
    client = FakeClient([resp([text("cortad")], "max_tokens")])
    r = agent.run_agent("algo", client=client)
    assert r.status == "error" and "longa demais" in r.answer


def test_cost_computation(offline):
    client = FakeClient([
        resp([tool_use("t1", "run_sql", {"sql": "SELECT 1", "purpose": "p"})], "tool_use",
             u=usage(1000, 200, 0, 5000)),
        resp([text("Fim.")], "end_turn", u=usage(300, 400, 5000, 0)),
    ])
    r = agent.run_agent("custo", client=client)
    p_in, p_out, p_cr, p_cw = config.PRICES[config.MODEL]
    expected = (1300 * p_in + 600 * p_out + 5000 * p_cr + 5000 * p_cw) / 1e6
    assert r.cost_usd == pytest.approx(expected, abs=1e-6)
    assert r.usage.input_tokens == 1300 and r.usage.cache_read_input_tokens == 5000
    assert agent.compute_cost(r.usage, config.MODEL) == pytest.approx(expected)


def test_api_errors_are_friendly(offline):
    import anthropic
    import httpx2 as httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = [
        (anthropic.RateLimitError("rate", response=httpx.Response(429, request=req), body=None), "error", "429"),
        (anthropic.AuthenticationError("bad sk-ant-abc123", response=httpx.Response(401, request=req), body=None),
         "error", "401"),
        (anthropic.APIConnectionError(request=req), "error", "conectar"),
        (anthropic.APITimeoutError(request=req), "timeout", "demorou"),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=req), body=None),
         "error", "500"),
        (TypeError("Could not resolve authentication method. Expected one of api_key"), "error", "credencial"),
    ]
    for exc, status, needle in cases:
        r = agent.run_agent("pergunta", client=FakeClient([exc]))
        assert r.status == status, exc
        assert needle in r.answer
        assert "sk-ant-abc123" not in (r.error or "")


def test_question_and_history_limits(offline):
    long_q = "x" * (agent.MAX_QUESTION_CHARS + 1)
    client = FakeClient([])
    r = agent.run_agent(long_q, client=client)
    assert r.status == "error" and not client.calls

    hist = [(f"q{i}", "a" * 5000) for i in range(10)]
    client = FakeClient([resp([text("ok")], "end_turn")])
    agent.run_agent("nova pergunta", history=hist, client=client)
    msgs = client.calls[0]["messages"]
    assert len(msgs) == 2 * agent.MAX_HISTORY_TURNS + 1
    assert msgs[0]["content"] == "q4"                      # só as 6 últimas
    assert all(len(m["content"]) <= agent.MAX_HISTORY_CHARS + 20 for m in msgs)


def test_wall_clock_timeout(offline, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(agent, "_clock", lambda: clock["t"])

    class SlowMessages(FakeMessages):
        def create(self, **kw):
            clock["t"] += 200.0
            return super().create(**kw)

    client = FakeClient([resp([tool_use("t1", "run_sql", {"sql": "SELECT 1", "purpose": "p"})], "tool_use")])
    client.beta = NS(messages=SlowMessages(client.beta.messages.script))
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "timeout"


def test_fallback_content_is_sanitized(offline):
    content = [thinking(), tool_use("x", "run_sql", {"sql": "SELECT 1", "purpose": "p"}),
               NS(type="fallback"), text("Resposta do modelo de fallback.")]
    client = FakeClient([resp(content, "end_turn", model="claude-opus-4-8")])
    r = agent.run_agent("pergunta", client=client)
    assert r.status == "ok" and r.answer == "Resposta do modelo de fallback."
    assert r.steps[0].extra["fallback"] is True
    assert r.model == "claude-opus-4-8"


def test_state_is_isolated_between_runs(offline):
    states = []
    orig = agent.tools.new_state

    def spy():
        s = orig()
        states.append(s)
        return s

    agent.tools.new_state = spy
    try:
        for _ in range(2):
            agent.run_agent("p", client=FakeClient([
                resp([tool_use("t", "run_sql", {"sql": "SELECT 1", "purpose": "p"})], "tool_use"),
                resp([text("ok")], "end_turn")]))
    finally:
        agent.tools.new_state = orig
    assert states[0] is not states[1]
    assert list(states[1]["results"]) == ["r1"]            # cada execução recomeça em r1


def test_save_and_load_trace(offline):
    import tempfile
    client = FakeClient([
        resp([tool_use("t1", "make_chart", {"result_id": "r1"})], "tool_use"),
        resp([text("Fim.")], "end_turn"),
    ])
    r = agent.run_agent("trace", client=client)
    with tempfile.TemporaryDirectory() as d:   # sem tmp_path: o basetemp do pytest pode estar bloqueado
        p = agent.save_trace(r, Path(d) / "sub" / "t.json")
        back = agent.load_trace(p)
    assert isinstance(back, AgentResult)
    assert back.to_dict() == r.to_dict()
    assert back.charts == [{"mark": "bar"}]


def test_tool_use_with_inconsistent_stop_reason_is_processed(offline):
    """Revisão do Codex: blocos tool_use com stop_reason != 'tool_use' não podem ser ignorados."""
    client = FakeClient([
        resp([tool_use("t1", "run_sql", {"sql": "SELECT 1", "purpose": "x"})], "end_turn"),
        resp([text("Resultado: 1.")], "end_turn"),
    ])
    r = agent.run_agent("Teste", client=client)
    assert r.status == "ok"
    assert [s.kind for s in r.steps][:2] == ["llm", "tool"]
    assert client.calls[1]["messages"][2]["content"][0]["type"] == "tool_result"


def test_haiku_omits_fallbacks(offline, monkeypatch):
    """Claude Haiku 5.5 não tem fallback no servidor: o parâmetro não deve ser enviado."""
    monkeypatch.setattr(config, "MODEL", "claude-haiku-5-5")
    client = FakeClient([resp([text("Resposta.")], "end_turn", model="claude-haiku-5-5")])
    r = agent.run_agent("Teste", client=client)
    assert r.status == "ok"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in call and "betas" not in call
