"""Loop do agente: Claude orquestra run_sql / make_chart / score_titles.

Pontos de design (Claude API, Opus 5.5):
- loop manual com ``client.beta.messages.create`` + beta ``server-side-fallback-2026-07-01``
  e ``fallbacks="default"`` (recusas de classificador são refeitas no servidor);
- thinking é adaptativo por padrão neste modelo: não enviamos ``thinking``; a
  profundidade é controlada por ``output_config={"effort": ...}``;
- ``tool_choice`` fica no padrão ``auto`` (forçar ferramenta dá 400 no Opus 5.5);
- ``cache_control`` no topo (cache automático); system + tools são byte-estáveis;
- ``stop_reason`` é verificado antes de ler o conteúdo; o ``response.content``
  inteiro (thinking incluído) volta como mensagem do assistente;
- todos os ``tool_result`` de um turno vão numa única mensagem ``user``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import anthropic

from insights_agent import config, grounding, prompts, tools
from insights_agent.trace import AgentResult, TraceStep, Usage

BETAS = ["server-side-fallback-2026-07-01"]
# Modelos com fallback no servidor (`fallbacks="default"`). O Claude Haiku 5.5 não tem: nele o parâmetro é omitido.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5"}


def _fallback_kwargs(model: str) -> dict:
    return {"betas": BETAS, "fallbacks": "default"} if model in FALLBACK_MODELS else {}

# Limites (podem ir para config.py; aqui com getattr para não depender disso).
MAX_QUESTION_CHARS = getattr(config, "MAX_QUESTION_CHARS", 2000)
MAX_HISTORY_TURNS = getattr(config, "MAX_HISTORY_TURNS", 6)
MAX_HISTORY_CHARS = getattr(config, "MAX_HISTORY_CHARS", 2000)
MAX_WALL_S = getattr(config, "MAX_WALL_S", 120.0)
MAX_CONSECUTIVE_TOOL_ERRORS = getattr(config, "MAX_CONSECUTIVE_TOOL_ERRORS", 3)

_clock = time.perf_counter   # substituível nos testes

FINISH_NOTE_LIMIT = (
    "Limite de chamadas de ferramenta desta pergunta atingido. Responda agora ao usuário com o "
    "que já foi obtido, sem chamar ferramentas, deixando claro o que ficou de fora."
)
FINISH_NOTE_ERRORS = (
    "Houve {n} erros seguidos de ferramenta. Não tente de novo: responda agora ao usuário com o "
    "que já foi obtido, sem chamar ferramentas, e explique brevemente a limitação."
)
_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _scrub(text: str) -> str:
    return _KEY_RE.sub("sk-ant-***", text or "")


def compute_cost(usage: Any, model: str | None = None) -> float:
    """Custo em US$ de um objeto usage (API ou trace.Usage) com os preços de config.PRICES."""
    prices = config.PRICES.get(model or "") or config.PRICES.get(config.MODEL)
    if not prices:
        return 0.0
    p_in, p_out, p_cr, p_cw = prices
    g = lambda k: (getattr(usage, k, 0) or 0)  # noqa: E731
    return (g("input_tokens") * p_in + g("output_tokens") * p_out
            + g("cache_read_input_tokens") * p_cr + g("cache_creation_input_tokens") * p_cw) / 1e6


def _btype(block: Any) -> str:
    return getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "")


def _sanitize_content(content: list) -> list:
    """Regras de eco após fallback no meio da saída: antes do último bloco ``fallback``,
    descarta thinking/redacted_thinking/tool_use/server_tool_use (texto fica)."""
    content = list(content or [])
    idx = [i for i, b in enumerate(content) if _btype(b) == "fallback"]
    if not idx:
        return content
    last = idx[-1]
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(content) if i >= last or _btype(b) not in drop]


def _text_of(content: Iterable) -> str:
    return "".join(getattr(b, "text", "") or "" for b in content if _btype(b) == "text").strip()


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + " …[truncado]"


def _history_messages(history: list | None) -> list[dict]:
    pairs = []
    for item in history or []:
        if isinstance(item, dict):
            q, a = item.get("question"), item.get("answer")
        else:
            try:
                q, a = item
            except (TypeError, ValueError):
                continue
        if q and a:
            pairs.append((_clip(q, MAX_HISTORY_CHARS).strip(), _clip(a, MAX_HISTORY_CHARS).strip()))
    msgs: list[dict] = []
    for q, a in pairs[-MAX_HISTORY_TURNS:]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _emit(on_step: Callable | None, step: TraceStep) -> None:
    if on_step is None:
        return
    try:
        on_step(step)
    except Exception:  # noqa: BLE001 - a UI nunca derruba o agente
        pass


def _friendly_api_error(e: Exception) -> tuple[str, str]:
    """(status, mensagem amigável) para exceções do SDK."""
    if isinstance(e, anthropic.APITimeoutError):
        return "timeout", "A API da Anthropic demorou demais para responder. Tente de novo em instantes."
    if isinstance(e, anthropic.AuthenticationError):
        return "error", "A chave da Anthropic foi recusada (401). Confira a chave ou use o modo demo."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "error", "A chave da Anthropic não tem permissão para usar este modelo (403)."
    if isinstance(e, anthropic.RateLimitError):
        return "error", "Limite de uso da API atingido (429). Aguarde um minuto e tente de novo."
    if isinstance(e, anthropic.BadRequestError):
        return "error", "A API recusou a requisição (400). Veja os detalhes no rastro."
    if isinstance(e, anthropic.APIStatusError):
        code = getattr(e, "status_code", 0) or 0
        if code >= 500:
            return "error", f"A API da Anthropic está instável ou sobrecarregada ({code}). Tente de novo em instantes."
        return "error", f"A API da Anthropic devolveu um erro ({code}). Veja os detalhes no rastro."
    if isinstance(e, anthropic.APIConnectionError):
        return "error", "Não consegui conectar à API da Anthropic. Verifique a conexão com a internet."
    # sem credencial o SDK levanta TypeError("Could not resolve authentication method ...") na requisição
    if isinstance(e, anthropic.AnthropicError) or (isinstance(e, TypeError) and "authentication" in str(e)):
        return "error", ("Não há credencial da Anthropic configurada ou ela é inválida. "
                         "Informe uma chave na barra lateral ou use o modo demo.")
    return "error", "Ocorreu um erro inesperado ao falar com o modelo. Veja os detalhes no rastro."


# --------------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------------- #
def run_agent(question: str, history: list | None = None, *, client: Any = None,
              on_step: Callable[[TraceStep], None] | None = None,
              provider: str | None = None) -> AgentResult:
    """Responde uma pergunta. ``provider``: "anthropic" ou "groq" (padrão: config.PROVIDER)."""
    if (provider or config.PROVIDER) == "groq":   # provedor gratuito (ver groq_agent.py)
        from insights_agent import groq_agent
        return groq_agent.run_agent(question, history, client=client, on_step=on_step)
    t0 = _clock()
    question = (question or "").strip()
    result = AgentResult(question=question, model=config.MODEL)
    usage = Usage()
    result.usage = usage

    def add_step(step: TraceStep) -> None:
        result.steps.append(step)
        _emit(on_step, step)

    def finish(status: str, answer: str, error: str | None = None) -> AgentResult:
        result.status = status
        result.answer = answer
        result.error = _scrub(error) if error else None
        result.latency_s = round(_clock() - t0, 3)
        result.cost_usd = round(result.cost_usd, 6)
        return result

    # ---- limites de entrada ----
    if not question:
        add_step(TraceStep(kind="guard", name="input_limits", ok=False, output_preview="pergunta vazia"))
        return finish("error", "Escreva uma pergunta sobre os vídeos em alta para eu analisar.")
    if len(question) > MAX_QUESTION_CHARS:
        add_step(TraceStep(kind="guard", name="input_limits", ok=False,
                           output_preview=f"pergunta com {len(question)} caracteres",
                           extra={"chars": len(question), "max": MAX_QUESTION_CHARS}))
        return finish("error", f"A pergunta é longa demais ({len(question)} caracteres; o limite é "
                               f"{MAX_QUESTION_CHARS}). Resuma o que você quer analisar.")

    try:
        system = prompts.get_system_prompt()
        tool_defs = tools.get_tools()
        if client is None:
            client = anthropic.Anthropic()
    except Exception as e:  # noqa: BLE001
        status, msg = _friendly_api_error(e)
        if not isinstance(e, anthropic.AnthropicError):
            msg = "Não foi possível preparar o agente (catálogo ou ferramentas indisponíveis)."
        add_step(TraceStep(kind="error", name="setup", ok=False, output_preview=_scrub(str(e))[:300]))
        return finish(status, msg, f"{type(e).__name__}: {e}")

    messages: list[dict] = _history_messages(history)
    messages.append({"role": "user", "content": question})
    state = tools.new_state()          # resultados (r1, r2, ...) só desta execução
    tool_calls = 0
    consecutive_errors = 0
    finishing = False
    max_llm_calls = config.MAX_TOOL_CALLS + 4

    while True:
        elapsed = _clock() - t0
        if elapsed >= MAX_WALL_S:
            add_step(TraceStep(kind="guard", name="wall_clock", ok=False,
                               output_preview=f"{elapsed:.1f}s >= {MAX_WALL_S}s"))
            return finish("timeout", f"A análise passou do limite de {MAX_WALL_S:.0f} s e foi interrompida. "
                                     "Tente uma pergunta mais específica.")
        if usage.llm_calls >= max_llm_calls:
            return finish("max_tool_calls", "A análise usou chamadas demais ao modelo e foi interrompida.",
                          "limite de chamadas ao modelo")

        t_call = _clock()
        try:
            resp = client.beta.messages.create(
                model=config.MODEL,
                max_tokens=config.MAX_TOKENS,
                system=system,
                tools=tool_defs,
                messages=messages,
                output_config={"effort": config.EFFORT},
                cache_control={"type": "ephemeral"},
                **_fallback_kwargs(config.MODEL),
                timeout=max(5.0, MAX_WALL_S - elapsed),
            )
        except Exception as e:  # noqa: BLE001
            status, msg = _friendly_api_error(e)
            detail = f"{type(e).__name__}: {getattr(e, 'message', None) or e}"
            add_step(TraceStep(kind="error", name="anthropic_api", ok=False,
                               output_preview=_scrub(detail)[:300],
                               elapsed_ms=round((_clock() - t_call) * 1000, 1),
                               extra={"status_code": getattr(e, "status_code", None)}))
            return finish(status, msg, detail)

        served_by = getattr(resp, "model", None) or config.MODEL
        u = getattr(resp, "usage", None)
        usage.add(u)
        result.cost_usd += compute_cost(u, served_by)
        stop = getattr(resp, "stop_reason", None)
        content = _sanitize_content(getattr(resp, "content", None) or [])
        tool_uses = [b for b in content if _btype(b) == "tool_use"]
        iterations = getattr(u, "iterations", None) or []
        fallback = any(_btype(b) == "fallback" for b in content) or \
            any(getattr(it, "type", None) == "fallback_message" for it in iterations)
        stop_details = getattr(resp, "stop_details", None)
        llm_extra = {
            "stop_reason": stop, "served_by": served_by, "fallback": fallback,
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        if stop_details is not None:
            llm_extra["refusal_category"] = getattr(stop_details, "category", None)
        preview = ("ferramentas: " + ", ".join(getattr(b, "name", "?") for b in tool_uses)) if tool_uses \
            else _text_of(content)[:200]
        add_step(TraceStep(kind="llm", name=served_by, input={"call": usage.llm_calls, "messages": len(messages)},
                           output_preview=preview, ok=stop not in ("refusal", "max_tokens"),
                           elapsed_ms=round((_clock() - t_call) * 1000, 1), extra=llm_extra))
        if fallback and served_by != config.MODEL:
            result.model = served_by

        # ---- stop_reason antes de ler o conteúdo ----
        if stop == "refusal":
            return finish("refused_by_model",
                          "O modelo recusou esta solicitação pelas políticas de segurança. Reformule a "
                          "pergunta com foco na análise dos vídeos em alta.",
                          f"refusal: {llm_extra.get('refusal_category')}")
        if stop in ("max_tokens", "model_context_window_exceeded"):
            return finish("error", "A resposta ficou longa demais e foi cortada. Tente uma pergunta "
                                   "mais específica.", f"stop_reason={stop}")
        if stop == "pause_turn":
            messages.append({"role": "assistant", "content": content})
            continue

        if tool_uses:  # processa mesmo se o stop_reason vier inconsistente
            if finishing:
                partial = _text_of(content)
                answer = (partial + "\n\n" if partial else "") + \
                    "_A análise atingiu o limite de chamadas de ferramenta; a resposta pode estar incompleta._"
                return finish("max_tool_calls", answer, "o modelo continuou chamando ferramentas após o limite")

            messages.append({"role": "assistant", "content": content})
            results: list[dict] = []
            for b in tool_uses:
                if tool_calls >= config.MAX_TOOL_CALLS:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                    "content": tools.wrap_untrusted(
                                        b.name, "Não executada: limite de chamadas de ferramenta atingido.")})
                    continue
                text, step, spec = tools.dispatch(b.name, getattr(b, "input", None), state)
                tool_calls += 1
                add_step(step)
                if spec is not None:
                    result.charts.append(spec)
                consecutive_errors = 0 if step.ok else consecutive_errors + 1
                block = {"type": "tool_result", "tool_use_id": b.id, "content": text}
                if not step.ok:
                    block["is_error"] = True
                results.append(block)

            if tool_calls >= config.MAX_TOOL_CALLS:
                finishing = True
                results.append({"type": "text", "text": FINISH_NOTE_LIMIT})
            elif consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                finishing = True
                results.append({"type": "text", "text": FINISH_NOTE_ERRORS.format(n=consecutive_errors)})
            messages.append({"role": "user", "content": results})
            continue

        # ---- resposta final (end_turn / stop_sequence / outros) ----
        answer = _text_of(content)
        if not answer:
            return finish("error", "O modelo não devolveu texto. Tente de novo.", f"stop_reason={stop}")
        g = grounding.check_numbers(answer, result.steps, question=question)
        add_step(TraceStep(kind="guard", name="grounding", ok=not g["unverified_numbers"],
                           output_preview=(f"{g['checked']} números checados; "
                                           f"{len(g['unverified_numbers'])} não verificados"),
                           extra=g))
        return finish("ok", answer)


# --------------------------------------------------------------------------- #
# Persistência do rastro (modo demo / evals)
# --------------------------------------------------------------------------- #
def save_trace(result: AgentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = result.to_dict()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def load_trace(path: str | Path) -> AgentResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AgentResult.from_dict(data)


def unverified_numbers(result: AgentResult) -> list[str]:
    """Números marcados pela checagem de grounding (lista vazia se não houver checagem)."""
    for s in result.steps:
        if s.kind == "guard" and s.name == "grounding":
            return list(s.extra.get("unverified_numbers") or [])
    return []
