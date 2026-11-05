"""UI do YouTube Insights Agent (Streamlit).

Rodar:  streamlit run app/streamlit_app.py

- Modo "Ao vivo": precisa de credencial da Anthropic (ANTHROPIC_API_KEY, st.secrets ou a
  chave digitada na barra lateral, guardada só em st.session_state).
- Modo "Demo": reproduz execuções gravadas em demo_traces/*.json, sem chamar a API.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import streamlit as st  # noqa: E402

from insights_agent import config  # noqa: E402
from insights_agent.agent import load_trace, run_agent, unverified_numbers  # noqa: E402
from insights_agent.trace import AgentResult, TraceStep  # noqa: E402

MAX_LIVE_ANSWERS = 20
CAVEAT_STRIP = "Base com apenas vídeos que entraram em alta · associação ≠ causalidade · out/2024–mai/2026"
EXAMPLES = [
    ("Análise", "Compare a mediana de views de vídeos em alta de 8 a 12 minutos contra vídeos de 4 a 8 "
                "minutos no Brasil."),
    ("Causal", "Qual duração causa mais views?"),
    ("Títulos", "Quero criar um vídeo sobre como estudar Python para entrevistas de dados. Sugira 3 títulos."),
    ("Por país", "Quais categorias mais aparecem em alta no Brasil nos últimos 90 dias da base?"),
]
STATUS_PT = {"ok": "concluída", "refused_by_model": "recusada pelo modelo",
             "max_tool_calls": "limite de ferramentas", "timeout": "tempo esgotado", "error": "erro"}

st.set_page_config(page_title="YouTube Insights Agent", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
_WIN_PATH = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_NIX_PATH = re.compile(r"(?<![\w.])/(?:Users|home|tmp|var|mnt|opt|root|private)/[^\s'\"<>|]*")


def strip_paths(text: str) -> str:
    """Troca caminhos absolutos locais pelo nome do arquivo (nada de C:\\Users\\... na UI)."""
    def base(m: re.Match) -> str:
        return re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1] or "…"
    return _NIX_PATH.sub(base, _WIN_PATH.sub(base, str(text)))


def _safe_json(obj) -> str:
    return strip_paths(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _secret_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[no-any-return]
    except Exception:  # sem secrets.toml
        return None


def _user_key() -> str:
    return (st.session_state.get("user_api_key") or "").strip()


def has_credentials() -> bool:
    return bool(_user_key() or _secret_key() or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def make_client():
    import anthropic
    key = _user_key() or _secret_key()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def warehouse_info() -> dict:
    wh = Path(config.WAREHOUSE_PATH)
    info = {"arquivo": wh.name}
    for name in (("manifest_sample.json",) if "sample" in wh.name else ()) + ("manifest.json",):
        mf = wh.parent / name
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("warehouse") and data["warehouse"] != wh.name:
            continue
        info["gerado em (UTC)"] = data.get("built_at_utc")
        if data.get("row_counts"):
            info["linhas"] = data["row_counts"]
        break
    return info


@st.cache_resource(show_spinner=False)
def model_info() -> dict | None:
    try:
        from insights_agent import predictor
        mi = getattr(predictor, "MODEL_INFO", None)
        return json.loads(strip_paths(json.dumps(mi, default=str))) if mi else None
    except Exception:
        return None


def list_demo_traces() -> dict[str, AgentResult]:
    out: dict[str, AgentResult] = {}
    for p in sorted(Path(config.DEMO_TRACES_DIR).glob("*.json")):
        try:
            out[p.name] = load_trace(p)
        except Exception:  # arquivo inválido: ignora
            continue
    return out


# --------------------------------------------------------------------------- #
# Renderização
# --------------------------------------------------------------------------- #
def describe_step(step: TraceStep) -> str:
    icon = "✅" if step.ok else "⚠️"
    if step.kind == "llm":
        sr = step.extra.get("stop_reason")
        return f"{icon} Modelo respondeu ({sr}, {step.elapsed_ms / 1000:.1f}s)"
    if step.kind == "tool":
        extra = step.extra
        if step.name == "run_sql":
            if step.ok:
                return f"{icon} SQL executado: {extra.get('row_count')} linhas em {extra.get('elapsed_ms')} ms"
            return f"{icon} SQL com erro ou rejeitado (o modelo vai corrigir)"
        if step.name == "make_chart":
            return f"{icon} Gráfico {'criado' if step.ok else 'com erro'}"
        if step.name == "score_titles":
            return f"{icon} Títulos pontuados pelo modelo" if step.ok else f"{icon} Erro ao pontuar títulos"
    if step.kind == "guard" and step.name == "grounding":
        n = len(step.extra.get("unverified_numbers") or [])
        return f"{icon} Checagem de números: {n} não verificados"
    return f"{icon} {step.kind}: {step.name}"


def render_chart(spec: dict, key: str) -> None:
    try:
        st.altair_chart(alt.Chart.from_dict(spec), width="stretch", key=key)
    except Exception:
        try:
            st.vega_lite_chart(spec=spec, width="stretch", key=key + "_vl")
        except Exception as e:  # noqa: BLE001
            st.caption(f"Não foi possível desenhar o gráfico ({type(e).__name__}).")


def render_trace(result: AgentResult) -> None:
    u = result.usage
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latência", f"{result.latency_s:.1f} s")
    c2.metric("Custo", f"US$ {result.cost_usd:.4f}")
    c3.metric("Tokens entrada", f"{u.input_tokens:,}".replace(",", "."))
    c4.metric("Tokens saída", f"{u.output_tokens:,}".replace(",", "."))
    c5.metric("Cache (leitura / escrita)", f"{u.cache_read_input_tokens} / {u.cache_creation_input_tokens}")
    st.caption(f"Modelo: {result.model or config.MODEL} · chamadas ao modelo: {u.llm_calls} · "
               f"status: {STATUS_PT.get(result.status, result.status)}")
    if result.error:
        st.caption("Detalhe técnico: " + strip_paths(result.error))

    for i, step in enumerate(result.steps, 1):
        label = (f"{i}. [{step.kind}] {step.name} · {'ok' if step.ok else 'falhou'} · "
                 f"{step.elapsed_ms:.0f} ms")
        with st.expander(strip_paths(label), expanded=False):
            extra = dict(step.extra or {})
            if step.kind == "tool":
                if extra.get("purpose"):
                    st.markdown(f"**Objetivo:** {strip_paths(extra['purpose'])}")
                if step.name == "run_sql":
                    st.code(strip_paths(extra.get("sql_validated") or step.input.get("sql", "")), language="sql")
                    if "row_count" in extra:
                        st.markdown(f"**Linhas:** {extra.get('row_count')}"
                                    + (" (truncado)" if extra.get("truncated") else "")
                                    + f" · **tempo da consulta:** {extra.get('elapsed_ms')} ms"
                                    + f" · **result_id:** `{extra.get('result_id')}`")
                else:
                    st.markdown("**Entrada:**")
                    st.code(_safe_json(step.input), language="json")
                if step.name == "score_titles" and extra.get("model_info"):
                    st.markdown("**Versão do modelo:**")
                    st.code(_safe_json(extra["model_info"]), language="json")
                st.markdown("**Resultado enviado ao modelo:**")
                st.text(strip_paths(extra.get("result_text") or step.output_preview))
            elif step.kind == "llm":
                st.markdown(
                    f"**stop_reason:** `{extra.get('stop_reason')}` · **servido por:** "
                    f"`{extra.get('served_by')}`" + (" (fallback)" if extra.get("fallback") else "")
                    + f" · **tokens:** entrada {extra.get('input_tokens')}, saída "
                      f"{extra.get('output_tokens')}, cache leitura {extra.get('cache_read_input_tokens')}, "
                      f"cache escrita {extra.get('cache_creation_input_tokens')}")
                if step.output_preview:
                    st.text(strip_paths(step.output_preview))
            elif step.kind == "guard" and step.name == "grounding":
                st.markdown(f"**Números checados:** {extra.get('checked')} · verificados: "
                            f"{extra.get('verified')} · ignorados (anos, ordinais, pergunta): "
                            f"{extra.get('ignored')}")
                if extra.get("unverified_numbers"):
                    st.markdown("**Não encontrados nas saídas das ferramentas:** "
                                + ", ".join(f"`{n}`" for n in extra["unverified_numbers"]))
            else:
                st.text(strip_paths(step.output_preview))
                if extra:
                    st.code(_safe_json(extra), language="json")


def render_result(result: AgentResult, key: str) -> None:
    st.markdown(f"**Pergunta:** {result.question}")
    if result.status != "ok":
        st.warning(f"Status: {STATUS_PT.get(result.status, result.status)}")
    flagged = unverified_numbers(result)
    tab_ans, tab_trace = st.tabs(["Resposta", "Rastro"])
    with tab_ans:
        if flagged:
            st.markdown(
                f":orange-badge[⚠️ números não verificados: {len(flagged)}]",
                help="Números da resposta que não aparecem em nenhuma saída de ferramenta desta execução "
                     "(podem ser cálculos derivados). Veja a aba Rastro.")
        st.markdown(result.answer or "_(sem resposta)_")
        for j, spec in enumerate(result.charts):
            render_chart(spec, key=f"{key}_chart_{j}")
    with tab_trace:
        render_trace(result)


# --------------------------------------------------------------------------- #
# Estado da sessão
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("question", "")
ss.setdefault("history", [])          # [(pergunta, resposta)] das respostas ao vivo bem-sucedidas
ss.setdefault("live_results", [])     # AgentResult desta sessão (nunca gravado em disco pela UI)
ss.setdefault("live_count", 0)
ss.setdefault("demo_choice", None)


def _set_question(q: str) -> None:
    ss["question"] = q


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Configuração")
    st.text_input("Sua chave da Anthropic (opcional, não é salva)", type="password", key="user_api_key",
                  help="Fica só na memória desta sessão do navegador; não é gravada nem exibida.")
    creds = has_credentials()
    modes = ["Ao vivo", "Demo"] if creds else ["Demo"]
    mode = st.radio("Modo", modes, index=0, horizontal=True)
    if not creds:
        st.caption("Sem credencial da Anthropic: só o modo demo está disponível.")
    st.markdown(f"**Modelo:** `{config.MODEL}` · **effort:** `{config.EFFORT}`")
    st.markdown(f"**Limites:** {config.MAX_TOOL_CALLS} ferramentas/pergunta · "
                f"{MAX_LIVE_ANSWERS} respostas ao vivo/sessão")
    with st.expander("Dados e modelo em uso"):
        st.markdown("**Warehouse**")
        st.json(warehouse_info(), expanded=False)
        mi = model_info()
        st.markdown("**Modelo de títulos (LightGBM)**")
        if mi:
            st.json(mi, expanded=False)
        else:
            st.caption("Informações do modelo indisponíveis.")
    if mode == "Ao vivo" and ss["history"]:
        if st.button("Nova conversa"):
            ss["history"] = []
    st.warning(CAVEAT_STRIP)

# --------------------------------------------------------------------------- #
# Área principal
# --------------------------------------------------------------------------- #
st.title("YouTube Insights Agent")
st.caption("O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos "
           "determinísticos; o sistema expõe suas limitações.")

cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES):
    col.button(label, help=q, on_click=_set_question, args=(q,), width="stretch")

if mode == "Ao vivo":
    question = st.text_area("Pergunta", key="question", height=90,
                            placeholder="Ex.: Qual a mediana de views de Shorts vs vídeos longos no Brasil?")
    remaining = MAX_LIVE_ANSWERS - ss["live_count"]
    ask = st.button("Perguntar", type="primary", disabled=remaining <= 0)
    if remaining <= 0:
        st.info(f"Limite de {MAX_LIVE_ANSWERS} respostas ao vivo nesta sessão atingido. "
                "Recarregue a página para começar outra sessão ou use o modo demo.")
    if ask and question.strip() and remaining > 0:
        try:
            client = make_client()
        except Exception:
            client = None   # run_agent devolve a mensagem amigável de credencial
        with st.status("Analisando…", expanded=True) as status:
            def on_step(step: TraceStep) -> None:
                status.write(describe_step(step))
            result = run_agent(question, history=list(ss["history"]), client=client, on_step=on_step)
            status.update(label=f"Análise {STATUS_PT.get(result.status, result.status)} "
                                f"em {result.latency_s:.1f} s",
                          state="complete" if result.status == "ok" else "error", expanded=False)
        ss["live_count"] += 1
        ss["live_results"].append(result)
        if result.status == "ok":
            ss["history"] = (ss["history"] + [(result.question, result.answer)])[-6:]
    if ss["live_results"]:
        render_result(ss["live_results"][-1], key=f"live_{len(ss['live_results'])}")
        older = ss["live_results"][:-1]
        if older:
            with st.expander(f"Respostas anteriores desta sessão ({len(older)})"):
                for k, r in enumerate(reversed(older)):
                    st.markdown(f"**{r.question}**")
                    st.markdown(r.answer)
                    st.divider()
else:
    st.info("Modo demo: reproduzindo execuções gravadas, sem chamar a API da Anthropic.")
    traces = list_demo_traces()
    if not traces:
        st.markdown("Ainda não há execuções gravadas em `demo_traces/`. Grave algumas no modo ao vivo "
                    "(ex.: `agent.save_trace(result, 'demo_traces/exemplo.json')`) e recarregue a página.")
    else:
        names = list(traces)
        # um clique num exemplo seleciona a gravação com a mesma pergunta, se houver
        q = (ss.get("question") or "").strip()
        match = next((n for n in names if traces[n].question.strip() == q), None)
        if match:
            ss["demo_choice"] = match
        if ss.get("demo_choice") not in names:
            ss["demo_choice"] = names[0]
        choice = st.selectbox("Execução gravada", names, key="demo_choice",
                              format_func=lambda n: f"{traces[n].question[:90]}  ({n})")
        render_result(traces[choice], key=f"demo_{choice}")
