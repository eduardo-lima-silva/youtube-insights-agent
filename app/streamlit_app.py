"""UI do YouTube Insights Agent (Streamlit).

Rodar:  streamlit run app/streamlit_app.py

- Provedor: Groq (plano gratuito, modelo aberto). A chave vem do arquivo `.env` na raiz do projeto
  (`GROQ_API_KEY=...`), da variável de ambiente ou de `st.secrets` (deploy no Streamlit Cloud).
- Sem chave: modo demo, que reproduz execuções gravadas em demo_traces/*.json sem chamar API.
- As tabelas das respostas viram tabelas interativas (ordenar, buscar, tela cheia) com aba de gráfico.
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
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from insights_agent import config  # noqa: E402  (carrega o .env)
from insights_agent.agent import load_trace, run_agent, unverified_numbers  # noqa: E402
from insights_agent.trace import AgentResult, TraceStep  # noqa: E402

PROVIDER = "groq"
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
ACCENT = "#C4302B"

st.set_page_config(page_title="YouTube Insights Agent", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")
# sem barra lateral: tudo fica na área principal
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
.block-container {padding-top: 2rem; max-width: 1150px;}
</style>""", unsafe_allow_html=True)


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


def _api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")  # type: ignore[no-any-return]
    except Exception:  # sem secrets.toml
        return None


def make_client():
    import groq
    return groq.Groq(api_key=_api_key())


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
# Tabelas interativas a partir do markdown da resposta
# --------------------------------------------------------------------------- #
_SEP_ROW = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_NUM_CELL = re.compile(r"^[≈~]?\s*(-?\d{1,3}(?:[.\s\u00a0\u202f]\d{3})+|-?\d+)(?:,(\d+))?\s*(%)?$")


def _clean_cell(c: str) -> str:
    c = re.sub(r"\*\*|__|`", "", c)
    c = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"\1", c)   # *itálico* -> itálico
    return c.strip()


def _to_number(c: str):
    """'54 683' / '1.234,5' / '12,5 %' / '63412' -> número; senão None."""
    m = _NUM_CELL.match(c.strip())
    if not m:
        return None
    ip = re.sub(r"[.\s\u00a0\u202f]", "", m.group(1))
    return float(f"{ip}.{m.group(2)}") if m.group(2) else float(ip)


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [_clean_cell(c) for c in line.split("|")]


def markdown_table_to_df(lines: list[str]) -> pd.DataFrame | None:
    rows = [_split_row(ln) for ln in lines if not _SEP_ROW.match(ln.strip())]
    if len(rows) < 2:
        return None
    header, body = rows[0], rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    header = [h or f"col{i + 1}" for i, h in enumerate(header)]
    if len(set(header)) != len(header):     # nomes repetidos quebram o DataFrame
        header = [f"{h} ({i + 1})" for i, h in enumerate(header)]
    df = pd.DataFrame(body, columns=header)
    for col in df.columns:
        vals = [v for v in df[col] if v not in ("", "-", "—")]
        nums = [_to_number(v) for v in vals]
        if vals and all(n is not None for n in nums):
            df[col] = pd.to_numeric(pd.Series([_to_number(v) if v not in ("", "-", "—") else None
                                               for v in df[col]], dtype="float64"))
            if df[col].dropna().apply(float.is_integer).all():
                df[col] = df[col].astype("Int64")      # contagens sem ",0"
    return df


def split_markdown(text: str) -> list[tuple[str, object]]:
    """Separa a resposta em blocos ('md', texto) e ('table', linhas)."""
    blocks: list[tuple[str, object]] = []
    buf: list[str] = []
    table: list[str] = []
    for line in (text or "").splitlines():
        if line.strip().startswith("|"):
            if buf:
                blocks.append(("md", "\n".join(buf)))
                buf = []
            table.append(line)
        else:
            if table:
                blocks.append(("table", table))
                table = []
            buf.append(line)
    if table:
        blocks.append(("table", table))
    if buf:
        blocks.append(("md", "\n".join(buf)))
    return blocks


def _pick_metric(df: pd.DataFrame, numeric: list[str]) -> str:
    for hint in ("median", "mediana", "views", "visualiza", "previst", "vídeos", "videos", "n_"):
        for c in numeric:
            if hint in c.lower():
                return c
    return numeric[-1]


def render_table(df: pd.DataFrame, key: str) -> None:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    text_cols = [c for c in df.columns if c not in numeric]
    metrics = [c for c in numeric if c.strip().lower() not in ("rank", "#", "posição", "posicao", "pos")]
    cfg = {c: st.column_config.NumberColumn(c, format="localized") for c in numeric}
    height = min(38 * (len(df) + 1) + 4, 420)

    def table() -> None:
        st.dataframe(df, hide_index=True, width="stretch", column_config=cfg, height=height, key=f"{key}_df")

    chartable = metrics and text_cols and 2 <= len(df) <= 40
    if not chartable:
        table()
        return
    t_tab, t_chart = st.tabs(["📋 Tabela", "📊 Gráfico"])
    with t_tab:
        table()
    with t_chart:
        label = text_cols[0]
        metric = st.selectbox("Métrica", metrics, index=metrics.index(_pick_metric(df, metrics)),
                              key=f"{key}_metric")
        # nomes internos simples: o Altair interpreta parênteses/dois-pontos nos nomes de campo
        data = df[[label, metric]].dropna().rename(columns={label: "rotulo", metric: "valor"})
        data["valor"] = data["valor"].astype(float)
        chart = (alt.Chart(data).mark_bar(color=ACCENT, cornerRadiusEnd=3)
                 .encode(y=alt.Y("rotulo:N", sort="-x", title=None),
                         x=alt.X("valor:Q", title=metric, axis=alt.Axis(format="~s")),
                         tooltip=[alt.Tooltip("rotulo:N", title=label),
                                  alt.Tooltip("valor:Q", title=metric, format=",.0f")])
                 .properties(height=max(120, 32 * len(data))))
        st.altair_chart(chart, width="stretch", key=f"{key}_chart")


def render_rich_answer(text: str, key: str) -> None:
    """Markdown normal + tabelas markdown viram tabelas interativas com aba de gráfico."""
    for i, (kind, content) in enumerate(split_markdown(text)):
        if kind == "md":
            if str(content).strip():
                st.markdown(content)
            continue
        df = markdown_table_to_df(content)  # type: ignore[arg-type]
        if df is None or df.empty:
            st.markdown("\n".join(content))  # type: ignore[arg-type]
        else:
            render_table(df, key=f"{key}_t{i}")


def query_tables(result: AgentResult) -> list[tuple[str, pd.DataFrame]]:
    """Resultados das consultas SQL desta execução, como DataFrames (para o painel de dados)."""
    out = []
    for s in result.steps:
        if s.kind == "tool" and s.name == "run_sql" and s.ok:
            txt = (s.extra or {}).get("result_text") or ""
            lines = [ln for ln in txt.splitlines() if ln.strip().startswith("|")]
            df = markdown_table_to_df(lines) if lines else None
            if df is not None and not df.empty:
                out.append(((s.extra or {}).get("result_id") or s.name, df))
    return out


# --------------------------------------------------------------------------- #
# Rastro
# --------------------------------------------------------------------------- #
def describe_step(step: TraceStep) -> str:
    icon = "✅" if step.ok else "⚠️"
    if step.kind == "llm":
        return f"{icon} Modelo pensou e respondeu ({step.elapsed_ms / 1000:.1f}s)"
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
    if step.kind == "guard" and step.name == "rate_limit_wait":
        return f"⏳ {step.output_preview}"
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latência", f"{result.latency_s:.1f} s")
    c2.metric("Custo", f"US$ {result.cost_usd:.2f}")
    c3.metric("Tokens entrada", f"{u.input_tokens:,}".replace(",", "."))
    c4.metric("Tokens saída", f"{u.output_tokens:,}".replace(",", "."))
    st.caption(f"Modelo: {result.model or config.GROQ_MODEL} · chamadas ao modelo: {u.llm_calls} · "
               f"status: {STATUS_PT.get(result.status, result.status)}")
    if result.error:
        st.caption("Detalhe técnico: " + strip_paths(result.error))

    for i, step in enumerate(result.steps, 1):
        label = (f"{i}. [{step.kind}] {step.name} · {'ok' if step.ok else 'falhou'} · "
                 f"{step.elapsed_ms:.0f} ms")
        with st.expander(strip_paths(label), expanded=False):
            extra = dict(step.extra or {})
            if step.kind == "tool":
                if step.input.get("purpose"):
                    st.markdown(f"**Objetivo:** {strip_paths(step.input['purpose'])}")
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
                reason = extra.get("stop_reason") or extra.get("finish_reason")
                st.markdown(f"**motivo de parada:** `{reason}` · **tokens:** entrada "
                            f"{extra.get('input_tokens')}, saída {extra.get('output_tokens')}")
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
    st.markdown(f"#### {result.question}")
    if result.status != "ok":
        st.warning(f"Status: {STATUS_PT.get(result.status, result.status)}")
    flagged = unverified_numbers(result)
    data = query_tables(result)
    tabs = st.tabs(["💬 Resposta", "🗃️ Dados da consulta", "🔎 Rastro"])
    with tabs[0]:
        if flagged:
            st.markdown(
                f":orange-badge[⚠️ números não verificados: {len(flagged)}]",
                help="Números da resposta que não aparecem em nenhuma saída de ferramenta desta execução "
                     "(podem ser cálculos derivados). Veja a aba Rastro.")
        render_rich_answer(result.answer or "_(sem resposta)_", key=f"{key}_ans")
        for j, spec in enumerate(result.charts):
            render_chart(spec, key=f"{key}_chart_{j}")
    with tabs[1]:
        if not data:
            st.caption("Esta resposta não usou consultas SQL.")
        for j, (rid, df) in enumerate(data):
            st.caption(f"Resultado `{rid}` · {len(df)} linha(s), exatamente como saiu do banco")
            render_table(df, key=f"{key}_data_{j}")
    with tabs[2]:
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
# Cabeçalho
# --------------------------------------------------------------------------- #
has_key = bool(_api_key())
head_l, head_r = st.columns([3, 1])
with head_l:
    st.title("YouTube Insights Agent")
    st.caption("O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos "
               "determinísticos; o sistema expõe suas limitações.")
with head_r:
    st.write("")
    if has_key:
        st.markdown(f":green-badge[● ao vivo] :gray-badge[{config.GROQ_MODEL} · grátis]")
        demo_mode = st.toggle("Ver execuções gravadas", value=False)
    else:
        st.markdown(":orange-badge[● modo demo]")
        st.caption("Sem GROQ_API_KEY no .env: só as execuções gravadas.")
        demo_mode = True

with st.expander("ℹ️ Sobre os dados, o modelo e os limites"):
    st.warning(CAVEAT_STRIP)
    st.markdown(f"**LLM:** `{config.GROQ_MODEL}` no Groq (plano gratuito: ~8 mil tokens/min e ~200 mil/dia; "
                f"o agente espera sozinho quando bate o limite por minuto). **Limites:** "
                f"{config.MAX_TOOL_CALLS} ferramentas por pergunta · {MAX_LIVE_ANSWERS} respostas por sessão.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Warehouse**")
        st.json(warehouse_info(), expanded=False)
    with c2:
        st.markdown("**Modelo de títulos (LightGBM)**")
        mi = model_info()
        if mi:
            st.json(mi, expanded=False)
        else:
            st.caption("Informações do modelo indisponíveis.")
    if ss["history"] and st.button("Nova conversa (esquecer o contexto)"):
        ss["history"] = []

cols = st.columns(len(EXAMPLES))
for col, (label, q) in zip(cols, EXAMPLES):
    col.button(label, help=q, on_click=_set_question, args=(q,), width="stretch")

# --------------------------------------------------------------------------- #
# Ao vivo
# --------------------------------------------------------------------------- #
if not demo_mode:
    question = st.text_area("Pergunta", key="question", height=90,
                            placeholder="Ex.: Qual a mediana de views de Shorts vs vídeos longos no Brasil?")
    remaining = MAX_LIVE_ANSWERS - ss["live_count"]
    ask = st.button("Perguntar", type="primary", disabled=remaining <= 0)
    if remaining <= 0:
        st.info(f"Limite de {MAX_LIVE_ANSWERS} respostas nesta sessão atingido. Recarregue a página para "
                "começar outra sessão.")
    if ask and question.strip() and remaining > 0:
        try:
            client = make_client()
        except Exception:
            client = None   # run_agent devolve a mensagem amigável de credencial
        with st.status("Analisando…", expanded=True) as status:
            def on_step(step: TraceStep) -> None:
                status.write(describe_step(step))
            result = run_agent(question, history=list(ss["history"]), client=client, on_step=on_step,
                               provider=PROVIDER)
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
                    render_rich_answer(r.answer, key=f"old_{k}")
                    st.divider()
# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
else:
    st.info("Reproduzindo execuções gravadas, sem chamar nenhuma API de LLM.")
    traces = list_demo_traces()
    if not traces:
        st.markdown("Ainda não há execuções gravadas em `demo_traces/`.")
    else:
        names = list(traces)
        q = (ss.get("question") or "").strip()
        match = next((n for n in names if traces[n].question.strip() == q), None)
        if match:
            ss["demo_choice"] = match
        if ss.get("demo_choice") not in names:
            ss["demo_choice"] = names[0]
        choice = st.selectbox("Execução gravada", names, key="demo_choice",
                              format_func=lambda n: f"{traces[n].question[:90]}")
        render_result(traces[choice], key=f"demo_{choice}")
