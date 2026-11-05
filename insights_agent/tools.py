"""Ferramentas expostas ao Claude: definições (JSON Schema estrito) e execução.

O LLM só enxerga três ferramentas:

- ``run_sql``: consulta read-only no warehouse (validada por ``sql_guard``);
- ``make_chart``: gráfico Altair a partir de um resultado anterior (``result_id``);
- ``score_titles``: score associativo de 1 a 3 títulos (LightGBM + SHAP).

``dispatch(name, input, state)`` executa a ferramenta e devolve
``(texto_para_o_llm, TraceStep, chart_spec | None)``. O texto vai embrulhado em
``<resultado_ferramenta ... confiavel="nao">`` porque contém dados vindos da base
(títulos, nomes de canal, mensagens de erro), que nunca devem ser tratados como
instruções.

Os módulos de dados (``db``, ``sql_guard``, ``charts``, ``predictor``) são
importados de forma preguiçosa: este arquivo importa sem o warehouse nem os
modelos, o que permite testar o agente offline.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import time
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any

from insights_agent import config
from insights_agent.trace import TraceStep

SCORE_DISCLAIMER = (
    "Scores refletem associação estatística em vídeos que entraram em alta, "
    "não causalidade nem qualidade editorial."
)
MAX_TITLES = 3
TIE_RATIO = 1.5            # diferença < 1,5x entre títulos = empate técnico
PREVIEW_CHARS = 400

# Palavras-chave do JSON Schema que o strict tool use não aceita (ou aceita mal).
_UNSUPPORTED_KEYS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                     "multipleOf", "minLength", "maxLength", "pattern",
                     "minItems", "maxItems", "uniqueItems", "default")


def _mod(name: str):
    """Importa ``insights_agent.<name>`` sob demanda (respeita sys.modules, útil nos testes)."""
    return importlib.import_module(f"insights_agent.{name}")


# --------------------------------------------------------------------------- #
# Definições das ferramentas
# --------------------------------------------------------------------------- #
_FALLBACK_CONTEXT_FIELDS: dict[str, tuple[str, str]] = {
    "category": ("string", "Categoria do YouTube em inglês (ex.: Education, Entertainment, Gaming). "
                           "Padrão: Entertainment."),
    "duration_s": ("integer", "Duração prevista do vídeo em segundos. Padrão: 600."),
    "channel_subs": ("integer", "Inscritos do canal. Padrão: 100000."),
    "channel_views": ("integer", "Views totais do canal. Padrão: 20000000."),
    "channel_videos": ("integer", "Número de vídeos do canal. Padrão: 300."),
    "channel_age_days": ("integer", "Idade do canal em dias. Padrão: 1500."),
    "market": ("string", "País-alvo em inglês (ex.: Brazil). Padrão: Brazil."),
    "pub_hour_utc": ("integer", "Hora de publicação em UTC (0-23; Brasília = UTC-3). Padrão: 21."),
    "pub_dow": ("integer", "Dia da semana (0 = segunda ... 6 = domingo). Padrão: 4."),
    "horizon_days": ("integer", "Horizonte da previsão em dias após a publicação. Padrão: 7."),
    "tags": ("string", "Tags separadas por vírgula (pode ser vazio). Padrão: vazio."),
    "desc_len": ("integer", "Tamanho da descrição em caracteres. Padrão: 500."),
}


def _fallback_context_schema() -> dict:
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d}
                       for k, (t, d) in _FALLBACK_CONTEXT_FIELDS.items()},
        "required": list(_FALLBACK_CONTEXT_FIELDS),
        "additionalProperties": False,
    }


def _strictify(schema: Any) -> Any:
    """Garante as regras do strict tool use em qualquer nível do schema.

    Todo objeto ganha ``additionalProperties: false`` e ``required`` com todas as
    propriedades; restrições numéricas/de tamanho são removidas (são validadas
    no código, não pela gramática).
    """
    if isinstance(schema, dict):
        out = {k: _strictify(v) for k, v in schema.items() if k not in _UNSUPPORTED_KEYS}
        if out.get("type") == "object" or "properties" in out:
            props = out.get("properties", {})
            out["properties"] = props
            out["required"] = list(props)
            out["additionalProperties"] = False
        return out
    if isinstance(schema, list):
        return [_strictify(s) for s in schema]
    return schema


def _context_schema() -> dict:
    try:
        schema = _mod("predictor").context_schema()
        if not isinstance(schema, dict) or not schema.get("properties"):
            raise ValueError("schema vazio")
    except Exception:  # predictor/modelos indisponíveis: usa o schema local
        schema = _fallback_context_schema()
    return _strictify(copy.deepcopy(schema))


def _build_tools() -> list[dict]:
    run_sql = {
        "name": "run_sql",
        "description": (
            "Executa UMA consulta SELECT (dialeto DuckDB) no warehouse read-only de vídeos em alta "
            "do YouTube (tabelas videos, video_country, daily_trending). Use sempre que a resposta "
            "depender de números da base: contagens, medianas, comparações entre grupos, rankings. "
            "A consulta é validada antes de rodar (só SELECT/WITH, só as tabelas do catálogo, "
            f"no máximo {config.MAX_ROWS} linhas). Devolve uma tabela markdown, o número de linhas "
            "e um result_id (ex.: r1) que pode ser usado em make_chart. Prefira agregar no SQL "
            "(GROUP BY, median, quantile_cont, count) em vez de trazer linhas brutas."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string",
                        "description": "Uma única instrução SELECT (CTEs permitidas), dialeto DuckDB."},
                "purpose": {"type": "string",
                            "description": "Em uma frase, o que esta consulta quer verificar."},
            },
            "required": ["sql", "purpose"],
            "additionalProperties": False,
        },
    }
    make_chart = {
        "name": "make_chart",
        "description": (
            "Cria um gráfico (Altair) a partir de um resultado anterior de run_sql, identificado "
            "pelo result_id. Use quando um gráfico ajudar a comparar grupos ou mostrar uma "
            "tendência; no máximo um ou dois por resposta. x, y e color devem ser nomes de colunas "
            "exatamente como aparecem no resultado. O gráfico é mostrado ao usuário; você recebe "
            "apenas a confirmação."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "result_id devolvido por run_sql (ex.: r1)."},
                "kind": {"type": "string", "enum": ["bar", "line", "scatter"],
                         "description": "bar para comparar grupos, line para séries temporais, "
                                        "scatter para relação entre duas medidas."},
                "x": {"type": "string", "description": "Coluna do eixo x."},
                "y": {"type": "string", "description": "Coluna do eixo y."},
                "color": {"anyOf": [{"type": "string"}, {"type": "null"}],
                          "description": "Coluna para cor/série, ou null."},
                "title": {"type": "string", "description": "Título curto do gráfico, em português."},
            },
            "required": ["result_id", "kind", "x", "y", "color", "title"],
            "additionalProperties": False,
        },
    }
    score_titles = {
        "name": "score_titles",
        "description": (
            "Pontua de 1 a 3 títulos candidatos com o modelo LightGBM treinado em vídeos que "
            "entraram em alta, no mesmo contexto de vídeo, e explica cada score com fatores SHAP. "
            "Use quando o usuário pedir sugestões ou comparação de títulos: escreva você mesmo os "
            f"títulos (no máximo {MAX_TITLES}) e chame esta ferramenta uma vez com todos eles. "
            "Preencha o contexto com o que o usuário informou e use os padrões descritos para o "
            "resto. Os títulos devem ser fiéis à ideia e ao formato do vídeo (sem #shorts em vídeo "
            "com mais de 180 s, sem caixa alta ou clickbait). A ferramenta ordena as alternativas "
            "pelo padrão estatístico aprendido em vídeos que entraram em alta: é associação, não "
            "garantia de views nem medida de qualidade."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "titles": {"type": "array", "items": {"type": "string"},
                           "description": f"De 1 a {MAX_TITLES} títulos candidatos."},
                "context": _context_schema(),
            },
            "required": ["titles", "context"],
            "additionalProperties": False,
        },
    }
    return [run_sql, make_chart, score_titles]


@lru_cache(maxsize=1)
def _tools_cached() -> str:
    # Serializado de forma determinística: mesma lista, mesmos bytes -> cache do prompt estável.
    return json.dumps(_build_tools(), ensure_ascii=False, sort_keys=False)


def get_tools() -> list[dict]:
    """Lista de ferramentas (cópia nova a cada chamada; conteúdo estável no processo)."""
    return json.loads(_tools_cached())


def __getattr__(name: str):  # PEP 562: ``tools.TOOLS`` sem importar o predictor no import do módulo
    if name == "TOOLS":
        return get_tools()
    raise AttributeError(name)


# --------------------------------------------------------------------------- #
# Estado de uma execução e utilidades
# --------------------------------------------------------------------------- #
def new_state() -> dict:
    """Estado isolado de UMA chamada de run_agent (resultados por result_id)."""
    return {"results": {}, "next_id": 1}


def truncate(text: str, limit: int | None = None) -> str:
    limit = limit or config.TOOL_RESULT_MAX_CHARS
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n… [truncado: {omitted} caracteres omitidos]"


def wrap_untrusted(source: str, text: str) -> str:
    """Embrulha o resultado como dado não confiável (impede 'fechar' a tag por dentro)."""
    safe = text.replace("</resultado_ferramenta", "&lt;/resultado_ferramenta")
    return f'<resultado_ferramenta fonte="{source}" confiavel="nao">\n{safe}\n</resultado_ferramenta>'


def _preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def _fmt_int(v: Any) -> str:
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Implementação de cada ferramenta: devolve (texto, ok, extra, chart_spec)
# --------------------------------------------------------------------------- #
def _run_sql(inp: dict, state: dict):
    sql = inp.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return "Erro: o campo 'sql' está vazio.", False, {}, None
    extra: dict[str, Any] = {"purpose": inp.get("purpose", "")}
    try:
        sql_guard = _mod("sql_guard")
        rejected_cls = sql_guard.SQLRejected
    except Exception:
        rejected_cls = ()  # sem sql_guard, db.run_query ainda valida/levanta o que tiver
    try:
        res = _mod("db").run_query(sql)
    except Exception as e:  # noqa: BLE001 - qualquer falha volta ao LLM como texto
        if rejected_cls and isinstance(e, rejected_cls):
            reason = getattr(e, "reason", None) or str(e)
            extra["rejected"] = True
            return (f"Consulta rejeitada pelo validador de SQL: {reason}\n"
                    "Reescreva como um único SELECT sobre videos, video_country ou daily_trending."), \
                False, extra, None
        msg = str(e).strip() or type(e).__name__
        return (f"Erro ao executar a consulta ({type(e).__name__}): {msg}\n"
                "Verifique nomes de colunas/tabelas no catálogo e tente de novo."), False, extra, None

    rid = f"r{state['next_id']}"
    state["next_id"] += 1
    state["results"][rid] = res
    extra.update({"result_id": rid, "sql_validated": res.sql, "row_count": res.row_count,
                  "truncated": bool(res.truncated), "elapsed_ms": round(float(res.elapsed_ms), 1),
                  "columns": list(res.columns)})
    head = [f"result_id: {rid}",
            f"linhas: {res.row_count}" + (f" (truncado no limite de {config.MAX_ROWS})" if res.truncated else ""),
            f"colunas: {', '.join(map(str, res.columns))}",
            f"tempo: {res.elapsed_ms:.0f} ms",
            "SQL validado:", res.sql, ""]
    try:
        table = res.to_markdown()
    except Exception:  # noqa: BLE001
        table = "\n".join(" | ".join(map(str, r)) for r in res.rows[:30])
    return "\n".join(head) + table, True, extra, None


def _make_chart(inp: dict, state: dict):
    rid = inp.get("result_id")
    res = state["results"].get(rid)
    extra = {"result_id": rid}
    if res is None:
        known = ", ".join(state["results"]) or "nenhum"
        return (f"Erro: result_id '{rid}' não existe nesta pergunta (disponíveis: {known}). "
                "Rode run_sql antes de pedir o gráfico."), False, extra, None
    try:
        import pandas as pd
        df = pd.DataFrame(res.rows, columns=res.columns)
        chart = _mod("charts").build_chart(df, inp.get("kind"), inp.get("x"), inp.get("y"),
                                           inp.get("color"), inp.get("title") or "")
        spec = chart.to_dict()
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip() or type(e).__name__
        return f"Erro ao criar o gráfico: {msg}", False, extra, None
    extra.update({"kind": inp.get("kind"), "x": inp.get("x"), "y": inp.get("y")})
    text = (f"Gráfico criado ({inp.get('kind')}: x={inp.get('x')}, y={inp.get('y')}) a partir de {rid}; "
            f"será exibido ao usuário junto da resposta ({len(df)} linhas no gráfico).")
    return text, True, extra, spec


def _score_titles(inp: dict, state: dict):
    titles = inp.get("titles")
    if not isinstance(titles, list) or not titles:
        return "Erro: envie de 1 a 3 títulos em 'titles'.", False, {}, None
    titles = [t.strip() for t in titles if isinstance(t, str) and t.strip()]
    if not 1 <= len(titles) <= MAX_TITLES:
        return f"Erro: envie de 1 a {MAX_TITLES} títulos não vazios (recebidos: {len(titles)}).", False, {}, None
    ctx_in = inp.get("context") or {}
    try:
        predictor = _mod("predictor")
        VideoContext = predictor.VideoContext
        known = {f.name for f in fields(VideoContext)} if is_dataclass(VideoContext) else set(ctx_in)
        ctx = VideoContext(**{k: v for k, v in ctx_in.items() if k in known and v is not None})
        scored = predictor.score_titles(titles, ctx)
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip() or type(e).__name__
        return f"Erro ao pontuar os títulos: {msg}", False, {"context": ctx_in}, None

    lines = ["| # | título | views previstas pelo modelo | faixa 2x | texto do título (SHAP) | principais fatores (SHAP, log) |",
             "|---|---|---|---|---|---|"]
    warnings: list[str] = []
    for i, item in enumerate(scored, 1):
        low, high = (item.get("interval_2x") or [None, None])[:2]
        factors = "; ".join(
            f"{f.get('label_pt') or f.get('feature')} {float(f.get('contribution_log', 0)):+.2f}"
            for f in (item.get("top_factors") or [])[:4])
        title = str(item.get("title", "")).replace("|", "/")
        tel = float(item.get("title_effect_log", 0) or 0)
        lines.append(f"| {i} | {title} | {_fmt_int(item.get('predicted_views'))} | "
                     f"{_fmt_int(low)}–{_fmt_int(high)} | {tel:+.2f} log ≈ ×{math.exp(tel):.2f} | {factors} |")
        for w in item.get("warnings") or []:
            if w not in warnings:
                warnings.append(str(w))
    preds = [float(s.get("predicted_views") or 0) for s in scored]
    ratio = (max(preds) / min(preds)) if len(preds) > 1 and min(preds) > 0 else None
    if ratio is None:
        spread = ""
    elif ratio < TIE_RATIO:
        spread = (f"Razão entre o maior e o menor score: ×{ratio:.2f} (< ×{TIE_RATIO}): empate técnico, "
                  "dentro da margem do modelo; escolha por critério editorial.")
    else:
        spread = f"Razão entre o maior e o menor score: ×{ratio:.2f}."
    ctx_used = vars(ctx) if hasattr(ctx, "__dict__") else ctx_in
    note = (scored[0].get("notes") if scored else None) or getattr(predictor, "INTERVAL_NOTE_PT", "")
    effect_label = getattr(predictor, "TITLE_EFFECT_LABEL_PT",
                           "contribuição das features de texto do título (SHAP), não é efeito causal")
    text = "\n".join([
        "O modelo ordena as alternativas pelo padrão estatístico aprendido em vídeos que entraram em alta.",
        "Contexto usado: " + ", ".join(f"{k}={v}" for k, v in ctx_used.items()),
        "",
        *lines,
        "",
        *([spread] if spread else []),
        f"Coluna 'texto do título': {effect_label}. Unidades em log(views): +0,69 ≈ ×2, −0,69 ≈ ×0,5.",
        "Tamanho do canal e horizonte (dias) costumam pesar muito mais que o título; trocar só o título "
        "move pouco a previsão.",
        *([f"Faixa 2x: {note}"] if note else []),
        *([("Avisos do modelo: " + " | ".join(warnings))] if warnings else []),
        SCORE_DISCLAIMER,
    ])
    extra = {"context": dict(ctx_used), "n_titles": len(scored), "warnings": warnings,
             "max_min_ratio": round(ratio, 3) if ratio else None,
             "model_info": copy.deepcopy(getattr(predictor, "MODEL_INFO", None)),
             "scores": [{"title": s.get("title"), "predicted_views": s.get("predicted_views"),
                         "interval_2x": s.get("interval_2x"),
                         "title_effect_log": s.get("title_effect_log")} for s in scored]}
    return text, True, extra, None


_IMPLS = {"run_sql": _run_sql, "make_chart": _make_chart, "score_titles": _score_titles}


def dispatch(name: str, input: dict, state: dict) -> tuple[str, TraceStep, dict | None]:
    """Executa a ferramenta ``name``. Nunca levanta exceção: falhas viram texto de erro."""
    t0 = time.perf_counter()
    inp = input if isinstance(input, dict) else {}
    impl = _IMPLS.get(name)
    if impl is None:
        text, ok, extra, spec = f"Erro: ferramenta desconhecida '{name}'.", False, {}, None
    elif not isinstance(input, dict):
        text, ok, extra, spec = "Erro: a entrada da ferramenta deve ser um objeto JSON.", False, {}, None
    else:
        try:
            text, ok, extra, spec = impl(inp, state)
        except Exception as e:  # noqa: BLE001 - rede de segurança
            text, ok, extra, spec = f"Erro inesperado na ferramenta: {type(e).__name__}: {e}", False, {}, None
    text = truncate(text)
    extra["result_text"] = text   # texto integral (truncado) usado pela checagem de números
    step = TraceStep(kind="tool", name=name, input=copy.deepcopy(inp), output_preview=_preview(text),
                     ok=ok, elapsed_ms=round((time.perf_counter() - t0) * 1000, 1), extra=extra)
    return wrap_untrusted(name, text), step, spec
