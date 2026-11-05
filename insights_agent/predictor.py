"""Score de títulos com o LightGBM do projeto 1 + explicação via SHAP.

O que este módulo faz
---------------------
Dado um contexto de vídeo (canal, categoria, duração, horário...) e 1 a 3 títulos
candidatos, monta as mesmas features usadas no treino do projeto
``youtube-views-prediction`` (lógica copiada e adaptada de ``src/predict.py`` e
``src/02_build_dataset.py``, sem importar de lá), calcula o ``text_score`` com o
modelo TF-IDF + Ridge e devolve, para cada título:

- ``predicted_views``: ``expm1(pred_log)``, onde ``pred_log`` é a saída do LightGBM
  (o alvo do treino é ``log1p(views)``). **Sem correção de viés** de retransformação:
  é uma estimativa do tipo mediana na escala original (tende a ficar abaixo da média);
- ``interval_2x``: ``[previsto/2, previsto*2]``. É uma faixa heurística (~84% das
  previsões de teste ficaram dentro de 2×; ``pct_dentro_2x`` em ``metadata.json``),
  **não** um intervalo de predição calibrado;
- ``top_factors``: maiores contribuições SHAP (em log-views), sem o termo de viés;
- ``title_effect_log``: contribuição das features de texto do título (SHAP) — não é
  efeito causal. Soma das contribuições SHAP de ``TITLE_FEATURES``;
- ``warnings``: avisos (ex.: categoria/mercado fora dos níveis do modelo);
- ``notes``: nota fixa sobre ``interval_2x``.

``MODEL_INFO`` descreve a versão do modelo (arquivo, tamanho, mtime, best_iteration,
nº de features) para o app/rastro mostrarem qual modelo gerou os scores.

Ressalvas importantes (o agente deve repassá-las)
-------------------------------------------------
- A base contém **apenas vídeos que entraram em alta** (trending). A previsão é
  "quantas views um vídeo como este teve, *dado que entrou em alta*", não a
  probabilidade de entrar em alta.
- É **associação estatística**, não causalidade: o modelo ordena títulos segundo o
  padrão aprendido na base; não garante que um título "cause" mais views.
- Sem histórico do canal: ``ch_prev_*`` = NaN e ``ch_prev_count`` = 0 (o modelo vê o
  vídeo como o primeiro do canal em alta).
"""
from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from insights_agent import config

__all__ = ["VideoContext", "score_titles", "context_schema", "FEATURE_LABELS_PT",
           "TITLE_FEATURES", "INTERVAL_NOTE_PT", "TITLE_EFFECT_LABEL_PT", "MODEL_INFO"]

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
MAX_TITLES = 3
MAX_TITLE_CHARS = 150
MAX_TAGS_CHARS = 500           # limite do próprio YouTube para o campo de tags
HORIZON_MIN_DAYS = 1
HORIZON_MAX_DAYS = 30          # treino: p99 da idade ~36 dias, máx. ~38
MAX_DURATION_S = 12 * 3600     # limite de upload do YouTube

INTERVAL_NOTE_PT = (
    "interval_2x é uma faixa heurística (~84% das previsões de teste ficaram dentro de 2×), "
    "não um intervalo de predição calibrado"
)
TITLE_EFFECT_LABEL_PT = "contribuição das features de texto do título (SHAP) — não é efeito causal"

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")

# Features que dependem do título / tags (texto). Usadas em title_effect_log.
TITLE_FEATURES = (
    "text_score", "title_len", "title_words", "title_upper_ratio", "title_non_ascii_ratio",
    "title_has_question", "title_has_exclam", "title_has_number", "title_has_pipe",
    "title_n_emojis", "title_n_hashtags", "title_has_official", "title_has_shorts_tag",
    "n_tags", "tags_len",
)

FEATURE_LABELS_PT = {
    "log_age_days": "horizonte (dias desde a publicação)",
    "log_channel_subs": "inscritos do canal",
    "log_channel_views": "views totais do canal",
    "log_channel_videos": "nº de vídeos do canal",
    "log_channel_views_per_video": "média de views por vídeo do canal",
    "channel_age_days": "idade do canal",
    "hidden_subs": "inscritos ocultos",
    "ch_prev_count": "nº de vídeos anteriores do canal em alta (não informado: 0)",
    "ch_prev_mean_log_views": "histórico do canal em alta, média (não informado)",
    "ch_prev_max_log_views": "histórico do canal em alta, máx. (não informado)",
    "log_duration": "duração do vídeo",
    "is_short": "é Short (≤ 3 min)",
    "is_hd": "vídeo em HD",
    "licensed": "conteúdo licenciado",
    "pub_hour": "hora de publicação (UTC)",
    "pub_dow": "dia da semana da publicação",
    "title_len": "tamanho do título (caracteres)",
    "title_words": "palavras no título",
    "title_upper_ratio": "% de maiúsculas no título",
    "title_non_ascii_ratio": "% de caracteres não-ASCII no título (acentos, emojis)",
    "title_has_question": "título com interrogação",
    "title_has_exclam": "título com exclamação",
    "title_has_number": "título com número",
    "title_has_pipe": "título com barra vertical (|)",
    "title_n_emojis": "nº de emojis no título",
    "title_n_hashtags": "nº de hashtags no título",
    "title_has_official": "título com \"oficial/official\"",
    "title_has_shorts_tag": "título com #shorts",
    "n_tags": "nº de tags",
    "tags_len": "tamanho das tags",
    "desc_len": "tamanho da descrição",
    "desc_n_links": "links na descrição",
    "desc_n_hashtags": "hashtags na descrição",
    "category": "categoria",
    "channel_country": "país do canal",
    "market": "mercado (país em alta)",
    "text_score": "padrão de texto do título/tags (NLP)",
}


# --------------------------------------------------------------------------- #
# Contexto
# --------------------------------------------------------------------------- #
@dataclass
class VideoContext:
    category: str = "Entertainment"
    duration_s: int = 600
    channel_subs: int = 100_000
    channel_views: int = 20_000_000
    channel_videos: int = 300
    channel_age_days: int = 1500
    market: str = "Brazil"
    pub_hour_utc: int = 21
    pub_dow: int = 4
    horizon_days: int = 7
    tags: str = ""
    desc_len: int = 500


_FIELD_DESCRIPTIONS_PT = {
    "category": "Categoria do YouTube do vídeo (em inglês, ex.: Sports, Gaming, Education).",
    "duration_s": "Duração do vídeo em segundos (Shorts: até 180).",
    "channel_subs": "Número de inscritos do canal.",
    "channel_views": "Views totais acumuladas do canal.",
    "channel_videos": "Número de vídeos publicados pelo canal.",
    "channel_age_days": "Idade do canal em dias (desde a criação até a publicação).",
    "market": "País-alvo do vídeo (nome em inglês, ex.: Brazil); também usado como país do canal.",
    "pub_hour_utc": "Hora de publicação em UTC (0 a 23). Brasília = UTC-3.",
    "pub_dow": "Dia da semana da publicação (0 = segunda ... 6 = domingo).",
    "horizon_days": f"Horizonte da previsão: views após N dias ({HORIZON_MIN_DAYS} a {HORIZON_MAX_DAYS}).",
    "tags": "Tags do vídeo separadas por vírgula (pode ser vazio).",
    "desc_len": "Tamanho da descrição em caracteres.",
}

# limites de validação (min, max) para os campos inteiros
_INT_BOUNDS = {
    "duration_s": (1, MAX_DURATION_S),
    "channel_subs": (0, 10**10),
    "channel_views": (0, 10**13),
    "channel_videos": (1, 10**7),
    "channel_age_days": (0, 8000),
    "pub_hour_utc": (0, 23),
    "pub_dow": (0, 6),
    "horizon_days": (HORIZON_MIN_DAYS, HORIZON_MAX_DAYS),
    "desc_len": (0, 5000),
}


# --------------------------------------------------------------------------- #
# Carga dos modelos (uma vez, preguiçosa, thread-safe)
# --------------------------------------------------------------------------- #
_MODELS: dict[str, Any] | None = None
_LOAD_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()


def _load_models() -> dict[str, Any]:
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    with _LOAD_LOCK:
        if _MODELS is None:
            import joblib
            import lightgbm as lgb

            for p in (config.LGBM_MODEL, config.TEXT_MODEL, config.MODEL_META):
                if not p.exists():
                    raise FileNotFoundError(f"Arquivo de modelo não encontrado: {p}")
            meta = json.loads(config.MODEL_META.read_text(encoding="utf-8"))
            vec, ridge = joblib.load(config.TEXT_MODEL)
            booster = lgb.Booster(model_file=str(config.LGBM_MODEL))
            _MODELS = {"meta": meta, "vec": vec, "ridge": ridge, "booster": booster}
    return _MODELS


def _file_info(p) -> dict:
    try:
        st = p.stat()
        return {"file": p.name, "size_bytes": st.st_size,
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    except OSError:
        return {"file": p.name, "size_bytes": None, "mtime_utc": None}


def _build_model_info() -> dict:
    """Metadados da versão do modelo (barato: não carrega o LightGBM)."""
    info: dict[str, Any] = {
        "lgbm": _file_info(config.LGBM_MODEL),
        "text_model": _file_info(config.TEXT_MODEL),
        "best_iteration": None, "n_features": None, "train_period": None,
        "test_period": None, "test_pct_within_2x": None,
    }
    try:
        meta = json.loads(config.MODEL_META.read_text(encoding="utf-8"))
        final = meta.get("results", {}).get("LightGBM + NLP (modelo final)", {})
        info.update(
            best_iteration=meta.get("best_iteration"),
            n_features=len(meta.get("features", [])),
            train_period=meta.get("split", {}).get("train_period"),
            test_period=meta.get("split", {}).get("test_period"),
            test_pct_within_2x=final.get("pct_dentro_2x"),
        )
    except (OSError, ValueError) as e:
        info["error"] = f"metadata.json indisponível: {e}"
    return info


MODEL_INFO: dict[str, Any] = _build_model_info()


def _meta() -> dict:
    return _load_models()["meta"]


# --------------------------------------------------------------------------- #
# Schema para a camada de ferramentas do LLM
# --------------------------------------------------------------------------- #
def context_schema() -> dict:
    """JSON Schema (compatível com strict tool use) do objeto ``context``.

    Todos os campos são obrigatórios, ``additionalProperties`` é ``false`` e só
    são usados ``type``, ``enum`` e ``description`` (sem minimum/maximum).
    Os enums de categoria e mercado vêm de ``cat_levels`` do metadata.json.
    """
    meta = _meta()
    cat_enum = [c for c in meta["cat_levels"]["category"] if c not in ("Unknown", "Other")]
    mkt_enum = [m for m in meta["cat_levels"]["market"] if m not in ("Unknown",)]
    props: dict[str, dict] = {}
    for f in fields(VideoContext):
        desc = _FIELD_DESCRIPTIONS_PT[f.name]
        if f.name == "category":
            props[f.name] = {"type": "string", "enum": cat_enum, "description": desc}
        elif f.name == "market":
            props[f.name] = {"type": "string", "enum": mkt_enum,
                             "description": desc + " Use 'Other' para países fora da lista."}
        elif f.name == "tags":
            props[f.name] = {"type": "string", "description": desc}
        else:
            lo, hi = _INT_BOUNDS[f.name]
            props[f.name] = {"type": "integer", "description": f"{desc} Faixa válida: {lo} a {hi}."}
    return {
        "type": "object",
        "properties": props,
        "required": [f.name for f in fields(VideoContext)],
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #
def _coerce_ctx(ctx: VideoContext | dict) -> VideoContext:
    if isinstance(ctx, VideoContext):
        return ctx
    if isinstance(ctx, dict):
        valid = {f.name for f in fields(VideoContext)}
        extra = set(ctx) - valid
        if extra:
            raise ValueError(f"Campos de contexto desconhecidos: {', '.join(sorted(extra))}.")
        return VideoContext(**ctx)
    raise ValueError("O contexto deve ser um VideoContext ou um dicionário.")


def _validate_titles(titles: Any) -> list[str]:
    if isinstance(titles, str) or not isinstance(titles, (list, tuple)):
        raise ValueError("'titles' deve ser uma lista de 1 a 3 títulos.")
    if not 1 <= len(titles) <= MAX_TITLES:
        raise ValueError(f"Envie de 1 a {MAX_TITLES} títulos (recebidos: {len(titles)}).")
    out = []
    for i, t in enumerate(titles, 1):
        if not isinstance(t, str):
            raise ValueError(f"O título {i} não é um texto.")
        t = t.strip()
        if not t:
            raise ValueError(f"O título {i} está vazio.")
        if len(t) > MAX_TITLE_CHARS:
            raise ValueError(f"O título {i} tem {len(t)} caracteres; o máximo é {MAX_TITLE_CHARS}.")
        out.append(t)
    return out


def _validate_ctx(ctx: VideoContext) -> None:
    for name, (lo, hi) in _INT_BOUNDS.items():
        v = getattr(ctx, name)
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"O campo '{name}' deve ser numérico (recebido: {v!r}).")
        if not math.isfinite(float(v)):
            raise ValueError(f"O campo '{name}' deve ser um número finito.")
        if not lo <= v <= hi:
            raise ValueError(f"O campo '{name}' deve estar entre {lo} e {hi} (recebido: {v}).")
    for name in ("category", "market", "tags"):
        if not isinstance(getattr(ctx, name), str):
            raise ValueError(f"O campo '{name}' deve ser um texto.")
    if len(ctx.tags) > MAX_TAGS_CHARS:
        raise ValueError(f"As tags têm {len(ctx.tags)} caracteres; o máximo é {MAX_TAGS_CHARS}.")


# --------------------------------------------------------------------------- #
# Features (adaptado de youtube-views-prediction/src/predict.py)
# --------------------------------------------------------------------------- #
def _normalize_tags(tags: str) -> str:
    """Formato do treino: tags separadas por vírgula, sem espaços extras."""
    return ",".join(p.strip() for p in tags.split(",") if p.strip())


def _resolve_level(value: str, levels: list[str]) -> tuple[str, bool]:
    """Nível categórico do treino. Devolve (nível, foi_substituído).

    Desconhecido -> 'Other' (como no treino). A coluna ``category`` não tem nível
    'Other' (todas as categorias tinham >= 200 vídeos), então nela o fallback é
    'Unknown', que é um nível real do treino.
    """
    if value in levels:
        return value, False
    low = {lv.lower(): lv for lv in levels}
    if value.strip().lower() in low:
        return low[value.strip().lower()], False
    return ("Other" if "Other" in levels else "Unknown"), True


def _level_warnings(ctx: VideoContext) -> list[str]:
    meta = _meta()
    out = []
    for name in ("category", "market"):
        value = getattr(ctx, name)
        lvl, replaced = _resolve_level(value, meta["cat_levels"][name])
        if replaced:
            out.append(f"'{name}' = {value!r} não está entre os níveis do modelo; usado {lvl!r} no lugar.")
    return out


def _title_row(t: str, tags: str) -> dict:
    letters = max(len(re.findall(r"[A-Za-z]", t)), 1)
    return {
        "title_len": len(t), "title_words": len(t.split()),
        "title_upper_ratio": len(re.findall(r"[A-Z]", t)) / letters,
        "title_non_ascii_ratio": sum(ord(c) > 127 for c in t) / max(len(t), 1),
        "title_has_question": int("?" in t), "title_has_exclam": int("!" in t),
        "title_has_number": int(bool(re.search(r"\d", t))),
        "title_has_pipe": int(bool(re.search(r"[|｜]", t))),   # igual ao treino
        "title_n_emojis": len(EMOJI.findall(t)), "title_n_hashtags": len(re.findall(r"#\w+", t)),
        "title_has_official": int(bool(re.search(r"official|oficial", t, re.I))),
        "title_has_shorts_tag": int(bool(re.search(r"#shorts?", t, re.I))),
        "n_tags": 0 if not tags else tags.count(",") + 1, "tags_len": len(tags),
    }


def _build_frame(titles: list[str], ctx: VideoContext) -> pd.DataFrame:
    m = _load_models()
    meta = m["meta"]
    tags = _normalize_tags(ctx.tags)
    base = {
        "log_age_days": np.log1p(ctx.horizon_days),
        "log_channel_subs": np.log1p(ctx.channel_subs),
        "log_channel_views": np.log1p(ctx.channel_views),
        "log_channel_videos": np.log1p(ctx.channel_videos),
        "log_channel_views_per_video": np.log1p(ctx.channel_views / max(ctx.channel_videos, 1)),
        "channel_age_days": float(ctx.channel_age_days),
        "hidden_subs": 0,
        "ch_prev_count": 0,                      # sem histórico conhecido do canal
        "ch_prev_mean_log_views": np.nan, "ch_prev_max_log_views": np.nan,
        "log_duration": np.log1p(ctx.duration_s), "is_short": int(ctx.duration_s <= 180),
        "is_hd": 1, "licensed": 1,
        "pub_hour": int(ctx.pub_hour_utc), "pub_dow": int(ctx.pub_dow),
        "desc_len": int(ctx.desc_len), "desc_n_links": 0, "desc_n_hashtags": 0,
        "category": ctx.category, "channel_country": ctx.market, "market": ctx.market,
    }
    rows = [{**base, **_title_row(t, tags)} for t in titles]
    X = pd.DataFrame(rows)
    texts = [(t + " " + tags.replace(",", " "))[:600] for t in titles]
    X["text_score"] = m["ridge"].predict(m["vec"].transform(texts))
    for c in meta["categorical"]:
        levels = meta["cat_levels"][c]
        X[c] = pd.Categorical([_resolve_level(v, levels)[0] for v in X[c]], categories=levels)
    return X[meta["features"]]


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def score_titles(titles: list[str], ctx: VideoContext, top_k_factors: int = 5) -> list[dict]:
    """Prevê views para cada título no mesmo contexto e explica com SHAP.

    Retorna uma lista (mesma ordem de ``titles``) de dicts com as chaves
    ``title``, ``predicted_views``, ``interval_2x``, ``top_factors``,
    ``title_effect_log``, ``warnings`` e ``notes``.

    - ``predicted_views`` = ``expm1(pred_log)`` sem correção de viés (estimativa
      do tipo mediana na escala original).
    - ``interval_2x``: faixa heurística (~84% das previsões de teste dentro de 2×),
      não um intervalo de predição calibrado.
    - ``title_effect_log``: contribuição das features de texto do título (SHAP) —
      não é efeito causal.
    - ``contribution_log`` e ``title_effect_log`` estão em log(views):
      +0,69 ≈ dobro, −0,69 ≈ metade.
    """
    titles = _validate_titles(titles)
    ctx = _coerce_ctx(ctx)
    _validate_ctx(ctx)
    if isinstance(top_k_factors, bool) or not isinstance(top_k_factors, int) or top_k_factors < 0:
        raise ValueError("'top_k_factors' deve ser um inteiro ≥ 0.")

    m = _load_models()
    warnings = _level_warnings(ctx)
    X = _build_frame(titles, ctx)
    feats = list(X.columns)
    with _PREDICT_LOCK:
        contrib = np.asarray(m["booster"].predict(X, pred_contrib=True), dtype=float)
    # colunas: uma por feature + viés (última). Soma = predição em log1p(views).
    pred_log = contrib.sum(axis=1)
    shap = contrib[:, :-1]
    title_idx = [feats.index(f) for f in TITLE_FEATURES if f in feats]

    out = []
    for i, t in enumerate(titles):
        pred = max(float(np.expm1(pred_log[i])), 0.0)
        order = np.argsort(-np.abs(shap[i]), kind="stable")[:top_k_factors]
        factors = [{
            "feature": feats[j],
            "label_pt": FEATURE_LABELS_PT.get(feats[j], feats[j]),
            "contribution_log": round(float(shap[i, j]), 4),
            "direction": "+" if shap[i, j] >= 0 else "-",
        } for j in order]
        out.append({
            "title": t,
            "predicted_views": int(round(pred)),
            "interval_2x": [int(round(pred / 2)), int(round(pred * 2))],
            "top_factors": factors,
            "title_effect_log": round(float(shap[i, title_idx].sum()), 4),
            "warnings": list(warnings),
            "notes": INTERVAL_NOTE_PT,
        })
    return out


def context_to_dict(ctx: VideoContext) -> dict:
    """Atalho para serializar o contexto (ex.: no rastro do agente)."""
    return asdict(ctx)
