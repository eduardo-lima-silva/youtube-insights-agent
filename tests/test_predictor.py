"""Testes do predictor (usa os modelos reais do projeto 1)."""
import math
import time
from dataclasses import replace

import numpy as np
import pytest

from insights_agent import config, predictor
from insights_agent.predictor import VideoContext, context_schema, score_titles

pytestmark = pytest.mark.skipif(
    not (config.LGBM_MODEL.exists() and config.TEXT_MODEL.exists() and config.MODEL_META.exists()),
    reason="modelos do projeto 1 não encontrados",
)

CTX = VideoContext(category="Sports", duration_s=540, channel_subs=850_000,
                   channel_views=120_000_000, channel_videos=900, channel_age_days=2200,
                   market="Brazil", pub_hour_utc=21, pub_dow=5, horizon_days=7,
                   tags="futebol,gols,reação", desc_len=800)
TITLES = [
    "Reagindo aos MELHORES gols da rodada! #futebol",
    "Gols da rodada",
    "VOCÊ NÃO VAI ACREDITAR NESSE GOL 😱🔥",
]


def test_shapes_and_keys():
    res = score_titles(TITLES, CTX)
    assert isinstance(res, list) and len(res) == 3
    for r, t in zip(res, TITLES):
        assert set(r) == {"title", "predicted_views", "interval_2x", "top_factors",
                          "title_effect_log", "warnings", "notes"}
        assert r["title"] == t
        assert r["warnings"] == []
        assert "faixa heurística" in r["notes"] and "não um intervalo de predição calibrado" in r["notes"]
        assert isinstance(r["predicted_views"], int) and r["predicted_views"] > 0
        lo, hi = r["interval_2x"]
        assert lo <= r["predicted_views"] <= hi
        assert abs(lo - r["predicted_views"] / 2) <= 1 and abs(hi - r["predicted_views"] * 2) <= 1
        assert isinstance(r["title_effect_log"], float) and math.isfinite(r["title_effect_log"])
        assert len(r["top_factors"]) == 5
        mags = [abs(f["contribution_log"]) for f in r["top_factors"]]
        assert mags == sorted(mags, reverse=True)
        for f in r["top_factors"]:
            assert set(f) == {"feature", "label_pt", "contribution_log", "direction"}
            assert f["direction"] == ("+" if f["contribution_log"] >= 0 else "-")
            assert f["feature"] in predictor.FEATURE_LABELS_PT
            assert f["label_pt"] and f["label_pt"] != f["feature"]


def test_top_k_param():
    assert len(score_titles(TITLES[:1], CTX, top_k_factors=2)[0]["top_factors"]) == 2
    assert score_titles(TITLES[:1], CTX, top_k_factors=0)[0]["top_factors"] == []


def test_deterministic():
    a = score_titles(TITLES, CTX)
    b = score_titles(TITLES, CTX)
    assert a == b


def test_batch_equals_single():
    batch = score_titles(TITLES, CTX)
    for t, r in zip(TITLES, batch):
        assert score_titles([t], CTX)[0] == r


def test_shap_sums_to_prediction():
    m = predictor._load_models()
    X = predictor._build_frame(TITLES, CTX)
    raw = m["booster"].predict(X)
    contrib = m["booster"].predict(X, pred_contrib=True)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, rtol=1e-6, atol=1e-6)
    res = score_titles(TITLES, CTX)
    for r, p in zip(res, raw):
        assert abs(r["predicted_views"] - np.expm1(p)) <= 1


def test_different_title_changes_text_features():
    X = predictor._build_frame(["Gols da rodada", "Tutorial de Python para iniciantes: aula 1"], CTX)
    assert X["text_score"].iloc[0] != X["text_score"].iloc[1]
    res = score_titles(["Gols da rodada", "Tutorial de Python para iniciantes: aula 1"], CTX)
    assert all(math.isfinite(r["predicted_views"]) for r in res)
    assert res[0]["title_effect_log"] != res[1]["title_effect_log"]


def test_unknown_levels_map_to_fallback():
    X = predictor._build_frame(["teste"], replace(CTX, market="Atlantis", category="Nonexistent"))
    assert X["market"].iloc[0] == "Other"
    assert X["channel_country"].iloc[0] == "Other"
    assert X["category"].iloc[0] == "Unknown"   # 'category' não tem nível 'Other' no treino
    X = predictor._build_frame(["teste"], replace(CTX, market="brazil", category="sports"))
    assert X["market"].iloc[0] == "Brazil" and X["category"].iloc[0] == "Sports"


def test_unknown_levels_reported_in_warnings():
    r = score_titles(["teste"], replace(CTX, market="Atlantis", category="Nonexistent"))[0]
    assert len(r["warnings"]) == 2
    assert any("market" in w and "Other" in w for w in r["warnings"])
    assert any("category" in w for w in r["warnings"])
    assert score_titles(["teste"], replace(CTX, market="brazil"))[0]["warnings"] == []


def test_model_info():
    info = predictor.MODEL_INFO
    assert info["lgbm"]["file"] == config.LGBM_MODEL.name and info["lgbm"]["size_bytes"] > 0
    assert info["lgbm"]["mtime_utc"]
    assert info["best_iteration"] == predictor._meta()["best_iteration"]
    assert info["n_features"] == len(predictor._meta()["features"]) == 37


def test_title_effect_label():
    assert predictor.TITLE_EFFECT_LABEL_PT == (
        "contribuição das features de texto do título (SHAP) — não é efeito causal")


def test_no_channel_history():
    X = predictor._build_frame(["teste"], CTX)
    assert X["ch_prev_count"].iloc[0] == 0
    assert np.isnan(X["ch_prev_mean_log_views"].iloc[0]) and np.isnan(X["ch_prev_max_log_views"].iloc[0])


def test_dict_context_accepted():
    from dataclasses import asdict
    assert score_titles(TITLES[:1], asdict(CTX)) == score_titles(TITLES[:1], CTX)


@pytest.mark.parametrize("titles", [
    [], ["a", "b", "c", "d"], [""], ["   "], ["x" * 151], "um título solto", [123], None,
])
def test_invalid_titles(titles):
    with pytest.raises(ValueError):
        score_titles(titles, CTX)


@pytest.mark.parametrize("change", [
    {"pub_hour_utc": 24}, {"pub_hour_utc": -1}, {"pub_dow": 7}, {"horizon_days": 0},
    {"horizon_days": 31}, {"channel_subs": -1}, {"channel_views": -1}, {"channel_videos": 0},
    {"duration_s": 0}, {"duration_s": 43201}, {"channel_age_days": 8001}, {"desc_len": 5001},
    {"channel_subs": float("nan")}, {"channel_subs": "muitos"}, {"category": 5},
    {"tags": "x" * 501}, {"pub_dow": True},
])
def test_invalid_context(change):
    with pytest.raises(ValueError):
        score_titles(TITLES[:1], replace(CTX, **change))


def test_invalid_context_type_and_extra_keys():
    with pytest.raises(ValueError):
        score_titles(TITLES[:1], "Sports")
    with pytest.raises(ValueError):
        score_titles(TITLES[:1], {"category": "Sports", "foo": 1})


def test_error_messages_in_portuguese():
    with pytest.raises(ValueError, match="título"):
        score_titles([""], CTX)


def test_horizon_monotonic():
    d1 = score_titles(TITLES[:1], replace(CTX, horizon_days=1))[0]["predicted_views"]
    d30 = score_titles(TITLES[:1], replace(CTX, horizon_days=30))[0]["predicted_views"]
    assert d30 >= d1


def test_context_schema_strict():
    s = context_schema()
    assert s["type"] == "object" and s["additionalProperties"] is False
    names = [f for f in VideoContext.__dataclass_fields__]
    assert set(s["properties"]) == set(names) and set(s["required"]) == set(names)
    allowed = {"type", "enum", "description"}
    for name, p in s["properties"].items():
        assert set(p) <= allowed, name
        assert p["description"]
    assert "Sports" in s["properties"]["category"]["enum"]
    assert "Brazil" in s["properties"]["market"]["enum"] and "Other" in s["properties"]["market"]["enum"]
    assert s["properties"]["duration_s"]["type"] == "integer"
    assert s["properties"]["tags"]["type"] == "string"


def test_runtime_after_warmup():
    score_titles(TITLES, CTX)  # aquecimento (carga dos modelos)
    t0 = time.perf_counter()
    score_titles(TITLES, CTX)
    assert time.perf_counter() - t0 < 2.0
