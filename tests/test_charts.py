"""Testes de charts.build_chart."""
import json

import altair as alt
import pandas as pd
import pytest

from insights_agent.charts import build_chart


@pytest.fixture
def cat_df():
    return pd.DataFrame({"category": ["Music", "Gaming", "Sports"], "n": [10, 30, 20],
                         "grp": ["a", "b", "a"]})


@pytest.fixture
def time_df():
    return pd.DataFrame({"m": ["2025-01-01", "2025-02-01", "2025-03-01"], "v": [1.5, 2.5, 2.0]})


def _spec(chart):
    d = chart.to_dict()
    json.dumps(d)          # serializável (vai para o trace)
    return d


def test_bar_sorted_desc(cat_df):
    d = _spec(build_chart(cat_df, "bar", "category", "n", None, "Vídeos por categoria"))
    assert d["mark"]["type"] == "bar"
    assert d["encoding"]["x"]["sort"] == "-y"
    assert d["encoding"]["x"]["type"] == "nominal"
    assert d["encoding"]["y"]["type"] == "quantitative"
    assert d["width"] == "container"
    assert d["title"] == "Vídeos por categoria"
    assert "tooltip" in d["encoding"]


def test_line_temporal(time_df):
    c = build_chart(time_df, "line", "m", "v", None, "Série")
    d = _spec(c)
    assert d["mark"]["type"] == "line"
    assert d["encoding"]["x"]["type"] == "temporal"


def test_scatter_with_color():
    df = pd.DataFrame({"dur": [10, 200, 600], "views": [1e3, 5e4, 2e5], "cat": ["x", "y", "x"]})
    d = _spec(build_chart(df, "scatter", "dur", "views", "cat", "Dispersão"))
    assert d["mark"]["type"] == "circle"
    assert d["encoding"]["x"]["type"] == "quantitative"
    assert d["encoding"]["color"]["field"] == "cat"


def test_bar_grouped_color(cat_df):
    d = _spec(build_chart(cat_df, "bar", "category", "n", "grp", ""))
    assert d["encoding"]["color"]["field"] == "grp"
    assert "xOffset" in d["encoding"]


def test_returns_altair_chart(cat_df):
    assert isinstance(build_chart(cat_df, "bar", "category", "n", None, "t"), alt.Chart)


def test_invalid_kind(cat_df):
    with pytest.raises(ValueError, match="Tipo de gráfico inválido"):
        build_chart(cat_df, "pie", "category", "n", None, "t")


def test_missing_column(cat_df):
    with pytest.raises(ValueError, match="inexistente"):
        build_chart(cat_df, "bar", "category", "views", None, "t")


def test_missing_color_column(cat_df):
    with pytest.raises(ValueError, match="inexistente"):
        build_chart(cat_df, "bar", "category", "n", "nope", "t")


def test_non_numeric_y_for_bar(cat_df):
    with pytest.raises(ValueError, match="numérico"):
        build_chart(cat_df, "bar", "n", "category", None, "t")


def test_empty_df():
    with pytest.raises(ValueError, match="vazio"):
        build_chart(pd.DataFrame({"a": [], "b": []}), "bar", "a", "b", None, "t")


def test_from_query_result_rows():
    """Integração com o formato de QueryResult (datas como str, números como float/int)."""
    df = pd.DataFrame([["2025-01-01", 3], ["2025-01-02", 5]], columns=["day", "n_videos"])
    d = _spec(build_chart(df, "line", "day", "n_videos", None, ""))
    assert d["encoding"]["x"]["type"] == "temporal"
