"""Spec de gráfico (vinda do LLM) -> altair.Chart, com validação e padrões sensatos."""
from __future__ import annotations

import altair as alt
import pandas as pd

KINDS = ("bar", "line", "scatter")
MAX_POINTS = 5000


def _is_temporal(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(20)
        if sample.empty:
            return False
        # datas ISO (AAAA-MM-DD[ hh:mm:ss]) vindas de QueryResult (serializadas como str)
        return bool(sample.str.match(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?").all())
    return False


def _enc_type(s: pd.Series) -> str:
    if _is_temporal(s):
        return "temporal"
    if pd.api.types.is_bool_dtype(s):
        return "nominal"
    if pd.api.types.is_numeric_dtype(s):
        return "quantitative"
    return "nominal"


def build_chart(df: pd.DataFrame, kind: str, x: str, y: str, color: str | None = None,
                title: str = "") -> alt.Chart:
    """Cria um gráfico Altair a partir de um DataFrame.

    kind ∈ {"bar", "line", "scatter"}; x/y (e color, se dado) precisam ser colunas de df.
    Levanta ValueError com mensagem em português se a spec for inválida.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("build_chart espera um pandas.DataFrame.")
    if kind not in KINDS:
        raise ValueError(f"Tipo de gráfico inválido: {kind!r}. Use um de: {', '.join(KINDS)}.")
    cols = list(df.columns)
    missing = [c for c in (x, y) + ((color,) if color else ()) if c not in cols]
    if missing:
        raise ValueError(f"Coluna(s) inexistente(s) no resultado: {', '.join(map(repr, missing))}. "
                         f"Colunas disponíveis: {', '.join(map(str, cols))}.")
    if x == y:
        raise ValueError("x e y precisam ser colunas diferentes.")
    if df.empty:
        raise ValueError("O resultado está vazio; não há o que plotar.")

    data = df.copy()
    if len(data) > MAX_POINTS:
        data = data.head(MAX_POINTS)

    x_type, y_type = _enc_type(data[x]), _enc_type(data[y])
    if x_type == "temporal":
        data[x] = pd.to_datetime(data[x], errors="coerce")
    if y_type == "temporal":
        data[y] = pd.to_datetime(data[y], errors="coerce")
    if kind in ("bar", "line") and y_type != "quantitative":
        num = pd.to_numeric(data[y], errors="coerce")
        if num.notna().sum() == 0:
            raise ValueError(f"Para gráfico '{kind}', y ({y!r}) precisa ser numérico.")
        data[y], y_type = num, "quantitative"
    if kind == "scatter" and x_type != "quantitative":
        num = pd.to_numeric(data[x], errors="coerce")
        if num.notna().all():
            data[x], x_type = num, "quantitative"

    x_title, y_title = str(x), str(y)
    if kind == "bar":
        if x_type == "nominal":
            x_enc = alt.X(x, type="nominal", sort="-y", title=x_title,
                          axis=alt.Axis(labelAngle=-40, labelLimit=160))
        else:
            x_enc = alt.X(x, type=x_type, title=x_title)
        mark = alt.Chart(data).mark_bar()
    elif kind == "line":
        x_enc = alt.X(x, type=x_type, title=x_title,
                      sort="ascending" if x_type != "nominal" else None)
        mark = alt.Chart(data).mark_line(point=len(data) <= 60)
    else:
        x_enc = alt.X(x, type=x_type, title=x_title)
        mark = alt.Chart(data).mark_circle(size=40, opacity=0.6)

    y_enc = alt.Y(y, type=y_type, title=y_title)
    y_tip = (alt.Tooltip(y, type=y_type, format=",.4~g") if y_type == "quantitative"
             else alt.Tooltip(y, type=y_type))
    tooltip = [alt.Tooltip(x, type=x_type), y_tip]
    enc = {"x": x_enc, "y": y_enc}
    if color:
        c_type = _enc_type(data[color])
        enc["color"] = alt.Color(color, type=c_type if c_type != "temporal" else "nominal",
                                 title=str(color))
        tooltip.append(alt.Tooltip(color, type=c_type if c_type != "temporal" else "nominal"))
    if kind == "bar" and color and x_type == "nominal":
        enc["xOffset"] = alt.XOffset(color, type="nominal")

    chart = mark.encode(**enc, tooltip=tooltip).properties(width="container", height=320)
    if title:
        chart = chart.properties(title=str(title))
    return chart
