"""Checagem determinística dos números da resposta contra as saídas das ferramentas."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insights_agent.grounding import _interpretations, check_numbers  # noqa: E402
from insights_agent.trace import TraceStep  # noqa: E402

TOOL_OUT = """result_id: r1
linhas: 2
| faixa | mediana_views | n |
|---|---|---|
| 4-8 min | 41000.0 | 2210 |
| 8-12 min | 63412.0 | 1520 |
| total | 1234567 | 3730 |
share_shorts: 0.453
"""


def steps(text=TOOL_OUT):
    return [TraceStep(kind="llm", name="m", output_preview="99999"),
            TraceStep(kind="tool", name="run_sql", extra={"result_text": text})]


def unverified(answer, question="", st=None):
    return check_numbers(answer, st if st is not None else steps(), question=question)["unverified_numbers"]


def test_interpretations_ptbr_and_en():
    assert (1234.5, 1) in _interpretations("1.234,5")
    assert (1234.5, 1) in _interpretations("1,234.5")
    vals = [v for v, _ in _interpretations("1.234")]
    assert 1234.0 in vals and 1.234 in vals                 # ambíguo: milhar ou decimal
    assert _interpretations("2,3") == [(2.3, 1)]
    assert _interpretations("63412") == [(63412.0, 0)]


def test_formats_are_verified():
    ans = ("A mediana foi de 63 mil views (n = 1.520) contra 41.000 (n = 2.210). "
           "No total, 1,2 milhão de views; também 1,23 M. Shorts são 45% e 45,3% da amostra. "
           "Valor exato: 63.412,0.")
    assert unverified(ans) == []


def test_invented_numbers_are_flagged():
    ans = "A mediana foi de 87 mil views (n = 1520) e 38% dos vídeos são Shorts; 57 canais."
    flagged = unverified(ans)
    assert "87 mil" in flagged
    assert "38%" in flagged
    assert "57" in flagged                                    # inteiro pequeno sem correspondência exata
    assert "1520" not in flagged


def test_tolerance_two_percent():
    assert unverified("cerca de 64.500 views") == []         # 1,7% de 63.412
    assert unverified("cerca de 66.000 views") == ["66.000"]  # 4%


def test_derived_percentage_from_two_tool_numbers():
    # 63412 / 41000 = 1,5466 -> +54,7%
    assert unverified("Vídeos de 8-12 min têm mediana 55% maior.") == []
    assert unverified("Vídeos de 8-12 min têm mediana 54,7% maior.") == []


def test_ignores_years_ordinals_and_question_numbers():
    ans = ("1. Em 2025 a base cresceu.\n2) Segundo ponto.\n- 3º lugar: Gaming.\n"
           "Comparei vídeos de 8 a 12 minutos com 4 a 8 minutos.")
    r = check_numbers(ans, steps(""), question="Compare vídeos de 8 a 12 minutos contra 4 a 8 minutos")
    assert r["unverified_numbers"] == []
    assert r["checked"] == 0 and r["ignored"] >= 7


def test_ignores_code_blocks_and_only_uses_tool_steps():
    ans = "Veja:\n```sql\nSELECT 777\n```\nResultado: 99999 views."
    assert unverified(ans) == ["99999"]                       # 99999 só aparece num passo llm


def test_small_integer_requires_exact():
    assert unverified("n = 2 grupos", st=steps("linhas: 2")) == []
    assert unverified("n = 3 grupos", st=steps("linhas: 2")) == ["3"]


def test_dict_steps_from_saved_trace():
    st = [{"kind": "tool", "name": "run_sql", "output_preview": "", "extra": {"result_text": "mediana 63412"}}]
    assert check_numbers("63 mil", st)["unverified_numbers"] == []


def test_counts():
    r = check_numbers("63 mil e 87 mil", steps())
    assert r["checked"] == 2 and r["verified"] == 1 and r["unverified_numbers"] == ["87 mil"]


def test_space_as_thousands_separator():
    """Achado no teste ao vivo com o Groq: o modelo escreveu "54 683" (espaço como milhar)."""
    from insights_agent.grounding import check_numbers
    from insights_agent.trace import TraceStep
    steps = [TraceStep(kind="tool", name="run_sql", extra={"result_text": "| n_videos_brasil |\n|---|\n| 54683 |"})]
    for ans in ("Foram 54 683 vídeos.", "Foram 54\u00a0683 vídeos.", "Foram 54\u202f683 vídeos."):
        g = check_numbers(ans, steps)
        assert g["unverified_numbers"] == [], (ans, g)
    # números separados por espaço que NÃO formam milhar continuam independentes
    g = check_numbers("Entre 12 e 54683 vídeos, n = 30.", steps)
    assert "12" in " ".join(map(str, g["unverified_numbers"])) or g["checked"] >= 2


def test_rank_column_in_markdown_table_is_ignored():
    """Achado no teste ao vivo: a coluna 'Rank' de uma tabela era marcada como número não verificado."""
    from insights_agent.grounding import check_numbers
    from insights_agent.trace import TraceStep
    steps = [TraceStep(kind="tool", name="run_sql", extra={"result_text": "| category | n |\n| Gaming | 9861 |\n| Music | 4906 |"})]
    ans = "| Rank | Categoria | Vídeos |\n|---|---|---|\n| 1 | Gaming | 9 861 |\n| 2 | Music | 4 906 |"
    g = check_numbers(ans, steps)
    assert g["unverified_numbers"] == [], g


def test_derived_difference_and_time_units_are_verified():
    """Achados da avaliação ao vivo (cmp_02 e tit_04): números derivados legítimos."""
    from insights_agent.grounding import check_numbers
    from insights_agent.trace import TraceStep
    steps = [TraceStep(kind="tool", name="run_sql",
                       extra={"result_text": "| country | n |\n| Brazil | 32170 |\n| Mexico | 27160 |"}),
             TraceStep(kind="tool", name="score_titles", extra={"result_text": "Contexto usado: duration_s=600"})]
    g = check_numbers("O Brasil teve ≈ 5 mil vídeos a mais; vídeo de 10 min.", steps)
    assert g["unverified_numbers"] == [], g
    # um número inventado continua sendo sinalizado
    g = check_numbers("O Brasil teve 41.300 vídeos.", steps)
    assert g["unverified_numbers"], g
