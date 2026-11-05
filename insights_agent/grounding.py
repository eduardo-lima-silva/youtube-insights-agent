"""Checagem determinística de números da resposta contra as saídas das ferramentas.

Ideia: todo número que o agente afirma deveria ter vindo de uma consulta ou do
modelo. ``check_numbers`` extrai os números da resposta (formatos pt-BR como
"1.234,5", "63 mil", "1,2 milhão", "2,3 M", "45%") e marca os que não aparecem,
dentro da tolerância, em nenhum texto de ferramenta desta execução.

Tolerância: 2% relativo OU meia unidade da precisão exibida (``1,2 milhão``
aceita 1,15–1,25 milhão), o que for maior. Inteiros pequenos (< 100) exigem
igualdade com o valor da ferramenta (ou com ele arredondado). Percentuais também
são aceitos se forem razão/variação entre dois números das ferramentas (ex.:
"+55%" calculado a partir de duas medianas; tolerância de meia unidade exibida).

Ignorados: anos (2000–2030), ordinais de lista ("1.", "2)", "3º"), números
copiados da pergunta do usuário e conteúdo de blocos de código.

O resultado é um sinal de auditoria ("não verificado"), não uma prova de erro:
números derivados (diferenças, somas) também aparecem como não verificados.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

_SCALES = {
    "mil": 1e3, "k": 1e3,
    "mi": 1e6, "m": 1e6, "milhão": 1e6, "milhao": 1e6, "milhões": 1e6, "milhoes": 1e6,
    "bi": 1e9, "b": 1e9, "bilhão": 1e9, "bilhao": 1e9, "bilhões": 1e9, "bilhoes": 1e9,
}

_NUM_RE = re.compile(
    r"(?<![\w.,])"
    r"(\d+(?:[.,]\d+)*)"
    r"(?:\s?(%)|\s?(milh[õo]es|milh[ãa]o|bilh[õo]es|bilh[ãa]o|mil|mi|bi|[kKmMbB])(?![\wÀ-ÿ]))?"
)
_FENCE_RE = re.compile(r"```.*?```", re.S)
_ORDINAL_SUFFIX = ("º", "ª", "°")

SMALL_INT = 100
REL_TOL = 0.02
MAX_RATIO_VALUES = 150
RATIO_MIN_VALUE = 10
PROMPT_CONSTANTS = (30.0, 180.0, 200.0)


@dataclass
class _Tok:
    raw: str
    cands: list[tuple[float, float]]   # (valor já escalado, meia unidade já escalada)
    pct: bool
    small_int: bool
    year: bool
    ordinal: bool


def _interpretations(s: str) -> list[tuple[float, int]]:
    """Valores possíveis de um número escrito (pt-BR ou en) -> [(valor, casas decimais)]."""
    has_dot, has_comma = "." in s, "," in s
    if not has_dot and not has_comma:
        return [(float(s), 0)]
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):          # 1.234,5 (pt-BR)
            ip, dp = s.replace(".", "").split(",", 1)
        else:                                      # 1,234.5 (en)
            ip, dp = s.replace(",", "").split(".", 1)
        dp = dp.replace(",", "").replace(".", "")
        return [(float(f"{ip}.{dp}"), len(dp))]
    sep = "." if has_dot else ","
    parts = s.split(sep)
    out: list[tuple[float, int]] = []
    if len(parts[0]) <= 3 and parts[0] != "0" and all(len(p) == 3 for p in parts[1:]):
        out.append((float("".join(parts)), 0))     # separador de milhar
    if len(parts) == 2:
        out.append((float(f"{parts[0]}.{parts[1]}"), len(parts[1])))   # separador decimal
    if not out:
        out.append((float("".join(parts)), 0))
    return out


def _is_ordinal(text: str, start: int, end: int) -> bool:
    nxt = text[end:end + 1]
    if nxt in _ORDINAL_SUFFIX:
        return True
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start].strip().strip("*_#>").strip()
    if prefix in ("", "-") and nxt in (".", ")") and text[end + 1:end + 2] in (" ", "\t", ""):
        return True
    # coluna de ranking: "| 2 | Music | ..." (primeira célula de uma linha de tabela)
    if prefix == "|" and text[end:].lstrip(" ").startswith("|") and (end - start) <= 3:
        return True
    return False


# "54 683" / "54 683": espaço (comum, não separável ou estreito) como separador de milhar.
# Troca por "." mantendo o comprimento do texto (as posições usadas em _is_ordinal não mudam).
_SPACE_THOUSANDS_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d.,]\d)")


def _normalize_spaces(text: str) -> str:
    return _SPACE_THOUSANDS_RE.sub(
        lambda m: m.group(0).replace(" ", ".").replace(" ", ".").replace(" ", "."), text)


def _tokens(text: str) -> list[_Tok]:
    toks = []
    text = _normalize_spaces(text)
    for m in _NUM_RE.finditer(text):
        num, pct, scale_word = m.group(1), m.group(2), m.group(3)
        scale = _SCALES.get(scale_word.lower(), 1.0) if scale_word else 1.0
        interps = _interpretations(num)
        cands = [(v * scale, 0.5 * (10 ** -d) * scale) for v, d in interps]
        plain_int = not pct and not scale_word and len(interps) == 1 and interps[0][1] == 0
        v0 = interps[0][0]
        toks.append(_Tok(
            raw=m.group(0).strip(),
            cands=cands,
            pct=bool(pct),
            small_int=plain_int and v0 < SMALL_INT and "." not in num and "," not in num,
            year=plain_int and 2000 <= v0 <= 2030 and len(num) == 4,
            ordinal=not pct and not scale_word and _is_ordinal(text, m.start(1), m.end(1)),
        ))
    return toks


def _close(v: float, t: float, half_unit: float) -> bool:
    return abs(v - t) <= max(REL_TOL * abs(t), half_unit) + 1e-9


def _tool_values(texts: Iterable[str]) -> list[float]:
    vals: set[float] = set()
    for txt in texts:
        for tok in _tokens(txt):
            for v, _ in tok.cands:
                vals.add(v)
    return sorted(vals)


def _tool_texts(steps: Iterable[Any]) -> list[str]:
    out = []
    for s in steps or []:
        kind = getattr(s, "kind", None) if not isinstance(s, dict) else s.get("kind")
        if kind != "tool":
            continue
        extra = (s.get("extra") if isinstance(s, dict) else getattr(s, "extra", None)) or {}
        preview = s.get("output_preview") if isinstance(s, dict) else getattr(s, "output_preview", "")
        out.append(str(extra.get("result_text") or preview or ""))
    return out


def _matches(tok: _Tok, values: list[float], ratio_pool: list[float]) -> bool:
    for v, hu in tok.cands:
        if tok.small_int:
            # exato, ou conversão exata de unidade de tempo (ex.: duration_s=600 -> "10 min")
            if any(v == t or v == round(t) or v == t / 60 or v == t / 3600 for t in values):
                return True
            continue
        if any(_close(v, t, hu) for t in values):
            return True
        # derivados aceitos (achados do teste ao vivo): conversão de unidade de tempo e
        # diferença absoluta entre dois números das ferramentas ("≈ 5 mil a mais")
        if any(_close(v, d, hu) for t in values for d in (t / 60, t / 3600, t * 60)):
            return True
        if not tok.pct:
            for i, a in enumerate(ratio_pool):
                for b in ratio_pool[i + 1:]:
                    if _close(v, abs(a - b), hu):
                        return True
        if tok.pct:
            if any(_close(v, t * 100, hu) for t in values if abs(t) <= 1.5):
                return True
            # derivado: variação relativa (a > b) ou participação (a < b) entre dois números
            for a in ratio_pool:
                for b in ratio_pool:
                    if a == b:
                        continue
                    r = (a / b - 1) * 100 if a > b else a / b * 100
                    if abs(v - r) <= hu + 1e-9:
                        return True
                    # queda relativa ("≈ 21 % menor" = 1 - 50.702/64.233)
                    if a < b and abs(v - (1 - a / b) * 100) <= hu + 1e-9:
                        return True
    return False


def check_numbers(answer: str, steps: Iterable[Any], question: str = "") -> dict:
    """Devolve ``{"unverified_numbers": [...], "checked": n, "verified": n, "ignored": n}``."""
    text = _FENCE_RE.sub(" ", answer or "")
    values = _tool_values(_tool_texts(steps))
    # só magnitudes "de verdade" (>= 10, fora anos) para evitar coincidências entre números pequenos
    ratio_pool = [v for v in values if v >= RATIO_MIN_VALUE and not (2000 <= v <= 2030 and v == int(v))]
    ratio_pool = ratio_pool[-MAX_RATIO_VALUES:]
    q_vals = [v for tok in _tokens(question or "") for v, _ in tok.cands]
    # constantes das regras do system prompt (n mínimo, limite dos Shorts em s, LIMIT do SQL)
    q_vals += list(PROMPT_CONSTANTS)

    unverified: list[str] = []
    checked = verified = ignored = 0
    for tok in _tokens(text):
        if tok.year or tok.ordinal:
            ignored += 1
            continue
        if any(_close(v, q, hu) if not tok.small_int else v == q
               for v, hu in tok.cands for q in q_vals):
            ignored += 1
            continue
        checked += 1
        if _matches(tok, values, ratio_pool):
            verified += 1
        elif tok.raw not in unverified:
            unverified.append(tok.raw)
    return {"unverified_numbers": unverified, "checked": checked,
            "verified": verified, "ignored": ignored}
