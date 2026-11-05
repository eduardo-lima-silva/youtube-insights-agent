"""Calcula as respostas de referência executando cada `reference_sql` no warehouse.

    python -m evals.build_references [--warehouse PATH] [--cases evals/cases.jsonl]

Escreve evals/reference_answers.json:
    {case_id: {"columns": [...], "rows": [...primeiras 50...], "row_count": n, "sql": str}}

Falha (exit 1, sem gravar) se: algum SQL der erro; a SQL citar tabela fora do warehouse;
a coluna usada em `answer_check` não existir no resultado; ou o resultado vier vazio/zero
sem o caso estar marcado com `expect_zero`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import math
import sys
from pathlib import Path

import duckdb

from insights_agent import config

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.jsonl"
REF_PATH = EVALS_DIR / "reference_answers.json"
MAX_SAVED_ROWS = 50


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"cases.jsonl linha {i}: JSON inválido: {e}")
    return cases


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    return str(v)


def referenced_tables(sql: str) -> set[str]:
    """Tabelas reais citadas (CTEs excluídas), via AST do sqlglot."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="duckdb")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes}


def _check_columns(case: dict) -> list[str]:
    spec = case.get("answer_check") or {}
    specs = spec if isinstance(spec, list) else [spec]
    return [s["column"] for s in specs if s.get("column")]


def build(warehouse: Path, cases: list[dict]) -> tuple[dict, list[str], list[str]]:
    con = duckdb.connect(str(warehouse), read_only=True)
    refs, errors, warnings = {}, [], []
    for c in cases:
        sql = c.get("reference_sql")
        if not sql:
            if c.get("expected_behavior") == "answer" and (c.get("answer_check") or {}).get("type", "none") != "none":
                errors.append(f"{c['id']}: caso com answer_check mas sem reference_sql")
            continue
        try:
            bad = referenced_tables(sql) - set(config.ALLOWED_TABLES)
        except Exception as e:
            errors.append(f"{c['id']}: SQL não parseia com sqlglot: {e}")
            continue
        if bad:
            errors.append(f"{c['id']}: tabelas fora do warehouse: {sorted(bad)}")
            continue
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        except Exception as e:
            errors.append(f"{c['id']}: ERRO SQL: {e}")
            continue
        for col in _check_columns(c):
            if col not in cols:
                errors.append(f"{c['id']}: coluna '{col}' do answer_check não está no resultado {cols}")
        first = rows[0][0] if rows and rows[0] else None
        empty = not rows or all(v in (None, 0) for v in rows[0])
        is_zero_like = not rows or first in (None, 0)
        if c.get("expect_zero"):
            if not is_zero_like:
                errors.append(f"{c['id']}: marcado expect_zero, mas a referência deu {rows[:1]}")
        elif empty:
            errors.append(f"{c['id']}: referência vazia/zero sem expect_zero")
        refs[c["id"]] = {"columns": cols, "rows": [[_jsonable(v) for v in r] for r in rows[:MAX_SAVED_ROWS]],
                         "row_count": len(rows), "sql": sql}
    con.close()
    return refs, errors, warnings


def _preview(ref: dict) -> str:
    if not ref["rows"]:
        return "(0 linhas)"
    row = ref["rows"][0]
    parts = []
    for k, v in zip(ref["columns"], row):
        if isinstance(v, float):
            v = f"{v:,.4g}"
        s = str(v)
        parts.append(f"{k}={s[:40]}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warehouse", type=Path, default=config.WAREHOUSE_FULL if config.WAREHOUSE_FULL.exists() else config.WAREHOUSE_PATH)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--out", type=Path, default=REF_PATH)
    args = ap.parse_args(argv)
    if not args.warehouse.exists():
        print(f"warehouse não encontrado: {args.warehouse}", file=sys.stderr)
        return 1
    cases = load_cases(args.cases)
    refs, errors, warnings = build(args.warehouse, cases)
    print(f"warehouse: {args.warehouse}")
    print(f"{len(cases)} casos; {len(refs)} referências calculadas\n")
    for cid, ref in refs.items():
        print(f"  {cid:8s} linhas={ref['row_count']:<4d} {_preview(ref)}")
    for w in warnings:
        print("AVISO:", w)
    if errors:
        print("\nFALHOU:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1
    payload = {"_meta": {"warehouse": args.warehouse.name, "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         "n_references": len(refs)}}
    payload.update(refs)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOK -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
