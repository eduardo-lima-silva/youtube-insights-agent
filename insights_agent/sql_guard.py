"""Validação do SQL gerado pelo LLM (AST do sqlglot, dialeto DuckDB — nunca regex).

Camadas, na ordem:
 1. tamanho máximo do texto;
 2. parse com sqlglot; exatamente 1 instrução;
 3. raiz precisa ser SELECT ou operação de conjunto (UNION/INTERSECT/EXCEPT) de SELECTs;
 4. varredura de TODOS os nós: nenhum nó de escrita/DDL/comando, nenhum SELECT ... INTO;
 5. tabelas: só config.ALLOWED_TABLES ou CTEs definidas na própria query, sem qualificador
    (schema/catálogo); nenhuma table function (read_csv, range, glob, query, ...);
 6. funções: denylist (arquivo, rede, ambiente, catálogo interno, pragma_*, duckdb_* ...);
 7. limites de complexidade (DoS): JOINs, CTEs, profundidade de subconsultas, CTE recursiva,
    CROSS JOIN / join sem condição, ORDER BY random(), SELECT * sobre tabelas-base;
 8. LIMIT <= max_rows no SELECT mais externo (adiciona ou reduz).

A conexão em db.py é a defesa seguinte (read-only, enable_external_access=false,
lock_configuration=true, timeout); este módulo não depende dela, e vice-versa.
"""
from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from . import config

DIALECT = "duckdb"
# O sqlglot avisa (WARNING) quando cai no fallback 'Command'; o guard já recusa esses casos.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_SQL_CHARS = 4000
MAX_JOINS = 4
MAX_CTES = 4
MAX_SUBQUERY_DEPTH = 3          # nº máximo de SELECTs aninhados dentro de outro SELECT
STAR_FORBIDDEN_TABLES = frozenset({"videos", "video_country"})

_ALLOWED = ", ".join(config.ALLOWED_TABLES)
_TABLES_MSG = f"Só são permitidas as tabelas {_ALLOWED} (e CTEs definidas com WITH na própria query)."

# Tipos de nó proibidos em qualquer posição da árvore.
_FORBIDDEN_NODE_NAMES = (
    "Command", "Create", "Insert", "Update", "Delete", "Drop", "Alter", "Copy", "Attach",
    "Detach", "Pragma", "Set", "SetItem", "Use", "Install", "Load", "LoadData", "Export",
    "Merge", "Transaction", "Commit", "Rollback", "Describe", "Show", "Summarize",
    "TruncateTable", "Analyze", "Grant", "Revoke", "Cache", "Uncache", "Refresh", "Kill",
    "Comment", "Execute", "Into", "DDL", "DML",
)
_FORBIDDEN_NODES = tuple(t for n in _FORBIDDEN_NODE_NAMES if isinstance(t := getattr(exp, n, None), type))

# Funções proibidas (nome em minúsculas), em qualquer posição.
_DENY_FUNCS = frozenset({
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_json_objects", "read_json_objects_auto", "read_ndjson", "read_ndjson_auto",
    "read_ndjson_objects", "read_text", "read_blob", "read_xlsx", "read_xls",
    "glob", "parquet_scan", "parquet_metadata", "parquet_schema", "parquet_file_metadata",
    "parquet_kv_metadata", "parquet_bloom_probe", "sniff_csv", "csv_scan", "json_scan",
    "query", "query_table", "getenv", "current_setting", "set_variable", "getvariable",
    "sqlite_scan", "sqlite_attach", "postgres_scan", "postgres_attach", "postgres_query",
    "mysql_scan", "mysql_query", "iceberg_scan", "iceberg_metadata", "iceberg_snapshots",
    "delta_scan", "arrow_scan", "arrow_scan_dumb", "pandas_scan", "python_map_function",
    "load_aws_credentials", "which_secret", "httpfs", "http_get", "http_post",
    "json_execute_serialized_sql", "json_serialize_sql", "json_deserialize_sql",
    "checkpoint", "force_checkpoint", "enable_profiling", "disable_profiling",
    "test_all_types", "test_vector_types", "summary", "repeat_row", "setseed",
    "range", "generate_series",   # só como table function; veja _check_tables
    "seq_scan", "index_scan", "unnest_table", "export_database", "import_database",
    "install_extension", "load_extension", "current_database", "current_schema",
    "current_schemas", "current_catalog", "current_user", "session_user", "user", "version",
})
_DENY_PREFIXES = ("duckdb_", "pragma_", "read_", "parquet_", "sqlite_", "postgres_", "mysql_",
                  "iceberg_", "delta_", "arrow_", "http", "s3", "gcs", "azure_", "pg_",
                  "json_execute", "sql_auto_complete", "icu_", "excel_")
_DENY_SUFFIXES = ("_scan", "_metadata", "_attach")
# 'range'/'generate_series' como função escalar (lista) são inofensivas; o perigo é como tabela.
_SCALAR_OK = frozenset({"version"})
# Funções que podem materializar listas/strings gigantes numa única linha (o LIMIT não protege).
_DOS_FUNCS = frozenset({"range", "generate_series", "repeat", "list_resize", "array_resize"})


class SQLRejected(Exception):
    """SQL recusado pelo guard. `reason` é uma mensagem em português, acionável para o LLM."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _func_names(node: exp.Func) -> set[str]:
    names: set[str] = set()
    if isinstance(node, exp.Anonymous):
        names.add(str(node.name).lower())
    else:
        try:
            names.add(node.sql_name().lower())
        except Exception:
            pass
        names.update(n.lower() for n in getattr(node, "_sql_names", []) or [])
        names.add(node.key.lower())
    return {n for n in names if n}


def _is_denied_func(name: str) -> bool:
    return (name in _DENY_FUNCS or name.startswith(_DENY_PREFIXES)
            or name.endswith(_DENY_SUFFIXES))


def _check_functions(tree: exp.Expression) -> None:
    for node in tree.find_all(exp.Func):
        names = _func_names(node)
        in_table = isinstance(node.parent, exp.Table)
        for name in names:
            if name in _DOS_FUNCS:
                raise SQLRejected(
                    f"A função {name.upper()} não é permitida: pode gerar volumes enormes de dados "
                    f"em uma única linha. Consulte as tabelas diretamente. {_TABLES_MSG}")
            if name in _SCALAR_OK and not in_table:
                continue
            if _is_denied_func(name):
                raise SQLRejected(
                    f"A função {name.upper()} não é permitida (acesso a arquivos, rede, ambiente "
                    f"ou catálogo interno). {_TABLES_MSG}")


def _cte_names(tree: exp.Expression) -> set[str]:
    return {str(c.alias_or_name).lower() for c in tree.find_all(exp.CTE)}


def _check_tables(tree: exp.Expression, ctes: set[str]) -> None:
    allowed = {t.lower() for t in config.ALLOWED_TABLES} | ctes
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            fn = t.this.sql(dialect=DIALECT)[:60] if t.this is not None else "?"
            raise SQLRejected(
                f"Table functions não são permitidas no FROM/JOIN (encontrado: {fn}). {_TABLES_MSG}")
        if t.args.get("db") or t.args.get("catalog"):
            raise SQLRejected(
                f"Use o nome da tabela sem schema/catálogo (encontrado: {t.sql(dialect=DIALECT)}). "
                f"{_TABLES_MSG}")
        name = str(t.name).lower()
        if name not in allowed:
            raise SQLRejected(f"Tabela não permitida: {t.name!r}. {_TABLES_MSG}")
    # Fontes de linhas que não são tabelas (VALUES, UNNEST, LATERAL) só podem gerar dados
    # inofensivos, mas não fazem parte do escopo analítico: recusamos para manter a superfície mínima.
    for node in tree.find_all(exp.Lateral):
        raise SQLRejected("LATERAL não é permitido. Use JOIN ... ON ou CTEs.")


def _select_depth(node: exp.Expression) -> int:
    depth, p = 0, node.parent
    while p is not None:
        if isinstance(p, exp.Select):
            depth += 1
        p = p.parent
    return depth


def _check_complexity(tree: exp.Expression) -> None:
    joins = list(tree.find_all(exp.Join))
    if len(joins) > MAX_JOINS:
        raise SQLRejected(f"Consulta com {len(joins)} JOINs; o máximo é {MAX_JOINS}. "
                          "Simplifique ou agregue em etapas com CTEs.")
    for j in joins:
        kind = (j.args.get("kind") or "").upper()
        method = (j.args.get("method") or "").upper()
        if kind == "CROSS" or method in ("NATURAL", "POSITIONAL"):
            raise SQLRejected(f"{kind or method} JOIN não é permitido (risco de produto cartesiano). "
                              "Use JOIN ... ON <condição> ou USING (coluna).")
        if not (j.args.get("on") or j.args.get("using")):
            raise SQLRejected("Todo JOIN precisa de condição ON ou USING; junções por vírgula e "
                              "CROSS JOIN não são permitidas (risco de produto cartesiano).")

    ctes = list(tree.find_all(exp.CTE))
    if len(ctes) > MAX_CTES:
        raise SQLRejected(f"Consulta com {len(ctes)} CTEs; o máximo é {MAX_CTES}.")
    for w in tree.find_all(exp.With):
        if w.args.get("recursive"):
            raise SQLRejected("CTEs recursivas (WITH RECURSIVE) não são permitidas.")

    for sel in tree.find_all(exp.Select):
        if _select_depth(sel) > MAX_SUBQUERY_DEPTH:
            raise SQLRejected(f"Subconsultas aninhadas demais (máximo {MAX_SUBQUERY_DEPTH} níveis). "
                              "Reescreva com CTEs.")

    for order in tree.find_all(exp.Order):
        for node in order.walk():
            if isinstance(node, exp.Rand) or (
                    isinstance(node, exp.Func) and _func_names(node) & {"random", "rand"}):
                raise SQLRejected("ORDER BY random() não é permitido. Ordene por uma coluna "
                                  "(ou use USING SAMPLE para amostrar).")


def _direct_sources(sel: exp.Select) -> list[exp.Expression]:
    out = []
    frm = sel.args.get("from_") or sel.args.get("from")
    if frm is not None and frm.this is not None:
        out.append(frm.this)
    for j in sel.args.get("joins") or []:
        out.append(j.this)
    return out


def _check_star(tree: exp.Expression, ctes: set[str]) -> None:
    for sel in tree.find_all(exp.Select):
        base = {}   # alias/nome (minúsculo) -> nome da tabela-base proibida
        for src in _direct_sources(sel):
            if isinstance(src, exp.Table) and isinstance(src.this, exp.Identifier):
                name = str(src.name).lower()
                if name in STAR_FORBIDDEN_TABLES and name not in ctes:
                    base[name] = name
                    if src.alias:
                        base[str(src.alias).lower()] = name
        if not base:
            continue
        for proj in sel.expressions:
            target = proj.this if isinstance(proj, exp.Alias) else proj
            if isinstance(target, exp.Star):
                hit = sorted(set(base.values()))[0]
            elif isinstance(target, exp.Column) and isinstance(target.this, exp.Star):
                hit = base.get(str(target.table).lower())
            else:
                continue
            if hit:
                raise SQLRejected(
                    f"SELECT * direto sobre a tabela {hit} não é permitido: liste só as colunas "
                    "necessárias (COUNT(*) é permitido).")


def _limit_value(node: exp.Expression) -> int | None:
    lim = node.args.get("limit")
    if lim is None:
        return None
    if isinstance(lim, exp.Fetch):     # FETCH FIRST n ROWS ONLY
        cnt = lim.args.get("count")
        ok = isinstance(cnt, exp.Literal) and str(cnt.this).isdigit() and not lim.args.get("percent")
        return int(cnt.this) if ok else -1
    e = lim.args.get("expression")
    if isinstance(e, exp.Literal) and not e.is_string and str(e.this).isdigit():
        opts = lim.args.get("limit_options")
        if (opts is not None and opts.args.get("percent")) or lim.args.get("percent"):
            return -1   # LIMIT n% vira LIMIT max_rows
        return int(e.this)
    return -1   # expressão não-literal: força o limite padrão


def _apply_limit(tree: exp.Expression, max_rows: int) -> None:
    current = _limit_value(tree)
    if current is None or current < 0 or current > max_rows:
        current = max_rows
    tree.set("limit", exp.Limit(expression=exp.Literal.number(current)))


def validate_sql(sql: str, max_rows: int | None = None) -> str:
    """Valida o SQL e devolve a versão normalizada (dialeto DuckDB, sem comentários, com LIMIT).

    Levanta SQLRejected(reason) se a consulta não for um SELECT seguro dentro das regras.
    `max_rows` (padrão config.MAX_ROWS) é o teto do LIMIT do SELECT mais externo.
    """
    max_rows = config.MAX_ROWS if max_rows is None else int(max_rows)
    if not isinstance(sql, str) or not sql.strip():
        raise SQLRejected("SQL vazio. Envie uma única instrução SELECT.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLRejected(f"SQL longo demais ({len(sql)} caracteres; máximo {MAX_SQL_CHARS}). "
                          "Simplifique a consulta.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except (ParseError, TokenError) as e:
        msg = str(e).splitlines()[0][:200]
        raise SQLRejected(f"SQL inválido (erro de sintaxe no dialeto DuckDB): {msg}") from None
    if len(statements) != 1:
        raise SQLRejected(f"Envie exatamente 1 instrução SQL (recebidas {len(statements)}). "
                          "Não use ';' para encadear comandos.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        raise SQLRejected(f"Só consultas SELECT são permitidas (recebido: {tree.key.upper()}). "
                          "Escrita, DDL, PRAGMA, SET, COPY, ATTACH, INSTALL/LOAD etc. são proibidos.")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            what = "SELECT ... INTO" if isinstance(node, exp.Into) else node.key.upper()
            raise SQLRejected(f"Comando não permitido dentro da consulta: {what}. "
                              "Só leitura com SELECT.")

    ctes = _cte_names(tree)
    _check_tables(tree, ctes)
    _check_functions(tree)
    _check_complexity(tree)
    _check_star(tree, ctes)
    _apply_limit(tree, max_rows)

    try:
        return tree.sql(dialect=DIALECT, comments=False)
    except Exception as e:  # pragma: no cover - geração raramente falha
        raise SQLRejected(f"Não foi possível normalizar o SQL: {e}") from None
