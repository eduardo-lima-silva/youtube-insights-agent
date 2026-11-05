"""Conexão DuckDB somente leitura e execução segura de consultas.

Defesa em camadas (independente do guard):
- arquivo aberto com read_only=True (escrita falha no próprio DuckDB);
- SET enable_external_access=false (nada de arquivos/rede: read_csv, COPY, ATTACH, httpfs...);
- memory_limit/threads de config; por fim SET lock_configuration=true (o SQL não consegue
  desfazer as opções). As opções de segurança são VERIFICADAS logo após conectar; se a
  verificação falhar, nenhuma consulta é executada;
- timeout por consulta (threading.Timer -> cursor.interrupt()); após um timeout a conexão é
  descartada e recriada na próxima chamada;
- no máximo MAX_ROWS + 1 linhas são buscadas (fetchmany).
"""
from __future__ import annotations

import datetime as _dt
import decimal
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb

from . import config
from .sql_guard import SQLRejected, validate_sql  # noqa: F401  (SQLRejected reexportado)

__all__ = ["QueryResult", "QueryError", "QueryTimeout", "SQLRejected", "get_connection",
           "run_query", "reset_connection"]


class QueryError(RuntimeError):
    """Erro ao executar a consulta (mensagem em PT, pronta para voltar ao LLM)."""


class QueryTimeout(QueryError):
    """A consulta excedeu config.SQL_TIMEOUT_S e foi interrompida."""


class InsecureConnectionError(RuntimeError):
    """As opções de segurança da conexão não puderam ser confirmadas."""


_LOCK = threading.RLock()
_CON: duckdb.DuckDBPyConnection | None = None
_CON_PATH: str | None = None

_REQUIRED = {"enable_external_access": False, "lock_configuration": True}


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return {"true": True, "false": False}.get(v.strip().lower())
    return None


def _verify(con: duckdb.DuckDBPyConnection) -> None:
    for name, expected in _REQUIRED.items():
        try:
            val = con.execute(f"SELECT current_setting('{name}')").fetchone()[0]
        except duckdb.Error as e:
            raise InsecureConnectionError(f"Não foi possível ler a opção {name}: {e}") from e
        if _as_bool(val) is not expected:
            raise InsecureConnectionError(
                f"Conexão insegura: {name}={val!r} (esperado {expected}). Consultas bloqueadas.")


_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
          "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def _to_bytes(text: str) -> float:
    import re
    m = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]+)\s*", str(text))
    if not m:
        raise ValueError(text)
    return float(m.group(1)) * _UNITS[m.group(2).lower()]


def _verify_resources(con: duckdb.DuckDBPyConnection) -> None:
    """Confirma que os limites de memória e de threads estão ativos (camada contra consultas caras)."""
    try:
        mem = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        thr = int(con.execute("SELECT current_setting('threads')").fetchone()[0])
        ok_mem = _to_bytes(mem) <= _to_bytes(config.DUCKDB_MEMORY_LIMIT) * 1.05
    except (duckdb.Error, ValueError, KeyError, TypeError) as e:
        raise InsecureConnectionError(f"Não foi possível confirmar os limites de recursos: {e}") from e
    if not ok_mem or thr > int(config.DUCKDB_THREADS):
        raise InsecureConnectionError(
            f"Limites de recursos não aplicados (memory_limit={mem}, threads={thr}). Consultas bloqueadas.")


def _connect() -> duckdb.DuckDBPyConnection:
    path = config.WAREHOUSE_PATH
    if not path.exists():
        raise QueryError(f"Warehouse não encontrado em {path}. Rode scripts/build_warehouse.py.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        for stmt in (
            "SET enable_external_access=false",
            f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'",
            f"SET threads={int(config.DUCKDB_THREADS)}",
            "SET lock_configuration=true",
        ):
            try:
                con.execute(stmt)
            except duckdb.Error:
                # Instância já travada (cache de instâncias do DuckDB no mesmo processo):
                # a verificação abaixo decide se ela é segura.
                pass
        _verify(con)
        _verify_resources(con)
    except BaseException:
        con.close()
        raise
    return con


def get_connection() -> duckdb.DuckDBPyConnection:
    """Conexão read-only, com acesso externo desligado e configuração travada (cache por processo)."""
    global _CON, _CON_PATH
    with _LOCK:
        if _CON is None or _CON_PATH != str(config.WAREHOUSE_PATH):
            reset_connection()
            _CON = _connect()
            _CON_PATH = str(config.WAREHOUSE_PATH)
        return _CON


def reset_connection() -> None:
    """Fecha e descarta a conexão em cache (a próxima chamada cria outra)."""
    global _CON, _CON_PATH
    with _LOCK:
        if _CON is not None:
            try:
                _CON.close()
            except Exception:
                pass
        _CON, _CON_PATH = None, None


# ---------------------------------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, decimal.Decimal):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return v.isoformat(sep=" ") if isinstance(v, _dt.datetime) else v.isoformat()
    if isinstance(v, _dt.timedelta):
        return str(v)
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item"):            # escalares numpy
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    return str(v)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1000:
            return f"{v:.0f}"
        if a >= 1:
            return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.4g}"
    s = str(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    return s if len(s) <= 80 else s[:77] + "..."


@dataclass
class QueryResult:
    """Resultado de run_query.

    - `sql`: o SQL validado/normalizado (com LIMIT <= config.MAX_ROWS), como exibido ao usuário.
    - `rows`: no máximo config.MAX_ROWS linhas, valores serializáveis em JSON
      (datas -> str ISO, decimais -> float, NaN/inf -> None).
    - `row_count`: len(rows).
    - `truncated`: True se a consulta como escrita (respeitando o LIMIT próprio dela, quando
      <= MAX_ROWS) teria devolvido MAIS de config.MAX_ROWS linhas, ou seja, há linhas que não
      vieram. Detectado executando com LIMIT MAX_ROWS+1 e buscando até MAX_ROWS+1 linhas.
    - `elapsed_ms`: tempo de execução + busca.
    """
    sql: str
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    elapsed_ms: float
    column_types: list[str] = field(default_factory=list)

    def to_markdown(self, max_rows: int = 30) -> str:
        if not self.columns:
            return "(consulta sem colunas)"
        head = "| " + " | ".join(_fmt(c) for c in self.columns) + " |"
        sep = "|" + "|".join("---" for _ in self.columns) + "|"
        body = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows]]
        lines = [head, sep, *body]
        if not self.rows:
            lines.append("(0 linhas)")
        notes = []
        if self.row_count > max_rows:
            notes.append(f"mostrando {max_rows} de {self.row_count} linhas")
        if self.truncated:
            notes.append(f"resultado truncado em {config.MAX_ROWS} linhas; agregue ou filtre mais")
        if notes:
            lines.append("(" + "; ".join(notes) + ")")
        return "\n".join(lines)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_query(sql: str, *, timeout_s: float | None = None) -> QueryResult:
    """validate_sql -> execução com timeout -> QueryResult.

    Levanta SQLRejected (guard), QueryTimeout (tempo esgotado) ou QueryError (erro do DuckDB,
    ex.: coluna inexistente — a mensagem deve voltar ao LLM para ele corrigir a consulta).
    """
    shown_sql = validate_sql(sql)                                 # LIMIT <= MAX_ROWS
    exec_sql = validate_sql(sql, max_rows=config.MAX_ROWS + 1)    # detecta truncamento
    timeout = float(config.SQL_TIMEOUT_S if timeout_s is None else timeout_s)

    with _LOCK:
        con = get_connection()
        cur = con.cursor()
        fired = threading.Event()

        def _interrupt() -> None:
            fired.set()
            try:
                cur.interrupt()
            except Exception:
                pass

        timer = threading.Timer(timeout, _interrupt)
        timer.daemon = True
        t0 = time.perf_counter()
        timer.start()
        try:
            cur.execute(exec_sql)
            columns = [d[0] for d in (cur.description or [])]
            types = [str(d[1]) for d in (cur.description or [])]
            raw = cur.fetchmany(config.MAX_ROWS + 1)
        except duckdb.InterruptException as e:
            reset_connection()
            raise QueryTimeout(
                f"A consulta passou do tempo limite de {timeout:g} s e foi interrompida. "
                "Filtre mais (país, categoria, período), agregue antes de juntar tabelas ou "
                "evite junções que multiplicam linhas.") from e
        except duckdb.Error as e:
            if fired.is_set():
                reset_connection()
                raise QueryTimeout(f"A consulta passou do tempo limite de {timeout:g} s.") from e
            msg = str(e).strip()
            raise QueryError(f"Erro do DuckDB ao executar a consulta: {msg[:800]}") from e
        finally:
            timer.cancel()
            try:
                cur.close()
            except Exception:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000

    truncated = len(raw) > config.MAX_ROWS
    rows = [[_jsonable(v) for v in r] for r in raw[: config.MAX_ROWS]]
    return QueryResult(sql=shown_sql, columns=columns, rows=rows, row_count=len(rows),
                       truncated=truncated, elapsed_ms=round(elapsed_ms, 1), column_types=types)
