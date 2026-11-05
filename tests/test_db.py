"""Testes da camada de banco (precisam de data/warehouse*.duckdb)."""
import json
import time

import duckdb
import pytest

from insights_agent import config, db
from insights_agent.db import QueryError, QueryTimeout, get_connection, run_query
from insights_agent.sql_guard import SQLRejected

pytestmark = pytest.mark.skipif(not config.WAREHOUSE_PATH.exists(),
                                reason="warehouse não construído (scripts/build_warehouse.py)")


# ---------------- read-only e segurança da conexão ----------------
def test_insert_rejected_by_guard():
    with pytest.raises(SQLRejected):
        run_query("INSERT INTO videos (video_id) VALUES ('x')")


def test_create_rejected_by_guard():
    with pytest.raises(SQLRejected):
        run_query("CREATE TABLE t AS SELECT 1")


def test_direct_write_on_connection_fails():
    """Mesmo sem o guard, a conexão é read-only."""
    cur = get_connection().cursor()
    try:
        for stmt in ("CREATE TABLE t (a INT)",
                     "INSERT INTO videos (video_id) VALUES ('x')",
                     "DELETE FROM videos",
                     "DROP TABLE daily_trending"):
            with pytest.raises(duckdb.Error):
                cur.execute(stmt)
    finally:
        cur.close()


def test_external_access_blocked_on_raw_connection():
    """Se uma query maliciosa passasse pelo guard, o DuckDB ainda recusaria ler/escrever arquivos."""
    existing = (config.ROOT / "requirements.txt").as_posix()      # arquivo local que existe
    out = config.DATA_DIR / "_nao_deve_existir.csv"
    cur = get_connection().cursor()
    try:
        for stmt in (f"SELECT * FROM read_csv('{existing}')",
                     f"SELECT * FROM read_text('{existing}')",
                     f"SELECT * FROM '{existing}'",
                     f"COPY (SELECT 1) TO '{out.as_posix()}'",
                     f"ATTACH '{(config.DATA_DIR / '_x.db').as_posix()}' AS x"):
            with pytest.raises(duckdb.Error):
                cur.execute(stmt).fetchall()
    finally:
        cur.close()
    assert not out.exists()
    assert not (config.DATA_DIR / "_x.db").exists()


def test_configuration_locked():
    cur = get_connection().cursor()
    try:
        with pytest.raises(duckdb.Error):
            cur.execute("SET enable_external_access=true")
        with pytest.raises(duckdb.Error):
            cur.execute("SET lock_configuration=false")
        assert cur.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
    finally:
        cur.close()


def test_verify_rejects_insecure_connection():
    con = duckdb.connect()          # em memória, sem as opções de segurança
    try:
        with pytest.raises(db.InsecureConnectionError):
            db._verify(con)
    finally:
        con.close()


# ---------------- LIMIT / truncamento ----------------
def test_limit_applied_and_truncated():
    r = run_query("SELECT video_id, views FROM videos")
    assert r.row_count == config.MAX_ROWS
    assert r.truncated is True
    assert r.sql.rstrip().endswith(f"LIMIT {config.MAX_ROWS}")


def test_small_result_not_truncated():
    r = run_query("SELECT category, count(*) AS n FROM videos GROUP BY category ORDER BY n DESC")
    assert 0 < r.row_count < config.MAX_ROWS
    assert r.truncated is False
    assert r.columns == ["category", "n"]


def test_user_limit_respected():
    r = run_query("SELECT video_id FROM videos ORDER BY views DESC LIMIT 5")
    assert r.row_count == 5 and r.truncated is False


def test_exact_max_rows_not_truncated():
    r = run_query(f"SELECT video_id FROM videos LIMIT {config.MAX_ROWS}")
    assert r.row_count == config.MAX_ROWS and r.truncated is False


# ---------------- timeout ----------------
def test_timeout_interrupts_heavy_join():
    heavy = ("SELECT count(*) FROM video_country a JOIN video_country b ON a.country = b.country "
             "JOIN video_country c ON c.country = b.country")
    t0 = time.perf_counter()
    with pytest.raises(QueryTimeout) as ei:
        run_query(heavy, timeout_s=1.0)
    assert time.perf_counter() - t0 < 10
    assert "tempo limite" in str(ei.value)
    # a conexão foi recriada e continua segura e funcional
    r = run_query("SELECT count(*) AS n FROM daily_trending")
    assert r.rows[0][0] > 0
    cur = get_connection().cursor()
    try:
        assert cur.execute("SELECT current_setting('enable_external_access')").fetchone()[0] is False
    finally:
        cur.close()


# ---------------- erros e serialização ----------------
def test_unknown_column_is_query_error():
    with pytest.raises(QueryError) as ei:
        run_query("SELECT nao_existe FROM videos")
    assert "nao_existe" in str(ei.value)


def test_result_json_serializable():
    r = run_query(
        "SELECT v.video_id, v.published_at, v.first_trending_date, v.is_short, v.likes, "
        "v.hours_to_trend, CAST(v.views AS DECIMAL(18,2)) AS dec_views, c.first_date "
        "FROM videos v JOIN video_country c USING (video_id) WHERE c.country = 'Brazil' LIMIT 20")
    s = json.dumps(r.to_dict())
    assert r.row_count == 20 and s
    assert isinstance(r.rows[0][1], str) and isinstance(r.rows[0][2], str)
    assert isinstance(r.rows[0][6], float)


def test_to_markdown():
    r = run_query("SELECT category, median(views) AS med FROM videos GROUP BY 1 ORDER BY 2 DESC")
    md = r.to_markdown(max_rows=3)
    lines = md.splitlines()
    assert lines[0] == "| category | med |"
    assert lines[1] == "|---|---|"
    assert len([ln for ln in lines if ln.startswith("| ")]) == 4      # cabeçalho + 3 linhas
    assert "mostrando 3 de" in md


def test_realistic_query_fast():
    sql = """
    SELECT CASE WHEN v.duration_s <= 60 THEN '0-1 min' WHEN v.duration_s <= 180 THEN '1-3 min'
                WHEN v.duration_s <= 480 THEN '3-8 min' WHEN v.duration_s <= 1200 THEN '8-20 min'
                ELSE '20+ min' END AS faixa,
           count(*) AS n, median(v.views) AS mediana_views
    FROM videos v JOIN video_country c ON c.video_id = v.video_id
    WHERE c.country = 'Brazil' GROUP BY 1 ORDER BY 1
    """
    r = run_query(sql)
    assert r.row_count == 5
    assert r.elapsed_ms < 3000
