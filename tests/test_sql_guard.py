"""Testes do guard de SQL (sem banco: só AST)."""
import re

import pytest
import sqlglot

from insights_agent import config
from insights_agent.sql_guard import MAX_SQL_CHARS, SQLRejected, validate_sql

MAX = config.MAX_ROWS

MALICIOUS = {
    # ---- múltiplas instruções / escrita / DDL ----
    "multi_drop": "SELECT 1; DROP TABLE videos",
    "multi_select": "SELECT 1; SELECT 2",
    "drop": "DROP TABLE videos",
    "delete": "DELETE FROM videos",
    "update": "UPDATE videos SET views = 0",
    "insert": "INSERT INTO videos (video_id) VALUES ('x')",
    "insert_select": "INSERT INTO videos SELECT * FROM videos",
    "create_table": "CREATE TABLE t AS SELECT 1",
    "create_view": "CREATE VIEW v AS SELECT video_id FROM videos",
    "create_macro": "CREATE MACRO m(a) AS a + 1",
    "alter": "ALTER TABLE videos ADD COLUMN x INT",
    "select_into": "SELECT video_id INTO t FROM videos",
    "truncate": "TRUNCATE videos",
    # ---- comandos de sistema ----
    "copy_to": "COPY videos TO 'out.csv'",
    "copy_select": "COPY (SELECT video_id FROM videos) TO 'out.parquet'",
    "attach": "ATTACH 'other.db' AS o",
    "detach": "DETACH o",
    "pragma": "PRAGMA database_list",
    "set": "SET enable_external_access = true",
    "reset": "RESET lock_configuration",
    "install": "INSTALL httpfs",
    "load": "LOAD httpfs",
    "export": "EXPORT DATABASE 'dump'",
    "use": "USE other",
    "call": "CALL pragma_version()",
    "describe": "DESCRIBE videos",
    "show": "SHOW TABLES",
    "summarize": "SUMMARIZE videos",
    "begin": "BEGIN TRANSACTION",
    "vacuum": "VACUUM",
    "checkpoint": "CHECKPOINT",
    # ---- arquivos / rede / ambiente ----
    "read_csv": "SELECT * FROM read_csv('C:/Windows/win.ini')",
    "read_csv_auto": "SELECT * FROM read_csv_auto('secrets.csv')",
    "read_parquet": "SELECT count(*) FROM read_parquet('../youtube-views-prediction/data/x.parquet')",
    "read_json_auto": "SELECT * FROM read_json_auto('a.json')",
    "read_text": "SELECT content FROM read_text('C:/Users/usuario/.ssh/id_rsa')",
    "read_blob": "SELECT * FROM read_blob('x')",
    "file_literal": "SELECT * FROM 'data/warehouse.duckdb'",
    "csv_literal": "SELECT * FROM 'secrets.csv'",
    "http_literal": "SELECT * FROM 'https://example.com/x.parquet'",
    "glob": "SELECT * FROM glob('C:/*')",
    "parquet_scan": "SELECT * FROM parquet_scan('x.parquet')",
    "parquet_metadata": "SELECT * FROM parquet_metadata('x.parquet')",
    "sniff_csv": "SELECT * FROM sniff_csv('x.csv')",
    "query_fn": "SELECT * FROM query('DROP TABLE videos')",
    "query_table": "SELECT * FROM query_table('videos')",
    "getenv": "SELECT getenv('ANTHROPIC_API_KEY')",
    "current_setting": "SELECT current_setting('enable_external_access')",
    "duckdb_settings": "SELECT * FROM duckdb_settings()",
    "duckdb_tables": "SELECT * FROM duckdb_tables()",
    "pragma_fn": "SELECT * FROM pragma_table_info('videos')",
    "sqlite_scan": "SELECT * FROM sqlite_scan('x.db', 't')",
    "read_csv_in_subquery": "SELECT video_id FROM videos WHERE video_id IN (SELECT a FROM read_csv('x.csv'))",
    "read_csv_in_cte": "WITH x AS (SELECT * FROM read_csv('x.csv')) SELECT * FROM x",
    "read_csv_scalar_pos": "SELECT video_id, (SELECT count(*) FROM read_csv('x')) FROM videos",
    "range_table": "SELECT * FROM range(1000000000)",
    # funções geradoras na forma escalar (DoS numa única linha; achado da revisão do Codex)
    "range_scalar_huge": "SELECT range(1000000000)",
    "generate_series_scalar": "SELECT generate_series(1, 1000000000)",
    "repeat_huge": "SELECT repeat('x', 1000000000)",
    "list_resize_huge": "SELECT list_resize([1], 1000000000)",
    "generate_series_table": "SELECT * FROM generate_series(1, 10)",
    # ---- tabelas não permitidas ----
    "unknown_table": "SELECT * FROM users",
    "information_schema": "SELECT * FROM information_schema.tables",
    "qualified": "SELECT video_id FROM main.videos",
    "catalog_qualified": "SELECT video_id FROM warehouse.main.videos",
    "sqlite_master": "SELECT * FROM sqlite_master",
    "union_unknown": "SELECT video_id FROM videos UNION ALL SELECT name FROM secrets",
    # ---- complexidade / DoS ----
    "cross_join": "SELECT count(*) FROM videos CROSS JOIN video_country",
    "comma_join": "SELECT count(*) FROM videos, video_country",
    "join_no_on": "SELECT count(*) FROM videos JOIN video_country",
    "natural_join": "SELECT count(*) FROM videos NATURAL JOIN video_country",
    "positional_join": "SELECT count(*) FROM videos POSITIONAL JOIN video_country",
    "lateral": "SELECT v.video_id FROM videos v, LATERAL (SELECT 1 AS x) l",
    "too_many_joins": (
        "SELECT count(*) FROM videos v "
        + " ".join(f"JOIN video_country c{i} ON c{i}.video_id = v.video_id" for i in range(5))
    ),
    "too_many_ctes": "WITH " + ", ".join(f"c{i} AS (SELECT {i} AS x)" for i in range(5))
                     + " SELECT x FROM c0",
    "recursive_cte": "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r",
    "deep_nesting": "SELECT a FROM (SELECT a FROM (SELECT a FROM (SELECT a FROM "
                    "(SELECT 1 AS a) t1) t2) t3) t4",
    "order_random": "SELECT video_id, title FROM videos ORDER BY random()",
    "order_random_expr": "SELECT video_id FROM videos ORDER BY views * random() DESC",
    "too_long": "SELECT video_id FROM videos WHERE title = '" + "x" * MAX_SQL_CHARS + "'",
    # ---- SELECT * em tabelas-base ----
    "star_videos": "SELECT * FROM videos",
    "from_first": "FROM videos",
    "star_video_country": "SELECT * FROM video_country WHERE country = 'Brazil'",
    "qualified_star": "SELECT v.*, c.country FROM videos v JOIN video_country c USING (video_id)",
    "star_exclude": "SELECT * EXCLUDE (title) FROM videos",
    # ---- sintaxe / vazio ----
    "empty": "   ",
    "only_semicolons": ";;",
    "garbage": "SELEC video_id FRM videos",
    "comment_only": "-- nada aqui",
}

VALID = {
    "simple_agg": "SELECT category, count(*) AS n FROM videos GROUP BY category ORDER BY n DESC",
    "where_limit": "SELECT video_id, title, views FROM videos WHERE is_short ORDER BY views DESC LIMIT 10",
    "join_on": ("SELECT c.country, median(v.views) AS med FROM videos v "
                "JOIN video_country c ON c.video_id = v.video_id GROUP BY c.country"),
    "join_using": ("SELECT count(DISTINCT v.video_id) FROM videos v JOIN video_country c USING (video_id) "
                   "WHERE c.country = 'Brazil'"),
    "cte": ("WITH br AS (SELECT video_id FROM video_country WHERE country = 'Brazil') "
            "SELECT v.category, count(*) FROM videos v JOIN br USING (video_id) GROUP BY ALL"),
    "cte_star": "WITH t AS (SELECT video_id, views FROM videos) SELECT * FROM t LIMIT 5",
    "window": ("SELECT category, video_id, views, row_number() OVER (PARTITION BY category "
               "ORDER BY views DESC) AS rk FROM videos QUALIFY rk <= 3"),
    "subquery_in": ("SELECT title FROM videos WHERE video_id IN "
                    "(SELECT video_id FROM video_country WHERE country = 'Portugal' AND days_trending > 5)"),
    "subquery_from": "SELECT avg(n) FROM (SELECT country, count(*) AS n FROM video_country GROUP BY 1) t",
    "scalar_subquery": "SELECT video_id, views / (SELECT median(views) FROM videos) AS rel FROM videos",
    "union": "SELECT 'a' AS g, count(*) FROM videos UNION ALL SELECT 'b', count(*) FROM video_country",
    "daily": ("SELECT date_trunc('month', trending_date) AS m, sum(n_videos) FROM daily_trending "
              "WHERE country = 'Brazil' GROUP BY 1 ORDER BY 1"),
    "star_daily": "SELECT * FROM daily_trending WHERE country = 'Brazil' LIMIT 20",
    "case_buckets": ("SELECT CASE WHEN duration_s < 60 THEN '<1m' WHEN duration_s < 600 THEN '1-10m' "
                     "ELSE '10m+' END AS b, quantile_cont(views, 0.5) FROM videos GROUP BY b"),
    "count_star": "SELECT count(*) FROM videos",
    "comment_trailing": "SELECT 1 -- ; DROP TABLE videos",
    "block_comment": "SELECT /* DROP TABLE videos; */ count(*) FROM videos",
    "trailing_semicolon": "SELECT count(*) FROM videos;",
    "quoted_table": 'SELECT count(*) FROM "videos"',
    "three_level_nesting": "SELECT a FROM (SELECT a FROM (SELECT a FROM (SELECT 1 AS a) t1) t2) t3",
    "left_join": ("SELECT v.video_id, c.days_trending FROM videos v LEFT JOIN video_country c "
                  "ON c.video_id = v.video_id AND c.country = 'Brazil' LIMIT 5"),
    "string_with_keywords": "SELECT count(*) FROM videos WHERE title ILIKE '%drop table%'",
    "using_sample": "SELECT video_id, views FROM videos USING SAMPLE 100 ROWS",
}


@pytest.mark.parametrize("name", sorted(MALICIOUS))
def test_rejects(name):
    with pytest.raises(SQLRejected) as ei:
        validate_sql(MALICIOUS[name])
    assert ei.value.reason and isinstance(ei.value.reason, str)


@pytest.mark.parametrize("name", sorted(VALID))
def test_accepts(name):
    out = validate_sql(VALID[name])
    assert isinstance(out, str) and out
    # o resultado normalizado continua 1 SELECT e passa de novo no guard (idempotente)
    assert validate_sql(out) == out
    assert len(sqlglot.parse(out, read="duckdb")) == 1


def test_counts():
    assert len(MALICIOUS) >= 25
    assert len(VALID) >= 10


def _limit(sql):
    m = re.search(r"LIMIT (\d+)\s*$", sql)
    return int(m.group(1)) if m else None


def test_limit_added_when_missing():
    assert _limit(validate_sql("SELECT video_id FROM videos")) == MAX


def test_limit_lowered_when_too_big():
    assert _limit(validate_sql("SELECT video_id FROM videos LIMIT 100000")) == MAX


def test_limit_kept_when_small():
    assert _limit(validate_sql("SELECT video_id FROM videos LIMIT 7")) == 7


def test_limit_percent_and_expression_replaced():
    assert _limit(validate_sql("SELECT video_id FROM videos LIMIT 50%")) == MAX
    assert _limit(validate_sql("SELECT video_id FROM videos LIMIT (SELECT 10**9)")) == MAX


def test_limit_on_union_outermost():
    out = validate_sql("SELECT video_id FROM videos UNION SELECT video_id FROM video_country")
    assert _limit(out) == MAX


def test_inner_limit_untouched_outer_added():
    out = validate_sql("SELECT x FROM (SELECT video_id AS x FROM videos LIMIT 5000) t")
    assert "LIMIT 5000" in out and _limit(out) == MAX


def test_fetch_first_converted():
    out = validate_sql("SELECT video_id FROM videos FETCH FIRST 5 ROWS ONLY")
    assert _limit(out) == 5


def test_custom_max_rows():
    assert _limit(validate_sql("SELECT video_id FROM videos", max_rows=MAX + 1)) == MAX + 1


def test_comments_stripped():
    out = validate_sql("SELECT 1 -- ; DROP TABLE videos")
    assert "DROP" not in out.upper() and "--" not in out


def test_multi_statement_message_in_portuguese():
    with pytest.raises(SQLRejected) as ei:
        validate_sql("SELECT 1; DROP TABLE x")
    assert "1 instrução" in ei.value.reason


def test_table_message_lists_allowed():
    with pytest.raises(SQLRejected) as ei:
        validate_sql("SELECT a FROM secrets")
    assert "videos, video_country, daily_trending" in ei.value.reason
