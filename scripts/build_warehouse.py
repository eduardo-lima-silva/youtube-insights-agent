"""Constrói o warehouse DuckDB (as únicas tabelas visíveis ao LLM).

Uso:
    python scripts/build_warehouse.py            # data/warehouse.duckdb (base completa)
    python scripts/build_warehouse.py --sample   # data/warehouse_sample.duckdb (< 90 MB, deploy)

Fontes (projeto 1): config.SLIM_PARQUET (vídeo x país x dia) e config.VIDEOS_PARQUET
(1 linha por vídeo, com features). Tabelas geradas: videos, video_country, daily_trending
(ver docs/02_arquitetura.md).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from insights_agent import config  # noqa: E402
from insights_agent.catalog import SAMPLE_COUNTRIES  # noqa: E402

# Amostra (deploy): um vídeo entra se o primeiro país em alta OU qualquer país em que esteve em
# alta estiver em catalog.SAMPLE_COUNTRIES.
SAMPLE_MAX_MB = 90

# Categorias oficiais do YouTube; qualquer outro valor (lixo de coleta, NULL) vira 'Unknown'.
VALID_CATEGORIES = (
    "Autos & Vehicles", "Comedy", "Education", "Entertainment", "Film & Animation", "Gaming",
    "Howto & Style", "Movies", "Music", "News & Politics", "Nonprofits & Activism",
    "People & Blogs", "Pets & Animals", "Science & Technology", "Sports", "Travel & Events",
)


def _lit(p: Path) -> str:
    """Caminho como literal SQL (barras normais, aspas escapadas)."""
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _cat_expr(col: str) -> str:
    cats = ", ".join("'" + c.replace("'", "''") + "'" for c in VALID_CATEGORIES)
    return f"CASE WHEN {col} IN ({cats}) THEN {col} ELSE 'Unknown' END"


def build(out_path: Path, sample: bool) -> None:
    for src in (config.SLIM_PARQUET, config.VIDEOS_PARQUET):
        if not src.exists():
            raise SystemExit(f"Fonte não encontrada: {src}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".building.duckdb")
    for p in (tmp_path, Path(str(tmp_path) + ".wal")):
        if p.exists():
            p.unlink()

    t0 = time.perf_counter()
    con = duckdb.connect(str(tmp_path))
    con.execute("SET memory_limit='3GB'")
    con.execute("SET preserve_insertion_order=true")
    slim, vids = _lit(config.SLIM_PARQUET), _lit(config.VIDEOS_PARQUET)

    # Linhas do slim já limpas (país não nulo).
    con.execute(f"""
        CREATE TEMP VIEW slim AS
        SELECT video_id, trending_date::DATE AS trending_date, trending_country AS country,
               {_cat_expr('category')} AS category, views
        FROM read_parquet({slim})
        WHERE trending_country IS NOT NULL AND video_id IS NOT NULL AND trending_date IS NOT NULL
    """)

    # Conjunto de vídeos a incluir.
    if sample:
        countries = ", ".join("'" + c + "'" for c in SAMPLE_COUNTRIES)
        con.execute(f"""
            CREATE TEMP TABLE keep_ids AS
            SELECT video_id FROM read_parquet({vids}) WHERE market IN ({countries})
            UNION
            SELECT DISTINCT s.video_id FROM slim s
            SEMI JOIN read_parquet({vids}) v ON v.video_id = s.video_id
            WHERE s.country IN ({countries})
        """)
    else:
        con.execute(f"CREATE TEMP TABLE keep_ids AS SELECT video_id FROM read_parquet({vids})")

    con.execute(f"""
        CREATE TABLE videos AS
        SELECT
            v.video_id::VARCHAR                                   AS video_id,
            v.title::VARCHAR                                      AS title,
            v.channel_title::VARCHAR                              AS channel_title,
            {_cat_expr('v.category')}::VARCHAR                    AS category,
            v.published_at::TIMESTAMP                             AS published_at,
            v.pub_hour::INTEGER                                   AS pub_hour,
            v.pub_dow::INTEGER                                    AS pub_dow,
            v.duration_s::INTEGER                                 AS duration_s,
            (v.duration_s <= 180)::BOOLEAN                        AS is_short,
            round(v.views)::BIGINT                                AS views,
            round(v.likes)::BIGINT                                AS likes,
            round(v.comments)::BIGINT                             AS comments,
            v.age_days::DOUBLE                                    AS age_days,
            round(exp(v.log_channel_subs) - 1)::BIGINT            AS channel_subs,
            coalesce(v.channel_country, 'Unknown')::VARCHAR       AS channel_country,
            v.market::VARCHAR                                     AS first_trending_country,
            v.first_trending_date::DATE                           AS first_trending_date,
            v.hours_to_trend::DOUBLE                              AS hours_to_trend,
            v.n_trending_countries::INTEGER                       AS n_trending_countries,
            v.n_trending_days::INTEGER                            AS n_trending_days
        FROM read_parquet({vids}) v
        SEMI JOIN keep_ids k ON k.video_id = v.video_id
        ORDER BY first_trending_country, first_trending_date, video_id
    """)

    con.execute("""
        CREATE TABLE video_country AS
        SELECT
            s.video_id::VARCHAR                        AS video_id,
            s.country::VARCHAR                         AS country,
            min(s.trending_date)::DATE                 AS first_date,
            max(s.trending_date)::DATE                 AS last_date,
            count(DISTINCT s.trending_date)::INTEGER   AS days_trending
        FROM slim s
        SEMI JOIN keep_ids k ON k.video_id = s.video_id
        GROUP BY 1, 2
        ORDER BY country, video_id
    """)

    # daily_trending é um agregado da base completa (pequeno), igual nos dois arquivos.
    con.execute("""
        CREATE TABLE daily_trending AS
        SELECT
            trending_date::DATE                  AS trending_date,
            country::VARCHAR                     AS country,
            category::VARCHAR                    AS category,
            count(DISTINCT video_id)::INTEGER    AS n_videos,
            median(views)::DOUBLE                AS median_views
        FROM slim
        GROUP BY 1, 2, 3
        ORDER BY country, trending_date, category
    """)

    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in config.ALLOWED_TABLES}

    # ---- verificações de integridade ----
    n_ids = con.execute("SELECT count(DISTINCT video_id) FROM videos").fetchone()[0]
    assert n_ids == counts["videos"], f"video_id não é único em videos ({n_ids} != {counts['videos']})"
    orphans = con.execute(
        "SELECT count(*) FROM video_country vc ANTI JOIN videos v ON v.video_id = vc.video_id"
    ).fetchone()[0]
    assert orphans == 0, f"{orphans} linhas de video_country sem vídeo correspondente"
    dup_vc = con.execute(
        "SELECT count(*) - count(DISTINCT (video_id, country)) FROM video_country").fetchone()[0]
    assert dup_vc == 0, "(video_id, country) duplicado em video_country"

    def _mm(table: str, col: str) -> list[str]:
        lo, hi = con.execute(f"SELECT min({col}), max({col}) FROM {table}").fetchone()
        return [str(lo), str(hi)]

    date_ranges = {
        "videos.published_at": _mm("videos", "published_at"),
        "videos.first_trending_date": _mm("videos", "first_trending_date"),
        "video_country.first_date": _mm("video_country", "first_date"),
        "video_country.last_date": _mm("video_country", "last_date"),
        "daily_trending.trending_date": _mm("daily_trending", "trending_date"),
    }
    con.execute("DROP TABLE keep_ids")
    con.execute("CHECKPOINT")
    con.close()

    if out_path.exists():
        out_path.unlink()          # falha (PermissionError) se outro processo tiver o arquivo aberto
    os.replace(tmp_path, out_path)
    wal = Path(str(tmp_path) + ".wal")
    if wal.exists():
        wal.unlink()

    size_mb = out_path.stat().st_size / 1e6
    manifest = {
        "warehouse": out_path.name,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": sample,
        "sample_countries": list(SAMPLE_COUNTRIES) if sample else None,
        "size_bytes": out_path.stat().st_size,
        "sources": {
            p.name: {
                "path": f"youtube-views-prediction/data/{p.name}",
                "size_bytes": p.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            for p in (config.SLIM_PARQUET, config.VIDEOS_PARQUET)
        },
        "row_counts": counts,
        "checks": {
            "videos_video_id_unique": True,
            "videos_distinct_video_id": n_ids,
            "video_country_orphans": orphans,
            "video_country_pk_duplicates": dup_vc,
        },
        "date_ranges": date_ranges,
    }
    manifest_path = out_path.parent / ("manifest_sample.json" if sample else "manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{out_path.name}: construído em {time.perf_counter() - t0:.1f} s")
    for t, n in counts.items():
        print(f"  {t:15s} {n:>12,d} linhas")
    print(f"  tamanho: {size_mb:.1f} MB  |  manifesto: {manifest_path.name}")
    if sample and size_mb >= SAMPLE_MAX_MB:
        raise SystemExit(f"ERRO: amostra com {size_mb:.1f} MB (limite {SAMPLE_MAX_MB} MB); "
                         "reduza SAMPLE_COUNTRIES.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", action="store_true",
                    help=f"gera {config.WAREHOUSE_SAMPLE.name} (< {SAMPLE_MAX_MB} MB) para deploy")
    args = ap.parse_args()
    out = config.WAREHOUSE_SAMPLE if args.sample else config.WAREHOUSE_FULL
    build(out, args.sample)


if __name__ == "__main__":
    main()
