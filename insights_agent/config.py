"""Configuração central do YouTube Insights Agent."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Carrega variáveis de um arquivo .env (KEY=valor) sem sobrescrever as já definidas no ambiente.

    O .env fica fora do Git (.gitignore); é onde a chave do Groq é configurada localmente.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


_load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
P1 = ROOT.parent / "youtube-views-prediction"          # projeto 1 (dados + modelo)

# ---- fontes (projeto 1) ----
SLIM_PARQUET = P1 / "data" / "trending_slim.parquet"
VIDEOS_PARQUET = P1 / "data" / "videos_features.parquet"
# modelos: cópia local em models/ (repositório autossuficiente); senão, os do projeto 1
_MODELS = ROOT / "models" if (ROOT / "models" / "lgbm_views.txt").exists() else P1 / "models"
LGBM_MODEL = Path(os.getenv("LGBM_MODEL", _MODELS / "lgbm_views.txt"))
TEXT_MODEL = Path(os.getenv("TEXT_MODEL", _MODELS / "text_tfidf_ridge.joblib"))
MODEL_META = Path(os.getenv("MODEL_META", _MODELS / "metadata.json"))

# ---- warehouse (o que o LLM enxerga) ----
WAREHOUSE_FULL = DATA_DIR / "warehouse.duckdb"
WAREHOUSE_SAMPLE = DATA_DIR / "warehouse_sample.duckdb"
WAREHOUSE_PATH = Path(os.getenv("WAREHOUSE_PATH",
                                WAREHOUSE_FULL if WAREHOUSE_FULL.exists() else WAREHOUSE_SAMPLE))
ALLOWED_TABLES = ("videos", "video_country", "daily_trending")

# ---- limites de segurança ----
MAX_ROWS = 200                 # linhas máximas devolvidas por consulta
SQL_TIMEOUT_S = 15.0           # tempo máximo por consulta
DUCKDB_MEMORY_LIMIT = "1GB"
DUCKDB_THREADS = 4
MAX_TOOL_CALLS = 8             # chamadas de ferramenta por pergunta
TOOL_RESULT_MAX_CHARS = 6000   # truncamento do texto devolvido ao LLM

# ---- LLM ----
MODEL = os.getenv("INSIGHTS_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("INSIGHTS_EFFORT", "medium")
MAX_TOKENS = 16000
# US$ por milhão de tokens (input, output, cache_read, cache_write_5m)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

DEMO_TRACES_DIR = ROOT / "demo_traces"

# ---- provedor do LLM ----
# "groq" (padrão; plano gratuito: ~30 req/min, 8 mil tokens/min, 200 mil tokens/dia) ou "anthropic"
PROVIDER = os.getenv("INSIGHTS_PROVIDER", "groq").strip().lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "medium")
GROQ_MAX_WALL_S = 300.0            # espera os limites do plano gratuito, então é mais longo
GROQ_TOOL_RESULT_MAX_CHARS = 2000  # resultados menores para caber em 8 mil tokens/min
GROQ_CONTEXT_BUDGET = 5800         # tokens estimados por requisição (limite do plano: ~8 mil)
GROQ_MAX_TOKENS = 4096
GROQ_MAX_RETRIES_429 = 4
# no plano gratuito do Groq o custo é zero (os limites é que são o gargalo)
PRICES.update({"openai/gpt-oss-120b": (0.0, 0.0, 0.0, 0.0), "openai/gpt-oss-20b": (0.0, 0.0, 0.0, 0.0),
               "qwen/qwen3.8-27b": (0.0, 0.0, 0.0, 0.0)})
