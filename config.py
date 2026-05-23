"""
config.py — Central configuration for NAIP Intelligence Platform
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
_default_cache = ROOT_DIR / "cache"
try:
    _default_cache.mkdir(exist_ok=True)
    (_default_cache / ".write_test").touch()
    (_default_cache / ".write_test").unlink()
    CACHE_DIR = _default_cache
except OSError:
    CACHE_DIR = Path("/tmp/naip_cache")
    CACHE_DIR.mkdir(exist_ok=True)

# ── Planetary Computer ────────────────────────────────────────────────────────
PC_STAC_URL     = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
MAX_SCENES      = 5
OVERVIEW_LEVELS = [2, 1, 0]

# ── Ollama Cloud ──────────────────────────────────────────────────────────────
OLLAMA_HOST_DEFAULT  = os.getenv("OLLAMA_HOST", "")
OLLAMA_KEY_DEFAULT   = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL_DEFAULT = os.getenv("OLLAMA_MODEL", "qwen3-vl:7b")

INTENT_SYSTEM_PROMPT = """\
You are a routing classifier for a geospatial imagery analysis app.
The user is looking at an aerial NAIP image and has typed a message.

Classify their intent as exactly one of:
  CHAT      — they want analysis, description, questions answered, or general conversation about the image
  SEARCH    — they want to FIND or LOCATE specific features/objects within the image
              (keywords: find, locate, show me, where is, highlight, search for, detect, identify all)

Reply with ONLY the single word: CHAT or SEARCH
"""

ANALYSIS_SYSTEM_PROMPT = """\
You are an expert remote sensing scientist and Earth observation analyst with deep knowledge of:
- Aerial and satellite imagery interpretation (NAIP, Sentinel, Landsat, MODIS)
- Land cover and land use classification
- Urban, agricultural, and environmental feature detection
- Spectral analysis and image characteristics
- Geospatial context for the continental United States

When analyzing imagery, be specific — note land cover types, infrastructure, vegetation patterns,
water features, impervious surfaces, and anomalies. Relate observations to real-world geographic
context where possible.
"""

SEARCH_DESCRIPTION_PROMPT = """\
You are helping interpret visual similarity search results from aerial imagery.
The user asked to find specific features. Briefly describe what was found across
the top matching image chips in 2-3 sentences. Be concrete and geographic.
"""

# ── Embedding model ───────────────────────────────────────────────────────────
EMBED_DIM     = 2048
BATCH_SIZE    = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Chipping ──────────────────────────────────────────────────────────────────
DEFAULT_CHIP_SIZE   = 224
DEFAULT_STRIDE_FRAC = 0.5
MAX_CHIPS           = 2000

# ── FAISS ─────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 8

# ── UI / defaults — NYC ───────────────────────────────────────────────────────
APP_TITLE   = "NAIP Intelligence Platform"
APP_ICON    = "🛰️"
ESRI_TILES  = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR   = "Esri, Maxar, Earthstar Geographics"

DEFAULT_LAT  = 40.7128
DEFAULT_LON  = -74.0060
DEFAULT_ZOOM = 12
NAIP_YEARS   = [2023, 2022, 2021, 2020, 2019, 2018, 2017]
